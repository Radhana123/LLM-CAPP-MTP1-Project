"""Does the plan confidence tell which plans need a human, and where should the review flag sit? (report 6.4, WP5)

The pipeline flags a plan for human review when its confidence (mean pairwise agreement of the N samples' routes after repair) is below
systems.CONFIDENCE_FLAG. This module looks at finished runs and answers, offline and without any API call:

  * which plans are "bad" (the label is a choice, see LABELS) and how often,
  * how well a score ranks them (AUROC; 0.5 = no information) for the recorded confidence and two diagnostics that are not flagged on
    today (the agreement of the RAW samples, before any repair, and the share of raw samples that were strict-valid),
  * for every possible threshold: the share of plans flagged, the share of bad plans caught (recall) and the share of flags that were
    bad (precision), and two recommendations: the threshold with the best recall within a review budget, and the one that reaches a
    recall target with the fewest flags,
  * the same numbers at any fixed threshold (`evaluate`), with 95% bootstrap intervals over PARTS (the repeats of a part are not
    independent). This is what a sealed split gets: no tuning there, only the threshold fixed on the development set is evaluated.

A score is "higher = more trustworthy"; a plan is flagged when score < threshold, the pipeline's own rule. Thresholds are placed
between two observed values, so the printed threshold reproduces the flagged set exactly.

Honest limits: the labels are proxies (on the development set the M4 reference is the Route Builder, not an expert; the efficiency gap
is against the best Route Builder variant), the sample is small, and picking a threshold on the data it is then measured on is
optimistic: the intervals do not account for that selection.
"""
import random

from capp import metrics as M
from capp import summary as SM
from capp import systems as SY

SIGNALS = ("confidence", "raw_agreement", "raw_valid_share")
SIGNAL_TEXT = {"confidence": "confidence (agreement after repair; what is flagged on)",
               "raw_agreement": "raw agreement (before repair; diagnostic)",
               "raw_valid_share": "raw valid share (samples strict-valid before repair; diagnostic)"}
LABELS = ("gap", "lcs", "not_raw")
LABEL_TEXT = {"gap": "efficiency gap to the best Route Builder variant above {bad_gap:g} points",
              "lcs": "M4 LCS ratio against the reference below {bad_lcs:g}",
              "not_raw": "final route is not the LLM's raw route (it was repaired or replaced)"}
DEFAULT_BAD_GAP = 3.0
DEFAULT_BAD_LCS = 0.8
DEFAULT_BUDGET = 0.20
DEFAULT_RECALL = 0.80
RESAMPLES = 1000


class CalibrationError(ValueError):
    pass


# ------------------------------------------------------------------ rows
def _sample_routes(rec, key):
    return [c[key] for c in rec["candidates"] if c.get("sample_id", -1) >= 0 and c.get("llm_ok") and c.get(key)]


def rows_from_records(records, bad_gap=DEFAULT_BAD_GAP, bad_lcs=DEFAULT_BAD_LCS) -> list:
    """One row per record (part x repeat): the three scores (None when they cannot be computed) and the three bad/not-bad labels
    (None when the label cannot be decided: no valid final route for `gap`, no reference for `lcs`)."""
    part_rows = SM.part_rows(records)
    rows = []
    for rec, pr in zip(records, part_rows):
        info = rec["info"]
        attempts = info.get("attempts") or 0
        raw = _sample_routes(rec, "llm_steps")
        gap, lcs = pr["eff_gap_to_rb_best"], pr["m4_lcs_ratio"]
        rows.append({
            "part_id": rec["part_id"], "repeat": rec.get("repeat", 0),
            "scores": {"confidence": info.get("confidence"),
                       "raw_agreement": SY.agreement(raw),
                       "raw_valid_share": (info.get("raw_strict_valid", 0) / attempts) if attempts else None},
            "labels": {"gap": None if gap is None else gap > bad_gap,
                       "lcs": None if lcs is None else lcs < bad_lcs,
                       "not_raw": pr["final_provenance"] != "raw_llm"},
            "eff_gap": gap, "lcs_ratio": lcs, "m4_source": pr["m4_source"], "provenance": pr["final_provenance"],
            "flagged_recorded": info.get("flagged"),
        })
    return rows


def _usable(rows, signal, label):
    """The rows that have both the score and the label."""
    return [r for r in rows if r["scores"].get(signal) is not None and r["labels"].get(label) is not None]


# ------------------------------------------------------------------ statistics
def auroc(pairs):
    """Probability that a bad plan has a LOWER score than a good one (ties count half). pairs = [(score, is_bad)]; None if one class is empty."""
    bad = [s for s, b in pairs if b]
    good = [s for s, b in pairs if not b]
    if not bad or not good:
        return None
    wins = sum((1.0 if sb < sg else 0.5 if sb == sg else 0.0) for sb in bad for sg in good)
    return wins / (len(bad) * len(good))


def at_threshold(pairs, threshold):
    """Flag score < threshold. pairs = [(score, is_bad)]."""
    n = len(pairs)
    flagged = [b for s, b in pairs if s < threshold]
    bad_total = sum(1 for _, b in pairs if b)
    tp = sum(1 for b in flagged if b)
    return {"n": n, "n_flagged": len(flagged), "flagged_share": len(flagged) / n if n else None,
            "n_bad": bad_total, "base_rate": bad_total / n if n else None, "caught": tp,
            "precision": tp / len(flagged) if flagged else None,
            "recall": tp / bad_total if bad_total else None,
            "lift": (tp / len(flagged)) / (bad_total / n) if flagged and bad_total else None}


def _percentile_ci(values, level=0.95):
    vals = sorted(v for v in values if v is not None)
    if len(vals) < 2:
        return None, None
    lo = vals[int(((1 - level) / 2) * (len(vals) - 1))]
    hi = vals[int((1 - (1 - level) / 2) * (len(vals) - 1))]
    return lo, hi


def _by_part(rows_scored):
    groups = {}
    for r in rows_scored:
        groups.setdefault(r["part_id"], []).append(r)
    return list(groups.values())


def _resample(groups, rng):
    out = []
    for g in rng.choices(groups, k=len(groups)):
        out.extend(g)
    return out


def bootstrap(rows_scored, stat, resamples=RESAMPLES, seed=7):
    """95% interval of stat(list of rows) when whole PARTS are resampled with replacement (a part's repeats stay together)."""
    groups = _by_part(rows_scored)
    if len(groups) < 2:
        return None, None
    rng = random.Random(seed)
    return _percentile_ci([stat(_resample(groups, rng)) for _ in range(resamples)])


def _pairs(rows_scored, signal, label):
    return [(r["scores"][signal], r["labels"][label]) for r in rows_scored]


def evaluate(rows, signal, label, threshold, resamples=RESAMPLES) -> dict:
    """Flagged share, precision, recall and lift at a fixed threshold, with part-bootstrap intervals."""
    use = _usable(rows, signal, label)
    if not use:
        raise CalibrationError(f"no plan has both a {signal} score and a {label} label")
    res = at_threshold(_pairs(use, signal, label), threshold)
    res["threshold"] = threshold
    for key in ("flagged_share", "precision", "recall"):
        res[key + "_ci95"] = bootstrap(use, lambda rs, k=key: at_threshold(_pairs(rs, signal, label), threshold)[k], resamples)
    return res


# ------------------------------------------------------------------ the sweep
def _midpoint(lo, hi):
    """A threshold t with lo < t <= hi: the midpoint, rounded to 3 decimals if the rounding stays inside the gap."""
    mid = (lo + hi) / 2
    for digits in range(3, 13):                      # the shortest rounding that still reproduces the flagged set
        r = round(mid, digits)
        if lo < r <= hi:
            return r
    return mid


def sweep(pairs) -> list:
    """Every distinct flagged set, from 'flag nobody' to 'flag everybody', with the threshold that gives it (flag score < threshold)."""
    values = sorted({s for s, _ in pairs})
    if not values:
        return []
    cuts = [values[0]] + [_midpoint(a, b) for a, b in zip(values, values[1:])] + [round(values[-1] + 0.001, 6)]
    out = []
    for t in cuts:
        row = at_threshold(pairs, t)
        row["threshold"] = t
        out.append(row)
    return out


def recommend_budget(table, budget):
    """The row with the best recall among those flagging at most `budget` of the plans (the largest such flagged set on ties)."""
    ok = [r for r in table if r["flagged_share"] <= budget + 1e-12]
    return max(ok, key=lambda r: (r["recall"] if r["recall"] is not None else -1, r["n_flagged"])) if ok else None


def recommend_recall(table, target):
    """The row with the fewest flags that still catches at least `target` of the bad plans; None if there are no bad plans."""
    ok = [r for r in table if r["recall"] is not None and r["recall"] >= target - 1e-12]
    return min(ok, key=lambda r: (r["n_flagged"], -r["threshold"])) if ok else None


def analyse_run(records, label="gap", signal="confidence", budget=DEFAULT_BUDGET, recall=DEFAULT_RECALL,
                bad_gap=DEFAULT_BAD_GAP, bad_lcs=DEFAULT_BAD_LCS, current=None, evaluate_at=None, resamples=RESAMPLES,
                tune=True) -> dict:
    """Everything for one run (the records of one system). `current` = the threshold in use now (default systems.CONFIDENCE_FLAG);
    `evaluate_at` = a fixed threshold to judge; tune=False (sealed splits) skips the sweep and the recommendations."""
    if label not in LABELS or signal not in SIGNALS:
        raise CalibrationError(f"label must be one of {LABELS}, signal one of {SIGNALS}")
    if not records:
        raise CalibrationError("no records")
    rows = rows_from_records(records, bad_gap, bad_lcs)
    current = SY.CONFIDENCE_FLAG if current is None else current
    n_parts = len({r["part_id"] for r in rows})
    out = {"system": records[0]["system"], "n_records": len(rows), "n_parts": n_parts, "label": label, "signal": signal,
           "label_text": LABEL_TEXT[label].format(bad_gap=bad_gap, bad_lcs=bad_lcs), "bad_gap": bad_gap, "bad_lcs": bad_lcs,
           "m4_sources": sorted({r["m4_source"] for r in rows if r["m4_source"]}), "current_threshold": current}
    out["scored"] = {s: sum(1 for r in rows if r["scores"][s] is not None) for s in SIGNALS}
    out["labelled"] = {l: sum(1 for r in rows if r["labels"][l] is not None) for l in LABELS}
    out["base_rates"] = {l: M.rate([r["labels"][l] for r in rows if r["labels"][l] is not None]) for l in LABELS}
    out["gap_percentiles"] = {f"p{q}": M.percentile([r["eff_gap"] for r in rows if r["eff_gap"] is not None], q / 100) for q in (10, 25, 50, 75, 90)}
    # unscored plans (fewer than two samples) are never flagged by the pipeline: how bad are they?
    unscored = [r for r in rows if r["scores"][signal] is None and r["labels"][label] is not None]
    out["unscored"] = {"n": len(unscored), "bad": sum(1 for r in unscored if r["labels"][label])}
    use = _usable(rows, signal, label)
    out["n_used"] = len(use)
    if not use:
        why = ""
        if signal == "confidence":
            if not any(_sample_routes(rec, "llm_steps") for rec in records):
                why = (" (no model answer in this run could be read as a route: cut off at the token limit, API errors or invalid JSON; "
                       "see parse, trunc and api-err in results/diagnostics.txt)")
            else:
                why = " (a run with a single sample has no confidence)"
        raise CalibrationError(f"no plan has both a {signal} score and a {label} label" + why)
    # how well does each score rank the bad plans, for every label
    out["auroc"] = {}
    for l in LABELS:
        for s in SIGNALS:
            u = _usable(rows, s, l)
            pairs = _pairs(u, s, l)
            val = auroc(pairs)
            out["auroc"][f"{s}|{l}"] = {"value": val, "n": len(u), "ci95": bootstrap(u, lambda rs, s=s, l=l: auroc(_pairs(rs, s, l)), resamples)
                                       if val is not None else (None, None)}
    out["current"] = evaluate(rows, signal, label, current, resamples)
    if evaluate_at is not None:
        out["evaluate_at"] = evaluate(rows, signal, label, evaluate_at, resamples)
    if tune:
        table = sweep(_pairs(use, signal, label))
        out["sweep"] = table
        rb = recommend_budget(table, budget)
        rr = recommend_recall(table, recall)
        out["budget"], out["recall_target"] = budget, recall
        out["recommend_budget"] = None if rb is None else evaluate(rows, signal, label, rb["threshold"], resamples)
        out["recommend_recall"] = None if rr is None else evaluate(rows, signal, label, rr["threshold"], resamples)
    return out


# ------------------------------------------------------------------ reliability (for the plot)
def reliability(rows, signal, label, bins=5) -> list:
    """Bad-plan rate in equal-count bins of the score (ties never split across bins)."""
    use = sorted(_usable(rows, signal, label), key=lambda r: r["scores"][signal])
    if not use:
        return []
    groups = []
    for r in use:
        if groups and groups[-1][-1]["scores"][signal] == r["scores"][signal]:
            groups[-1].append(r)
        else:
            groups.append([r])
    target = max(1, len(use) / bins)
    out, cur = [], []
    for g in groups:
        cur.extend(g)
        if len(cur) >= target:
            out.append(cur)
            cur = []
    if cur:
        if out and len(cur) < target / 2:
            out[-1].extend(cur)
        else:
            out.append(cur)
    return [{"score_min": min(r["scores"][signal] for r in b), "score_max": max(r["scores"][signal] for r in b), "n": len(b),
             "bad_rate": sum(1 for r in b if r["labels"][label]) / len(b)} for b in out]


# ------------------------------------------------------------------ text
def _f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _ci(ci, nd=2):
    return "" if not ci or ci[0] is None else f" [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def _line(title, r):
    return (f"  {title:<28} threshold {r['threshold']:<7g} flags {_pct(r['flagged_share'])}{_ci(r['flagged_share_ci95'])} "
            f"({r['n_flagged']}/{r['n']}), catches {r['caught']}/{r['n_bad']} bad plans: recall {_f(r['recall'])}{_ci(r['recall_ci95'])}, "
            f"precision {_f(r['precision'])}{_ci(r['precision_ci95'])}")


def format_report(res, rows_shown=14) -> str:
    L = [f"=== {res['system']}: {res['n_records']} plans of {res['n_parts']} parts ===",
         f"bad plan = {res['label_text']}",
         f"  base rate {_pct(res['base_rates'][res['label']])} of the {res['labelled'][res['label']]} labelled plans;"
         f" signal: {SIGNAL_TEXT[res['signal']]}; scored {res['scored'][res['signal']]}/{res['n_records']}"]
    if res["label"] == "lcs":
        L.append(f"  M4 reference: {'/'.join(res['m4_sources']) or 'none'}"
                 + ("  (proxy_rb is the Route Builder, not an expert: a weak label)" if "proxy_rb" in res["m4_sources"] else ""))
    if res["label"] == "gap":
        g = res["gap_percentiles"]
        L.append("  efficiency gap (points) p10/p25/p50/p75/p90: " + " / ".join(_f(g[k], 1) for k in ("p10", "p25", "p50", "p75", "p90"))
                 + "   (move --bad-gap if the base rate is near 0 or 1)")
    if res["unscored"]["n"]:
        L.append(f"  {res['unscored']['n']} plans have no {res['signal']} (fewer than two samples) and are never flagged;"
                 f" {res['unscored']['bad']} of them are bad")
    L.append("")
    L.append("Does the score rank bad plans below good ones? AUROC (0.5 = no information, 1 = perfect; 95% bootstrap over parts)")
    L.append(f"  {'score':<18}" + "".join(f"{l:>26}" for l in LABELS))
    for s in SIGNALS:
        cells = []
        for l in LABELS:
            a = res["auroc"][f"{s}|{l}"]
            cells.append(("n/a" if a["value"] is None else f"{a['value']:.2f}{_ci(a['ci95'])}") + f" n={a['n']}")
        L.append(f"  {s:<18}" + "".join(f"{c:>26}" for c in cells))
    L.append("")
    L.append(_line("now (CONFIDENCE_FLAG)", res["current"]))
    if "evaluate_at" in res:
        L.append(_line("fixed threshold asked for", res["evaluate_at"]))
    if res["current"]["n_bad"] in (0, res["current"]["n"]):
        which = "no plan" if res["current"]["n_bad"] == 0 else "every plan"
        L += ["", f"{which} is bad under this label, so no threshold can separate bad plans from good ones and none is recommended."
                  " Change the label (--bad-gap, --bad-lcs, or --label) so that both kinds exist."]
    elif "sweep" in res:
        tab = res["sweep"]
        if len(tab) > rows_shown:
            keep = {round(i * (len(tab) - 1) / (rows_shown - 1)) for i in range(rows_shown)}
            tab = [r for i, r in enumerate(tab) if i in keep]
        L += ["", f"Every threshold ({len(res['sweep'])} distinct flagged sets; flag when score < threshold):",
              f"  {'threshold':>10} {'flagged':>9} {'recall':>8} {'precision':>10} {'lift':>6}"]
        for r in tab:
            L.append(f"  {r['threshold']:>10g} {_pct(r['flagged_share']):>9} {_f(r['recall']):>8} {_f(r['precision']):>10} {_f(r['lift'], 1):>6}")
        L.append("")
        L.append(f"Recommendations (on this data: optimistic, the intervals ignore that the threshold was picked here):")
        if res["recommend_budget"]:
            L.append(_line(f"best recall within {_pct(res['budget'])} flagged", res["recommend_budget"]))
        else:
            L.append(f"  no threshold flags at most {_pct(res['budget'])} of the plans (the lowest-score group alone is bigger)")
        if res["recommend_recall"]:
            L.append(_line(f"fewest flags for recall {_pct(res['recall_target'])}", res["recommend_recall"]))
        else:
            L.append("  no bad plans in this label: nothing to catch")
        L += ["", "Use a threshold T:  python scripts/run_eval.py --system <system> --tag <tag> --confidence-flag T    (cached: no new API call)",
              "                    python scripts/run_eval.py --summarize-only <results folder> --confidence-flag T"]
    else:
        L.append("  (sealed split: no sweep, no recommendation; judge the threshold fixed on the development set)")
    return "\n".join(L)


def sweep_rows(res) -> list:
    """The sweep as flat rows (for the CSV)."""
    return [{"system": res["system"], "label": res["label"], "signal": res["signal"], "threshold": r["threshold"],
             "n_flagged": r["n_flagged"], "flagged_share": r["flagged_share"], "n_bad": r["n_bad"], "caught": r["caught"],
             "recall": r["recall"], "precision": r["precision"], "lift": r["lift"]} for r in res.get("sweep", [])]


# ------------------------------------------------------------------ plot
def plot(records, res, path):
    """Two panels: the bad-plan rate by score bin (reliability), and the share of bad plans caught against the share of plans flagged.
    Needs matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = rows_from_records(records, res["bad_gap"], res["bad_lcs"])
    signal, label = res["signal"], res["label"]
    blue, orange, grey = "#2a78d6", "#eb6834", "#8a8984"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    bins = reliability(rows, signal, label)
    xs = list(range(len(bins)))
    a1.bar(xs, [b["bad_rate"] for b in bins], color=blue, width=0.62)
    a1.axhline(res["base_rates"][label], color=grey, lw=1, ls="--")
    a1.text(len(bins) - 0.5, res["base_rates"][label], " all plans", color=grey, fontsize=8, va="bottom", ha="right")
    a1.set_xticks(xs)
    a1.set_xticklabels([f"{b['score_min']:.2f}" if b["score_min"] == b["score_max"] else f"{b['score_min']:.2f}-\n{b['score_max']:.2f}"
                        for b in bins], fontsize=8)
    for x, b in zip(xs, bins):
        a1.text(x, b["bad_rate"], f"n={b['n']}", ha="center", va="bottom", fontsize=8, color="#52514e")
    a1.set_ylim(0, 1.08)
    a1.set_xlabel(f"{signal} (low to high)")
    a1.set_ylabel("share of plans that are bad")
    a1.set_title("Do low scores go with bad plans?", fontsize=10)
    tab = res.get("sweep")
    if tab:
        a2.plot([0, 1], [0, 1], color=grey, lw=1, ls="--")
        a2.text(0.5, 0.47, "flagging at random", color=grey, fontsize=8, rotation=38, ha="center", va="top")
        a2.step([r["flagged_share"] for r in tab], [r["recall"] or 0 for r in tab], where="post", color=blue, lw=2)
        for key, name in (("recommend_budget", "budget"), ("recommend_recall", "recall target")):
            r = res.get(key)
            if r and r["recall"] is not None:
                a2.scatter([r["flagged_share"]], [r["recall"]], s=40, color=orange, zorder=3, edgecolor="white", linewidth=1.5)
                a2.annotate(f"{name}: < {r['threshold']:g}", (r["flagged_share"], r["recall"]), textcoords="offset points", xytext=(8, -12), fontsize=8)
        if res["current"]["n_bad"] == 0:
            a2.text(0.5, 0.5, "no bad plans under this label", ha="center", va="center", fontsize=9, color=grey)
    a2.set_xlim(0, 1.02)
    a2.set_ylim(0, 1.02)
    a2.set_xlabel("share of plans flagged for review")
    a2.set_ylabel("share of bad plans caught")
    a2.set_title("What a threshold buys", fontsize=10)
    for ax in (a1, a2):
        ax.grid(axis="y", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"{res['system']}: {res['label_text']}", fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
