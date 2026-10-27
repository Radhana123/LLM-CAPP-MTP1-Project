"""Risk-aware comparison of two finished runs (Step 21): is the LLM route really worse than the Route Builder route, once the physical
risks of the two routes are looked at as well as the v1 Efficiency?

Why: the v1 Efficiency (time, cost, energy, changeovers) does not see physical risk. A route that adds Reaming for a 0.01 mm hole
tolerance is slower and costs more, so its Efficiency is lower, and it clears a feasibility risk that the shorter route carries. A gap in
Efficiency between the LLM and the Route Builder may therefore be partly an artefact of what the score cannot see. compare_runs.py and
part_diff.py show both columns side by side; this module puts them in one picture. Pure functions on the pair rows of capp/partdiff.py
(no API calls). The v1 Efficiency itself is not changed.

Route flags of one route = its hard infeasibilities + its route risks (capp/feasibility.py; with physics=True also: over machine power,
and Ra risk 'high' in capp/physics.py). A count of findings, not a severity: two minor risks weigh the same as one hard one.

What is computed on the comparable pairs (both finals strict-valid) that have feasibility data in both runs:
    mean Efficiency difference (B minus A), mean flags of A and of B, the mean reduction (flags of A minus flags of B)
    a 3 x 3 table: Efficiency better / equal / worse x B has fewer / the same / more flags than A
    the pairs where B is worse on Efficiency but has fewer flags, and the opposite: which rules were cleared or added, which operations B
        adds (that is what the lost Efficiency bought)
    pairs split by which side has flags (neither / only A / only B / both), with the mean Efficiency difference of each group
    the penalty view: adjusted difference = Efficiency difference + P x (flags of A - flags of B), for P = 0, 1, 2, 5, 10 Efficiency
        points per flag, with a 95% interval; and the break-even P*, the value of one flag at which B matches A on average
    per rule: on how many pairs A / B has it, cleared, added

P is not estimated here: it is a statement of how much one flag is worth, which the thesis (or the supervisor) has to argue. The table
says what the comparison looks like for each P, and P* says how large it must be for the gap to close.

Limits: a count of findings is a coarse risk measure; the feasibility rules are indicative until their sources are confirmed
(capp/feasibility.py), so a cleared flag is a cleared model finding, not a verified shop-floor risk; the intervals resample PARTS (repeats
of a part are not independent); with fewer than about 30 parts every interval is wide.

    rows, _ = partdiff.pair_rows(base_records, other_records)
    s = riskview.analyze(rows)
    print(riskview.render(s, "rb", "n5_r1_s3"))
"""
import csv
import random
from collections import Counter, defaultdict

from capp import partdiff as PD

EPS = PD.EPS
PENALTIES = (0.0, 1.0, 2.0, 5.0, 10.0)
RESAMPLES = 1000
SEED = 7
INF = float("inf")


class RiskViewError(ValueError):
    pass


# ---------------------------------------------------------------- flags of one route
def _count(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def route_flags(row: dict, side: str, physics: bool = False):
    """Number of findings of one route of a pair row (side 'base' or 'other'), or None when the run did not analyse it."""
    hard, risk = _count(row.get("feas_infeasible_" + side)), _count(row.get("feas_risk_" + side))
    if hard is None or risk is None:
        return None
    n = int(hard) + int(risk)
    if physics:
        over, ra = row.get("over_power_" + side, ""), row.get("ra_risk_" + side, "")
        if over == "" or ra == "":
            return None
        n += int(bool(over)) + int(ra == "high")
    return n


def _rules(row: dict, side: str) -> set:
    s = row.get("feas_rules_" + side) or ""
    return {x for x in str(s).split(";") if x}


def pairs_for_analysis(rows: list, physics: bool = False):
    """(pairs, counts). Comparable pairs (both finals valid) whose two routes have flags; counts = {"comparable", "no_risk_data"}."""
    pairs, comparable, no_data = [], 0, 0
    for r in rows:
        if r.get("eff_diff") is None:
            continue
        comparable += 1
        fb, fo = route_flags(r, "base", physics), route_flags(r, "other", physics)
        if fb is None or fo is None:
            no_data += 1
            continue
        rb, ro = _rules(r, "base"), _rules(r, "other")
        pairs.append({"part_id": r["part_id"], "repeat": r["repeat"], "eff_diff": r["eff_diff"], "flags_base": fb, "flags_other": fo,
                      "reduction": fb - fo, "rules_base": sorted(rb), "rules_other": sorted(ro), "rules_cleared": sorted(rb - ro),
                      "rules_added": sorted(ro - rb), "ops_extra": list(r.get("only_other") or []),
                      "ops_missing": list(r.get("only_base") or []), "cause": r.get("cause", ""),
                      "route_base": r.get("route_base", ""), "route_other": r.get("route_other", "")})
    return pairs, {"comparable": comparable, "no_risk_data": no_data}


# ---------------------------------------------------------------- small statistics
def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _risk_verdict(reduction: int) -> str:
    return "fewer" if reduction > 0 else "more" if reduction < 0 else "same"


def _stats(pairs: list) -> dict:
    d = [p["eff_diff"] for p in pairs]
    return {"n": len(pairs), "mean_eff": _mean(d), "mean_reduction": _mean(p["reduction"] for p in pairs),
            "better": sum(1 for x in d if x > EPS), "equal": sum(1 for x in d if abs(x) <= EPS), "worse": sum(1 for x in d if x < -EPS)}


def break_even(mean_eff, mean_reduction):
    """(P*, status). P*: Efficiency points one flag must be worth for B to match A on average.
    status 'not worse' (P* = 0: B is not worse than A on Efficiency even without valuing flags), 'never' (P* = INF: B does not have fewer
    flags than A on average, so no value of P closes the gap), 'value', or 'none' (P* None: no pairs)."""
    if mean_eff is None or mean_reduction is None:
        return None, "none"
    if mean_eff >= -EPS:
        return 0.0, "not worse"
    if mean_reduction <= EPS:
        return INF, "never"
    return -mean_eff / mean_reduction, "value"


def _percentile_ci(values, lo=0.025, hi=0.975):
    v = sorted(values)
    n = len(v)
    return v[int(lo * (n - 1))], v[int(round(hi * (n - 1)))]


def _bootstrap(pairs: list, penalties, resamples: int, seed: int):
    """Percentile intervals by resampling PARTS (a part's repeats stay together). Returns (penalty intervals, break-even interval) or
    (None, None) with fewer than 2 parts."""
    by_part = defaultdict(lambda: [0.0, 0.0, 0])
    for p in pairs:
        c = by_part[p["part_id"]]
        c[0] += p["eff_diff"]
        c[1] += p["reduction"]
        c[2] += 1
    clusters = list(by_part.values())
    k = len(clusters)
    if k < 2:
        return None, None
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        se = sr = n = 0
        for _ in range(k):
            c = clusters[rng.randrange(k)]
            se += c[0]
            sr += c[1]
            n += c[2]
        means.append((se / n, sr / n))
    ci = {pen: _percentile_ci([e + pen * r for e, r in means]) for pen in penalties}
    star = _percentile_ci([break_even(e, r)[0] for e, r in means])
    return ci, star


# ---------------------------------------------------------------- the analysis
def analyze(rows: list, physics: bool = False, penalties=PENALTIES, resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    pairs, counts = pairs_for_analysis(rows, physics)
    out = {"physics": physics, "n_rows": len(rows), "n_comparable": counts["comparable"], "n_no_risk_data": counts["no_risk_data"],
           "n_pairs": len(pairs), "pairs": pairs}
    if not pairs:
        return out
    overall = _stats(pairs)
    out["overall"] = overall
    out["flags_base"] = _mean(p["flags_base"] for p in pairs)
    out["flags_other"] = _mean(p["flags_other"] for p in pairs)
    out["parts"] = len({p["part_id"] for p in pairs})

    quad = {}
    for ev in ("better", "equal", "worse"):
        for rv in ("fewer", "same", "more"):
            sel = [p for p in pairs if PD._verdict(p["eff_diff"]) == ev and _risk_verdict(p["reduction"]) == rv]
            quad[(ev, rv)] = {"n": len(sel), "mean_eff": _mean(p["eff_diff"] for p in sel)}
    out["quadrants"] = quad

    def trade(eff_v, risk_v):
        sel = [p for p in pairs if PD._verdict(p["eff_diff"]) == eff_v and _risk_verdict(p["reduction"]) == risk_v]
        cleared, added, extra, missing = Counter(), Counter(), Counter(), Counter()
        for p in sel:
            cleared.update(p["rules_cleared"])
            added.update(p["rules_added"])
            extra.update(set(p["ops_extra"]))
            missing.update(set(p["ops_missing"]))
        return {"n": len(sel), "mean_eff": _mean(p["eff_diff"] for p in sel), "mean_reduction": _mean(p["reduction"] for p in sel),
                "rules_cleared": cleared.most_common(), "rules_added": added.most_common(),
                "ops_extra": extra.most_common(), "ops_missing": missing.most_common(),
                "pairs": sorted(sel, key=lambda p: p["eff_diff"])}
    out["worse_fewer"] = trade("worse", "fewer")
    out["better_more"] = trade("better", "more")

    split = {"neither": [], "only base": [], "only other": [], "both": []}
    for p in pairs:
        b, o = p["flags_base"] > 0, p["flags_other"] > 0
        split["both" if b and o else "only base" if b else "only other" if o else "neither"].append(p)
    out["split"] = {k: _stats(v) for k, v in split.items()}

    star, status = break_even(overall["mean_eff"], overall["mean_reduction"])
    ci, star_ci = _bootstrap(pairs, tuple(penalties), resamples, seed)
    rows_p = []
    for pen in penalties:
        adj = [p["eff_diff"] + pen * p["reduction"] for p in pairs]
        rows_p.append({"penalty": pen, "mean": _mean(adj), "ci95": None if ci is None else ci[pen],
                       "better": sum(1 for x in adj if x > EPS), "equal": sum(1 for x in adj if abs(x) <= EPS),
                       "worse": sum(1 for x in adj if x < -EPS)})
    out["penalties"] = rows_p
    out["break_even"] = {"value": star, "status": status, "ci95": star_ci}

    names = sorted({r for p in pairs for r in p["rules_base"] + p["rules_other"]})
    per_rule = []
    for name in names:
        cleared = [p for p in pairs if name in p["rules_cleared"]]
        added = [p for p in pairs if name in p["rules_added"]]
        per_rule.append({"rule": name, "base_has": sum(name in p["rules_base"] for p in pairs),
                         "other_has": sum(name in p["rules_other"] for p in pairs), "cleared": len(cleared), "added": len(added),
                         "mean_eff_when_cleared": _mean(p["eff_diff"] for p in cleared),
                         "mean_eff_when_added": _mean(p["eff_diff"] for p in added)})
    out["rules"] = sorted(per_rule, key=lambda r: (-(r["cleared"] + r["added"]), r["rule"]))
    return out


# ---------------------------------------------------------------- report
def _f(x, nd=2):
    return "n/a" if x is None else ("never" if x == INF else f"{x:.{nd}f}")


def _ci(ci, nd=2):
    return "" if not ci else f" [{_f(ci[0], nd)}, {_f(ci[1], nd)}]"


def _top(counter, k=6):
    return ", ".join(f"{name} x{n}" for name, n in counter[:k]) or "none"


def _pair_line(p, base_label, other_label):
    ops = " ".join([f"-{o}" for o in p["ops_missing"]] + [f"+{o}" for o in p["ops_extra"]]) or "same operations"
    rules = ""
    if p["rules_cleared"]:
        rules += f"  cleared {', '.join(p['rules_cleared'])}"
    if p["rules_added"]:
        rules += f"  added {', '.join(p['rules_added'])}"
    return (f"  {p['part_id']}/{p['repeat']:<2} Efficiency {p['eff_diff']:+.2f}  flags {p['flags_base']} -> {p['flags_other']}{rules}  ({ops})")


def render(s: dict, base_label: str = "A", other_label: str = "B", top: int = 8) -> str:
    kind = "hard infeasibilities + route risks" + (" + over power + Ra risk high" if s["physics"] else "")
    head = [f"Risk-aware comparison, {other_label} minus {base_label} (the first run is the reference; a negative Efficiency difference means {other_label} is worse)",
            f"Flags of a route = {kind}: a count of findings, not a severity. The v1 Efficiency does not see them.",
            f"{s['n_rows']} pairs, {s['n_comparable']} comparable (both finals strict-valid), {s['n_pairs']} with feasibility data in both runs"
            + (f" ({s['n_no_risk_data']} left out: not analysed)" if s["n_no_risk_data"] else "") + "."]
    if not s["n_pairs"]:
        reason = ("no pair is comparable: a final is not strict-valid" if not s["n_comparable"]
                  else "feasibility was not analysed in these runs (run them with the Step 11 harness)")
        return "\n".join(head + ["", "Nothing to compare: " + reason + "."])
    o = s["overall"]
    out = head + ["",
                  f"Mean Efficiency difference {_f(o['mean_eff'])} pts over {s['n_pairs']} pairs ({s['parts']} parts): better {o['better']}, equal {o['equal']}, worse {o['worse']}.",
                  f"Mean flags per route: {base_label} {_f(s['flags_base'])}, {other_label} {_f(s['flags_other'])}; reduction {_f(o['mean_reduction'])} per route "
                  f"({'positive: ' + other_label + ' has fewer' if o['mean_reduction'] > EPS else 'negative: ' + other_label + ' has more' if o['mean_reduction'] < -EPS else 'none on average'}).",
                  "", f"Efficiency against flags (pairs, mean Efficiency difference in brackets); columns: {other_label} has ... flags than {base_label}:",
                  f"  {'':<22}{'fewer':>16}{'the same':>16}{'more':>16}"]
    for ev in ("better", "equal", "worse"):
        cells = []
        for rv in ("fewer", "same", "more"):
            q = s["quadrants"][(ev, rv)]
            cells.append(f"{q['n']} ({_f(q['mean_eff'], 1)})" if q["n"] else "0")
        out.append(f"  {other_label + ' ' + ev:<22}" + "".join(f"{c:>16}" for c in cells))

    wf = s["worse_fewer"]
    out += ["", f"{other_label} worse on Efficiency but with fewer flags: {wf['n']} pair(s)"
            + (f", mean Efficiency {_f(wf['mean_eff'])}, mean flags cleared {_f(wf['mean_reduction'])}" if wf["n"] else "")]
    if wf["n"]:
        out += [f"  rules cleared: {_top(wf['rules_cleared'])}", f"  operations {other_label} adds: {_top(wf['ops_extra'])}",
                f"  operations {other_label} lacks: {_top(wf['ops_missing'])}"]
        out += [_pair_line(p, base_label, other_label) for p in wf["pairs"][:top]]
        if wf["n"] > top:
            out.append(f"  ... {wf['n'] - top} more (--top)")
    bm = s["better_more"]
    out += ["", f"{other_label} better on Efficiency but with more flags: {bm['n']} pair(s)"
            + (f", mean Efficiency {_f(bm['mean_eff'])}, mean flags added {_f(-bm['mean_reduction'])}" if bm["n"] else "")]
    if bm["n"]:
        out += [f"  rules added: {_top(bm['rules_added'])}", f"  operations {other_label} lacks: {_top(bm['ops_missing'])}"]
        out += [_pair_line(p, base_label, other_label) for p in sorted(bm["pairs"], key=lambda p: -p["eff_diff"])[:top]]

    out += ["", "Pairs by which route has flags (mean Efficiency difference of each group):"]
    groups = (("neither", "neither route has flags"), ("only base", f"only {base_label} has flags"),
              ("only other", f"only {other_label} has flags"), ("both", "both have flags"))
    w = max(len(label) for _, label in groups)
    for key, label in groups:
        g = s["split"][key]
        out.append(f"  {label:<{w}} n={g['n']:<3} mean {_f(g['mean_eff']):>6}  better {g['better']:<3} equal {g['equal']:<3} worse {g['worse']}")
    out.append("  The 'neither' row is the comparison on equal footing: no flag on either side to value.")

    out += ["", f"If one flag were worth P Efficiency points (adjusted difference = Efficiency difference + P x flags {base_label} has more than {other_label}):"]
    for r in s["penalties"]:
        out.append(f"  P = {r['penalty']:<4g} mean {_f(r['mean']):>6}{_ci(r['ci95'])}   better {r['better']}, equal {r['equal']}, worse {r['worse']}")
    be = s["break_even"]
    if be["status"] == "not worse":
        out.append(f"  Break-even P*: not needed, {other_label} is not worse than {base_label} on Efficiency even at P = 0{_ci(be['ci95'])}.")
    elif be["status"] == "never":
        out.append(f"  Break-even P*: none{_ci(be['ci95'])}. {other_label} does not have fewer flags than {base_label} on average, "
                   "so valuing flags does not close the gap.")
    else:
        out.append(f"  Break-even P* = {_f(be['value'])}{_ci(be['ci95'])} Efficiency points per flag: at this value {other_label} matches {base_label} on average.")

    if s["rules"]:
        wb, wo = max(8, len(base_label) + 5), max(8, len(other_label) + 5)
        out += ["", "Per rule (pairs where the rule is raised on the route; cleared = raised on " + base_label + " only, added = on " + other_label + " only):",
                f"  {'rule':<26}{base_label + ' has':>{wb}}{other_label + ' has':>{wo}}{'cleared':>9}{'added':>7}   mean Efficiency when cleared / added"]
        for r in s["rules"][:12]:
            out.append(f"  {r['rule']:<26}{r['base_has']:>{wb}}{r['other_has']:>{wo}}{r['cleared']:>9}{r['added']:>7}   "
                       f"{_f(r['mean_eff_when_cleared'], 1)} / {_f(r['mean_eff_when_added'], 1)}")
    out += ["", "Read with care: P is a value judgement, not an estimate; flags are model findings from indicative rules, not shop-floor facts; "
            "intervals resample parts, and with a few parts they are wide. Look at the routes of the listed pairs before concluding anything."]
    return "\n".join(out)


def write_csv(path: str, s: dict):
    """One row per analysed pair."""
    if not s["pairs"]:
        raise RiskViewError("no pairs to write")
    cols = ["part_id", "repeat", "eff_diff", "flags_base", "flags_other", "reduction", "rules_cleared", "rules_added", "ops_extra", "ops_missing",
            "cause", "route_base", "route_other"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in s["pairs"]:
            w.writerow([";".join(p[c]) if isinstance(p[c], list) else p[c] for c in cols])
