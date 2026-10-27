"""M9, extraction quality (report 6.5 and 7.x, WP7, RQ6): precision, recall and F1 of the feature extraction on annotated drawings, per pass,
for the union of both passes (v1), for their intersection and for the agreement-based variant, with the corrections each one asks of the user.

    from capp import extraction_eval as EE
    truth = EE.load_ground_truth("data/extraction_truth.json")
    records = EE.run("drawings/", truth, "results/fe1_extraction", tag="fe1")      # live API calls (cached); scripts/extraction_eval.py run
    EE.format_report(EE.score(records))                                            # offline; scripts/extraction_eval.py score

Per drawing G = the ground-truth feature set, g = what the geometry pass reported, a = annotation pass, u = g | a (v1), b = g & a.
Variants (all on the drawings where BOTH passes worked, so that they are compared on the same drawings):
  geometry, annotation   a pass alone: list shown = g or a
  union                  v1: list shown = u
  both                   list shown = b (what the agreement rule pre-accepts)
  agreement              report 6.5: b is pre-accepted, f = u - b is highlighted and the user confirms or rejects each. Scored AFTER the
                         confirmation of the highlighted features by an ORACLE user (accepts f in G, rejects f not in G): the list is
                         L = b | (f & G). That is an upper bound for what a real user does; the corrections log of the interface
                         (capp/extraction_review.py) measures the real one.
Corrections per drawing = the edits that turn the list shown into G: union, passes, both: |list - G| + |G - list|. Agreement: wrong
pre-accepted features to remove + features both passes missed to add = |b - G| + |G - u|; the |f| decisions are counted apart as
"confirmations", because that is the price of the rule. RQ6 asks whether corrections fall without the confirmed list getting worse.

Limits: the ground truth is as good as the person who wrote it (two annotators on a subset give the agreement ceiling); a feature list
is a set (a drawing with three holes lists Hole once, as in v1); the drawings used to improve prompts must not be those used to
report M9 (develop on a development subset, freeze the prompts, report on the rest); with 30-50 drawings every interval is wide.
"""
import csv
import json
import os
import random
import statistics
from collections import Counter

from capp import extraction_review as ER
from capp import legacy  # noqa: F401
import feature_vocab as fv

IMAGE_EXT = (".png", ".jpg", ".jpeg")
VARIANTS = ("geometry", "annotation", "union", "both", "agreement")
LABELS = {"geometry": "geometry pass alone", "annotation": "annotation pass alone", "union": "union of both passes (v1)",
          "both": "both passes (intersection)", "agreement": "agreement-based, oracle-confirmed"}
RESAMPLES = 1000
HOLE_TOL_MM = 0.05


class ExtractionEvalError(RuntimeError):
    pass


# ---------------------------------------------------------------- ground truth and drawings
def load_ground_truth(path: str) -> dict:
    """{drawing file name: {"features": [...], "hole_diameters": [mm] or None}} from a JSON file:
    {"a.png": ["Hole", "Slot"], "b.png": {"features": ["Hole"], "hole_diameters": [8, 6.5]}} (optionally inside {"drawings": {...}}).
    Every feature must be one of the vocabulary; an empty list is refused (a template entry that was never filled in)."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        raise ExtractionEvalError(f"cannot read the ground truth {path}: {exc}") from exc
    if isinstance(raw, dict) and isinstance(raw.get("drawings"), dict):
        raw = raw["drawings"]
    if not isinstance(raw, dict) or not raw:
        raise ExtractionEvalError(f"{path}: expected an object {{drawing file: features}}")
    out, problems = {}, []
    for name, v in raw.items():
        feats, holes = (v.get("features"), v.get("hole_diameters")) if isinstance(v, dict) else (v, None)
        if not isinstance(feats, list) or not all(isinstance(x, str) for x in feats):
            problems.append(f"{name}: features must be a list of names")
            continue
        if not feats:
            problems.append(f"{name}: no features written down yet")
            continue
        bad = [x for x in feats if not fv.is_valid_feature(x)]
        if bad:
            problems.append(f"{name}: not in the vocabulary: {', '.join(bad)}")
            continue
        if len(set(feats)) != len(feats):
            problems.append(f"{name}: a feature is listed twice")
            continue
        if holes is not None and (not isinstance(holes, list) or not all(isinstance(h, (int, float)) and not isinstance(h, bool) and h > 0 for h in holes)):
            problems.append(f"{name}: hole_diameters must be a list of positive numbers (mm)")
            continue
        out[name] = {"features": list(feats), "hole_diameters": None if holes is None else [float(h) for h in holes]}
    if problems:
        raise ExtractionEvalError(f"{path}: " + "; ".join(problems[:10]) + (f" (and {len(problems) - 10} more)" if len(problems) > 10 else ""))
    return out


def list_images(folder: str) -> list:
    if not os.path.isdir(folder):
        raise ExtractionEvalError(f"no such folder: {folder}")
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXT))


def template(folder: str) -> dict:
    """A ground-truth file to fill in: every drawing of the folder with an empty feature list."""
    return {name: {"features": [], "hole_diameters": None} for name in list_images(folder)}


def match_drawings(folder: str, truth: dict) -> tuple:
    """(pairs [(name, path)], truth without drawing, drawings without truth). A truth key may leave out the file extension."""
    images = list_images(folder)
    stems = {}
    for f in images:
        stems.setdefault(os.path.splitext(f)[0], []).append(f)
    pairs, no_image, used = [], [], set()
    for key in truth:
        if key in images:
            pairs.append((key, os.path.join(folder, key)))
            used.add(key)
        elif len(stems.get(key, [])) == 1:
            pairs.append((key, os.path.join(folder, stems[key][0])))
            used.add(stems[key][0])
        else:
            no_image.append(key)
    return pairs, no_image, [f for f in images if f not in used]


# ---------------------------------------------------------------- running the extractor
def _format_of(path: str) -> str:
    return "png" if path.lower().endswith(".png") else "jpeg"


def record_from_extraction(name: str, extraction: dict, truth: dict) -> dict:
    """One record of the run: what the extractor gave for a drawing, next to the ground truth. Keeps everything the scoring and the review
    logic need (the per-pass lists), not the image."""
    ok = bool(extraction.get("success"))
    rec = {"drawing": name, "truth": list(truth["features"]), "truth_holes": truth.get("hole_diameters"), "success": ok,
           "error": "" if ok else str(extraction.get("error", "")), "features": list(extraction.get("features") or []),
           "rejected_features": list(extraction.get("rejected_features") or []), "passes": extraction.get("passes") or {},
           "passes_ok": extraction.get("passes_ok"), "sample_agreement": extraction.get("sample_agreement"),
           "confidence": extraction.get("confidence"), "notes": extraction.get("notes", "")}
    if "dimensions" in extraction:
        rec["hole_diameters"] = list(extraction.get("hole_diameters") or [])
        rec["dimensions_ok"] = bool(extraction["dimensions"].get("success"))
    return rec


def run(folder: str, truth: dict, out_dir: str, tag: str = "fe", with_dimensions: bool = False, with_evidence: bool = False,
        limit: int = None, extractor=None, progress=None) -> list:
    """Extract every drawing that has a ground truth, write results/<out_dir>/records.jsonl and run.json, return the records. The calls go
    through capp.llm_client (cache; CAPP_LLM_MODE live | replay | nocache), under the run tag f'{tag}_extraction'."""
    pairs, no_image, no_truth = match_drawings(folder, truth)
    if not pairs:
        raise ExtractionEvalError("no drawing of the folder has a ground truth: check the file names in the truth file")
    if limit:
        pairs = pairs[:limit]
    if extractor is None:
        import image_feature_extractor as ife
        extractor = ife.extract_features_from_image
    old = os.environ.get("CAPP_RUN_TAG")
    os.environ["CAPP_RUN_TAG"] = f"{tag}_extraction"
    records = []
    try:
        for i, (name, path) in enumerate(pairs, start=1):
            with open(path, "rb") as f:
                data = f.read()
            kw = {"with_dimensions": True} if with_dimensions else {}
            if with_evidence:
                kw["with_evidence"] = True
            try:
                ex = extractor(data, image_format=_format_of(path), **kw)
            except Exception as exc:                                  # one bad drawing must not lose the others
                ex = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
            records.append(record_from_extraction(name, ex, truth[name]))
            if i == 1 and not ex.get("success") and "not configured" in str(ex.get("error", "")):
                raise ExtractionEvalError("GROQ_API_KEY is not configured: no extraction can run (use --mode replay on a cached run)")
            if progress:
                progress(f"[{i}/{len(pairs)}] {name}: " + ("ok" if ex.get("success") else f"FAILED {ex.get('error', '')}"))
    finally:
        if old is None:
            os.environ.pop("CAPP_RUN_TAG", None)
        else:
            os.environ["CAPP_RUN_TAG"] = old
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "records.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {"tag": tag, "folder": os.path.abspath(folder), "n_drawings": len(records), "with_dimensions": with_dimensions,
            "with_evidence": with_evidence, "truth_without_image": no_image, "images_without_truth": no_truth,
            "llm_mode": os.getenv("CAPP_LLM_MODE", "live")}
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return records


def load_records(run_dir: str) -> list:
    path = os.path.join(run_dir, "records.jsonl")
    if not os.path.exists(path):
        raise ExtractionEvalError(f"no records.jsonl in {run_dir}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- scoring
def _sets(rec) -> dict:
    """The sets of one complete record: g, a, u, b, f (highlighted), L (list after oracle confirmation), G."""
    g = set(rec["passes"]["geometry"].get("features", []))
    a = set(rec["passes"]["annotation"].get("features", []))
    G = set(rec["truth"])
    u, b = g | a, g & a
    return {"geometry": g, "annotation": a, "union": u, "both": b, "G": G, "flagged": u - b, "agreement": b | ((u - b) & G)}


def _complete(rec) -> bool:
    p = rec.get("passes") or {}
    return bool(rec.get("success")) and all(isinstance(p.get(n), dict) and p[n].get("success") for n in ER.PASSES)


def _counts(pred: set, G: set) -> tuple:
    return len(pred & G), len(pred - G), len(G - pred)


def _drawing_f1(pred: set, G: set) -> float:
    tp, fp, fn = _counts(pred, G)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0


def _corrections(variant: str, s: dict) -> int:
    if variant == "agreement":
        return len(s["both"] - s["G"]) + len(s["G"] - s["union"])
    pred = s[variant]
    return len(pred - s["G"]) + len(s["G"] - pred)


def _ci(values, stat, resamples=RESAMPLES, seed=7):
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    est = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    return est[int(0.025 * (resamples - 1))], est[int(round(0.975 * (resamples - 1)))]


def _micro_f1(triples):
    tp, fp, fn = (sum(t[i] for t in triples) for i in range(3))
    return ER.prf(tp, fp, fn)["f1"] or 0.0


def score(records: list, resamples: int = RESAMPLES) -> dict:
    """The M9 tables from a run's records (offline). Records whose extraction failed, or in which a pass failed, are left out of every
    row and listed in "excluded": the variants are compared on the same drawings."""
    use = [r for r in records if _complete(r)]
    excluded = [{"drawing": r["drawing"], "reason": r.get("error") or "a pass failed"} for r in records if not _complete(r)]
    if not use:
        raise ExtractionEvalError("no drawing has both passes: nothing to score" + (f" ({excluded[0]['reason']})" if excluded else ""))
    S = [_sets(r) for r in use]
    rows = {}
    for v in VARIANTS:
        triples = [_counts(s[v], s["G"]) for s in S]
        pooled = ER.prf(sum(t[0] for t in triples), sum(t[1] for t in triples), sum(t[2] for t in triples))
        corr = [_corrections(v, s) for s in S]
        rows[v] = {"variant": v, "label": LABELS[v], **pooled,
                   "f1_ci95": _ci(triples, _micro_f1, resamples),
                   "drawing_f1_mean": statistics.mean(_drawing_f1(s[v], s["G"]) for s in S),
                   "exact_share": sum(s[v] == s["G"] for s in S) / len(S),
                   "corrections_mean": statistics.mean(corr), "corrections_ci95": _ci(corr, statistics.mean, resamples),
                   "confirmations_mean": statistics.mean(len(s["flagged"]) for s in S) if v == "agreement" else 0.0,
                   "shown_mean": statistics.mean(len(s["union"] if v == "agreement" else s[v]) for s in S)}
    flagged = [s["flagged"] for s in S]
    n_flag = sum(len(f) for f in flagged)
    n_flag_ok = sum(len(s["flagged"] & s["G"]) for s in S)
    n_pre = sum(len(s["both"]) for s in S)
    n_pre_ok = sum(len(s["both"] & s["G"]) for s in S)
    overlap = {"union_features": n_flag + n_pre, "seen_by_both": n_pre, "seen_by_one": n_flag,
               "both_correct_share": n_pre_ok / n_pre if n_pre else None, "one_correct_share": n_flag_ok / n_flag if n_flag else None,
               "flagged_share": n_flag / (n_flag + n_pre) if n_flag + n_pre else None,
               "flagged_per_drawing": n_flag / len(S), "drawings_with_nothing_flagged": sum(not f for f in flagged) / len(S)}
    feats = sorted({f for s in S for k in ("G", "union") for f in s[k]})
    per_feature = []
    for f in feats:
        row = {"feature": f, "support": sum(f in s["G"] for s in S)}
        for v in ("geometry", "annotation", "union", "both"):
            tp = sum(f in s[v] and f in s["G"] for s in S)
            fp = sum(f in s[v] and f not in s["G"] for s in S)
            fn = sum(f not in s[v] and f in s["G"] for s in S)
            row[v] = ER.prf(tp, fp, fn)
        per_feature.append(row)
    unrec = Counter()
    for r in use:
        unrec.update(r.get("rejected_features") or [])
    holes = _hole_scores([r for r in use if r.get("truth_holes") is not None and "hole_diameters" in r])
    return {"n_drawings": len(records), "n_scored": len(use), "excluded": excluded, "variants": [rows[v] for v in VARIANTS],
            "overlap": overlap, "per_feature": per_feature, "holes": holes,
            "unrecognised": [{"name": n, "count": c, "suggestion": ER.nearest_vocabulary(n)} for n, c in unrec.most_common()],
            "per_drawing": [{"drawing": r["drawing"], "truth": sorted(s["G"]), "geometry": sorted(s["geometry"]), "annotation": sorted(s["annotation"]),
                             "highlighted": sorted(s["flagged"]), "missed_by_both": sorted(s["G"] - s["union"]),
                             **{f"corrections_{v}": _corrections(v, s) for v in VARIANTS}} for r, s in zip(use, S)]}


def _hole_scores(recs) -> dict:
    """The dimension pass (Step 14) against the hole diameters of the ground truth, where the truth has them: a read diameter matches a
    true one within HOLE_TOL_MM; each true diameter is matched once."""
    if not recs:
        return {}
    tp = fp = fn = exact = 0
    for r in recs:
        truth, read = list(r["truth_holes"]), list(r["hole_diameters"])
        matched = 0
        left = list(truth)
        for h in read:
            hit = next((t for t in left if abs(t - h) <= HOLE_TOL_MM), None)
            if hit is not None:
                left.remove(hit)
                matched += 1
        tp += matched
        fp += len(read) - matched
        fn += len(truth) - matched
        exact += (len(read) == len(truth) == matched)
    out = ER.prf(tp, fp, fn)
    out.update({"drawings": len(recs), "exact_set_share": exact / len(recs), "failed_calls": sum(not r.get("dimensions_ok", True) for r in recs)})
    return out


# ---------------------------------------------------------------- report
def _f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ci_text(ci, nd=2):
    return "" if not ci or ci[0] is None else f" [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def format_report(s: dict, label: str = "") -> str:
    L = [f"=== M9 extraction quality{' - ' + label if label else ''}: {s['n_scored']} of {s['n_drawings']} drawings scored ==="]
    if s["excluded"]:
        L.append("Left out (extraction or a pass failed): " + "; ".join(f"{e['drawing']} ({e['reason'][:60]})" for e in s["excluded"][:6])
                 + (" ..." if len(s["excluded"]) > 6 else ""))
    L += ["", "Per variant (pooled over drawings; 95% intervals resample drawings):",
          f"  {'variant':36} {'precision':>9} {'recall':>7} {'F1':>6} {'F1 95% CI':>14} {'exact':>6} {'corr/dwg':>9} {'confirm':>8} {'shown':>6}"]
    for r in s["variants"]:
        ci = r["f1_ci95"]
        L.append(f"  {r['label']:36} {_f(r['precision']):>9} {_f(r['recall']):>7} {_f(r['f1']):>6} "
                 f"{('[%.2f, %.2f]' % ci) if ci and ci[0] is not None else '':>14} {r['exact_share'] * 100:>5.0f}% "
                 f"{r['corrections_mean']:>9.2f} {r['confirmations_mean']:>8.2f} {r['shown_mean']:>6.1f}")
    u, a = s["variants"][2], s["variants"][4]
    L += ["",
          f"  corrections per drawing: union {u['corrections_mean']:.2f}{_ci_text(u['corrections_ci95'])}, "
          f"agreement {a['corrections_mean']:.2f}{_ci_text(a['corrections_ci95'])} + {a['confirmations_mean']:.2f} confirmations",
          "  corr/dwg = edits that turn the list shown into the ground truth; confirm = decisions on highlighted features (agreement variant only);",
          "  'agreement' is scored after an ORACLE user confirmed the highlighted features: an upper bound, the corrections log has the real number."]
    o = s["overlap"]
    L += ["", f"Agreement of the passes: of {o['union_features']} features in the union, {o['seen_by_both']} were seen by both passes "
              f"(correct: {_f(o['both_correct_share'])}) and {o['seen_by_one']} by one pass only (correct: {_f(o['one_correct_share'])}); "
              f"{o['flagged_per_drawing']:.2f} highlighted per drawing, {o['drawings_with_nothing_flagged'] * 100:.0f}% of drawings with none."]
    L += ["", "Per feature (support = drawings whose truth has it; precision / recall):",
          f"  {'feature':16} {'n':>3}  {'geometry':>11} {'annotation':>11} {'union':>11} {'both':>11}"]
    for r in sorted(s["per_feature"], key=lambda r: -r["support"]):
        cells = [f"{_f(r[v]['precision'], 1)}/{_f(r[v]['recall'], 1)}" for v in ("geometry", "annotation", "union", "both")]
        L.append(f"  {r['feature']:16} {r['support']:>3}  " + " ".join(f"{c:>11}" for c in cells))
    if s["holes"]:
        h = s["holes"]
        L += ["", f"Hole diameters (dimension pass, {h['drawings']} drawings whose truth lists them): precision {_f(h['precision'])}, recall {_f(h['recall'])}, "
                  f"exact set on {h['exact_set_share'] * 100:.0f}%, {h['failed_calls']} failed call(s)."]
    if s["unrecognised"]:
        L += ["", "Names outside the vocabulary (vocabulary gaps or wording): " +
              ", ".join(f"{u['name']} x{u['count']}" + (f" (~ {u['suggestion']})" if u["suggestion"] else "") for u in s["unrecognised"][:10])]
    L += ["", "Read with care: a handful of drawings gives wide intervals; the ground truth is one annotator's; prompts must be frozen before the "
              "drawings used for this table are looked at."]
    return "\n".join(L)


def write_csv(path_prefix: str, s: dict) -> list:
    """<prefix>_variants.csv, <prefix>_features.csv and <prefix>_drawings.csv; returns the paths."""
    os.makedirs(os.path.dirname(os.path.abspath(path_prefix)), exist_ok=True)
    paths = []
    p = path_prefix + "_variants.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["variant", "precision", "recall", "f1", "f1_ci_low", "f1_ci_high", "exact_share", "corrections_mean", "corr_ci_low", "corr_ci_high",
                    "confirmations_mean", "shown_mean", "tp", "fp", "fn"])
        for r in s["variants"]:
            f1c, cc = r["f1_ci95"] or (None, None), r["corrections_ci95"] or (None, None)
            w.writerow([r["variant"], r["precision"], r["recall"], r["f1"], f1c[0], f1c[1], r["exact_share"], r["corrections_mean"], cc[0], cc[1],
                        r["confirmations_mean"], r["shown_mean"], r["tp"], r["fp"], r["fn"]])
    paths.append(p)
    p = path_prefix + "_features.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["feature", "support"] + [f"{v}_{m}" for v in ("geometry", "annotation", "union", "both") for m in ("precision", "recall", "f1")])
        for r in s["per_feature"]:
            w.writerow([r["feature"], r["support"]] + [r[v][m] for v in ("geometry", "annotation", "union", "both") for m in ("precision", "recall", "f1")])
    paths.append(p)
    p = path_prefix + "_drawings.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        cols = ["drawing", "truth", "geometry", "annotation", "highlighted", "missed_by_both"] + [f"corrections_{v}" for v in VARIANTS]
        w = csv.writer(f)
        w.writerow(cols)
        for r in s["per_drawing"]:
            w.writerow([";".join(r[c]) if isinstance(r[c], list) else r[c] for c in cols])
    paths.append(p)
    return paths
