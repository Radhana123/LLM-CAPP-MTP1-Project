"""Evidence, selective confirmation and the corrections log of the extraction front end (report 6.5, WP7, RQ6). Pure functions: no network,
no Streamlit; week6/app.py and scripts/extraction_eval.py call them.

    from capp import extraction_review as ER
    review = ER.build_review(extraction)            # extraction = week1/image_feature_extractor.extract_features_from_image(...)
    review["preaccepted"], review["flagged"]        # seen by both passes / seen by one pass only
    features = ER.initial_features(review)          # what the editable list starts with
    ER.pending(review, decisions, features)         # highlighted features nobody has accepted or rejected yet
    entry = ER.build_log_entry(extraction, final_features, decisions)   # one line of the corrections log
    ER.append_log("results/corrections_log.jsonl", entry)

Report 6.5: a feature reported by BOTH passes (geometry and annotation) is pre-accepted and marked high-confidence; a feature reported by
ONE pass only is highlighted, and the route cannot be generated until each highlighted feature has been accepted, edited or rejected.
Names outside the 36-feature vocabulary are listed as unrecognised (with the nearest vocabulary item) instead of being dropped in silence.

What a "correction" is (the corrections log, and the number M9 reports per drawing): a feature the user ADDS (the model did not give it),
a PRE-ACCEPTED feature the user REMOVES, or an unrecognised name the user MAPS to a vocabulary feature. Accepting or rejecting a
HIGHLIGHTED feature is a DECISION, not a correction: the interface asked for it. The two are counted apart, because RQ6 asks whether the
agreement rule lowers the corrections without making the confirmed list worse, and the price of it is the number of decisions.

Old extraction results (no per-pass data, no agreement counts) have no agreement information: nothing is highlighted and everything is
pre-accepted, as before Step 20. If only one of the two passes worked, no feature can be seen by both, so every feature is highlighted.
"""
import datetime
import difflib
import hashlib
import json
import os
import re
import statistics
from collections import Counter

from capp import legacy  # noqa: F401
import feature_vocab as fv

PASSES = ("geometry", "annotation")
BOTH, ONE, UNKNOWN = "both", "one", "unknown"
ACCEPT, REJECT = "accept", "reject"
CONF_RANK = {"high": 3, "medium": 2, "low": 1}
LOG_VERSION = 1
SUGGEST_CUTOFF = 0.72


class ExtractionReviewError(ValueError):
    pass


# ---------------------------------------------------------------- unrecognised names
def _key(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def nearest_vocabulary(name, cutoff: float = SUGGEST_CUTOFF):
    """The vocabulary feature a name most likely means ('Counter Bore' -> Counterbore, 'Threading' -> Thread), or None. A suggestion only:
    the user decides."""
    keys = {_key(f): f for f in fv.GEOMETRY_FEATURES}
    k = _key(name)
    if not k:
        return None
    if k in keys:
        return keys[k]
    close = difflib.get_close_matches(k, list(keys), n=1, cutoff=cutoff)
    return keys[close[0]] if close else None


# ---------------------------------------------------------------- the review of one extraction
def _cap(conf):
    """A single pass's own confidence, capped at 'medium': the other pass did not see the feature. None if it gave none."""
    if conf not in CONF_RANK:
        return None
    return "medium" if conf == "high" else conf


def build_review(extraction: dict) -> dict:
    """What the interface shows for one extraction. Rows are in the order of extraction["features"] (the union, as in v1):
    {"feature", "agreement": both|one|unknown, "seen_by": [pass names] ([] when only counts are known), "needs_confirmation",
     "confidence": 'high' (both passes) | the pass's own confidence capped at 'medium' | None, "evidence": {pass: text}, "evidence_text"}.
    Also "flagged" / "preaccepted" (feature names), "unrecognised" ([{"name", "seen_by", "suggestion"}]), "has_agreement", "passes_ok",
    "n_passes" and "notes" (what the user should be told)."""
    union = list(extraction.get("features") or [])
    reports = extraction.get("passes") if isinstance(extraction.get("passes"), dict) else {}
    counts = extraction.get("sample_agreement") if isinstance(extraction.get("sample_agreement"), dict) else None
    ok = {n: r for n, r in reports.items() if isinstance(r, dict) and r.get("success")}
    n_passes = len(PASSES)
    notes = []
    if reports:
        has_agreement, passes_ok = True, len(ok)
    elif counts is not None:
        has_agreement, passes_ok = True, extraction.get("passes_ok", n_passes)
    else:
        has_agreement, passes_ok = False, None
    if has_agreement and passes_ok is not None and passes_ok < n_passes:
        notes.append(f"Only {passes_ok} of the {n_passes} passes worked, so no feature was seen by both: every feature needs your confirmation.")
    rows = []
    for f in union:
        seen = [n for n in PASSES if n in ok and f in (ok[n].get("features") or [])] if reports else []
        if not has_agreement:
            agreement = UNKNOWN
        elif reports:
            agreement = BOTH if len(seen) == n_passes else ONE
        else:
            agreement = BOTH if counts.get(f, 0) >= n_passes and passes_ok >= n_passes else ONE
        if agreement == BOTH:
            conf = "high"
        elif agreement == ONE and seen:
            conf = _cap(ok[seen[0]].get("confidence"))
        else:
            conf = None
        ev = {n: ok[n]["evidence"][f] for n in seen if isinstance(ok[n].get("evidence"), dict) and ok[n]["evidence"].get(f)}
        rows.append({"feature": f, "agreement": agreement, "seen_by": seen, "needs_confirmation": agreement == ONE, "confidence": conf,
                     "evidence": ev, "evidence_text": "; ".join(f"{n}: {t}" for n, t in ev.items())})
    unrec = []
    names = list(extraction.get("rejected_features") or [])
    for r in ok.values():
        names += [x for x in (r.get("rejected_features") or []) if x not in names]
    for name in dict.fromkeys(names):
        unrec.append({"name": name, "seen_by": [n for n in PASSES if n in ok and name in (ok[n].get("rejected_features") or [])],
                      "suggestion": nearest_vocabulary(name)})
    return {"rows": rows, "flagged": [r["feature"] for r in rows if r["needs_confirmation"]],
            "preaccepted": [r["feature"] for r in rows if not r["needs_confirmation"]], "unrecognised": unrec,
            "has_agreement": has_agreement, "passes_ok": passes_ok, "n_passes": n_passes, "notes": notes}


def pass_notes(extraction: dict) -> dict:
    """{pass: the one-sentence note it gave}, for the passes that gave one (shown when there is no per-feature evidence)."""
    reports = extraction.get("passes") if isinstance(extraction.get("passes"), dict) else {}
    return {n: r["notes"] for n, r in reports.items() if isinstance(r, dict) and r.get("success") and r.get("notes")}


# ---------------------------------------------------------------- confirmation state
def initial_features(review: dict) -> list:
    """The editable list at the start: the pre-accepted features (everything when there is no agreement information)."""
    return list(review["preaccepted"])


def pending(review: dict, decisions: dict, features) -> list:
    """Highlighted features still without a decision. One counts as reviewed when it was accepted or rejected, or when the user put it
    in the list by editing it (that is an acceptance)."""
    decisions, have = decisions or {}, set(features or [])
    return [f for f in review["flagged"] if decisions.get(f) not in (ACCEPT, REJECT) and f not in have]


def can_generate(review: dict, decisions: dict, features) -> bool:
    """Generate Route stays disabled until nothing is pending (and there is at least one feature)."""
    return bool(features) and not pending(review, decisions, features)


def decide(features, decisions, feature: str, decision: str):
    """(features, decisions) after a decision on one feature; the inputs are not modified. Accept puts the feature in the list, reject
    takes it out; a later decision replaces an earlier one."""
    if decision not in (ACCEPT, REJECT):
        raise ExtractionReviewError(f"decision must be {ACCEPT!r} or {REJECT!r}, not {decision!r}")
    feats, dec = list(features or []), dict(decisions or {})
    dec[feature] = decision
    if decision == ACCEPT and feature not in feats:
        feats.append(feature)
    if decision == REJECT:
        feats = [f for f in feats if f != feature]
    return feats, dec


def map_unrecognised(features, mappings, name: str, feature: str):
    """(features, mappings) after the user maps an unrecognised name to a vocabulary feature."""
    if not fv.is_valid_feature(feature):
        raise ExtractionReviewError(f"{feature!r} is not a feature of the vocabulary")
    feats, mp = list(features or []), dict(mappings or {})
    mp[name] = feature
    if feature not in feats:
        feats.append(feature)
    return feats, mp


def status(review: dict, decisions: dict, features) -> dict:
    """{feature: label} for the table: pre-accepted | needs confirmation | accepted | rejected."""
    decisions, have = decisions or {}, set(features or [])
    out = {}
    for r in review["rows"]:
        f = r["feature"]
        if not r["needs_confirmation"]:
            out[f] = "pre-accepted" if f in have else "removed"
        elif decisions.get(f) == REJECT or (f not in have and decisions.get(f) == ACCEPT):
            out[f] = "rejected"
        elif f in have:
            out[f] = "accepted"
        else:
            out[f] = "needs confirmation"
    return out


# ---------------------------------------------------------------- the corrections log
def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_log_entry(extraction: dict, features, decisions: dict = None, mappings: dict = None, drawing_id: str = "",
                    image_bytes: bytes = None, now: str = None, review: dict = None) -> dict:
    """One line of the corrections log: the model's original output, the user's final list and the edits between them (report 6.5).
    The drawing itself is not stored: only a short hash of it, so the same drawing can be recognised."""
    review = review or build_review(extraction)
    decisions, mappings = decisions or {}, mappings or {}
    final = list(dict.fromkeys(features or []))
    union = [r["feature"] for r in review["rows"]]
    pre, flagged = set(review["preaccepted"]), set(review["flagged"])
    events = []
    for f in union:
        if f in pre and f not in final:
            events.append({"type": "removed", "feature": f, "was": "pre-accepted", "correction": True})
        if f in flagged:
            events.append({"type": "accepted" if f in final else "rejected", "feature": f, "was": "highlighted", "correction": False})
    mapped_to = set()
    for name, f in mappings.items():
        if f in final and f not in union:
            events.append({"type": "mapped", "from": name, "feature": f, "correction": True})
            mapped_to.add(f)
    for f in final:
        if f not in union and f not in mapped_to:
            events.append({"type": "added", "feature": f, "correction": True})
    sha = hashlib.sha256(image_bytes).hexdigest()[:12] if image_bytes else ""
    return {"version": LOG_VERSION, "timestamp": now or _now(), "drawing_id": drawing_id or sha, "image_sha256": sha,
            "model_output": {"features": union, "agreement": {r["feature"]: r["agreement"] for r in review["rows"]},
                             "seen_by": {r["feature"]: r["seen_by"] for r in review["rows"]},
                             "highlighted": list(review["flagged"]), "rejected_features": list(extraction.get("rejected_features") or []),
                             "unrecognised": [u["name"] for u in review["unrecognised"]],
                             "passes_ok": review["passes_ok"], "has_agreement": review["has_agreement"]},
            "final_features": final, "events": events,
            "n_corrections": sum(e["correction"] for e in events),
            "n_decisions": sum(e["type"] in ("accepted", "rejected") for e in events), "n_highlighted": len(flagged)}


def append_log(path: str, entry: dict) -> None:
    """Add one entry to the JSONL log (the folder is created). Raises OSError; a caller that must not stop on it catches it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_log(path: str):
    """(entries, n_bad_lines). Blank lines are skipped; a line that is not a log entry is counted, not fatal."""
    entries, bad = [], 0
    if not os.path.exists(path):
        return entries, bad
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                if not isinstance(e, dict) or "model_output" not in e or "final_features" not in e:
                    raise ValueError
                entries.append(e)
            except ValueError:
                bad += 1
    return entries, bad


def prf(tp: int, fp: int, fn: int) -> dict:
    """Precision, recall, F1 (None where undefined: no prediction, no truth)."""
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r else (0.0 if p is not None and r is not None else None)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def summarize_log(entries: list) -> dict:
    """What the log says (report 6.5): corrections per drawing, decisions per drawing, how far the user's final list is from the model's
    output (the final list is the user's, not an independent truth), which features are most often missed or invented, and which
    unrecognised names occur and what they were mapped to."""
    n = len(entries)
    if not n:
        return {"n_drawings": 0}
    corr = [e["n_corrections"] for e in entries]
    tp = fp = fn = 0
    missed, invented, mapped, unrec = Counter(), Counter(), Counter(), Counter()
    for e in entries:
        model, final = set(e["model_output"]["features"]), set(e["final_features"])
        tp += len(model & final)
        fp += len(model - final)
        fn += len(final - model)
        invented.update(model - final)
        missed.update(final - model)
        for ev in e.get("events", []):
            if ev["type"] == "mapped":
                mapped[(ev["from"], ev["feature"])] += 1
        unrec.update(e["model_output"].get("rejected_features") or [])
    return {"n_drawings": n,
            "corrections": {"mean": statistics.mean(corr), "median": statistics.median(corr), "max": max(corr),
                            "share_zero": sum(c == 0 for c in corr) / n},
            "decisions_mean": statistics.mean(e["n_decisions"] for e in entries),
            "highlighted_mean": statistics.mean(e["n_highlighted"] for e in entries),
            "model_vs_final": prf(tp, fp, fn),
            "exact_share": sum(set(e["model_output"]["features"]) == set(e["final_features"]) for e in entries) / n,
            "most_missed": missed.most_common(), "most_invented": invented.most_common(),
            "unrecognised": unrec.most_common(), "mapped": [(a, b, c) for (a, b), c in mapped.most_common()],
            "without_agreement_info": sum(not e["model_output"].get("has_agreement") for e in entries)}


def _fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def format_log_report(s: dict, path: str = "", bad: int = 0) -> str:
    if not s.get("n_drawings"):
        return "The corrections log is empty." + (f" ({path})" if path else "")
    c, m = s["corrections"], s["model_vs_final"]
    L = [f"Corrections log{': ' + path if path else ''}: {s['n_drawings']} drawing(s)" + (f", {bad} unreadable line(s) skipped" if bad else ""),
         f"  corrections per drawing: mean {c['mean']:.2f}, median {c['median']:g}, max {c['max']}; drawings needing none: {c['share_zero'] * 100:.0f}%",
         f"  decisions on highlighted features per drawing: {s['decisions_mean']:.2f} (of {s['highlighted_mean']:.2f} highlighted)",
         f"  model output vs the user's final list: precision {_fmt(m['precision'])}, recall {_fmt(m['recall'])}, F1 {_fmt(m['f1'])}; "
         f"identical on {s['exact_share'] * 100:.0f}% of drawings",
         "  most often MISSED (the user added): " + (", ".join(f"{f} x{k}" for f, k in s["most_missed"][:8]) or "none"),
         "  most often INVENTED (the user dropped): " + (", ".join(f"{f} x{k}" for f, k in s["most_invented"][:8]) or "none")]
    if s["unrecognised"]:
        L.append("  unrecognised names the model gave: " + ", ".join(f"{f} x{k}" for f, k in s["unrecognised"][:8]))
    if s["mapped"]:
        L.append("  mapped by the user: " + ", ".join(f"{a} -> {b} x{k}" for a, b, k in s["mapped"][:8]))
    if s["without_agreement_info"]:
        L.append(f"  {s['without_agreement_info']} entr(ies) came from an extraction without per-pass data: nothing was highlighted there.")
    L.append("The final list is what the user confirmed, not an independent ground truth: it measures how much the user changed, "
             "and a user can miss an error. Accuracy against ground truth is scripts/extraction_eval.py (M9).")
    return "\n".join(L)
