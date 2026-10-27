"""Failure case studies (report 7.5, WP8): a fixed selection rule and, for every selected case, the facts from the records. Offline: it reads
records.jsonl (and run.json) and makes no API call. scripts/case_studies.py is the command line.

What the report asks for. "For each main failure type of M3, three to five cases are selected ... Each case is documented as the part and
its features, the LLM route, the violation report, the repair applied and the final route, with a short commentary on why the model failed
and whether the repair preserved the decisions that were correct. Cases in which the final route is valid but differs from the expert
reference are included as well ... Cases are selected by a fixed rule (the first failures of each type in the order of the part
identifiers), supplemented by a few chosen for being instructive, which are labelled as such."

The rule implemented here, per run and per type:
  * a failure of type T is a raw LLM route (one sample of one repeat) whose strict check lists a violation of kind T (order, missing,
    invented, duplicate, machine_grouping), or whose physical-feasibility check lists an 'infeasible' finding (infeasible; needs the records
    of a run with the analysis on). A route can be a failure of several types: it is a candidate for each.
  * one case per part: the first such route of the part in the order (repeat, sample id). Parts are taken in the natural order of their
    identifiers (DEV_002 before DEV_010). The first `per_type` parts are the cases. Nothing else decides: not the metric values, not
    whether the repair worked.
  * 'differs': the part's final route is strict-valid but is not an exact match of any reference route (expert routes when the part has
    them, otherwise the Route Builder variants, which is only a proxy and is labelled as such). Same one-per-part, first-in-order rule.
  * instructive cases are named by the author on the command line, each with the reason, and are labelled "instructive" everywhere.
The facts are mechanical (what the verifier said, what the repair removed, inserted and kept, what the final route is). The commentary on
WHY the model failed and whether the repair kept what was right is the author's: the generator leaves a marked gap, or fills it from a
commentary file (so that regenerating the report keeps what was written).
"""
import csv
import json
import os
import re

from capp import dataset, metrics as M, partdiff as PD, summary as S, verify as V

TYPES = ("order", "missing", "invented", "duplicate", "machine_grouping", "infeasible")     # the six main types of M3
OPTIONAL_TYPES = ("risk",)                                                                  # soft: not part of the report's six
RECORD_TYPES = ("differs",)                                                                 # final valid, not like the reference
ALL_TYPES = TYPES + OPTIONAL_TYPES + RECORD_TYPES
DEFAULT_TYPES = TYPES + RECORD_TYPES
DEFAULT_PER_TYPE = 3
MAX_PER_TYPE = 10
GAP = "_TO WRITE: why the model failed, and whether the repair kept the decisions that were correct._"

TYPE_TITLES = {
    "order": "Order violation: an operation is placed before one that must precede it",
    "missing": "Missing operation: a feature is not covered",
    "invented": "Invented operation: not in the operation vocabulary",
    "duplicate": "Duplicate operation",
    "machine_grouping": "Wrong machine grouping: more machine changeovers than needed (soft)",
    "infeasible": "Physical infeasibility: the route cannot be carried out as planned",
    "risk": "Physical risk: carried out only with a support, a smaller feed or a finishing step (soft)",
    "differs": "Valid final route that differs from the reference",
    "instructive": "Instructive cases (chosen by the author)",
}


class CaseError(ValueError):
    """A problem with what was asked for (the runs, the types, an instructive entry, the commentary file)."""


def natural_key(part_id: str):
    """DEV_002 sorts before DEV_010 whatever the zero padding."""
    return tuple((0, int(t)) if t.isdigit() else (1, t) for t in re.findall(r"\d+|\D+", str(part_id)))


# ---------------------------------------------------------------- reading the runs
def load_run(folder: str) -> dict:
    from capp import harness
    if not os.path.exists(os.path.join(folder, "records.jsonl")):
        raise CaseError(f"no records.jsonl in {folder}")
    try:
        records = harness.load_records(folder)
    except (OSError, ValueError) as exc:
        raise CaseError(f"{folder}: records.jsonl cannot be read ({exc})") from None
    if not records:
        raise CaseError(f"{folder}: records.jsonl is empty")
    for rec in records:
        missing = [k for k in ("part_id", "repeat", "part", "candidates", "final", "info", "references") if k not in rec]
        if missing:
            raise CaseError(f"{folder}: a record lacks {missing}: not a run of this harness")
    info = {}
    try:
        with open(os.path.join(folder, "run.json"), encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError):
        pass
    label = os.path.basename(folder.rstrip("/\\")) or folder
    return {"label": label, "folder": folder, "records": records, "run_info": info}


# ---------------------------------------------------------------- what is wrong with a raw route
def parsed(rec):
    return [c for c in rec["candidates"] if c["llm_ok"] and c.get("first_pass") is not None and c.get("llm_steps")]


def feasibility_of(c):
    """The physical-feasibility findings of the raw route, or None when the run did not analyse it."""
    f = c.get("raw_feasibility")
    if not f or not isinstance(f, dict) or "violations" not in f:
        return None
    return f


def failure_types(c) -> list:
    """The types of a raw route, in the order of verify.KINDS: the strict check's kinds, plus 'infeasible' from the feasibility check."""
    kinds = set(c["first_pass"]["kinds"])
    f = feasibility_of(c)
    if f is not None and any(v.get("kind") == "infeasible" for v in f["violations"]):
        kinds.add("infeasible")
    return sorted(kinds, key=V.KINDS.index)


def _key(rec, c):
    return (natural_key(rec["part_id"]), rec["repeat"], c["sample_id"])


def failures_of_type(records, t):
    """[(record, candidate)], one per part (the first by repeat and sample id), in the natural order of the part ids."""
    best = {}
    for rec in records:
        for c in parsed(rec):
            if t in failure_types(c):
                k = _key(rec, c)
                if rec["part_id"] not in best or k < best[rec["part_id"]][0]:
                    best[rec["part_id"]] = (k, rec, c)
    return [(rec, c) for _, rec, c in sorted(best.values(), key=lambda x: x[0])]


def differs_records(records):
    """[record], one per part (the first repeat), whose final route is strict-valid but not an exact match of any reference route."""
    best = {}
    for rec in records:
        refs, _ = S._refs_for(rec)
        fin = rec["final"]
        if not refs or not fin["check"]["valid"] or not fin["steps"]:
            continue
        if M.best_similarity(fin["steps"], refs)["exact"]:
            continue
        k = (natural_key(rec["part_id"]), rec["repeat"])
        if rec["part_id"] not in best or k < best[rec["part_id"]][0]:
            best[rec["part_id"]] = (k, rec)
    return [rec for _, rec in sorted(best.values(), key=lambda x: x[0])]


def type_table(records, types) -> list:
    """The aggregate beside the cases: for every type, how many raw routes show it, in how many parts, and how many cases it could give."""
    routes = [(rec, c) for rec in records for c in parsed(rec)]
    n_routes = len(routes)
    n_calls = sum(1 for rec in records for c in rec["candidates"] if c["sample_id"] >= 0)
    n_feas = sum(1 for _, c in routes if feasibility_of(c) is not None)
    rows = []
    for t in types:
        if t == "differs":
            eligible = differs_records(records)
            rows.append({"type": t, "routes": None, "share": None, "parts": len(eligible), "eligible": len(eligible), "denominator": None,
                         "note": f"{len(eligible)} part(s) with a valid final route that is not an exact match of the reference"})
            continue
        hit = [(rec, c) for rec, c in routes if t in failure_types(c)]
        denom = n_feas if t == "infeasible" else n_routes
        note = ""
        if t == "infeasible" and n_feas == 0:
            note = "no feasibility analysis in these records (run without --no-analysis)"
        rows.append({"type": t, "routes": len(hit), "share": (len(hit) / denom) if denom else None,
                     "parts": len({rec["part_id"] for rec, _ in hit}), "eligible": len({rec["part_id"] for rec, _ in hit}),
                     "denominator": denom, "note": note})
    return rows


# ---------------------------------------------------------------- one case
def _ops(route):
    return V.clean_steps(route or [])


def route_diff(llm, fixed) -> dict:
    """What a repair or replacement did to the LLM's route, from the two routes alone."""
    a, b = _ops(llm), _ops(fixed)
    d = PD.route_delta(a, b)
    tau = M.kendall_tau_distance(a, b)
    return {"removed": d["only_base"], "inserted": d["only_other"], "kept_in_order": M.lcs_length(a, b), "n_llm_ops": len(a), "n_fixed_ops": len(b),
            "survived_share": M.survived_share(a, b), "edit_distance": M.edit_distance(a, b),
            "order_changed": bool(tau), "kendall_tau": tau, "identical": a == b}


def repair_outcome(c) -> str:
    rep = c.get("repair") or {}
    if c["replaced"]:
        tried = rep.get("tried") or []
        return "replaced by the Route Builder's route (R0)" + (f" after {', '.join(t.upper() for t in tried)} did not give a valid route" if tried else "")
    if c.get("repaired"):
        return {"r1": "repaired by R1 (minimal edit by rules)", "r2": "repaired by R2 (the model was asked again with the violation report)"}.get(
            rep.get("stage"), "repaired")
    if not c["steps"]:
        return "no route: the Route Builder found no replacement either"
    if c["provenance"] == "raw_llm" and rep:
        return "kept as sampled: only a risk was to be fixed and R1 could not fix it"
    if c["first_pass"]["valid"]:
        return "valid as sampled: no repair needed"
    return "not repaired"


def _violation(v):
    return {"kind": v["kind"], "message": v["message"], "ops": list(v.get("ops") or []), "feature": v.get("feature", ""),
            "rule_id": v.get("rule_id", ""), "positions": list(v.get("positions") or []), "hard": v["kind"] in V.HARD_KINDS}


def _final_block(rec) -> dict:
    fin = rec["final"]
    refs, kind = S._refs_for(rec)
    steps = _ops(fin["steps"])
    sim = M.best_similarity(steps, refs) if refs and steps else None
    nearest = None
    if refs and steps:
        nearest = max(refs, key=lambda r: M.lcs_ratio(steps, r))
        nearest = _ops(nearest)
    gap = S._gap_to_rb_best(fin["scores_eval"], rec["best_scores"]) if fin["check"]["valid"] and rec.get("best_scores") else None
    feas = fin.get("feasibility")
    return {"name": fin["name"], "provenance": fin["provenance"], "route": steps, "valid": bool(fin["check"]["valid"]),
            "kinds": list(fin["check"]["kinds"]), "scores": fin["scores_eval"],
            "efficiency_gap_to_rb_best": None if not gap else gap["efficiency_gap"],
            "reference_kind": kind if refs else None, "n_references": len(refs or []),
            "similarity": sim, "nearest_reference": nearest,
            "feasibility": None if not isinstance(feas, dict) else [_violation(v) for v in feas.get("violations", [])]}


def _part_block(rec) -> dict:
    p = rec["part"]
    return {"part_id": p["part_id"], "archetype": p.get("archetype", ""), "material": p["material"], "features": list(p["features"]),
            "tolerance": p.get("tolerance", ""), "batch_size": p.get("batch_size", ""), "dims": p.get("dims"),
            "machine_preference": p.get("machine_preference", ""), "machine_profile": p.get("machine_profile", ""),
            "in_distribution": p.get("in_distribution"), "required_ra_um": p.get("required_ra_um"),
            "has_expert_reference": bool(p.get("reference_routes"))}


def case_id(run_label, type_, rec, sample) -> str:
    return f"{run_label}:{type_}:{rec['part_id']}:r{rec['repeat']}:{'final' if sample is None else 's' + str(sample)}"


def candidate_case(run, rec, c, type_, source="rule", rank=None, note="") -> dict:
    """The facts of one raw route and what happened to it. type_ is the failure type, or 'instructive'."""
    llm = _ops(c["llm_steps"])
    fixed = _ops(c["steps"])
    rep = c.get("repair") or {}
    f = feasibility_of(c)
    final = _final_block(rec)
    selected = bool(c["in_pool"] and rec["final"]["name"] == c["name"])
    out = {
        "id": case_id(run["label"], type_, rec, c["sample_id"]), "type": type_, "source": source, "rank": rank, "note": note,
        "run": run["label"], "part": _part_block(rec), "repeat": rec["repeat"], "sample_id": c["sample_id"], "candidate": c["name"],
        "llm_route": llm, "types": failure_types(c),
        "violations": [_violation(v) for v in c["first_pass"]["violations"]], "strict_valid_as_sampled": bool(c["first_pass"]["valid"]),
        "feasibility": None if f is None else [_violation(v) for v in f["violations"]],
        "repair": {"outcome": repair_outcome(c), "stage": rep.get("stage"), "tried": list(rep.get("tried") or []),
                   "fallback": rep.get("fallback"), "r1": rep.get("r1"),
                   "r2": None if not rep.get("r2") else {k: rep["r2"].get(k) for k in ("ok", "reason", "kinds", "steps")}},
        "candidate_route": fixed, "diff": route_diff(llm, fixed) if fixed else None,
        "selected_as_final": selected, "final": final,
        "cascade": c.get("cascade") and {k: c["cascade"].get(k) for k in ("tier", "escalated", "kept")},
    }
    out["facts"] = _facts(out)
    return out


def _join(ops):
    return ", ".join(ops) if ops else "none"


def _repair_facts(rp, d, llm, candidate_route) -> list:
    """What R1 and R2 did, from the repair record and the two routes."""
    out = []
    r1 = rp["r1"]
    if r1:
        if r1.get("ok") is False:
            out.append(f"R1 could not produce a valid route{(': ' + r1['reason']) if r1.get('reason') else ''}.")
        else:
            bits = []
            if r1.get("removed_unknown"):
                bits.append(f"removed invented operation(s) {_join(r1['removed_unknown'])}")
            if r1.get("removed_duplicates"):
                bits.append(f"removed duplicate(s) {_join(r1['removed_duplicates'])}")
            if r1.get("inserted"):
                fz = r1.get("fz_inserted") or []
                bits.append(f"inserted {_join(r1['inserted'])}" + (f" (for feasibility: {_join(fz)})" if fz else ""))
            if r1.get("fz_edges"):
                bits.append("added the ordering " + ", ".join(f"{a} before {b}" for a, b in r1["fz_edges"]))
            if r1.get("discordant_pairs"):
                bits.append(f"reordered (it resolved {r1['discordant_pairs']} operation pair(s) that were in the wrong order)")
            out.append("R1 " + (", ".join(bits) if bits else "changed nothing in the operations") + ".")
            if r1.get("unrepairable"):
                out.append(f"R1 left hard feasibility finding(s) it cannot repair: {_join(r1['unrepairable'])}.")
            if r1.get("flagged"):
                out.append(f"Flagged for the engineer, not changed: {_join(r1['flagged'])}.")
    r2 = rp["r2"]
    if r2:
        out.append(f"R2 re-prompt: {'the answer was accepted' if r2.get('ok') else 'the answer was not accepted'}"
                   f"{(' (' + r2['reason'] + ')') if r2.get('reason') else ''}.")
    if d is not None and (rp["stage"] or candidate_route != llm):
        out.append(f"{d['kept_in_order']} of the LLM's {d['n_llm_ops']} operations are in the new route in the same order (share {d['survived_share']:.2f}); "
                   f"edit distance {d['edit_distance']}; removed: {_join(d['removed'])}; inserted: {_join(d['inserted'])}"
                   f"{'; relative order changed' if d['order_changed'] else ''}.")
    return out


def _facts(case) -> list:
    """Mechanical sentences: what the records say, nothing about why."""
    out = []
    llm, d, rp, fin = case["llm_route"], case["diff"], case["repair"], case["final"]
    kinds = [v["kind"] for v in case["violations"]]
    if kinds:
        out.append(f"The strict check of the route as sampled ({len(llm)} operations) lists {len(kinds)} violation(s): {_join(sorted(set(kinds), key=V.KINDS.index))}.")
    else:
        out.append(f"The strict check of the route as sampled ({len(llm)} operations) lists no violation.")
    if len(case["types"]) > 1:
        out.append(f"This route fails in several ways ({_join(case['types'])}); it is a candidate under each of these types.")
    if case["feasibility"]:
        hard = [v for v in case["feasibility"] if v["hard"]]
        out.append(f"The physical-feasibility check lists {len(hard)} hard and {len(case['feasibility']) - len(hard)} soft finding(s) "
                   f"({_join(sorted({v['rule_id'] for v in case['feasibility']}))}).")
    elif case["feasibility"] is None:
        out.append("The physical-feasibility check was not run on this route (the run was made without the analysis).")
    out.append(f"Outcome for this route: {rp['outcome']}.")
    out += _repair_facts(rp, d, llm, case["candidate_route"])
    out.append(f"This route {'is' if case['selected_as_final'] else 'is not'} the one selected as the final route of the part.")
    eff = ""
    if fin["valid"] and fin["scores"]:
        g = fin["efficiency_gap_to_rb_best"]
        eff = f"; its Efficiency is {fin['scores']['efficiency_score']:.2f}" + ("" if g is None else (
            f", {g:.2f} points below the best Route Builder variant" if g > 0 else ", not below the best Route Builder variant"))
    out.append(f"The final route of the part comes from {fin['provenance']} and is {'strict-valid' if fin['valid'] else 'NOT strict-valid'}{eff}.")
    if fin["similarity"]:
        ref = {"expert": "an expert reference route", "proxy_rb": "the Route Builder variants (a proxy, not an expert)"}[fin["reference_kind"]]
        out.append(f"Against {ref}: exact match {'yes' if fin['similarity']['exact'] else 'no'}, LCS ratio {fin['similarity']['lcs_ratio']:.2f}.")
    return out


def differs_case(run, rec, rank=None) -> dict:
    """A part whose final route is valid but not like the reference. Carries the facts of the candidate that became the final route, if any."""
    final = _final_block(rec)
    steps = final["route"]
    ref = final["nearest_reference"] or []
    d = PD.route_delta(ref, steps)
    src = next((c for c in rec["candidates"] if c["in_pool"] and c["name"] == rec["final"]["name"] and c["sample_id"] >= 0 and c.get("llm_steps")), None)
    base = candidate_case(run, rec, src, "differs", "rule", rank) if src is not None and src.get("first_pass") is not None else None
    case = {"id": case_id(run["label"], "differs", rec, None), "type": "differs", "source": "rule", "rank": rank, "note": "", "run": run["label"],
            "part": _part_block(rec), "repeat": rec["repeat"], "sample_id": None, "candidate": rec["final"]["name"], "final": final,
            "reference_delta": {"only_in_reference": d["only_base"], "only_in_final": d["only_other"],
                                "order_changed": bool(M.kendall_tau_distance(ref, steps)), "kendall_tau": M.kendall_tau_distance(ref, steps)},
            "from_candidate": base}
    kind = {"expert": "the references are expert routes", "proxy_rb": "the references are Route Builder variants, a proxy, not an expert",
            None: "there is no reference"}[final["reference_kind"]]
    facts = [f"The final route comes from {final['provenance']} and is strict-valid; it is not an exact match of any of the {final['n_references']} reference route(s) ({kind})."]
    if final["similarity"]:
        facts.append(f"Closest reference: LCS ratio {final['similarity']['lcs_ratio']:.2f}; only in the reference: {_join(d['only_base'])}; "
                     f"only in the final route: {_join(d['only_other'])}{'; relative order differs' if case['reference_delta']['order_changed'] else ''}.")
    if final["efficiency_gap_to_rb_best"] is not None:
        facts.append(f"Efficiency {final['scores']['efficiency_score']:.2f}: "
                     + ("not below the best Route Builder variant." if final["efficiency_gap_to_rb_best"] <= 0 else
                        f"{final['efficiency_gap_to_rb_best']:.2f} points below the best Route Builder variant."))
    if base is not None:
        facts.append(f"It was built from sample {base['sample_id']}, whose raw route " + (
            f"has {len(base['violations'])} strict violation(s) ({_join(sorted({v['kind'] for v in base['violations']}, key=V.KINDS.index))})"
            if base["violations"] else "is strict-valid as sampled") + (
            f", with {len(base['feasibility'])} physical-feasibility finding(s) ({_join(sorted({v['rule_id'] for v in base['feasibility']}))})"
            if base["feasibility"] else "") + ".")
        facts += _repair_facts(base["repair"], base["diff"], base["llm_route"], base["candidate_route"])
    case["facts"] = facts
    return case


# ---------------------------------------------------------------- selection
def parse_instructive(text: str) -> dict:
    """'DEV_016:1:0=why' -> part DEV_016, repeat 1, sample 0, note 'why'. Repeat and sample are optional. The reason is required."""
    if "=" not in text:
        raise CaseError(f"--instructive {text!r}: write PART[:REPEAT[:SAMPLE]]=why it is instructive (the reason is required: the case is labelled with it)")
    spec, note = text.split("=", 1)
    note = note.strip()
    parts = spec.strip().split(":")
    if not note or not parts[0] or len(parts) > 3:
        raise CaseError(f"--instructive {text!r}: write PART[:REPEAT[:SAMPLE]]=why it is instructive")
    try:
        repeat = int(parts[1]) if len(parts) > 1 and parts[1] != "" else None
        sample = int(parts[2]) if len(parts) > 2 and parts[2] != "" else None
    except ValueError:
        raise CaseError(f"--instructive {text!r}: repeat and sample must be whole numbers") from None
    return {"part_id": parts[0].strip(), "repeat": repeat, "sample_id": sample, "note": note}


def instructive_case(run, spec, rank):
    """The case for one instructive entry, or None if this run has no such route (no such part or repeat, no parsed LLM route, no such sample:
    a Route Builder run has none). Without a sample: the first route of the part with a violation (else its first route)."""
    recs = [r for r in run["records"] if r["part_id"] == spec["part_id"] and (spec["repeat"] is None or r["repeat"] == spec["repeat"])]
    for rec in sorted(recs, key=lambda r: r["repeat"]):
        cands = parsed(rec)
        if spec["sample_id"] is not None:
            cands = [c for c in cands if c["sample_id"] == spec["sample_id"]]
        if not cands:
            continue
        c = next((c for c in cands if not c["first_pass"]["valid"]), cands[0])
        return candidate_case(run, rec, c, "instructive", "instructive", rank, spec["note"])
    return None


def validate_types(types):
    types = list(types)
    bad = [t for t in types if t not in ALL_TYPES]
    if bad:
        raise CaseError(f"unknown type(s) {bad}: use {', '.join(ALL_TYPES)}")
    if len(set(types)) != len(types):
        raise CaseError("a type is listed twice")
    if not types:
        raise CaseError("no type to select")
    return types


def analyse(runs: list, types=DEFAULT_TYPES, per_type: int = DEFAULT_PER_TYPE, instructive=(), commentary=None) -> dict:
    """The cases of every run. runs: load_run() results. instructive: parse_instructive() results. commentary: {case id: text}."""
    types = validate_types(types)
    if not isinstance(per_type, int) or isinstance(per_type, bool) or not 1 <= per_type <= MAX_PER_TYPE:
        raise CaseError(f"--per-type must be a whole number from 1 to {MAX_PER_TYPE} (the report asks for three to five)")
    if not runs:
        raise CaseError("no run given")
    commentary = commentary or {}
    out_runs, cases, found = [], [], set()
    for run in runs:
        recs = run["records"]
        table = type_table(recs, types)
        mine = []
        for t in types:
            pool = [differs_case(run, rec) for rec in differs_records(recs)[:per_type]] if t == "differs" else \
                   [candidate_case(run, rec, c, t) for rec, c in failures_of_type(recs, t)[:per_type]]
            for i, case in enumerate(pool, 1):
                case["rank"] = i
            mine.extend(pool)
        for i, spec in enumerate(instructive, 1):
            case = instructive_case(run, spec, i)
            if case is not None:
                mine.append(case)
                found.add(i)
        for case in mine:
            case["commentary"] = (commentary.get(case["id"]) or "").strip()
        cases.extend(mine)
        info = run["run_info"]
        out_runs.append({"label": run["label"], "folder": run["folder"], "system": recs[0]["system"], "n_records": len(recs),
                         "n_parts": len({r["part_id"] for r in recs}), "repeats": sorted({r["repeat"] for r in recs}),
                         "n_parsed_routes": sum(len(parsed(r)) for r in recs),
                         "n_unparsed_calls": sum(1 for r in recs for c in r["candidates"] if c["sample_id"] >= 0 and not (c["llm_ok"] and c.get("llm_steps"))),
                         "dataset": info.get("dataset"), "sealed": bool(info.get("dataset") and dataset.is_sealed(info["dataset"])),
                         "tag": info.get("tag"), "model": info.get("model"), "git_commit": info.get("git_commit"), "table": table})
    for i, spec in enumerate(instructive, 1):
        if i not in found:
            where = f" repeat {spec['repeat']}" if spec["repeat"] is not None else ""
            what = f" sample {spec['sample_id']}" if spec["sample_id"] is not None else ""
            raise CaseError(f"--instructive {spec['part_id']}{where}{what}: no run has a parsed LLM route for it "
                            "(check the part id, the repeat and the sample id; a Route Builder run has no LLM routes)")
    ids = {c["id"] for c in cases}
    return {"types": types, "per_type": per_type, "runs": out_runs, "cases": cases,
            "n_without_commentary": sum(1 for c in cases if not c["commentary"]),
            "orphan_commentary": sorted(k for k, v in commentary.items() if k not in ids and v.strip())}


# ---------------------------------------------------------------- commentary file
def read_commentary(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise CaseError(f"{path} is not a valid commentary file ({exc})") from None
    if not isinstance(data, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in data.items()):
        raise CaseError(f"{path}: expected one JSON object {{case id: text}}")
    return data


def sync_commentary(path: str, cases: list) -> int:
    """Adds an empty entry for every case that has none; never removes or changes what is written. Returns the number added."""
    data = read_commentary(path)
    added = 0
    for c in cases:
        if c["id"] not in data:
            data[c["id"]] = ""
            added += 1
    if added or not os.path.exists(path):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return added


# ---------------------------------------------------------------- rendering
def _route(ops, strike=(), bold=()):
    if not ops:
        return "_(empty)_"
    left_s, left_b = list(strike), list(bold)
    out = []
    for i, op in enumerate(ops, 1):
        if op in left_s:
            left_s.remove(op)
            out.append(f"{i}. ~~{op}~~")
        elif op in left_b:
            left_b.remove(op)
            out.append(f"{i}. **{op}**")
        else:
            out.append(f"{i}. {op}")
    return " → ".join(out)


def _part_line(p) -> str:
    bits = [f"{p['material']}", f"features: {', '.join(p['features'])}", f"tolerance {p['tolerance']}", f"batch {p['batch_size']}"]
    if p["machine_preference"]:
        bits.append(f"machine preference {p['machine_preference']}")
    if p["archetype"]:
        bits.append(f"archetype {p['archetype']}")
    ref = "an expert reference route exists" if p["has_expert_reference"] else "no expert reference route (the Route Builder variants are the proxy)"
    return "; ".join(bits) + f"; {ref}."


def _violations(vs, empty):
    if not vs:
        return [f"- {empty}"]
    return [f"- **{v['kind']}**{'' if v['hard'] else ' (soft)'} `{v['rule_id']}`: {v['message']}" for v in vs]


def render_case(case: dict, number: str) -> str:
    label = TYPE_TITLES.get(case["type"], case["type"])
    tag = f"instructive, chosen by the author: {case['note']}" if case["source"] == "instructive" else f"rule: number {case['rank']} of its type in part-id order"
    out = [f"#### {number} {case['part']['part_id']}, repeat {case['repeat']}" + ("" if case["sample_id"] is None else f", sample {case['sample_id']}"),
           "", f"*{tag}.* Case id: `{case['id']}`", "", f"**Part.** {_part_line(case['part'])}", ""]
    if case["type"] == "differs":
        fin, d = case["final"], case["reference_delta"]
        out += [f"**Final route** ({fin['provenance']}, strict-valid): {_route(fin['route'], bold=d['only_in_final'])}", ""]
        if fin["nearest_reference"] is not None:
            ref_label = "expert reference route" if fin["reference_kind"] == "expert" else "Route Builder variant, a proxy, not an expert"
            out += [f"**Closest reference** ({ref_label}): {_route(fin['nearest_reference'], strike=d['only_in_reference'])}", ""]
        if case["from_candidate"]:
            fc = case["from_candidate"]
            out += [f"**The final route was built from sample {fc['sample_id']}.** LLM route as sampled: {_route(fc['llm_route'], strike=(fc['diff'] or {}).get('removed', []))}", ""]
            out += ["Violation report of that route (strict check):"] + _violations(fc["violations"], "none: valid as sampled")
            if fc["feasibility"]:
                out += ["", "Physical-feasibility findings on that route:"] + _violations(fc["feasibility"], "none")
            out += ["", f"Repair: {fc['repair']['outcome']}.", ""]
    else:
        d = case["diff"]
        out += [f"**LLM route** (as sampled): {_route(case['llm_route'], strike=(d or {}).get('removed', []))}", "", "**Violation report** (strict check):"]
        out += _violations(case["violations"], "no violation")
        if case["feasibility"]:
            out += ["", "**Physical-feasibility findings** on the raw route:"] + _violations(case["feasibility"], "none")
        out += ["", f"**Repair applied:** {case['repair']['outcome']}."]
        if case["repair"]["r2"] and case["repair"]["r2"].get("steps"):
            out += ["", f"The model's second answer (R2): {_route(case['repair']['r2']['steps'])}"]
        if case["candidate_route"] and not (d or {}).get("identical"):
            out += ["", f"**Route after the repair:** {_route(case['candidate_route'], bold=(d or {}).get('inserted', []))}"]
        out += ["", f"**Final route of the part** ({case['final']['provenance']}, {'strict-valid' if case['final']['valid'] else 'NOT strict-valid'}"
                    f"{', the route above' if case['selected_as_final'] else ', from another sample'}): {_route(case['final']['route'])}"]
    out += ["", "**Facts from the records:**"] + [f"- {f}" for f in case["facts"]]
    out += ["", f"**Commentary.** {case['commentary'] or GAP}", ""]
    return "\n".join(out)


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_markdown(res: dict) -> str:
    out = ["# Failure case studies", "",
           "Generated by `scripts/case_studies.py` from finished runs (report section 7.5). The numbers and routes are copied from the records; "
           "the commentary is the author's.", "", "## How the cases were selected", "",
           f"- Types: {', '.join(res['types'])}. Per run and type, the first {res['per_type']} parts in the natural order of their identifiers, one case per part "
           "(its first failing route by repeat and sample id). Nothing else decides: not the metrics, not whether the repair worked.",
           "- A failure is a raw LLM route whose strict check lists a violation of that kind, or (infeasible) whose physical-feasibility check lists an "
           "infeasible finding. A route that fails in two ways is a candidate for both types.",
           "- `differs`: the final route is strict-valid but is not an exact match of any reference route. Without expert routes the reference is the set of "
           "Route Builder variants, a proxy: the case then shows where the *rule-based* plan differs, not an expert.",
           "- Cases chosen for being instructive are named by the author, carry the reason, and are labelled; they are not part of the rule.", ""]
    for run in res["runs"]:
        out += [f"## Run `{run['label']}`", "",
                f"System `{run['system']}`" + (f", model `{run['model']}`" if run["model"] else "") + (f", dataset {run['dataset']}" if run["dataset"] else "")
                + f", {run['n_parts']} parts x {len(run['repeats'])} repeat(s), {run['n_parsed_routes']} parsed raw routes"
                + (f", {run['n_unparsed_calls']} call(s) without a parsable route" if run["n_unparsed_calls"] else "")
                + (f", commit {run['git_commit']}" if run["git_commit"] else "") + ".", ""]
        if run["sealed"]:
            out += [f"> Dataset {run['dataset']} is sealed: cases from it belong in the report only if they come from the final run, after the configuration was frozen (7.5).", ""]
        if not run["n_parsed_routes"]:
            out += ["This run has no raw LLM routes (a Route Builder run, or every call failed): there is no failure to study in it.", ""]
            continue
        out += ["| Type | Raw routes with it | Share | Parts | Cases here |", "|---|---|---|---|---|"]
        for row in run["table"]:
            n_cases = sum(1 for c in res["cases"] if c["run"] == run["label"] and c["type"] == row["type"])
            if row["type"] == "differs":
                out.append(f"| {row['type']} | – | – | {row['parts']} | {n_cases} |")
            else:
                out.append(f"| {row['type']} | {row['routes']} | {_pct(row['share'])} | {row['parts']} | {n_cases} |")
        notes = [f"- {r['type']}: {r['note']}" for r in run["table"] if r["note"] and r["type"] != "differs"]
        out += [""] + notes + ([""] if notes else [])
        out += ["The share is of the parsed raw routes (infeasible: of the raw routes that were checked for feasibility); a route can count for several types.", ""]
        mine = [c for c in res["cases"] if c["run"] == run["label"]]
        for t in (*res["types"], "instructive"):
            group = [c for c in mine if c["type"] == t]
            if t != "instructive" and not group:
                out += [f"### {TYPE_TITLES[t]}", "", "No case: " + next((r["note"] for r in run["table"] if r["type"] == t and r["note"]), "no failure of this type in this run") + ".", ""]
                continue
            if not group:
                continue
            out += [f"### {TYPE_TITLES[t]}", ""]
            for i, c in enumerate(group, 1):
                out.append(render_case(c, f"{t[:3] if t != 'instructive' else 'I'}-{i}"))
    out += [f"*{res['n_without_commentary']} of {len(res['cases'])} case(s) still have no commentary.*"
            + (f" *{len(res['orphan_commentary'])} written commentary entr{'y' if len(res['orphan_commentary']) == 1 else 'ies'} match no case of this selection "
               "(kept in the commentary file).*" if res["orphan_commentary"] else ""), ""]
    return "\n".join(out)


SELECTION_FIELDS = ["run", "case_id", "type", "source", "rank", "part_id", "repeat", "sample_id", "candidate", "types", "repair_outcome",
                    "selected_as_final", "final_provenance", "final_valid", "note", "has_commentary"]


def selection_rows(res: dict) -> list:
    rows = []
    for c in res["cases"]:
        rows.append({"run": c["run"], "case_id": c["id"], "type": c["type"], "source": c["source"], "rank": c["rank"],
                     "part_id": c["part"]["part_id"], "repeat": c["repeat"], "sample_id": "" if c["sample_id"] is None else c["sample_id"],
                     "candidate": c["candidate"], "types": ";".join(c.get("types") or []),
                     "repair_outcome": c["repair"]["outcome"] if "repair" in c else (c["from_candidate"]["repair"]["outcome"] if c.get("from_candidate") else ""),
                     "selected_as_final": c.get("selected_as_final", ""), "final_provenance": c["final"]["provenance"], "final_valid": c["final"]["valid"],
                     "note": c["note"], "has_commentary": bool(c["commentary"])})
    return rows


def write_outputs(res: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    p = os.path.join(out_dir, "case_studies.md")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_markdown(res))
    paths["case_studies.md"] = p
    p = os.path.join(out_dir, "cases.json")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
        f.write("\n")
    paths["cases.json"] = p
    p = os.path.join(out_dir, "selection.csv")
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SELECTION_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(selection_rows(res))
    paths["selection.csv"] = p
    return paths
