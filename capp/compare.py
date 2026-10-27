"""Side-by-side comparison of finished runs (results/<tag>_<system>/records.jsonl).

    python scripts/compare_runs.py results/dev_rb results/dev_b1 results/dev_b0 results/dev_b0_r1

Prints one table with the key metrics of every run, then, for every run after the first, a paired comparison with the first
run on the same parts (and repeats): how often the final route is better, equal or worse in Efficiency, and the mean difference.
If one run repaired with R0 and another with R1 on the same raw LLM route (same part, repeat and sample), it also compares the
two repairs on exactly those routes: edit distance and share of the LLM's operations that survive.
"""
import os

from capp import harness, metrics as M, summary as S, systems


def _label(path, records):
    return f"{records[0]['system']}" if records else os.path.basename(path.rstrip("/\\"))


def _fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _stage_val(s, stage, key):
    return s["M5_localized_repair"]["by_stage"].get(stage, {}).get(key)


def _m4_source(s):
    """The reference the M4 numbers of a run are measured against: expert routes if the parts have them, else the Route Builder."""
    m4 = s.get("M4") or {}
    return "expert" if "expert" in m4 else "proxy_rb" if "proxy_rb" in m4 else None


def _m4_val(s, kind, key):
    src = _m4_source(s)
    return "n/a" if src is None else _fmt(s["M4"][src][kind][key])


def _models(s):
    by = (s.get("cost") or {}).get("by_model")
    return "n/a" if not by else "+".join(k.split("/")[-1] for k in by)


def _usd(s):
    c = s.get("cost") or {}
    if not c.get("n"):
        return "n/a"
    return "n/a" if c["usd_per_part"] is None else f"{c['usd_per_part']:.6f}"


def table(runs: list) -> str:
    """runs: list of (label, summary). Returns an aligned text table."""
    def prov(s, name):
        return s["M2"]["provenance_share"].get(name, 0.0)
    rows = [
        ("parts x repeats", lambda s: f"{s['n_parts']}x{s['n_records'] // max(1, s['n_parts'])}"),
        ("LLM samples / part", lambda s: _fmt(s["M7"]["calls_per_part"], 2)),
        ("API calls / part (incl. 2nd step, repair)", lambda s: _fmt(s["M7"]["api_calls_per_part"], 2)),
        ("M1 raw strict-valid", lambda s: _fmt(s["M1"]["strict_valid_of_parsed"])),
        ("M2 final strict-valid", lambda s: _fmt(s["M2"]["final_strict_valid"])),
        ("final raw_llm share", lambda s: _fmt(prov(s, "raw_llm"), 2)),
        ("final repaired_llm share", lambda s: _fmt(prov(s, "repaired_llm"), 2)),
        ("final route_builder share", lambda s: _fmt(prov(s, "route_builder"), 2)),
        ("R0 replacements / part", lambda s: _fmt(s["M2"]["whole_route_replacements_per_part"], 2)),
        ("localized repairs / part", lambda s: _fmt(s["M2"]["localized_repairs_per_part"], 2)),
        ("  by R1 / by R2 (total)", lambda s: f"{s['M2']['repaired_by_r1_total']}/{s['M2']['repaired_by_r2_total']}"),
        ("  fell back to R0 after R1 / R2", lambda s: f"{s['M2']['r1_failed_total']}/{s['M2']['r2_failed_total']}"),
        ("extra re-prompt calls / part", lambda s: _fmt(s["M7"]["repair_calls_per_part"], 2)),
        ("replacement found no route", lambda s: str(s["M2"]["replacement_failed_total"])),
        ("M4 reference (expert, else proxy_rb)", lambda s: _m4_source(s) or "n/a"),
        ("M4 final: exact-match rate", lambda s: _m4_val(s, "final", "exact_rate")),
        ("M4 final: LCS ratio", lambda s: _m4_val(s, "final", "lcs_ratio_mean")),
        ("M4 final: Kendall tau (0 = same order)", lambda s: _m4_val(s, "final", "kendall_tau_mean")),
        ("M4 raw LLM: LCS ratio", lambda s: _m4_val(s, "raw_llm", "lcs_ratio_mean")),
        ("M5 edit distance (R0)", lambda s: _fmt(s["M5_whole_route_replacement"]["edit_distance_mean"], 2)),
        ("M5 LLM ops surviving (R0)", lambda s: _fmt(s["M5_whole_route_replacement"]["survived_share_mean"])),
        ("M5 edit distance (R1 stage)", lambda s: _fmt(_stage_val(s, "r1", "edit_distance_mean"), 2)),
        ("M5 LLM ops surviving (R1 stage)", lambda s: _fmt(_stage_val(s, "r1", "survived_share_mean"))),
        ("M5 edit distance (R2 stage)", lambda s: _fmt(_stage_val(s, "r2", "edit_distance_mean"), 2)),
        ("M5 LLM ops surviving (R2 stage)", lambda s: _fmt(_stage_val(s, "r2", "survived_share_mean"))),
        ("M6 efficiency", lambda s: _fmt(s["M6"]["efficiency_mean"], 2)),
        ("M6 gap to best RB (pts)", lambda s: _fmt(s["M6"]["efficiency_gap_to_rb_best_mean"], 2)),
        ("M6 time / cost / energy gap", lambda s: "/".join(_fmt(s["M6"][k], 3) for k in (
            "time_gap_to_rb_best_mean", "cost_gap_to_rb_best_mean", "energy_gap_to_rb_best_mean"))),
        ("prompt tokens / part (samples)", lambda s: _fmt(s["M7"]["prompt_tokens_per_part"], 0)),
        ("models called", lambda s: _models(s)),
        ("cost: list-price USD / part", lambda s: _usd(s)),
        ("cost: tokens / part (all calls)", lambda s: _fmt((s.get("cost") or {}).get("tokens_per_part"), 0)),
        ("cost: sum of call latencies s / part", lambda s: _fmt((s.get("cost") or {}).get("latency_s_per_part"), 2)),
        ("cascade: escalated sample slots", lambda s: _fmt((s.get("cascade") or {}).get("escalation_rate"), 2) if (s.get("cascade") or {}).get("n") else "-"),
        ("cascade: final route from large model", lambda s: _fmt((s.get("cascade") or {}).get("parts_with_final_from_large"), 2) if (s.get("cascade") or {}).get("n") else "-"),
        ("retrieved examples / call", lambda s: _fmt((s.get("planner_prompt") or {}).get("examples_per_call"), 2)),
        ("physics: cutting energy kWh/part", lambda s: _fmt((s.get("physics") or {}).get("energy_kwh_mean"), 3)),
        ("physics: tool wear INR/part", lambda s: _fmt((s.get("physics") or {}).get("wear_cost_inr_mean"), 1)),
        ("physics: parts over machine power", lambda s: _fmt((s.get("physics") or {}).get("over_power_share"), 2)),
        ("physics: Ra risk high share", lambda s: _fmt(((s.get("physics") or {}).get("ra_risk_share") or {}).get("high"), 2)),
        ("feasibility: finals with a hard infeasibility", lambda s: _fmt((s.get("feasibility") or {}).get("final_infeasible_share"), 2)),
        ("feasibility: finals with risks", lambda s: _fmt((s.get("feasibility") or {}).get("final_risk_share"), 2)),
        ("feasibility: strict-valid raw LLM routes infeasible", lambda s: _fmt((s.get("feasibility") or {}).get("raw_valid_infeasible_share"), 3)),
        ("2-step: stated order kept by step 2", lambda s: _fmt((s.get("two_step") or {}).get("faithful_share"), 2)),
        ("constrained: choice calls / route", lambda s: _fmt((s.get("constrained") or {}).get("choice_calls_per_sample_mean"), 1) if (s.get("constrained") or {}).get("n") else "-"),
        ("constrained: unreadable answers (share of choices)", lambda s: _fmt((s.get("constrained") or {}).get("fallback_share_of_choices"), 3) if (s.get("constrained") or {}).get("n") else "-"),
        ("front size mean", lambda s: _fmt(s["front_size_mean"], 2)),
        ("selection", lambda s: "/".join(s.get("selection") or []) or "n/a"),
        ("plan confidence mean", lambda s: _fmt((s.get("confidence") or {}).get("mean"))),
        ("plans flagged for review", lambda s: _fmt((s.get("confidence") or {}).get("flagged_share"), 2)),
    ]
    w0 = max(len(r[0]) for r in rows)
    cols = [max(len(lab), 10) for lab, _ in runs]
    out = [" " * w0 + "  " + "  ".join(lab.rjust(w) for (lab, _), w in zip(runs, cols))]
    for name, fn in rows:
        out.append(name.ljust(w0) + "  " + "  ".join(fn(s).rjust(w) for (_, s), w in zip(runs, cols)))
    return "\n".join(out)


def _final_eff(rec):
    f = rec["final"]
    return f["scores_eval"]["efficiency_score"] if f["check"]["valid"] else None


def paired_quality(base_recs: list, other_recs: list) -> dict:
    """Per (part, repeat) difference in final Efficiency, other minus base, on parts where both finals are strict-valid."""
    base = {(r["part_id"], r["repeat"]): _final_eff(r) for r in base_recs}
    diffs = []
    for r in other_recs:
        b, o = base.get((r["part_id"], r["repeat"])), _final_eff(r)
        if b is not None and o is not None:
            diffs.append(o - b)
    eps = 1e-9
    return {"n": len(diffs), "mean_diff": M.mean_std(diffs)[0],
            "better": sum(1 for d in diffs if d > eps), "equal": sum(1 for d in diffs if abs(d) <= eps),
            "worse": sum(1 for d in diffs if d < -eps)}


def _repair_index(records):
    out = {}
    for rec in records:
        for c in rec["candidates"]:
            if c["llm_steps"] and (c["replaced"] or c.get("repaired")):
                out[(rec["part_id"], rec["repeat"], c["sample_id"])] = c
    return out


def paired_repairs(recs_r0: list, recs_r1: list) -> dict:
    """R0 (replaced) against R1 (repaired) on the very same raw LLM routes."""
    a, b = _repair_index(recs_r0), _repair_index(recs_r1)
    ed0, ed1, sv0, sv1 = [], [], [], []
    for k, c0 in a.items():
        c1 = b.get(k)
        if c1 is None or not c0["replaced"] or not c1.get("repaired") or c0["llm_steps"] != c1["llm_steps"]:
            continue
        ed0.append(M.edit_distance(c0["llm_steps"], c0["steps"]))
        ed1.append(M.edit_distance(c1["llm_steps"], c1["steps"]))
        sv0.append(M.survived_share(c0["llm_steps"], c0["steps"]))
        sv1.append(M.survived_share(c1["llm_steps"], c1["steps"]))
    return {"n": len(ed0), "edit_distance_r0": M.mean_std(ed0)[0], "edit_distance_r1": M.mean_std(ed1)[0],
            "survived_r0": M.mean_std(sv0)[0], "survived_r1": M.mean_std(sv1)[0]}


def _strategy(records):
    sysname = records[0]["system"] if records else ""
    try:
        spec = systems.parse_system(sysname)
    except ValueError:
        spec = None
    return spec[1].upper() if spec else "R0"


def compare(dirs: list) -> str:
    loaded = []
    for d in dirs:
        recs = harness.load_records(d)
        loaded.append((_label(d, recs), recs))
    labels = [lab for lab, _ in loaded]
    out = [table([(lab, S.summarize(recs)) for lab, recs in loaded])]
    base_lab, base_recs = loaded[0]
    for lab, recs in loaded[1:]:
        q = paired_quality(base_recs, recs)
        out.append(f"\nPaired final Efficiency, {lab} vs {base_lab}: {q['n']} (part, repeat) pairs, mean difference "
                   f"{_fmt(q['mean_diff'], 2)} pts; better {q['better']}, equal {q['equal']}, worse {q['worse']}")
    for lab0, r0 in loaded:
        for lab1, r1 in loaded:
            if lab0 == lab1 or not any(c["replaced"] for r in r0 for c in r["candidates"]):
                continue
            if not any(c.get("repaired") for r in r1 for c in r["candidates"]):
                continue
            p = paired_repairs(r0, r1)
            if p["n"]:
                out.append(f"\nSame raw LLM routes, R0 ({lab0}) vs {_strategy(r1)} ({lab1}): n={p['n']}; edit distance "
                           f"{_fmt(p['edit_distance_r0'], 2)} vs {_fmt(p['edit_distance_r1'], 2)}; LLM operations surviving "
                           f"{_fmt(p['survived_r0'])} vs {_fmt(p['survived_r1'])}")
    if len(set(labels)) != len(labels):
        out.append("\nNote: two runs share a system name; label columns by directory if this is confusing.")
    return "\n".join(out)
