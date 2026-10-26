"""Turns run records (RunRecord.to_dict() plus the part and the repeat number) into flat tables and the metric summary.

Pure functions: no API calls, no files. The harness (capp/harness.py) writes what this returns.

Metrics reported (Table 3 of the report)
  M1  first-pass validity of the raw LLM routes, v1 notion and strict notion, reported overall and per part class
  M2  final validity and provenance of the final routes, number of whole-route replacements (fallbacks)
  M3  failure kinds of the invalid raw LLM routes (a route can have several kinds)
  M4  plan similarity: against expert reference routes if the part has them, otherwise against the Route Builder
      variants (labelled "proxy_rb": agreement with the rule-based routes, not with an expert)
  M5  repair minimality: edit distance LLM route -> final route and share of LLM operations that survive, separately for
      v1's whole-route replacement (R0) and for localized repair (R1)
  M6  plan quality of the final routes and the gap to the best Route Builder variant (negative gap = better than all of them).
      Scored fairly: every route gets one changeover marker per machine switch first (v1 charges only markers that are in
      the route, and LLM routes never contain any)
  M7  inference cost: calls, tokens, latency
  M8  stability over repeated runs (mean and std over repeats of M1, M4 and M6)
"""
from capp import metrics as M

COMPLEXITY_BUCKETS = (("simple", 1, 3), ("medium", 4, 5), ("complex", 6, 99))


def complexity(n_features: int) -> str:
    for name, lo, hi in COMPLEXITY_BUCKETS:
        if lo <= n_features <= hi:
            return name
    return "complex"


def _refs_for(rec):
    expert = rec["part"].get("reference_routes")
    if expert:
        return expert, "expert"
    return rec["references"], "proxy_rb"


def _gap_to_rb_best(scores, best_rb):
    if not scores or not best_rb:
        return None
    return M.quality_gap(scores, M.best_known(best_rb))


def part_rows(records: list) -> list:
    rows = []
    for rec in records:
        part, info, fin = rec["part"], rec["info"], rec["final"]
        refs, src = _refs_for(rec)
        sim = M.best_similarity(fin["steps"], refs) if fin["steps"] else None
        gap = _gap_to_rb_best(fin["scores_eval"], rec["best_scores"]) if fin["check"]["valid"] else None
        cands = rec["candidates"]
        rows.append({
            "system": rec["system"], "repeat": rec["repeat"], "part_id": rec["part_id"],
            "archetype": part.get("archetype", ""), "machine_profile": part.get("machine_profile", ""),
            "complexity": complexity(len(part["features"])), "n_features": len(part["features"]),
            "in_distribution": part.get("in_distribution", ""),
            "material": part["material"], "tolerance": part.get("tolerance", ""), "batch_size": part.get("batch_size", ""),
            "attempts": info["attempts"], "llm_parsed": sum(1 for c in cands if c["llm_ok"]),
            "raw_v1_valid": info["raw_v1_valid"], "raw_strict_valid": info.get("raw_strict_valid", 0),
            "replaced": info["replaced"], "repaired": info.get("repaired", 0), "r1_failed": info.get("r1_failed", 0),
            "deduped": info["deduped"], "safety_net": info["safety_net"],
            "population": info["population"], "front_size": len(rec["front"]),
            "final_name": fin["name"], "final_provenance": fin["provenance"],
            "final_strict_valid": fin["check"]["valid"], "final_v1_valid": fin["check"]["v1_valid"],
            "final_kinds": ";".join(fin["check"]["kinds"]), "final_route": " > ".join(
                s for s in fin["steps"] if not s.startswith("---")),
            "time_min": fin["scores_eval"]["time_min"], "cost_inr": fin["scores_eval"]["cost_inr"],
            "energy_kwh": fin["scores_eval"]["energy_kwh"], "efficiency": fin["scores_eval"]["efficiency_score"],
            "efficiency_v1": fin["scores"]["efficiency_score"],
            "eff_gap_to_rb_best": None if gap is None else gap["efficiency_gap"],
            "m4_source": src if sim else "", "m4_exact": None if not sim else sim["exact"],
            "m4_lcs_ratio": None if not sim else sim["lcs_ratio"],
            "m4_kendall_tau": None if not sim else sim["kendall_tau"],
        })
    return rows


def _repair_label(c):
    r = c.get("repair")
    if not r:
        return ""
    if r.get("fallback"):
        return r["strategy"] + "->" + r["fallback"]
    stage = r.get("stage") or r["strategy"]
    return r["strategy"] if stage == r["strategy"] else f"{r['strategy']}:{stage}"


def _stage(c):
    """Which stage produced a repaired route: r1 or r2 (records written before R2 existed have no 'stage': r1)."""
    return c["repair"].get("stage") or "r1"


def _r1_details(c):
    """R1 details of a repair attempt (new records nest them under 'r1'; Step 5 records have them at the top level)."""
    r = c["repair"]
    return r.get("r1") or (r if "discordant_pairs" in r else None)


def candidate_rows(records: list) -> list:
    rows = []
    for rec in records:
        for c in rec["candidates"]:
            fp = c["first_pass"]
            llm, fin = c["llm_steps"], c["steps"]
            rows.append({
                "system": rec["system"], "repeat": rec["repeat"], "part_id": rec["part_id"],
                "sample_id": c["sample_id"], "name": c["name"], "provenance": c["provenance"],
                "llm_ok": c["llm_ok"], "in_pool": c["in_pool"], "deduped": c["deduped"], "replaced": c["replaced"],
                "repaired": c.get("repaired", False), "repair_strategy": _repair_label(c),
                "v1_valid": c["v1_valid"], "strict_valid": None if fp is None else fp["valid"],
                "kinds": "" if fp is None else ";".join(fp["kinds"]),
                "llm_route": "" if not llm else " > ".join(llm), "route": " > ".join(
                    s for s in fin if not s.startswith("---")),
                "edit_distance": M.edit_distance(llm, fin) if ((c["replaced"] or c.get("repaired")) and llm) else None,
                "survived_share": M.survived_share(llm, fin) if ((c["replaced"] or c.get("repaired")) and llm) else None,
                "finish_reason": c["meta"].get("finish_reason", ""), "prompt_tokens": c["meta"].get("prompt_tokens"),
                "completion_tokens": c["meta"].get("completion_tokens"), "latency_s": c["meta"].get("latency_s"),
                "cached": c["meta"].get("cached"),
            })
    return rows


def _parsed(records):
    return [(rec, c) for rec in records for c in rec["candidates"] if c["llm_ok"] and c["first_pass"] is not None]


def _calls(records):
    return [(rec, c) for rec in records for c in rec["candidates"] if c["meta"]]


def _m4(records):
    out = {}
    for src in ("expert", "proxy_rb"):
        fin, raw = [], []
        for rec in records:
            refs, s = _refs_for(rec)
            if s != src or not refs:
                continue
            if rec["final"]["steps"]:
                fin.append(M.best_similarity(rec["final"]["steps"], refs))
            raw += [M.best_similarity(c["llm_steps"], refs) for c in rec["candidates"] if c["llm_ok"] and c["llm_steps"]]
        if not fin and not raw:
            continue

        def agg(sims):
            return {"n": len(sims), "exact_rate": M.rate([s["exact"] for s in sims]),
                    "lcs_ratio_mean": M.mean_std([s["lcs_ratio"] for s in sims])[0],
                    "kendall_tau_mean": M.mean_std([s["kendall_tau"] for s in sims])[0]}
        out[src] = {"final": agg(fin), "raw_llm": agg(raw)}
    return out


def summarize_set(records: list) -> dict:
    """Metric summary for one set of records (one system, any number of repeats)."""
    if not records:
        return {"n_records": 0}
    parsed = _parsed(records)
    attempts = sum(r["info"]["attempts"] for r in records)
    pooled = [c for r in records for c in r["candidates"] if c["llm_ok"]]
    kinds = {}
    for _, c in parsed:
        if not c["first_pass"]["valid"]:
            for k in c["first_pass"]["kinds"]:
                kinds[k] = kinds.get(k, 0) + 1
    soft = sum(1 for _, c in parsed if "machine_grouping" in c["first_pass"]["kinds"])
    finals = [r["final"] for r in records]
    prov = {}
    for f in finals:
        prov[f["provenance"]] = prov.get(f["provenance"], 0) + 1
    replaced = [c for r in records for c in r["candidates"] if c["replaced"] and c["llm_steps"]]
    repaired = [c for r in records for c in r["candidates"] if c.get("repaired") and c["llm_steps"]]
    repaired_r1 = [c for c in repaired if _stage(c) == "r1"]
    repaired_r2 = [c for c in repaired if _stage(c) == "r2"]
    reprompts = [c["repair"]["r2"] for r in records for c in r["candidates"] if c.get("repair") and c["repair"].get("r2")]
    ok_finals = [(r, r["final"]) for r in records if r["final"]["check"]["valid"]]
    gaps = [_gap_to_rb_best(f["scores_eval"], r["best_scores"]) for r, f in ok_finals]
    gaps = [g for g in gaps if g]
    calls = _calls(records)
    lat = [c["meta"].get("latency_s") for _, c in calls]
    n = len(records)
    out = {
        "n_records": n,
        "n_parts": len({r["part_id"] for r in records}),
        "M1": {
            "llm_calls": attempts, "parsed_routes": len(parsed), "parse_rate": M.rate([True] * len(parsed) + [False] * (attempts - len(parsed))),
            "v1_valid_of_parsed": M.rate([c["v1_valid"] for _, c in parsed]),
            "strict_valid_of_parsed": M.rate([c["first_pass"]["valid"] for _, c in parsed]),
            "v1_valid_but_strict_invalid": sum(1 for _, c in parsed if c["v1_valid"] and not c["first_pass"]["valid"]),
        },
        "M2": {
            "final_strict_valid": M.rate([f["check"]["valid"] for f in finals]),
            "final_v1_valid": M.rate([f["check"]["v1_valid"] for f in finals]),
            "provenance_share": {k: v / n for k, v in sorted(prov.items())},
            "whole_route_replacements_total": sum(r["info"]["replaced"] for r in records),
            "whole_route_replacements_per_part": sum(r["info"]["replaced"] for r in records) / n,
            "replacement_failed_total": sum(r["info"].get("replacement_failed", 0) for r in records),
            "localized_repairs_total": sum(r["info"].get("repaired", 0) for r in records),
            "localized_repairs_per_part": sum(r["info"].get("repaired", 0) for r in records) / n,
            "r1_failed_total": sum(r["info"].get("r1_failed", 0) for r in records),
            "repaired_by_r1_total": sum(r["info"].get("repaired_r1", 0) for r in records),
            "repaired_by_r2_total": sum(r["info"].get("repaired_r2", 0) for r in records),
            "r2_failed_total": sum(r["info"].get("r2_failed", 0) for r in records),
            "safety_net_parts": sum(1 for r in records if r["info"]["safety_net"]),
            "incomplete_fallback_parts": sum(1 for r in records if r["info"]["fallback_incomplete"]),
            "raw_llm_finals_strict_invalid": sum(1 for f in finals if f["provenance"] == "raw_llm" and not f["check"]["valid"]),
        },
        "M3": {"invalid_raw_routes": sum(1 for _, c in parsed if not c["first_pass"]["valid"]),
               "kinds_among_invalid": dict(sorted(kinds.items())), "soft_machine_grouping_in_parsed": soft},
        "M4": _m4(records),
        "M5_whole_route_replacement": {
            "n_replaced": len(replaced),
            "edit_distance_mean": M.mean_std([M.edit_distance(c["llm_steps"], c["steps"]) for c in replaced])[0],
            "survived_share_mean": M.mean_std([M.survived_share(c["llm_steps"], c["steps"]) for c in replaced])[0]},
        "M5_localized_repair": {
            "n_repaired": len(repaired),
            "edit_distance_mean": M.mean_std([M.edit_distance(c["llm_steps"], c["steps"]) for c in repaired])[0],
            "survived_share_mean": M.mean_std([M.survived_share(c["llm_steps"], c["steps"]) for c in repaired])[0],
            "discordant_pairs_mean": M.mean_std([_r1_details(c)["discordant_pairs"] for c in repaired_r1])[0],
            "operations_inserted_mean": M.mean_std([len(_r1_details(c)["inserted"]) for c in repaired_r1])[0],
            "operations_removed_mean": M.mean_std([len(_r1_details(c)["removed_unknown"]) + len(_r1_details(c)["removed_duplicates"])
                                                   for c in repaired_r1])[0],
            "by_stage": {st: {"n": len(cs),
                              "edit_distance_mean": M.mean_std([M.edit_distance(c["llm_steps"], c["steps"]) for c in cs])[0],
                              "survived_share_mean": M.mean_std([M.survived_share(c["llm_steps"], c["steps"]) for c in cs])[0]}
                         for st, cs in (("r1", repaired_r1), ("r2", repaired_r2)) if cs}},
        "M6": {
            "n_valid_finals": len(ok_finals),
            "time_min_mean": M.mean_std([f["scores_eval"]["time_min"] for _, f in ok_finals])[0],
            "cost_inr_mean": M.mean_std([f["scores_eval"]["cost_inr"] for _, f in ok_finals])[0],
            "energy_kwh_mean": M.mean_std([f["scores_eval"]["energy_kwh"] for _, f in ok_finals])[0],
            "efficiency_mean": M.mean_std([f["scores_eval"]["efficiency_score"] for _, f in ok_finals])[0],
            "efficiency_gap_to_rb_best_mean": M.mean_std([g["efficiency_gap"] for g in gaps])[0],
            "time_gap_to_rb_best_mean": M.mean_std([g["time_min_gap"] for g in gaps])[0],
            "cost_gap_to_rb_best_mean": M.mean_std([g["cost_inr_gap"] for g in gaps])[0],
            "energy_gap_to_rb_best_mean": M.mean_std([g["energy_kwh_gap"] for g in gaps])[0]},
        "M7": {
            "calls_per_part": attempts / n,
            "prompt_tokens_per_part": sum(c["meta"].get("prompt_tokens") or 0 for _, c in calls) / n,
            "completion_tokens_per_part": sum(c["meta"].get("completion_tokens") or 0 for _, c in calls) / n,
            "latency_mean_s": M.mean_std(lat)[0], "latency_p50_s": M.percentile(lat, 0.5), "latency_p95_s": M.percentile(lat, 0.95),
            "truncated_calls": sum(1 for _, c in calls if c["meta"].get("finish_reason") == "length"),
            "cache_hit_rate": M.rate([bool(c["meta"].get("cached")) for _, c in calls]),
            "repair_calls_per_part": sum(r["info"].get("repair_calls", 0) for r in records) / n,
            "total_calls_per_part": (attempts + sum(r["info"].get("repair_calls", 0) for r in records)) / n,
            "repair_prompt_tokens_per_part": sum(m["meta"].get("prompt_tokens") or 0 for m in reprompts) / n,
            "repair_completion_tokens_per_part": sum(m["meta"].get("completion_tokens") or 0 for m in reprompts) / n,
            "repair_latency_mean_s": M.mean_std([m["meta"].get("latency_s") for m in reprompts])[0],
            "repair_cache_hit_rate": M.rate([bool(m["meta"].get("cached")) for m in reprompts])},
        "front_size_mean": M.mean_std([len(r["front"]) for r in records if r["front"]])[0],
        "pooled_llm_parsed": len(pooled),
    }
    return out


def _compact(records):
    s = summarize_set(records)
    if s["n_records"] == 0:
        return s
    return {"n_records": s["n_records"], "M1_strict_valid_of_parsed": s["M1"]["strict_valid_of_parsed"],
            "M2_final_strict_valid": s["M2"]["final_strict_valid"],
            "M6_efficiency_gap_to_rb_best": s["M6"]["efficiency_gap_to_rb_best_mean"]}


def summarize(records: list) -> dict:
    """Overall summary of one system's records, plus per-class breakdowns and stability over repeats (M8)."""
    out = summarize_set(records)
    if not records:
        return out
    by_class = {}
    for key, getter in (("machine_profile", lambda r: r["part"].get("machine_profile", "?")),
                        ("complexity", lambda r: complexity(len(r["part"]["features"]))),
                        ("in_distribution", lambda r: str(r["part"].get("in_distribution", "?")))):
        groups = {}
        for r in records:
            groups.setdefault(getter(r), []).append(r)
        by_class[key] = {k: _compact(v) for k, v in sorted(groups.items())}
    out["by_class"] = by_class
    repeats = sorted({r["repeat"] for r in records})
    if len(repeats) > 1:
        per = [summarize_set([r for r in records if r["repeat"] == k]) for k in repeats]

        def ms(vals):
            m, s = M.mean_std(vals)
            return {"mean": m, "std": s, "values": vals}
        out["M8"] = {
            "repeats": len(repeats),
            "M1_strict_valid_of_parsed": ms([p["M1"]["strict_valid_of_parsed"] for p in per]),
            "M2_final_strict_valid": ms([p["M2"]["final_strict_valid"] for p in per]),
            "M6_efficiency_mean": ms([p["M6"]["efficiency_mean"] for p in per]),
            "M4_raw_lcs_ratio_mean": ms([(p["M4"].get("expert") or p["M4"].get("proxy_rb") or {}).get("raw_llm", {}).get("lcs_ratio_mean")
                                         for p in per]),
        }
    return out


def _fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def format_summary(system: str, s: dict) -> str:
    if s["n_records"] == 0:
        return f"{system}: no records"
    m1, m2, m3, m5, m6, m7 = s["M1"], s["M2"], s["M3"], s["M5_whole_route_replacement"], s["M6"], s["M7"]
    m5r = s["M5_localized_repair"]
    L = [f"=== {system}: {s['n_parts']} parts, {s['n_records']} runs ==="]
    L.append(f"M1 first pass   : LLM calls {m1['llm_calls']}, parsed {m1['parsed_routes']} (parse rate {_fmt(m1['parse_rate'])}); "
             f"v1-valid {_fmt(m1['v1_valid_of_parsed'])}, strict-valid {_fmt(m1['strict_valid_of_parsed'])} "
             f"(v1-valid but strict-invalid: {m1['v1_valid_but_strict_invalid']})")
    L.append(f"M2 final        : strict-valid {_fmt(m2['final_strict_valid'])}, v1-valid {_fmt(m2['final_v1_valid'])}; "
             f"provenance {', '.join(f'{k} {v:.2f}' for k, v in m2['provenance_share'].items())}; "
             f"replacements/part {_fmt(m2['whole_route_replacements_per_part'], 2)} (replacement found no route: "
             f"{m2['replacement_failed_total']}), safety-net parts {m2['safety_net_parts']}, "
             f"raw-LLM finals that are strict-invalid {m2['raw_llm_finals_strict_invalid']}")
    L.append(f"M3 failure kinds: {m3['kinds_among_invalid'] or 'none'} among {m3['invalid_raw_routes']} invalid raw routes "
             f"(soft machine-grouping flags: {m3['soft_machine_grouping_in_parsed']})")
    for src, v in s["M4"].items():
        L.append(f"M4 similarity [{src}] final: exact {_fmt(v['final']['exact_rate'])}, LCS {_fmt(v['final']['lcs_ratio_mean'])}, "
                 f"tau {_fmt(v['final']['kendall_tau_mean'])} | raw LLM: exact {_fmt(v['raw_llm']['exact_rate'])}, "
                 f"LCS {_fmt(v['raw_llm']['lcs_ratio_mean'])}, tau {_fmt(v['raw_llm']['kendall_tau_mean'])}")
    L.append(f"M5 replacement  : {m5['n_replaced']} replaced; edit distance {_fmt(m5['edit_distance_mean'], 2)}, "
             f"LLM operations surviving {_fmt(m5['survived_share_mean'])}")
    if m5r["n_repaired"] or m2["r1_failed_total"] or m2["r2_failed_total"]:
        L.append(f"M5 localized    : {m5r['n_repaired']} repaired (by R1 {m2['repaired_by_r1_total']}, by R2 {m2['repaired_by_r2_total']}; "
                 f"fell back to R0 after R1 failed {m2['r1_failed_total']}, after R2 failed {m2['r2_failed_total']}); "
                 f"edit distance {_fmt(m5r['edit_distance_mean'], 2)}, LLM operations surviving {_fmt(m5r['survived_share_mean'])}")
        for st, v in m5r["by_stage"].items():
            L.append(f"                  {st.upper()}: n={v['n']}, edit distance {_fmt(v['edit_distance_mean'], 2)}, "
                     f"surviving {_fmt(v['survived_share_mean'])}")
        if m5r["discordant_pairs_mean"] is not None:
            L.append(f"                  R1 edits: discordant pairs {_fmt(m5r['discordant_pairs_mean'], 2)}, ops inserted "
                     f"{_fmt(m5r['operations_inserted_mean'], 2)}, ops removed {_fmt(m5r['operations_removed_mean'], 2)}")
    L.append(f"M6 quality      : efficiency {_fmt(m6['efficiency_mean'], 2)}, gap to best RB variant: efficiency "
             f"{_fmt(m6['efficiency_gap_to_rb_best_mean'], 2)} pts, time {_fmt(m6['time_gap_to_rb_best_mean'])}, "
             f"cost {_fmt(m6['cost_gap_to_rb_best_mean'])}, energy {_fmt(m6['energy_gap_to_rb_best_mean'])} (relative)")
    rp = (f" + {_fmt(m7['repair_calls_per_part'], 2)} repair calls ({_fmt(m7['repair_prompt_tokens_per_part'], 0)}+"
          f"{_fmt(m7['repair_completion_tokens_per_part'], 0)} tokens, latency {_fmt(m7['repair_latency_mean_s'], 2)}s, "
          f"cache hit {_fmt(m7['repair_cache_hit_rate'], 2)})") if m7["repair_calls_per_part"] else ""
    L.append(f"M7 cost         : {_fmt(m7['calls_per_part'], 2)} calls/part{rp}, {_fmt(m7['prompt_tokens_per_part'], 0)}+"
             f"{_fmt(m7['completion_tokens_per_part'], 0)} tokens/part (samples), latency mean {_fmt(m7['latency_mean_s'], 2)}s "
             f"p50 {_fmt(m7['latency_p50_s'], 2)}s p95 {_fmt(m7['latency_p95_s'], 2)}s, truncated {m7['truncated_calls']}, "
             f"cache hit rate {_fmt(m7['cache_hit_rate'], 2)}")
    L.append(f"front size mean : {_fmt(s['front_size_mean'], 2)}")
    for key, groups in s.get("by_class", {}).items():
        L.append(f"  by {key}: " + "; ".join(
            f"{k} (n={v['n_records']}) M1 {_fmt(v['M1_strict_valid_of_parsed'], 2)} M2 {_fmt(v['M2_final_strict_valid'], 2)}"
            for k, v in groups.items()))
    if "M8" in s:
        m8 = s["M8"]
        L.append(f"M8 stability over {m8['repeats']} repeats (mean +/- std): "
                 f"M1 {_fmt(m8['M1_strict_valid_of_parsed']['mean'])} +/- {_fmt(m8['M1_strict_valid_of_parsed']['std'])}, "
                 f"M2 {_fmt(m8['M2_final_strict_valid']['mean'])} +/- {_fmt(m8['M2_final_strict_valid']['std'])}, "
                 f"efficiency {_fmt(m8['M6_efficiency_mean']['mean'], 2)} +/- {_fmt(m8['M6_efficiency_mean']['std'], 2)}")
    return "\n".join(L)
