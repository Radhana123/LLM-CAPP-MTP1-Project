"""Per-part comparison of two finished runs (Step 15): on which parts is run B worse than run A, and what is different in the routes there.

compare_runs.py gives the averages ("n5_r1_s3 is 3.2 pts below rb, worse on 7 of 20 pairs"). This module answers the next question: WHICH
pairs, and what differs. Pure functions on the records of two runs (results/<tag>_<system>/records.jsonl), no API calls. The first run
(A, "base", normally rb) is the reference, the second (B, "other", the LLM system) is judged against it, as in compare_runs.

What is computed for every (part, repeat) that both runs have and whose features and material agree:
    eff_diff          final Efficiency of B minus A (the v1 Efficiency, scored with changeover markers added); None unless both finals
                      are strict-valid. verdict: better / equal / worse (|diff| <= 1e-9 is equal), n/a if not comparable
    time / cost / energy     the three numbers behind Efficiency, with the relative difference (B - A) / A
    route delta       operations only in A, only in B (in route order), number of machine changeovers, and one cause tag:
                      identical / order only / other has extra operations / other lacks operations / different operations
    provenance        raw_llm / repaired_llm / route_builder of both finals
    physics, feasibility     cutting energy, tool wear, peak spindle power, over-power, Ra risk, hard infeasibilities, risks, both runs

What the summary adds: the same pairs grouped by part class (machine profile, complexity, material, tolerance ... any column of the
rows), by cause tag and by provenance of B's final, the operations that B has extra or lacks on the pairs it is worse on, and which of
time / cost / energy moves the Efficiency difference.

Read it as description, not as causation: a cause tag says how the routes differ, not why Efficiency differs; an operation that shows up
on the worse pairs may simply be common. With few pairs (n < 10 per row of a table) nothing here is statistically meaningful.

    rows, unmatched = partdiff.pair_rows(base_records, other_records)
    summ = partdiff.summarize(rows)
    print(partdiff.render(rows, summ, "rb", "n5_r1_s3", top=10))
    partdiff.write_csv("pairs.csv", rows)
"""
import csv
from collections import Counter, defaultdict

from capp import metrics as M
from capp import summary as S
from capp import verify as V

EPS = 1e-9
GROUPS = ("machine_profile", "complexity", "material", "tolerance")
CAUSES = ("identical", "order only", "other has extra operations", "other lacks operations", "different operations")


def route_delta(base: list, other: list) -> dict:
    """How two routes differ, markers ignored: operations only in base / only in other (in route order) and one cause tag."""
    b, o = V.clean_steps(base), V.clean_steps(other)
    cb, co = Counter(b), Counter(o)
    miss, extra = cb - co, co - cb                       # multiset differences: what other lacks, what other adds
    only_base, only_other = [], []
    left = Counter(miss)
    for s in b:
        if left[s] > 0:
            only_base.append(s)
            left[s] -= 1
    left = Counter(extra)
    for s in o:
        if left[s] > 0:
            only_other.append(s)
            left[s] -= 1
    if b == o:
        cause = "identical"
    elif not miss and not extra:
        cause = "order only"
    elif not miss:
        cause = "other has extra operations"
    elif not extra:
        cause = "other lacks operations"
    else:
        cause = "different operations"
    return {"only_base": only_base, "only_other": only_other, "cause": cause,
            "changeovers_base": V.implied_changeovers(b), "changeovers_other": V.implied_changeovers(o)}


def _verdict(diff):
    if diff is None:
        return "n/a"
    return "better" if diff > EPS else "worse" if diff < -EPS else "equal"


def _rel(base, other):
    if base is None or other is None or abs(base) < EPS:
        return None
    return (other - base) / base


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _by_key(recs):
    return {(r["part_id"], r["repeat"]): r for r in recs}


def pair_rows(base_recs: list, other_recs: list):
    """(rows, unmatched). Rows are sorted as in `other_recs`. unmatched = {"only_base": [...], "only_other": [...], "mismatch": [...]}
    (lists of (part_id, repeat)); a pair whose features or material differ between the runs comes from a different dataset and is
    not compared."""
    rb, ro = _by_key(base_recs), _by_key(other_recs)
    pb = {(r["part_id"], r["repeat"]): r for r in S.part_rows(base_recs)}
    po = {(r["part_id"], r["repeat"]): r for r in S.part_rows(other_recs)}
    unmatched = {"only_base": [k for k in rb if k not in ro], "only_other": [k for k in ro if k not in rb], "mismatch": []}
    rows = []
    for key, rec_o in ro.items():
        rec_b = rb.get(key)
        if rec_b is None:
            continue
        part = rec_o["part"]
        if rec_b["part"]["features"] != part["features"] or rec_b["part"]["material"] != part["material"]:
            unmatched["mismatch"].append(key)
            continue
        fb, fo = rec_b["final"], rec_o["final"]
        vb, vo = bool(fb["check"]["valid"]), bool(fo["check"]["valid"])
        sb, so = fb["scores_eval"], fo["scores_eval"]
        eff_b = sb["efficiency_score"] if vb else None
        eff_o = so["efficiency_score"] if vo else None
        diff = None if eff_b is None or eff_o is None else eff_o - eff_b
        delta = route_delta(fb["steps"], fo["steps"])
        rb_row, ro_row = pb[key], po[key]
        row = {
            "part_id": key[0], "repeat": key[1], "archetype": part.get("archetype", ""),
            "machine_profile": part.get("machine_profile", ""), "complexity": ro_row["complexity"], "material": part["material"],
            "tolerance": part.get("tolerance", ""), "batch_size": part.get("batch_size", ""),
            "n_features": len(part["features"]), "features": ";".join(part["features"]),
            "valid_base": vb, "valid_other": vo, "eff_base": eff_b, "eff_other": eff_o, "eff_diff": diff, "verdict": _verdict(diff),
            "provenance_base": fb["provenance"], "provenance_other": fo["provenance"],
            "route_base": " > ".join(V.clean_steps(fb["steps"])), "route_other": " > ".join(V.clean_steps(fo["steps"])),
            "n_ops_base": len(V.clean_steps(fb["steps"])), "n_ops_other": len(V.clean_steps(fo["steps"])),
            "cause": delta["cause"], "only_base": delta["only_base"], "only_other": delta["only_other"],
            "changeovers_base": delta["changeovers_base"], "changeovers_other": delta["changeovers_other"],
        }
        for name in ("time_min", "cost_inr", "energy_kwh"):
            row[name + "_base"], row[name + "_other"] = sb[name], so[name]
            row[name + "_rel"] = _rel(sb[name], so[name])
        for name in ("phys_energy_kwh", "phys_wear_inr", "phys_peak_spindle_kw", "over_power", "ra_risk", "feas_infeasible", "feas_risk",
                     "feas_rules"):
            row[name + "_base"], row[name + "_other"] = rb_row.get(name, ""), ro_row.get(name, "")
        rows.append(row)
    return rows, unmatched


def _stats(diffs):
    return {"n": len(diffs), "mean_diff": M.mean_std(diffs)[0] if diffs else None,
            "better": sum(1 for d in diffs if d > EPS), "equal": sum(1 for d in diffs if abs(d) <= EPS),
            "worse": sum(1 for d in diffs if d < -EPS), "worst": min(diffs) if diffs else None}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(rows: list, groups=GROUPS) -> dict:
    comp = [r for r in rows if r["eff_diff"] is not None]
    out = {"n_pairs": len(rows), "n_comparable": len(comp), "n_not_comparable": len(rows) - len(comp),
           "overall": _stats([r["eff_diff"] for r in comp]), "groups": {}}
    for g in groups:
        by = defaultdict(list)
        for r in comp:
            by[r.get(g, "")].append(r["eff_diff"])
        out["groups"][g] = {str(k): _stats(v) for k, v in sorted(by.items(), key=lambda kv: str(kv[0]))}
    for key, label in (("cause", "causes"), ("provenance_other", "provenance_other")):
        by = defaultdict(list)
        for r in comp:
            by[r[key]].append(r["eff_diff"])
        out[label] = {k: _stats(v) for k, v in by.items()}
    ops = defaultdict(list)
    for r in comp:
        for op in set(r["only_other"]):
            ops[("other has", op)].append(r["eff_diff"])
        for op in set(r["only_base"]):
            ops[("other lacks", op)].append(r["eff_diff"])
    out["operations"] = sorted(({"direction": d, "op": op, **_stats(v)} for (d, op), v in ops.items()),
                               key=lambda x: (x["mean_diff"], -x["n"]))
    worse = [r for r in comp if r["verdict"] == "worse"]
    out["components"] = {
        "all": {k: _mean([r[k + "_rel"] for r in comp]) for k in ("time_min", "cost_inr", "energy_kwh")},
        "worse": {k: _mean([r[k + "_rel"] for r in worse]) for k in ("time_min", "cost_inr", "energy_kwh")},
        "changeovers": {"base": _mean([r["changeovers_base"] for r in comp]), "other": _mean([r["changeovers_other"] for r in comp])},
        "n_ops": {"base": _mean([r["n_ops_base"] for r in comp]), "other": _mean([r["n_ops_other"] for r in comp])},
    }
    ph = {}
    for name in ("phys_energy_kwh", "phys_wear_inr", "phys_peak_spindle_kw", "feas_infeasible", "feas_risk"):
        both = [(_num(r[name + "_base"]), _num(r[name + "_other"])) for r in rows]
        both = [(a, b) for a, b in both if a is not None and b is not None]
        ph[name] = {"n": len(both), "base": _mean([a for a, _ in both]), "other": _mean([b for _, b in both])}
    op = [(r["over_power_base"], r["over_power_other"]) for r in rows if r["over_power_base"] != "" and r["over_power_other"] != ""]
    ph["over_power"] = {"n": len(op), "both": sum(1 for a, b in op if a and b), "only_base": sum(1 for a, b in op if a and not b),
                        "only_other": sum(1 for a, b in op if b and not a), "neither": sum(1 for a, b in op if not a and not b)}
    out["physics"] = ph
    return out


def _f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _pct(x):
    return "n/a" if x is None else f"{100 * x:+.1f}%"


def _stat_line(label, s, w):
    return (f"  {label.ljust(w)}  n={s['n']:<3} mean {_f(s['mean_diff']):>6}  better {s['better']:<3} equal {s['equal']:<3} "
            f"worse {s['worse']:<3} worst {_f(s['worst']):>6}")


def _mark(route, only, sign):
    left = Counter(only)
    out = []
    for s in route.split(" > ") if route else []:
        if left[s] > 0:
            out.append(f"{sign}{s}")
            left[s] -= 1
        else:
            out.append(s)
    return " > ".join(out)


def _pair_block(r, base_label, other_label):
    head = (f"{r['part_id']}" + (f" (repeat {r['repeat']})" if r["repeat"] != 1 else "") + f"  {r['archetype']}; {r['material']}; "
            f"{r['features'].replace(';', ', ')}; tol {r['tolerance']}; batch {r['batch_size']}")
    if r["eff_diff"] is None:
        verdict = "not comparable: " + ", ".join(f"{lab} final is not strict-valid" for lab, v in
                                                 ((base_label, r["valid_base"]), (other_label, r["valid_other"])) if not v)
    else:
        verdict = f"Efficiency {_f(r['eff_base'])} -> {_f(r['eff_other'])} ({r['eff_diff']:+.2f}), {r['cause']}"
    lines = [head, "  " + verdict,
             f"  {base_label} [{r['provenance_base']}, {r['changeovers_base']} changeovers]: {_mark(r['route_base'], r['only_base'], '-')}",
             f"  {other_label} [{r['provenance_other']}, {r['changeovers_other']} changeovers]: {_mark(r['route_other'], r['only_other'], '+')}",
             f"  time {_f(r['time_min_base'], 1)} -> {_f(r['time_min_other'], 1)} min ({_pct(r['time_min_rel'])}), cost {_f(r['cost_inr_base'], 0)} -> "
             f"{_f(r['cost_inr_other'], 0)} INR ({_pct(r['cost_inr_rel'])}), energy {_f(r['energy_kwh_base'], 2)} -> "
             f"{_f(r['energy_kwh_other'], 2)} kWh ({_pct(r['energy_kwh_rel'])})"]
    extra = []
    if r["over_power_base"] != r["over_power_other"] and r["over_power_base"] != "" and r["over_power_other"] != "":
        extra.append(f"over machine power {bool(r['over_power_base'])} -> {bool(r['over_power_other'])}")
    for name, label in (("feas_infeasible", "hard infeasibilities"), ("feas_risk", "route risks")):
        a, b = _num(r[name + "_base"]), _num(r[name + "_other"])
        if a is not None and b is not None and a != b:
            extra.append(f"{label} {a:g} -> {b:g}")
    a, b = _num(r["phys_wear_inr_base"]), _num(r["phys_wear_inr_other"])
    if a is not None and b is not None and abs(b - a) > 0.5:
        extra.append(f"tool wear {a:.0f} -> {b:.0f} INR")
    if extra:
        lines.append("  " + "; ".join(extra))
    return "\n".join(lines)


def render(rows: list, summ: dict, base_label: str, other_label: str, top: int = 10, only: str = "worse", unmatched: dict = None) -> str:
    """Text report. `only` picks which pairs are listed in the last part: worse, better or all (best/worst first)."""
    o = summ["overall"]
    out = [f"Per-part difference, {other_label} minus {base_label} (the first run is the reference; a negative difference means {other_label} is worse)",
           f"{summ['n_pairs']} pairs, {summ['n_comparable']} comparable (both finals strict-valid). Efficiency difference: mean {_f(o['mean_diff'])} pts, "
           f"better {o['better']}, equal {o['equal']}, worse {o['worse']}, worst {_f(o['worst'])}"]
    if summ["n_not_comparable"]:
        out.append(f"{summ['n_not_comparable']} pair(s) not comparable because a final is not strict-valid (listed below with their routes)")
    if unmatched and any(unmatched.values()):
        out.append("Not paired: " + ", ".join(f"{len(v)} {k.replace('_', ' ')}" for k, v in unmatched.items() if v)
                   + ("  (mismatch: same part id but other features or material, so another dataset)" if unmatched.get("mismatch") else ""))
    out.append("")
    for g, table in summ["groups"].items():
        if table:
            w = max(len(k) for k in table)
            out += [f"By {g}:"] + [_stat_line(k, s, w) for k, s in table.items()] + [""]
    out += ["By how the routes differ (cause tag: description, not causation):"]
    w = max(len(k) for k in CAUSES)
    out += [_stat_line(k, summ["causes"][k], w) for k in CAUSES if k in summ["causes"]] + [""]
    pv = summ["provenance_other"]
    if pv:
        out += [f"By provenance of the {other_label} final:"] + [_stat_line(k, s, max(len(x) for x in pv)) for k, s in sorted(pv.items())] + [""]
    ops = summ["operations"]
    if ops:
        out.append("Operations that differ (mean Efficiency difference of the comparable pairs where it differs; worst first):")
        wop = max(len(x["op"]) for x in ops[:12])
        for x in ops[:12]:
            out.append(f"  {x['direction']:<11} {x['op']:<{wop}}  n={x['n']:<3} mean {_f(x['mean_diff']):>6}  worse {x['worse']}/{x['n']}")
        out.append("")
    c = summ["components"]
    out += ["What moves Efficiency (mean relative difference of the numbers behind it, B vs A):",
            f"  all comparable pairs   time {_pct(c['all']['time_min'])}  cost {_pct(c['all']['cost_inr'])}  energy {_pct(c['all']['energy_kwh'])}",
            f"  pairs where B is worse time {_pct(c['worse']['time_min'])}  cost {_pct(c['worse']['cost_inr'])}  energy {_pct(c['worse']['energy_kwh'])}",
            f"  operations / route {_f(c['n_ops']['base'], 1)} vs {_f(c['n_ops']['other'], 1)};  machine changeovers / route "
            f"{_f(c['changeovers']['base'], 2)} vs {_f(c['changeovers']['other'], 2)}", ""]
    ph = summ["physics"]
    out.append("Physics and feasibility of the finals (means over the pairs where both runs have them; A vs B):")
    for name, label, nd in (("phys_energy_kwh", "cutting energy kWh", 3), ("phys_wear_inr", "tool wear INR", 1), ("phys_peak_spindle_kw", "peak spindle kW", 2),
                            ("feas_infeasible", "hard infeasibilities", 2), ("feas_risk", "route risks", 2)):
        out.append(f"  {label:<22} {_f(ph[name]['base'], nd)} vs {_f(ph[name]['other'], nd)}   (n={ph[name]['n']})")
    op = ph["over_power"]
    out += [f"  over machine power     both {op['both']}, only {base_label} {op['only_base']}, only {other_label} {op['only_other']}, neither {op['neither']}"
            f"   (n={op['n']})", ""]
    if only == "worse":
        sel = sorted((r for r in rows if r["verdict"] == "worse"), key=lambda r: r["eff_diff"])
    elif only == "better":
        sel = sorted((r for r in rows if r["verdict"] == "better"), key=lambda r: -r["eff_diff"])
    else:
        sel = sorted((r for r in rows if r["eff_diff"] is not None), key=lambda r: r["eff_diff"])
    bad = [r for r in rows if r["eff_diff"] is None]
    out.append(f"{only.capitalize()} pairs ({len(sel)}), first {min(top, len(sel))} (-op: only in {base_label}, +op: only in {other_label}):")
    for r in sel[:top]:
        out.append(_pair_block(r, base_label, other_label))
        out.append("")
    if bad:
        out.append(f"Not comparable ({len(bad)}):")
        for r in bad[:top]:
            out.append(_pair_block(r, base_label, other_label))
            out.append("")
    out.append("Description, not causation. With few pairs these tables are anecdotes: look at the routes before drawing a conclusion.")
    return "\n".join(out)


def write_csv(path: str, rows: list):
    if not rows:
        raise ValueError("no pairs to write")
    flat = [{k: (";".join(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
