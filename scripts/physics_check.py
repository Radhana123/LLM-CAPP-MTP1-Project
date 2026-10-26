"""What do the physics agents say about the dataset, and do they measure anything the v1 agents do not? Offline, no API calls.

For every part it takes the Route Builder's valid route variants (the same ones objective_check.py uses), runs the physics
analysis of capp/physics.py (specific cutting energy kc, Taylor tool-wear cost, Ra risk, machine power) and compares it with the
v1 Energy / Cost / Time agents:
  * coverage: every operation must be modelled (or declared non-cutting);
  * kc, cutting energy and tool wear by material (sanity: titanium and steel wear tools, aluminium and plastic do not);
  * how far the v1 Energy agent is from the energy of the cut, and whether it ranks route variants like physics does
    (Spearman, pooled and within a part: a within-part correlation near 1 means the new agent adds no information to selection);
  * how often a route's cutting parameters need more power than the machine has, and which operations do it;
  * Ra risk against the dataset's required Ra, by required-Ra class;
  * sensitivity to the cutting speed (Taylor): vc_factor 0.8 / 1.0 / 1.2.

    python scripts/physics_check.py --dataset dev_v2
    python scripts/physics_check.py --dataset dev_v2 --json results/physics_check.json
"""
import argparse
import collections
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import dataset, legacy, metrics, physics, systems      # noqa: E402,F401
import route_builder as rb                                       # noqa: E402


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def analyse(parts, max_routes=15):
    first, variants, ops_seen = [], [], collections.Counter()
    for part in parts:
        p = systems.part_inputs(part)
        routes, _ = systems._rb_routes(p["features"], p["pref"], max_routes)
        if not routes:
            continue
        per_route = []
        for i, r in enumerate(routes):
            a = physics.analyse_part_route(r["steps"], part)
            v1 = systems.eval_score(r["steps"], p)
            per_route.append((a, v1))
            for row in a["ops"]:
                ops_seen[(row["op"], row["process"])] += 1
        first.append((part, per_route[0][0], per_route[0][1]))
        variants.append((part, per_route))
    return first, variants, ops_seen


def _within_part(variants, phys_key, v1_key):
    rhos, constant_phys, constant_v1 = [], 0, 0
    for part, per_route in variants:
        x = [phys_key(a) for a, _ in per_route]
        y = [v1[v1_key] for _, v1 in per_route]
        constant_phys += len(set(round(v, 9) for v in x)) == 1
        constant_v1 += len(set(y)) == 1
        if len(x) >= 3 and len(set(x)) > 1 and len(set(y)) > 1:
            rhos.append(metrics.spearman(x, y))
    return {"parts_compared": len(rhos), "rho_mean": _mean(rhos), "parts_physics_constant": constant_phys,
            "parts_v1_constant": constant_v1, "parts": len(variants)}


def report(parts):
    first, variants, ops_seen = analyse(parts)
    out = {"parts": len(first)}
    out["coverage"] = {"operations_seen": sum(ops_seen.values()),
                       "by_process": dict(collections.Counter(proc for (_, proc), n in ops_seen.items() for _ in range(n))),
                       "unknown": sorted({op for (op, proc) in ops_seen if proc == "unknown"})}
    by_mat = collections.defaultdict(list)
    for part, a, v1 in first:
        by_mat[part["material"]].append((a["totals"], v1))
    out["by_material"] = {m: {"parts": len(v), "kc_mean_n_mm2": _mean([t["kc_mean_n_mm2"] for t, _ in v]),
                              "cutting_energy_kwh": _mean([t["cutting_energy_kwh"] for t, _ in v]),
                              "v1_energy_kwh": _mean([s["energy_kwh"] for _, s in v]),
                              "wear_cost_inr": _mean([t["wear_cost_inr"] for t, _ in v]),
                              "v1_cost_inr": _mean([s["cost_inr"] for _, s in v]),
                              "over_power_share": _mean([bool(t["ops_over_power"]) for t, _ in v])}
                          for m, v in sorted(by_mat.items())}
    all_t = [(a["totals"], v1) for _, a, v1 in first]
    ratios = [v1["energy_kwh"] / t["cutting_energy_kwh"] for t, v1 in all_t if t["cutting_energy_kwh"] > 0]
    out["v1_energy_over_physics"] = {"mean_of_ratios": _mean(ratios), "median": statistics.median(ratios),
                                     "ratio_of_means": _mean([v1["energy_kwh"] for _, v1 in all_t]) / _mean([t["cutting_energy_kwh"] for t, _ in all_t])}
    pooled_e = [(a["totals"]["cutting_energy_kwh"], v1["energy_kwh"]) for _, pr in variants for a, v1 in pr]
    pooled_w = [(a["totals"]["wear_cost_inr"], v1["cost_inr"]) for _, pr in variants for a, v1 in pr]
    out["rank_agreement"] = {
        "pooled_energy_physics_vs_v1": metrics.spearman([x for x, _ in pooled_e], [y for _, y in pooled_e]),
        "pooled_wear_vs_v1_cost": metrics.spearman([x for x, _ in pooled_w], [y for _, y in pooled_w]),
        "within_part_energy": _within_part(variants, lambda a: a["totals"]["cutting_energy_kwh"], "energy_kwh"),
        "within_part_wear_vs_cost": _within_part(variants, lambda a: a["totals"]["wear_cost_inr"], "cost_inr"),
        "within_part_cut_time_vs_time": _within_part(variants, lambda a: a["totals"]["cutting_time_min"], "time_min"),
    }
    over = collections.Counter()
    d_over, d_ok = [], []
    for part, a, _ in first:
        t = a["totals"]
        (d_over if t["ops_over_power"] else d_ok).append(part["dims"]["diameter"])
        for op in t["ops_over_power"]:
            over[op] += 1
    out["power"] = {"over_power_share": len(d_over) / max(1, len(first)), "ops_over_power": dict(over.most_common()),
                    "diameter_median_over": _mean([statistics.median(d_over)]) if d_over else None,
                    "diameter_median_ok": _mean([statistics.median(d_ok)]) if d_ok else None,
                    "machine_power_kw": physics.SPINDLE_POWER_KW}
    ra = collections.defaultdict(collections.Counter)
    best_op = collections.Counter()
    for part, a, _ in first:
        ra[part.get("required_ra_um")][a["totals"]["ra"]["risk"]] += 1
        best_op[a["totals"]["ra"]["best_op"]] += 1
    out["ra"] = {"by_required_um": {str(k): dict(v) for k, v in sorted(ra.items(), key=lambda kv: (kv[0] is None, kv[0]))},
                 "finest_process_op": dict(best_op.most_common())}
    sens = {}
    for vf in (0.8, 1.0, 1.2):
        ts = [physics.analyse_part_route(a_steps, part, vc_factor=vf)["totals"] for part, a_steps in
              ((part, [r["op"] for r in a["ops"]]) for part, a, _ in first)]
        sens[str(vf)] = {"wear_cost_inr": _mean([t["wear_cost_inr"] for t in ts]), "edges_per_part": _mean([t["edges_per_part"] for t in ts]),
                         "over_power_share": _mean([bool(t["ops_over_power"]) for t in ts]),
                         "cutting_energy_kwh": _mean([t["cutting_energy_kwh"] for t in ts])}
    out["speed_sensitivity"] = sens
    return out


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def format_report(r):
    L = [f"=== physics check: {r['parts']} parts (Route Builder routes) ==="]
    c = r["coverage"]
    L.append(f"coverage        : {c['operations_seen']} operation instances, by process {c['by_process']}; unknown operations: {c['unknown'] or 'none'}")
    L.append("by material     :  parts  kc[N/mm2]  E_cut[kWh]  E_v1[kWh]  wear[INR]  cost_v1[INR]  over-power")
    for m, v in r["by_material"].items():
        L.append(f"  {m:<10}      {v['parts']:>5}  {_f(v['kc_mean_n_mm2'], 0):>9}  {_f(v['cutting_energy_kwh']):>10}  {_f(v['v1_energy_kwh'], 2):>9}  "
                 f"{_f(v['wear_cost_inr'], 1):>9}  {_f(v['v1_cost_inr'], 0):>12}  {_f(v['over_power_share'], 2):>10}")
    e = r["v1_energy_over_physics"]
    L.append(f"v1 Energy agent / energy of the cut: x{_f(e['ratio_of_means'], 1)} (ratio of means), median x{_f(e['median'], 1)}, "
             f"mean of per-part ratios x{_f(e['mean_of_ratios'], 1)} (small parts dominate this one); the v1 number does not depend on part size")
    ra = r["rank_agreement"]
    we, ww, wt = ra["within_part_energy"], ra["within_part_wear_vs_cost"], ra["within_part_cut_time_vs_time"]
    L.append(f"rank agreement  : pooled Spearman energy(physics, v1) {_f(ra['pooled_energy_physics_vs_v1'], 2)}, wear vs v1 cost "
             f"{_f(ra['pooled_wear_vs_v1_cost'], 2)}")
    for name, w in (("energy", we), ("wear vs cost", ww), ("cut time vs time", wt)):
        L.append(f"  within a part, {name}: mean rho {_f(w['rho_mean'], 2)} over {w['parts_compared']} parts; variants identical for "
                 f"physics in {w['parts_physics_constant']}/{w['parts']} parts, for v1 in {w['parts_v1_constant']}/{w['parts']}")
    p = r["power"]
    L.append(f"machine power   : {_f(p['over_power_share'], 2)} of parts have an operation above {p['machine_power_kw']:g} kW at the spindle; "
             f"operations {p['ops_over_power']}; median part diameter {_f(p['diameter_median_over'], 0)} mm (over) vs {_f(p['diameter_median_ok'], 0)} mm (ok)")
    L.append("Ra risk         : by required Ra (um): " + "; ".join(f"{k}: {v}" for k, v in r["ra"]["by_required_um"].items()))
    L.append(f"                  finest process on the route: {r['ra']['finest_process_op']}")
    L.append("speed what-if   : " + "; ".join(f"Vc x{k}: wear {_f(v['wear_cost_inr'], 1)} INR, {_f(v['edges_per_part'], 2)} edges/part, "
                                              f"over power {_f(v['over_power_share'], 2)}, E_cut {_f(v['cutting_energy_kwh'])} kWh"
                                              for k, v in r["speed_sensitivity"].items()))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dev_v2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", metavar="FILE", help="also write the numbers as JSON")
    args = ap.parse_args()
    parts = dataset.load_dataset(args.dataset)
    if args.limit:
        parts = parts[:args.limit]
    r = report(parts)
    print(format_report(r))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
