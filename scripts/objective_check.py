"""Do Time, Cost and Energy really measure different things? Offline check, no API calls.

For every part, take the Route Builder's valid route variants, score them fairly (part dimensions, one changeover charge per
machine switch) and look at (a) the Spearman correlation between the objectives across the variants of a part, (b) how many
distinct (time, cost, energy) triples exist, (c) the size of the non-dominated front, and how many of its members are exact
objective ties (the same triple, only another operation order): for those the Efficiency score cannot choose, and v1 simply
keeps the first. If the objectives are almost perfectly correlated, NSGA-II has one objective in three costumes and a front
of size 1 or 2.

    python scripts/objective_check.py --dataset dev_v2
"""
import argparse
import csv
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import dataset, legacy, metrics, systems      # noqa: E402,F401
import nsga2                                            # noqa: E402
import route_builder as rb                              # noqa: E402


class _P:
    def __init__(self, obj):
        self.objectives = obj
        self.rank = 0


def analyse(parts, max_routes=15):
    rows = []
    for part in parts:
        p = systems.part_inputs(part)
        routes = rb.generate_valid_routes(p["features"], max_routes=max_routes, machine_preference="auto")
        if not routes:
            continue
        sc = [systems.eval_score(r["steps"], p) for r in routes]
        objs = [(s["time_min"], s["cost_inr"], s["energy_kwh"]) for s in sc]
        pop = [_P(o) for o in objs]
        front = nsga2.non_dominated_sort(pop)[0]
        front_triples = [objs[i] for i in front]
        col = lambda i: [o[i] for o in objs]          # noqa: E731
        rows.append({
            "part_id": part["part_id"], "archetype": part.get("archetype", ""), "machine_profile": part.get("machine_profile", ""),
            "routes": len(routes), "distinct_triples": len(set(objs)), "front_size": len(front),
            "front_distinct_triples": len(set(front_triples)),
            "rho_time_cost": metrics.spearman(col(0), col(1)), "rho_time_energy": metrics.spearman(col(0), col(2)),
            "rho_cost_energy": metrics.spearman(col(1), col(2)),
        })
    return rows


def summarize(rows):
    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.fmean(vals) if vals else None
    n = len(rows)
    fs = [r["front_size"] for r in rows]
    return {
        "parts": n, "routes_per_part_mean": mean("routes"), "distinct_triples_mean": mean("distinct_triples"),
        "front_size_mean": statistics.fmean(fs), "front_size_median": statistics.median(fs),
        "parts_front_size_1": sum(1 for f in fs if f == 1) / n, "parts_front_size_le_2": sum(1 for f in fs if f <= 2) / n,
        "front_distinct_triples_mean": mean("front_distinct_triples"),
        "front_members_that_are_ties_share": 1 - sum(r["front_distinct_triples"] for r in rows) / sum(r["front_size"] for r in rows),
        "parts_with_tied_front_members": sum(1 for r in rows if r["front_distinct_triples"] < r["front_size"]) / n,
        "parts_whose_front_is_one_triple": sum(1 for r in rows if r["front_distinct_triples"] == 1) / n,
        "rho_time_cost_mean": mean("rho_time_cost"), "rho_time_energy_mean": mean("rho_time_energy"),
        "rho_cost_energy_mean": mean("rho_cost_energy"),
        "parts_where_correlation_undefined": sum(1 for r in rows if r["rho_time_cost"] is None) / n,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dev_v2")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    rows = analyse(dataset.load_dataset(args.dataset))
    s = summarize(rows)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"objective_check_{args.dataset}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, f"objective_check_{args.dataset}.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    print(f"Objective check on {args.dataset}: {s['parts']} parts, Route Builder variants scored fairly")
    for k, v in s.items():
        print(f"  {k:<36} {v:.3f}" if isinstance(v, float) else f"  {k:<36} {v}")


if __name__ == "__main__":
    main()
