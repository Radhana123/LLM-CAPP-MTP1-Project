"""What do the physical-feasibility rules say about the dataset, and about the Route Builder's own routes? Offline, no API calls.

For every part it takes the Route Builder's first route (the RB baseline) and its valid variants (the same ones objective_check.py uses)
and runs the 15 rules of capp/feasibility.py:
  * per rule: how many parts fire it on the RB first route, on how many of the RB variants, and for how many parts it cannot be evaluated
    (missing data, typically dims.hole_diameter, which dataset v2 does not have);
  * hard findings on RB routes are defects of the rule base itself (the Route Builder builds a route that cannot be machined): listed by part;
  * specification findings (the tolerance is beyond the operation vocabulary) by tolerance value;
  * support needed (L/D) by class;
  * the sources: how many rules rest on a web source, on the vocabulary, on the cutting-force model, on an assumption.

    python scripts/feasibility_check.py --dataset dev_v2
    python scripts/feasibility_check.py --dataset dev_v2 --json results/feasibility_check.json
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import dataset, feasibility as F, systems       # noqa: E402,F401


def analyse(parts, max_routes=15):
    first = collections.Counter()
    variants = collections.Counter()
    variant_total = 0
    not_eval = collections.Counter()
    hard_parts = collections.defaultdict(list)
    spec_by_tol = collections.defaultdict(lambda: [0, 0])
    support = collections.Counter()
    n = 0
    for part in parts:
        p = systems.part_inputs(part)
        routes, _ = systems._rb_routes(p["features"], p["pref"], max_routes)
        if not routes:
            continue
        n += 1
        rep = F.evaluate(routes[0]["steps"], part)
        for rid in rep.rule_ids:
            first[rid] += 1
        for rid in rep.not_evaluated:
            not_eval[rid] += 1
        for v in rep.infeasible:
            hard_parts[v.rule_id[5:]].append(part["part_id"])
        tol = part.get("tolerance", "")
        spec_by_tol[tol][0] += 1
        spec_by_tol[tol][1] += "TOL-SPEC" in rep.rule_ids
        support["tailstock" if "HOLD-TAILSTOCK" in rep.rule_ids else "steady rest" if "HOLD-STEADY" in rep.rule_ids
                else "beyond L/D 20" if "HOLD-SLENDER-MAX" in rep.rule_ids else "none"] += 1
        for r in routes:
            variant_total += 1
            for rid in F.evaluate(r["steps"], part).rule_ids:
                variants[rid] += 1
    rules = {}
    for r in F.RULES:
        rules[r.id] = {"kind": r.kind, "scope": r.scope, "category": r.category, "source_status": r.source_status,
                       "parts_first_route": first[r.id], "share_first_route": first[r.id] / n if n else None,
                       "variants_firing": variants[r.id], "share_variants": variants[r.id] / variant_total if variant_total else None,
                       "not_evaluable_parts": not_eval[r.id]}
    return {"parts": n, "route_builder_variants": variant_total, "rules": rules, "hard_findings_first_route": dict(hard_parts),
            "spec_by_tolerance": {k: {"parts": v[0], "beyond_vocabulary": v[1]} for k, v in sorted(spec_by_tol.items())},
            "support": dict(support), "status_counts": dict(collections.Counter(r.source_status for r in F.RULES)),
            "not_encoded": F.NOT_ENCODED}


def format_report(r):
    L = [f"=== feasibility check: {r['parts']} parts, {r['route_builder_variants']} Route Builder routes ==="]
    L.append(f"{'rules':<17} {'kind':<10} {'scope':<6} {'status':<10} {'RB first route':>15} {'RB variants':>12} {'not evaluable':>14}")
    for rid, x in r["rules"].items():
        L.append(f"  {rid:<15} {x['kind']:<10} {x['scope']:<6} {x['source_status']:<10} "
                 f"{x['parts_first_route']:>5} ({x['share_first_route']:.2f})   {x['variants_firing']:>5} ({x['share_variants']:.2f})   {x['not_evaluable_parts']:>10}")
    h = r["hard_findings_first_route"]
    L.append("hard findings on the RB first route (a defect of the rule base if any): " + (", ".join(f"{k}: {v}" for k, v in h.items()) if h else "none"))
    L.append("specification beyond the operation vocabulary, by tolerance: " + "; ".join(
        f"{k}: {v['beyond_vocabulary']}/{v['parts']}" for k, v in r["spec_by_tolerance"].items()))
    L.append("support needed (RB first route): " + ", ".join(f"{k} {v}" for k, v in r["support"].items()))
    L.append("sources         : " + ", ".join(f"{k} {v}" for k, v in sorted(r["status_counts"].items())) + "; none is confirmed against a handbook yet")
    L.append("not encoded (no source): " + "; ".join(r["not_encoded"]))
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
    r = analyse(parts)
    print(format_report(r))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
