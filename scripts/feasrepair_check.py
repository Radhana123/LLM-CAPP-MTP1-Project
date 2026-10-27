"""What does the feasibility-aware R1 (capp/feasrepair.py, the +fz1 / +fz2 options) change? Offline, no API calls, no LLM.

Two experiments on the dataset, both with deterministic stand-ins for LLM errors, so they test the repair, not a model:

  A. The Route Builder's own routes (first route and valid variants). Some carry a HARD feasibility finding (the rule base lets an
     Internal Grooving come before any Drilling, BASELINE.md Step 11). How many does fz1 fix? How many risks does fz2 act on?
  B. Damaged routes: one of the part's valid Route Builder routes (chosen at random, so that every alternative operation chain of a
     feature can occur, Internal Grooving included), with a seeded random shuffle and, half of the time, one operation dropped:
     the kind of route an LLM produces when it gets the order wrong. Plain R1 makes it strict-valid; how many of the repaired routes
     are still physically infeasible, and what does the feasibility-aware repair cost in extra edits (operations inserted, pairs of
     operations whose order had to be reversed)?

It reports honestly where the repair stops: findings that no route edit can fix (machine capability) or that are only flagged
(supports, feed, finishing outside the vocabulary).

    python scripts/feasrepair_check.py --dataset dev_v2
    python scripts/feasrepair_check.py --dataset dev_v2 --seed 5 --damage 3 --json results/feasrepair_check.json
"""
import argparse
import collections
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import dataset, feasibility as F, feasrepair as FR, repair as R, systems, verify as V    # noqa: E402


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _rate(a, b):
    return a / b if b else None


def experiment_a(parts, max_routes=15):
    n = hard = fixed1 = failed1 = 0
    risk_routes = risk_fixed = 0
    by_rule = collections.Counter()
    inserted = collections.Counter()
    for part in parts:
        p = systems.part_inputs(part)
        routes, _ = systems._rb_routes(p["features"], p["pref"], max_routes)
        for r in routes:
            steps = V.clean_steps(r["steps"])
            n += 1
            found = F.evaluate(steps, part).violations
            h = FR.repairable_hard(found)
            if h:
                hard += 1
                for rid in {v.rule_id[5:] for v in h}:
                    by_rule[rid] += 1
                rep = FR.r1_fz(steps, part["features"], part)
                if rep.ok and not FR.repairable_hard(F.evaluate(rep.steps, part).violations):
                    fixed1 += 1
                    for op in rep.fz_inserted:
                        inserted[op] += 1
                else:
                    failed1 += 1
            if FR.insertable_risks(found):
                risk_routes += 1
                rep2 = FR.r1_fz(steps, part["features"], part, risks=True)
                if rep2.ok and not FR.insertable_risks(F.evaluate(rep2.steps, part).violations):
                    risk_fixed += 1
    return {"routes": n, "with_fixable_hard": hard, "hard_by_rule": dict(by_rule), "fz1_fixed": fixed1, "fz1_failed": failed1,
            "fz1_operations_inserted": dict(inserted), "with_insertable_risk": risk_routes, "fz2_fixed": risk_fixed}


def damage(rng, base):
    bad = list(base)
    rng.shuffle(bad)
    if rng.random() < 0.5 and len(bad) > 3:
        bad.pop(rng.randrange(len(bad)))
    return bad


def experiment_b(parts, seed, per_part):
    rng = random.Random(seed)
    rows = []
    t0 = time.time()
    for part in parts:
        p = systems.part_inputs(part)
        routes, _ = systems._rb_routes(p["features"], p["pref"], 15)
        if not routes:
            continue
        for _ in range(per_part):
            bad = damage(rng, V.clean_steps(rng.choice(routes)["steps"]))
            plain = R.r1(bad, part["features"])
            row = {"part": part["part_id"], "plain_ok": plain.ok}
            if plain.ok:
                f = F.evaluate(plain.steps, part).violations
                row.update(plain_hard=len(FR.repairable_hard(f)), plain_risk=len(FR.insertable_risks(f)),
                           plain_ins=len(plain.inserted), plain_disc=plain.discordant_pairs)
            for name, risks in (("fz1", False), ("fz2", True)):
                rep = FR.r1_fz(bad, part["features"], part, risks=risks)
                row[name + "_ok"] = rep.ok
                if rep.ok:
                    f = F.evaluate(rep.steps, part).violations
                    row.update({name + "_hard": len(FR.repairable_hard(f)), name + "_risk": len(FR.insertable_risks(f)),
                                name + "_ins": len(rep.inserted), name + "_disc": rep.discordant_pairs,
                                name + "_unrepairable": len(rep.unrepairable)})
            rows.append(row)
    secs = time.time() - t0
    out = {"damaged_routes": len(rows), "seconds": round(secs, 1)}
    for name in ("plain", "fz1", "fz2"):
        ok = [r for r in rows if r.get(name + "_ok")]
        out[name] = {"repaired": len(ok), "share_repaired": _rate(len(ok), len(rows)),
                     "still_hard_infeasible": _rate(sum(1 for r in ok if r[name + "_hard"]), len(ok)),
                     "still_insertable_risk": _rate(sum(1 for r in ok if r[name + "_risk"]), len(ok)),
                     "operations_inserted_mean": _mean(r[name + "_ins"] for r in ok),
                     "discordant_pairs_mean": _mean(r[name + "_disc"] for r in ok)}
    return out


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dev_v2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--damage", type=int, default=3, help="damaged routes per part (experiment B)")
    ap.add_argument("--json", metavar="FILE", help="also write the numbers as JSON")
    args = ap.parse_args()
    parts = dataset.load_dataset(args.dataset)[:args.limit]

    a = experiment_a(parts)
    print(f"A. Route Builder routes of {len(parts)} parts: {a['routes']} routes (first routes and valid variants)")
    print(f"   with a hard finding a route edit can fix : {a['with_fixable_hard']} ({fmt(_rate(a['with_fixable_hard'], a['routes']))}), by rule {a['hard_by_rule']}")
    print(f"   fixed by +fz1                             : {a['fz1_fixed']}   not fixed {a['fz1_failed']}   operations inserted {a['fz1_operations_inserted']}")
    print(f"   with a risk whose action is an insertion  : {a['with_insertable_risk']} ({fmt(_rate(a['with_insertable_risk'], a['routes']))});"
          f" fixed by +fz2 {a['fz2_fixed']}")
    b = experiment_b(parts, args.seed, args.damage)
    print(f"\nB. {b['damaged_routes']} damaged routes (seed {args.seed}, {args.damage} per part), {b['seconds']} s")
    print(f"   {'':6} {'repaired':>9} {'still hard-infeasible':>22} {'still fixable risk':>19} {'ops inserted':>13} {'discordant pairs':>17}")
    for name, label in (("plain", "R1"), ("fz1", "R1+fz1"), ("fz2", "R1+fz2")):
        r = b[name]
        print(f"   {label:6} {fmt(r['share_repaired']):>9} {fmt(r['still_hard_infeasible']):>22} {fmt(r['still_insertable_risk']):>19} "
              f"{fmt(r['operations_inserted_mean'], 2):>13} {fmt(r['discordant_pairs_mean'], 2):>17}")
    print("\n   'still ...' is the share of the REPAIRED routes that keep a finding. Machine-capability findings are not counted (no edit fixes them);\n"
          "   supports, feed and finishing findings are only flagged. The damage is synthetic: this tests the repair, not an LLM.")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"dataset": args.dataset, "n_parts": len(parts), "A": a, "B": b}, f, indent=2)


if __name__ == "__main__":
    main()
