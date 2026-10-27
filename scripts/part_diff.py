"""Per part: where is run B worse than run A, and what is different in the routes there. No API calls.

    python scripts/part_diff.py results/dev_rb results/dev_n5_r1_s3
    python scripts/part_diff.py results/dev_rb results/dev_n5_r1_s3 --top 20 --only all --csv results/pairs.csv
    python scripts/part_diff.py results/dev_rb results/dev_n5_r1_s3 --group archetype,material,n_features

The first folder is the reference (normally rb), the second is judged against it, like compare_runs.py. Both must come from the same
dataset. Output: totals, tables by part class / by how the routes differ / by provenance, the operations that differ, which of time,
cost and energy moves Efficiency, physics and feasibility, then the worst pairs with both routes (-op = only in the first run,
+op = only in the second). --csv writes one row per pair for your own analysis. Description, not causation.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import compare, harness, partdiff   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="reference run folder (contains records.jsonl)")
    ap.add_argument("other", help="run folder to judge against the reference")
    ap.add_argument("--top", type=int, default=10, help="how many pairs to list with their routes (default 10)")
    ap.add_argument("--only", choices=("worse", "better", "all"), default="worse", help="which pairs to list (default worse)")
    ap.add_argument("--group", default=",".join(partdiff.GROUPS),
                    help="comma-separated columns for the group tables (default %(default)s; others: archetype, n_features, batch_size)")
    ap.add_argument("--csv", metavar="FILE", help="write one row per pair")
    ap.add_argument("--json", metavar="FILE", help="write the summary tables as JSON")
    args = ap.parse_args(argv)

    for d in (args.base, args.other):
        if not os.path.exists(os.path.join(d, "records.jsonl")):
            print(f"no records.jsonl in {d}", file=sys.stderr)
            return 1
    base_recs, other_recs = harness.load_records(args.base), harness.load_records(args.other)
    base_label, other_label = compare._label(args.base, base_recs), compare._label(args.other, other_recs)
    if base_label == other_label:
        base_label, other_label = os.path.basename(args.base.rstrip("/\\")), os.path.basename(args.other.rstrip("/\\"))
    rows, unmatched = partdiff.pair_rows(base_recs, other_recs)
    if not rows:
        print("no pairs: the two runs share no (part, repeat) with the same features and material", file=sys.stderr)
        return 1
    groups = tuple(g.strip() for g in args.group.split(",") if g.strip())
    unknown = [g for g in groups if g not in rows[0]]
    if unknown:
        print(f"unknown --group column(s): {', '.join(unknown)}. Columns: {', '.join(rows[0].keys())}", file=sys.stderr)
        return 1
    summ = partdiff.summarize(rows, groups)
    print(partdiff.render(rows, summ, base_label, other_label, top=args.top, only=args.only, unmatched=unmatched))
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        partdiff.write_csv(args.csv, rows)
        print(f"\nwrote {len(rows)} pairs to {args.csv}")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
