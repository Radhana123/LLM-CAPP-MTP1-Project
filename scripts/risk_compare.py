"""Risk-aware comparison of two finished runs (Step 21): is the LLM route really worse than the Route Builder route once the physical
risks (feasibility findings) of both routes are looked at, not only the v1 Efficiency?

    python scripts/risk_compare.py results/dev_rb results/dev_n5_r1_s3
    python scripts/risk_compare.py results/dev_rb results/dev_n5_r1_s3 --physics --top 20 --csv results/risk_pairs.csv
    python scripts/risk_compare.py results/dev_rb results/dev_n5_r1_s3 --penalties 0,1,3,10

The first folder is the reference (normally rb), the second is judged against it, as in compare_runs.py and part_diff.py. Offline: it
only reads records.jsonl, no API calls. Both runs must come from the same dataset and parts. The v1 Efficiency is not changed: the
penalty table says what the comparison looks like if one flag is worth P Efficiency points, and the break-even P* says how large P must
be for the two to match. P is a value judgement to argue, not something this script estimates.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import harness, partdiff, riskview   # noqa: E402


def _label(path):
    return os.path.basename(path.rstrip("/\\")) or path


def _load(path):
    if not os.path.exists(os.path.join(path, "records.jsonl")):
        print(f"no records.jsonl in {path}", file=sys.stderr)
        sys.exit(1)
    return harness.load_records(path)


def _penalties(text):
    try:
        values = tuple(float(x) for x in text.split(",") if x.strip())
    except ValueError:
        values = ()
    if not values or any(v < 0 for v in values):
        print(f"--penalties: expected non-negative numbers separated by commas, got {text!r}", file=sys.stderr)
        sys.exit(1)
    return values


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="reference run folder (normally rb)")
    ap.add_argument("other", help="run folder judged against it")
    ap.add_argument("--physics", action="store_true", help="also count over machine power and Ra risk 'high' as flags")
    ap.add_argument("--penalties", default=",".join(f"{p:g}" for p in riskview.PENALTIES),
                    help="Efficiency points one flag is worth, comma separated (default %(default)s)")
    ap.add_argument("--top", type=int, default=8, help="pairs listed per group")
    ap.add_argument("--resamples", type=int, default=riskview.RESAMPLES)
    ap.add_argument("--csv", metavar="PATH", help="write one row per analysed pair")
    args = ap.parse_args(argv)

    penalties = _penalties(args.penalties)
    base, other = _load(args.base), _load(args.other)
    rows, unmatched = partdiff.pair_rows(base, other)
    s = riskview.analyze(rows, physics=args.physics, penalties=penalties, resamples=args.resamples)
    print(riskview.render(s, _label(args.base), _label(args.other), top=args.top))
    if any(unmatched.values()):
        print("\nNot paired: " + ", ".join(f"{len(v)} {k.replace('_', ' ')}" for k, v in unmatched.items() if v))
    if args.csv:
        try:
            riskview.write_csv(args.csv, s)
            print(f"\nwrote {args.csv}")
        except riskview.RiskViewError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
