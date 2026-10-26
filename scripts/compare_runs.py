"""Compare finished runs side by side (key metrics table, paired quality, R0 vs R1 on the same raw LLM routes).

    python scripts/compare_runs.py results/dev_rb results/dev_b1 results/dev_b0 results/dev_b0_r1

The first folder is the reference for the paired comparison. All folders must come from the same dataset and parts.
No API calls.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import compare   # noqa: E402


def main():
    dirs = sys.argv[1:]
    if len(dirs) < 2:
        print(__doc__)
        sys.exit(1)
    for d in dirs:
        if not os.path.exists(os.path.join(d, "records.jsonl")):
            print(f"no records.jsonl in {d}", file=sys.stderr)
            sys.exit(1)
    print(compare.compare(dirs))


if __name__ == "__main__":
    main()
