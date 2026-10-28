"""The G1 baselines (rb, b1, b0) in one command, then the diagnostics file (MTP-1 Step 34).

    python scripts/g1.py                  # run what is missing, then write results/diagnostics.txt (attach that file)
    python scripts/g1.py --dry-run        # what it would do and the plan of each API run; nothing is run
    python scripts/g1.py --yes            # do not ask before a run that calls the API
    python scripts/g1.py --max-calls 150  # also stop each API run after 150 live calls (run again to go on)

The commands are the ones of RUNBOOK.md, G1. Runs that are complete are not repeated; a run with API errors is run again with the same
command (the cache makes that cost only the missing calls). Details: capp/g1.py. Exit code: 0 when all three baselines are there at the end,
1 when the sequence stopped before that, 2 for a wrong argument.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import diagnostics, g1     # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the G1 baselines (rb, b1, b0) and write the diagnostics file.")
    ap.add_argument("--tag", default=diagnostics.DEFAULT_TAG, help="run tag (default dev: results/dev_rb, dev_b1, dev_b0)")
    ap.add_argument("--dataset", default=diagnostics.DEFAULT_DATASET, help="dataset (default dev_v2; a sealed set is not for G1)")
    ap.add_argument("--yes", action="store_true", help="do not ask before a run that calls the API")
    ap.add_argument("--dry-run", action="store_true", help="show what would be run, with the plan of each API run; run nothing")
    ap.add_argument("--max-calls", type=int, default=None, metavar="N", help="stop each API run after N live calls (the same command later continues)")
    ap.add_argument("--no-diagnostics", action="store_true", help="do not write results/diagnostics.txt at the end")
    ap.add_argument("--root", default=ROOT, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    from capp import dataset
    if dataset.is_sealed(a.dataset):
        print(f"STOPPED: {a.dataset} is sealed (final runs only); the G1 baselines are made on dev_v2.", file=sys.stderr)
        return 2
    try:
        return g1.run_g1(a.root, tag=a.tag, dataset=a.dataset, yes=a.yes, dry_run=a.dry_run, max_calls=a.max_calls, diag=not a.no_diagnostics)
    except g1.G1Error as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
