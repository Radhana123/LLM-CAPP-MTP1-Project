"""Where the project stands and the command to run next (MTP-1 Step 41).

    python scripts/where.py

About fifteen lines, offline, a second: it reads results/, configs/, FREEZE.json and .env, runs nothing, calls no API, writes nothing. The first
line that says TODO is where you are, the commands under NEXT are the ones to type, and a screenshot of the output is what to send instead of
asking "what next?". It follows RUNBOOK.md: key and smoke test, G1 baselines, G2 grid and freeze, G3/G4, release. Details: capp/where.py.
Exit code: always 0 (2 for a wrong argument).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import diagnostics, where   # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Where the project stands, and the command to run next (offline, reads only).")
    ap.add_argument("--tag", default=diagnostics.DEFAULT_TAG, help="run tag of the G1 baselines (default dev)")
    ap.add_argument("--dataset", default=diagnostics.DEFAULT_DATASET, help="dataset of the G1 baselines (default dev_v2)")
    ap.add_argument("--grid", default=where.GRID, help="the development grid (default configs/ablation_template.json)")
    ap.add_argument("--final", default=where.FINAL, help="the final grid (default configs/final.json)")
    ap.add_argument("--root", default=ROOT, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    print(where.report(a.root, tag=a.tag, dataset=a.dataset, grid=a.grid, final=a.final))
    return 0


if __name__ == "__main__":
    sys.exit(main())
