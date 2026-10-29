"""The G2 ablation grid on the development set in one command (MTP-1 Step 40).

    python scripts/g2.py configs/ablation_template.json --dry-run     # the runs still to make, the plan per column, nothing is run
    python scripts/g2.py configs/ablation_template.json               # run them column by column (asks before each column)
    python scripts/g2.py configs/ablation_template.json --only qwen --max-calls 300     # one column; each run stops after 300 live calls
    python scripts/g2.py configs/ablation_template.json --yes         # do not ask

The same runs as `python scripts/make_tables.py commands <config>`, started for you, with the care of scripts/g1.py: what is present and usable
is not run again, a run that is cut off or made with old model settings is run again with the same --tag, the plan and the question come
before the API calls, the first run that stops stops everything, and the same command later continues from the cache. A sealed dataset is
refused. Exit code: 0 nothing left to run in the selection, 1 something is left or the sequence stopped, 2 a wrong argument or config.
Details: capp/g2.py.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import g2, tables   # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the runs of the G2 ablation grid that are still to make.")
    ap.add_argument("config", help="the grid definition (JSON), for example configs/ablation_template.json")
    ap.add_argument("--only", nargs="+", metavar="COLUMN", help="only these columns (the model ids of the config; 'shared' = the rows that choose their own models)")
    ap.add_argument("--yes", action="store_true", help="do not ask before the runs of a column that call the API")
    ap.add_argument("--dry-run", action="store_true", help="list the runs and the plan of each column; run nothing")
    ap.add_argument("--max-calls", type=int, default=None, metavar="N", help="stop each API run after N live calls (the same command later continues)")
    ap.add_argument("--root", default=ROOT, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    try:
        cfg = tables.load_config(a.config)
    except tables.TableError as exc:
        print(f"STOPPED: config error: {exc}", file=sys.stderr)
        return 2
    try:
        return g2.run_g2(cfg, a.root, only=a.only, yes=a.yes, dry_run=a.dry_run, max_calls=a.max_calls)
    except g2.G2Error as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
