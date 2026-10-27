"""Table 4 of the report (ablation grid A0-A9) from finished runs, by one command per step (report 7.3-7.4).

    python scripts/make_tables.py status   configs/ablation_template.json    which runs exist and are usable (offline, reads only)
    python scripts/make_tables.py commands configs/ablation_template.json    the run_eval lines (PowerShell) for the runs still to make
    python scripts/make_tables.py build    configs/ablation_template.json    Table 4 + paired tests + figures into results/tables/<name>/
    python scripts/make_tables.py build    my_grid.json --out tables_final --strict --no-figures

The config (configs/ablation_template.json) says which system is which row, which LLMs are the columns and which run tag each used.
Runs are made with scripts/run_eval.py (the `commands` step prints the exact lines); this script makes no API call and never starts a
run. `build` writes table4.md / .csv / .tex, paired.md / .csv (bootstrap CIs, McNemar), consistency.md (across LLMs), per_part.csv,
issues.txt, fig_ablation.png, fig_cost.png and manifest.json (config hash and the hash of every run it read). A run counts for a row
only if its records and run.json say that it is that system, on that dataset, with that model; otherwise it is left out and listed.

Exit code: 0 (also with missing runs, which are listed); 1 for a bad config, or with --strict if anything at all is listed in issues.txt.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import tables   # noqa: E402


def _load(path):
    try:
        return tables.load_config(path)
    except tables.TableError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        sys.exit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["status", "commands", "build"])
    ap.add_argument("config", help="the grid definition (JSON)")
    ap.add_argument("--results", metavar="DIR", help="folder with the run folders (default: results_dir of the config, relative to the repo)")
    ap.add_argument("--out", metavar="DIR", help="build: output folder (default <results>/tables/<name>)")
    ap.add_argument("--no-figures", action="store_true", help="build: skip the PNG figures")
    ap.add_argument("--strict", action="store_true", help="build: exit 1 if anything is listed in issues.txt (a missing or partial run, a warning)")
    args = ap.parse_args(argv)

    cfg = _load(args.config)
    if args.command in ("status", "commands"):
        loaded = tables.load_all(cfg, args.results)
        if args.command == "status":
            print(tables.status_grid(cfg, loaded))
            for (rid, mid), run in loaded["runs"].items():
                for msg in run["errors"] + run["warnings"]:
                    print(f"  {rid} {mid}: {msg}")
        else:
            print("\n".join(tables.run_commands(cfg, loaded)))
        return 0

    results_dir = tables.resolve_results_dir(cfg, args.results)
    out_dir = args.out or os.path.join(results_dir, "tables", cfg["name"])
    res = tables.build(cfg, out_dir, results_dir=args.results, figures=not args.no_figures)
    print(tables.render_issues(res))
    for name in sorted(res["files"]):
        print(f"wrote {os.path.join(out_dir, name)}")
    if args.strict and res["issues"]:
        print(f"--strict: {len(res['issues'])} issue(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
