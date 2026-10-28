"""One file that says how the project stands: environment, freeze, datasets, runs, and the offline checks (MTP-1 Step 31).

    python scripts/diagnostics.py                      # results/diagnostics.txt: attach that one file to the chat
    python scripts/diagnostics.py --tag dev --dataset dev_v2
    python scripts/diagnostics.py --no-checks          # only the quick sections (a second); the checks take about ten seconds
    python scripts/diagnostics.py --out C:\\temp\\d.txt

What is in it (capp/diagnostics.py has the details): the problems on top, git and the API key (set or not, never its value), FREEZE.json, the
datasets, one row per run folder in results/, which G1 baseline runs are still to make (with the command), and the offline scripts (physics_check,
feasibility_check, objective_check, feasrepair_check, grid status, compare_runs, part_diff, confidence_tune, risk_compare, cascade_pareto,
freeze verify), each with its exit code and output; one whose input is not there yet is skipped with the reason.

No API call is made and no run is started. The only file written is --out (default results/diagnostics.txt). A line that looks like a secret is
removed from it. Exit code: 0 when the file is written (problems are listed in it, they are not an error here), 2 for a wrong argument or a file
that cannot be written.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import diagnostics as G     # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="One diagnostics file: environment, freeze, datasets, runs and the offline checks. No API call.")
    ap.add_argument("--tag", default=G.DEFAULT_TAG, help="run tag of the baseline runs (default dev: results/dev_rb, dev_b1, dev_b0, dev_b0_r1)")
    ap.add_argument("--dataset", default=G.DEFAULT_DATASET, help="dataset of the offline checks (default dev_v2)")
    ap.add_argument("--grid", default=G.DEFAULT_GRID, help="grid config whose status is shown (default configs/ablation_template.json)")
    ap.add_argument("--out", default=os.path.join(ROOT, *G.DEFAULT_OUT.split("/")), help="the file to write (default results/diagnostics.txt)")
    ap.add_argument("--no-checks", action="store_true", help="skip the offline scripts, only the quick sections")
    ap.add_argument("--timeout", type=int, default=G.TIMEOUT, help=f"seconds allowed per check (default {G.TIMEOUT})")
    ap.add_argument("--max-lines", type=int, default=G.MAX_LINES, help=f"lines kept per check (default {G.MAX_LINES}: the first three quarters and the last quarter)")
    a = ap.parse_args(argv)
    if a.timeout < 1 or a.max_lines < 8:
        print("STOPPED: --timeout must be at least 1 and --max-lines at least 8", file=sys.stderr)
        return 2
    try:
        rep = G.build_report(ROOT, tag=a.tag, dataset=a.dataset, grid=a.grid, checks=not a.no_checks, timeout=a.timeout, max_lines=a.max_lines,
                             progress=lambda m: print(m, file=sys.stderr, flush=True))
        G.write_text(rep["text"], a.out)
    except G.DiagnosticsError as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"STOPPED: {a.out} cannot be written ({exc})", file=sys.stderr)
        return 2
    n = len(rep["problems"])
    print(f"{n} problem(s)" + ("" if not n else ":"))
    for p in rep["problems"]:
        print(f"  - {p}")
    if rep["removed"]:
        print(f"{len(rep['removed'])} line(s) that looked like a secret were removed from the file")
    c = rep["counts"]
    if not a.no_checks:
        print("checks: " + ", ".join(f"{c[k]} {k}" for k in G.STATUS_ORDER if c[k]))
    print(f"\nWritten: {a.out}\nAttach this one file to the chat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
