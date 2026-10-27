"""Where should the human-review flag sit? Threshold sweep for the plan confidence on finished runs (report 6.4, WP5). No API call.

    python scripts/confidence_tune.py results/dev_n5_r1_s3                       # label: efficiency gap above 3 points (default)
    python scripts/confidence_tune.py results/dev_n5_r1_s3 --label lcs --bad-lcs 0.8
    python scripts/confidence_tune.py results/dev_n5_r1_s3 --budget 0.15 --recall 0.9 --plot results/conf.png --csv results/conf.csv
    python scripts/confidence_tune.py results/test_n5_r1_s3 --evaluate-at 0.93    # a sealed split: no tuning, only judge a fixed threshold

What it says, per run folder (each system is judged on its own, runs are not pooled):
  * how often a plan is "bad" under the label you chose (--label gap | lcs | not_raw) and how well each score ranks the bad plans (AUROC),
  * every threshold: share of plans flagged, share of bad plans caught (recall), share of flags that were bad (precision),
  * the threshold with the best recall within a review budget (--budget, share of plans) and the one that reaches a recall target
    (--recall) with the fewest flags,
  * 95% bootstrap intervals over parts. Tune on the development set only; test_v2 / bench_v1 runs are refused unless --evaluate-at.

The labels are proxies (on dev_v2 the M4 reference is the Route Builder, not an expert), so read the numbers as a guide to a starting
threshold, then confirm it once on the sealed split. Use the threshold with:
    python scripts/run_eval.py --system <system> --tag <tag> --confidence-flag T
    python scripts/run_eval.py --summarize-only <folder> --confidence-flag T
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import calibration as CAL   # noqa: E402
from capp import dataset, harness     # noqa: E402


def _dataset_of(folder):
    path = os.path.join(folder, "run.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("dataset")
    except (OSError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="result folders (each with records.jsonl)")
    ap.add_argument("--label", choices=CAL.LABELS, default="gap", help="what a bad plan is (default gap)")
    ap.add_argument("--signal", choices=CAL.SIGNALS, default="confidence",
                    help="the score to threshold (default confidence, the one the pipeline flags on; the others are diagnostics)")
    ap.add_argument("--bad-gap", type=float, default=CAL.DEFAULT_BAD_GAP, help="label gap: efficiency points below the best Route Builder variant (default 3)")
    ap.add_argument("--bad-lcs", type=float, default=CAL.DEFAULT_BAD_LCS, help="label lcs: M4 LCS ratio against the reference below this (default 0.8)")
    ap.add_argument("--budget", type=float, default=CAL.DEFAULT_BUDGET, help="largest share of plans you accept to flag (default 0.2)")
    ap.add_argument("--recall", type=float, default=CAL.DEFAULT_RECALL, help="share of bad plans the flag must catch (default 0.8)")
    ap.add_argument("--current", type=float, help="threshold in use now (default systems.CONFIDENCE_FLAG)")
    ap.add_argument("--evaluate-at", type=float, metavar="T", help="also judge this fixed threshold; the only mode allowed on sealed splits")
    ap.add_argument("--resamples", type=int, default=CAL.RESAMPLES, help="bootstrap resamples (default 1000)")
    ap.add_argument("--csv", help="write the threshold table to this CSV")
    ap.add_argument("--json", help="write the whole analysis to this JSON")
    ap.add_argument("--plot", help="write a PNG per run (this name, with the system appended when there are several runs; needs matplotlib)")
    a = ap.parse_args(argv)

    for name, val in (("--budget", a.budget), ("--recall", a.recall)):
        if not 0.0 < val <= 1.0:
            ap.error(f"{name} must be in (0, 1]")
    if a.bad_gap < 0 or not 0.0 < a.bad_lcs <= 1.0:
        ap.error("--bad-gap must be >= 0 and --bad-lcs in (0, 1]")

    results, csv_rows, status = [], [], 0
    for folder in a.runs:
        if not os.path.exists(os.path.join(folder, "records.jsonl")):
            print(f"no records.jsonl in {folder}", file=sys.stderr)
            return 1
        records = harness.load_records(folder)
        if not records:
            print(f"{folder}: no records", file=sys.stderr)
            return 1
        sealed = bool(_dataset_of(folder)) and dataset.is_sealed(_dataset_of(folder))
        if sealed and a.evaluate_at is None:
            print(f"{folder}: this is a run on the sealed split {_dataset_of(folder)!r}. Thresholds are tuned on the development set only; "
                  "to judge the threshold you chose there, add --evaluate-at T.", file=sys.stderr)
            return 1
        try:
            res = CAL.analyse_run(records, label=a.label, signal=a.signal, budget=a.budget, recall=a.recall, bad_gap=a.bad_gap,
                                  bad_lcs=a.bad_lcs, current=a.current, evaluate_at=a.evaluate_at, resamples=a.resamples,
                                  tune=not sealed)
        except CAL.CalibrationError as exc:
            print(f"{folder}: {exc}", file=sys.stderr)
            return 1
        res["folder"], res["sealed"] = folder, sealed
        print(CAL.format_report(res))
        print()
        results.append(res)
        csv_rows += CAL.sweep_rows(res)
        if a.plot:
            if sealed:
                print("(no plot for a sealed split: there is no sweep)", file=sys.stderr)
            else:
                path = a.plot if len(a.runs) == 1 else f"{os.path.splitext(a.plot)[0]}_{res['system'].replace('+', '_')}{os.path.splitext(a.plot)[1] or '.png'}"
                try:
                    CAL.plot(records, res, path)
                    print(f"plot: {path}")
                except ImportError:
                    print("--plot needs matplotlib (pip install matplotlib)", file=sys.stderr)
                    status = 1
    if a.csv:
        if csv_rows:
            with open(a.csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
                w.writeheader()
                w.writerows(csv_rows)
        else:
            print("--csv: nothing to write (sealed runs have no sweep)", file=sys.stderr)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(results if len(results) > 1 else results[0], f, indent=2, ensure_ascii=False, default=str)
    return status


if __name__ == "__main__":
    sys.exit(main())
