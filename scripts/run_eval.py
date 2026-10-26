"""Run a reference system (rb, b1, b0) over a dataset and write results to results/<tag>_<system>/.

Systems: rb (Route Builder), b1 / b0 (v1 baselines, 1 / 5 samples, R0 = whole-route replacement), b1_r1 / b0_r1 (same, but
invalid samples are repaired by R1, minimal edit, R0 as fallback).

Examples (repo root, PowerShell):
    python scripts/run_eval.py --system rb --dataset dev_v2                       # no API calls
    python scripts/run_eval.py --system b0 --dataset dev_v2 --limit 20 --tag dev  # first 20 parts, live API (cached)
    python scripts/run_eval.py --system b1 --dataset dev_v2 --limit 20 --tag dev  # same tag: B1 reuses B0's first sample
    python scripts/run_eval.py --system b0_r1 --dataset dev_v2 --limit 20 --tag dev   # same tag: B0's samples from cache, no API
    python scripts/run_eval.py --system b0 --dataset dev_v2 --repeats 5 --tag dev # five repeated runs (M8)
    python scripts/run_eval.py --summarize-only results/dev_b0                    # re-print a finished run

Do NOT run the sealed test set (test_v2) until the final runs (WP11).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import dataset, harness, summary, systems   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=sorted(systems.SYSTEMS))
    ap.add_argument("--dataset", default="dev_v2", help="dev_v2 (default) or test_v2 (sealed)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N parts")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1, help="repeated runs for B1/B0 (default 1; 5 for the final runs)")
    ap.add_argument("--tag", default="dev", help="run label; also part of the cache key. Same tag => B1 shares B0's cache")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--mode", choices=["live", "replay", "nocache"], help="sets CAPP_LLM_MODE (default: live)")
    ap.add_argument("--throttle", type=float, default=0.0, help="seconds to sleep after a part that made live API calls")
    ap.add_argument("--summarize-only", metavar="DIR", help="print the summary of an existing results folder and exit")
    args = ap.parse_args()

    if args.summarize_only:
        recs = harness.load_records(args.summarize_only)
        print(summary.format_summary(recs[0]["system"], summary.summarize(recs)))
        return
    if not args.system:
        ap.error("--system is required (or use --summarize-only)")
    if args.dataset.startswith("test") and args.system != "rb":
        print("WARNING: test_v2 is sealed (final runs only). Continue only if this is the final run.", file=sys.stderr)
    if args.mode:
        os.environ["CAPP_LLM_MODE"] = args.mode

    parts = dataset.load_dataset(args.dataset)[args.offset:]
    if args.limit:
        parts = parts[:args.limit]
    try:
        out = harness.run_experiment(args.system, parts, repeats=args.repeats, tag=args.tag, out_dir=args.out,
                                     dataset_name=args.dataset, throttle=args.throttle)
    except harness.HarnessError as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        sys.exit(2)
    print()
    print(summary.format_summary(args.system, out["summary"]))
    print(f"\nFiles: {out['out_dir']}")


if __name__ == "__main__":
    main()
