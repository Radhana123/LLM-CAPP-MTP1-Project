"""Run a reference system (rb, b1, b0) over a dataset and write results to results/<tag>_<system>/.

Systems: rb (Route Builder), b1 / b0 (v1 baselines, 1 / 5 samples, R0 = whole-route replacement), and the repair variants
b1_r1 / b0_r1 (R1: minimal-edit repair by rules, no API calls), b1_r2 / b0_r2 (R2: one LLM re-prompt with the violation report,
extra API calls, cached), b1_r3 / b0_r3 (R3: R2, then R1 if the re-prompt fails). R0 is the last resort of every variant.
Generic names n<N>_r<0-3>_s<0-3> choose everything: N samples (1-20), repair R0-R3, selection S0-S3 (report 6.4):
  s0 first valid candidate | s1 v1 (front, then best Efficiency) | s2 weighted time/cost/energy on the front (--weights)
  s3 medoid: the front route that agrees most with the other samples.   n5_r0_s1 is b0, n1_r0_s1 is b1.
  e.g. n8_r1_s3 = 8 samples, minimal-edit repair, consistency-aware selection.

Planner options (report 6.2), appended with '+' to any LLM system. A different prompt always means new API calls:
  +fs2 / +fs4   k retrieved examples (most similar development parts by Jaccard overlap of their feature sets, then material) as
                earlier turns of the chat; the part itself is never an example. +fs2x / +fs4x also leave out every part with the
                SAME feature set (a combination the pool has not seen). Examples come from --pool (default dev_v2).
  +h1           the Route Builder's route is given in the prompt as a hint (H1). +h0 / +fs0 = off: exactly the v1 prompt.
  +ts           two-step planner: call 1 reasons in plain text (sampled), call 2 turns the reasoning into the operation list under
                structured output (json_schema with the closed vocabulary; falls back to json_object, then plain text, and records
                which one was used). TWO API calls per sample. With +fs/+h1 the examples and hint go into the reasoning prompt.
                CAPP_STRUCTURED=json_object|none sets the first structured-output mode to try.
  e.g. n5_r1_s3+fs2, b0+h1, n5_r1_s3+fs4x+h1, n5_r1_s3+ts, n5_r1_s3+ts+h1

Feasibility-aware repair (row A7, capp/feasrepair.py), also appended with '+', needs repair r1, r2 or r3. It does NOT change the prompt,
so it uses the same cached LLM answers as the system without it: no new API calls (only R2/R3 re-prompts that carry a feasibility finding are new).
  +fz1          the HARD physical-feasibility violations (an operation that needs a hole before any Drilling/Boring, a lathe operation after
                Parting-off) are repaired too, by R1 (insert the hole-making operation, constrain the order); selection prefers routes with
                the fewest hard findings. Machine-capability findings cannot be fixed by editing a route: reported only.
  +fz2          +fz1, and R1 also inserts what the risk rules ask for (Center Drilling before Drilling, Reaming for a tight hole tolerance);
                selection prefers the fewest hard findings, then the fewest risks.
  e.g. n5_r1_s3+fz1, n5_r1_s3+fz2, n5_r1_s3+fs2+fz2   (compare with n5_r1_s3 on the same --tag: "feasibility" lines of the summary)

Examples (repo root, PowerShell):
    python scripts/run_eval.py --system rb --dataset dev_v2                       # no API calls
    python scripts/run_eval.py --system b0 --dataset dev_v2 --limit 20 --tag dev  # first 20 parts, live API (cached)
    python scripts/run_eval.py --system b1 --dataset dev_v2 --limit 20 --tag dev  # same tag: B1 reuses B0's first sample
    python scripts/run_eval.py --system b0_r1 --dataset dev_v2 --limit 20 --tag dev   # same tag: B0's samples from cache, no API
    python scripts/run_eval.py --system b0_r2 --dataset dev_v2 --limit 20 --tag dev   # B0's samples from cache + live re-prompts
    python scripts/run_eval.py --system b0_r3 --dataset dev_v2 --limit 20 --tag dev   # reuses R2's cached re-prompts
    python scripts/run_eval.py --system n5_r1_s3 --dataset dev_v2 --limit 20 --tag dev   # B0's samples + R1 + medoid selection
    python scripts/run_eval.py --system n5_r1_s2 --weights 0.6,0.2,0.2 --limit 20 --tag dev  # time-first priorities
    python scripts/run_eval.py --system n5_r1_s3+fs2 --limit 20 --tag dev          # few-shot, 2 retrieved examples (new API calls)
    python scripts/run_eval.py --system b0+h1 --limit 20 --tag dev                 # with the Route Builder hint
    python scripts/run_eval.py --system n5_r1_s3+ts --limit 10 --tag dev           # two-step planner (2 calls per sample, new API calls)
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
    ap.add_argument("--system", help="rb, b0, b1, b0_r1 ... b1_r3, or generic n<N>_r<0-3>_s<0-3>, plus options +fs2/+fs4/+h1 (see above)")
    ap.add_argument("--pool", default="dev_v2", help="where +fs2/+fs4 take their examples from (default dev_v2; never test_v2)")
    ap.add_argument("--no-analysis", action="store_true", help="skip the physics analysis (kc, tool wear, Ra) and the physical-feasibility check of the final and raw routes")
    ap.add_argument("--weights", help="selection S2 priorities as time,cost,energy (default equal), e.g. 0.5,0.3,0.2")
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
    try:
        systems.parse_system(args.system)
        pspec = systems.planner_spec(args.system)
    except ValueError as exc:
        ap.error(str(exc))
    if pspec.few_shot and args.pool.startswith("test"):
        ap.error("--pool: few-shot examples must not come from the sealed test set")
    weights = None
    if args.weights:
        try:
            weights = [float(x) for x in args.weights.split(",")]
            systems.normalize_weights(weights)
        except ValueError as exc:
            ap.error(f"--weights: {exc}")
    if args.dataset.startswith("test") and args.system != "rb":
        print("WARNING: test_v2 is sealed (final runs only). Continue only if this is the final run.", file=sys.stderr)
    if args.mode:
        os.environ["CAPP_LLM_MODE"] = args.mode

    parts = dataset.load_dataset(args.dataset)[args.offset:]
    if args.limit:
        parts = parts[:args.limit]
    try:
        out = harness.run_experiment(args.system, parts, repeats=args.repeats, tag=args.tag, out_dir=args.out,
                                     dataset_name=args.dataset, throttle=args.throttle, weights=weights,
                                     pool=args.pool, analysis=not args.no_analysis)
    except harness.HarnessError as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        sys.exit(2)
    print()
    print(summary.format_summary(args.system, out["summary"]))
    print(f"\nFiles: {out['out_dir']}")


if __name__ == "__main__":
    main()
