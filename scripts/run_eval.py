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

Constrained (graph-guided) planner (row A9, RQ5, capp/constrained.py):
  +cg1          the route is built one operation at a time: the operation SET is the Route Builder's, and at each step the LLM sees only the
                operations whose prerequisites are placed and answers with the number of the next one. Valid by construction (nothing to
                repair), but it cannot add an operation the set lacks. One short call per step that has a real choice (about 2 per route
                on the development set; forced steps cost nothing). Own prompts, so it cannot be combined with +fs, +h1 or +ts.
  e.g. n1_r0_s1+cg1 (one route), n5_r0_s1+cg1 (five routes, compare with b0_r1 = A2), n5_r0_s3+cg1 (with selection)

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
    python scripts/run_eval.py --system n5_r1_s3+ms --limit 20 --tag dev          # small model only (gpt-oss-20b); +ml = large only (gpt-oss-120b)
    python scripts/run_eval.py --system n5_r1_s3+cas1 --limit 20 --tag dev        # cascade: small first, large where repair fails (A8); +cas2 = eager
    python scripts/cascade_pareto.py results/dev_n5_r1_s3+ms results/dev_n5_r1_s3+ml results/dev_n5_r1_s3+cas1   # validity against cost
    python scripts/run_eval.py --system n5_r0_s1+cg1 --limit 20 --tag dev         # constrained planner (A9): about 2 short calls per sample
    python scripts/run_eval.py --system b0 --dataset dev_v2 --repeats 5 --tag dev # five repeated runs (M8)
    python scripts/run_eval.py --summarize-only results/dev_b0                    # re-print a finished run
    python scripts/run_eval.py --summarize-only results/dev_n5_r1_s3 --confidence-flag 0.93   # same run, plans flagged below confidence 0.93 (scripts/confidence_tune.py picks T)

Do NOT run the sealed sets (test_v2, the expert benchmark bench_v1) until the final runs (WP11). A sealed run with LLM calls needs a freeze:
`python scripts/freeze.py create --grid <config>` first (capp/freeze.py); run_eval stops if FREEZE.json is missing or the working tree has
drifted from it (--no-freeze-check to override; the freeze status of every run is written to run.json).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import config, dataset, freeze, harness, summary, systems   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", help="rb, b0, b1, b0_r1 ... b1_r3, or generic n<N>_r<0-3>_s<0-3>, plus options +fs2/+fs4/+h1/+ts/+fz1/+ms/+ml/+cas1/+cas2/+cg1 (see above)")
    ap.add_argument("--freeze", help="freeze manifest checked before a sealed dataset is run (default FREEZE.json in the repository)")
    ap.add_argument("--no-freeze-check", action="store_true",
                    help="run a sealed dataset although the freeze is missing or differs (the status is still written to run.json and flagged in Table 4)")
    ap.add_argument("--pool", default="dev_v2", help="where +fs2/+fs4 take their examples from (default dev_v2; never test_v2)")
    ap.add_argument("--no-analysis", action="store_true", help="skip the physics analysis (kc, tool wear, Ra) and the physical-feasibility check of the final and raw routes")
    ap.add_argument("--weights", help="selection S2 priorities as time,cost,energy (default equal), e.g. 0.5,0.3,0.2")
    ap.add_argument("--dataset", default="dev_v2", help="dev_v2 (default), or test_v2 / bench_v1 (sealed: final runs only)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N parts")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1, help="repeated runs for B1/B0 (default 1; 5 for the final runs)")
    ap.add_argument("--tag", default="dev", help="run label; also part of the cache key. Same tag => B1 shares B0's cache")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--mode", choices=["live", "replay", "nocache"], help="sets CAPP_LLM_MODE (default: live)")
    ap.add_argument("--throttle", type=float, default=0.0, help="seconds to sleep after a part that made live API calls")
    ap.add_argument("--summarize-only", metavar="DIR", help="print the summary of an existing results folder and exit")
    ap.add_argument("--confidence-flag", type=float, metavar="T",
                    help="flag a plan for human review when its confidence is below T (default systems.CONFIDENCE_FLAG = 0.8; "
                         "pick T with scripts/confidence_tune.py on the development set). With --summarize-only the existing records are re-flagged.")
    args = ap.parse_args()

    if args.confidence_flag is not None:
        if not 0.0 < args.confidence_flag <= 1.0 + 1e-9:
            ap.error("--confidence-flag must be in (0, 1]: confidence is an agreement between 0 and 1")
        systems.CONFIDENCE_FLAG = args.confidence_flag
    if args.summarize_only:
        recs = harness.load_records(args.summarize_only)
        if args.confidence_flag is not None:
            recs = summary.reflag(recs, args.confidence_flag)
        print(summary.format_summary(recs[0]["system"], summary.summarize(recs)))
        return
    if not args.system:
        ap.error("--system is required (or use --summarize-only)")
    try:
        systems.parse_system(args.system)
        pspec = systems.planner_spec(args.system)
    except ValueError as exc:
        ap.error(str(exc))
    if pspec.few_shot and dataset.is_sealed(args.pool):
        ap.error("--pool: few-shot examples must not come from the sealed test set or the expert benchmark")
    weights = None
    if args.weights:
        try:
            weights = [float(x) for x in args.weights.split(",")]
            systems.normalize_weights(weights)
        except ValueError as exc:
            ap.error(f"--weights: {exc}")
    if dataset.is_sealed(args.dataset) and args.system != "rb":
        print(f"WARNING: {args.dataset} is sealed (final runs only: no prompt or setting may change after its results are seen). "
              "Continue only if this is the final run.", file=sys.stderr)
        fz = freeze.status(config.ROOT, args.freeze)
        if fz["status"] in freeze.OK_STATUSES:
            print(f"Freeze {fz['id']}: {fz['status']} ({fz['manifest']})", file=sys.stderr)
        elif args.no_freeze_check:
            print(f"WARNING: freeze status is {fz['status']!r}; continuing because of --no-freeze-check. The status is written to run.json "
                  "and Table 4 will flag this run.", file=sys.stderr)
        else:
            why = {"none": f"no freeze manifest ({fz['manifest']}): create it with `python scripts/freeze.py create --grid <config>`, then commit and tag",
                   "drift": "the working tree differs from the freeze: " + "; ".join(fz.get("differences", [])[:5]) + " (python scripts/freeze.py verify)",
                   "unreadable": f"the freeze manifest cannot be used: {fz.get('why')}"}.get(fz["status"], fz["status"])
            print(f"\nSTOPPED: sealed dataset {args.dataset} and {why}. For a deliberate exception use --no-freeze-check.", file=sys.stderr)
            sys.exit(2)
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
