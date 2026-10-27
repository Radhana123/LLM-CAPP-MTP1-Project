"""Expert benchmark (MTP-1 WP2): parts spec, annotation workbooks, import of the plans, human-human agreement (M10), datasheet.
No API calls. The plans are written by people; this is the tooling around them (see capp/benchmark.py).

    python scripts/benchmark.py template --out bench/spec.xlsx                      # 1. empty parts spec (Excel); fill one row per part
    python scripts/benchmark.py sheets --spec bench/spec.xlsx --out-dir bench \\
        --annotator "Prof. Deb" --second-name "Second annotator" --second-n 18       # 2. workbooks for the annotators
    python scripts/benchmark.py import --spec bench/bench_spec.json \\
        --primary bench/annotate_primary.xlsx --second bench/annotate_second.xlsx    # 3. read the filled workbooks, build data/bench_v1.json
    python scripts/benchmark.py import ... --dry-run                                 #    only report problems, write nothing

To rehearse the whole chain before real parts exist, draw a PLACEHOLDER spec from a synthetic set (labelled as such in the data):

    python scripts/benchmark.py select --from test_v2 --n 50 --out bench/spec_placeholder.xlsx

The benchmark is sealed like test_v2: final runs only. Plans are written independently, nothing is pre-filled.
"""
import argparse
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import benchmark as B, dataset as D   # noqa: E402


def _errors(errs, what, limit=30):
    print(f"{what}: {len(errs)} problem(s)", file=sys.stderr)
    for e in errs[:limit]:
        print("  " + e, file=sys.stderr)
    if len(errs) > limit:
        print(f"  ... and {len(errs) - limit} more", file=sys.stderr)


def cmd_template(a):
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    print(f"wrote {B.write_spec_template(a.out)}: one row per part; fill it in, then run: sheets --spec {a.out}")
    return 0


def cmd_select(a):
    parts = D.load_dataset(a.source)
    picked = B.stratified_pick(parts, a.n, a.seed)
    rows = B.spec_from_dataset(picked, source=f"synthetic {a.source} (placeholder)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    B.write_spec_rows(rows, a.out)
    print(f"wrote {a.out}: {len(rows)} placeholder parts from {a.source} (synthetic, labelled as such). Next: sheets --spec {a.out}")
    return 0


def cmd_sheets(a):
    spec, errs = B.read_spec(a.spec)
    if errs:
        _errors(errs, f"{a.spec}")
        return 1
    os.makedirs(a.out_dir, exist_ok=True)
    primary = os.path.join(a.out_dir, "annotate_primary.xlsx")
    B.write_annotation_workbook(spec, primary, a.annotator, "primary", a.alternatives)
    with open(os.path.join(a.out_dir, "bench_spec.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"{len(spec)} parts. Wrote {primary} (primary: {a.annotator}, with review columns) and bench_spec.json (use it as --spec at import)")
    if a.second_n:
        subset = B.second_subset(spec, a.second_n, a.seed)
        second = os.path.join(a.out_dir, "annotate_second.xlsx")
        B.write_annotation_workbook(subset, second, a.second_name, "second", a.alternatives)
        print(f"Wrote {second} (second annotator: {a.second_name}) with {len(subset)} parts: {', '.join(p['part_id'] for p in subset)}")
    print("Give each person only their own workbook. They plan independently; nothing is pre-filled.")
    return 0


def cmd_import(a):
    spec, errs = B.read_spec(a.spec)
    if errs:
        _errors(errs, a.spec)
        return 1
    primary = B.read_annotations(a.primary)
    second = B.read_annotations(a.second) if a.second else None
    bench, rep = B.build_benchmark(spec, primary, second, allow_missing=a.allow_missing)
    print(f"{len(spec)} parts in the spec, {len(bench)} with a reference plan (annotator: {rep['annotator']}"
          + (f"; second: {rep['second_annotator']}" if rep["second_annotator"] else "") + ")")
    if rep["review"]:
        print("supervisor review: " + ", ".join(f"{k} {v}" for k, v in sorted(rep["review"].items())))
    if rep["unknown_parts"]:
        print(f"part ids in the workbooks but not in the spec (ignored): {', '.join(rep['unknown_parts'])}")
    if rep["missing"]:
        print(f"without a plan: {', '.join(rep['missing'])}" + ("" if a.allow_missing else "  (blocking; --allow-missing leaves them out)"))
    if rep["conflicts"]:
        print(f"\n{len(rep['conflicts'])} plan(s) the rule base rejects (kept as references: settle each with the supervisor):")
        for c in rep["conflicts"][:40]:
            print(f"  {c['part_id']} route {c['route']}: {', '.join(c['kinds'])}: {'; '.join(c['messages'][:2])}")
    if rep["physical"]:
        print(f"\n{len(rep['physical'])} plan(s) with a physical-feasibility finding on the real dimensions:")
        for c in rep["physical"][:40]:
            print(f"  {c['part_id']} route {c['route']}: " + ", ".join(c["infeasible"] + c["risks"]))
    if rep["errors"]:
        print()
        _errors(rep["errors"], "blocking")
        print("nothing written", file=sys.stderr)
        return 1
    agree = B.agreement(bench)
    if agree["n"]:
        print(f"\nM10 (second plan vs reference, {agree['n']} parts): exact {agree['exact_rate']:.2f} {B._ci(agree['exact_rate_ci95'])}, "
              f"LCS ratio {agree['lcs_ratio_mean']:.2f} {B._ci(agree['lcs_ratio_mean_ci95'])}, "
              f"Kendall tau {B._f(agree['kendall_tau_mean'])} {B._ci(agree['kendall_tau_mean_ci95'])}")
    if a.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    path = D.save_dataset(bench, a.out)
    sheet = a.datasheet or os.path.splitext(path)[0] + "_datasheet.md"
    with open(sheet, "w", encoding="utf-8", newline="\n") as f:
        f.write(B.datasheet(bench, rep, agree, generated=datetime.date.today().isoformat()))
    print(f"\nwrote {path} (+ .csv) and {sheet}. The benchmark is sealed: final runs only (run_eval --dataset {os.path.splitext(os.path.basename(path))[0]})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template", help="write an empty parts spec")
    t.add_argument("--out", default=os.path.join("bench", "spec.xlsx"), help=".xlsx (default) or .csv")
    t.set_defaults(fn=cmd_template)
    s = sub.add_parser("select", help="write a PLACEHOLDER spec drawn from a synthetic dataset (to rehearse the chain)")
    s.add_argument("--from", dest="source", default="test_v2", help="dataset to draw from (default test_v2: held-out archetypes)")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default=os.path.join("bench", "spec_placeholder.xlsx"), help=".xlsx or .csv")
    s.set_defaults(fn=cmd_select)
    w = sub.add_parser("sheets", help="write the annotation workbooks from a spec")
    w.add_argument("--spec", required=True)
    w.add_argument("--out-dir", default="bench")
    w.add_argument("--annotator", default="primary annotator", help="name of the person who writes the reference plans")
    w.add_argument("--second-name", default="second annotator")
    w.add_argument("--second-n", type=int, default=18, help="parts for the second annotator (protocol: 15 to 20; 0 = no second workbook)")
    w.add_argument("--seed", type=int, default=7)
    w.add_argument("--alternatives", type=int, default=2, help="plan rows per part (1 to 3)")
    w.set_defaults(fn=cmd_sheets)
    i = sub.add_parser("import", help="read the filled workbooks, check the plans, write the benchmark dataset and datasheet")
    i.add_argument("--spec", required=True, help="bench_spec.json written by 'sheets' (or the spec itself)")
    i.add_argument("--primary", required=True, help="the filled primary workbook")
    i.add_argument("--second", help="the filled second-annotator workbook")
    i.add_argument("--out", default="bench_v1", help="dataset name or path (default bench_v1 -> data/bench_v1.json)")
    i.add_argument("--datasheet", help="datasheet path (default: next to the dataset)")
    i.add_argument("--allow-missing", action="store_true", help="leave out parts without a plan instead of stopping")
    i.add_argument("--dry-run", action="store_true", help="report problems and M10, write nothing")
    i.set_defaults(fn=cmd_import)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
