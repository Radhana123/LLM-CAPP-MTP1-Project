"""Failure case studies (report 7.5) from finished runs: the cases chosen by a fixed rule, with the facts of each from the records.

    python scripts/case_studies.py results/n5_r1_s3_ts_h1_qwen
    python scripts/case_studies.py results/b0_qwen results/n5_qwen --per-type 4
    python scripts/case_studies.py results/n5_qwen --types order,missing,differs
    python scripts/case_studies.py results/n5_qwen --instructive "DEV_016:1:0=the repair kept a wrong boring-before-drilling decision"
    python scripts/case_studies.py results/n5_qwen --out case_studies_final --strict

Per run and per failure type (order, missing, invented, duplicate, machine_grouping, infeasible) the first K parts in the order of the part
identifiers are taken, K = --per-type (default 3; the report asks for three to five), each with its first failing raw route (repeat, then
sample id). `differs` adds the parts whose final route is valid but not an exact match of the reference. --instructive adds a case you chose,
with the reason, labelled as instructive. Offline: reads records.jsonl and run.json, makes no API call.

Writes into --out (default <results folder>/case_studies/<run>):
    case_studies.md    the section: how the cases were chosen, the M3 aggregate per type, then every case with its facts
    cases.json         everything above in machine-readable form
    selection.csv      one row per case (which part, repeat, sample, type, repair outcome)
    commentary.json    {case id: text}. YOU write the commentary there (why the model failed; whether the repair kept what was right);
                       regenerating never overwrites it, and new cases get an empty entry. Cases without commentary show a marked gap.

Exit code: 0; 1 for a bad argument or run, or with --strict if any case still has no commentary.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import casestudy as C   # noqa: E402


def _fail(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def default_out(folders):
    first = folders[0].rstrip("/\\")
    label = os.path.basename(first)
    if len(folders) > 1:
        label += f"+{len(folders) - 1}more"
    return os.path.join(os.path.dirname(first) or ".", "case_studies", label)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", metavar="RUN", help="run folder(s) with records.jsonl (made with scripts/run_eval.py)")
    ap.add_argument("--per-type", type=int, default=C.DEFAULT_PER_TYPE, metavar="K", help=f"cases per type and run (default {C.DEFAULT_PER_TYPE})")
    ap.add_argument("--types", metavar="LIST", help=f"comma-separated, default {','.join(C.DEFAULT_TYPES)}; also available: {','.join(C.OPTIONAL_TYPES)}")
    ap.add_argument("--instructive", action="append", default=[], metavar="PART[:REPEAT[:SAMPLE]]=WHY",
                    help="add a case you chose, with the reason (repeatable; the reason is required)")
    ap.add_argument("--commentary", metavar="FILE", help="the commentary file (default <out>/commentary.json)")
    ap.add_argument("--out", metavar="DIR", help="output folder")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any case has no commentary yet")
    args = ap.parse_args(argv)

    types = [t.strip() for t in args.types.split(",") if t.strip()] if args.types else list(C.DEFAULT_TYPES)
    out_dir = args.out or default_out(args.runs)
    com_path = args.commentary or os.path.join(out_dir, "commentary.json")
    try:
        specs = [C.parse_instructive(t) for t in args.instructive]
        runs = [C.load_run(f) for f in args.runs]
        labels = [r["label"] for r in runs]
        if len(set(labels)) != len(labels):
            raise C.CaseError(f"two runs have the same folder name {sorted({x for x in labels if labels.count(x) > 1})}: the case ids would collide")
        res = C.analyse(runs, types=types, per_type=args.per_type, instructive=specs, commentary=C.read_commentary(com_path))
        paths = C.write_outputs(res, out_dir)
        added = C.sync_commentary(com_path, res["cases"])
    except C.CaseError as exc:
        _fail(str(exc))

    for run in res["runs"]:
        print(f"{run['label']}: system {run['system']}, {run['n_parts']} parts x {len(run['repeats'])} repeat(s), {run['n_parsed_routes']} raw LLM routes")
        if not run["n_parsed_routes"]:
            print("  no raw LLM routes: nothing to study")
            continue
        for row in run["table"]:
            n = sum(1 for c in res["cases"] if c["run"] == run["label"] and c["type"] == row["type"])
            share = "" if row["share"] is None else f"  ({100 * row['share']:.1f}% of routes)"
            what = f"{row['routes']} routes in {row['parts']} parts{share}" if row["type"] != "differs" else f"{row['parts']} parts"
            print(f"  {row['type']:<17} {what:<38} -> {n} case(s){('   ' + row['note']) if row['note'] and row['type'] != 'differs' else ''}")
        if run["sealed"]:
            print(f"  note: {run['dataset']} is sealed: use cases from the final run only, after the configuration was frozen")
    print(f"{len(res['cases'])} case(s); {res['n_without_commentary']} without commentary")
    for name in sorted(paths):
        print(f"  {paths[name]}")
    print(f"  {com_path}" + (f"  ({added} new empty entr{'y' if added == 1 else 'ies'})" if added else ""))
    if res["orphan_commentary"]:
        print(f"  note: {len(res['orphan_commentary'])} written commentary entr{'y' if len(res['orphan_commentary']) == 1 else 'ies'} match no case now (kept in the file)")
    if args.strict and res["n_without_commentary"]:
        print(f"--strict: {res['n_without_commentary']} case(s) have no commentary", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
