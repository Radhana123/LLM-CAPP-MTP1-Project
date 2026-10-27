"""Route sheet / process plan (WP9): per operation machine, tool, cutting parameters, estimated time and the rule that puts it there.
No API calls.

    # a part typed in: the Route Builder's route, as the benchmark's RB system selects it
    python scripts/route_sheet.py make --material Steel --features Hole,Chamfer,Slot --diameter 40 --length 120 --depth 15 --width 32 \\
        --hole-diameter 8 --ra 1.6 --batch 100 --out results/sheet.pdf results/sheet.xlsx results/sheet.html

    # the final route of one part of a run (any system): provenance, plan confidence, review flag, before/after repair, reasoning
    python scripts/route_sheet.py from-run results/dev_n5_r1_s3 --part DEV_004 --out results/DEV_004.pdf
    python scripts/route_sheet.py from-run results/dev_n5_r1_s3 --part DEV_004 --repeat 1 --confidence-flag 0.7

    # the tool rules and reference ranges with their status and sources, for you to check against the handbook
    python scripts/route_sheet.py tables
    python scripts/route_sheet.py tables --csv results/process_tables.csv

--out takes one or more files; the extension chooses the format (.pdf needs reportlab, .xlsx needs openpyxl, .html, .md, .json need
nothing). Without --out the sheet is printed as Markdown. Without dimensions the reference part of v1 (Ø25, L50, depth 20) is used and the
sheet says so. The sheet is a DRAFT: the parameters are the v1 model's, the ranges come from web pages, the tool rules from general
practice, and none of it is confirmed against the handbook (see `tables`). The route is chosen first; the sheet never changes it.
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import analysis_view as AV      # noqa: E402
from capp import harness, systems         # noqa: E402
from capp import process_plan as PP       # noqa: E402
from capp import process_tables as PT     # noqa: E402
from capp import routesheet as RS         # noqa: E402

import feature_vocab as fv                # noqa: E402  (after capp: capp.legacy puts the v1 modules on the path)

DIM_NAMES = ("diameter", "length", "depth", "width")


def _emit(plan, outs, generated_at=None):
    if not outs:
        sys.stdout.write(RS.to_markdown(plan, generated_at))
        return
    for path in outs:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        RS.write(plan, path, generated_at)
        print(f"wrote {path}")


def _summary(plan) -> str:
    t, st = plan["totals"], plan["tables"]["status"]
    flags = sum(1 for e in plan["operations"] if e["flags"])
    return (f"{t['n_operations']} operations, {t['n_changeovers']} changeover(s), estimated {t['total_min']:.1f} min per part; "
            f"{flags} operation(s) with a note; reference ranges used: {sum(st.values())} (confirmed {st['confirmed']}, web {st['web']}, "
            f"mixed {st['mixed']}, indicative {st['indicative']})")


def cmd_make(args):
    feats = [f.strip() for f in args.features.split(",") if f.strip()]
    bad = [f for f in feats if not fv.is_valid_feature(f)]
    if not feats or bad:
        print(f"unknown feature(s): {', '.join(bad) or '(none given)'}. Features: {', '.join(sorted(fv.get_all_features()))}", file=sys.stderr)
        return 1
    given = {k: getattr(args, k) for k in DIM_NAMES if getattr(args, k) is not None}
    if given and len(given) < len(DIM_NAMES):
        print("give all four dimensions (--diameter --length --depth --width) or none", file=sys.stderr)
        return 1
    if args.hole_diameter and not given:
        print("--hole-diameter needs the part dimensions as well", file=sys.stderr)
        return 1
    part = AV.make_part(args.material, feats, args.tolerance, args.batch, dims=given or None, required_ra_um=args.ra, hole_diameter=args.hole_diameter)
    part["part_id"] = args.part_id
    part["machine_preference"] = args.machine
    if args.material not in PT.MATERIALS:
        print(f"unknown material {args.material!r}. Materials: {', '.join(PT.MATERIALS)}", file=sys.stderr)
        return 1
    if args.route:
        steps = [s.strip() for s in args.route.split(">") if s.strip()]
        ctx = {"system": "given route", "provenance": "given", "source": "route given on the command line"}
    else:
        rec = systems.run_rb(dict(part, dims=part["dims"] or None)).to_dict()
        steps = rec["final"]["steps"]
        ctx = {"system": "rb", "provenance": "route_builder", "source": "route_builder"}
    plan = PP.build_plan(steps, part, context=ctx)
    _emit(plan, args.out, args.generated_at)
    print(_summary(plan), file=sys.stderr)
    return 0


def cmd_from_run(args):
    if not os.path.exists(os.path.join(args.run, "records.jsonl")):
        print(f"no records.jsonl in {args.run}", file=sys.stderr)
        return 1
    recs = harness.load_records(args.run)
    mine = [r for r in recs if r["part_id"] == args.part]
    if not mine:
        ids = sorted({r["part_id"] for r in recs})
        print(f"part {args.part!r} is not in this run. Parts: {', '.join(ids[:8])}{' ...' if len(ids) > 8 else ''}", file=sys.stderr)
        return 1
    if args.repeat is None:
        rec = mine[0]
        if len(mine) > 1:
            print(f"{len(mine)} repeats of {args.part} in this run: using repeat {rec.get('repeat')} (--repeat N to choose)", file=sys.stderr)
    else:
        rec = next((r for r in mine if r.get("repeat") == args.repeat), None)
        if rec is None:
            print(f"no repeat {args.repeat} of {args.part}; repeats: {', '.join(str(r.get('repeat')) for r in mine)}", file=sys.stderr)
            return 1
    thr = args.confidence_flag
    if thr is None:
        try:
            import json
            with open(os.path.join(args.run, "run.json"), encoding="utf-8") as f:
                thr = json.load(f).get("confidence_flag_threshold")
        except (OSError, ValueError):
            thr = None
    plan = PP.plan_from_record(rec, flag_threshold=thr)
    _emit(plan, args.out, args.generated_at)
    print(_summary(plan), file=sys.stderr)
    return 0


def cmd_tables(args):
    rows = PT.rows()
    problems = PT.validate()
    counts = PT.status_counts([r for r in PT.RANGES.values()])
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["table", "key", "value", "status", "sources", "note"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}")
    else:
        for r in rows:
            if args.table in ("all", r["table"]):
                print(f"{r['table']:5} {r['key']:34} {r['status']:10} {r['sources']:8} {r['value']}")
    print(f"\n{len(rows)} rows; reference ranges by status: " + ", ".join(f"{k} {v}" for k, v in counts.items()) +
          "; every tool rule is 'indicative'. Nothing here is confirmed against the handbook until a row says 'confirmed'.")
    for k, (what, url) in PT.SOURCES.items():
        print(f"[{k}] {what}: {url}")
    if problems:
        print("\nTABLE PROBLEMS:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", nargs="+", metavar="FILE", help="write the sheet to these files (.pdf .xlsx .html .md .json); default: Markdown on screen")
        p.add_argument("--generated-at", default=None, help=argparse.SUPPRESS)

    m = sub.add_parser("make", help="sheet for a part typed in (route: the Route Builder's)")
    m.add_argument("--material", required=True, help=f"one of {', '.join(PT.MATERIALS)}")
    m.add_argument("--features", required=True, help="comma-separated, e.g. Hole,Chamfer,Slot")
    m.add_argument("--tolerance", default="0.05mm")
    m.add_argument("--batch", type=int, default=100)
    for d in DIM_NAMES:
        m.add_argument(f"--{d}", type=float, help="mm (all four dimensions, or none)")
    m.add_argument("--hole-diameter", type=float, help="mm")
    m.add_argument("--ra", type=float, help="required surface roughness Ra in um")
    m.add_argument("--machine", choices=("auto", "lathe", "milling"), default="auto", help="machine preference of the Route Builder")
    m.add_argument("--part-id", default="UI")
    m.add_argument("--route", help="use this route instead of the Route Builder's, e.g. 'Facing > Drilling > Inspection'")
    common(m)
    m.set_defaults(fn=cmd_make)

    r = sub.add_parser("from-run", help="sheet for the final route of one part of a run folder")
    r.add_argument("run", help="run folder (contains records.jsonl)")
    r.add_argument("--part", required=True, help="part id, e.g. DEV_004")
    r.add_argument("--repeat", type=int, help="repeat number (default: the first)")
    r.add_argument("--confidence-flag", type=float, help="review threshold for the flag (default: the one of the run)")
    common(r)
    r.set_defaults(fn=cmd_from_run)

    t = sub.add_parser("tables", help="print or export the tool rule and reference range tables with status and sources")
    t.add_argument("--csv", metavar="FILE")
    t.add_argument("--table", choices=("all", "tool", "range"), default="range")
    t.set_defaults(fn=cmd_tables)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except RS.RouteSheetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
