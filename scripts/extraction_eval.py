"""M9, extraction quality (report 6.5, WP7, RQ6), and the corrections log of the interface.

    # 1. ground truth: a file to fill in, one entry per drawing of a folder (features from the 36-feature vocabulary, hole diameters optional)
    python scripts/extraction_eval.py template drawings/ --out data/extraction_truth.json

    # 2. extract every drawing (live API calls, cached; one run folder per configuration), then score it
    python scripts/extraction_eval.py run drawings/ --truth data/extraction_truth.json --tag fe1
    python scripts/extraction_eval.py run drawings/ --truth data/extraction_truth.json --tag fe2 --evidence --dimensions
    python scripts/extraction_eval.py run drawings/ --truth data/extraction_truth.json --tag fe1 --mode replay    # from the cache only

    # 3. scoring is offline and can be repeated: precision / recall / F1 per pass, union (v1), intersection, agreement-based; corrections per drawing
    python scripts/extraction_eval.py score results/fe1_extraction --csv results/m9_fe1
    python scripts/extraction_eval.py score results/fe1_extraction results/fe2_extraction          # two configurations, one after the other

    # 4. what real users changed in the interface (week6/app.py writes results/corrections_log.jsonl each time Generate is pressed)
    python scripts/extraction_eval.py log results/corrections_log.jsonl

Variants: geometry pass alone, annotation pass alone, union (v1), intersection, and the agreement rule of report 6.5 (features both passes saw are
pre-accepted, the rest are highlighted and confirmed) scored after an ORACLE user confirmed the highlighted ones: an upper bound; `log` has the
real users' numbers. Develop prompts on a development subset of the drawings, freeze them, and report M9 on the rest.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import extraction_eval as EE      # noqa: E402
from capp import extraction_review as ER    # noqa: E402


def _print_score(run_dir, csv_prefix=None):
    records = EE.load_records(run_dir)
    s = EE.score(records)
    info = {}
    try:
        with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError):
        pass
    label = os.path.basename(run_dir.rstrip("/\\")) + (" (evidence prompt)" if info.get("with_evidence") else "")
    print(EE.format_report(s, label))
    if csv_prefix:
        for p in EE.write_csv(csv_prefix, s):
            print(f"wrote {p}")
    return s


def cmd_template(args):
    t = EE.template(args.folder)
    if not t:
        print(f"no drawing (.png .jpg .jpeg) in {args.folder}", file=sys.stderr)
        return 1
    if os.path.exists(args.out) and not args.force:
        print(f"{args.out} exists: not overwritten (--force to replace it)", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(t, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}: {len(t)} drawings. Fill in each \"features\" (names from the vocabulary) and, if the drawing writes them, "
          "\"hole_diameters\" (mm).")
    print("Vocabulary: " + ", ".join(sorted(EE.fv.GEOMETRY_FEATURES)))
    return 0


def cmd_run(args):
    truth = EE.load_ground_truth(args.truth)
    if args.mode:
        os.environ["CAPP_LLM_MODE"] = args.mode
    out = args.out or os.path.join(ROOT, "results", f"{args.tag}_extraction")
    pairs, no_image, no_truth = EE.match_drawings(args.folder, truth)
    if no_image:
        print(f"truth entries without a drawing: {', '.join(no_image[:8])}", file=sys.stderr)
    if no_truth:
        print(f"drawings without a truth (not run): {', '.join(no_truth[:8])}", file=sys.stderr)
    EE.run(args.folder, truth, out, tag=args.tag, with_dimensions=args.dimensions, with_evidence=args.evidence, limit=args.limit,
           progress=lambda m: print(m, file=sys.stderr))
    print(f"records: {out}")
    _print_score(out, args.csv)
    return 0


def cmd_score(args):
    for i, d in enumerate(args.runs):
        if i:
            print("\n" + "=" * 100 + "\n")
        prefix = args.csv
        if args.csv and len(args.runs) > 1:                 # one set of CSV files per run folder
            prefix = args.csv + "_" + os.path.basename(d.rstrip("/\\"))
        _print_score(d, prefix)
    return 0


def cmd_log(args):
    entries, bad = ER.read_log(args.path)
    print(ER.format_log_report(ER.summarize_log(entries), args.path, bad))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template", help="write a ground-truth file to fill in, one entry per drawing of a folder")
    t.add_argument("folder")
    t.add_argument("--out", required=True)
    t.add_argument("--force", action="store_true")
    t.set_defaults(fn=cmd_template)

    r = sub.add_parser("run", help="extract every drawing that has a ground truth, then score")
    r.add_argument("folder", help="folder of drawings (.png .jpg .jpeg)")
    r.add_argument("--truth", required=True, help="ground-truth JSON (see `template`)")
    r.add_argument("--tag", default="fe", help="run label, also part of the cache key; the run folder is results/<tag>_extraction")
    r.add_argument("--out", help="run folder (default results/<tag>_extraction)")
    r.add_argument("--evidence", action="store_true", help="use the evidence prompt (a short evidence phrase per feature; a different prompt: new API calls)")
    r.add_argument("--dimensions", action="store_true", help="also run the hole-diameter pass (one more call per drawing) and score it where the truth has diameters")
    r.add_argument("--limit", type=int, help="only the first N drawings")
    r.add_argument("--mode", choices=["live", "replay", "nocache"], help="sets CAPP_LLM_MODE (default: live)")
    r.add_argument("--csv", metavar="PREFIX", help="also write PREFIX_variants.csv, PREFIX_features.csv, PREFIX_drawings.csv")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("score", help="score one or more run folders (offline)")
    s.add_argument("runs", nargs="+", help="run folder(s) with records.jsonl")
    s.add_argument("--csv", metavar="PREFIX")
    s.set_defaults(fn=cmd_score)

    g = sub.add_parser("log", help="summarize the corrections log of the interface")
    g.add_argument("path", nargs="?", default=os.path.join(ROOT, "results", "corrections_log.jsonl"))
    g.set_defaults(fn=cmd_log)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except EE.ExtractionEvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
