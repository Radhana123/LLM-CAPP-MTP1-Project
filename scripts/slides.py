"""Presentation skeleton (MTP-1 WP12, report 8.4 item vi): the slides of the final talk, with the facts in place and your sentences in a
separate file, and a check of what is still missing.

    python scripts/slides.py release/v1                                   # from a release folder (scripts/release.py): its tables, cases, benchmark
    python scripts/slides.py release/v1 --format marp --out slides_final --strict
    python scripts/slides.py configs/ablation_template.json               # rehearsal on the development grid
    python scripts/slides.py configs/final.json --benchmark data/bench_v1.json --case-studies results/case_studies/final \\
        --extraction results/fe1_extraction --freeze FREEZE.json --tables-dir results/tables/final

Writes <out>/slides.md (default out = slides), <out>/readiness.txt and creates <out>/slides_text.json (or --text FILE): one short entry per
slide, empty. Write your sentences there; regenerating puts them in place and never touches them. The script writes the tables and the
numbers and, in the speaker notes, what to say about them; it never writes what the numbers mean. One `## ` heading is one slide.

Convert:  pandoc slides/slides.md -o slides/slides.pptx        (--format pandoc, the default; notes become speaker notes)
          Marp for VS Code, or  marp slides/slides.md --pptx   (--format marp)
Check the layout yourself: wide tables may need a smaller font. --strict: exit 1 if a blocking item is open (the files are still written).
No API call is made and no run is started; see capp/slides.py.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import paperdraft as P, release as R, slides as S, tables as T      # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="a release folder (made by scripts/release.py), or the ablation config of the grid")
    ap.add_argument("--benchmark", help="grid config only: the expert benchmark")
    ap.add_argument("--case-studies", help="grid config only: folder written by scripts/case_studies.py")
    ap.add_argument("--extraction", action="append", default=[], help="grid config only: folder of an extraction_eval run (M9); repeatable")
    ap.add_argument("--freeze", help="grid config only: freeze manifest (default FREEZE.json in the repository, if it exists)")
    ap.add_argument("--tables-dir", help="grid config only: where make_tables.py build wrote its figures")
    ap.add_argument("--results-dir", help="where the runs are (default: the config's results_dir)")
    ap.add_argument("--text", help="the file with your sentences (default <out>/slides_text.json)")
    ap.add_argument("--out", default="slides", help="output folder (default slides)")
    ap.add_argument("--format", choices=S.FORMATS, default="pandoc", help="pandoc (default) or marp")
    ap.add_argument("--strict", action="store_true", help="exit 1 if a blocking item is open")
    a = ap.parse_args(argv)
    try:
        if os.path.isdir(a.source):
            ctx = S.collect_from_release(a.source, ROOT, a.results_dir)
        else:
            ctx = P.collect(a.source, a.benchmark, a.case_studies, a.extraction, a.freeze, a.tables_dir, a.results_dir)
        out = S.write_all(ctx, a.text or os.path.join(a.out, S.STATUS_FILE), a.out, a.format)
    except (T.TableError, P.PaperError, R.ReleaseError, OSError, UnicodeError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    with open(out["paths"]["readiness"], encoding="utf-8") as f:
        print(f.read())
    print(f"Slides:    {out['paths']['slides']}  ({out['slides']} slides, {a.format})\nYour text: {out['paths']['text']}"
          + (f"  ({out['added']} empty entr{'y' if out['added'] == 1 else 'ies'} added)" if out["added"] else ""))
    if out["orphans"]:
        print(f"Note: {len(out['orphans'])} entr{'y' if len(out['orphans']) == 1 else 'ies'} of the text file match no slide ({', '.join(out['orphans'])}); kept.")
    if a.strict and not out["ready"]:
        print("--strict: blocking item(s) open.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
