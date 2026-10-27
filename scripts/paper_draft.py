"""Paper draft (MTP-1 WP12, report 8.4): the facts of the paper from the finished artefacts, the prose from you, and a check of what is
still missing before the numbers can be used.

    python scripts/paper_draft.py configs/ablation_template.json --out paper                       # rehearsal on the development grid
    python scripts/paper_draft.py configs/final.json --benchmark data/bench_v1.json --case-studies results/case_studies/final \\
        --extraction results/fe1_extraction --freeze FREEZE.json --tables-dir results/tables/final --out paper --strict

Writes <out>/paper_draft.md and <out>/readiness.txt, and creates <out>/paper_text.json (or --text FILE): one entry per section, empty.
Write your sentences there; regenerating the draft puts them in place and never touches them. The script writes the numbers (Table 4,
the paired comparisons per research question, M3, M9, the benchmark, the freeze) and never a sentence that says what they mean.
--strict: exit 1 if a blocking item is open (the draft is still written). No API call is made and no run is started; see capp/paperdraft.py.
Convert with pandoc if you want Word or LaTeX:  pandoc paper/paper_draft.md -o paper/paper_draft.docx
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import paperdraft as P, tables as T      # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="ablation config of the grid (scripts/make_tables.py)")
    ap.add_argument("--benchmark", help="the expert benchmark (data/bench_v1.json)")
    ap.add_argument("--case-studies", help="folder written by scripts/case_studies.py (cases.json)")
    ap.add_argument("--extraction", action="append", default=[], help="folder of an extraction_eval run (M9); repeatable")
    ap.add_argument("--freeze", help="freeze manifest (default FREEZE.json in the repository, if it exists)")
    ap.add_argument("--tables-dir", help="where make_tables.py build wrote its figures (linked from the draft)")
    ap.add_argument("--results-dir", help="where the runs are (default: the config's results_dir)")
    ap.add_argument("--text", help="the file with your prose (default <out>/paper_text.json)")
    ap.add_argument("--out", default="paper", help="output folder (default paper)")
    ap.add_argument("--strict", action="store_true", help="exit 1 if a blocking item is open")
    a = ap.parse_args(argv)
    try:
        ctx = P.collect(a.config, a.benchmark, a.case_studies, a.extraction, a.freeze, a.tables_dir, a.results_dir)
        out = P.write_all(ctx, a.text or os.path.join(a.out, "paper_text.json"), a.out)
    except (T.TableError, P.PaperError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    with open(out["paths"]["readiness"], encoding="utf-8") as f:
        print(f.read())
    print(f"Draft:     {out['paths']['draft']}\nYour text: {out['paths']['text']}" + (f"  ({out['added_sections']} empty section(s) added)" if out["added_sections"] else ""))
    if out["orphans"]:
        print(f"Note: {len(out['orphans'])} entr{'y' if len(out['orphans']) == 1 else 'ies'} of the text file match no section ({', '.join(out['orphans'])}); kept.")
    if a.strict and not out["ready"]:
        print("--strict: blocking item(s) open.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
