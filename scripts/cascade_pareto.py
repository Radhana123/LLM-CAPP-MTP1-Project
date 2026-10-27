"""Validity against cost for the model cascade: one point per finished run, Pareto frontier marked (report 6.4, row A8).

    python scripts/cascade_pareto.py results/dev_n5_r1_s3+ms results/dev_n5_r1_s3+ml results/dev_n5_r1_s3+cas1 results/dev_n5_r1_s3+cas2
    python scripts/cascade_pareto.py results/dev_n5_r1_s3 results/dev_n5_r1_s3+ms results/dev_n5_r1_s3+ml results/dev_n5_r1_s3+cas1 \\
        --plot results/pareto.png --csv results/pareto.csv

The runs are made with the same --tag on the same parts, so the cascade reuses the cache of the small-only and large-only runs:

    python scripts/run_eval.py --system n5_r1_s3+ms   --limit 20 --tag dev      # the cheap end (new API calls: small model)
    python scripts/run_eval.py --system n5_r1_s3+ml   --limit 20 --tag dev      # the strong end (new API calls: large model)
    python scripts/run_eval.py --system n5_r1_s3+cas1 --limit 20 --tag dev      # between them, no new sample call
    python scripts/run_eval.py --system n5_r1_s3+cas2 --limit 20 --tag dev

No API calls here. Prices are list prices (capp/config.py PRICE_USD_PER_MTOK, to be confirmed in the Groq console).
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import harness, pareto   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="result folders (each with records.jsonl)")
    ap.add_argument("--csv", help="write the points to this CSV")
    ap.add_argument("--json", help="write the whole analysis to this JSON")
    ap.add_argument("--plot", help="write a PNG of cost against the share of LLM routes (needs matplotlib)")
    a = ap.parse_args(argv)
    runs = []
    for d in a.runs:
        if not os.path.exists(os.path.join(d, "records.jsonl")):
            print(f"no records.jsonl in {d}", file=sys.stderr)
            return 1
        recs = harness.load_records(d)
        if not recs:
            print(f"{d}: no records", file=sys.stderr)
            return 1
        runs.append((recs[0]["system"], recs))
    labels = [lab for lab, _ in runs]
    if len(set(labels)) != len(labels):
        print("two runs of the same system: " + ", ".join(sorted({x for x in labels if labels.count(x) > 1})), file=sys.stderr)
        return 1
    res = pareto.analyse(runs)
    print(pareto.render(res))
    if a.csv:
        cols = [k for k in res["points"][0] if k not in ("models", "unpriced")] + ["models"]
        with open(a.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for p in res["points"]:
                w.writerow([";".join(p["models"]) if c == "models" else p[c] for c in cols])
        print(f"\nwrote {a.csv}")
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"wrote {a.json}")
    if a.plot:
        try:
            pareto.plot(res, a.plot)
            print(f"wrote {a.plot}")
        except ImportError:
            print("matplotlib is not installed: no plot (pip install matplotlib)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
