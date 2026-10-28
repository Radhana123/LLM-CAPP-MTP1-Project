# Release: mtp1_rehearsal

> **DEVELOPMENT RESULTS, NOT PAPER RESULTS.** The grid `ablation_dev` runs on the development set `dev_v2`: this folder is a rehearsal of the release.

Made by `python scripts/release.py build <recipe>` (the recipe is `inputs/recipe.json`). Check it later with `python scripts/release.py verify <this folder>`: it tests that no file changed since the build and that building again from the same sources gives the same files.

Rehearsal on the development grid. For the final release: grid = your copy of the ablation config on test_v2 / bench_v1, benchmark = data/bench_v1.json, case_studies.runs = the baseline-row run folder of each LLM, extraction = the M9 run folder(s).

## Contents

| path | what |
|---|---|
| `RELEASE.md` | this file |
| `FREEZE.json` | the freeze manifest: the configuration, pinned before the first run on the sealed set |
| `freeze_verify.txt` | whether the working tree still matched the freeze when this was built |
| `SHA256SUMS.txt` | SHA-256 of every file here except release_manifest.json (text files with LF line endings) |
| `release_manifest.json` | the build record: time, commit, file hashes (not part of the comparison in `verify`) |
| `inputs/` | the recipe and copies of the files written by hand (prose, case commentary), so the folder says what it was made from |
| `tables/` | Table 4 (md, csv, tex), paired comparisons, consistency across LLMs, per-part results, issues, figures (`scripts/make_tables.py`) |
| `paper/` | `paper_draft.md` (facts and your prose) and `readiness.txt` |

## What it was made from

- Grid `configs/ablation_template.json` (`ablation_dev`): dataset `dev_v2` (development), 3 LLM(s), 10/10 rows built, 0/30 runs usable.
- Freeze: `2d59d8b0a213`, status **drift** when this was built.

## Readiness of the numbers

Not ready: 4 blocking item(s) open.
- [BLOCKING] the grid runs on dev_v2, a development set: these are development results, not paper results
- [BLOCKING] 30 run(s) missing, partial or excluded: A0/gpt120=missing, A0/gpt20=missing, A0/qwen=missing, A1/gpt120=missing, A1/gpt20=missing, A1/qwen=missing, A2/gpt120=missing, A2/gpt20=missing ...
- [BLOCKING] the working tree has drifted from freeze 2d59d8b0a213: results are not those of the frozen configuration
- [BLOCKING] 17 required section(s) without text: title, authors, abstract, introduction, related_work, method, setup, rq1, rq2, rq3, rq4, rq5, rq6, case_studies, discussion, limitations, conclusion
- [advisory] M4 is measured against the Route Builder variants (proxy_rb), not against an expert: say so, or run the benchmark
- [advisory] no case studies given (--case-studies): report 7.5 asks for 3 to 5 cases per main failure type
- [advisory] no extraction run given (--extraction): RQ6 has no evidence

## Before you publish it

```
python -m pytest
python scripts/freeze.py verify
python scripts/release.py verify <this folder>
git tag baseline-v0 <commit of the internship code>    # once, if it is not there yet (see BASELINE.md)
git tag freeze-v1                                     # the commit with FREEZE.json, made before the first run on the sealed set
git add <this folder>; git commit -m "Release mtp1_rehearsal"; git tag release-mtp1_rehearsal
git push origin --tags
```

## What this does not show

- `verify` shows that the folder is what the recipe and the inputs produce. It does not show that the inputs are right: the runs, the prose, the benchmark annotation and the case commentary are yours.
- The freeze shows that the files are the ones that were pinned. It does not show that no test result was seen before it was made.
- A release built from a development grid is a rehearsal, not a result.
