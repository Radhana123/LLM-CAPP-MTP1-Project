# _TO WRITE: title_

_TO WRITE: authors_

> **DEVELOPMENT RESULTS, NOT PAPER RESULTS.** This draft was generated from the grid `ablation_dev` on the development set `dev_v2`. Use it to rehearse the paper; the numbers of the paper come from the final run on the sealed benchmark, after the freeze. 4 blocking item(s) in Appendix A.

## Abstract

_TO WRITE (150-250 words): Problem, what was built, the evaluation (data, LLMs, repeats), the main results with their numbers from the facts below, the limits._

**Facts for the abstract** (generated; for each LLM the baseline row and the deepest row, the one with the most components added: a rule that does not look at the results, not a selection of the best):

| LLM | Row | Configuration | M1 first-pass valid | M2 LLM-final | M6 gap (pts) | M7 calls/part |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | – | no usable run | – | – | – | – |
| gpt-oss-120b | – | no usable run | – | – | – | – |
| gpt-oss-20b | – | no usable run | – | – | – | – |

## 1 Introduction

_TO WRITE (400-700 words): Why reliability of LLM process planning matters, the gaps (report 3.2), the questions below, the contributions as a list._

Research questions (MTP-1 report, section 3.3):

- **RQ1**: How often and in what ways does the v1 LLM planner fail, in terms of precedence violations, missing or invented operations and physical infeasibility, when evaluated against expert-validated plans?
- **RQ2**: Does two-step reason-then-formalise planning reduce first-pass failures, and how much of any gain depends on the rule-based hint?
- **RQ3**: Does localized minimal-edit repair preserve more of the LLM's valid decisions than whole-route replacement, without loss of validity?
- **RQ4**: Do multi-candidate generation and consistency-aware selection improve plan quality and stability, and at what cost in calls, tokens and latency?
- **RQ5**: Is it better to repair an LLM plan after generation, or to constrain generation step by step with the precedence graph? (stretch)
- **RQ6**: Do per-feature evidence and agreement-based selective confirmation reduce the number of user corrections per drawing without lowering the accuracy of the confirmed feature list?

## 2 Related work

_TO WRITE (350-600 words): LLM-based CAPP studies by input, planning method, verification and repair, evaluation data and metrics (the table of report 8.4), and where this work differs._

## 3 System and method

_TO WRITE (500-900 words): The planner, the verifier and the repair, selection, the extended verifier, the constrained planner: what each component does, in the words of the configurations table._

### Configurations evaluated

| ID | Configuration | System | Adds to | Component(s) changed | Research question |
|---|---|---|---|---|---|
| A0 | B1: one LLM call, whole-route replacement | `b1` | – | – | – |
| A1 | B0: five samples, whole-route replacement (v1 default) | `b0` | A0 | samples | RQ4 |
| A2 | A1 + localized minimal-edit repair | `b0_r1` | A1 | repair | RQ3 |
| A3 | A2 + two-step planner, with Route Builder hint | `b0_r1+ts+h1` | A2 | two-step, hint | RQ2 |
| A4 | A2 + two-step planner, without hint | `b0_r1+ts` | A2 | two-step | RQ2 |
| A5 | Better of A3 and A4 + multi-candidate selection | `n5_r1_s3+ts+h1` | A3 | selection | RQ4 |
| A6 | A5 + retrieval-based few-shot | `n5_r1_s3+ts+h1+fs2` | A5 | few-shot | RQ2 |
| A7 | A6 + extended verifier (feasibility rules) | `n5_r1_s3+ts+h1+fs2+fz2` | A6 | feasibility | – |
| A8 | A7 + model cascade (optional) | `n5_r1_s3+ts+h1+fs2+fz2+cas1` | A7 | cascade | – |
| A9 | Constrained planner (optional) | `n5_r0_s1+cg1` | A2 | repair, constrained | RQ5 |

System names:

- `b1`: one LLM call per part.
- `b0`: five samples (v1 default).
- `_r1`: localized minimal-edit repair (R1).
- `_s3`: verifier-guided selection with the confidence score.
- `+ts`: two-step planner: a reasoning call, then a formalisation call.
- `+h1`: the Route Builder's route is given to the LLM as a hint.
- `+fs`: retrieval-based few-shot examples from the development set.
- `+fz`: feasibility-aware repair with the physical rules.
- `+cas`: model cascade: the small model first, the large one where repair fails.
- `+cg1`: constrained planner: the LLM picks the next operation among those whose prerequisites are placed.

## 4 Experimental setup

### Data

- Grid data: `dev_v2` (development set), 20 parts, 13 part archetypes, feature coverage 18/36, features per part 3: 2, 4: 14, 5: 2, 7: 2.
- Materials: Aluminum 5, Brass 2, Cast Iron 3, Copper 1, Plastic 1, Steel 6, Titanium 2. Tolerances: 0.005mm 3, 0.01mm 2, 0.02mm 4, 0.05mm 3, 0.1mm 6, 0.5mm 2.
- 3 repeats per cell by default; bootstrap 1000 resamples, seed 7.

### LLMs

| ID | LLM | Model id | List price in / out (USD per million tokens) |
|---|---|---|---|
| qwen | Qwen3.8-27B | `qwen/qwen3.8-27b` | 0.8 / 4 |
| gpt120 | gpt-oss-120b | `openai/gpt-oss-120b` | 0.15 / 0.6 |
| gpt20 | gpt-oss-20b | `openai/gpt-oss-20b` | 0.075 / 0.3 |

### Metrics and statistics

- M1 first-pass valid: share of the parsed raw LLM routes that pass the strict verifier.
- M2 final valid: share of final routes that are strict-valid (the Route Builder fallback keeps this near 1). M2 LLM-final: share of final routes that are LLM routes, raw or repaired, not the Route Builder's.
- M5 ops kept: share of the LLM's operations that survive replacement or repair (higher = more minimal); – where nothing had to be fixed.
- M6 gap: mean Efficiency of the final route below the best Route Builder variant, in points (lower is better, negative = better).
- M7: API calls per part (all calls) and list-price USD per part (n/a: no price for a model, see capp/config.py).
- M4: no reference route available.

Paired comparisons (each row against the row it adds a component to): per part the mean over repeats, 95% percentile bootstrap over parts (1000 resamples, seed 7); validity as a paired binary outcome with McNemar's exact test; the same comparison for every LLM, because the claim that a component helps needs the same answer for all of them.

### Reproducibility

- Freeze `2d59d8b0a213` created 2026-10-27T22:46:22+00:00, commit `8b29b0076d`; the working tree now: **drift**.
- Tests at freeze time: 1530 passed, 0 failed.
- Models: large `openai/gpt-oss-120b`, primary `qwen/qwen3.8-27b`, small `openai/gpt-oss-20b`.
- Runs were made under freeze id(s): none recorded.

_TO WRITE (80-300 words): Anything about the data or the protocol that the facts do not say: how the benchmark was annotated, why these LLMs, how the development set was used._

## 5 Results

### Table 4: ablation grid

Dataset dev_v2 (parts 1-20), 3 repeats by default. A cell is mean ± SD over the repeats; a plain value is one repeat; – = no data.

#### Qwen3.8-27B (`qwen/qwen3.8-27b`)

| ID | Configuration | System | M1 first-pass valid | M2 final valid | M2 LLM-final | M4 similarity | M5 ops kept | M6 gap (pts) | M7 calls/part | M7 USD/part |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | B1: one LLM call, whole-route replacement | `b1` | – | – | – | – | – | – | – | – |
| A1 | B0: five samples, whole-route replacement (v1 default) | `b0` | – | – | – | – | – | – | – | – |
| A2 | A1 + localized minimal-edit repair | `b0_r1` | – | – | – | – | – | – | – | – |
| A3 | A2 + two-step planner, with Route Builder hint | `b0_r1+ts+h1` | – | – | – | – | – | – | – | – |
| A4 | A2 + two-step planner, without hint | `b0_r1+ts` | – | – | – | – | – | – | – | – |
| A5 | Better of A3 and A4 + multi-candidate selection | `n5_r1_s3+ts+h1` | – | – | – | – | – | – | – | – |
| A6 | A5 + retrieval-based few-shot | `n5_r1_s3+ts+h1+fs2` | – | – | – | – | – | – | – | – |
| A7 | A6 + extended verifier (feasibility rules) | `n5_r1_s3+ts+h1+fs2+fz2` | – | – | – | – | – | – | – | – |
| A8 | A7 + model cascade (optional) | `n5_r1_s3+ts+h1+fs2+fz2+cas1` | – | – | – | – | – | – | – | – |
| A9 | Constrained planner (optional) | `n5_r0_s1+cg1` | – | – | – | – | – | – | – | – |

- A0: no run yet
- A1: no run yet
- A2: no run yet
- A3: no run yet
- A4: no run yet
- A5: no run yet
- A6: no run yet
- A7: no run yet
- A8: no run yet
- A9: no run yet
- A5: decide: this takes A3 (with hint). If A4 is better, use "n5_r1_s3+ts" and parent "A4"; decide on the development set, before the sealed set.
- A6: decide: k = 2 retrieved examples (fs2); fs4 is the other choice. Examples come only from the development data.
- A7: decide: fz2 = hard findings and risk insertions; fz1 = hard findings only. fz does not change the prompt, so it costs no new sample calls.
- A8: optional. The cascade chooses its own models (small gpt-oss-20b, then large gpt-oss-120b), so it is the same run in every column and A8 vs A7 mixes the effect of the cascade with a change of model: read it with scripts/cascade_pareto.py (validity against cost).
- A9: optional, WP10 (RQ5). Same five samples and the same selection as A2 (b0_r1), but each sample is built step by step under the precedence graph (+cg1), so nothing needs repair: paired with A2 it answers 'repair after generation or constraint during generation'. It cannot add an operation the Route Builder's set lacks, so read it with the flags and M4, not validity alone. Calls: about 2 per sample on the development set, not one per step, because most steps have a single option. For a comparison with the full pipeline add a row n5_r0_s3+cg1 next to A5.

#### gpt-oss-120b (`openai/gpt-oss-120b`)

| ID | Configuration | System | M1 first-pass valid | M2 final valid | M2 LLM-final | M4 similarity | M5 ops kept | M6 gap (pts) | M7 calls/part | M7 USD/part |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | B1: one LLM call, whole-route replacement | `b1` | – | – | – | – | – | – | – | – |
| A1 | B0: five samples, whole-route replacement (v1 default) | `b0` | – | – | – | – | – | – | – | – |
| A2 | A1 + localized minimal-edit repair | `b0_r1` | – | – | – | – | – | – | – | – |
| A3 | A2 + two-step planner, with Route Builder hint | `b0_r1+ts+h1` | – | – | – | – | – | – | – | – |
| A4 | A2 + two-step planner, without hint | `b0_r1+ts` | – | – | – | – | – | – | – | – |
| A5 | Better of A3 and A4 + multi-candidate selection | `n5_r1_s3+ts+h1` | – | – | – | – | – | – | – | – |
| A6 | A5 + retrieval-based few-shot | `n5_r1_s3+ts+h1+fs2` | – | – | – | – | – | – | – | – |
| A7 | A6 + extended verifier (feasibility rules) | `n5_r1_s3+ts+h1+fs2+fz2` | – | – | – | – | – | – | – | – |
| A8 | A7 + model cascade (optional) | `n5_r1_s3+ts+h1+fs2+fz2+cas1` | – | – | – | – | – | – | – | – |
| A9 | Constrained planner (optional) | `n5_r0_s1+cg1` | – | – | – | – | – | – | – | – |

- A0: no run yet
- A1: no run yet
- A2: no run yet
- A3: no run yet
- A4: no run yet
- A5: no run yet
- A6: no run yet
- A7: no run yet
- A8: no run yet
- A9: no run yet
- A5: decide: this takes A3 (with hint). If A4 is better, use "n5_r1_s3+ts" and parent "A4"; decide on the development set, before the sealed set.
- A6: decide: k = 2 retrieved examples (fs2); fs4 is the other choice. Examples come only from the development data.
- A7: decide: fz2 = hard findings and risk insertions; fz1 = hard findings only. fz does not change the prompt, so it costs no new sample calls.
- A8: optional. The cascade chooses its own models (small gpt-oss-20b, then large gpt-oss-120b), so it is the same run in every column and A8 vs A7 mixes the effect of the cascade with a change of model: read it with scripts/cascade_pareto.py (validity against cost).
- A9: optional, WP10 (RQ5). Same five samples and the same selection as A2 (b0_r1), but each sample is built step by step under the precedence graph (+cg1), so nothing needs repair: paired with A2 it answers 'repair after generation or constraint during generation'. It cannot add an operation the Route Builder's set lacks, so read it with the flags and M4, not validity alone. Calls: about 2 per sample on the development set, not one per step, because most steps have a single option. For a comparison with the full pipeline add a row n5_r0_s3+cg1 next to A5.

#### gpt-oss-20b (`openai/gpt-oss-20b`)

| ID | Configuration | System | M1 first-pass valid | M2 final valid | M2 LLM-final | M4 similarity | M5 ops kept | M6 gap (pts) | M7 calls/part | M7 USD/part |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | B1: one LLM call, whole-route replacement | `b1` | – | – | – | – | – | – | – | – |
| A1 | B0: five samples, whole-route replacement (v1 default) | `b0` | – | – | – | – | – | – | – | – |
| A2 | A1 + localized minimal-edit repair | `b0_r1` | – | – | – | – | – | – | – | – |
| A3 | A2 + two-step planner, with Route Builder hint | `b0_r1+ts+h1` | – | – | – | – | – | – | – | – |
| A4 | A2 + two-step planner, without hint | `b0_r1+ts` | – | – | – | – | – | – | – | – |
| A5 | Better of A3 and A4 + multi-candidate selection | `n5_r1_s3+ts+h1` | – | – | – | – | – | – | – | – |
| A6 | A5 + retrieval-based few-shot | `n5_r1_s3+ts+h1+fs2` | – | – | – | – | – | – | – | – |
| A7 | A6 + extended verifier (feasibility rules) | `n5_r1_s3+ts+h1+fs2+fz2` | – | – | – | – | – | – | – | – |
| A8 | A7 + model cascade (optional) | `n5_r1_s3+ts+h1+fs2+fz2+cas1` | – | – | – | – | – | – | – | – |
| A9 | Constrained planner (optional) | `n5_r0_s1+cg1` | – | – | – | – | – | – | – | – |

- A0: no run yet
- A1: no run yet
- A2: no run yet
- A3: no run yet
- A4: no run yet
- A5: no run yet
- A6: no run yet
- A7: no run yet
- A8: no run yet
- A9: no run yet
- A5: decide: this takes A3 (with hint). If A4 is better, use "n5_r1_s3+ts" and parent "A4"; decide on the development set, before the sealed set.
- A6: decide: k = 2 retrieved examples (fs2); fs4 is the other choice. Examples come only from the development data.
- A7: decide: fz2 = hard findings and risk insertions; fz1 = hard findings only. fz does not change the prompt, so it costs no new sample calls.
- A8: optional. The cascade chooses its own models (small gpt-oss-20b, then large gpt-oss-120b), so it is the same run in every column and A8 vs A7 mixes the effect of the cascade with a change of model: read it with scripts/cascade_pareto.py (validity against cost).
- A9: optional, WP10 (RQ5). Same five samples and the same selection as A2 (b0_r1), but each sample is built step by step under the precedence graph (+cg1), so nothing needs repair: paired with A2 it answers 'repair after generation or constraint during generation'. It cannot add an operation the Route Builder's set lacks, so read it with the flags and M4, not validity alone. Calls: about 2 per sample on the development set, not one per step, because most steps have a single option. For a comparison with the full pipeline add a row n5_r0_s3+cg1 next to A5.

### RQ1: How often and in what ways does the v1 LLM planner fail, in terms of precedence violations, missing or invented operations and physical infeasibility, when evaluated against expert-validated plans?

Baseline rows (no component added):

| Row | Configuration | LLM | M1 first-pass valid | M2 LLM-final | M6 gap (pts) |
|---|---|---|---|---|---|
| A0 | B1: one LLM call, whole-route replacement | Qwen3.8-27B | – | – | – |
| A0 | B1: one LLM call, whole-route replacement | gpt-oss-120b | – | – | – |
| A0 | B1: one LLM call, whole-route replacement | gpt-oss-20b | – | – | – |
| A1 | B0: five samples, whole-route replacement (v1 default) | Qwen3.8-27B | – | – | – |
| A1 | B0: five samples, whole-route replacement (v1 default) | gpt-oss-120b | – | – | – |
| A1 | B0: five samples, whole-route replacement (v1 default) | gpt-oss-20b | – | – | – |

Failure types of the raw LLM routes (M3):

No case-study folder was given (`--case-studies`), so the failure types (M3) are not tabulated here.

_TO WRITE (80-250 words): What the baseline failure rates and the failure types show. Say what the numbers support and what they do not._

### RQ2: Does two-step reason-then-formalise planning reduce first-pass failures, and how much of any gain depends on the rule-based hint?

Comparisons that isolate this component:

No paired comparison has data yet (a run of the pair is missing or unusable).

_TO WRITE (80-250 words): Two-step, hint and few-shot: what the paired comparisons show across the LLMs._

### RQ3: Does localized minimal-edit repair preserve more of the LLM's valid decisions than whole-route replacement, without loss of validity?

Comparisons that isolate this component:

No paired comparison has data yet (a run of the pair is missing or unusable).

M5 (share of the LLM's operations that survive the fix; higher = more minimal; – where nothing had to be fixed):

| Row | Qwen3.8-27B | gpt-oss-120b | gpt-oss-20b |
|---|---|---|---|
| A1: B0: five samples, whole-route replacement (v1 default) | – | – | – |
| A2: A1 + localized minimal-edit repair | – | – | – |

_TO WRITE (80-250 words): Repair against whole-route replacement: validity and how much of the LLM's route survives (M5)._

### RQ4: Do multi-candidate generation and consistency-aware selection improve plan quality and stability, and at what cost in calls, tokens and latency?

Comparisons that isolate this component:

No paired comparison has data yet (a run of the pair is missing or unusable).

_TO WRITE (80-250 words): Candidates and selection: quality, stability and the cost in calls and tokens._

### RQ5: Is it better to repair an LLM plan after generation, or to constrain generation step by step with the precedence graph? (stretch)

Comparisons that isolate this component:

No paired comparison has data yet (a run of the pair is missing or unusable).

The constrained planner cannot add an operation the Route Builder's set lacks, and asks the LLM only where the precedence graph leaves a choice (calls per part in M7).

_TO WRITE (80-250 words): Repair after generation or constraint during generation; or say that it was not run._

### RQ6: Do per-feature evidence and agreement-based selective confirmation reduce the number of user corrections per drawing without lowering the accuracy of the confirmed feature list?

No extraction run was given (`--extraction`): M9 is not available in this draft.

_TO WRITE (80-250 words): Corrections per drawing and the accuracy of the confirmed list; the oracle is an upper bound._

### Further components (objectives 5 and 7)

Feasibility rules and the model cascade are not research questions of their own; their effect on the same metrics:

No paired comparison has data yet (a run of the pair is missing or unusable).

### Failure case studies

No case-study folder was given (`--case-studies`), so the failure types (M3) are not tabulated here.

_TO WRITE (200-500 words): What the failure cases show that the aggregate numbers do not; cite the cases by id._

## 6 Discussion

_TO WRITE (300-600 words): What holds across the LLMs, what does not, what surprised you._

## 7 Limitations

_TO WRITE (150-350 words): In your words, beyond the generated list._

Limitations that follow from this run (generated):

- The grid is on the development set `dev_v2` (synthetic parts): these are not test results.
- M4: no reference route available.
- Cells without a complete, checked run: A0/gpt120, A0/gpt20, A0/qwen, A1/gpt120, A1/gpt20, A1/qwen, A2/gpt120, A2/gpt20, A2/qwen, A3/gpt120, A3/gpt20, A3/qwen, A4/gpt120, A4/gpt20, A4/qwen, A5/gpt120, A5/gpt20, A5/qwen, A6/gpt120, A6/gpt20, A6/qwen, A7/gpt120, A7/gpt20, A7/qwen, A8/gpt120, A8/gpt20, A8/qwen, A9/gpt120, A9/gpt20, A9/qwen.
- Intervals and p-values are not corrected for the number of comparisons. With several repeats the McNemar pairs are clustered by part, so those p-values are approximate; the bootstrap over parts is the primary evidence.
- The models are hosted and can be retired or change behaviour: the identifiers are pinned in `capp/config.py` and every response is cached, but a re-run against the live endpoint may differ.
- The constrained planner depends on the completeness of the rule base: it cannot invent an operation the Route Builder's set lacks.

## 8 Conclusion

_TO WRITE (120-250 words): Answers in one paragraph and the next steps._

## Appendix A: readiness

```
NOT READY: 4 blocking, 3 advisory item(s) open

[BLOCKING] the grid runs on dev_v2, a development set: these are development results, not paper results
[ok] 3 LLM(s) in the grid (at least 3 are required)
[BLOCKING] 30 run(s) missing, partial or excluded: A0/gpt120=missing, A0/gpt20=missing, A0/qwen=missing, A1/gpt120=missing, A1/gpt20=missing, A1/qwen=missing, A2/gpt120=missing, A2/gpt20=missing ...
[ok] no error in the table issues
[BLOCKING] the working tree has drifted from freeze 2d59d8b0a213: results are not those of the frozen configuration
[advisory] M4 is measured against the Route Builder variants (proxy_rb), not against an expert: say so, or run the benchmark
[advisory] no case studies given (--case-studies): report 7.5 asks for 3 to 5 cases per main failure type
[advisory] no extraction run given (--extraction): RQ6 has no evidence
[BLOCKING] 17 required section(s) without text: title, authors, abstract, introduction, related_work, method, setup, rq1, rq2, rq3, rq4, rq5, rq6, case_studies, discussion, limitations, conclusion
[ok] no TODO or placeholder left in the prose
[ok] every written section is within its word budget
[ok] 0 words of prose (6 to 8 pages is roughly 4000 to 5500 words with the tables)

Prose:
  Title                  0 words  (3-20)  empty
  Authors                0 words  (1-40)  empty
  Abstract               0 words  (150-250)  empty
  Introduction           0 words  (400-700)  empty
  Related work           0 words  (350-600)  empty
  Method                 0 words  (500-900)  empty
  Setup remarks          0 words  (80-300)  empty
  RQ1 answer             0 words  (80-250)  empty
  RQ2 answer             0 words  (80-250)  empty
  RQ3 answer             0 words  (80-250)  empty
  RQ4 answer             0 words  (80-250)  empty
  RQ5 answer             0 words  (80-250)  empty
  RQ6 answer             0 words  (80-250)  empty
  Case studies           0 words  (200-500)  empty
  Discussion             0 words  (300-600)  empty
  Limitations            0 words  (150-350)  empty
  Conclusion             0 words  (120-250)  empty
  Acknowledgements       0 words  (0-100)  empty
```

## Appendix B: how to reproduce

```
python scripts/release.py build <recipe>     # the recipe is inputs/recipe.json
python scripts/release.py verify <this folder>
```

## Appendix C: issues of the table build

```
MISSING  A0 gpt120: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt120_b1
MISSING  A0 gpt20: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt20_b1
MISSING  A0 qwen: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_qwen_b1
MISSING  A1 gpt120: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt120_b0
MISSING  A1 gpt20: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt20_b0
MISSING  A1 qwen: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_qwen_b0
MISSING  A2 gpt120: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt120_b0_r1
MISSING  A2 gpt20: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt20_b0_r1
MISSING  A2 qwen: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_qwen_b0_r1
MISSING  A3 gpt120: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt120_b0_r1+ts+h1
MISSING  A3 gpt20: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt20_b0_r1+ts+h1
MISSING  A3 qwen: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_qwen_b0_r1+ts+h1
MISSING  A4 gpt120: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt120_b0_r1+ts
MISSING  A4 gpt20: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt20_b0_r1+ts
MISSING  A4 qwen: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_qwen_b0_r1+ts
MISSING  A5 gpt120: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt120_n5_r1_s3+ts+h1
MISSING  A5 gpt20: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt20_n5_r1_s3+ts+h1
MISSING  A5 qwen: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_qwen_n5_r1_s3+ts+h1
MISSING  A6 gpt120: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt120_n5_r1_s3+ts+h1+fs2
MISSING  A6 gpt20: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt20_n5_r1_s3+ts+h1+fs2
MISSING  A6 qwen: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_qwen_n5_r1_s3+ts+h1+fs2
MISSING  A7 gpt120: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt120_n5_r1_s3+ts+h1+fs2+fz2
MISSING  A7 gpt20: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt20_n5_r1_s3+ts+h1+fs2+fz2
MISSING  A7 qwen: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_qwen_n5_r1_s3+ts+h1+fs2+fz2
MISSING  A8: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_shared_n5_r1_s3+ts+h1+fs2+fz2+cas1
MISSING  A9 gpt120: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt120_n5_r0_s1+cg1
MISSING  A9 gpt20: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_gpt20_n5_r0_s1+cg1
MISSING  A9 qwen: no run: C:\All-Projects\LLM-CAPP-MTP1-Project\results\abl_qwen_n5_r0_s1+cg1
```
