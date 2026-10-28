---
title: "TO WRITE: title"
author: "TO WRITE: authors and date"
subtitle: "DEVELOPMENT RESULTS, NOT PAPER RESULTS"
---


## Problem and gap

_TO WRITE (15-70 words): Why an LLM process plan cannot be trusted as it is, and what is missing in earlier work: three or four short bullets (report 3.2)._

::: notes
- State the problem in one sentence before the bullets.
- Name the gap that no earlier study closes (report 3.2).
:::


## Foundation: LLM-CAPP v1

_TO WRITE (15-70 words): The Route Builder, the LLM planner, the verifier and the scoring of v1, and what MTP-1 adds._

::: notes
- Say what v1 already does and what it cannot guarantee.
- Say what MTP-1 adds, in one sentence.
:::


## Research questions

- **RQ1**: How often and in what ways does the v1 LLM planner fail, in terms of precedence violations, missing or invented operations and physical infeasibility, when evaluated against expert-validated plans?
- **RQ2**: Does two-step reason-then-formalise planning reduce first-pass failures, and how much of any gain depends on the rule-based hint?
- **RQ3**: Does localized minimal-edit repair preserve more of the LLM's valid decisions than whole-route replacement, without loss of validity?
- **RQ4**: Do multi-candidate generation and consistency-aware selection improve plan quality and stability, and at what cost in calls, tokens and latency?
- **RQ5**: Is it better to repair an LLM plan after generation, or to constrain generation step by step with the precedence graph? (stretch)
- **RQ6**: Do per-feature evidence and agreement-based selective confirmation reduce the number of user corrections per drawing without lowering the accuracy of the confirmed feature list?

::: notes
- Read the questions in your own words, not off the slide.
- Say which of them the data of this talk can answer and which not.
:::


## Components under test

_TO WRITE (15-70 words): One short line per component, in plain words._

| Row | Configuration | Adds to its parent | Answers |
|---|---|---|---|
| A0 | B1: one LLM call, whole-route replacement | – | – |
| A1 | B0: five samples, whole-route replacement (v1 default) | samples | RQ4 |
| A2 | A1 + localized minimal-edit repair | repair | RQ3 |
| A3 | A2 + two-step planner, with Route Builder hint | two-step, hint | RQ2 |
| A4 | A2 + two-step planner, without hint | two-step | RQ2 |
| A5 | Better of A3 and A4 + multi-candidate selection | selection | RQ4 |
| A6 | A5 + retrieval-based few-shot | few-shot | RQ2 |
| A7 | A6 + extended verifier (feasibility rules) | feasibility | further |
| A8 | A7 + model cascade (optional) | cascade | further |
| A9 | Constrained planner (optional) | repair, constrained | RQ5 |

::: notes
- Walk down the rows: each row adds one component to its parent.
- The last column says which research question the row answers.
:::


## Data and protocol

- Grid `ablation_dev`: dataset `dev_v2` (development set), 20 parts, 3 repeats per cell.
- LLMs: Qwen3.8-27B (`qwen/qwen3.8-27b`), gpt-oss-120b (`openai/gpt-oss-120b`), gpt-oss-20b (`openai/gpt-oss-20b`).
- Statistics: paired bootstrap, 95% interval over parts (1000 resamples); McNemar's exact test for validity; no correction for the number of comparisons.
- Configuration freeze `2d59d8b0a213`, working tree drift.

_TO WRITE (10-60 words): Why these LLMs, how many repeats, how the benchmark was annotated and the development set used._

::: notes
- Say why three LLMs: no conclusion should rest on one model.
- Say how the development set and the sealed set were kept apart.
:::


## First results

_Development results (`dev_v2`), not paper results._

_TO WRITE (15-70 words): What the audience should take from the first results slide; say what the numbers support and what they do not._

| LLM | Row | Configuration | M1 first-pass valid | M2 LLM-final | M6 gap (pts) | M7 calls/part |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | – | no usable run | – | – | – | – |
| gpt-oss-120b | – | no usable run | – | – | – | – |
| gpt-oss-20b | – | no usable run | – | – | – | – |

::: notes
- The two rows per LLM are the first and the deepest row of the grid, chosen by a rule that does not look at the results.
- Read M2 LLM-final, not M2 final: the Route Builder fallback makes M2 final almost always 1.0.
:::


## RQ1: Baseline failures

**How often and in what ways does the v1 LLM planner fail, in terms of precedence violations, missing or invented operations and physical infeasibility, when evaluated against expert-validated plans?**

_Development results (`dev_v2`), not paper results._

_TO WRITE (15-70 words): What the baseline failure rates and failure types show._

| Row | Configuration | LLM | M1 first-pass valid | M2 LLM-final | M6 gap (pts) |
|---|---|---|---|---|---|
| A0 | B1: one LLM call, whole-route replacement | Qwen3.8-27B | – | – | – |
| A0 | B1: one LLM call, whole-route replacement | gpt-oss-120b | – | – | – |
| A0 | B1: one LLM call, whole-route replacement | gpt-oss-20b | – | – | – |
| A1 | B0: five samples, whole-route replacement (v1 default) | Qwen3.8-27B | – | – | – |
| A1 | B0: five samples, whole-route replacement (v1 default) | gpt-oss-120b | – | – | – |
| A1 | B0: five samples, whole-route replacement (v1 default) | gpt-oss-20b | – | – | – |

::: notes
- Say what fails first and how often, in the words of the failure types.
- The share is of raw LLM routes, before any repair or fallback.
:::


## RQ1: Failure types of the raw routes

_Development results (`dev_v2`), not paper results._

_No case studies given._

::: notes
- Name the failure types in the order of the table.
- The cases of the case-study slide are drawn from these types.
:::


## RQ2: Two-step planning and the hint

**Does two-step reason-then-formalise planning reduce first-pass failures, and how much of any gain depends on the rule-based hint?**

_Development results (`dev_v2`), not paper results._

_TO WRITE (15-70 words): Two-step, hint, few-shot: what the paired comparisons show, and whether the LLMs agree._

No paired comparison has data yet (a run of the pair is missing or unusable).

::: notes
- Name the rows that answer this question: A3 vs A2, A4 vs A2, A6 vs A5.
- Read the interval, not the point estimate; say whether the LLMs agree (the last column).
- Say what was not tested.
:::


## RQ3: Localized repair

**Does localized minimal-edit repair preserve more of the LLM's valid decisions than whole-route replacement, without loss of validity?**

_Development results (`dev_v2`), not paper results._

_TO WRITE (15-70 words): Repair against whole-route replacement: validity and how much of the LLM route survives._

No paired comparison has data yet (a run of the pair is missing or unusable).

::: notes
- Name the rows that answer this question: A2 vs A1.
- Read the interval, not the point estimate; say whether the LLMs agree (the last column).
- Say what was not tested.
:::


## RQ3: How much of the LLM route survives

_Development results (`dev_v2`), not paper results._

M5 (share of the LLM's operations that survive the fix; higher = more minimal; – where nothing had to be fixed):

| Row | Qwen3.8-27B | gpt-oss-120b | gpt-oss-20b |
|---|---|---|---|
| A1: B0: five samples, whole-route replacement (v1 default) | – | – | – |
| A2: A1 + localized minimal-edit repair | – | – | – |

::: notes
- M5 is the share of the LLM's operations that is still in the route after the fix: the minimal-edit claim is about this number.
:::


## RQ4: Candidates and selection

**Do multi-candidate generation and consistency-aware selection improve plan quality and stability, and at what cost in calls, tokens and latency?**

_Development results (`dev_v2`), not paper results._

_TO WRITE (15-70 words): Candidates and selection: quality, stability and what they cost._

No paired comparison has data yet (a run of the pair is missing or unusable).

::: notes
- Name the rows that answer this question: A1 vs A0, A5 vs A3.
- Read the interval, not the point estimate; say whether the LLMs agree (the last column).
- Say what was not tested.
:::


## RQ5: Repair or constrain

**Is it better to repair an LLM plan after generation, or to constrain generation step by step with the precedence graph? (stretch)**

_Development results (`dev_v2`), not paper results._

_TO WRITE (15-70 words): Repair after generation or constraint during generation; or say it was not run._

No paired comparison has data yet (a run of the pair is missing or unusable).

::: notes
- Name the rows that answer this question: A9 vs A2.
- Read the interval, not the point estimate; say whether the LLMs agree (the last column).
- Say what was not tested.
:::


## RQ6: Extraction and confirmation

**Do per-feature evidence and agreement-based selective confirmation reduce the number of user corrections per drawing without lowering the accuracy of the confirmed feature list?**

_Development results (`dev_v2`), not paper results._

_TO WRITE (15-70 words): Corrections per drawing; the oracle figure is an upper bound._

_No extraction run given: M9 is not available._

::: notes
- The agreement-based figure is scored after an oracle user: an upper bound, not what a real user does.
- Say how many drawings this rests on.
:::


## Failure case studies

_No case studies given._

_TO WRITE (15-70 words): One case that shows what the aggregate numbers do not._

::: notes
- Show one case in full: the part, the route of the LLM, the violation, what the repair kept.
- Cite the case by its id.
:::


## Limitations

- The grid is on the development set `dev_v2` (synthetic parts): these are not test results.
- M4: no reference route available.
- Cells without a complete, checked run: A0/gpt120, A0/gpt20, A0/qwen, A1/gpt120, A1/gpt20, A1/qwen, A2/gpt120, A2/gpt20, A2/qwen, A3/gpt120, A3/gpt20,…
- Intervals and p-values are not corrected for the number of comparisons.
- The models are hosted and can be retired or change behaviour: the identifiers are pinned in `capp/config.py` and every response is cached, but a re-r…

_TO WRITE (15-70 words): The limits you want the audience to hear from you._

::: notes
- The bullets above are generated from this run; the rest of the list is in the paper draft.
- Say the one limit that matters most for reading the results.
:::


## Reproducibility

- Freeze `2d59d8b0a213` created 2026-10-27T22:46:22+00:00, commit `8b29b0076d`; the working tree now: **drift**.
- Tests at freeze time: 1530 passed, 0 failed.
- Models: large `openai/gpt-oss-120b`, primary `qwen/qwen3.8-27b`, small `openai/gpt-oss-20b`.
- Tables: `python scripts/make_tables.py build <grid>`; draft: `python scripts/paper_draft.py <grid>`.

::: notes
- Say that every figure and table is regenerated by one command from the finished runs.
:::


## Conclusion

_TO WRITE (15-70 words): The answers in a few lines, and the next steps._

::: notes
- Answer each research question in one clause, as far as the data goes.
- End with the next steps, not with the limitations.
:::
