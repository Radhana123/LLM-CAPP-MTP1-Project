# Table 4: ablation grid, ablation_dev

Table 4 (ablation grid A0-A9) on the development set: first 20 parts, 3 repeats, three LLMs on Groq. A template: copy it, then fix the choices of the rows marked 'decide' once the rows before them are in. For the final grid use a copy with "dataset": "test_v2" (or bench_v1), "limit": null, "repeats": 5 and run it only for the final results.

Dataset dev_v2 (parts 1-20), 3 repeats by default. A cell is mean ± SD over the repeats; a plain value is one repeat; – = no data.

- M1 first-pass valid: share of the parsed raw LLM routes that pass the strict verifier.
- M2 final valid: share of final routes that are strict-valid (the Route Builder fallback keeps this near 1). M2 LLM-final: share of final routes that are LLM routes, raw or repaired, not the Route Builder's.
- M5 ops kept: share of the LLM's operations that survive replacement or repair (higher = more minimal); – where nothing had to be fixed.
- M6 gap: mean Efficiency of the final route below the best Route Builder variant, in points (lower is better, negative = better).
- M7: API calls per part (all calls) and list-price USD per part (n/a: no price for a model, see capp/config.py).
- M4: no reference route available.

## Qwen3.8-27B (`qwen/qwen3.8-27b`)

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

## gpt-oss-120b (`openai/gpt-oss-120b`)

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

## gpt-oss-20b (`openai/gpt-oss-20b`)

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
