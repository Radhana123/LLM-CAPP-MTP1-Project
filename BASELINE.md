# baseline-v0 (MTP-1, WP0)

LLM-CAPP v1 (internship final code) + reproducibility and test fixes. Ye wo frozen baseline hai jisse
MTP-1 ke har component ka comparison hoga. `git tag baseline-v0` is commit par lagta hai.

## Tests

Repo root se: `python -m pytest`  ->  72 passed (week1 24, week2 11, week3 11, week4 14, week5 12)

## Internship final code se kya badla

1. Tests: week2/test_planner.py aur week4/test_week4.py purane API par the (import error), week3/test_agents.py
   me 2 stale tests the. Teeno current code ke hisaab se update. week4 tests ab LLM ko fake karte hain (koi API call nahi)
   aur llm_first (B0) ka current behaviour pin karte hain.
2. Bug fix: week3/multi_agent_eval.py feature-aware mode crash hota tha (dict ko step-list samajh raha tha).
3. Bug fix: week2/planner.py `process_steps` me poora dict return karta tha, ab step-list.
4. Reproducibility: week1/route_builder.py ab same input par same routes deta hai (pehle random.shuffle + set order
   ki wajah se har run me alag). Naya test: week1/tests/test_route_builder_determinism.py.
5. `mystery_files_diff.txt` hata di. `.gitignore` aur `pytest.ini` add kiye.

## Known issues (abhi fix nahi kiye, MTP-1 me handle honge)

- nsga2.run_nsga2: invalid LLM route ki jagah Route Builder ka route "Route_LLM_n_corrected" naam se aata hai
  (provenance galat). MTP-1 me raw LLM / repaired / Route Builder alag track hoga.
- Route Builder kabhi kabhi steps me marker string "--- Machine Changeover ---" daalta hai (ek valid operation nahi hai).
  Metrics aur repair me ise alag handle karna padega.

## Step 2 (WP0): config, LLM client, cache

- `capp/config.py`: model IDs (`qwen/qwen3.8-27b` etc.), `.env` loading, cache mode, run tag. Model ID ab sirf yahan badalta hai.
- `capp/llm_client.py`: har LLM call yahin se jata hai (planner, extractor). Response cache (`cache/llm/`, gitignored),
  `sample_id` + run tag cache key me (5 samples = 5 alag samples, par har ek reproducible), retries, call log (tokens, latency,
  finish_reason), `CAPP_LLM_MODE=replay` se offline replay.
- `week2/llm_planner.py`, `week1/image_feature_extractor.py`: Groq ko seedha call karne ki jagah `llm_client.chat`. Prompts aur
  settings (T=0.3, max_tokens=200) bilkul v1 jaise, to baseline B0 nahi badla.
- `week4/nsga2.py`: llm_first loop ab `sample_id=attempt` bhejta hai (warna cache 5 attempts ko ek hi jawab deta).
- `scripts/smoke_llm.py`: real API ka chhota test (do baar chalao, doosri baar cached=True aana chahiye).
- Tests: 92 passed (72 + 20 naye). Koi test network use nahi karta.
