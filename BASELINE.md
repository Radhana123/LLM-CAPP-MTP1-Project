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

## Step 3 (WP0/WP2 prep): dataset v2, material constants

- `capp/archetypes.py`: 43 part archetypes. 19 purane (v1) + 17 naye dev + 7 held-out. Purana dataset (`data/parts_dataset.*`,
  `data/generate_dataset.py`) sirf 19/36 features cover karta tha, 17 features (Bore, T_Slot, Eccentric, Parting_Off, Rib, ...) kisi
  part me the hi nahi. Dev pool ab 36/36 cover karta hai. Purani files baseline ke liye jyon ki tyon rakhi hain.
- `capp/dataset.py`: part schema + `validate_part` + `load_dataset` / `save_dataset` + `summarize`. Har part me ab dims
  (diameter/length/depth/width, UI jaise limits), `required_ra_um` (tolerance se), `machine_preference`, `in_distribution`.
- `capp/datagen.py` + `scripts/make_dataset_v2.py`: fixed seed, deterministic. `data/dev_v2.json|csv` (200 parts, design/tuning ke liye),
  `data/test_v2.json|csv` (120 parts: 60 dev jaise + 60 held-out archetypes, jinke feature-combos dev me kabhi nahi aate).
  **Test set sealed hai: ispe prompts/rules tune nahi karne, sirf final runs (WP11).**
- `capp/materials.py`: kc1.1 + Kienzle mc (Sandvik-style table, machiningdoctor.com), Taylor n (carbide 0.20-0.25), t_ref, tool edge cost.
  Assumed (table me nahi): Plastic ka kc, sab materials ka `t_ref_min` aur `tool_edge_cost_inr`. Thesis me inhe "assumed" likho ya sensitivity chalao.
- Tests: 138 passed (92 + 46 naye). Committed data == regenerated data check bhi hai.

### Dataset ki limits (thesis me honestly likhna)
- Archetypes synthetic hain (maine feature combos socha, expert ne validate nahi kiye). Isliye WP2 ka expert benchmark alag se chahiye.
- Dev me Plastic aur Copper sirf 7-7 parts me hain (kam archetypes unhe use karte hain), in materials ke results pe zyada bharosa mat karna.
- `required_ra_um` tolerance se rule-of-thumb mapping hai (ISO 1302 values), standard nahi. Default feed pe 56/200 dev parts ka theoretical Ra
  required se zyada aata hai (Ra = f^2/(32 r), r = 0.8 mm), yani Ra-risk rule ke liye achhi variety hai.
- Dims uniform random hain, real drawings se nahi.

## Step 4 (WP1): evaluation harness

Code: `capp/verify.py` (violation objects, error taxonomy M3), `capp/metrics.py` (M4, M5, M6 helpers), `capp/systems.py` (RB, B1, B0),
`capp/summary.py` (tables + metric summary), `capp/harness.py` (runner), `scripts/run_eval.py`, `scripts/objective_check.py`.
Tests: 213 passed. B0 ka v1 se match karna test hai (`tests/test_systems.py`: 8 parts x 3 planners pe `run_nsga2` + UI selection ke saath same front, same final route, same counts).

Chalane ka tareeka (repo root):
```
python scripts/run_eval.py --system rb --dataset dev_v2                        # API nahi lagti
python scripts/run_eval.py --system b0 --dataset dev_v2 --limit 20 --tag dev   # pehle chhota run
python scripts/run_eval.py --system b1 --dataset dev_v2 --limit 20 --tag dev   # same tag: B0 ka pehla sample reuse hota hai
python scripts/objective_check.py --dataset dev_v2                             # Time/Cost/Energy ka correlation check
```
Results `results/<tag>_<system>/` me (gitignored): `summary.json`, `per_part.csv`, `per_candidate.csv`, `records.jsonl`, `run.json`.
Test set (`test_v2`) abhi mat chalana, wo sealed hai.

### Baseline (v1) ki weaknesses jo harness ke saath mili (thesis ke liye asli results)
1. **v1 ka "valid" kam check karta hai.** Sirf precedence order + feature coverage. LLM ka invented operation (jaise "Deburring") ya duplicate step
   bhi valid pass ho jata hai. Harness `v1_valid` aur `strict_valid` dono report karta hai.
2. **LLM routes ko changeover ka charge nahi lagta.** Agents sirf route me maujood marker string ("--- Machine Changeover ---") count karte hain.
   Route Builder routes me marker hota hai, LLM routes me kabhi nahi. Dev set ke 44/200 mixed parts pe ye roughly 10 efficiency points
   (+14 min, +43% cost, +0.45 kWh mean) ka free advantage hai. Isliye M6 me har route ko pehle `normalize_markers` se fair banate hain; B0 ka selection v1 jaisa hi rehta hai.
3. **v1 ka whole-route replacement kabhi route nahi dhoondhta.** `fix_sequence_with_builder` sirf 1 route maangta hai. Valve Body jaisi
   8-feature parts pe (dev me 6/200) kuch nahi milta, to B0 safety net pe girta hai. `replacement_failed` me count hota hai.
4. **"NSGA-II" llm_first mode me sirf non-dominated sort + crowding distance hai** (koi crossover, mutation ya generations nahi; population = 5 LLM samples ya unki Route Builder replacements). Route Builder ke variants (mean 13.3 routes/part, 5.5 distinct Time/Cost/Energy triples)
   pe Time-Cost correlation 0.98, Time-Energy 0.91, Cost-Energy 0.88; front ka median size 3, 19.5% parts me front size 1 (`objective_check.py`).
5. **NSGA-II objectives default dims (25/50/20) se banate hain**, part ki asli dims se nahi. Front default dims se banta hai, final selection asli dims se.
   Harness yehi karta hai (`front_uses_dims=False`).

### Harness ke design decisions
- B1 aur B0 ek hi `--tag` pe chalao: B1 ka ek call B0 ka sample 0 hai (same cache entry), extra API call nahi.
- Missing key, replay-mode cache miss, ya 20 API errors lagatar: run ruk jata hai (`STOPPED: ...`). Warna wo chupke se "LLM failed, Route Builder used" ban jata.
- M4 abhi proxy hai (Route Builder variants se similarity, `m4_source=proxy_rb`). Expert `reference_routes` milte hi wo `expert` ban jata hai.
- M9 (extraction) aur M10 (human-human) baad ke steps me.
