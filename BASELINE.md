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

## Step 5 (WP3): violation objects aur R1 localized repair

Code: `capp/repair.py` (R1), `capp/verify.py` (violation objects), `capp/systems.py` (`b0_r1`, `b1_r1`), `capp/compare.py` + `scripts/compare_runs.py`.
Tests: 243 passed (naye: `tests/test_repair.py`, `tests/test_compare.py`, plus additions in test_verify/test_systems/test_harness).

### Kya naya hai
- **Violation objects:** har violation me ab `rule_id` (`ORD-FIRST`, `ORD-LAST`, `ORD-EDGE:a>b`, `COV:<feature>`, `SET-UNKNOWN`, `SET-DUP`, `MACH-GROUP`)
  aur `positions` (route me index) hain. `records.jsonl` me ye sab dikhta hai.
- **R1 (report 6.3):** LLM route ko pheko mat, minimum edit se theek karo.
  1. *Set repair:* unknown aur repeated ops hatao (pehla occurrence rehta hai), missing ops insert karo (Facing, Inspection, aur jis feature ki koi chain poori nahi, uski sabse chhoti chain).
     Valid extra ops delete nahi hote.
  2. *Order repair:* precedence rules ke saath saare valid orderings me se wo jo LLM ke order se sabse kam discordant pairs (Kendall-tau) pe ho; tie me kam changeovers; phir canonical order (sirf determinism ke liye).
     Dynamic programming hai, exact hai (test brute force se match karta hai), 15-op routes pe ~0.02 s.
  3. Result strict verifier se check hota hai. R1 fail ho (precedence conflict ya search bahut bada) to R0 fallback, to validity guarantee v1 jaisa hi rehta hai.
- **Systems:** `b0_r1` / `b1_r1` = B0 / B1 jaisa hi sab kuch, bas repair R1 hai. R1 *strict* criterion pe trigger hota hai (invented/duplicate ops bhi), jabki v1 (`b0`, `b1`) sirf v1-invalid ko replace karta hai.
  Provenance label: `repaired_llm`. `b0` aur `b1` bilkul pehle jaise (faithful) hain.
- **M5** ab R0 aur R1 ke liye alag report hota hai (edit distance, LLM ke kitne ops bache, discordant pairs, inserted/removed ops).

### Chalane ka tareeka (same tag => B0 ke samples cache se, API nahi lagti)
```
python scripts/run_eval.py --system b0_r1 --dataset dev_v2 --limit 20 --tag dev
python scripts/run_eval.py --system b1_r1 --dataset dev_v2 --limit 20 --tag dev
python scripts/compare_runs.py results/dev_rb results/dev_b1 results/dev_b0 results/dev_b0_r1
```
`compare_runs.py` table banata hai, pehle folder (rb) ke saath paired Efficiency comparison deta hai, aur R0 vs R1 ko *wahi raw LLM routes* pe compare karta hai.

### Abhi kya nahi hai
R2 (LLM re-prompt with violation report) aur R3 (hybrid) agle step me. Selection abhi bhi v1 jaisa hai (as-is scores), fair-scoring wala selection Step 7 (agents me changeover) ke saath aayega.
Offline simulation (synthetic planner) se sirf pipeline check hua, asli numbers tumhare live cache se aayenge.

## Step 6 (WP3): R2 (LLM re-prompt) aur R3 (hybrid)

Code: `capp/reprompt.py` (R2), `capp/systems.py` (`b0_r2`, `b1_r2`, `b0_r3`, `b1_r3`), `capp/harness.py` (repair planner guard), `capp/summary.py` + `capp/compare.py` (R2 numbers), `capp/config.py` (`REPAIR_TEMPERATURE`).
Tests: 266 passed (naya: `tests/test_reprompt.py`, plus additions in test_systems/test_harness).

### Kya naya hai
- **R2:** invalid route ko LLM ko wapas bhejte hain, violation report ke saath. Conversation: v1 ka system prompt + v1 ka user prompt (byte-for-byte same, test se check) →
  LLM ka rejected plan (assistant) → ek naya user message jisme saari *hard* violations ki list hai ("'Drilling' must come before 'Reaming'", "feature 'Hole' is not covered (absent: Drilling)", ...)
  aur "sirf utna badlo jitna zaruri hai". Temperature 0.0 (repair reproducible rahe). Jawab strict verifier se pass hona chahiye, warna R2 fail.
- **R3 (report 6.3):** pehle R2; R2 fail ho to R1 *original* LLM route pe; sab fail ho to R0 last resort. R2/R3 ke `repair` record me `tried`, `stage` (kis stage ne route diya), `r1 {...}`, `r2 {...}` hote hain.
- **Systems:** `b0_r2`, `b1_r2`, `b0_r3`, `b1_r3`. Baaki sab (dedupe, safety net, selection) B0/B1 jaisa; fark sirf repair strategy ka.
- **Cost (M7):** R2 extra LLM calls hain. Summary me ab `repair_calls_per_part`, repair tokens, latency aur cache-hit alag dikhte hain. `calls/part` sirf samples ki ginti hai.
- **Cache:** same `--tag` pe B0 ke samples aur R2 ke repair calls cache se aate hain; `b1_r2` ka sample 0 aur uska repair `b0_r2` jaisa hi cache entry hai. `b0_r3` R2 ke cached repairs reuse karta hai.
- **Guards:** missing key, replay-mode cache miss ya lagatar 20 API errors R2 calls pe bhi run rok dete hain (`STOPPED: ...`).
- **M5** ab stage ke hisaab se: R1 stage aur R2 stage ka edit distance aur LLM ops surviving alag.

### Chalane ka tareeka (same `--tag dev`; R2/R3 ke naye Groq calls sirf invalid samples ke liye, 20 parts pe lagbhag 15 calls)
```
python scripts/run_eval.py --system b0_r2 --dataset dev_v2 --limit 20 --tag dev
python scripts/run_eval.py --system b0_r3 --dataset dev_v2 --limit 20 --tag dev
python scripts/compare_runs.py results/dev_rb results/dev_b0 results/dev_b0_r1 results/dev_b0_r2 results/dev_b0_r3
```
Rate limit lage to `--throttle 2` jodo; cache ki wajah se jo calls ho chuki hain wo dobara nahi lagengi.

### Abhi kya nahi hai
Retrieval few-shot, multi-candidate selection strategies (S0-S3), physics agents (kc, Taylor, Ra), extended verifier, UI. R2 ko sirf ek round milta hai (report: single re-prompt).

## Step 7 (WP4): selection S0-S3, plan confidence, aur ek reproducibility fix

Code: `capp/systems.py` (selection, confidence, system names), `capp/harness.py` (`weights`), `capp/summary.py` + `capp/compare.py` (confidence numbers), `scripts/run_eval.py` (`--weights`),
`scripts/objective_check.py` (tie numbers), `week1/route_builder.py` (fix). Tests: 289 passed (naye: `tests/test_selection.py`, `tests/test_determinism.py`).

### 1. Reproducibility fix (zaroori)
Route Builder ke `_classify_and_assign` me shared ops pe `for op in list(shared_ops)` string-set ke hash order me chalta tha, aur har op ka machine assignment is pe depend karta hai
ki dono machines pe abhi kitne ops hain. Is wajah se ek 7-feature archetype (Turning, Step, Groove, Thread, Keyway, Chamfer, Undercut) pe kabhi 15 kabhi 12 routes milte the,
process-to-process (`PYTHONHASHSEED`). Fix: `sorted(shared_ops)` (aur warnings bhi sorted). `tests/test_determinism.py` ab 4 alag hash seeds pe saare 43 archetypes ke RB routes,
prefer_lathe routes, warnings, R0 replacement aur R1 repair compare karta hai; purane code pe ye test fail hota hai, fix ke saath pass.
**Asar:** RB baseline (first route) same hai: efficiency 64.46, gap 1.06 (har seed pe check kiya). Sirf us ek archetype ke *variants* badle, isliye objective_check ke numbers thode shift hue:
routes/part 13.275 -> 13.185, distinct triples 5.54 -> 5.51, front size mean 2.97 -> 3.00, correlations 0.979/0.908/0.881 -> 0.979/0.907/0.877.
Pehle wale numbers hash seed pe depend karte the; ab har run me yehi aayenge.

### 2. Selection (report 6.4)
Selection ab parameter hai (`s0`-`s3`), default `s1` = v1, to `b0`/`b1` waise hi rehte hain.
- **S0** pehla valid candidate (sample order). **S1** v1: pehla non-dominated front, usme se sabse zyada Efficiency.
- **S2** pehla front, phir time/cost/energy ka weighted scalarisation (front ke andar min-max scale; `--weights 0.6,0.2,0.2`, default barabar).
- **S3** pehla front, phir *medoid*: wo route jiska doosre samples ke routes se mean distance sabse kam ho. Duplicates gine jaate hain (3 samples ka same route = 3 votes), dedupe se votes nahi ghatte.
  Tie me zyada Efficiency, phir pehla.
- **Distance** (`route_distance`): Kendall-tau distance operations ke *union* pe. Dono routes me maujood ops ki jodi ulti ho to 1; jis jodi me koi op ek route me nahi hai wo 0.5 (neutral, Fagin ka K^(p)).
  Same ops wale routes pe ye plain tau hai. Sirf common ops pe tau lene se alag alternative chain wale routes "agree" dikhte the, ye test se pakda gaya.
- **Plan confidence** = sabhi samples ke (repair ke baad) routes ka mean pairwise agreement (1 - distance). 0.8 se kam => human review flag (`CONFIDENCE_FLAG`, abhi tune nahi hua; dev set pe karenge).
  Har run me record hota hai, chahe selection kuch bhi ho. Summary me: mean, flagged share, flagged vs unflagged ka gap, Spearman(confidence, gap).
- **System names:** `n<N>_r<0-3>_s<0-3>`, N = 1..20 samples. `n5_r0_s1` = b0, `n1_r0_s1` = b1. Purane naam (b0, b1_r2, ...) chalte rahenge.

### 3. Naya finding: front ke members zyadatar exact ties hain
`objective_check.py` ab front ke andar ties bhi ginta hai (dev_v2, Route Builder variants):
- front ke members me se **61.3%** ka (time, cost, energy) kisi aur member jaisa bilkul same hai (sirf operation order alag);
- **77.5%** parts me front me ties hain; **84.0%** parts me poora front ek hi (time, cost, energy) triple hai (front me distinct triples ka mean 1.16).
Matlab 84% parts me "multi-objective selection" ke liye koi trade-off hai hi nahi: v1 ka Efficiency selection un order-variants me se bas pehla utha leta hai, jo arbitrary hai.
S3 (medoid) is tie ko ek principled tareeke se todta hai (jo order LLM samples me sabse common hai). Ye RB variants pe nikala gaya hai, LLM samples pe nahi; live runs me S1 vs S3 ka fark dikhega.

### Chalane ka tareeka (same `--tag dev`)
```
python scripts/objective_check.py --dataset dev_v2                                     # API nahi
python scripts/run_eval.py --system n5_r1_s3 --dataset dev_v2 --limit 20 --tag dev     # B0 ke samples cache se, R1, medoid; API nahi
python scripts/run_eval.py --system n5_r1_s2 --weights 0.6,0.2,0.2 --dataset dev_v2 --limit 20 --tag dev
python scripts/run_eval.py --system n3_r1_s3 --dataset dev_v2 --limit 20 --tag dev     # sirf pehle 3 samples, cache se
python scripts/run_eval.py --system n8_r1_s3 --dataset dev_v2 --limit 20 --tag dev     # samples 5,6,7 naye: lagbhag 60 Groq calls
python scripts/compare_runs.py results/dev_rb results/dev_b0 results/dev_n5_r1_s1 results/dev_n5_r1_s3
```

### Abhi kya nahi hai
N candidates ki diversity abhi sirf T=0.3 se aati hai (samples lagbhag same dikhte hain). Report ke hisaab se diversity reasoning paths (hint ke saath/bina) aur retrieved examples se aani hai: agla step (retrieval few-shot).
Confidence threshold tuning, model cascade (optional), physics agents, extended verifier, UI.

## Step 8 (WP5): retrieval few-shot (k = 0/2/4) aur Route Builder hint (H0/H1)

Code: naya `capp/plannerspec.py` (naam parsing), naya `capp/planners.py` (pool, retrieval, prompt), `capp/systems.py` (`parse_system` ab `+...` suffix leta hai, `planner_spec`),
`capp/harness.py` (`pool=`, `planner_factory=`, run.json me planner options), `capp/summary.py` + `capp/compare.py` (prompt numbers), `scripts/run_eval.py` (`--pool`).
Tests: 340 passed (naya: `tests/test_planners.py`, 51 tests).

### Kya hai
Koi naya LLM call nahi, sirf prompt badalta hai. Kisi bhi LLM system ke naam ke baad `+` se options lagao:
- `+fs2`, `+fs4`: k retrieved examples. Pool (default `dev_v2`, kabhi test nahi) me se wo parts jinke feature sets query part se sabse zyada milte hain
  (Jaccard overlap, tie me same material, phir same material-class, phir part_id). Har *alag feature set* ek candidate hai, aur jinka plan pehle chuni example jaisa hi ho wo hata diye jaate hain,
  isliye k examples k alag problems hote hain, ek hi problem ki k copies nahi. Part khud kabhi example nahi banta.
- `+fs2x`, `+fs4x`: upar se, jis feature set ka query part hai wo feature set wale *saare* parts bhi nikal diye jaate hain (naya combination, jo pool ne dekha hi nahi).
- `+h1`: Route Builder ka pehla route prompt me "A rule-based planner proposes this route: [...]" ke roop me. `+h0` (default) = hint nahi.
- `+fs0+h0` / koi suffix nahi = bilkul v1 prompt (byte-for-byte, test se pakka), isliye uske LLM answers cache se milte hain. Naya prompt = naya cache key = naye Groq calls.
- Examples chat ke pehle turns hote hain (user: v1 wala prompt, assistant: JSON plan), sabse kam similar pehle, sabse similar sawaal ke bilkul pehle. R2 re-prompt prompts nahi badalte (examples/hint nahi jaate).
- Pool ke har part ka "validated plan": expert `reference_routes[0]` agar hai (WP2 ke baad), warna Route Builder ka pehla route (strict verifier se check hota hai, dev ke 200 me se 200 valid).
- Prompt chhota rehta hai: system ~1200 chars, fs4 + hint ~2800 chars total, matlab call ke hisaab se kuch sau tokens ka fark.
- Record me har candidate ke `meta.prompt_context` me example part_ids aur hint hai; `per_candidate.csv` me `examples`, `hint` columns; summary me "planner prompt" line; `compare_runs.py` me "retrieved examples / call" aur "prompt tokens / part".

### Thesis ke liye dhyan rakhne wali baatein (honest)
1. **Examples ke plans Route Builder ke hain** (jab tak expert routes nahi). Isliye few-shot LLM ko RB jaisa banata hai; ye "LLM ne domain seekha" ka saboot nahi, "RB ka style copy kiya" ka hai. WP2 ke expert routes aane par pool unse banega.
2. **dev_v2 me 200 parts, sirf 36 distinct feature sets.** `+fs2` me aksar ek *bilkul same problem* ka solved example mil jaata hai (Jaccard 1.0), yani answer lagbhag leak. Ye "seen archetype" ka upper bound hai.
   Generalisation ke liye `+fs2x` / `+fs4x` report karo (same feature set nikal ke). Dono ko saath dikhana sahi rahega.
3. **`+h1` me LLM RB ka route dekhta hai**, to efficiency RB ke paas aani chahiye. Ye ablation hai (hint kitna kaam karta hai), final system nahi.
4. Ye sab dev set pe hai. Test_v2 pe sirf final run me, pool tab bhi dev hi rahega (`--pool test_v2` jaan-boojh ke refuse hota hai).
5. Sirf prompt badla hai, temperature 0.3 wahi. Diversity ka asar dekhne ke liye n5/n8 ke saath chalana.

### Chalane ka tareeka (naye Groq calls: 20 parts x 5 samples = lagbhag 100 calls har naye prompt ke liye; `--throttle 2` rate limit ke liye)
```
python scripts/run_eval.py --system n5_r1_s3+fs2x --dataset dev_v2 --limit 20 --tag dev
python scripts/run_eval.py --system n5_r1_s3+fs4x --dataset dev_v2 --limit 20 --tag dev
python scripts/run_eval.py --system n5_r1_s3+fs2  --dataset dev_v2 --limit 20 --tag dev
python scripts/run_eval.py --system n5_r1_s3+h1   --dataset dev_v2 --limit 20 --tag dev
python scripts/compare_runs.py results/dev_rb results/dev_n5_r1_s3 results/dev_n5_r1_s3+fs2x results/dev_n5_r1_s3+fs4x results/dev_n5_r1_s3+fs2 results/dev_n5_r1_s3+h1
```
Same tag + same prompt options => alag repair/selection (r1/r3, s1/s3) ke saath dobara chalane par samples cache se aate hain, API nahi lagti. Pehle ek hi variant (`n5_r1_s3+fs2x`) chalao aur dekho.

### Abhi kya nahi hai
Two-step planner (pehle reasoning, phir formalisation, structured output) agla step hai. Uske baad physics agents (kc, Taylor, Ra), extended verifier, UI, expert benchmark.

## Step 9 (WP4): two-step planner (reasoning call, phir structured-output formalisation), option `+ts`

Code: naya `capp/twostep.py`, naya `scripts/show_plan.py`; `capp/plannerspec.py` (`+ts`), `capp/planners.py` (dispatch), `capp/config.py` (constants, `CAPP_STRUCTURED`),
`capp/summary.py` + `capp/compare.py` + `capp/harness.py` (API calls, two-step numbers, run.json), `scripts/run_eval.py` (docs). Tests: `tests/test_twostep.py` (50).

### Kya hai (report 6.2)
Kisi bhi LLM system ke naam me `+ts` lagao (`n5_r1_s3+ts`, `b0+ts+h1`, `n5_r1_s3+fs2x+h1+ts`; options saath me chalte hain).
- **Call 1, reasoning:** LLM ko part + closed operation list milti hai; wo chhota plain-text plan likhta hai (har op kis feature ke liye, har ordering decision kis precedence rule pe),
  aur aakhri line `FINAL ORDER: a > b > c`. v1 ke rules 1-3 wahi hain (Facing pehle, Inspection aakhir me, closed list, precedence); JSON nahi. Sampled (T=0.3), isliye N samples = N reasoning paths.
  `+fs` ke examples aur `+h1` ka hint isi call ke prompt me text ki tarah jaate hain.
- **Call 2, formalisation:** reasoning text ko operation list me badalta hai, greedy (T=0), **structured output** me: JSON schema jisme operation names closed vocabulary ka `enum` hain.
  Prompt kehta hai "stated order copy karo, correct mat karo", taaki reasoning ki galti verifier ko dikhe, chhup na jaye.
- **Fallback chain:** `json_schema` -> platform/model mana kare to `json_object` -> `none` (plain text). Kaunsa chala, har sample me record hota hai (`meta.structured_used`), summary me
  "two-step" line aur `per_candidate.csv` ke `structured`, `parser` columns me. v1 ka multi-strategy parser sirf aakhri fallback hai (`parser = fallback`). `CAPP_STRUCTURED=json_object` se pehla mode badal sakte ho.
- **Reasoning text sample ke saath store hota hai** (`records.jsonl` -> `candidates[].meta.reasoning`), saath me `reasoning_order` (reasoning khud jo order kehti hai) aur `faithful`
  (step 2 ne wahi order diya ya nahi). `python scripts/show_plan.py results/dev_b1+ts DEV_007` ek part ka poora sach dikhata hai: reasoning, LLM route, verifier ka reason, repair, final.
- **Cost:** har sample ke 2 API calls. `n5` = part ke 10 calls. M7 me ab "samples/part" aur "API calls/part (incl. 2nd step, repair)" alag dikhte hain; tokens aur latency dono calls ka sum hai.
- R2 re-prompt (agar r2/r3 chalao) abhi bhi one-step conversation hai (v1 prompt + rejected plan + feedback), ts me nahi badla.

### Do cheezein jo thesis me honestly likhni hain
1. **Structured output closed vocabulary force karta hai**, to "invented operation" errors ts me *construction se* gayab ho jaate hain. Agar ts ka M1 validity v1 se behtar nikle, to pehle M3 (failure kinds) dekho:
   kitna fark vocabulary se aaya aur kitna order/missing me reasoning se. RQ2 ke liye sahi control "one-step + structured output, bina reasoning" hoga; agar zaroorat padi to agla chhota step.
2. **v1 prompt ki valid-ops list me "Inspection" hai hi nahi** (36 ops), jabki rule 1 use maangta hai; kuch canonical ops (Form Milling, Forming, Gang Milling, Polishing/Burnishing) list me nahi hain.
   v1 prompt ko baseline ke liye waise hi rakha hai. ts ke prompt/enum me list + "Inspection" hai (37), warna schema me Inspection aa hi nahi sakta aur har plan "missing" hota.
3. **Live API pe structured output abhi test nahi hua** (qwen3.8-27b pe json_schema chalta hai ya nahi, yahan se nahi dikhta). Isliye chain hai aur record hota hai; pehle run ka "two-step: structured output used {...}" dekho.

### Chalane ka tareeka (naye Groq calls; pehle chhota)
```
python scripts/run_eval.py --system b1+ts --dataset dev_v2 --limit 5 --tag dev          # smoke test: 10 calls
python scripts/show_plan.py results/dev_b1+ts DEV_001                                    # reasoning + verifier + final
python scripts/run_eval.py --system n5_r1_s3+ts --dataset dev_v2 --limit 20 --tag dev   # 200 calls (b1+ts ka sample 0 cache se)
python scripts/run_eval.py --system n5_r1_s3+ts+h1 --dataset dev_v2 --limit 20 --tag dev
python scripts/compare_runs.py results/dev_rb results/dev_n5_r1_s3 results/dev_n5_r1_s3+ts results/dev_n5_r1_s3+ts+h1
```
Same tag + same options => dobara chalane par sab cache se (API nahi). `--throttle 2` rate limit ke liye.

### Abhi kya nahi hai
Physics agents (kc, Taylor wear, Ra), extended verifier (15-20 physical rules), UI ke dono modes, extraction front end, expert benchmark (WP2), model cascade, confidence threshold tuning.

## Step 10 (WP6 prep): physics agents, kc cutting energy, Taylor tool-wear cost, Ra risk

Code: naya `capp/physics.py`, naya `scripts/physics_check.py`; `capp/harness.py` (har final route ke saath analysis), `capp/summary.py` ("physics" line + `per_part.csv` columns),
`capp/compare.py` (4 rows), `scripts/run_eval.py` (`--no-analysis`). Tests: `tests/test_physics.py` (49).

### Kya hai (report: "Route + Full Analysis" mode ke teen numbers)
Ye **analysis layer** hai. Route select karne wale objectives (v1 Time/Cost/Energy/Efficiency) **nahi badle**, isliye B0/B1/RB aur ab tak ke saare numbers wahi hain. Physics sirf final route ko score/explain karta hai.
Time agent ke speeds, feeds, passes aur machining time reuse hote hain, to har number Time agent se consistent hai.
- **Specific cutting energy kc:** Kienzle `kc = kc1.1 * h^-mc` (`capp/materials.py` ki table), chip thickness h feed aur tool geometry se (turning h = f, drilling h = f/2 sin(59 deg), milling h = fz sqrt(ae/D)...).
  `1 N/mm2 = 1 mJ/mm3`, yani kc hi specific cutting energy hai (steel ~2 J/mm3). Phir MRR Q, power `Pc = kc*Q/6e7` kW, spindle power `Pc/0.8`, utilisation vs 7.5 kW, energy `E = Pc*t/60` kWh.
- **Taylor tool wear:** `V*T^n = C`; handbook speed pe tool life = `t_ref` (materials.py), wear fraction `t/T`, wear cost = fraction x ek cutting edge ki cost. Batch ke liye edges/batch bhi.
- **Ra risk:** best case `f^2/(32 r)` (turning, boring, milling), process floor ke saath (`RA_FLOOR_UM`); route ka sabse fine process required Ra se compare hota hai: ratio <= 0.5 low, <= 1.0 medium, >1.0 high.
  Saath me `max_turning_feed_mm_rev` (required Ra ke liye kitna feed chalega) milta hai: UI me "feed itna kam karo" ka seedha input.
- **Speed what-if:** `physics.analyse_route(..., vc_factor=1.2)`: cut ka time 1/k, power k guna, energy same, wear per part `k^(1/n - 1)` guna (Taylor).
- Har record me `final.physics` (totals + per-op short rows); `python scripts/run_eval.py --system rb --limit 20` ki summary me ek "physics" line; `--no-analysis` se band.
  Pura per-op detail: `physics.analyse_part_route(steps, part)`.

### Pehla look: `python scripts/physics_check.py --dataset dev_v2` (offline, API nahi), 200 dev parts, RB routes
- Coverage: 18262 operation instances, koi unknown nahi; forming ops (Knurling, Forming, Polishing) aur Inspection me cutting energy/wear nahi (time rehta hai).
- **Cutting energy 0.49 kWh/part, kc 2051 N/mm2 (mean), tool wear 95 INR/part** (72 edges/batch). Material ke hisaab se wear: Titanium 367, Cast Iron 106, Steel 93, Aluminum 9 INR/part. Ordering sahi hai (Ti, Steel, Cast Iron, Al, Plastic).
- **v1 Energy agent bahut alag cheez hai.** `energy_agent` = har op ka fixed constant (`_BASE_ENERGY`) + setup events, material factor se; **part ke size pe depend hi nahi karta**.
  Uska 84% hissa per-op constants ka hai. Isliye v1 2.85 kWh vs physics 0.49 kWh (ratio of means x5.8, median x14.5) ek-ek compare nahi hota: v1 me setup/non-cutting energy bhi hai, physics me sirf cut. Ye mat likhna ki "v1 galat hai".
  Pooled Spearman 0.41 (parts ke beech v1 size nahi dekhta), par ek hi part ke route variants me 0.81: variants ko rank karne me dono kaafi milte hain, to **selection me physics energy se bahut fark nahi padega**; fark dikhane wali cheez size-dependence aur wear hai.
- Wear vs v1 cost: pooled 0.94, within-part 0.89. Matlab wear zyada tar machining time ko follow karta hai; Taylor ka naya information material (Ti 4x steel) aur speed sensitivity me hai, route ordering me nahi.
- **Machine power:** 23% parts me koi op 7.5 kW se upar jaata hai (Drilling 40, Counterboring 13, Facing 10, Spotfacing 4, Countersinking 3). **Ye zyada tar dataset ki limit hai, finding nahi:** dataset me hole diameter nahi hai, Time agent ki tarah
  hole ops part ka diameter use karte hain, to 149 mm ke part me 149 mm ka drill lagta hai (over-power parts ka median diameter 149 mm, baaki 28 mm). Facing ke 10 cases asli hain (v1 speed/feed, bada depth).
  Isliye is number ko report me "dataset me hole diameter aane par dobara" ke saath hi likhna.
- **Ra risk** (best predicted Ra / required Ra): required 0.4 um wale 17 parts me koi low nahi (6 medium, 11 high); 0.8 um me 24 medium, 3 high, 6 low; 3.2 um me 54 low, 3 medium, 1 high; 6.3 aur 12.5 um me lagbhag sab low. Yani v1 ka ek hi feed (steel 0.15 mm/rev) fine finish ke liye roughing feed hai: physics bolta hai "finish pass ka feed alag chahiye" (v1 me parameters route ke baad aate hain, report 6.7).
- **Speed sensitivity** (Vc x0.8 / 1.0 / 1.2): wear 45 / 95 / 178 INR/part, over-power parts 12% / 23% / 35%, cutting energy 0.489 kWh teeno me (volume same).

### Jo honestly likhna hai
1. **Ye assumptions hain, measurements nahi:** `ASSUMPTIONS` in `capp/physics.py` (machine power 7.5 kW, efficiency 0.8, insert nose radius 0.8/0.4, plunge tool widths, boring/reaming stock, `RA_FLOOR_UM`, mill engagement) aur materials.py ke `t_ref`, `tool_edge_cost_inr` (sab materials ke liye), Plastic ka kc. Absolute INR quote mat karo; comparisons aur sensitivity quote karo. `RA_FLOOR_UM` ke liye koi achha handbook source abhi nahi mila, quote karne se pehle check karna.
2. **Energy = sirf cutting ka energy**, lower bound. Idle power, spindle acceleration, coolant, auxiliaries nahi hain; real machine ka electrical energy kaafi zyada hota hai.
3. **Ra risk optimistic hai:** maanta hai ki route ka sabse fine process hi critical surface banata hai (dataset me ye mapping nahi hai). "Low" ka matlab "route ke parameters ye finish de sakte hain", guarantee nahi; "high" pakka signal hai.
4. Wear model me tool edge, coolant, coating, BUE nahi hai; Taylor n carbide range (0.2-0.25) ka assumption hai.
5. Time agent ka machining time ek hi tarah ka formula hai (per-operation), physics usi pe chalta hai: agar Time agent galat hai to physics ka energy aur wear bhi proportionally galat hoga.

### Chalane ka tareeka
```
python scripts/physics_check.py --dataset dev_v2 --json results/physics_check.json
python scripts/run_eval.py --system rb --dataset dev_v2 --tag dev                  # summary me "physics" line
python scripts/run_eval.py --system rb --dataset dev_v2 --tag dev --no-analysis
```

### Abhi kya nahi hai
Extended verifier (15-20 physical-feasibility rules; power limit aur Ra yahan Violation objects ban sakte hain taaki R1 repair ya flag kare), UI ke dono modes, extraction front end, expert benchmark (WP2), model cascade, confidence threshold tuning.

## Step 11 (WP6): extended verifier, 15 physical-feasibility rules

Code: naya `capp/feasibility.py`, naya `scripts/feasibility_check.py`; `capp/verify.py` (naya soft kind `risk`, Violation me `action` field), `capp/harness.py` (final aur raw LLM route ka check),
`capp/summary.py` + `capp/compare.py` (feasibility numbers). Tests: `tests/test_feasibility.py` (47).

### Kya hai (report 6.6)
Precedence-valid route bhi physically galat ho sakta hai (hole hi nahi hai jispe reaming ho, part machine me fit nahi hota, slender part bina support). Ye rules wahi pakadte hain.
Har finding ek `verify.Violation` hai (rule id `PHYS:<RULE>`, positions, aur `action` jaise `insert:Reaming`, `support:tailstock`, `reduce:feed to 0.098 mm/rev`), yani precedence violations jaisa format.
- **Do kinds:** `infeasible` (hard: route as planned ho hi nahi sakta) aur `risk` (soft: support/feed/finishing chahiye ya assumption pe tika hai; route ko invalid nahi karta).
- **Scope:** `route` rules route judge karte hain; `part` rule (TOL-SPEC) specification judge karta hai, route se independent, isliye part me ek baar gina jata hai.
- **Dims chahiye wale rules sirf tab chalte hain jab dims hon** (report 6.6). Data na ho to rule `not_evaluated` me jaata hai (reason ke saath), chupchap pass nahi hota.
- **Analysis-only:** M2 validity, repair, selection, baselines (B0/B1/RB) **nahi badle**. `verify.PHYSICAL_RULES` abhi bhi khaali hai. Agar kabhi strict check me jodna ho: `feasibility.install()`
  (tab `infeasible` route ko strict-invalid karega, `risk` nahi); test me dikhaya hai.
- Har final route aur har raw LLM route ke saath `feasibility` store hota hai (`--no-analysis` se band). Summary me "feasibility" lines, `per_part.csv` me `feas_*` columns, `per_candidate.csv` me `raw_feas_*`.
  **Report ka promised number:** "raw LLM routes: strict-valid but physically infeasible" (summary line 2) = precedence-valid par physically infeasible routes ka share.

### Rules (id, kind, source status)
| Category | Rule | Kind | Status |
|---|---|---|---|
| capability | CAP-LATHE-DIA, CAP-LATHE-LEN, CAP-MILL-ENV (part machine envelope me fit) | infeasible | assumption (Machine defaults) |
| capability | CAP-POWER (spindle power > machine power; hole ops tab tak skip jab tak `hole_diameter` na ho) | risk | model |
| surface | RA-FEED (standard feed pe predicted Ra > required; message me kitna feed chalega) | risk | model |
| surface | RA-FLOOR (required Ra route ke finest cutting process ke floor se neeche) | risk | assumption |
| workholding | HOLD-TAILSTOCK (L/D 3-6), HOLD-STEADY (6-20, 12 ke upar reduced parameters), HOLD-SLENDER-MAX (>20) | risk | web (CADEM) |
| hole | HOLE-DEEP (depth > 3 x D peck, > 5 x D multi-peck; `hole_diameter` chahiye) | risk | web (Mt Hood CC) |
| hole | HOLE-EXISTS (Reaming/Tapping/Thread Milling/Counterboring/Countersinking/Internal Grooving se pehle Drilling ya Boring) | infeasible | derived |
| hole | HOLE-PILOT (Drilling se pehle Center Drilling) | risk | web (CIMCO) |
| finishing | TOL-HOLE (hole tolerance drilling IT10 / boring IT8 se fine ho to Reaming IT6) | risk | web (ISO 286, Engineering ToolBox) |
| finishing | TOL-SPEC (tolerance vocabulary ke best process se bhi fine: turned diameter IT7, hole IT6; grinding/honing/lapping vocabulary me nahi) | risk, part | web |
| sequence | SEQ-PARTOFF (Parting-off ke baad koi lathe op nahi) | infeasible | derived |

### Dev set pe pehla look: `python scripts/feasibility_check.py --dataset dev_v2` (offline), 200 parts, RB routes
- **RB ke first route me koi hard finding nahi.** Risks: TOL-HOLE 47 parts (23%), HOLD-TAILSTOCK 27, HOLD-STEADY 24, TOL-SPEC 24 (12%), RA-FEED 15, RA-FLOOR 14, CAP-POWER 10, HOLD-SLENDER-MAX 1. 51 parts (25%) ko tailstock/steady rest chahiye.
- **Rule base ka asli defect mila:** HOLE-EXISTS RB ke 2637 variants me se **60 (2.3%)** pe fire karta hai, hamesha `Internal Grooving` Drilling se *pehle*. Precedence graph me `Drilling -> Internal Grooving` edge hai hi nahi. First route me ye nahi aata (isliye RB baseline sahi dikhta hai),
  par variants (M4/M6 ke references, R0 fallback) me aata hai. Baselines frozen hain, to maine graph nahi badla; ek edge jodna agle kisi step ka candidate hai (decision tumhara/supervisor ka).
- **TOL-SPEC dataset ka finding hai:** tolerance 0.005 mm wale 10/17 aur 0.01 mm wale 14/33 parts (jinme turned diameter ya hole feature hai) ki tolerance vocabulary ke kisi operation se hold nahi hoti (grinding/honing/lapping nahi hain). Tolerance ko +-T padha hai (band 2T); band padhte to aur zyada parts aate.
- **HOLE-DEEP 61% parts me evaluate nahi hota** (Drilling hai par `hole_diameter` nahi). CAP-POWER bhi isi wajah se hole ops skip karta hai: isliye feasibility me 5% parts over-power dikhte hain, physics line ke 23% nahi (wo hole = part diameter wala artefact tha).
  Benchmark parts aur extraction front end ko hole diameter record karna chahiye.
- CAP-LATHE-*/CAP-MILL-ENV dev par kabhi fire nahi hote (machine envelope assumed, parts chhote); unit tests se check hue hain.

### Jo honestly likhna hai
1. **Report kehti hai "har rule ka handbook citation". Mere paas handbook nahi tha, isliye koi rule "handbook" status ka nahi hai.** Status: web 7, derived 2, model 2, assumption 4 (`feasibility.rule_table()`). Har rule me jo source maine padha wahi likha hai (URL ke saath):
   CADEM blog (L/D thumb rule, 2025), Mt Hood CC CNC course (3 x D / 5 x D), CIMCO CNC handbook ch. 7, Wikipedia "IT Grade" (ISO 286 table), Engineering ToolBox (process -> IT grade, ANSI B4.1 cite karta hai).
   **Tumhe / supervisor ko har web rule ke liye handbook page (Kalpakjian, Groover, Machinery's Handbook, HMT jo bhi) lagana hai**; tab status "handbook" hoga. Report ka "pehle 10 rules jinka handbook basis sabse clear" abhi 9 (web + derived) hain.
2. **15 rules implement hue, par ye sab source-backed nahi:** 4 assumption pe hain (machine envelope x3, Ra floor). Jin rules ka source nahi mila wo **encode nahi kiye** (`feasibility.NOT_ENCODED`): minimum chuck grip, boring bar overhang, thin walls, end-mill L/D, material-wise operation bans.
   Ra floors ke liye web tables (turning 0.05-25 um jaise) itne wide hain ki floor tay nahi hota; `physics.RA_FLOOR_UM` assumption hi hai.
3. **Support rules route edit se theek nahi hote:** vocabulary me tailstock/steady rest operation nahi hai, to ye `risk` hain jo plan sheet/UI me dikhenge. "Precedence-valid par physically infeasible" ka number sirf hard rules (HOLE-EXISTS, SEQ-PARTOFF, machine capability) se banta hai. Ye zyada tar LLM galtiyan pakadega; RB par ~0.
4. Ra rules me optimism bacha hai (Facing ko floor se nikala hai kyunki har route me hota hai, par ye nahi pata ki critical surface kaun si hai). TOL-HOLE maanta hai ki part tolerance hole ki hai; part-level tolerance ko feature se jodna dataset me nahi hai.
5. ISO 286 values sirf 6-250 mm ke liye load hain (verified table); isse bade size pe TOL rules `not_evaluated`.
6. R1 abhi in actions ko use nahi karta (`insert:Reaming`, `insert:Drilling`...). Violation format wahi hai, to agla step (feasibility-aware R1/selection) iske upar ban sakta hai.

### Chalane ka tareeka
```
python scripts/feasibility_check.py --dataset dev_v2 --json results/feasibility_check.json
python scripts/run_eval.py --system rb --dataset dev_v2 --tag dev                     # summary me "feasibility" lines
python scripts/run_eval.py --system n5_r1_s3 --dataset dev_v2 --limit 20 --tag dev    # raw LLM routes ka strict-valid-but-infeasible number (agar b0 ka cache is tag ke saath hai to API nahi)
python -c "import json; from capp import feasibility as F; print(json.dumps(F.rule_table(), indent=1))"
```

### Abhi kya nahi hai
Feasibility-aware repair/selection, UI ke dono modes, extraction front end (hole diameter record karna), expert benchmark (WP2), model cascade, confidence threshold tuning, process plan aur route sheet (WP9).
