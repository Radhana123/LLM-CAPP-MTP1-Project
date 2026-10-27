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


## Step 12 (WP9 prep): UI ke dono modes, physics aur feasibility ke saath

`streamlit run week6/app.py` ab dono modes me Step 10 (physics) aur Step 11 (15 feasibility rules) dikhata hai. Naye numbers `capp/analysis_view.py` me bante hain (Streamlit import nahi karta, isliye browser ke bina test hota hai); `week6/app.py` sirf unhe draw karta hai.
**Ye sirf analysis hai:** route ab bhi v1 ke efficiency score se chunta hai. Naye columns/panels route ko describe karte hain aur problems flag karte hain, selection nahi badalte (pehle isse measure karna hai, phir selection me jodna; wo agla step hai).

### Route Only
- Naya "🛡️ Feasibility Check" block, best route ke neeche. Dimensions nahi hain, isliye sirf wo rules chalte hain jinhe dimension nahi chahiye (HOLE-EXISTS, HOLE-PILOT, SEQ-PARTOFF, TOL-HOLE, TOL-SPEC jab tolerance se ho sake).
- Baaki rules "ℹ️ N rule(s) not evaluated — and why" expander me reason ke saath aate hain ("dims.diameter missing" jaise). Chup-chaap pass nahi dikhaye jaate.
- Physics (energy, kc, wear, Ra) is mode me nahi hai; caption batata hai ki Full Analysis me milega. Is mode ke default dims (Ø25, L50...) sirf v1 ke rough time/cost me use hote hain, naye checks me nahi (unke liye `dims=None`).

### Route + Full Analysis
- Sidebar me do naye inputs: **Required Ra (µm)** (not specified / 0.4 / 0.8 / 1.6 / 3.2 / 6.3 / 12.5) aur **Hole diameter (mm)** (0 = unknown). Ye sirf physics/feasibility ko jaate hain; v1 agents unhe nahi dekhte.
- Step 4 ki route table me 6 naye columns: Cutting energy (kWh), kc, Tool wear (₹), Ra risk, Hard issues, Risks. Pehle 7 columns aur efficiency wahi hain (test: `test_physics_columns_do_not_enter_the_efficiency_score`).
- Naya **STEP 5** (best route ke liye): feasibility findings (hard = lal, risk = peela, har ek ke saath rule id, source status aur suggested action), "not evaluated" list, 5 metric cards (cutting energy, mean kc, tool wear ₹/part aur ₹/batch, peak spindle power aur utilisation, Ra risk), per-operation table, energy aur wear ke bar plots, Ra panel (predicted vs required, max feed), speed what-if table (x0.8 / x1.0 / x1.2), "Assumptions" expander, "15 rules aur unke sources" expander (status web/derived/model/assumption).
- Speed what-if **fixed table hai, slider nahi:** app ka poora result `if not run_btn ... else:` ke andar hai, results ke andar koi bhi widget rerun karke result mita deta. Test `test_step_5_adds_no_input_widget` isi ko pakadta hai.

### Jo honestly likhna hai
1. **Default demo part (Aluminium, Ø25) pe CAP-POWER risk dikhta hai:** Facing 7.7 kW maangta hai, machine 7.5 kW (assumed). Model ke standard aluminium parameters (Vc, feed, depth) aggressive hain aur machine power assumed hai; isse "machine me nahi chalega" mat padho. Supervisor ko dikhane se pehle department ki machine ka asli power daalna hai (`feasibility.Machine`).
2. **UI me features sirf image extraction se aate hain** (existing behaviour: `if True:` block me sirf uploader + multiselect jo extraction ke baad dikhta hai). Groq key ke bina ya image ke bina koi run start nahi hota. Tests `st.session_state` me extraction result daal ke chalate hain. Manual feature input extraction front end wale step me aayega.
3. Ra risk optimistic hai (route ka sabse fine process critical surface maana gaya hai); "high" pakka warning hai, "low" guarantee nahi. Hole diameter na ho to hole operations physics me part diameter use karte hain (v1 Time agent ki tarah) aur CAP-POWER unhe skip karta hai. Dono baatein "Assumptions" expander me likhi hain.
4. Cutting energy sirf cut ki energy hai (lower bound); ye v1 ka "Energy" score nahi hai (wo fixed per-operation constants hai). Dono table me saath dikhte hain, naam alag hain.
5. Streamlit 1.66 me `use_container_width` deprecated warning deta hai (kaam karta hai). Poori app me wahi use hota hai; maine nahi badla kyunki `width="stretch"` purane Streamlit versions me nahi hai. Streamlit upgrade karo to ek saath badalna hoga.
6. `.streamlit/config.toml` me BOM tha, jisse Streamlit "Error parsing config toml" deta tha aur `gatherUsageStats = false` kabhi lagta hi nahi tha. BOM hata diya (sirf encoding badli, content wahi).
7. Tests: `tests/test_analysis_view.py` (40, view model) aur `tests/test_app.py` (17, Streamlit `AppTest` se headless: dono modes, saare 7 materials, kuch feature sets, Ra/hole diameter inputs, "koi feature nahi" wala purana error). Tests replay mode me chalte hain (`CAPP_LLM_MODE=replay`, key khali), isliye API call nahi hoti, chahe tumhare `.env` me key ho.
   Do screenshots (headless Chromium) se layout bhi dekha: metric cards pehle truncate ho rahe the, theek kiye.

### Chalane ka tareeka
```
streamlit run week6/app.py
python -m pytest tests/test_analysis_view.py tests/test_app.py
```

### Abhi kya nahi hai
Feasibility-aware repair/selection (R1 ke `Violation.action` use karke), extraction front end (hole diameter record karna, manual input), expert benchmark (WP2), model cascade, confidence threshold tuning, process plan aur route sheet (WP9).


## Step 13 (WP3 + WP6): feasibility-aware R1 (row A7), options `+fz1` aur `+fz2`

Step 11 ke feasibility rules ab repair aur selection me bhi kaam karte hain. Opt-in hai: system name ke aakhir me `+fz1` ya `+fz2` (repair r1, r2 ya r3 ke saath; r0 ke saath error). Bina `+fz` ke koi output nahi badla (test: purane 454 tests + `test_a_feasible_route_is_untouched_at_every_level`).
**Prompt wahi rehta hai, to LLM answers cache se aate hain: `n5_r1_s3+fz1` ko `n5_r1_s3` ke same `--tag` pe chalao, API call nahi lagti** (sirf R2/R3 re-prompts naye hote hain, kyunki unme feasibility finding bhi jaati hai).

### Kya karta hai (`capp/feasrepair.py`)
| level | repair | selection |
|---|---|---|
| `+fz1` | strict-valid par **HARD** feasibility violation wala route bhi repair hota hai: HOLE-EXISTS (hole banane wala operation, Drilling nahi to Boring, dependent operation se pehle; route me na ho to Drilling insert), SEQ-PARTOFF (har lathe op Parting-off se pehle). | sabse kam hard findings wale candidates pehle |
| `+fz2` | fz1 + risk rules ke insert actions: HOLE-PILOT (Center Drilling), TOL-HOLE (Reaming). | pehle sabse kam hard, phir sabse kam risks, phir wahi s0/s1/s2/s3 |

- Algorithm: R1 jaisa set repair + order repair, phir rules chalao, findings ko insertions aur ordering constraints me badlo (`order_repair(..., extra_edges)`), dobara order repair; jab kuch actionable na bache tab ruko (max 6 rounds). Exact DP hai, deterministic, API nahi. Finding na ho to output R1 se byte-for-byte same (test).
- **Machine capability** (CAP-LATHE-*, CAP-MILL-ENV) route edit se theek nahi hoti: sirf report hoti hai (`unrepairable`), repair trigger nahi karti. **Supports, feed, finishing (HOLD-*, RA-*, CAP-POWER, HOLE-DEEP, TOL-SPEC)** sirf flag hote hain (`flagged`), route me nahi aa sakte (vocabulary me tailstock/steady rest nahi).
- Sirf LLM candidates repair hote hain. R0/Route Builder routes ko haath nahi lagta. Risk-only wale route ka repair fail ho to R0 se replace nahi hota, LLM ka route rehta hai (`fz_risk_unfixed`); risk ke liye LLM re-prompt bhi nahi hota.
- `first_pass` (M1) abhi bhi precedence-level strict check hai. Physical infeasibility alag naap hoti hai (summary ka "feasibility" block, report ka "reported separately"): A6 aur A7 ko `final_infeasible_share` pe compare karo.
- Naya output sirf `+fz` systems me: `info.fz_*`, `Candidate.repair.r1.fz_inserted/fz_edges/fz_rounds/unrepairable/flagged/remaining`, per_part me `fz_pool/fz_filtered`, summary me "feasibility fix" line (`feasibility_repair` block), run.json me `feasibility_repair`.

### Offline numbers: `python scripts/feasrepair_check.py --dataset dev_v2` (no API; LLM nahi, synthetic damage)
- **A. Route Builder ke 2637 routes (200 parts):** 60 (2.3%) me fixable hard finding (sab HOLE-EXISTS). `+fz1` ne **60/60 theek kiye** (48 me Drilling insert, baaki me sirf order). 501 (19%) me insertable risk (HOLE-PILOT / TOL-HOLE), `+fz2` ne 501/501 theek kiye.
- **B. 600 damaged routes** (part ka random valid RB route, shuffle + aadhe me ek op drop; seed 5; seed 11 x10 pe bhi wahi picture):

| | repaired | repair ke baad hard-infeasible | fixable risk bacha | ops inserted | discordant pairs |
|---|---|---|---|---|---|
| R1 | 1.000 | **0.035** | 0.185 | 0.45 | 6.69 |
| R1+fz1 | 1.000 | **0.000** | 0.215 | 0.48 | 6.71 |
| R1+fz2 | 1.000 | 0.000 | **0.000** | 0.70 | 6.71 |

  Matlab: plain R1 ke repaired routes me 3.5% physically infeasible rehte hain; fz1 unhe 0 karta hai sirf +0.03 ops ke saath; fz2 risks bhi hata deta hai (+0.25 ops/route). Order me extra badlav (discordant pairs) +0.02 hi.

### Asli LLM run (Radhana ka, `dev_v2`, 20 parts, 1 repeat, cache se) — ye numbers screenshots se hain
| | rb | n5_r1_s3 | +fz1 | +fz2 |
|---|---|---|---|---|
| M6 efficiency | 64.46 | 63.82 | 63.82 | 63.53 |
| M6 gap to best RB (pts) | 1.06 | 3.64 | 3.64 | 3.92 |
| finals with a hard infeasibility | 0.00 | 0.00 | 0.00 | 0.00 |
| finals with risks | 0.54 | 0.40 | 0.40 | 0.35 |
| strict-valid raw LLM routes infeasible | n/a | 0.000 | 0.000 | 0.000 |
| M5 edit distance (R1 stage) / ops surviving | n/a | 2.00 / 0.863 | 2.00 / 0.863 | 1.65 / 0.911 |
| physics: cutting energy kWh/part | 0.489 | 0.559 | 0.559 | 0.559 |
| physics: parts over machine power | 0.23 | 0.45 | 0.45 | 0.45 |
| physics: tool wear INR/part | 95.1 | 78.7 | 78.7 | 79.0 |
| plan confidence mean / flagged | n/a | 0.947 / 0.00 | 0.947 / 0.00 | 0.951 / 0.00 |

Paired final Efficiency vs rb (20 pairs): n5_r1_s3 -3.19 pts (better 4, equal 9, worse 7); +fz1 -3.19 (same); +fz2 -3.48 (better 4, equal 8, worse 8).
- **Asli LLM routes pe `+fz1` ne kuch nahi badla:** 20 parts x 5 samples me koi strict-valid raw route physically infeasible nahi tha (HOLE-EXISTS Route Builder ke bhi sirf 2.3% routes me aata hai). Is chhote set pe A7 ka fz1 effect zero hai; ye natija hai, nuksan nahi. Bade ya Groove-heavy subset pe dobara dekhna hai.
- `+fz2`: risky finals 0.40 -> 0.35, efficiency -0.29 pts (offline trade-off jaisa). 20 pairs me ye significant nahi.
- LLM-first RB se 3.19 pts peeche hai, aur machine power se upar jaane wale parts zyada hain (0.45 vs 0.23), par risky finals kam (0.40 vs 0.54). Kaaran abhi nahi pata: 7 "worse" parts ka per-part diff baaki hai.
- Confidence ~0.95 aur flagged 0.00: threshold is set pe kisi plan ko flag nahi karta (tuning baaki).

### Jo honestly likhna hai
1. **Ye offline numbers repair mechanism ke hain, LLM ke nahi.** Damage synthetic hai. Asli A7 effect (A6 vs A7, real LLM errors) live cache ke saath `n5_r1_s3` vs `+fz1` vs `+fz2` chalake naapna hai (neeche).
2. **fz1 Drilling insert kar sakta hai aisi part me jisme Hole feature hi nahi** (jaise `Turning, Step, Groove, ...`, jahan RB variant me Internal Grooving aata hai): rule kehta hai internal groove ke liye pehle bore chahiye. Dataset ka "Groove" feature internal hai ya external (Grooving/Necking), ye saaf nahi; supervisor se confirm karna hai. Ye wahi rule-base defect hai (Drilling -> Internal Grooving edge graph me nahi); fz1 use sirf tab lagata hai jab finding aaye, graph abhi bhi nahi badla.
3. **fz1 akela Drilling insert karta hai bina Center Drilling ke** (action wahi hai), isliye "fixable risk" 0.185 -> 0.215 badh jata hai. Isi wajah se asli comparison ke liye `+fz2` hai; `+fz1` "sirf hard" reference hai.
4. **fz2 ka TOL-HOLE maanta hai ki part ki tolerance hole ki hai** (Step 11 ka assumption), isliye tight tolerance + hole feature pe Reaming insert hoga. Reaming time/cost badhata hai; fz2 selection (feasibility pehle) isliye costlier route chun sakta hai. Plan quality M6 (efficiency gap) aur feasibility dono saath report karna.
5. Risk "kam count" selection saare risks ko barabar tolta hai (Center Drilling ka missing hona aur Reaming ka missing hona same). Machine-capability aur part-level (TOL-SPEC) findings sab routes ke liye same hain, isliye selection key me nahi.
6. **Selection ka naya pool `fz_pool` me record hota hai;** front size ab filtered pool ka hai (`population` purana hi hai). Agar tum front size ko purane runs se compare karo to dhyan rakhna.
7. Tests: `tests/test_feasrepair.py` (48): order_repair brute-force (extra edges ke saath), har fix ka case, levels, idempotent/deterministic, dev set pe saare 60 HOLE-EXISTS variants + random damage property, system names, run me (R1/R2/R3, fallback, risk-only, selection filter, rule error count), harness aur summary, script.

### Chalane ka tareeka
```
python scripts/feasrepair_check.py --dataset dev_v2                       # offline, no API
python scripts/run_eval.py --system n5_r1_s3     --dataset dev_v2 --limit 20 --tag dev   # LLM samples (cache)
python scripts/run_eval.py --system n5_r1_s3+fz1 --dataset dev_v2 --limit 20 --tag dev   # same cache, no new API calls
python scripts/run_eval.py --system n5_r1_s3+fz2 --dataset dev_v2 --limit 20 --tag dev
python scripts/compare_runs.py results/dev_rb results/dev_n5_r1_s3 results/dev_n5_r1_s3+fz1 results/dev_n5_r1_s3+fz2
```

### Abhi kya nahi hai
Extraction front end (hole diameter record karna, manual input), expert benchmark (WP2), model cascade, confidence threshold tuning, process plan aur route sheet (WP9). `Drilling -> Internal Grooving` edge ka decision tumhara/supervisor ka.


## Step 14 (WP9 prep): extraction front end, manual feature input aur drawing se hole diameter

Pehle features sirf image extraction se set ho sakte the (image + API key chahiye), aur hole diameter sirf haath se type hota tha. Ab dono behtar hain. Batch evaluation (`run_eval`, dataset v2, baselines) pe is step ka koi asar nahi: sirf `week6/app.py`, `week1/image_feature_extractor.py` aur naya `capp/dimensions.py` badle.

### Kya badla
1. **Manual feature input** (`week6/app.py`): sidebar me radio "How do you give the features?": *Upload a drawing (AI extraction)* (default, pehle jaisa) ya *Select the features myself*. Manual me image aur API key nahi chahiye: saare `GEOMETRY_FEATURES` ka multiselect, aage pipeline wahi (tokenizer, NSGA-II, Step 5 physics, feasibility).
   - Khaali manual selection "select at least one feature" deta hai: pichle extraction ke features pe fall back **nahi** karta (pehle ye fall-back hi sirf ek tareeka tha, ab sirf upload mode me chalta hai).
   - Manual selection tab bhi bachti hai jab tum upload mode me jaake wapas aate ho (Streamlit hidden widget ki state hata deta hai, usse bachaya).
2. **Drawing se hole diameter** (`extract_features_from_image(..., with_dimensions=True)`, app use karta hai): do feature passes ke baad **ek teesra call** (alag prompt `DIMENSION_SYSTEM_PROMPT`): sirf wo hole diameters jo drawing pe **likhe** hain (mm). Andaaza lagana ya compute karna mana hai; thread callout (M8), PCD, shaft/boss ka bahari diameter, radius, inches (khaali list + note) nahi.
   - **Do feature passes ke messages aur params byte-same hain** (test), isliye unke cache keys aur recall nahi badle: tumhara purana extraction cache chalta rahega. Naya sirf dimension call hai (ek image pe ek extra call, phir cached).
   - Default `with_dimensions=False`: purane callers aur `test_extractor_goes_through_client_one_call_per_pass` bilkul same. Dimension call fail/gandi reply de to extraction fail **nahi** hota: `hole_diameters = []`, `dimensions.error` me wajah, aur tum haath se type karte ho.
3. **`capp/dimensions.py`** (pure, no network): model ka JSON saaf karta hai. Maanta hai: number, "8", "8 mm", "Ø8", "⌀8", "dia 8", "Ø12 H7", "Ø8 ±0.1", "2x Ø6 THRU", "8,5". Rad karta hai (aur `dropped` me batata hai): M-callout, "10x1.25", "5 in", bool, NaN, [0.5, 200] mm ke bahar (app ki input limit). Unique, bada pehle, max 8.
4. **App me pre-fill** (Full Analysis mode): hole diameter field me **sabse bada** detected diameter aata hai; extraction block me caption sab detected sizes dikhata hai aur "check it against the drawing" kehta hai. Ek detection ek hi baar lagti hai (tumhara edit rerun pe nahi mitta), naya drawing dobara lagta hai. Route Only se Full me wapas aane pe field khaali ho jaata tha (hidden widget ki state): AppTest ne ye pakda, ab wahi detection wapas lagti hai. Jis extraction me diameter nahi mila wo field ko haath nahi lagata. Number hamesha editable field me aata hai, plan me chup-chaap nahi jaata ("AI is verified, not trusted").

### Jo honestly likhna hai
1. **Dimension pass live test nahi hua** (mere sandbox me API nahi): sirf fake transport se. Vision model Ø callouts galat bhi pad sakta hai. Pehle 3-4 asli drawings pe `streamlit run week6/app.py` me check karo ki caption sahi diameters dikhata hai.
2. **Ek hi hole_diameter poore part ke liye.** Kai holes hon to sabse bada pre-fill hota hai: CAP-POWER (hard rule) sabse bade hole pe sabse kharab hai, HOLE-DEEP sabse chhote pe. Ek number dono ke liye conservative nahi ho sakta; isliye sab sizes dikhte hain aur field editable hai.
3. M-callout wale (tapped) holes ka diameter nahi aata (nominal size drill size nahi hai): inke liye HOLE-DEEP "not evaluated" hi rahega.
4. **Dataset v2 me hole_diameter abhi bhi nahi hai** (baselines frozen): `run_eval` me "HOLE-DEEP 61% parts me evaluate nahi hota" wali baat waisi hi rahegi. WP2 benchmark me hole_diameter record karna ho to uska schema alag se tay karna hai.
5. Baaki dims (diameter, length, depth, width) abhi bhi haath se; AI se nahi padhe.
6. Route Only <-> Full toggle me Diameter/Depth/Length/Width inputs reset ho jaate hain (pehle se aisa hi tha, Step 12 ka behaviour); sirf hole field aur manual features bachaye gaye.

### Tests
`tests/test_dimensions.py` (71): parser ke accepted/rejected forms, sorting/dedupe/cap, pre-fill rule (ek baar, naya detection, field ki state gayi), extractor ka dimension pass (ek extra call, feature passes byte-same, pinned model, cache, gandi/failed reply, inches, out-of-range). `tests/test_frontend.py` (22, AppTest): input-method radio, manual mode (pipeline, route only, khaali selection, extraction pe fall back nahi, selection bachti hai), pre-fill (bada diameter, caption, edit rerun pe bachta hai, naya detection, bina detection field untouched, Route Only se wapas, HOLE-DEEP tak pahunchta hai, results me koi widget nahi).

### Chalane ka tareeka
```
streamlit run week6/app.py          # sidebar: "Select the features myself" ya drawing upload karke Extract
python -m pytest tests/test_dimensions.py tests/test_frontend.py
```

### Abhi kya nahi hai
Baaki dimensions ka extraction, expert benchmark (WP2), model cascade, confidence threshold tuning, process plan aur route sheet (WP9). `Drilling -> Internal Grooving` edge ka decision tumhara/supervisor ka.


## Step 15 (WP3 analysis): per-part diff, `scripts/part_diff.py`

`compare_runs.py` averages deta hai ("n5_r1_s3 rb se 3.19 pts peeche, 7 pairs me worse"). Ye script batata hai **kaun se parts** aur wahan **routes me kya alag hai**. Offline hai, API nahi, sirf `records.jsonl` padhta hai. Pehla folder reference (rb), doosra us par judge hota hai (compare_runs jaisa).

```
python scripts/part_diff.py results/dev_rb results/dev_n5_r1_s3                 # LLM vs RB: kaun se parts worse
python scripts/part_diff.py results/dev_n5_r1_s3 results/dev_n5_r1_s3+fz2       # fz2 ne kaun se parts kharab kiye
python scripts/part_diff.py results/dev_rb results/dev_n5_r1_s3 --top 20 --only all --csv results/pairs.csv
python scripts/part_diff.py results/dev_rb results/dev_n5_r1_s3 --group archetype,material,n_features
```

### Kya dikhata hai (`capp/partdiff.py`)
- Har (part, repeat) pair: final Efficiency (dono strict-valid ho tabhi comparable), time/cost/energy aur unka relative diff, dono routes, provenance, physics (energy, wear, peak kW, over-power, Ra risk) aur feasibility (hard, risks).
- **Cause tag** (description, causation nahi): `identical`, `order only`, `other has extra operations`, `other lacks operations`, `different operations`. Operations route-order me, multiset ke hisaab se; markers ignore.
- Tables: part class se (machine profile, complexity, material, tolerance; `--group` se archetype, n_features, batch_size bhi), cause tag se, doosre run ke final ki provenance se (raw_llm / repaired_llm / route_builder).
- **Operations jo alag hain:** worse pairs me kaun sa op doosre run me extra (`other has`) ya kam (`other lacks`) hai, mean Efficiency diff ke saath, worst pehle.
- **Efficiency kya hilata hai:** time / cost / energy ka mean relative diff (sab pairs aur worse pairs), operations aur changeovers per route.
- Fir sabse kharab pairs, dono routes ke saath (`-op` sirf pehle run me, `+op` sirf doosre me). `--csv` har pair ki ek row deta hai.
- Consistency test: summary ke better/equal/worse aur mean `compare_runs.paired_quality` se exactly same hain.

### Jo honestly likhna hai
1. **Description hai, causation nahi.** Cause tag batata hai routes kaise alag hain, ye nahi ki Efficiency kyun alag hai. Jo op worse pairs me dikhe wo common bhi ho sakta hai: dono taraf ki frequency table me nahi hai (sirf "jahan alag hai wahan mean diff").
2. 20 parts pe har table me n bahut chhota hai: ye anecdotes hain, statistics nahi. Dekhna routes ko hai.
3. Efficiency v1 ka function hai (time, cost, energy, changeover markers ke saath scored). Physics aur feasibility alag columns hain aur Efficiency me nahi jaate.
4. Synthetic LLM (fake planner, tests me) pe ye script dikhata hai ki `+fz2` ke Reaming insertions (TOL-HOLE) worse pairs me aate hain: ye test ka artefact hai, tumhare asli run ka nahi. Asli run pe apna output dekhna.

### Tests
`tests/test_partdiff.py` (31): route delta (har cause, multiset, markers, changeovers, mirror), do asli runs (harness + deterministic messy planner) pe pair rows, compare_runs se agreement, run vs khud (sab identical/equal), runs swap karne pe har pair ulta, unmatched/mismatch, invalid final, summary tables ka jod, operations table, report ke sections, `--top` aur ordering, `--only`, CSV, script ke options aur errors.

### Abhi kya nahi hai
Model cascade, confidence threshold tuning, expert benchmark (WP2), process plan aur route sheet (WP9), baaki dimensions ka extraction. `Drilling -> Internal Grooving` edge ka decision tumhara/supervisor ka.


## Step 16 (WP2): expert benchmark pipeline, `scripts/benchmark.py`

Report §7.1: 40-60 asli parts, supervisor review, 15-20 parts pe doosra annotator, aur M10 (do logon ke plans ki aapas ki agreement) jo M4 ka ceiling hai. Ye step uska **tooling** hai: plans insaan likhte hain, code spec padhta hai, workbooks banata hai, bhare hue workbooks import karta hai, plans ko rule base aur physics se check karta hai, M10 nikalta hai aur datasheet likhta hai. API call nahi.

### Workflow (repo root, PowerShell)
```
python scripts/benchmark.py template --out bench/spec.xlsx                     # 1. khali parts spec (Excel), har part ek row
#   ... spec bharo: material, features, tolerance, batch, ASLI dimensions, hole_diameter (optional), Ra (drawing se), source ...
python scripts/benchmark.py sheets --spec bench/spec.xlsx --out-dir bench `
    --annotator "Prof. Deb" --second-name "Second annotator" --second-n 18      # 2. annotators ke workbooks + bench/bench_spec.json
#   Prof. Deb annotate_primary.xlsx bharte hain (review columns ke saath), doosra annotate_second.xlsx (18 parts), alag-alag
python scripts/benchmark.py import --spec bench/bench_spec.json `
    --primary bench/annotate_primary.xlsx --second bench/annotate_second.xlsx --dry-run    # 3. sirf problems dikhao
python scripts/benchmark.py import --spec bench/bench_spec.json `
    --primary bench/annotate_primary.xlsx --second bench/annotate_second.xlsx              # data/bench_v1.json + .csv + datasheet
```
Asli parts aane se pehle poori chain rehearse karne ke liye: `python scripts/benchmark.py select --from test_v2 --n 50 --out bench/spec_placeholder.xlsx`. Ye synthetic parts se **PLACEHOLDER** spec banata hai (source aur notes me "placeholder" likha hota hai, datasheet ke upar WARNING aati hai). Isse M4 ka koi number "expert" nahi hai.

### Kya karta hai (`capp/benchmark.py`)
- **Spec:** `parse_part_row` har row ke saare errors ek saath deta hai (empty cell, material/feature/tolerance galat, typo pe "did you mean", limits). Material aliases (`aluminium`, `castiron`), tolerance `0.05` / `0.05 mm` / `0.05MM`, features `;` ya `,` se. Row numbers file ke hisaab se (header = row 1).
- **Workbooks:** Routes sheet me har part ke liye row A (plan) aur B, C (sirf agar doosra plan *barabar* acceptable hai), har step ek dropdown cell (41 operations), `Operations` sheet me machine aur usually-serves-features. Primary workbook me supervisor review columns (accepted / changed / rejected, reviewed_by, comment); second me nahi, aur sirf uska subset.
- **Import:** operation ka naam case/space ignore karke match hota hai; galat naam blocking error hai, suggestion ke saath (`Drillng` -> `Drilling`). Plan ke bina part blocking hai (`--allow-missing` se bahar). Har plan rule base (precedence, missing, invented) aur 15 feasibility rules se **asli dimensions pe** check hota hai.
- **Record:** `capp/dataset.py` ka bench split: `dims` me optional `hole_diameter`, `required_ra_um` drawing se (None ho sakta hai, tolerance se derive nahi), `reference_routes` zaroori, `second_annotator_routes` alag, `reference_meta` (annotator, review, conflicts, physical).
- **M10:** doosre annotator ke pehle plan ka M4 (exact, LCS ratio, Kendall tau) reference plan(s) ke saath, jo parts dono ne kiye unpe, 95% bootstrap interval ke saath.
- **Datasheet:** markdown, data se generate (composition, plans, review, conflicts, physical findings, M10, limits, `[TO CONFIRM]` items).
- **Harness / compare:** jin parts me `reference_routes` hain unpe M4 expert routes se naapa jata hai (pehle se tha), ab `compare_runs.py` table me "M4 reference (expert, else proxy_rb)" aur M4 ki 4 rows aati hain. `bench_*` sealed hai (`test_*` ki tarah): `run_eval` warning deta hai aur retrieval pool (`--pool`) ke roop me refuse hota hai.

### Jo honestly likhna hai
1. **Placeholder asli benchmark nahi hai.** `select` sirf pipeline rehearse karne ke liye hai; usse bane koi number report me expert result ke naam se nahi jaane chahiye.
2. **Pre-fill nahi.** Workbook me koi plan pehle se bhara nahi hota: bhara plan annotator ko anchor karta hai, aur Route Builder ka bhara plan har baad ki comparison me RB ko favour karta. Ye jaanbujhkar hai.
3. **Rule base se reject hua plan bhi reference rehta hai** (expert ground truth hai) aur conflicts me listed hota hai. Har conflict ya expert ki slip hai ya rule base ki kami: kaun sa, ye supervisor ke saath tay hota hai, code ke nahi.
4. **Reference plans ek expert ki practice dikhate hain.** Alternatives (row B, C) aur supervisor review isse kam karte hain, hatate nahi.
5. **M10 15-20 parts pe indicative hai**, precise nahi: interval ki chaudai dekhni hai. "Inspection last" rule base ka convention hai aur annotators ko wahi follow karna kaha gaya hai.
6. `hole_diameter` optional hai (dataset v2 me nahi hai, isliye wahan HOLE-DEEP 61% parts pe evaluate nahi hota); benchmark me asli dimension honi chahiye.
7. Benchmark sealed hai: sirf final runs (WP11). Uske results dekhne ke baad koi prompt, rule ya threshold nahi badalna.

### Test fix (flaky throttle tests)
Teen throttle tests kabhi-kabhi full run me fail hote the. Asli kaaran mila: harness `git rev-parse` ko `subprocess.run(timeout=...)` se chalata hai, jo `time.sleep` se poll karta hai. Tests `time.sleep` ko poore process me patch karte the, to loaded machine pe woh polling bhi "harness ke sleeps" me gin li jaati thi (6000+ entries). Ab tests sirf harness ka apna `time` replace karte hain (`tests/harness_clock.py`). Pehle ek aisa `git` jo pipes band karke der se exit ho, purane tests ko har baar fail karta tha; ab pass. (Pehle "thread filter se fix" likha tha: wo ghalat tha, thread ka isse lena-dena nahi tha.)

Do chhote bugs tests ne pakde: `save_dataset` bench part par (jisme `hole_diameter` ho) crash karta tha, aur tolerance `0.05MM` (capital) reject hota tha. Dono fix.

### Tests
`tests/test_benchmark.py` (80): spec parsing aur errors, template/CSV/XLSX round trip, placeholder select, workbook structure (koi pre-fill nahi, dropdown, review columns), simulated annotators se fill aur import, typo suggestions, missing plan aur `--allow-missing`, rule conflicts aur physical findings, alternatives, M10 (handcrafted pairs, bootstrap), datasheet, bench split ki validation, save/load, sealing (`is_sealed`, pool, `run_eval`), harness pe expert-M4, compare table, aur `scripts/benchmark.py` ke saare subcommands (`--dry-run`, errors).

### Abhi kya nahi hai
Asli parts aur plans (tumhara/supervisor ka kaam; tooling taiyaar hai), model cascade, confidence threshold tuning, process plan aur route sheet (WP9), baaki dimensions ka extraction. `Drilling -> Internal Grooving` edge ka decision tumhara/supervisor ka.


## Step 17 (WP3, row A8): model cascade, options `+ms`, `+ml`, `+cas1`, `+cas2`, aur `scripts/cascade_pareto.py`

Report §6.4 row A8: pehle chhota model, jahan wo nahi chale wahan bada model. Ye step cascade, cost accounting aur validity-against-cost Pareto ka tooling hai. **Ye abhi asli API ke saath chala nahi hai** (tests me fake transport hai): chhote/bade model ka asli fark tumhare live runs se aayega.

### Models aur options
| option | matlab |
|---|---|
| (kuch nahi) / `+mp` | primary model (`qwen/qwen3.8-27b`): purane saare results isi ke hain |
| `+ms` | sirf small model (`openai/gpt-oss-20b`), samples aur R2 re-prompts dono |
| `+ml` | sirf large model (`openai/gpt-oss-120b`) |
| `+cas1` | small se sample; agar slot ka final route Route Builder ka bana (repair fail), to usi slot ko large model se dobara sample karo |
| `+cas2` | small se sample; agar raw route **jaisa tha waisa accept nahi hua** (invalid, ya repair ki zaroorat), to large se dobara; dono me jo behtar provenance wala ho wo rakho, barabari par large |

Env se badal sakte ho: `CAPP_MODEL_PRIMARY`, `CAPP_MODEL_SMALL`, `CAPP_MODEL_LARGE`. Options baaki sab ke saath chalte hain (`+fs2`, `+h1`, `+ts`, `+fz1`...), sirf `+cas` ke saath `+ms/+ml/+mp` nahi (cascade ke dono models fixed hain). Cascade me R2 re-prompt usi model ko jaata hai jisne sample diya.

### Workflow (same `--tag`, same parts)
```
python scripts/run_eval.py --system n5_r1_s3+ms   --limit 20 --tag dev      # sasta end: small model ke naye calls
python scripts/run_eval.py --system n5_r1_s3+ml   --limit 20 --tag dev      # mehnga end: large model ke naye calls
python scripts/run_eval.py --system n5_r1_s3+cas1 --limit 20 --tag dev      # in dono ke cache se bana: naya sample call nahi
python scripts/run_eval.py --system n5_r1_s3+cas2 --limit 20 --tag dev
python scripts/cascade_pareto.py results/dev_n5_r1_s3+ms results/dev_n5_r1_s3+ml results/dev_n5_r1_s3+cas1 results/dev_n5_r1_s3+cas2 `
    --plot results/pareto.png --csv results/pareto.csv
```
Cache trick: cascade ke prompts `+ms` / `+ml` ke prompts ke same hain (sirf model alag), to `+ms` aur `+ml` pehle chalane par cascade bina naye sample call ke bante hain (test me replay mode me saabit: zero naye calls). R2 systems (`_r2_`) me re-prompts alag hote hain, unke thode naye calls aate hain.

### Cost
- **Cost = list-price USD** = tokens x `capp/config.py::PRICE_USD_PER_MTOK`. Cache hits poore gine jaate hain (cache se bhi wo call "karni padti"), to number cached aur live run me same aata hai. Free tier me asli bill nahi hai: ye notional hai.
- **Prices third-party sources se hain (2026-10-13), `[TO CONFIRM]` Groq console se.** Jab tak confirm nahi, absolute USD ko tez nahi, sirf systems ke aapas ke order/ratio ko dekho. Kisi model ka price table me nahi hai to uska USD `n/a` aata hai (zero nahi) aur summary likhti hai kaunsa.
- Summary me naye blocks: `cost` (model-wise calls, tokens, USD, USD/part, tokens/part, latencies ka jod) aur `cascade` (slots, escalation rate, rescue rate, small ne as-is accept hone ki rate, large se aaye finals, large ke baad bhi small rakha). `compare_runs.py` me "models called", "cost: ..." aur "cascade: ..." rows. Per-candidate CSV me `model` hamesha, `cascade_tier`/`escalated`... sirf cascade runs me.
- **Purane records** (model/cascade keys ke bina) bhi summarise hote hain: unke calls primary maane jaate hain aur summary ye likhti hai ("assumed the primary model").

### Pareto (`scripts/cascade_pareto.py`)
Har run ek point: USD/part, tokens/part, latency, **LLM routes** (= final route Route Builder ka nahi, LLM ka hai: raw ya repaired; 1 - route_builder share), raw strict-valid, efficiency gap, M4 LCS, escalation rate. Do frontiers: **V** = (cost, LLM routes) aur **Q** = (cost, efficiency gap). "Price of validity" table V frontier ke har agle step ka extra USD/part per extra point of LLM routes batati hai. Alag parts par chale runs "not comparable" flag hote hain. `--plot` ke liye matplotlib chahiye.

### Jo honestly likhna hai
1. **R1 ke saath `+cas1` shayad hi trigger hoga.** R1 (exact search) lagbhag har baar repair kar leta hai, to final route Route Builder ka bahut kam banta hai aur cas1 ka trigger ("repair fail") rare hai. cas1 `r0`/`r2` systems me ya khaali/unparsable answers par kaam ka hai. R1 systems ke liye meaningful cascade `+cas2` hai (raw route accept na hone par bada model). Tumhare live run me escalation rate dekhni hai: 0 aaye to ye bug nahi, trigger hi nahi laga.
2. **"LLM routes" validity ka sirf ek axis hai.** R1 se repaired route bhi LLM route gina jaata hai; raw quality alag column (raw valid, M4 LCS, efficiency gap) me dekho.
3. **Same route do tiers ne diya to credit small ko jaata hai** (dedupe pehle wale route ko rakhta hai), to "final from large" kam dikh sakta hai jab dono same jawab dein.
4. **Cascade ka cost = dono attempts.** Escalated slot me small ka call bhi gina jaata hai. Cascade tabhi jeetega jab escalation rate kam ho.
5. **Small/large ka exact choice dev set par tay hota hai.** `gpt-oss-20b/120b` abhi anuman hai; qwen primary ke comparison me bhi `+mp` run rakho.
6. Edit distance par k-threshold (sirf bade repair par escalate) ek sambhav agla knob hai, abhi nahi hai.
7. Dev set par hi tune karo; test/bench par nahi.

### Refactor safety
`run_llm_first` ka sample loop `sample_slot` me refactor hua. 6 systems x 25 parts ke records purane code se **byte-identical** nikle (hash check), aur cascade ke bina record me koi naya key nahi aata (`cascade_level`, `escalations`, candidate ka `cascade` sirf cascade runs me). Ek hi naya field har record me asli planner se aata hai: `meta.model` (kaunsa model, cost ke liye zaroori); `run.json` me `model`, `model_role`, `price_usd_per_mtok`. Purane records bina inke bhi padhe jaate hain.

### Tests
`tests/test_cascade.py` (43): option grammar aur errors, prices/cost math, escalate/keep rules, `run_llm_first` fake planners ke saath (cas1 sirf failed slot, cas2 raw-not-ok par, large ke worse hone par small rakhna, tie par large, dono fail par Route Builder, fatal error, R2 re-prompt sahi tier ko, set-aside attempt ke repair calls), asli planner path pe model ids (samples aur R2 dono), default system ke cache keys na badalna, **cascade = ms + ml ka cache (replay me zero naye calls)**, cost = tokens x price (dono attempts), `+ts`/`+fs` ke saath cascade, unpriced model, purane records, CSV/run.json, printed summary aur compare table, throttle, replay identical, Pareto frontier logic/price steps/not-comparable, CLI (`--csv`, `--json`, errors) aur `--plot`.

### Abhi kya nahi hai
Asli API pe cascade ka koi run, confirmed prices, confidence threshold tuning, process plan aur route sheet (WP9), baaki dimensions ka extraction, asli benchmark parts/plans. `Drilling -> Internal Grooving` edge ka decision tumhara/supervisor ka.


## Step 18 (WP5): confidence threshold tuning, `scripts/confidence_tune.py`

Report §6.4: "plans with low confidence are flagged for human review". Abhi flag `confidence < 0.8` par lagta hai (`systems.CONFIDENCE_FLAG`, kabhi tune nahi hua); tumhare live run me confidence ~0.95 aur 0 plans flagged the, matlab flag kuch nahi kar raha. Ye step finished runs par offline (API call nahi) batata hai: kaunse plans "bad" hain, confidence unhe kitna pakadti hai, aur threshold kahan rakhna chahiye.

### Workflow (repo root, PowerShell)
```
python scripts/confidence_tune.py results/dev_n5_r1_s3 --plot results/conf.png --csv results/conf.csv
python scripts/confidence_tune.py results/dev_n5_r1_s3 --budget 0.15 --recall 0.9          # kitna review manzoor hai / kitne bad plans pakadne hain
python scripts/confidence_tune.py results/dev_n5_r1_s3 --label not_raw                    # bad = LLM ka apna route use nahi hua
python scripts/confidence_tune.py results/dev_n5_r1_s3 --label lcs --bad-lcs 0.8          # bad = reference se door (expert refs ho to asli label)
python scripts/run_eval.py --summarize-only results/dev_n5_r1_s3 --confidence-flag 0.93   # wahi run, 0.93 se neeche wale flagged
python scripts/run_eval.py --system n5_r1_s3 --limit 20 --tag dev --confidence-flag 0.93  # cache se, naya API call nahi
```
Har run folder alag judge hota hai (systems pool nahi hote). Plots ke liye matplotlib.

### Kya batata hai (`capp/calibration.py`)
- **Bad plan ka label (tum chunti ho, `--label`):** `gap` = efficiency gap best Route Builder variant se `--bad-gap` points (default 3) se zyada; `lcs` = M4 LCS ratio reference se `--bad-lcs` (default 0.8) se kam; `not_raw` = final route LLM ka raw route nahi (repair ya replace hua). Base rate aur gap ke percentiles chhapte hain: base rate 0 ya 100% ho to script saaf bolti hai ki koi threshold nahi ban sakta, label badlo.
- **Teen scores, AUROC ke saath (0.5 = koi information nahi):** `confidence` (repair ke baad agreement; abhi isi par flag lagta hai), aur do diagnostics: `raw_agreement` (repair se pehle raw samples ka agreement) aur `raw_valid_share` (kitne raw samples strict-valid the). Teeno x teeno labels ki table.
- **Har threshold:** flagged share, recall (kitne bad plans pakde), precision, lift. Threshold do observed values ke beech rakha jaata hai, to chhapa hua T bilkul wahi flagged set deta hai (pipeline ka apna rule: `confidence < T`).
- **Do recommendations:** `--budget` (default 20%) ke andar best recall, aur `--recall` (default 0.8) ke liye sabse kam flags.
- **95% bootstrap interval parts par** (ek part ke repeats saath chalte hain, independent nahi).
- **Sealed split:** `test_v2` / `bench_v1` runs par tuning refuse hoti hai. Wahan sirf `--evaluate-at T` chalta hai: dev par chuna hua T fixed rakh ke flagged share, recall, precision (sweep ya recommendation nahi). Protocol: dev par tune, T freeze, sealed par ek baar evaluate.
- `summary.reflag(records, T)` records ko nayi threshold par flag karta hai (disk wale records nahi badalte); summary line me "confidence below T" tabhi aata hai jab T badla ho.

### Jo honestly likhna hai
1. **Labels proxy hain.** dev_v2 par M4 ka reference Route Builder hai, expert nahi; `gap` best RB variant se hai. Asli "kaunsa plan insaan ko dikhana chahiye" ka label expert benchmark par hi banega (sealed hai, to wahan sirf evaluate).
2. **Confidence repair ke baad hai, to R1/R0 use inflate karte hain.** Repair samples ko ek hi route ki taraf kheench deta hai, isiliye 0.95 aur 0 flagged aaya. Agar `raw_agreement` ya `raw_valid_share` ka AUROC `confidence` se saaf upar aaye to flag unpar lagana ek agla (chhota) step hai; abhi flag sirf `confidence` par lagta hai, baaki sirf diagnostic.
3. **Chhota sample.** 20-25 dev parts par 3-5 bad plans: AUROC aur recall ke intervals bahut chaude aate hain, aur jab saare bad plans pakde jaayein to bootstrap interval [1.0, 1.0] jaisa jhootha-tight dikhta hai. Isse "T = ye" nahi, "T ka ilaaka ye" padho.
4. **Selection optimism.** Recommended T usi data se chuna jaata hai jis par naapa jaata hai; intervals is selection ko count nahi karte. Isiliye ek baar sealed par confirm.
5. `gap` label degenerate ho sakta hai (jaise synthetic planner par p50 = 0, koi bad plan nahi): tab `--bad-gap` kam karo ya `--label not_raw` dekho. Label tum chunti ho, script nahi.
6. Flag ka matlab "insaan ko dikhao" hai, "plan galat hai" nahi. Review budget (kitne plans supervisor/engineer dekh sakta hai) design ka faisla hai: `--budget` usi se aata hai.

### Tests ne jo bugs pakde
Threshold ka midpoint rounding bahut chhote gap me flagged set badal deta tha (printed T se wo plans flag nahi hote the): ab sabse chhoti rounding chunti hai jo gap ke andar rahe. Aur `--plot` ek aise label par crash karta tha jisme koi bad plan nahi: ab "no bad plans under this label" likhta hai.

### Tests
`tests/test_calibration.py` (33): AUROC, at-threshold, sweep aur recommendations 12 records par jinke confidence/gap test set karta hai (haath se nikale jawab), midpoint, part-bootstrap (repeats saath), reliability bins, raw signals, unscored plans, one-class labels, errors, sealed mode, report text, `reflag` aur summary line, `run_eval --confidence-flag` (range check, run.json me threshold, `--summarize-only`), script ke options/CSV/JSON, kai runs, sealed refusal aur `--evaluate-at`, `--plot`, purane records.

### Abhi kya nahi hai
Asli live runs par threshold (tumhare dev results chahiye), raw signals par flag, dashboard me confidence/flag ka dikhna (WP9 me jaayega), process plan aur route sheet, baaki dimensions ka extraction, asli benchmark parts/plans, cascade ka live run. `Drilling -> Internal Grooving` edge ka decision tumhara/supervisor ka.


## Step 19 (WP9): process plan + route sheet, `scripts/route_sheet.py`

Report §6.7: "route pehle, parameters baad me". Route chunne ke baad har operation ke liye machine, tool (type, size, material), cutting parameters (Vc, rpm, feed, ap x passes), estimated time aur "ye operation kyun" (feature + rule) ek **route sheet** me aata hai: PDF / Excel / HTML. Sheet route kabhi nahi badalti; sirf use describe karti hai. API call nahi.

### Workflow (repo root, PowerShell)
```
pip install reportlab openpyxl                                     # PDF aur Excel ke liye (HTML/Markdown/JSON bina iske chalte hain)
# dashboard: python -m streamlit run week6/app.py  ->  "Route Sheet" section, dono modes me, download buttons
python scripts/route_sheet.py make --material Steel --features Hole,Chamfer,Slot --diameter 40 --length 120 --depth 15 --width 32 --hole-diameter 8 --ra 1.6 --batch 100 --out results/sheet.pdf results/sheet.xlsx results/sheet.html
python scripts/route_sheet.py from-run results/dev_n5_r1_s3 --part DEV_004 --out results/DEV_004.pdf
python scripts/route_sheet.py from-run results/dev_n5_r1_s3 --part DEV_004 --repeat 1 --confidence-flag 0.9
python scripts/route_sheet.py tables --table all --csv results/process_tables.csv      # tool rules + ranges, status aur sources ke saath
```
`--out` ka extension format chunta hai (`.pdf .xlsx .html .md .json`); `--out` na do to Markdown screen par. Dimensions na do to v1 ka reference part (Ø25, L50, depth 20) lagta hai aur sheet ye likhti hai. `from-run` kisi bhi system ke record se banta hai (final route, provenance, plan confidence + review flag, repair se pehle/baad ka route, two-step planner ka reasoning).

### Sheet me kya hai
Part ka header, **DRAFT banner**, operations ki table (No., operation + description, machine, tool, Vc, n, feed (mm/rev ya mm/tooth, mm/min ke saath), ap x passes, time, feature / rule, notes), machine changeover ki alag rows, totals (cutting + tool change + position change + changeover), checks (strict verifier, physical feasibility, physics flags), "Before and after repair", planner ka reasoning, assumptions, tables ka status aur sources, aur Prepared / Checked / Approved ka khaali block. PDF landscape A4, har page par DRAFT footer.

### Parameters kahan se aate hain
- **Parameters = v1 model ke per-material speeds/feeds** (`week3/agents.py`, wahi numbers jo Time agent use karta hai; `capp/physics.py` ke raaste). Isliye estimated time = v1 cutting time + v1 ke set-up penalties (tool change 2, position change 1, changeover 15 min) aur **plan ka total `agents.time_agent` se match karta hai** (jab hole diameter nahi diya; test 12 dev parts par check karta hai). Route search ko kuch nahi chhua.
- **Reference ranges** (`capp/process_tables.py`): har (speed class, material) ke liye Vc aur feed range, har row par `status`: `web` (range ke saare bounds padhe hue pages se), `mixed` (ek bound page se, doosra typical practice), `indicative` (koi page nahi padha), `confirmed` (handbook [8] se tumne/supervisor ne check kiya). Sources S1-S4 (Aims Industrial, UFlorida MAE, UToronto drill press, incoherency lathe speeds) tables ke upar aur `tables` command me URLs ke saath hain. Plan ke parameters in ranges se compare hote hain: **"above" = check karo**, "below" = conservative (zyada time, kam wear; sirf info), range nahi = tapping/reaming/forming/engraving (koi range nahi), threading pitch se feed hoti hai (sirf Vc check).
- **Tool rules** (operation x material -> tool type + tool material): general machining practice, sab `indicative`.

### Jo honestly likhna hai (sheet me bhi likha hai)
1. **Koi row handbook se confirmed nahi hai** (handbook [8] mere paas nahi tha; `capp/feasibility.py` jaisa hi). Banner har page par "DRAFT" kehta hai aur ranges ke status counts chhapta hai; banner tabhi badlega jab use hui saari ranges `confirmed` hon. Confirm karne ka tareeqa: row check karo, `status = "confirmed"` aur `source_note` me page number likho (`capp/process_tables.py` edit karne ke liye hi bani hai; `tests/test_process_plan.py` table ko consistent rakhta hai).
2. **Web sources handbook nahi hain**: tool-maker aur course pages hain, aur aapas me carbide turning par ~2x tak alag hain, isliye range unhe span karti hai. Drilling feeds 6-12 mm drills ke hain. Koi bhi tumhare grade, hardness, tool ya machine ke liye nahi. Plastic drilling `indicative` hai (koi page nahi).
3. **v1 ke parameters ek real finding dete hain**: jaise Titanium drilling ka v1 Vc (20 m/min) HSS-Co range (6-14) se upar hai, to sheet "above: check" flag karti hai. Ye model ki kamzori hai, sheet ki nahi; thesis me likhne layak.
4. **Facing / Chamfering / Boring milling machine par** alag tool se hote hain (face mill, chamfer mill, boring head); sheet wo tool dikhati hai, par speed/feed v1 ke turning-type values hain, aur flag ye kehta hai ("set the milling feed per tooth at the machine").
5. **Assumed (sheet me `*` aur Assumptions)**: hole diameter na ho to hole operations part diameter lete hain; machine (7.5 kW, `feasibility.DEFAULT_MACHINE`) assumed hai, spindle load usi ka hai (rpm/torque limits modelled nahi); jis segment me koi machine-decider operation nahi use lathe maana jaata hai (part ki preference ho to wahi, bina `*`); stock = part diameter ki bar; milling cutter sizes v1 ke standard.
6. Estimated time shop-floor standard time **nahi** hai (batch set-up, waiting aur batch inspection nahi). Koi machining trial nahi hua.
7. Excel/PDF libraries optional hain (`requirements.txt` me nahi): na hon to saaf error "pip install ..." aata hai aur HTML (browser se Print -> PDF) hamesha chalta hai.

### UI
"Route Sheet" section dono modes me (Route Only par reference part par), table + DRAFT banner + download buttons (HTML / PDF / Excel). Download button `on_click="ignore"` se chalta hai, to click par app rerun nahi hota aur results gayab nahi hote (purane Streamlit par ye option nahi, wahan plain button). Sheet ban na paye to page chalta rehta hai ("could not be built").

### Tests ne jo bug pakda
Excel me freeze-pane / print-title row header se ek row upar thi (blank row ke baad ka row count galat tha): ab header append karke uski asli row li jaati hai. Aur feasibility lines me rule id gayab tha (reviewer ko pata nahi chalta kaunsa rule): ab `risk PHYS:HOLE-PILOT: ...` aata hai.

### Tests
`tests/test_process_plan.py` (53): tables complete (41 operations x 7 materials, 329 rows), validate() ke error cases, koi range "confirmed" nahi, range check within/below/above, machine assignment (segments, preference, mixed segment), plan route nahi badalta, **time = v1 Time agent** (haath ke route par aur 12 dev parts par), changeover next operation par charge, har operation x material, threading pitch feed, Titanium above flag, hole diameter given/assumed, Route Only reference part, unknown operation / material, feasibility flags, justification rules, repair view, run records (confidence, flag threshold, reasoning, purane records). `tests/test_routesheet.py` (45): display rows, banner (status counts, "confirmed" par badalna), header, sections, HTML escaping + well-formed, Markdown table, JSON round trip, XLSX (numbers numbers hain, time column total se match, sheets), PDF (landscape, har page par DRAFT footer, awkward characters, library na ho to error), `write()`, CLI (make / from-run / tables, errors). `tests/test_app.py` (+3): dono modes me sheet + downloads, sheet fail ho to page chalta rahe.

### Abhi kya nahi hai
Handbook se confirmation (tumhare/supervisor ke haath me; ek-ek row `confirmed` karo), per-operation multi-objective parameter refinement (abhi route pehle, v1 parameters baad me), tool-life se tool selection, machining trials, baaki dimensions ka extraction, confidence/flag ka dashboard me dikhna (sheet me hai, route cards me nahi), asli live results par threshold (Step 18) aur cascade ke results.

## Step 20 (WP7): extraction front end, agreement-based confirmation, corrections log aur M9, `scripts/extraction_eval.py`

Report §6.5 / Objective 6 / RQ6: drawing se features nikalne wala front end sirf "AI ne kya bola" nahi dikhata, balki **kis par bharosa karna hai** bhi batata hai. Do passes (geometry, annotation) ek hi drawing padhte hain. **Dono ne jo feature dekha wo pre-accepted** (confidence high). **Sirf ek pass ne dekha to highlight**: jab tak Accept ya Reject nahi, **Generate band**. User jo badalta hai wo **corrections log** me jaata hai, aur **M9** (precision / recall / F1 per pass, union, intersection, agreement-based + har drawing par corrections) ek script se nikalta hai.

### Workflow (repo root, PowerShell)
```
python -m streamlit run week6/app.py        # drawing upload -> "Extract Features" -> main area me review panel (drawing + table + Accept/Reject)

# M9: pehle ground truth (drawings ka folder, jaise drawings/)
python scripts/extraction_eval.py template drawings/ --out data/extraction_truth.json     # har drawing ke liye khaali entry; "features" (aur ho to "hole_diameters") bharo
python scripts/extraction_eval.py run drawings/ --truth data/extraction_truth.json --tag fe1                       # live API calls (cache me), phir score
python scripts/extraction_eval.py run drawings/ --truth data/extraction_truth.json --tag fe2 --evidence --dimensions   # evidence prompt + hole-diameter pass
python scripts/extraction_eval.py run drawings/ --truth data/extraction_truth.json --tag fe1 --mode replay           # sirf cache se
python scripts/extraction_eval.py score results/fe1_extraction results/fe2_extraction --csv results/m9              # offline, kitni bhi baar
python scripts/extraction_eval.py log                                                                              # asli users ne kya badla (results/corrections_log.jsonl)
```
Truth file: `{"a.png": ["Hole", "Slot"], "b.png": {"features": ["Hole"], "hole_diameters": [8, 6.5]}}` (key me extension chhod sakti ho). Feature names 36-feature vocabulary se hone chahiye, warna saaf error aata hai; khaali entry (template jo bhari nahi) bhi refuse hoti hai. API calls: har drawing par 2 (+1 `--dimensions`); `--evidence` alag prompt hai, to nayi calls lagti hain.

### Interface me kya badla
- **Review panel** (main area, drawing ke saath): table me Feature, Seen by (both passes / geometry only / annotation only), Confidence, Status (pre-accepted / needs confirmation / accepted / rejected), Evidence. Highlighted feature ke saamne **Accept / Reject**. Decision badal sakti ho. Multiselect me highlighted feature khud jodna bhi acceptance gina jaata hai.
- **Generate** disabled rehta hai jab tak koi highlighted feature undecided hai (sidebar me warning + caption batati hai kaun sa). Manual mode kabhi block nahi hota.
- **Confidence** = dono passes dekhe to "high"; ek pass dekhe to us pass ki apni confidence **medium par cap** (doosre pass ne nahi dekha).
- **Unrecognised names** (vocabulary ke bahar, jaise "Counter Bore") chupchaap drop nahi hote: panel me nearest feature ke saath "Use Counterbore" button (sirf suggestion, user decide karta hai); close match na ho to note.
- **Purane results** (jisme per-pass data nahi) pura pre-accepted rehte hain, pehle jaisa. Agar sirf ek pass chala to koi feature "dono ne dekha" nahi ho sakta, to sab highlighted aur panel kehta hai "Only 1 of the 2 passes worked".
- **Evidence (opt-in)**: sidebar checkbox "Ask the AI for a short evidence note per feature", default **off**. Alag prompt hai (har feature ke saath annotation text ya view, max 12 words), to **default prompt, uske cache keys aur recall bilkul nahi badle** (test: default prompt ka hash Step 19 wale jaisa, `33c64a8869bd6e13`). Evidence na ho to table me passes ke notes dikhte hain.

### "Correction" kya gina jaata hai
**Correction** = user ne feature **jodi** (model ne nahi di), **pre-accepted feature hataya**, ya unrecognised name **map** kiya. **Highlighted feature par Accept / Reject ek decision hai, correction nahi**: interface ne khud maanga tha. Dono alag gine jaate hain, kyunki RQ6 poochta hai ki agreement rule corrections ghataata hai ya nahi, aur uski keemat decisions ki ginti hai. Corrections log (`results/corrections_log.jsonl`, `CAPP_CORRECTIONS_LOG` se badal sakti ho; `results/` gitignored hai, to push nahi hota) me har Generate par ek line: model ka original output (union, agreement, kis pass ne dekha, highlighted, unrecognised), user ki final list, events, `n_corrections`, `n_decisions`, drawing ka naam aur **short hash** (drawing khud store nahi hoti). Same drawing same state me dobara Generate karo to doosri line nahi banti (material badalne se drawing do baar nahi giniyegi). Log likhne me dikkat ho to page chalta rehta hai.

### M9 ke variants (`capp/extraction_eval.py`)
Har drawing par G = truth, g / a = dono passes, u = g ∪ a (v1), b = g ∩ a. Sab variants **sirf unhi drawings par** jahan dono passes chale (taaki same drawings par compare ho; baaki "left out" me naam ke saath):

| variant | dikhayi gayi list | corrections / drawing |
|---|---|---|
| geometry, annotation | ek pass akela | list ko G banane ke edits (galat hatao + chhoote jodo) |
| union (v1) | u | wahi |
| both (intersection) | b | wahi |
| agreement | b pre-accepted, u − b highlighted | `|b − G| + |G − u|` (galat pre-accepted hatao + dono ne jo chhoda wo jodo); `|u − b|` decisions alag "confirmations" me |

Pooled precision / recall / F1 (95% interval drawings par bootstrap, seed 7), drawing-average F1, exact-match share, per-feature table (support ke saath), passes ka overlap ("dono ne dekha: itna sahi; ek ne dekha: itna sahi"), unrecognised names, hole diameters (dimension pass, 0.05 mm ke andar match, har true diameter ek baar), CSV (`_variants`, `_features`, `_drawings`; drawings me `missed_by_both` bhi).

### Jo honestly likhna hai
1. **M9 abhi nikal nahi sakta: repo me annotated drawings nahi hain** (sirf `sample_drawings/flange_plate.png`). 30-50 drawings aur unki ground truth (tumhe ya Prof. Deb ko likhni hai; do annotators ek subset par ho to agreement ceiling bhi milti hai). Module aur CLI synthetic records par haath se nikaale numbers se test hue hain (3 drawings ka table test me comment me hai).
2. **"agreement" variant ek ORACLE upper bound hai**: maanta hai user har highlighted feature sahi accept/reject karta hai. Asli users ka number `log` command deta hai; report me dono ko alag likho. Corrections log me final list user ki hai, independent truth nahi (user galti miss kar sakta hai): wo "kitna badla" naapta hai, "kitna sahi" nahi.
3. **Dono passes ek hi model hain, sirf prompt alag**: unki agreement independent saboot nahi (galtiyan correlated ho sakti hain). Jo feature dono miss karein wo kabhi highlight nahi hoga; table me `missed_by_both` isi ke liye hai.
4. **Evidence prompt aur dimension pass maine real API par nahi chalaye** (fake transport par hi test hue); pehle live run me model ka format alag aa sakta hai (`_clean_evidence` aur parser bachaate hain, par recall naapna tumhara kaam hai). Evidence ke saath aur bina features ka farak `fe1` vs `fe2` score se dekho.
5. Feature list ek **set** hai (3 holes wali drawing me Hole ek baar), v1 ki tarah. Sirf hole diameters ki extraction hai; baaki dimensions (depth, width, Ra) abhi user type karta hai.
6. **Protocol**: prompts ek development subset par badlo, **freeze** karo, M9 baaki drawings par report karo. 30-50 drawings par har interval chauda hoga, wahi likho.
7. `use_container_width` jaise Streamlit ke deprecation warnings purane code ke hain, behaviour nahi badla.

### Tests (+177)
`tests/test_extraction_review.py` (64): agreement rule (both / one / legacy / counts-only / ek pass fail), confidence cap, evidence per pass, unrecognised + nearest feature, pending / decide / status, mapping, log entry ke har event type, log file (bad lines), summary aur report. `tests/test_extraction_eval.py` (57): truth file ke saare errors, template, drawings ka matching, `run` (fake extractor: records, options, exception, key-missing par turant stop, run tag restore), **3-drawing case haath se** (har variant ke tp/fp/fn, precision/recall/F1, corrections, confirmations, overlap), left-out drawings, holes, report, CSV, CLI. `tests/test_extraction_passes.py` (25): per-pass data, default call byte-same, default prompt hash, evidence prompt alag + cache entries alag, failed pass. `tests/test_review_ui.py` (31): Generate disabled / enabled, Accept / Reject, mapping, evidence, image, manual mode, log line ke events, duplicate nahi, log fail ho to page chale, evidence checkbox extractor tak pahunchta hai (file upload ke saath).

### Abhi kya nahi hai
Live runs aur annotated drawings (upar 1), asli users ka log, baaki dimensions ka extraction, dono passes ke alawa teesra independent pass ya alag model se cross-check, flag ko raw signals (agreement, valid share) par rakhna (Step 18 ke `confidence_tune` ke baad), dashboard route cards me confidence / flag ka dikhna.
