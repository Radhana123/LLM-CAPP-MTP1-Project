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

## Step 21 (WP3 analysis): risk-aware RB vs LLM comparison, `scripts/risk_compare.py`

Sawaal: `compare_runs` kehta hai "n5_r1_s3, rb se 3.19 pts peeche". Par v1 **Efficiency physical risk dekhti hi nahi** (time, cost, energy, changeover). DEV_016 jaisa pair: LLM ne `+Reaming` (tolerance 0.01 mm) joda, Efficiency 6.29 pts gira, par route risks 1 se 0. To gap ka kuch hissa **scoring artefact** ho sakta hai. Ye script wahi test karta hai: dono runs ke finals ko Efficiency **aur** feasibility flags dono se ek saath dekhta hai. Offline, API nahi, sirf `records.jsonl`. v1 Efficiency nahi badli.

```
python scripts/risk_compare.py results/dev_rb results/dev_n5_r1_s3                        # pehla folder reference (rb), doosra us par judge hota hai
python scripts/risk_compare.py results/dev_rb results/dev_n5_r1_s3 --physics --top 20 --csv results/risk_pairs.csv
python scripts/risk_compare.py results/dev_rb results/dev_n5_r1_s3+fz2 --penalties 0,1,3,10
```

### Kya nikalta hai (`capp/riskview.py`)
**Flags** = route ke hard infeasibilities + route risks (`capp/feasibility.py`); `--physics` se over machine power aur Ra risk "high" bhi. Ye findings ki **ginti** hai, severity nahi.
1. Mean Efficiency difference, dono taraf ke mean flags, mean reduction (rb ke flags minus LLM ke flags).
2. **3 x 3 table**: LLM Efficiency me better / equal / worse x LLM ke flags kam / barabar / zyada (har cell me pairs aur mean Efficiency diff).
3. **"LLM worse par flags kam"** wale pairs: kaun se rules clear hue (`TOL-HOLE`, ...), LLM ne kaun se operations jode (`Reaming`...), har pair ek line me (Efficiency, flags a -> b, `+op/-op`). Ulta group bhi ("LLM better par flags zyada").
4. Pairs ka split: **neither** route par flag / sirf rb par / sirf LLM par / dono par, har group ka mean Efficiency diff. **"neither" row barabar ki zameen par comparison hai**: wahan koi flag value karna hi nahi.
5. **Penalty view**: adjusted diff = Efficiency diff + P x (rb ke flags - LLM ke flags), P = 0, 1, 2, 5, 10 Efficiency pts per flag, har ek ka 95% interval, aur **break-even P\***: ek flag kitne pts ka ho to LLM aur rb average me barabar. P\* bada ho to gap sirf tab band hota hai jab flag ko bahut value do; "none" ka matlab LLM ke average me flags kam hi nahi, to gap kisi P se band nahi hota (tab hypothesis us run ke liye galat hai, ye bhi result hai).
6. Per-rule table: kis rule par kitne pairs me rb/LLM ko flag, kitne cleared / added, aur tab ka mean Efficiency diff.

### Jo honestly likhna hai
1. **P ek value judgement hai, estimate nahi.** Script nahi bata sakti ek flag ki asli keemat kya hai (thesis me tumhe ya Prof. Deb ko argue karna hai). Tabular output batata hai "P ke har value par tasveer kaisi dikhti hai".
2. **Flags indicative rules ke model findings hain**: `capp/feasibility.py` ke rules ke sources abhi confirmed nahi (Step 11). Cleared flag = model ka finding cleared, shop-floor ka verified risk nahi. Aur ginti hai: do chhote risks ek bade hard infeasibility ke barabar gine jaate hain.
3. **Intervals parts par resample hote hain** (ek part ke repeats saath rehte hain, independent nahi), seed 7, 1000 resamples. 20-30 parts par interval chauda hoga; chhote n par isse conclusion nahi, tasveer lo.
4. Sirf unhi pairs me jahan dono finals strict-valid hain **aur** dono runs me feasibility analyse hui (baaki "left out" me gine jaate hain, `--physics` me physics data bhi chahiye).
5. **Mere dekhe numbers (jaise "fz2 ne 10 pairs me 8.3 pts khoye, break-even ~8 pts per flag") test ke nakli planner (`tests/test_partdiff.py` ka messy planner) ke hain, tumhare asli run ke nahi.** Asli run pe apna output dekho; DEV_016 wala pattern wahan sach hai ya nahi, ye script se pata chalega.
6. Ye comparison hai, causation nahi: cleared flag aur gira Efficiency saath dikhte hain, par ye nahi ki ek ne doosre ko "kharid" liya (extra operation dono ka reason ho sakta hai).

### Tests (+34)
`tests/test_riskview.py`: flags (hard + risk, physics, bool/missing), kaun se pairs analyse hote hain (not comparable aur no-data left out), **6 pairs haath se**: overall, 3 x 3 table, trade pairs ke rules/ops, split, penalty table (har P ka mean aur better/equal/worse), **break-even exactly gap band karta hai**, per-rule table; break-even ke special cases (not worse / never / none), **intervals parts par resample hote hain** (3 repeats wala part: sirf -10 / -7.5 / 0 aate hain, pair-level resample se -5 aur -2.5 bhi aate), reproducible seed, ek part par interval nahi par report, physics flags, khaali cases, report ke sentences, `--top`, CSV, runs swap karne par sab ulta, do asli harness runs par `part_diff` se agreement (pairs, mean, better/equal/worse, flags ke means), tables ka jod, run vs khud (kuch nahi badalta), CLI (report, CSV, `--physics`, `--penalties`, errors).

### Abhi kya nahi hai
Risk ko Efficiency ke andar daalna (v1 score ka badalna, alag design decision hai), severity-weighted flags, asli run ka output (tumhare `results/` par chalana hai), dashboard me ye view.

## Step 22 (WP11 prep): Table 4 regenerator, `scripts/make_tables.py`

Report 7.3-7.4: "har experiment ek config file se define hota hai ... aur sab tables aur figures **ek command se** regenerate ho sakte hain". Ye wahi hai. Ye script **koi API call nahi karti aur koi run shuru nahi karti**: sirf `results/<tag>_<system>/` ke finished runs padhti hai aur Table 4 (A0-A9), paired tests, figures banati hai. Runs `scripts/run_eval.py` se hi bante hain (`commands` step exact lines deta hai).

```
python scripts/make_tables.py status   configs/ablation_template.json    # kaun sa run hai, kaun sa kaam ka hai
python scripts/make_tables.py commands configs/ablation_template.json    # jo runs baaki hain unki run_eval lines (PowerShell)
python scripts/make_tables.py build    configs/ablation_template.json    # results/tables/ablation_dev/ me sab kuch
python scripts/make_tables.py build my_grid.json --out tables_final --strict --no-figures
```

### Config (`configs/ablation_template.json`)
Dataset, `limit`, `repeats`, 3 LLMs (`id`, `label`, `model`, `tag`) aur har row ki ek line (`id`, `label`, `system`, `parent`). `parent` = jis row me ye ek component jodti hai; paired comparison usi se hota hai. Template dev set par hai (20 parts, 3 repeats). **Final grid ke liye copy banao**: `"dataset": "test_v2"` (ya `bench_v1`), `"limit": null`, `"repeats": 5`, aur sirf final run (WP11) me chalao.
Rows A0-A8 report ke hisaab se: `b1`, `b0`, `b0_r1`, `b0_r1+ts+h1`, `b0_r1+ts`, `n5_r1_s3+ts+h1`, `...+fs2`, `...+fz2`, `...+cas1`; A9 (constrained planner) = `n5_r0_s1+cg1` (Step 24).
**Faisle jo tumhe lene hain** (config me `note` me likhe hain): A5 me A3 ya A4 (hint ke saath ya bina), A6 me k = 2 ya 4, A7 me fz1 ya fz2. Ye dev set par A3/A4 ke results dekh kar tay karo, **sealed set dekhne se pehle**. Sealed set ke results dekhne ke baad koi row, prompt ya setting nahi badal sakti (`commands` sealed dataset par WARNING chhapta hai).

### Ek run kisi cell me tabhi ginta hai jab...
records aur `run.json` dono kahein ki wo wahi system hai, wahi dataset, wahi model (`CAPP_MODEL_PRIMARY` jo column ke `model` se milna chahiye), aur parts ke features/material dataset se match karein. Warna wo cell `–` hota hai aur `issues.txt` me wajah aati hai (galat system/dataset/model, duplicate (part, repeat), corrupt file ...). **Adhoore run** (kam repeats, kuch parts gayab) table me **†** ke saath aate hain aur footnote batata hai; `--strict` me koi bhi issue exit code 1 deta hai. `+ms`, `+ml`, `+cas1` jaisi rows apne models khud chunti hain: ek hi run (`shared_tag`) har column me dikhta hai.

### Kya nikalta hai
`table4.md` / `.csv` / `.tex` (har LLM ke liye ek table; LaTeX booktabs fragment), `paired.md` / `.csv`, `consistency.md`, `per_part.csv`, `issues.txt`, `fig_ablation.png`, `fig_cost.png`, `manifest.json` (config ka hash aur har run ki records file ka hash; same runs + config = byte-identical text files).
Metrics wahi jo `run_eval` ke summary me (`summary.summarize_set`): **M1** first-pass valid, **M2** final valid aur **M2 LLM-final** (kitne finals LLM route hain, Route Builder fallback nahi), **M4** similarity, **M5** ops kept (LLM ke kitne operations repair/replacement ke baad bache), **M6** efficiency gap (pts), **M7** API calls/part aur USD/part. Cell = mean +- SD over repeats (ek repeat ho to sirf value).
**Paired tests (7.4):** har row apne parent se, parts par paired: continuous metrics par bootstrap 95% interval (parts resample hote hain, ek part ke repeats saath), validity par McNemar exact. Phir `consistency.md`: har LLM ka verdict aur "sab LLMs me same hai ya nahi".

### Jo honestly likhna hai
1. **Mere dekhe numbers nakli planner ke hain** (tests ke messy planner wale 3 "LLMs"), tumhare asli runs ke nahi. Asli Table 4 tabhi banega jab tumhare live runs aayenge.
2. **M2 "final valid" lagbhag hamesha 1.0 hota hai by construction** (Route Builder fallback). Isliye validity ka asli number **"M2 LLM-final"** hai, aur McNemar bhi usi par (aur raw samples ki M1 par) chalta hai.
3. **M4 dev set par "proxy_rb" hai**: rule-based plans se closeness, expert se nahi. Table ke upar ye likha aata hai. Expert reference sirf `bench_v1` par hota hai.
4. **McNemar me repeats > 1 ho to pairs part ke andar clustered hain**, to p-value approximate hai; asli saboot parts par bootstrap hai. Aur **8-10 comparisons ke liye koi multiple-comparison correction nahi** hai.
5. 20 parts par intervals chaude honge; "no clear difference" ka matlab "koi farq nahi" nahi, "ye data kuch nahi keh sakta" hai.
6. **A8 (cascade) ek alag model setup hai** (small phir large), to A8 vs A7 cascade ka asar aur model ka badlav mila hua hai. Isko `scripts/cascade_pareto.py` (validity vs cost) ke saath padho.
7. USD list price par hai (`capp/config.py`, abhi [TO CONFIRM]); jis model ka price nahi, uska USD `n/a`, kabhi 0 nahi. Purane records (Step 17 se pehle) ke bina model-id wale calls column ke model ke hisaab se gine jaate hain.
8. Figures sirf 3 LLMs tak (validated palette 3 series ke liye hai); chautha LLM tables me rehta hai.
9. **API budget:** template ke size par har LLM ke liye lagbhag 2,100 naye calls (20 parts x 3 repeats x ~35 calls: `b0` 5, `ts` rows 10 + 10, `fs2` row 10; `b1`, `b0_r1`, `n5_r1_s3`, `fz2` same cache se, koi naya call nahi). Ye mera andaza hai, Groq console se check karo; `--limit` aur `repeats` config me kam kar sakti ho.

### Tests (+125)
`tests/test_tables.py`: config (template A0-A9 report ke hisaab se, har system parse hota hai, 25 galat configs ke reasons), **haath se nikle statistics** (McNemar ke exact values, paired counts, bootstrap ke special cases, verdict ki direction), cells `summarize_set` ke barabar (mean aur SD over repeats, M5 do tarah ki fix ka pooled), API calls summary ke barabar, purane records ka cost column ke model par, run-integrity (galat system/dataset/model, kam repeats/parts, duplicate, features dataset se alag, corrupt, analyse na ho sake), shared run, paired aur consistency labels, missing/partial/not-built rendering, Markdown/CSV/LaTeX (escaping), manifest hashes, **byte-identical regeneration**, figures (PNG, palette, 4th LLM), `commands` (template par 28 lines, cache ka order, partial/unusable runs, sealed warning), CLI (`status`, `commands`, `build`, `--strict`, bad config).

### Abhi kya nahi hai
Asli runs ka Table 4 (tumhare live runs), M9 ka table (annotated drawings chahiye), mixed-effects ya multiplicity correction.

## Step 23 (WP8): failure case studies, `scripts/case_studies.py`

Report 7.5: har main M3 failure type ke 3-5 cases, ek **fixed rule** se chune hue, har case me part + features, LLM route, violation report, repair, final route aur ek chhota commentary. Ye script **koi API call nahi karti**: sirf finished runs (`records.jsonl`, `run.json`) padhti hai. **Facts script nikalti hai; commentary (model kyun fail hua, repair ne sahi decisions bachaye ya nahi) tumhe likhni hai** — wo ek alag file me rehti hai aur regenerate karne par kabhi overwrite nahi hoti.

```
python scripts/case_studies.py results/n5_r1_s3_ts_h1_qwen
python scripts/case_studies.py results/b0_qwen results/n5_qwen --per-type 4
python scripts/case_studies.py results/n5_qwen --types order,missing,differs
python scripts/case_studies.py results/n5_qwen --instructive "DEV_016:1:0=repair ne ek galat boring-before-drilling decision rakha"
python scripts/case_studies.py results/n5_qwen --out case_studies_final --strict
```

### Selection rule (jo report me likhna hai, wahi code me hai)
- **Failure** = ek raw LLM route (ek repeat ka ek sample) jiska strict check us type ka violation de: `order`, `missing`, `invented`, `duplicate`, `machine_grouping`; ya jiska physical-feasibility check `infeasible` finding de (iske liye run `analysis` ke saath bana hona chahiye, default hai). Ek route do tarah se fail ho to dono types ka candidate hai.
- **Har type me**: parts ko id ke natural order me lo (DEV_002, DEV_010 se pehle), har part ka **pehla** failing route (repeat, phir sample id), aur pehle K parts. K = `--per-type` (default 3, 1-10; report 3-5 maangti hai). **Ek part ek type me ek hi case.** Koi aur cheez decide nahi karti: na metric, na ye ki repair chala ya nahi.
- **`differs`**: final route strict-valid hai par kisi reference route se exact match nahi. Expert routes ho to unse, warna Route Builder variants se (**proxy**: tab case batata hai rule-based plan se kahan alag hai, expert se nahi; har case me ye likha aata hai).
- **`--instructive PART[:REPEAT[:SAMPLE]]=wajah`**: tumhara chuna hua case. **Wajah zaroori hai**; case "instructive, chosen by the author" label ke saath alag section me aata hai, rule ke cases ko nahi chhuta. Sample na do to us part ka pehla invalid route.

### Kya nikalta hai (`--out`, default `<results>/case_studies/<run>`)
`case_studies.md` (selection rule, har run ka **M3 aggregate table**: har type kitne raw routes me, kitne % me, kitne parts me, kitne cases; phir har case), `cases.json`, `selection.csv`, `commentary.json`.
**Har case me**: part (material, features, tolerance, batch, archetype, expert reference hai ya nahi), LLM route (jo operation repair ne hataya wo ~~kata hua~~), strict violations (rule id + message, soft wale "(soft)"), feasibility findings (hard/soft), repair ka outcome (R1 / R2 / R0 ke baad kaun sa stage chala), R1 ne kya hataya/joda/reorder kiya (feasibility ke liye joda alag likha, flagged findings bhi), R2 ka jawab accept hua ya nahi, repair ke baad ka route (jodi hui ops **bold**), part ka final route (ye wahi sample hai ya doosra), aur "Facts" ki list: kitne LLM operations same order me bache (survived share), edit distance, final route ki Efficiency best Route Builder variant se kitni neeche, reference se LCS ratio.
Jis case ki commentary nahi likhi, usme `_TO WRITE: ..._` ka nishaan rehta hai. `commentary.json` me `{case id: text}` hai; nayi cases ke liye khaali entry jud jaati hai. Rule badalne se jo purani commentary kisi case se match nahi karti wo file me rehti hai aur report me gini jaati hai. `--strict`: koi case bina commentary ka ho to exit 1 (report me daalne se pehle chalao).

### Jo honestly likhna hai
1. **Mere dekhe cases nakli planner ke hain** (tests ka messy planner, jo har tarah ki galti banata hai), tumhare asli runs ke nahi. Asli cases tumhare live runs se aayenge; commentary tumhari hogi.
2. **Facts "kya hua" batate hain, "kyun" nahi.** "Repair ne sahi decisions bachaye" ka mechanical hissa (survived share, kya hata/juda) script deti hai; ye ki wo decisions sahi the ya nahi, engineer ka judgement hai.
3. **Dev set ke cases report me tabhi jaayein jab clearly dev kahe ja rahe hon.** Sealed dataset (`test_v2`, `bench_v1`) ke run par report me note aata hai: 7.5 ke hisaab se wahan ke cases **final run se** hi lo, configuration freeze hone ke baad.
4. `infeasible` cases tabhi milte hain jab records me `raw_feasibility` ho (`analysis=True`, jo default hai). Bina analysis wale run me table me "no feasibility analysis in these records" aata hai, case nahi.
5. `differs` ka matlab "expert se alag" tabhi hai jab part me expert reference route ho (`bench_v1`). Dev set par ye Route Builder se alag hona hai; kai baar LLM ka alag route bhi sahi hota hai (feasibility repair ne `Reaming` jod diya jo RB variant me nahi tha), isliye ise failure nahi, "dekhne layak" cases samjho.
6. Ek hi route do type me aa sakta hai (jaise order aur infeasible); aggregate table me wo dono me gina jaata hai, to type ki ginti ka jod raw routes se zyada ho sakta hai.
7. Rule "pehle K parts" hai, to cases dataset ke shuru ke parts ki taraf jhukte hain. Report me ye line likh do ya `--instructive` se alag parts jodo (label ke saath).

### Tests (+89)
`tests/test_casestudy.py`: natural order, `--instructive` parsing (aur 7 galat forms), route diff, repair outcome ke shabd, R1/R2 facts, run loading ke errors, **selection rule har type par records se alag likh kar match** (pehla failing route, id order, ek part ek case, records ka order shuffle karne par same), `per-type` aur types ki validation, ek route do types me, soft `machine_grouping`, diff/survival metrics se match, violations verifier ke shabdon me, final route block, expert vs proxy references, **R2 accept hone aur refuse hone (R0 par girna) wale asli harness runs**, aggregate table `summarize_set` aur `per_candidate.csv` se match, bina analysis ke run, Route Builder run, `differs` (expert/proxy, invalid final kabhi nahi), instructive cases ki labelling aur errors, commentary (rehti hai, nayi entries, orphan, bad file, whitespace), deterministic Markdown, sealed note, teen output files (LF newlines), CLI (files, commentary bachna, default out, `--strict`, errors par kuch adha nahi likha jaata). Maine 6 jaan-boojh kar bugs daal kar bhi check kiya ki tests unhe pakadte hain.

### Abhi kya nahi hai
Asli runs ke cases aur commentary, cascade runs ke liye alag case view (small ne kya kiya, large ne kya), cases ka docx/LaTeX output (Markdown se pandoc se ban jaata hai).

## Step 24 (WP10, row A9, RQ5): constrained / graph-guided planner, `+cg1`

Report 6.8 aur RQ5: "repair generation ke **baad**, ya constraint generation ke **dauraan**?" Ab dono ka paired comparison ho sakta hai: A2 (`b0_r1`, LLM route + repair) vs A9 (`n5_r0_s1+cg1`, route precedence graph ke andar hi banta hai). Ye ek **planner option** hai, to har cheez (harness, repair, selection, summary, tables, case studies, cost) pehle jaisi chalti hai.

```
python scripts/run_eval.py --system n5_r0_s1+cg1 --limit 20 --tag dev
python scripts/run_eval.py --system n3_r0_s3+cg1+fz2 --limit 20 --tag dev      # cg ke baad bhi repair/selection chal sakte hain
python scripts/make_tables.py commands configs/ablation_template.json            # ab 28 lines, A9 inme hai
```

### Design (`capp/constrained.py`)
- **Operation set fixed hai**: Route Builder ke pehle route ke operations (Facing ... Inspection, markers hata kar). LLM koi operation jod ya hata nahi sakta.
- **Har step par frontier** = bache hue ops jinke prerequisite (precedence graph `pg.PRECEDENCE_EDGES`, sirf is set ke andar) sab placed ho chuke. Facing sabse pehle, Inspection tabhi jab sirf wahi bacha ho. Options alphabetical.
- **LLM sirf tab poocha jaata hai jab frontier me 1 se zyada option ho** (63-64% steps forced hote hain, call nahi lagti). Jawab sirf ek number. Prompt me: ab tak ka route, last operation ki machine, aur wo ops jo abhi baaki hain; machine grouping ka hint bhi.
- **Jawab parse nahi hua** (khaali, number range se bahar, nonsense, ya do names): us step par Route Builder ke route ka sabse pehle wala option liya jaata hai, `fallback_steps` me ginta hai, **retry nahi**. API error par sample `success False` (jo calls hui unka meta saath).
- **Valid by construction**: 200 dev parts par 1200 random-choice routes me **0 hard violations** (sirf `machine_grouping` aur physical `risk` baaki reh sakte hain, ye soft hain).
- **`cg2` hataya**: physical order rules (HOLE-EXISTS, SEQ-PARTOFF) 3 tak ke saare 7806 feature sets ke liye graph se pehle hi implied hain, to alag level ki zarurat nahi.
- Meta me `cg` = `n_ops`, `set_source`, `choice_steps`, `forced_steps`, `fallback_steps`, `unsatisfiable`, aur har choice (`pos`, `options`, `choice`, `how` = forced/llm/fallback, `answer`). `n_calls` ke through cost/summary/tables khud sahi calls ginte hain.
- **Option combinations**: `+cg1` ko `+fs`, `+h1`, `+ts` ke saath jodna ValueError hai (prompts apne hain). `+fz1/2`, `+ms/+ml` aur `+cas` ke saath ban sakta hai.

### Jo findings rule base ke baare me mili (report me likhne layak)
1. **Route Builder ke paas kuch feature sets ka route hi nahi**: [Turning, Slot, Hole, Chamfer] jaise. Wajah: lathe-first construction me shared op Chamfering lathe phase me aa jaata hai, par graph ke hisaab se Slot Milling uske pehle aana chahiye. 3 tak ke saare 7806 feature sets ke route hain; random 4-feature me ~0.4% (6/1500), 5-feature me ~0.3% (5/1500) ke nahi. Dev set ka koi part affected nahi. Aise sets par constrained planner **graph se set banata hai** (Facing + har feature ki pehli alternative chain + Inspection, `meta.cg.set_source == "graph"`, summary me gini jaati hai); un par "same ops as RB" wali baat lagu nahi hoti.
2. **RB ka apna route kabhi HOLE-EXISTS tod deta hai**: Chamfer+Groove+Pocket aur Chamfer+Groove+Slot ke route me Internal Grooving hai par Drilling/Boring nahi. Constrained planner ke `unsatisfiable` record me ye aata hai (`HOLE-EXISTS:<op>`), summary me count. Ye wahi precedence edge `Drilling -> Internal Grooving` ka sawaal hai jo pehle se pending hai.

### Cost (estimate, apni Groq console se match karo)
Dev set par ek route me ~6.2 ops, ~**2.1 real choice calls** (0 se 6; 12/200 parts me koi choice hi nahi). Report ka "ek call per step" assumption asal me ~2 per route hai. A9 N=5 samples ke saath ~10 calls per part, to template size (20 parts x 3 repeats) par ek LLM ke liye ~630 extra calls. Har call chhoti (jawab sirf ek number).

### Jo honestly likhna hai
1. **Saare numbers nakli transport se hain.** Asli API par kya hoga ye test nahi hua. Khaas risk: **reasoning models** (qwen3 jaise) apne 200 `max_tokens` thinking me kharch kar sakte hain aur jawab khaali aayega, jo `fallback_steps` ke roop me dikhega (plan RB jaisa ban jaayega, LLM ka koi yogdaan nahi). Pehle run ke baad `summary.json` ka `constrained.fallback_share_of_choices` dekho. Zyada ho to `capp/config.py` me `CG_MAX_TOKENS` badhao.
2. **LLM ka yogdaan chhota hai**: dev set par ~2 real choices per route, baaki forced. Isliye A9 ka validity ~100% aana expected hai, ye ek design property hai, LLM ki quality nahi. Asli sawal ye hai ki **choices ki quality** (Efficiency, machine grouping, M4) A2 se behtar/kharab hai ya nahi.
3. **Ye op add nahi kar sakta.** Physical risks jo extra operation maangte hain (TOL-HOLE ke liye Reaming) rehte hain; `+fz2` baad me Reaming jod sakta hai (test me dikhaya).
4. **Valid by construction sirf precedence graph ke liye hai.** Graph ke bahar ki cheezein (physical feasibility, risk) alag checks hain.
5. A9 template row **optional** hai (note me likha). Full pipeline se compare karna ho to `n5_r0_s3+cg1` ki row A5 ke baraabar jodo.

### Tests (+67)
`tests/test_constrained.py`: grammar aur rejections, `required_ops` = `planners.rb_plan` (saare dev parts), edges/frontier/cycle/machine classes, physical rules graph se implied (singles aur pairs), Internal Grooving gap, `parse_choice` (word boundary, nested names, out of range), question ka content, **validity har dev part par first/last/random policy se**, always-first = alphabetical topological order, calls = sirf real choices, meta sums, zero-choice parts, model id passthrough, unreadable jawab -> RB route, think blocks, API errors, replay mode, factory/roles, machine preference, **graph se set (RB route ke bina)**, vocabulary ke bahar ka feature, cycle error, aur asli harness runs (`n1_r0_s1+cg1`, `n3_r0_s3+cg1`, `n3_r1_s3+cg1+fz2`, `b1`): summary block records se match, M7, two-step se confuse nahi, text summary/compare rows, selection, repair ke baad Reaming, tables option jaanta hai, case study cg run par chalta hai. 9 jaan-boojh kar bugs daal kar bhi check kiya ki tests unhe pakadte hain. `test_tables.py` me A9 ab built hai (28 commands).

### Abhi kya nahi hai
Asli API par koi run (reasoning model ka behaviour dekhna baaki), choice quality ka M4 se analysis, `cg` ke saath `+fs`/`+ts` (nahi banaya, prompts alag hain), graph ke baahar ke constraints (feasibility) generation ke dauraan.

## Step 25 (WP0 / WP11): configuration freeze, `scripts/freeze.py`

Report 7.2: test benchmark "sirf final runs ke liye, jab saare components frozen hon, aur test results dekhne ke baad koi prompt ya setting nahi badalti". Pehle ye sirf ek warning thi. Ab ye **checkable** hai: ek manifest (`FREEZE.json`) jo likhta hai ki final run kis cheez se bana hai, aur ek `verify` jo batata hai ki freeze ke baad kuch badla ya nahi. **Koi API call nahi, koi run shuru nahi.**

```
python scripts/freeze.py create --grid configs/ablation_template.json --run-tests --note "final" [--extra results\threshold.json] [--name tier1]
git add FREEZE.json; git commit -m "Freeze"; git tag freeze-v1
python scripts/freeze.py verify                  # exit 0 ok / sirf analysis code badla; exit 1 drift; exit 2 manifest nahi mila
python scripts/freeze.py verify --strict         # analysis code badalne par bhi exit 1
python scripts/freeze.py show
python scripts/freeze.py diff FREEZE_tier1.json FREEZE.json    # do freezes ke beech kya badla
```

### Kya fingerprint hota hai (SHA-256; line endings normalise, to Windows aur Linux checkout ka hash same)
- **files**: `capp/`, `week1`-`week6`, `scripts/` ke top-level `.py` aur `.json` (tests nahi). Har file ka **kind**: `analysis` (wo code jo sirf finished runs padhta hai: tables, case studies, summary, route sheet, benchmark tools, UI, saari scripts) ya `behaviour` (baaki sab; **nayi ya anjaani file behaviour maani jaati hai**).
- **config**: `capp/config.py` ke saare UPPER_CASE constants (temperature, max_tokens ...), har role ka model id **jaisa abhi resolve hota hai** (to `CAPP_MODEL_PRIMARY` freeze se alag set ho to drift; **ek apwaad**: primary model grid ke `models` me se koi bhi ho to drift nahi, kyunki 3 LLM ke runs ek ke baad ek isi tarah chalte hain, dekho niche "Step 26 me fix"), structured-output mode. Prices sirf record hoti hain, compare nahi (wo baad me token ko cost banati hain).
- **datasets**: dev_v2, test_v2, bench_v1 (jo exist karein) + `--dataset`. Hash + parts ki ginti.
- **grids**: final runs ka ablation config (canonical JSON, indentation se fark nahi). Rows ki list (kaun sa system) is freeze ka hissa hai, isliye `--grid` zaroori hai.
- **extras**: koi aur file (`--extra`): tuned confidence threshold, A5/A6/A7 ka decision.

### Verdict
| status | matlab |
|---|---|
| `ok` | kuch nahi badla |
| `analysis-changed` | sirf analysis code badla (table bug fix wagairah): allowed, report hota hai |
| `drift` | behaviour file, config constant, model id, dataset, grid ya extra badla (nayi behaviour file bhi): results frozen configuration ke nahi |
| `none` / `unreadable` | manifest nahi hai / padha nahi ja raha (sirf run.json me) |

Freeze ke baad **bench_v1 ka nayi bann jaana drift nahi** (information hai); dataset badalna ya hatna drift hai.

### Kahan use hota hai
1. **`run.json`** (har run, dev ho ya sealed): `freeze` = `{status, id, manifest, differences (pehli 10)}`.
2. **`run_eval.py` sealed dataset par** (`test_v2`, `bench_v1`, LLM system): manifest nahi hai ya `drift` hai to **STOPPED, exit 2**, koi call nahi. `--freeze FILE` se dusra manifest, `--no-freeze-check` se jaan-boojh kar override (loud warning; status phir bhi run.json me jaata hai). `rb` system aur dev set par gate nahi.
3. **`make_tables.py`**: sealed config ke har run me freeze status `ok`/`analysis-changed` nahi hai (ya run.json me status hi nahi) to issue (warning); alag-alag freeze ids ke runs ek grid me hon to issue; manifest.json me har run ka `freeze` aur `freeze_id`; `commands` sealed config par freeze karne ki line bhi chhapta hai.

### Kab freeze karna hai (abhi nahi)
- **Gate G2** (Tier-1 features frozen): dev set par A3/A4 (hint ke saath ya bina), A5 (k, selection) tay karke. `--name tier1`.
- **Gate G4**: baaki rows (A6, A7, A8, A9) tay hone ke baad, **test_v2 / bench_v1 par pehla LLM run chalne se pehle**. Ek aur freeze, `diff` se dikha sakti ho ki tier1 ke baad kya jud gaya.
- **Abhi mat karo**: aage ke steps nayi behaviour files jodenge to wo drift ginenge (`create --force --note ...` se dobara freeze karna padega; purana id `replaces` me rehta hai).
- **`baseline-v0` tag**: report kehta hai foundation system "kisi bhi modification se pehle" tag hona chahiye. Agar tag nahi laga to `git log --oneline --reverse` me wo commit dhundo jahan MTP-1 ka internship code aaya tha: `git tag baseline-v0 <hash>; git push origin baseline-v0`. Mujhe nahi pata kaun sa commit hai, isliye maine ye nahi kiya.

### Jo honestly likhna hai
1. **Ye saabit karta hai ki files wahi hain jo freeze me thi; ye nahi ki tumne freeze se pehle test results nahi dekhe.** Order of events (freeze, phir pehla sealed run) manifest ki `created_utc`, run.json ke `started_utc` aur git tag se dikhta hai, par imaandari tumhari hai. Report me aise likho: "the configuration was frozen (commit X, tag freeze-v1, manifest id Y) before the first run on the test set".
2. **Analysis code ko "allowed" maanna ek faisla hai**: tables ya summary me bug fix routes badal nahi sakta, par numbers badal sakta hai (galat metric theek karna). Isliye `analysis-changed` report hota hai, chhupta nahi. Metrics ka code (`capp/metrics.py`, `verify.py`) maine **behaviour** me rakha hai (conservative).
3. `scripts/` ko analysis maana hai (CLI wrappers). `run_eval.py` ke defaults jo planner ko chhoote hain wo `config.py` aur `systems.py` me hain, jo behaviour hain.
4. Hash sirf us tree ka hai jo chal raha hai: installed library versions (openai/groq SDK, numpy) nahi. `python` aur `platform` manifest me hain par compare nahi hote.
5. `--run-tests` do minute lagata hai; failing tests ke saath freeze refuse hota hai.
6. Mujhe `git tag` ya `git commit` chalane ka haq nahi hai tumhari machine par: script sirf batati hai ki kya chalana hai (dirty working tree ka warning bhi).

### Step 26 me fix (multi-LLM)
Pehle version me `CAPP_MODEL_PRIMARY` ko frozen primary se alag set karna drift tha. Lekin paper ke grid me 3 LLM hain aur har LLM ka run alag `CAPP_MODEL_PRIMARY` se chalta hai, to dusre aur teesre LLM par `run_eval` ka gate (aur freeze status) galat drift dikhata. Ab `grid_snapshot` har grid ke `models` bhi record karta hai, aur compare me primary model ko tab theek maana jaata hai jab wo kisi frozen grid ke `models` me ho. Grid ke bahar ka model abhi bhi `config MODELS` drift hai. **Isliye Step 25 ka zip freeze karne ke liye mat use karo: Step 26 ka (cumulative) zip use karo.** Fix ke tests `test_freeze.py` me (ab 71 tests; `run_eval` har grid-LLM par chalta hai, grid ke bahar ka model rukta hai).

**Step 26b (hotfix):** `run.json` me `freeze` status pehle hamesha repo ke `FREEZE.json` ka likha jaata tha, chahe `run_eval --freeze FILE` se dusra manifest check hua ho. Ab `run_eval` wahi manifest harness ko deta hai (`run_experiment(..., freeze_path=)`), jis se gate ne check kiya. Ek test (`test_the_route_builder_and_the_development_set_are_not_gated`) isi wajah se tab fail hota tha jab repo me asli `FREEZE.json` pada ho; ab test hermetic hai (asli `FREEZE.json` ke saath poora suite chala kar dekha). Naya test: `test_run_json_records_the_manifest_the_gate_used_not_the_one_of_the_repository`.

### Tests (+68, ab 71 fix ke baad)
`tests/test_freeze.py`: hashing (CRLF, JSON layout), kinds (real repo ki har analysis module exist karti hai), scan (sirf top-level `.py`/`.json`), config snapshot (real config env se badalta hai), datasets (naya/hata/badla), grid/extras (layout se fark nahi, bahar ke paths), pytest summary parsing, git info (asli git repo me dirty/tag), create (once, `--force` se `replaces`, grid zaroori, failing tests refuse, id state se banti hai time/note se nahi, sorted keys), **verify** (har behaviour file, nayi/hata hui file, analysis, tests/README/subpackage/`__pycache__` ka na badalna, Windows checkout, config/model/structured mode/price, dataset, grid, extras, manifest ka na milna), status (none/ok/analysis/drift/unreadable, 10 se zyada differences, kabhi raise nahi), do manifests ka compare, text output, CLI (create/verify/show/diff/strict/exit codes), **harness run.json**, **run_eval subprocess** (manifest nahi, drift, unreadable, ok, override, rb aur dev par gate nahi), **Table 4** flags aur manifest keys. 14 jaan-boojh kar bugs daal kar bhi check kiya, sab pakde gaye.

### Abhi kya nahi hai
Release ke time ka freeze report (Markdown) jo paper me chipka sako, aur WP12 ke baaki (datasheet pehle se hai Step 16 me).

## Step 26 (WP12): paper draft, `scripts/paper_draft.py`

Report 8.4 ka deliverable "paper draft" ka ek hissa: results ke saare **facts** (tables, paired comparisons, case studies, benchmark datasheet, M9, freeze id) draft me khud bhar jaate hain; **prose tumhari** hai, alag file me. Saath me ek **readiness check**: paper ke numbers use karne se pehle kya baaki hai. **Koi API call nahi, koi run shuru nahi** (`tables.analyse` read-only chalta hai).

```
# rehearsal (dev grid): kuch bhi tayyar na ho to bhi chalta hai
python scripts\paper_draft.py configs\ablation_template.json --out paper

# final form (ek hi line me)
python scripts\paper_draft.py configs\final.json --benchmark data\bench_v1.json --case-studies results\case_studies\final --extraction results\fe1_extraction --freeze FREEZE.json --tables-dir results\tables\final --out paper --strict
```
`--extraction` repeat ho sakta hai (har M9 run ke liye). `--strict` me blocking item khula ho to exit 1 (draft phir bhi likha jaata hai).

### Teen files (`--out`, default `paper/`)
| file | kya hai | regenerate par |
|---|---|---|
| `paper_draft.md` | Title se Conclusion tak, tables aur facts ke saath, aakhir me Appendix A (readiness), B (reproduce commands), C (table build ke issues) | har baar naya banta hai |
| `readiness.txt` | neeche wali checks | har baar naya |
| `paper_text.json` | **tumhari prose**: har section ek key (`title`, `authors`, `abstract`, `introduction`, `related_work`, `method`, `setup`, `rq1`..`rq6`, `case_studies`, `discussion`, `limitations`, `conclusion`, `acknowledgements` optional). Pehli baar khali strings | **kabhi overwrite nahi**: sirf missing keys jud'te hain; kuch jodna na ho to file ko chhoota bhi nahi |

Draft me har section ke niche tumhari prose aati hai; khali ho to `_TO WRITE: <section>_` placeholder (word budget aur hint ke saath). Unknown keys (orphans) file me rehti hain, ek note chhapta hai. `--text FILE` se prose kahin aur rakho.

### Script kya likhti hai aur kya nahi
- **Likhti hai**: configurations table (har row ka system, parent, option ka matlab), data (parts, sources), LLMs (id, price jahan hai), metrics aur statistics ka legend, reproducibility (freeze id, commit, commands), **Table 4** (ek hi code `make_tables` ke saath, isliye identical), headline numbers, har RQ ke paired comparisons (CI ke saath, McNemar), M3, M5, baseline failure table, further components, case study list, limitations ki **generated** list.
- **Kabhi nahi likhti**: koi vaakya jo batata ho ki numbers ka *matlab* kya hai ("improves", "proves", "confirms"). Ye test se pakda jaata hai (regex). Matlab nikalna tumhara kaam hai.
- **Headline rows**: *baseline* = pehli built row jisme data hai; *deepest* = jis row ki parent-chain sabse lambi hai (barabari par baad wali row). Ye rule results nahi dekhta, isliye "best row chun li" wala bias nahi.

### RQ mapping (kaun si row kis question ka jawab hai)
Har row ko apne parent se compare karke dekhte hain ki kya badla (`components`): samples, repair, selection, two-step, hint, few-shot, feasibility, cascade, model, constrained. Phir:
**constrained -> RQ5; two-step / hint / few-shot -> RQ2; repair -> RQ3; samples / selection -> RQ4; baaki (feasibility, cascade, model) -> "Further components"**. Agar ek row ek saath kai cheezein badle to upar wali priority jeetti hai (test se pinned). `ablation_template.json` par: A1:RQ4, A2:RQ3, A3:RQ2, A4:RQ2, A5:RQ4, A6:RQ2, A7 aur A8: further, A9:RQ5. RQ1 (baseline failure types) aur RQ6 (M9) alag data se aate hain.

### Readiness checks (`blocking` ho to `--strict` exit 1; draft phir bhi likha jaata hai)
| check | level | kab fail |
|---|---|---|
| `sealed` | blocking | grid dev set par hai (draft par "DEVELOPMENT RESULTS, NOT PAPER RESULTS" banner) |
| `llms` | blocking | grid me 3 se kam LLM |
| `runs` | blocking | kisi built row ka koi run incomplete / unchecked |
| `unbuilt` | advisory | kuch rows built nahi (Table 4 se bahar) |
| `errors` | blocking | table issues me error |
| `freeze`, `freeze_runs` | blocking | FREEZE.json nahi / drift; ya runs alag freeze ids ke (sealed grid par) |
| `input` | blocking | koi optional input (benchmark, cases, extraction) padha nahi gaya |
| `bench_real` | blocking | benchmark me synthetic PLACEHOLDER part (`source` me "placeholder", jo `benchmark.py` ke rehearsal se aate hain) |
| `bench_size`, `bench_review`, `m10` | advisory | 40 se kam parts / supervisor review baaki / do annotators ka agreement 15 se kam parts par |
| `bench_given` | blocking | grid benchmark par hai par `--benchmark` nahi diya |
| `m4` | blocking (benchmark grid) / advisory | M4 expert plans ke against nahi, `proxy_rb` hai |
| `cases`, `m9` | advisory | case studies nahi / commentary nahi; extraction 30 drawings se kam |
| `text_missing`, `text_placeholders` | blocking | koi required section khali; `TODO`, `_TO WRITE`, `[TO CONFIRM]` bacha hai |
| `text_budget`, `text_total` | advisory | section apne word budget ke bahar; kul prose 2500-6000 words ke bahar |

### Honestly
1. **Dev grid par draft ek rehearsal hai.** Isme dev_v2 ke numbers hain, paper ke nahi; banner isliye hai. Asli draft tab: sealed grid, freeze `ok`, 3 LLM, benchmark.
2. **M4 abhi `proxy_rb` hai**, expert plan ke against nahi (jab tak benchmark par run nahi). Draft me yehi likha aata hai aur check advisory hai; benchmark grid par blocking ho jaata hai.
3. **Oracle (RQ6)** upper bound hai, result nahi: draft me alag likha jaata hai.
4. **Benchmark placeholder check**: `bench_real` tab tak blocking hai jab tak benchmark me sample/TODO parts hain. Ye check data ki imaandari ke liye hai, isse bypass karne ka option nahi diya.
5. Word budgets mere andaze hain (6-8 pages ke hisaab se); apne supervisor ke format ke hisaab se `TEXT_FIELDS` me badal sakti ho.
6. Word/LaTeX chahiye to: `pandoc paper\paper_draft.md -o paper\paper_draft.docx` (pandoc install karna padega; tables pipe tables hain).

### Tests (+54 test_paperdraft, ab 62)
`tests/test_paperdraft.py`: component diff aur RQ mapping (priority samet), text file semantics (khali shuruaat, kabhi overwrite nahi, missing key judna, untouched file, unknown keys), draft ke sections ka order aur placeholders, **Table 4 wahi hai jo `make_tables` banata hai**, headline rows ka rule, har RQ me sahi paired comparisons, **hand se nikala hua "+4.00 [+4.00, +4.00] costlier"**, "interpretation" regex (koi claim nahi), readiness ke saare variants (sealed ready path, drift, dusra freeze id, freeze nahi, benchmark placeholders, benchmark grid, cases, extraction, unreadable inputs), CLI (exit codes, `--strict`, `--text`, regenerate). Sealed fixture: asli `FREEZE.json` wala fake repo. 14 jaan-boojh kar bugs daal kar bhi check kiya (galat RQ order, headline = best row, text overwrite, blocking check ignore, dev banner hata dena ...): **ek bach gaya tha (RQ priority order), uska test jodh diya**, ab sab pakde gaye.

### Abhi kya nahi hai
Asli prose (title se conclusion), asli numbers (sealed runs), benchmark annotation, figures ka paper-quality version (Table 4 ke figures `make_tables.py build` se aate hain, draft unhe link karta hai), LaTeX template, bibliography.

## Step 27 (WP12): release ek command me, `scripts/release.py`

Report 8.4 ke deliverables (iii) benchmark datasheet ke saath aur (iv) "results tables aur figures **ek command se** regenerate, aur failure case studies". Ab ek **recipe** (JSON) se poora release ek folder me banta hai, aur `verify` batata hai ki folder badla nahi aur dobara banane par wahi files aati hain. **Koi API call nahi, koi run shuru nahi**: sirf finished runs padhta hai.

```
python scripts\release.py check  configs\release_final.json                  # build ko kya milega (kuch likhta nahi)
python scripts\release.py build  configs\release_final.json                  # -> release\<name>\
python scripts\release.py build  configs\release_final.json --out release\v1 --strict
python scripts\release.py verify release\v1                                  # files badli? aur dobara banao: wahi hai?
python scripts\release.py verify release\v1 --no-regenerate                  # sirf file hashes
```
`--strict`: paper readiness ka koi blocking item khula ho to exit 1 (folder phir bhi likha jaata hai). Exit 2: recipe ya input me problem. `verify`: 0 verified, 1 not verified, 2 release folder nahi hai.

### Recipe (`configs/release_template.json`, copy karke `configs/release_final.json`)
Paths repo ke relative. `name` aur `grid` ke siva sab optional; galat ya anjaana key ek saath saari galtiyon ke saath refuse hota hai.
| key | matlab |
|---|---|
| `name` | folder ka naam aur release tag (`release-<name>`) |
| `grid` | final runs ka ablation config (Step 22) |
| `results` | run folders ka folder (default: config ka `results_dir`) |
| `freeze` | freeze manifest (default `FREEZE.json`) |
| `benchmark` | `data/bench_v1.json` (agar hai) |
| `text` | tumhari prose ki file (default `paper/paper_text.json`) |
| `case_studies` | `{runs: [run folders], commentary: file, per_type, types, instructive}`: runs me har LLM ki **baseline row** ka run folder rakho |
| `extraction` | M9 ke run folders |
| `notes` | RELEASE.md me chhapne ke liye ek paragraph |

### Build kya karta hai (is order me)
freeze check -> Table 4 + paired tests + figures (`tables.build`) -> case studies (`casestudy`) -> benchmark + datasheet -> paper draft (`paperdraft`) -> `RELEASE.md`, `SHA256SUMS.txt`, `release_manifest.json`.
```
release\v1\
  RELEASE.md  FREEZE.json  freeze_verify.txt  SHA256SUMS.txt  release_manifest.json
  inputs\         recipe, grid config, tumhari prose aur commentary ki **copies**
  tables\         table4 (md, csv, tex), paired, consistency, per_part, issues, figures, manifest.json
  case_studies\   case_studies.md, cases.json, selection.csv
  benchmark\      bench file + DATASHEET.md
  paper\          paper_draft.md, readiness.txt
```
- **Benchmark datasheet:** `data/<bench>_datasheet.md` (jo `benchmark.py import` ne likhi) pehle use hoti hai (usme conflicts section hai); na ho to benchmark file se dobara banti hai (bina import report ke, aur bina tareekh ke taaki rebuild byte-identical rahe).
- **`RELEASE.md`** batata hai ki folder kis se bana (grid, dataset sealed/dev, kitne runs usable, freeze id aur status, benchmark, cases), readiness (blocking aur advisory), tag commands (`baseline-v0`, `freeze-v1`, `release-<name>`) aur "what this does not show".
- **Readiness** wahi hai jo `paper_draft.py` ki hai (Step 26); release use dobara nahi banata. Dev grid par RELEASE.md par "DEVELOPMENT RESULTS, NOT PAPER RESULTS" banner aata hai.

### `verify` kya jaanchta hai
1. **Files badli?** Har file ka hash `release_manifest.json` se (text files CRLF-normalised: Windows checkout ka hash same). Badli (`modified`), hata di gayi (`missing`) ya jodi gayi (`not in the manifest`).
2. **Dobara banao.** `inputs/recipe.json` se temp folder me build, phir har file compare. Same runs + same inputs = same bytes (alag folder par bhi, CRLF ke saath bhi). Jo source file release ke baad badli (aur prose, commentary, grid, koi run) wo `inputs/` ke neeche difference ke roop me dikhti hai: yehi iska kaam hai. **Figures** (PNG) ka sirf bytes me fark **warning** hai (image encoder repo ka hissa nahi); folder ke andar PNG ka badalna phir bhi failure hai.
3. Freeze: agar build ke baad koi behaviour file badli to `freeze_verify.txt` aur `RELEASE.md` ka rebuild alag aata hai, to verify fail hota hai.

### Safety
- Build pehle `.release-xxxx` temp folder me hota hai aur end me andar move hota hai: beech me fail hua to **purana release jaisa tha waisa rehta hai**.
- Khali na ho aur release na ho (`release_manifest.json` nahi) to folder refuse (tumhari file ghalti se na mite). Purane release par build sirf apni jagah (`inputs`, `tables`, `case_studies`, `benchmark`, `paper` aur chaar files) badalta hai; baaki files bachi rehti hain.
- **Tumhari hath se likhi files kabhi nahi badalti** (prose, commentary): release copies par kaam karta hai. Is liye `paper_text.json` me missing sections ki khali entries tabhi judti hain jab tum `paper_draft.py` chalao; release me wo copy me judti hain.
- Koi bhi machine ka path (repo folder, temp folder) problem messages se nikaal diya jaata hai, warna doosri machine par verify fail hota.
- `git commit`/`git tag` ye script nahi chalati: sirf batati hai kya karna hai (dirty tree aur kaun se tag abhi nahi hain).

### Jo honestly likhna hai
1. **`verify` saabit karta hai ki folder wahi hai jo recipe aur inputs se banta hai. Ye nahi ki inputs sahi hain**: runs, prose, benchmark annotation, commentary tumhari hain. Aur freeze sirf saabit karta hai ki files wahi hain, ye nahi ki test results freeze se pehle nahi dekhe.
2. **Run folders release me nahi aate** (bade hote hain): `tables/manifest.json` me har run ki records file ka hash hai. Agar `results/` badal gaya ya hat gaya to verify (rebuild) fail hoga, jo sahi hai: `results/` ko commit ya archive karo (Zenodo/zip) release ke saath.
3. Doosri machine par numbers ki bytes tab alag aa sakti hain jab numpy/Python ka random ya float formatting alag ho (bootstrap seeded hai, par library versions manifest me compare nahi hote). Aisa ho to `verify` difference batayega; chhipata nahi.
4. Dev grid par release ek rehearsal hai.

### Tests (+76, `tests/test_release.py`)
Recipe (defaults, idempotent, 27 galat recipes ke reasons, sab problems ek saath, shipped template valid), check, build (listed files hi likhe, sums sha256sum-compatible, Table 4 draft ke barabar, RELEASE.md ki baatein, prose kabhi nahi chhuti, copy LF), **same sources = same bytes alag folder par**, **koi machine path nahi**, CRLF, **verify** (fresh release, file badli/hata/jodi, source badla, run jodi gayi, release nahi hai), output folder (purane release ko badalna, foreign file, refuse, missing input, **fail hone par purana release bachta hai**, temp folder nahi bachta), case studies (commentary ke saath), benchmark (datasheet regenerate / import wali, placeholder blocking), extraction, figures (hard aur soft), sealed path (matching freeze = ready, drift, unreadable, named manifest missing), CLI (exit codes, `--strict`, subprocess). Jaan-boojh kar daale gaye bugs ka hisaab neeche.
**25 jaan-boojh kar bugs** daal kar check kiya (CRLF hash na karna, purane owned folder na hatana, `extra`/`modified` na dekhna, PNG ko hard/soft ulta karna, machine path na hatana, prose source par likhna, folder-refuse hata dena, temp folder na hatana, `ready` hamesha True, freeze copy hata dena, datasheet par tareekh ...). **Do bach gaye the** (verify me "sirf rebuild me hai" aur "sirf release me hai" wali files ka test nahi tha): test joda, ab sab pakde gaye.

### Abhi kya nahi hai
Asli runs aur asli final release, presentation (report 8.4 vi), release ko ek zip ya Zenodo upload me pack karna, `baseline-v0` aur `freeze-v1` tags (tumhare haath se).

## Step 28 (WP12): presentation skeleton, `scripts/slides.py`

Report 8.4 ke deliverables (vi) "presentation". Release folder (ya grid config) se **slides ka Markdown** banta hai: tables aur paired results jagah par, tumhari prose alag file me, speaker notes me sirf "kya bolna hai" ke neutral prompts. **Koi API call nahi, koi run shuru nahi.**

```
python scripts\slides.py release\v1                                  # release folder se: uske tables, cases, benchmark
python scripts\slides.py release\v1 --format marp --out slides_final --strict
python scripts\slides.py configs\ablation_template.json              # dev grid par rehearsal
python scripts\slides.py configs\final.json --benchmark data\bench_v1.json --case-studies results\case_studies\final --freeze FREEZE.json
```
Likhta hai: `slides\slides.md`, `slides\readiness.txt`, aur `slides\slides_text.json` (har slide ki ek khali entry; `--text FILE` se alag file). Exit 0 ok, 1 `--strict` me blocking item, 2 input me problem (STOPPED).

**Convert:** `pandoc slides\slides.md -o slides\slides.pptx` (pandoc installed ho; notes speaker notes ban jaate hain), ya Marp (`--format marp`, VS Code extension ya `marp slides\slides.md --pptx`). **Design/layout tumhara**: script sirf content aur order deta hai. Chaude tables par font chhota karna pad sakta hai.

### Slides (ek `## ` = ek slide)
Problem and gap, Foundation: LLM-CAPP v1, Research questions, Components under test (table), Data and protocol, [Expert benchmark], First results (wahi headline table jo paper draft me), [First results: the ablation path (figure)], RQ1: Baseline failures, RQ1: Failure types of the raw routes, RQ2: Two-step planning and the hint, RQ3: Localized repair, RQ3: How much of the LLM route survives (M3), RQ4: Candidates and selection, RQ5: Repair or constrain, RQ6: Extraction and confirmation (M9), [Cost and quality (figure)], Failure case studies, Limitations, Reproducibility, Conclusion. Bracket wale tabhi aate hain jab benchmark / figure / cost figure ho.
- Har RQ slide par wahi paired comparisons (A_i vs A_j, 95% interval) aur wahi metrics jo us RQ ke liye matter karte hain (RQ2/3: M1, M2p; RQ4: M2p, M4, M7; RQ5: M2p, M7). Jo comparison us RQ ka nahi, wo wahan nahi.
- Table 8 rows (`MAX_ROWS`) se lambi ho to bacha hua paper draft me; slide par line aati hai "+N more rows".
- Dev grid par har results slide par "Development results (...), not paper results." Sealed grid par jab tak blocking item khula ho: "Not final: n blocking item(s)".
- Limitations slide par paper draft ki generated list ke **pehle 5 items ke pehle sentence**.

### Ek zaroori design rule (pandoc)
pandoc pptx me table (ya image) ke baad jo bhi aaye wo **nayi slide** ban jaati hai. Isliye har slide par pehle sentences/notes, **table ya figure sabse last**, aur ek slide par ek hi table. Jo slide ko do tables chahiye uske liye do slides. Test ek asli pandoc conversion chalata hai: 18 headings + title = 19 slides, aur notes ki ginti.

### Tumhari prose: `slides_text.json`
Ek entry har slide ki, word budget ke saath (jaise 15-70 words; title 3-15). Dobara generate karne par tumhari entries **kabhi nahi chhuti**; sirf missing keys khali judti hain, aur jo key kisi slide se match nahi karti wo orphan ke roop me rehti hai (note aata hai). Khali entry par slide par `_TO WRITE (15-70 words): <hint>_` dikhta hai. Script kabhi nahi likhta ki numbers ka **matlab** kya hai: speaker notes me sirf prompts hain ("Read the interval, not the point estimate", ...).

### Readiness
Paper draft ke checks (jo `text_*` nahi hain, kyunki slides ki apni prose alag hai) plus: `slides_missing` (blocking: required slide ki entry khali, benchmark slide tabhi required jab benchmark hai), `slides_placeholders` (blocking: `TODO`, `_TO WRITE`, `[TO CONFIRM]` bache hain), `slides_budget` (advisory: words budget se kam ya zyada).

### Release folder se
`collect_from_release` release ke `tables\` aur `case_studies\` use karta hai (case-study ke asli run folders hat gaye ho tab bhi chalta hai), recipe se benchmark, freeze, extraction, results padhta hai. Release folder na ho, `recipe.json` na ho, grid ya input gum ho to STOPPED (nam ke saath).

### Honestly
1. Ye **talk ka dhaancha** hai, taiyaar talk nahi: kya bolna hai, kis slide par kya zor dena hai, tumhara kaam hai.
2. Dev grid par deck rehearsal hai (banner). Asli numbers ke liye asli runs aur release chahiye.
3. Layout ka koi test nahi (pandoc ka default template); wide tables aankh se dekho.
4. Slides ka matlab nahi ki claims support ho gaye: ek test check karta hai ki script "significant", "outperform", "demonstrate", "prove", "we show", "superior", "confirms", "shows that", "supports", "improve" jaisa kuch likhta hi nahi (RQ ke sawaal ke text ke siva).

### Tests (+51, `tests/test_slides.py`)
Text file semantics (kabhi overwrite nahi, orphans, order), slide ka order aur titles, benchmark slide, YAML/Marp titles (quoting), separators aur notes, **tables last** (grid par aur case studies+benchmark+extraction wale release par), **asli pandoc conversion**, headline = paper ka headline table, RQ routing aur metrics, `MAX_ROWS` aur note ki jagah, status line, case studies, limitations, M9, figure, reproducibility, no-claims, readiness ke saare variants, `write_all`, `collect_from_release` ke errors, CLI (grid, release, strict, errors, subprocess).
**30 jaan-boojh kar bugs** daal kar check kiya (text overwrite, benchmark required na rakhna, note table ke baad, legend table ke baad, case table ke baad extra line, galat RQ routing, galat metrics, dev banner hata dena, blocking banner hata dena, placeholders/budget/missing checks band, marp separators aur notes, YAML quoting, text sync band, galat tables dir, ...). **Teen bach gaye the**: cases slide par table ke baad text (tables-last test me case studies nahi the), limitations ka 5-item cap (fixture me 5 se kam limitations), RQ6 par "+N more run(s)" note ki jagah (ek hi extraction run). Teeno ke test jode, ab sab pakde gaye.

### Abhi kya nahi hai
Asli talk ki prose (tumhari), asli runs aur release, `.pptx` ka design/template, `baseline-v0` aur `freeze-v1` tags (tumhare haath se).
