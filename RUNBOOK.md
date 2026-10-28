# RUNBOOK: setup se Zenodo tak, ek hi order me

Ye file **order** batati hai: kaunsa command kab, kis gate par, aur mujhe kya paste karna hai. Har step ka detail `BASELINE.md` me hai. Sab commands PowerShell me, repo root se (`C:\All-Projects\LLM-CAPP-MTP1-Project`). Report ke gates G1 to G4 (Table 6) ke hisaab se.

`tests\test_runbook.py` har command ke flags aur `--system` ke naam asli scripts se match karta hai, isliye ye file code se alag nahi ho sakti.

**Kisi bhi gate par mujhe ek hi cheez bhejo:**
```
python scripts\diagnostics.py
```
phir `results\diagnostics.txt` attach karo (paste ya screenshot nahi). Upar **PROBLEMS** (kya ruka hai ya kharab hai), phir git / API key (set hai ya nahi, value kabhi nahi) / `FREEZE.json` / datasets ki halat, `results\` ke har run folder ki ek line, G1 ke kaun se baseline runs baaki hain (exact command ke saath), aur saare offline checks ka output (physics, feasibility, objective, feasrepair, compare_runs, part_diff, confidence_tune, risk_compare, cascade_pareto, grid status, freeze verify). Jis check ka input abhi bana nahi, wo wajah ke saath skip hota hai. Koi API call nahi, sirf wahi ek file likhi jaati hai, aur secret jaisi koi line file se hata di jaati hai. Neeche ke "Mujhe paste karo" me jo offline outputs hain wo isme already aa jaate hain; sirf wo cheezein alag se bhejo jo diagnostics me nahi hain (jaise `table4.md`, `case_studies.md`, tumhare 'decide' choices).

## 0. Ek baar (setup)

```
pip install -r requirements.txt
python -m pytest
python scripts\make_dataset_v2.py
python scripts\smoke_llm.py
python scripts\smoke_llm.py
```
- `make_dataset_v2.py` ke baad `git status` clean rehna chahiye (dataset deterministic hai).
- `smoke_llm.py` **do baar**: pehli baar live API, doosri baar `cached=True` aur turant. `finish_reason == 'length'` dikhe to answer kat raha hai: mujhe batao.
- `.env` me `GROQ_API_KEY=...`. `.env`, `results\` aur `cache\` git me nahi jaate (`.gitignore`). **`results\` aur `cache\` hi tumhare saboot hain**: unki alag backup rakho (OneDrive ya zip). Release ke saath `pack --with-runs` unhe archive me daalta hai.
- `baseline-v0` tag: `git log --oneline` me wo commit dhundo jo MTP-1 ke Step 1 se pehle ka aakhri hai, phir `git tag baseline-v0 <hash>`. Sahi commit mujhe nahi pata, ye tum hi pehchanogi.
- Slides/paper ko Word ya pptx banane ke liye pandoc alag install karna padta hai (pip se nahi).

## G1 (Week 1): baselines aur benchmark v0

Development set `dev_v2` par teen baselines. Pehle 20 parts (kam API), phir chaaho to poora.
```
python scripts\run_eval.py --system rb --dataset dev_v2 --tag dev
python scripts\run_eval.py --system b1 --dataset dev_v2 --tag dev --limit 20
python scripts\run_eval.py --system b0 --dataset dev_v2 --tag dev --limit 20 --throttle 1
python scripts\compare_runs.py results\dev_rb results\dev_b1 results\dev_b0
python scripts\part_diff.py results\dev_rb results\dev_b0
python scripts\case_studies.py results\dev_b0
```
Run folder ka naam `results\<tag>_<system>` hota hai. Pehla folder (`rb`) reference hai.

Benchmark v0 (lagbhag 20 parts) supervisor ko bhejne ke liye. Pehle rehearsal placeholder se, phir asli parts se:
```
python scripts\benchmark.py select --from test_v2 --n 20 --out bench\spec_placeholder.xlsx
python scripts\benchmark.py template --out bench\spec.xlsx
python scripts\benchmark.py sheets --spec bench\spec.xlsx --out-dir bench --annotator "Prof. Deb" --second-name "Second annotator" --second-n 18
python scripts\benchmark.py import --spec bench\bench_spec.json --primary bench\annotate_primary.xlsx --second bench\annotate_second.xlsx --dry-run
python scripts\benchmark.py import --spec bench\bench_spec.json --primary bench\annotate_primary.xlsx --second bench\annotate_second.xlsx
```
Plans annotators khud likhte hain, kuch pre-filled nahi hota. `data\bench_v1.json` **sealed** hai: sirf final runs.

**Mujhe paste karo:** `compare_runs` ka output, `case_studies.md` ka pehla hissa, `import --dry-run` ki problems.

## G2 (Week 3): sab Tier-1 components dev set par, phir FREEZE

Ablation grid dev par. `commands` exact lines deta hai (har LLM ke liye, same tag, taaki cache bache):
```
python scripts\make_tables.py commands configs\ablation_template.json
python scripts\make_tables.py status   configs\ablation_template.json
python scripts\make_tables.py build    configs\ablation_template.json
```
`commands` jo lines de unhe order me chalao, phir `status`, phir `build`. Template ki **'decide' rows** (A5 hint ya bina hint, A6 fs2 ya fs4, A7 fz1 ya fz2) dev ke numbers dekhke tay karo, **sealed set dekhne se pehle**.

Madad ke offline checks (koi API call nahi):
```
python scripts\confidence_tune.py results\dev_b0_r1
python scripts\risk_compare.py results\dev_rb results\dev_b0_r1 --physics
python scripts\cascade_pareto.py results\dev_b0_r1 results\dev_b0
python scripts\physics_check.py --dataset dev_v2
python scripts\feasibility_check.py --dataset dev_v2
python scripts\objective_check.py --dataset dev_v2
python scripts\feasrepair_check.py --dataset dev_v2
python scripts\show_plan.py results\dev_b0 DEV_007
```
`confidence_tune` ka threshold `run_eval ... --confidence-flag T` me jaata hai. Cascade ke runs ke liye `cascade_pareto.py` ke docstring me exact lines hain.

**Freeze (G2 par, Tier 1 ke liye).** Pehle final grid ki config banao:
```
copy configs\ablation_template.json configs\final.json
```
`final.json` me badlo: `"name": "ablation_final"`, `"dataset": "test_v2"` (ya `"bench_v1"`), `"limit": null`, `"repeats": 5`, aur **tags naye** (`fin_qwen`, `fin_gpt120`, `fin_gpt20`, `shared_tag` `fin_shared`). Run folder ka naam dataset nahi, sirf `<tag>_<system>` hota hai: purane tags rakhoge to dev ke results folder overwrite honge.
```
python scripts\freeze.py create --grid configs\final.json --run-tests --note "Tier 1"
python scripts\freeze.py verify
```
Phir `git add FREEZE.json; git commit -m "Freeze"; git tag freeze-v1; git push origin --tags`. Tier 2 aur 3 ke components ka code ab bhi bana hua hai, isliye ek hi freeze sab rows ko cover karta hai. **Is ke baad behaviour wala code mat badlo.** Bug fix ke liye badalna hi pade to naya `freeze.py create` karo aur `python scripts\freeze.py diff FREEZE_tier1.json FREEZE.json` se dikhao kya badla (pehle `copy FREEZE.json FREEZE_tier1.json` rakh lo); jin runs ka freeze id alag hoga unhe Table 4 warning ke saath dikhata hai. `run_eval` sealed dataset par freeze ke bina ya drift ke saath chalne se mana kar deta hai (`--no-freeze-check` mat lagao).

**Mujhe paste karo:** `table4.md` (dev), `issues.txt`, `confidence_tune` ka output, `physics_check` / `feasibility_check` / `objective_check` ke outputs, aur tumhare 'decide' choices.

## G3 (Week 5): Tier 1 (rows A0 se A6) sealed set par, kam se kam 3 LLMs

```
python scripts\make_tables.py commands configs\final.json
python scripts\make_tables.py status   configs\final.json
python scripts\make_tables.py build    configs\final.json --strict
python scripts\freeze.py verify
```
`commands` ki lines (`--dataset test_v2` / `bench_v1`, `--repeats 5`) order me chalao, **abhi sirf A0 se A6 ki** (Tier 1: baselines, repair, two-step, selection, few-shot). Har LLM ke liye pehle baseline rows. Phir failure case studies, **har LLM ki baseline row (A1) ke run se**:
```
python scripts\case_studies.py results\fin_qwen_b0 results\fin_gpt120_b0 results\fin_gpt20_b0 --out results\case_studies\final
```
`commentary.json` me **tum** likhogi: model kyun fail hua, repair ne kya sahi rakha. Bina commentary ke cases `--strict` me atakte hain.

Extraction front end (M9), annotated drawings par: ground truth tum bharti ho, development subset par prompts tay karo, baaki par M9:
```
python scripts\extraction_eval.py template drawings --out data\extraction_truth.json
python scripts\extraction_eval.py run drawings --truth data\extraction_truth.json --tag fe1
python scripts\extraction_eval.py score results\fe1_extraction
```

**Mujhe paste karo:** `status` ka output, `table4.md`, `paired.md`, `issues.txt`, `freeze.py verify` ka output.

## G4 (Week 7): baaki configurations ek baar, statistics poori, code frozen

Bachi hui rows ek-ek baar sealed set par: A7 (feasibility rules) aur A8 (cascade) Tier 2, A9 (constrained planner) Tier 3. Phir:
```
python scripts\make_tables.py build configs\final.json --strict
python scripts\freeze.py verify --strict
```
`verify --strict` analysis code ka badlav bhi pakadta hai. Koi run dobara chalana pade to wajah likho: sealed set par bar-bar chalana selection bias hai.

Time kam pade to report ka fixed order: pehle Tier 3 (A9), phir cascade (A8), phir route sheet ke tool/parameter columns, phir feasibility rules 15-20 se ~10. **Tier 1 kabhi nahi ghatta.**

## Week 8: release, paper, slides, pack

```
copy configs\release_template.json configs\release_final.json
python scripts\release.py check  configs\release_final.json
python scripts\release.py build  configs\release_final.json --out release\v1 --strict
python scripts\release.py verify release\v1
python scripts\paper_draft.py configs\final.json --out paper
python scripts\slides.py release\v1
copy configs\zenodo_template.json configs\zenodo.json
python scripts\pack.py check  release\v1 --with-runs
python scripts\pack.py build  release\v1 --with-runs
python scripts\pack.py verify release\v1.zip
```
- `release_final.json` me grid, benchmark, case-study runs, extraction runs, aur apni prose ki file. Prose (`paper\paper_text.json`, `slides\slides_text.json`) **tum likhti ho**; script kabhi usse overwrite nahi karta aur kabhi nahi likhta ki numbers ka matlab kya hai.
- `--strict` tab tak fail karega jab tak readiness ke blocking items khule hain (jaise commentary, prose, benchmark datasheet). Wahi tumhari checklist hai.
- `zenodo.json` me `creators` aur `license` tum bharogi.
- Tags: `git tag release-v1`, `git push origin --tags`. `release\v1` commit karo, **zip nahi** (`.gitignore` me `release/*.zip` jod do); zip Zenodo par jaata hai.
- Zenodo par upload tum khud karti ho; script upload nahi karta.

## Jab bhi dekhna ho

```
python scripts\diagnostics.py
streamlit run week6\app.py
python scripts\route_sheet.py from-run results\dev_b0_r1 --part DEV_004 --out results\DEV_004.pdf
python scripts\show_plan.py results\dev_b0 DEV_007
```

## Mat karo

1. Sealed set (`test_v2`, `bench_v1`) par freeze se pehle koi LLM run nahi.
2. Sealed results dekhke rows ya prompts mat badlo.
3. `--no-freeze-check` mat lagao.
4. Freeze ke baad behaviour wala code mat badlo (`freeze.py verify` batata hai).
5. `.env` ya API key kahin commit/paste mat karo. Key kabhi chat me aa gayi ho to Groq console me rotate karo.
6. Dev release ka pack Zenodo par result ke roop me mat daalo (rehearsal hai).
