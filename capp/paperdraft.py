"""Paper draft generator (MTP-1 WP12, report 8.4: "a paper draft of 6 to 8 pages"): the facts of the paper from the finished artefacts, the prose
from the author, and a check of whether the draft may be written up as results at all.

    python scripts/paper_draft.py configs/final.json --benchmark data/bench_v1.json --case-studies results/case_studies/final \
        --extraction results/fe1_extraction --freeze FREEZE.json --tables-dir results/tables/final --out paper

Inputs (all read-only; nothing is run and no API call is made):
    config        the ablation grid (scripts/make_tables.py): Table 4, the paired comparisons and their consistency across LLMs are
                  recomputed from the runs with capp/tables.py, so the draft and `make_tables.py build` can never disagree
    --benchmark   the expert benchmark (data/bench_v1.json): size, sources, supervisor review, M10 (human-human agreement)
    --case-studies  a folder written by scripts/case_studies.py (cases.json): the M3 failure table and how many cases lack commentary
    --extraction  run folders of scripts/extraction_eval.py (M9, RQ6); repeatable
    --freeze      FREEZE.json (scripts/freeze.py): commit, tag, id, and whether the working tree still matches it
    --tables-dir  where `make_tables.py build` wrote its figures (linked, not copied)

What the script writes into the draft: the data and setup facts, the table of configurations, Table 4, for every research question the
paired comparisons that isolate the component it asks about (which rows those are follows from the system names: a row that adds
`+ts`, `+h1` or `+fs` answers RQ2, repair RQ3, samples or selection RQ4, `+cg1` RQ5), the M3 table, M9, the limitations that this
particular run implies, and how to reproduce it. What it does NOT write: any sentence that says what the numbers mean. That is
`paper_text.json`, one entry per section, written by the author; the script only creates the missing entries (empty) and never changes
or deletes one, so a regenerated draft keeps the prose. An empty section shows what to write and the word budget.

readiness.txt (and Appendix A of the draft) checks what stands between this draft and a paper: results from the sealed set, at least three
LLMs, every run complete, one freeze that still matches, no placeholder parts in the benchmark, expert references for M4, case-study
commentary written, no TODO left in the prose. Each check is `blocking` (the results cannot be used as they are) or `advisory`.
The draft is generated even when it is not ready: a development grid gives a draft to rehearse with, and says so on its first line.
"""
import json
import os

from capp import config, dataset, freeze, tables as T

PLACEHOLDER_MARKS = ("TODO", "_TO WRITE", "[TO CONFIRM]")
UNCLEAR_SHORT = "n.s."
MIN_BENCH_PARTS, MIN_SECOND, MIN_DRAWINGS = 40, 15, 30
WORDS_PAGES = "6 to 8 pages is roughly 4000 to 5500 words with the tables"

RQ = {
    1: "How often and in what ways does the v1 LLM planner fail, in terms of precedence violations, missing or invented operations and "
       "physical infeasibility, when evaluated against expert-validated plans?",
    2: "Does two-step reason-then-formalise planning reduce first-pass failures, and how much of any gain depends on the rule-based hint?",
    3: "Does localized minimal-edit repair preserve more of the LLM's valid decisions than whole-route replacement, without loss of validity?",
    4: "Do multi-candidate generation and consistency-aware selection improve plan quality and stability, and at what cost in calls, tokens "
       "and latency?",
    5: "Is it better to repair an LLM plan after generation, or to constrain generation step by step with the precedence graph? (stretch)",
    6: "Do per-feature evidence and agreement-based selective confirmation reduce the number of user corrections per drawing without lowering "
       "the accuracy of the confirmed feature list?",
}

# key: (title, required, (min words, max words), what to write). The order is the order of the draft.
TEXT_FIELDS = {
    "title": ("Title", True, (3, 20), "A title that names the problem and the approach."),
    "authors": ("Authors", True, (1, 40), "Authors and affiliations, one line."),
    "abstract": ("Abstract", True, (150, 250), "Problem, what was built, the evaluation (data, LLMs, repeats), the main results with their numbers "
                 "from the facts below, the limits."),
    "introduction": ("Introduction", True, (400, 700), "Why reliability of LLM process planning matters, the gaps (report 3.2), the questions below, "
                     "the contributions as a list."),
    "related_work": ("Related work", True, (350, 600), "LLM-based CAPP studies by input, planning method, verification and repair, evaluation data "
                     "and metrics (the table of report 8.4), and where this work differs."),
    "method": ("Method", True, (500, 900), "The planner, the verifier and the repair, selection, the extended verifier, the constrained planner: "
               "what each component does, in the words of the configurations table."),
    "setup": ("Setup remarks", True, (80, 300), "Anything about the data or the protocol that the facts do not say: how the benchmark was "
              "annotated, why these LLMs, how the development set was used."),
    "rq1": ("RQ1 answer", True, (80, 250), "What the baseline failure rates and the failure types show. Say what the numbers support and what they do not."),
    "rq2": ("RQ2 answer", True, (80, 250), "Two-step, hint and few-shot: what the paired comparisons show across the LLMs."),
    "rq3": ("RQ3 answer", True, (80, 250), "Repair against whole-route replacement: validity and how much of the LLM's route survives (M5)."),
    "rq4": ("RQ4 answer", True, (80, 250), "Candidates and selection: quality, stability and the cost in calls and tokens."),
    "rq5": ("RQ5 answer", True, (80, 250), "Repair after generation or constraint during generation; or say that it was not run."),
    "rq6": ("RQ6 answer", True, (80, 250), "Corrections per drawing and the accuracy of the confirmed list; the oracle is an upper bound."),
    "case_studies": ("Case studies", True, (200, 500), "What the failure cases show that the aggregate numbers do not; cite the cases by id."),
    "discussion": ("Discussion", True, (300, 600), "What holds across the LLMs, what does not, what surprised you."),
    "limitations": ("Limitations", True, (150, 350), "In your words, beyond the generated list."),
    "conclusion": ("Conclusion", True, (120, 250), "Answers in one paragraph and the next steps."),
    "acknowledgements": ("Acknowledgements", False, (0, 100), "Supervisor, annotators, compute."),
}


class PaperError(Exception):
    pass


# ---------------------------------------------------------------- the author's text
def load_text(path: str) -> dict:
    """{key: text}. A missing file is an empty dict. Raises PaperError for a file that is not a JSON object of strings."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except ValueError as exc:
        raise PaperError(f"{path} is not valid JSON: {exc}")
    if not isinstance(data, dict) or any(not isinstance(v, str) for v in data.values()):
        raise PaperError(f"{path}: expected a JSON object {{section: text}} with string values")
    return data


def sync_text(path: str, text: dict) -> int:
    """Adds the missing sections (empty) to the file, in the order of the draft; written text is never touched and unknown keys stay.
    Returns the number added."""
    merged = {k: text.get(k, "") for k in TEXT_FIELDS}
    merged.update({k: v for k, v in text.items() if k not in TEXT_FIELDS})
    added = sum(1 for k in TEXT_FIELDS if k not in text)
    if added or not os.path.isfile(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return added


def words(s: str) -> int:
    return len((s or "").split())


def writing_status(text: dict) -> list:
    """One dict per section: key, title, words, budget, state (empty | short | ok | long)."""
    out = []
    for key, (title, required, (lo, hi), _) in TEXT_FIELDS.items():
        n = words(text.get(key, ""))
        state = "empty" if n == 0 else "short" if n < lo else "long" if n > hi else "ok"
        out.append({"key": key, "title": title, "required": required, "words": n, "budget": (lo, hi), "state": state})
    return out


# ---------------------------------------------------------------- which comparison answers which question
def components(system: str, parent: str) -> list:
    """What a row adds to its parent, from the two system names: samples, selection, repair, two-step, hint, few-shot, feasibility,
    cascade, model, constrained. [] if they cannot be compared (rb)."""
    from capp import systems
    a, b = systems.parse_system(system), systems.parse_system(parent)
    if a is None or b is None:
        return []
    pa, pb = systems.planner_spec(system), systems.planner_spec(parent)
    out = []
    for label, differs in (("samples", a[0] != b[0]), ("repair", a[1] != b[1]), ("selection", a[2] != b[2]), ("two-step", pa.two_step != pb.two_step),
                           ("hint", pa.hint != pb.hint), ("few-shot", pa.few_shot != pb.few_shot), ("feasibility", pa.feas != pb.feas),
                           ("cascade", pa.cascade != pb.cascade), ("model", pa.model != pb.model),
                           ("constrained", pa.constrained != pb.constrained)):
        if differs:
            out.append(label)
    return out


def rq_of(labels: list):
    """The research question a change answers: 5 constrained, 2 two-step / hint / few-shot, 3 repair, 4 samples / selection; None for the rest
    (feasibility, cascade, model: objectives 5 and 7, not a research question)."""
    for rq, group in ((5, ("constrained",)), (2, ("two-step", "hint", "few-shot")), (3, ("repair",)), (4, ("samples", "selection"))):
        if any(x in labels for x in group):
            return rq
    return None


def comparisons(cfg: dict) -> list:
    """[{'row', 'parent', 'labels', 'rq'}] for every row that has a parent and whose two systems are built."""
    by_id = {r["id"]: r for r in cfg["rows"]}
    out = []
    for r in cfg["rows"]:
        p = by_id.get(r["parent"]) if r["parent"] else None
        if not p or r["system"] is None or p["system"] is None:
            continue
        labels = components(r["system"], p["system"])
        out.append({"row": r["id"], "parent": p["id"], "labels": labels, "rq": rq_of(labels)})
    return out


# ---------------------------------------------------------------- collecting
def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except (OSError, ValueError) as exc:
        return None, str(exc)


def collect(config_path: str, benchmark: str = None, case_dir: str = None, extraction=(), freeze_path: str = None, tables_dir: str = None,
            results_dir: str = None, root: str = None) -> dict:
    """Everything the draft is made of. Raises T.TableError for a bad grid config; an optional input that cannot be read is kept in
    ctx['problems'] (the readiness check names it) instead of stopping."""
    root = root or str(config.ROOT)
    cfg = T.load_config(config_path)
    res = T.analyse(cfg, results_dir)
    ctx = {"cfg": cfg, "res": res, "config_path": config_path, "root": root, "problems": [], "tables_dir": tables_dir,
           "benchmark": None, "bench_path": benchmark, "cases": None, "case_dir": case_dir, "extraction": [], "freeze": None}
    parts = None
    try:
        parts = dataset.load_dataset(cfg["dataset"])[cfg["offset"]:]
        if cfg["limit"]:
            parts = parts[:cfg["limit"]]
    except (OSError, ValueError, KeyError) as exc:
        ctx["problems"].append(f"dataset {cfg['dataset']!r} could not be loaded: {exc}")
    ctx["parts"] = parts
    if benchmark:
        try:
            from capp import benchmark as B
            bparts = dataset.load_dataset(benchmark, validate=False)
            ctx["benchmark"] = {"parts": bparts, "agreement": B.agreement(bparts), "summary": dataset.summarize(bparts) if bparts else {}}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            ctx["problems"].append(f"benchmark {benchmark} could not be read: {exc}")
    if case_dir:
        data, why = _read_json(os.path.join(case_dir, "cases.json"))
        if data is None or not isinstance(data, dict) or "cases" not in data:
            ctx["problems"].append(f"case studies: {os.path.join(case_dir, 'cases.json')} could not be read ({why or 'not a case-study file'})")
        else:
            ctx["cases"] = data
    if extraction:
        from capp import extraction_eval as EE
        for folder in extraction:
            try:
                s = EE.score(EE.load_records(folder))
                ctx["extraction"].append({"folder": folder, "score": s, "text": EE.format_report(s, os.path.basename(os.path.normpath(folder)))})
            except Exception as exc:                                       # a folder that is not an extraction run
                ctx["problems"].append(f"extraction run {folder} could not be scored: {exc}")
    fpath = freeze_path or os.path.join(root, freeze.MANIFEST_NAME)
    if os.path.isfile(fpath):
        try:
            man = freeze.load(fpath)
            ctx["freeze"] = {"path": fpath, "manifest": man, "verify": freeze.verify(root, fpath)}
        except freeze.FreezeError as exc:
            ctx["problems"].append(f"freeze manifest {fpath} could not be used: {exc}")
    elif freeze_path:
        ctx["problems"].append(f"freeze manifest {freeze_path} not found")
    return ctx


# ---------------------------------------------------------------- small helpers
def demote(md: str, by: int) -> str:
    return "\n".join(("#" * by + ln) if ln.startswith("#") else ln for ln in md.splitlines())


def _built_rows(cfg):
    return [r for r in cfg["rows"] if r["system"] is not None]


def _cell(res, rid, mid):
    run = res["usable"].get((rid, mid))
    return run["cell"] if run else None


def _ct(cell, metric):
    return T.DASH if cell is None else T.cell_text(cell[metric], metric)


def _short(verdict: str) -> str:
    return UNCLEAR_SHORT if verdict == T.UNCLEAR else verdict


def _paired(res, mid, rid, metric):
    return next((p for p in res["paired"] if p["model"] == mid and p["row"] == rid and p["metric"] == metric), None)


def _consistency(res, rid, metric):
    return next((c for c in res["consistency"] if c["row"] == rid and c["metric"] == metric), None)


def _run_ids(res) -> list:
    return sorted({(r.get("run_info") or {}).get("freeze", {}).get("id") for r in res["runs"].values()
                   if isinstance((r.get("run_info") or {}).get("freeze"), dict) and (r["run_info"]["freeze"].get("id"))})


def is_final(cfg) -> bool:
    return dataset.is_sealed(cfg["dataset"])


# ---------------------------------------------------------------- generated blocks
def banner(ctx, checks) -> str:
    cfg = ctx["cfg"]
    blocking = [c for c in checks if c["level"] == "blocking" and not c["ok"]]
    if not is_final(cfg):
        return (f"> **DEVELOPMENT RESULTS, NOT PAPER RESULTS.** This draft was generated from the grid `{cfg['name']}` on the development set "
                f"`{cfg['dataset']}`. Use it to rehearse the paper; the numbers of the paper come from the final run on the sealed benchmark, "
                f"after the freeze. {len(blocking)} blocking item(s) in Appendix A.")
    if blocking:
        return f"> **NOT READY.** {len(blocking)} blocking item(s) in Appendix A: the numbers below are not yet final results."
    return (f"> Generated from the sealed set `{cfg['dataset']}` under freeze `{(ctx['freeze'] or {}).get('manifest', {}).get('id', '?')}`; "
            "no blocking item in Appendix A.")


def configurations_table(cfg) -> str:
    out = ["| ID | Configuration | System | Adds to | Component(s) changed | Research question |", "|---|---|---|---|---|---|"]
    comp = {c["row"]: c for c in comparisons(cfg)}
    for r in cfg["rows"]:
        c = comp.get(r["id"])
        out.append(f"| {r['id']} | {r['label']} | {('`' + r['system'] + '`') if r['system'] else 'not built'} | {r['parent'] or T.DASH} | "
                   f"{', '.join(c['labels']) if c and c['labels'] else T.DASH} | {('RQ%d' % c['rq']) if c and c['rq'] else T.DASH} |")
    return "\n".join(out)


OPTION_MEANING = (
    ("b1", "one LLM call per part"), ("b0", "five samples (v1 default)"), ("_r1", "localized minimal-edit repair (R1)"),
    ("_r2", "R1, then the model is asked again with the violation report (R2)"), ("_r3", "R1, R2 and the Route Builder as last resort (hybrid)"),
    ("_s3", "verifier-guided selection with the confidence score"), ("+ts", "two-step planner: a reasoning call, then a formalisation call"),
    ("+h1", "the Route Builder's route is given to the LLM as a hint"), ("+fs", "retrieval-based few-shot examples from the development set"),
    ("+fz", "feasibility-aware repair with the physical rules"), ("+cas", "model cascade: the small model first, the large one where repair fails"),
    ("+cg1", "constrained planner: the LLM picks the next operation among those whose prerequisites are placed"),
)


def table4_body(res) -> str:
    """Table 4 as `make_tables.py` renders it, without its own title and metric legend (the setup section has the legend)."""
    lines = T.render_markdown(res).splitlines()
    first_model = next(i for i, ln in enumerate(lines) if ln.startswith("## "))
    intro = next((ln for ln in lines[:first_model] if ln.startswith("Dataset ")), "")
    return "\n".join([intro, ""] + [("##" + ln) if ln.startswith("#") else ln for ln in lines[first_model:]])


def option_legend(cfg) -> str:
    used = [(tok, meaning) for tok, meaning in OPTION_MEANING if any(r["system"] and tok in r["system"] for r in cfg["rows"])]
    return "\n".join(f"- `{tok}`: {meaning}." for tok, meaning in used)


def depth(cfg, rid: str) -> int:
    """How many rows lie between a row and the root of its chain of parents: the number of components added so far."""
    by_id, n = {r["id"]: r for r in cfg["rows"]}, 0
    while by_id[rid]["parent"] in by_id and n < 100:
        rid, n = by_id[rid]["parent"], n + 1
    return n


def headline_rows(cfg, res, mid: str) -> list:
    """For one LLM the first built row with a usable run (the baseline) and the DEEPEST one (the most components added; the later row on a
    tie): a rule that does not look at the results, so it cannot pick the best-looking row."""
    have = [(i, r) for i, r in enumerate(cfg["rows"]) if r["system"] is not None and _cell(res, r["id"], mid)]
    if not have:
        return []
    first = have[0][1]
    deepest = max(have, key=lambda ir: (depth(cfg, ir[1]["id"]), ir[0]))[1]
    return [("baseline", first)] if deepest["id"] == first["id"] else [("baseline", first), ("deepest", deepest)]


def headline_table(ctx) -> str:
    cfg, res = ctx["cfg"], ctx["res"]
    out = ["| LLM | Row | Configuration | M1 first-pass valid | M2 LLM-final | M6 gap (pts) | M7 calls/part |", "|---|---|---|---|---|---|---|"]
    for m in cfg["models"]:
        rows = headline_rows(cfg, res, m["id"])
        if not rows:
            out.append(f"| {m['label']} | – | no usable run | – | – | – | – |")
        for tag, r in rows:
            c = _cell(res, r["id"], m["id"])
            out.append(f"| {m['label']} | {r['id']} ({tag}) | {r['label']} | {_ct(c, 'M1')} | {_ct(c, 'M2p')} | {_ct(c, 'M6')} | {_ct(c, 'M7')} |")
    return "\n".join(out)


def data_facts(ctx) -> list:
    cfg, parts = ctx["cfg"], ctx["parts"]
    L = []
    if parts:
        s = dataset.summarize(parts)
        sealed = "sealed final set" if is_final(cfg) else "development set"
        L.append(f"- Grid data: `{cfg['dataset']}` ({sealed}), {len(parts)} parts, {s['archetypes']} part archetypes, feature coverage "
                 f"{s['feature_coverage']}, features per part " + ", ".join(f"{k}: {v}" for k, v in sorted(s["n_features_hist"].items())) + ".")
        L.append("- Materials: " + ", ".join(f"{k} {v}" for k, v in sorted(s["materials"].items())) + ". Tolerances: "
                 + ", ".join(f"{k} {v}" for k, v in sorted(s["tolerances"].items())) + ".")
    elif ctx["problems"]:
        L.append("- Grid data: could not be loaded (see Appendix A).")
    L.append(f"- {cfg['repeats']} repeats per cell by default; bootstrap {cfg['resamples']} resamples, seed {cfg['seed']}.")
    b = ctx["benchmark"]
    if b:
        bp, s, ag = b["parts"], b["summary"], b["agreement"]
        sources = sorted({p.get("source") or "(none)" for p in bp})
        L.append(f"- Expert benchmark `{ctx['bench_path']}`: {len(bp)} parts"
                 + (f", feature coverage {s['feature_coverage']}" if s else "") + f"; sources: {', '.join(sources[:6])}{' ...' if len(sources) > 6 else ''}.")
        if ag["n"]:
            L.append(f"- Human-human agreement (M10) on {ag['n']} parts: exact match {ag['exact_rate']:.2f}, LCS ratio {ag['lcs_ratio_mean']:.2f}, "
                     f"Kendall tau distance {ag['kendall_tau_mean']:.2f} (95% bootstrap intervals in the benchmark datasheet).")
        else:
            L.append("- Human-human agreement (M10): no second-annotator plans in the benchmark.")
    return L


def llm_table(cfg) -> str:
    out = ["| ID | LLM | Model id | List price in / out (USD per million tokens) |", "|---|---|---|---|"]
    for m in cfg["models"]:
        pr = config.price(m["model"])
        out.append(f"| {m['id']} | {m['label']} | `{m['model']}` | " + (f"{pr[0]:g} / {pr[1]:g}" if pr else "n/a") + " |")
    return "\n".join(out)


def pair_table(ctx, pairs, metrics=("M1", "M2p", "M4", "M6", "M7")) -> str:
    cfg, res = ctx["cfg"], ctx["res"]
    heads = [m["label"] for m in cfg["models"]]
    out = ["| Comparison | Metric | " + " | ".join(heads) + " | Across LLMs |", "|---|---|" + "---|" * len(heads) + "---|"]
    for pr in pairs:
        for metric in metrics:
            cells, any_data = [], False
            for m in cfg["models"]:
                p = _paired(res, m["id"], pr["row"], metric)
                if p and p["mean"] is not None:
                    any_data = True
                    cells.append(f"{T._ci_text(p, T.DECIMALS[metric])} {_short(p['verdict'])}")
                else:
                    cells.append(T.DASH)
            if not any_data:
                continue
            cons = _consistency(res, pr["row"], metric)
            out.append(f"| {pr['row']} vs {pr['parent']} ({', '.join(pr['labels'])}) | {T.PAIRED_LABELS[metric]} | " + " | ".join(cells)
                       + f" | {cons['label'] if cons else T.DASH} |")
    if len(out) == 2:
        return "No paired comparison has data yet (a run of the pair is missing or unusable)."
    return "\n".join(out) + (f"\n\nCells: difference row minus parent (per part, mean over repeats) with a 95% bootstrap interval over parts; "
                              f"{UNCLEAR_SHORT} = the interval includes 0. Not corrected for the number of comparisons.")


def m5_table(ctx, pairs) -> str:
    cfg, res = ctx["cfg"], ctx["res"]
    rows = []
    for pr in pairs:
        for rid in (pr["parent"], pr["row"]):
            if rid not in [x[0] for x in rows]:
                rows.append((rid, next(r["label"] for r in cfg["rows"] if r["id"] == rid)))
    out = ["| Row | " + " | ".join(m["label"] for m in cfg["models"]) + " |", "|---|" + "---|" * len(cfg["models"])]
    for rid, label in rows:
        out.append(f"| {rid}: {label} | " + " | ".join(_ct(_cell(res, rid, m["id"]), "M5") for m in cfg["models"]) + " |")
    return "M5 (share of the LLM's operations that survive the fix; higher = more minimal; – where nothing had to be fixed):\n\n" + "\n".join(out)


def baseline_table(ctx) -> str:
    cfg, res = ctx["cfg"], ctx["res"]
    comp = {c["row"]: c for c in comparisons(cfg)}
    roots = {r["id"] for r in cfg["rows"] if not r["parent"]}
    # the baselines: the rows that add nothing to another row, and the row that only changes the number of samples of such a row (B1 -> B0)
    base = [r for r in _built_rows(cfg) if r["id"] in roots or (r["parent"] in roots and comp.get(r["id"]) and comp[r["id"]]["labels"] == ["samples"])]
    out = ["| Row | Configuration | LLM | M1 first-pass valid | M2 LLM-final | M6 gap (pts) |", "|---|---|---|---|---|---|"]
    for r in base:
        for m in cfg["models"]:
            c = _cell(res, r["id"], m["id"])
            out.append(f"| {r['id']} | {r['label']} | {m['label']} | {_ct(c, 'M1')} | {_ct(c, 'M2p')} | {_ct(c, 'M6')} |")
    return "\n".join(out)


def m3_tables(ctx) -> str:
    cases = ctx["cases"]
    if not cases:
        return "No case-study folder was given (`--case-studies`), so the failure types (M3) are not tabulated here."
    out = []
    for run in cases.get("runs", []):
        out += [f"Run `{run.get('label')}` (system `{run.get('system')}`, dataset {run.get('dataset')}, {run.get('n_parts')} parts, "
                f"{run.get('n_parsed_routes')} parsed raw routes):", "", "| Type | Raw routes with it | Share | Parts |", "|---|---|---|---|"]
        for row in run.get("table", []):
            if row["type"] == "differs":
                out.append(f"| {row['type']} | – | – | {row['parts']} |")
            else:
                share = "n/a" if row["share"] is None else f"{100 * row['share']:.1f}%"
                out.append(f"| {row['type']} | {row['routes']} | {share} | {row['parts']} |")
        out.append("")
    out.append(f"{len(cases.get('cases', []))} case(s) selected by the rule of `scripts/case_studies.py`; "
               f"{cases.get('n_without_commentary', '?')} still without commentary.")
    return "\n".join(out)


def rq_block(ctx, n: int) -> str:
    cfg, res = ctx["cfg"], ctx["res"]
    pairs = [c for c in comparisons(cfg) if c["rq"] == n]
    L = []
    if n == 1:
        L += ["Baseline rows (no component added):", "", baseline_table(ctx), "", "Failure types of the raw LLM routes (M3):", "", m3_tables(ctx)]
    elif n == 6:
        if ctx["extraction"]:
            for e in ctx["extraction"]:
                L += ["```", e["text"], "```", ""]
        else:
            L += ["No extraction run was given (`--extraction`): M9 is not available in this draft."]
    else:
        if not pairs:
            L += ["No row of this grid isolates this component, so no paired comparison answers this question here."]
        else:
            L += ["Comparisons that isolate this component:", "", pair_table(ctx, pairs)]
            if n == 3:
                L += ["", m5_table(ctx, pairs)]
            if n == 5:
                L += ["", "The constrained planner cannot add an operation the Route Builder's set lacks, and asks the LLM only where the "
                          "precedence graph leaves a choice (calls per part in M7)."]
    return "\n".join(L)


def other_components(ctx) -> str:
    pairs = [c for c in comparisons(ctx["cfg"]) if c["rq"] is None and c["labels"]]
    if not pairs:
        return ""
    return "\n".join(["### Further components (objectives 5 and 7)", "", "Feasibility rules and the model cascade are not research questions of "
                      "their own; their effect on the same metrics:", "", pair_table(ctx, pairs)])


def limitations(ctx) -> list:
    cfg, res = ctx["cfg"], ctx["res"]
    L = []
    if not is_final(cfg):
        L.append(f"The grid is on the development set `{cfg['dataset']}` (synthetic parts): these are not test results.")
    L.append(T._m4_note(res))
    b = ctx["benchmark"]
    if b:
        L.append(f"The benchmark has {len(b['parts'])} parts; every per-class statement on it is anecdotal. The reference plans reflect one expert's "
                 "practice; the alternatives recorded per part and the supervisor review reduce this, they do not remove it.")
        if b["agreement"]["n"]:
            L.append(f"The human-human agreement rests on {b['agreement']['n']} parts: indicative, not precise.")
    if len(cfg["models"]) < 3:
        L.append(f"Only {len(cfg['models'])} LLM(s): the report requires at least three so that no conclusion rests on one model.")
    bad = sorted({f"{rid}/{mid}" for (rid, mid), r in res["runs"].items() if r["status"] in ("partial", "missing", "error")})
    if bad:
        L.append("Cells without a complete, checked run: " + ", ".join(bad) + ".")
    L.append("Intervals and p-values are not corrected for the number of comparisons. With several repeats the McNemar pairs are clustered by part, so "
             "those p-values are approximate; the bootstrap over parts is the primary evidence.")
    L.append("The models are hosted and can be retired or change behaviour: the identifiers are pinned in `capp/config.py` and every response is "
             "cached, but a re-run against the live endpoint may differ.")
    if any(c["rq"] == 5 for c in comparisons(cfg)):
        L.append("The constrained planner depends on the completeness of the rule base: it cannot invent an operation the Route Builder's set lacks.")
    if ctx["extraction"]:
        L.append("The agreement-based variant of M9 is scored after an oracle user confirmed the highlighted features: an upper bound for what a real "
                 "user does (the corrections log has the real number).")
    return L


def reproducibility(ctx) -> list:
    cfg, fr = ctx["cfg"], ctx["freeze"]
    L = []
    if fr:
        man, ver = fr["manifest"], fr["verify"]
        g = man.get("git") or {}
        L.append(f"- Freeze `{man['id']}` created {man.get('created_utc')}"
                 + (f", commit `{(g.get('commit') or '')[:10]}`" if g.get("commit") else "") + (f", tag `{', '.join(g['tags'])}`" if g.get("tags") else "")
                 + f"; the working tree now: **{ver['status']}**.")
        if man.get("tests"):
            L.append(f"- Tests at freeze time: {man['tests']['passed']} passed, {man['tests']['failed']} failed.")
        L.append("- Models: " + ", ".join(f"{r} `{m}`" for r, m in sorted(man["config"].get("MODELS", {}).items())) + ".")
    else:
        L.append("- No freeze manifest was found: the configuration of the runs is not pinned (`python scripts/freeze.py create --grid <config>`).")
    ids = _run_ids(ctx["res"])
    L.append(f"- Runs were made under freeze id(s): {', '.join(ids) if ids else 'none recorded'}.")
    return L


def commands(ctx) -> str:
    cfg = ctx["cfg"]
    tdir = ctx["tables_dir"] or f"results/tables/{cfg['name']}"
    L = ["```", "python scripts/freeze.py verify", f"python scripts/make_tables.py build {ctx['config_path']} --out {tdir}"]
    if ctx["case_dir"]:
        L.append(f"python scripts/case_studies.py <run folders> --out {ctx['case_dir']}")
    for e in ctx["extraction"]:
        L.append(f"python scripts/extraction_eval.py score {e['folder']}")
    L += ["python scripts/paper_draft.py " + ctx["config_path"] + (f" --benchmark {ctx['bench_path']}" if ctx["bench_path"] else "")
          + (f" --case-studies {ctx['case_dir']}" if ctx["case_dir"] else "") + "".join(f" --extraction {e['folder']}" for e in ctx["extraction"])
          + " --out paper", "```"]
    return "\n".join(L)


# ---------------------------------------------------------------- readiness
def check(cid, level, ok, message):
    return {"id": cid, "level": level, "ok": bool(ok), "message": message}


def readiness(ctx, text: dict) -> list:
    cfg, res = ctx["cfg"], ctx["res"]
    C = []
    C.append(check("sealed", "blocking", is_final(cfg), f"the grid runs on the sealed set ({cfg['dataset']})" if is_final(cfg) else
                   f"the grid runs on {cfg['dataset']}, a development set: these are development results, not paper results"))
    C.append(check("llms", "blocking", len(cfg["models"]) >= 3, f"{len(cfg['models'])} LLM(s) in the grid (at least 3 are required)"))
    built_ids = {r["id"] for r in _built_rows(cfg)}
    not_ok = sorted(f"{rid}/{mid}={r['status']}" for (rid, mid), r in res["runs"].items() if rid in built_ids and r["status"] != "ok")
    C.append(check("runs", "blocking", not not_ok, "every run of every built row is complete and checked" if not not_ok else
                   f"{len(not_ok)} run(s) missing, partial or excluded: " + ", ".join(not_ok[:8]) + (" ..." if len(not_ok) > 8 else "")))
    unbuilt = [r["id"] for r in cfg["rows"] if r["system"] is None]
    if unbuilt:
        C.append(check("unbuilt", "advisory", False, "rows not built (left out of Table 4): " + ", ".join(unbuilt)))
    errs = [i for i in res["issues"] if i["level"] == "error"]
    C.append(check("errors", "blocking", not errs, "no error in the table issues" if not errs else f"{len(errs)} error(s) in the table issues"))
    if is_final(cfg) or ctx["freeze"]:
        fr = ctx["freeze"]
        if fr is None:
            C.append(check("freeze", "blocking", False, "no freeze manifest (FREEZE.json): the configuration is not pinned"))
        else:
            st = fr["verify"]["status"]
            C.append(check("freeze", "blocking", st in freeze.OK_STATUSES,
                           f"the working tree matches freeze {fr['manifest']['id']} ({st})" if st in freeze.OK_STATUSES else
                           f"the working tree has drifted from freeze {fr['manifest']['id']}: results are not those of the frozen configuration"))
            if is_final(cfg):
                ids = _run_ids(res)
                ok_ids = ids == [fr["manifest"]["id"]]
                C.append(check("freeze_runs", "blocking", ok_ids, "all runs were made under the freeze" if ok_ids else
                               f"runs were made under freeze id(s) {', '.join(ids) or 'none'}, the manifest is {fr['manifest']['id']}"))
    for p in ctx["problems"]:
        C.append(check("input", "blocking", False, p))
    b = ctx["benchmark"]
    if b:
        bp = b["parts"]
        ph = [p["part_id"] for p in bp if "placeholder" in (p.get("source") or "").lower()]
        C.append(check("bench_real", "blocking", not ph, "no placeholder part in the benchmark" if not ph else
                       f"{len(ph)} synthetic PLACEHOLDER part(s) in the benchmark ({', '.join(ph[:4])}{' ...' if len(ph) > 4 else ''}): not the expert benchmark"))
        C.append(check("bench_size", "advisory", len(bp) >= MIN_BENCH_PARTS, f"benchmark has {len(bp)} parts (the plan is {MIN_BENCH_PARTS} to 60)"))
        statuses = [((p.get("reference_meta") or {}).get("review_status") or "not reviewed") for p in bp]
        unrev = sum(1 for s in statuses if s == "not reviewed")
        C.append(check("bench_review", "advisory", unrev == 0, "every benchmark plan has a supervisor review status" if not unrev else
                       f"{unrev} benchmark plan(s) without a supervisor review"))
        C.append(check("m10", "advisory", b["agreement"]["n"] >= MIN_SECOND, f"human-human agreement on {b['agreement']['n']} parts (the plan is {MIN_SECOND} to 20)"))
    elif cfg["dataset"].startswith("bench"):
        C.append(check("bench_given", "blocking", False, "the grid is on the benchmark but no --benchmark file was given (size, sources and M10 are missing)"))
    sources = {r["cell"]["m4_source"] for r in res["usable"].values() if r.get("cell")}
    C.append(check("m4", "blocking" if cfg["dataset"].startswith("bench") else "advisory", sources == {"expert"},
                   "M4 is measured against expert routes" if sources == {"expert"} else
                   "M4 is measured against the Route Builder variants (proxy_rb), not against an expert: say so, or run the benchmark"))
    cs = ctx["cases"]
    if cs is None:
        C.append(check("cases", "advisory", False, "no case studies given (--case-studies): report 7.5 asks for 3 to 5 cases per main failure type"))
    else:
        n_no = cs.get("n_without_commentary", 0)
        C.append(check("cases", "advisory", n_no == 0, "every selected case has commentary" if not n_no else f"{n_no} case(s) without commentary"))
    n_dr = sum(e["score"]["n_scored"] for e in ctx["extraction"][:1])
    C.append(check("m9", "advisory", bool(ctx["extraction"]) and n_dr >= MIN_DRAWINGS,
                   f"M9 scored on {n_dr} drawings" if ctx["extraction"] else "no extraction run given (--extraction): RQ6 has no evidence"))
    st = writing_status(text)
    missing = [s["key"] for s in st if s["required"] and s["state"] == "empty"]
    C.append(check("text_missing", "blocking", not missing, "every required section has text" if not missing else
                   f"{len(missing)} required section(s) without text: " + ", ".join(missing)))
    left = sorted(k for k, v in text.items() if k in TEXT_FIELDS and any(m in v for m in PLACEHOLDER_MARKS))
    C.append(check("text_placeholders", "blocking", not left, "no TODO or placeholder left in the prose" if not left else
                   "placeholder left in: " + ", ".join(left)))
    off = [f"{s['key']} ({s['words']} words, {s['budget'][0]}-{s['budget'][1]})" for s in st if s["state"] in ("short", "long")]
    C.append(check("text_budget", "advisory", not off, "every written section is within its word budget" if not off else
                   "outside the word budget: " + "; ".join(off)))
    total = sum(s["words"] for s in st)
    C.append(check("text_total", "advisory", 2500 <= total <= 6000 or total == 0, f"{total} words of prose ({WORDS_PAGES})"))
    return C


def render_readiness(checks, text) -> str:
    blocking = [c for c in checks if c["level"] == "blocking" and not c["ok"]]
    adv = [c for c in checks if c["level"] == "advisory" and not c["ok"]]
    L = [("NOT READY" if blocking else "READY") + f": {len(blocking)} blocking, {len(adv)} advisory item(s) open", ""]
    for c in checks:
        L.append(f"[{'ok' if c['ok'] else ('BLOCKING' if c['level'] == 'blocking' else 'advisory')}] {c['message']}")
    L += ["", "Prose:"]
    for s in writing_status(text):
        L.append(f"  {s['title']:18s} {s['words']:5d} words  ({s['budget'][0]}-{s['budget'][1]})  {s['state']}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- the draft
def section(text: dict, key: str) -> str:
    title, required, (lo, hi), hint = TEXT_FIELDS[key]
    body = (text.get(key) or "").strip()
    return body if body else f"_TO WRITE ({lo}-{hi} words): {hint}_"


def render(ctx, text: dict, out_dir: str = ".") -> str:
    cfg, res = ctx["cfg"], ctx["res"]
    checks = readiness(ctx, text)
    L = []
    title = (text.get("title") or "").strip()
    L += [f"# {title or '_TO WRITE: title_'}", "", (text.get("authors") or "").strip() or "_TO WRITE: authors_", "", banner(ctx, checks), ""]
    L += ["## Abstract", "", section(text, "abstract"), "", "**Facts for the abstract** (generated; for each LLM the baseline row and the deepest row, the one with the most components added: "
          "a rule that does not look at the results, not a selection of the best):", "", headline_table(ctx), ""]
    L += ["## 1 Introduction", "", section(text, "introduction"), "", "Research questions (MTP-1 report, section 3.3):", ""]
    L += [f"- **RQ{n}**: {q}" for n, q in RQ.items()] + [""]
    L += ["## 2 Related work", "", section(text, "related_work"), ""]
    L += ["## 3 System and method", "", section(text, "method"), "", "### Configurations evaluated", "", configurations_table(cfg), ""]
    legend = option_legend(cfg)
    if legend:
        L += ["System names:", "", legend, ""]
    L += ["## 4 Experimental setup", "", "### Data", ""] + data_facts(ctx) + ["", "### LLMs", "", llm_table(cfg), "", "### Metrics and statistics", ""]
    L += [f"- {line}" for line in T.METRIC_LEGEND] + [f"- {T._m4_note(res)}", ""]
    L += ["Paired comparisons (each row against the row it adds a component to): per part the mean over repeats, 95% percentile bootstrap over parts "
          f"({cfg['resamples']} resamples, seed {cfg['seed']}); validity as a paired binary outcome with McNemar's exact test; the same comparison "
          "for every LLM, because the claim that a component helps needs the same answer for all of them.", "",
          "### Reproducibility", ""] + reproducibility(ctx) + ["", section(text, "setup"), ""]
    L += ["## 5 Results", "", "### Table 4: ablation grid", "", table4_body(res), ""]
    tdir = ctx["tables_dir"]
    if tdir:
        for name, caption in (("fig_ablation.png", "Ablation path per configuration"), ("fig_cost.png", "Cost against plan quality")):
            if os.path.isfile(os.path.join(tdir, name)):
                L += [f"![{caption}]({os.path.relpath(os.path.join(tdir, name), out_dir).replace(chr(92), '/')})", ""]
    for n, q in RQ.items():
        L += [f"### RQ{n}: {q}", "", rq_block(ctx, n), "", section(text, f"rq{n}"), ""]
    other = other_components(ctx)
    if other:
        L += [other, ""]
    L += ["### Failure case studies", "", m3_tables(ctx), "", section(text, "case_studies"), ""]
    L += ["## 6 Discussion", "", section(text, "discussion"), "", "## 7 Limitations", "", section(text, "limitations"), "",
          "Limitations that follow from this run (generated):", ""] + [f"- {x}" for x in limitations(ctx)] + [""]
    L += ["## 8 Conclusion", "", section(text, "conclusion"), ""]
    ack = (text.get("acknowledgements") or "").strip()
    if ack:
        L += ["## Acknowledgements", "", ack, ""]
    L += ["## Appendix A: readiness", "", "```", render_readiness(checks, text).rstrip(), "```", "", "## Appendix B: how to reproduce", "", commands(ctx), ""]
    issues = T.render_issues(res).rstrip()
    L += ["## Appendix C: issues of the table build", "", "```", issues, "```", ""]
    return "\n".join(L)


def write_all(ctx, text_path: str, out_dir: str) -> dict:
    """Writes paper_draft.md and readiness.txt into out_dir and adds the missing sections to the text file. Returns the paths and the checks."""
    os.makedirs(out_dir, exist_ok=True)
    text = load_text(text_path)
    added = sync_text(text_path, text)
    text = load_text(text_path)
    checks = readiness(ctx, text)
    paths = {"draft": os.path.join(out_dir, "paper_draft.md"), "readiness": os.path.join(out_dir, "readiness.txt"), "text": text_path}
    with open(paths["draft"], "w", encoding="utf-8", newline="\n") as f:
        f.write(render(ctx, text, out_dir).rstrip() + "\n")
    with open(paths["readiness"], "w", encoding="utf-8", newline="\n") as f:
        f.write(render_readiness(checks, text))
    orphans = sorted(k for k, v in text.items() if k not in TEXT_FIELDS and v.strip())
    return {"paths": paths, "checks": checks, "added_sections": added, "orphans": orphans,
            "ready": not any(c["level"] == "blocking" and not c["ok"] for c in checks)}
