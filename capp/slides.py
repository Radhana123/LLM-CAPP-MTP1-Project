"""Presentation skeleton (MTP-1 WP12, report 8.4 item vi): the slides of the final talk, with the facts of the results in place, from the
same finished artefacts as the paper draft (capp/paperdraft.py): the grid, the paired comparisons, the case studies, the benchmark, the freeze.

    python scripts/slides.py release/v1                      # from a release folder made by scripts/release.py
    python scripts/slides.py configs/final.json --benchmark data/bench_v1.json ...     # or from a grid config, as paper_draft.py

It writes slides.md (pandoc or Marp Markdown: one `## ` heading is one slide), readiness.txt and slides_text.json. The script writes the
tables and the numbers and, in the speaker notes, what to say about them; it never writes a sentence about what the numbers MEAN. The
sentences on the slides are yours, in slides_text.json (one short entry per slide, regenerating puts them in place and never touches them).
Layout, colours and the final look are yours too: this is the content and the order, ready to convert (pandoc -> .pptx, or Marp in VS Code).
Nothing is run and no API call is made.
"""
import json
import os

from capp import paperdraft as P, release as R, tables as T

FORMATS = ("pandoc", "marp")
MAX_ROWS = 8
PLACEHOLDER_MARKS = P.PLACEHOLDER_MARKS
STATUS_FILE = "slides_text.json"

# key: (title, required, (min words, max words), what to write). A slide holds short bullets, not paragraphs: the budgets are small on purpose.
SLIDE_TEXT = {
    "title": ("Title", True, (3, 15), "The title of the talk."),
    "authors": ("Authors and date", True, (1, 25), "Your name, supervisor, institute, date."),
    "problem": ("Problem and gap", True, (15, 70), "Why an LLM process plan cannot be trusted as it is, and what is missing in earlier work: three or four short bullets (report 3.2)."),
    "foundation": ("Foundation: LLM-CAPP v1", True, (15, 70), "The Route Builder, the LLM planner, the verifier and the scoring of v1, and what MTP-1 adds."),
    "method": ("Components under test", True, (15, 70), "One short line per component, in plain words."),
    "protocol": ("Data and protocol", True, (10, 60), "Why these LLMs, how many repeats, how the benchmark was annotated and the development set used."),
    "benchmark": ("Expert benchmark", True, (10, 60), "Who annotated, how the plans were reviewed, what the agreement figure says."),
    "headline": ("First results", True, (15, 70), "What the audience should take from the first results slide; say what the numbers support and what they do not."),
    "rq1": ("RQ1", True, (15, 70), "What the baseline failure rates and failure types show."),
    "rq2": ("RQ2", True, (15, 70), "Two-step, hint, few-shot: what the paired comparisons show, and whether the LLMs agree."),
    "rq3": ("RQ3", True, (15, 70), "Repair against whole-route replacement: validity and how much of the LLM route survives."),
    "rq4": ("RQ4", True, (15, 70), "Candidates and selection: quality, stability and what they cost."),
    "rq5": ("RQ5", True, (15, 70), "Repair after generation or constraint during generation; or say it was not run."),
    "rq6": ("RQ6", True, (15, 70), "Corrections per drawing; the oracle figure is an upper bound."),
    "cost": ("Cost and quality", False, (0, 60), "What the cost figure shows (optional)."),
    "cases": ("Failure case studies", True, (15, 70), "One case that shows what the aggregate numbers do not."),
    "limitations": ("Limitations", True, (15, 70), "The limits you want the audience to hear from you."),
    "conclusion": ("Conclusion", True, (15, 70), "The answers in a few lines, and the next steps."),
}

RQ_METRICS = {2: ("M1", "M2p"), 3: ("M1", "M2p"), 4: ("M2p", "M4", "M7"), 5: ("M2p", "M7")}


# ---------------------------------------------------------------- the author's text
def required(key: str, ctx) -> bool:
    if key == "benchmark":
        return bool(ctx.get("benchmark"))
    return SLIDE_TEXT[key][1]


def sync_text(path: str, text: dict) -> int:
    """Adds the missing entries (empty) to the file, in the order of the talk; written text is never touched and unknown keys stay."""
    merged = {k: text.get(k, "") for k in SLIDE_TEXT}
    merged.update({k: v for k, v in text.items() if k not in SLIDE_TEXT})
    added = sum(1 for k in SLIDE_TEXT if k not in text)
    if added or not os.path.isfile(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return added


def writing_status(text: dict, ctx) -> list:
    out = []
    for key, (title, _, (lo, hi), _h) in SLIDE_TEXT.items():
        if key == "benchmark" and not ctx.get("benchmark"):
            continue
        n = P.words(text.get(key, ""))
        out.append({"key": key, "title": title, "required": required(key, ctx), "words": n, "budget": (lo, hi),
                    "state": "empty" if n == 0 else "short" if n < lo else "long" if n > hi else "ok"})
    return out


def body_text(text: dict, key: str) -> str:
    title, _, (lo, hi), hint = SLIDE_TEXT[key]
    body = (text.get(key) or "").strip()
    return body if body else f"_TO WRITE ({lo}-{hi} words): {hint}_"


# ---------------------------------------------------------------- facts, compact
def _cap_table(md: str, n: int = None) -> str:
    """A pipe table cut to n data rows. The note about the rows left out goes BEFORE the table: pandoc starts a new slide for whatever
    follows a table."""
    n = MAX_ROWS if n is None else n
    lines = md.splitlines()
    head, rows = lines[:2], lines[2:]
    if len(rows) <= n:
        return md
    return f"_(+{len(rows) - n} more rows in the paper draft)_\n\n" + "\n".join(head + rows[:n])


def pair_slide_table(ctx, pairs, metrics) -> str:
    s = P.pair_table(ctx, pairs, metrics)
    if s.startswith("No paired"):
        return s
    table = s.split("\n\nCells:", 1)[0]
    return f"Cells: difference row minus parent, 95% interval over parts; {P.UNCLEAR_SHORT} = the interval includes 0.\n\n" + _cap_table(table)


def m3_compact(ctx) -> str:
    """The failure types of the first case-study run: the sentences first, the table last."""
    cases = ctx.get("cases")
    if not cases or not cases.get("runs"):
        return "_No case studies given._"
    run = cases["runs"][0]
    more = len(cases["runs"]) - 1
    out = [f"Run `{run.get('label')}` ({run.get('n_parsed_routes')} raw LLM routes); {len(cases.get('cases', []))} case(s) selected by a fixed rule, "
           f"{cases.get('n_without_commentary', '?')} still without commentary."]
    if more > 0:
        out.append(f"_(+{more} more run(s) in the paper draft)_")
    out += ["", "| Type | Raw routes | Share | Parts |", "|---|---|---|---|"]
    for row in run.get("table", []):
        if row["type"] == "differs":
            out.append(f"| {row['type']} | – | – | {row['parts']} |")
        else:
            share = "n/a" if row["share"] is None else f"{100 * row['share']:.1f}%"
            out.append(f"| {row['type']} | {row['routes']} | {share} | {row['parts']} |")
    return "\n".join(out)


def method_table(cfg) -> str:
    comp = {c["row"]: c for c in P.comparisons(cfg)}
    out = ["| Row | Configuration | Adds to its parent | Answers |", "|---|---|---|---|"]
    for r in cfg["rows"]:
        if r["system"] is None:
            out.append(f"| {r['id']} | {r['label']} | not built | – |")
            continue
        c = comp.get(r["id"])
        adds = ", ".join(c["labels"]) if c and c["labels"] else "–"
        answers = "–" if not c or not c["labels"] else (f"RQ{c['rq']}" if c["rq"] else "further")
        out.append(f"| {r['id']} | {r['label']} | {adds} | {answers} |")
    return _cap_table("\n".join(out), 12)


def first_sentence(text: str, limit: int = 150) -> str:
    s = text.strip()
    cut = s.find(". ")
    s = s[:cut + 1] if cut != -1 else s
    return s if len(s) <= limit else s[:limit - 1].rstrip() + "…"


def status_line(ctx, checks) -> str:
    if not P.is_final(ctx["cfg"]):
        return f"_Development results (`{ctx['cfg']['dataset']}`), not paper results._"
    blocking = [c for c in checks if c["level"] == "blocking" and not c["ok"]]
    return f"_Not final: {len(blocking)} blocking item(s), see the readiness check._" if blocking else ""


def _figure(ctx, name, caption, out_dir):
    tdir = ctx.get("tables_dir")
    if tdir and os.path.isfile(os.path.join(tdir, name)):
        return f"![{caption}]({os.path.relpath(os.path.join(tdir, name), out_dir).replace(chr(92), '/')})"
    return ""


# ---------------------------------------------------------------- the slides
RQ_TITLES = {1: "Baseline failures", 2: "Two-step planning and the hint", 3: "Localized repair", 4: "Candidates and selection",
             5: "Repair or constrain", 6: "Extraction and confirmation"}


def slide(key, title, body, notes):
    """One slide: {'key', 'title', 'body' (lines of Markdown), 'notes' (speaker notes)}."""
    return {"key": key, "title": title, "body": list(body), "notes": list(notes)}


def build_slides(ctx, text: dict, checks, out_dir: str = ".") -> list:
    """The slides in order. pandoc starts a new slide for anything that follows a table or an image, so on every slide the sentences come
    first and at most one table (or figure) comes last; a slide that needs two gets a second slide."""
    cfg = ctx["cfg"]
    st = status_line(ctx, checks)
    note = [st, ""] if st else []
    S = []
    S.append(slide("problem", "Problem and gap", [body_text(text, "problem")],
                   ["State the problem in one sentence before the bullets.", "Name the gap that no earlier study closes (report 3.2)."]))
    S.append(slide("foundation", "Foundation: LLM-CAPP v1", [body_text(text, "foundation")],
                   ["Say what v1 already does and what it cannot guarantee.", "Say what MTP-1 adds, in one sentence."]))
    S.append(slide("rqs", "Research questions", [f"- **RQ{n}**: {q}" for n, q in P.RQ.items()],
                   ["Read the questions in your own words, not off the slide.", "Say which of them the data of this talk can answer and which not."]))
    S.append(slide("method", "Components under test", [body_text(text, "method"), "", method_table(cfg)],
                   ["Walk down the rows: each row adds one component to its parent.", "The last column says which research question the row answers."]))
    llms = ", ".join(f"{m['label']} (`{m['model']}`)" for m in cfg["models"])
    parts = ctx.get("parts")
    facts = [f"- Grid `{cfg['name']}`: dataset `{cfg['dataset']}` ({'sealed final set' if P.is_final(cfg) else 'development set'})"
             + (f", {len(parts)} parts" if parts else "") + f", {cfg['repeats']} repeats per cell.",
             f"- LLMs: {llms}.",
             f"- Statistics: paired bootstrap, 95% interval over parts ({cfg['resamples']} resamples); McNemar's exact test for validity; no correction for "
             "the number of comparisons."]
    fr = ctx.get("freeze")
    facts.append(f"- Configuration freeze `{fr['manifest']['id']}`, working tree {fr['verify']['status']}." if fr else "- No freeze manifest: the configuration is not pinned.")
    S.append(slide("protocol", "Data and protocol", facts + ["", body_text(text, "protocol")],
                   ["Say why three LLMs: no conclusion should rest on one model.", "Say how the development set and the sealed set were kept apart."]))
    b = ctx.get("benchmark")
    if b:
        bp, ag = b["parts"], b["agreement"]
        sources = sorted({p.get("source") or "(none)" for p in bp})
        bl = [f"- {len(bp)} parts; sources: {', '.join(sources[:4])}{' ...' if len(sources) > 4 else ''}."]
        bl.append(f"- Human-human agreement (M10) on {ag['n']} parts: exact match {ag['exact_rate']:.2f}, LCS ratio {ag['lcs_ratio_mean']:.2f}, "
                  f"Kendall tau distance {ag['kendall_tau_mean']:.2f}." if ag["n"] else "- Human-human agreement (M10): no second-annotator plans.")
        S.append(slide("benchmark", "Expert benchmark", bl + ["", body_text(text, "benchmark")],
                       ["Say who wrote the reference plans and who reviewed them.", "The agreement figure is the ceiling for any match with the reference."]))
    S.append(slide("headline", "First results", note + [body_text(text, "headline"), "", P.headline_table(ctx)],
                   ["The two rows per LLM are the first and the deepest row of the grid, chosen by a rule that does not look at the results.",
                    "Read M2 LLM-final, not M2 final: the Route Builder fallback makes M2 final almost always 1.0."]))
    fig = _figure(ctx, "fig_ablation.png", "Ablation path per configuration", out_dir)
    if fig:
        S.append(slide("headline_fig", "First results: the ablation path", note + [fig], ["Each step adds one component; say which step moved which metric."]))
    for n in range(1, 7):
        pairs = [c for c in P.comparisons(cfg) if c["rq"] == n]
        head = [f"**{P.RQ[n]}**", ""] + note + [body_text(text, f"rq{n}"), ""]
        extra = None
        if n == 1:
            body = head + [_cap_table(P.baseline_table(ctx))]
            notes = ["Say what fails first and how often, in the words of the failure types.", "The share is of raw LLM routes, before any repair or fallback."]
            extra = slide("rq1_types", "RQ1: Failure types of the raw routes", note + [m3_compact(ctx)],
                          ["Name the failure types in the order of the table.", "The cases of the case-study slide are drawn from these types."])
        elif n == 6:
            ex = ctx.get("extraction")
            body = head + (([f"_(+{len(ex) - 1} more run(s) in the paper draft)_", ""] if len(ex) > 1 else []) + ["```"] + ex[0]["text"].splitlines()[:14] + ["```"]
                           if ex else ["_No extraction run given: M9 is not available._"])
            notes = ["The agreement-based figure is scored after an oracle user: an upper bound, not what a real user does.", "Say how many drawings this rests on."]
        else:
            body = head + [pair_slide_table(ctx, pairs, RQ_METRICS[n]) if pairs else "_No row of this grid isolates this component._"]
            notes = [f"Name the rows that answer this question: {', '.join(c['row'] + ' vs ' + c['parent'] for c in pairs) or 'none in this grid'}.",
                     "Read the interval, not the point estimate; say whether the LLMs agree (the last column).", "Say what was not tested."]
            if n == 3 and pairs:
                extra = slide("rq3_m5", "RQ3: How much of the LLM route survives", note + [P.m5_table(ctx, pairs)],
                              ["M5 is the share of the LLM's operations that is still in the route after the fix: the minimal-edit claim is about this number."])
        S.append(slide(f"rq{n}", f"RQ{n}: {RQ_TITLES[n]}", body, notes))
        if extra:
            S.append(extra)
    fig = _figure(ctx, "fig_cost.png", "Cost against plan quality", out_dir)
    if fig:
        cost_text = (text.get("cost") or "").strip()
        S.append(slide("cost", "Cost and quality", note + ([cost_text, ""] if cost_text else []) + [fig],
                       ["Cost is list price in USD where a price is configured; a model without a price shows n/a, never 0."]))
    cases = ctx.get("cases")
    count = (f"{len(cases.get('cases', []))} case(s) selected by a fixed rule; {cases.get('n_without_commentary', '?')} still without commentary."
             if cases else "_No case studies given._")
    S.append(slide("cases", "Failure case studies", [count, "", body_text(text, "cases")],
                   ["Show one case in full: the part, the route of the LLM, the violation, what the repair kept.", "Cite the case by its id."]))
    lim = [f"- {first_sentence(x)}" for x in P.limitations(ctx)[:5]]
    S.append(slide("limitations", "Limitations", lim + ["", body_text(text, "limitations")],
                   ["The bullets above are generated from this run; the rest of the list is in the paper draft.", "Say the one limit that matters most for reading the results."]))
    repro = ["- " + ln.lstrip("- ") for ln in P.reproducibility(ctx)[:3]]
    rel = ctx.get("slides_release")
    repro += [f"- Rebuild and check: `python scripts/release.py verify {rel}`." if rel else "- Tables: `python scripts/make_tables.py build <grid>`; draft: `python scripts/paper_draft.py <grid>`."]
    S.append(slide("repro", "Reproducibility", repro, ["Say that every figure and table is regenerated by one command from the finished runs."]))
    S.append(slide("conclusion", "Conclusion", [body_text(text, "conclusion")],
                   ["Answer each research question in one clause, as far as the data goes.", "End with the next steps, not with the limitations."]))
    return S


# ---------------------------------------------------------------- readiness
def check(cid, level, ok, message):
    return P.check(cid, level, ok, message)


def readiness(ctx, text: dict) -> list:
    """The checks of the paper (the numbers must be usable) without the ones about the paper's prose, and the checks of the slides' prose."""
    C = [c for c in P.readiness(ctx, {}) if not c["id"].startswith("text_")]
    st = writing_status(text, ctx)
    missing = [s["title"] for s in st if s["required"] and s["state"] == "empty"]
    C.append(check("slides_missing", "blocking", not missing, "every slide has its sentences" if not missing else
                   f"{len(missing)} slide(s) without your sentences: {', '.join(missing)}"))
    left = [f"{k}: {m}" for k, v in text.items() if k in SLIDE_TEXT for m in PLACEHOLDER_MARKS if m in v]
    C.append(check("slides_placeholders", "blocking", not left, "no TODO or placeholder left in the slides' text" if not left else "placeholder left in: " + ", ".join(left)))
    off = [f"{s['title']} ({s['words']} words, budget {s['budget'][0]}-{s['budget'][1]})" for s in st if s["state"] in ("short", "long")]
    C.append(check("slides_budget", "advisory", not off, "every slide's text is within its budget" if not off else "outside the budget: " + "; ".join(off)))
    return C


def render_readiness(checks, text, ctx) -> str:
    blocking = [c for c in checks if c["level"] == "blocking" and not c["ok"]]
    adv = [c for c in checks if c["level"] == "advisory" and not c["ok"]]
    L = [("NOT READY" if blocking else "READY") + f": {len(blocking)} blocking, {len(adv)} advisory item(s) open", ""]
    for c in checks:
        L.append(f"[{'ok' if c['ok'] else ('BLOCKING' if c['level'] == 'blocking' else 'advisory')}] {c['message']}")
    L += ["", "Slides' text:"]
    for s in writing_status(text, ctx):
        L.append(f"  {s['title']:26s} {s['words']:4d} words  ({s['budget'][0]}-{s['budget'][1]})  {s['state']}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- rendering
def _notes_pandoc(notes):
    return ["", "::: notes"] + [f"- {n}" for n in notes] + [":::"] if notes else []


def _notes_marp(notes):
    return ["", "<!--", *notes, "-->"] if notes else []


def render(ctx, text: dict, out_dir: str = ".", fmt: str = "pandoc") -> str:
    if fmt not in FORMATS:
        raise P.PaperError(f"unknown format {fmt!r}: use {' or '.join(FORMATS)}")
    checks = readiness(ctx, text)
    S = build_slides(ctx, text, checks, out_dir)
    title = (text.get("title") or "").strip() or "TO WRITE: title"
    authors = (text.get("authors") or "").strip() or "TO WRITE: authors and date"
    dev = not P.is_final(ctx["cfg"])
    q = lambda s: json.dumps(s, ensure_ascii=False)                   # a YAML double-quoted scalar
    L = []
    if fmt == "pandoc":
        L += ["---", f"title: {q(title)}", f"author: {q(authors)}"] + ([f"subtitle: {q('DEVELOPMENT RESULTS, NOT PAPER RESULTS')}"] if dev else []) + ["---", ""]
    else:
        L += ["---", "marp: true", "paginate: true", "---", "", f"# {title}", "", authors] + (["", "**DEVELOPMENT RESULTS, NOT PAPER RESULTS**"] if dev else [])
    for s in S:
        L += [""] + (["---", ""] if fmt == "marp" else []) + [f"## {s['title']}", ""] + s["body"] + (_notes_marp(s["notes"]) if fmt == "marp" else _notes_pandoc(s["notes"]))
        if fmt == "pandoc":
            L.append("")
    return "\n".join(L).rstrip() + "\n"


def write_all(ctx, text_path: str, out_dir: str, fmt: str = "pandoc") -> dict:
    """Writes slides.md and readiness.txt into out_dir and adds the missing entries to the text file."""
    os.makedirs(out_dir, exist_ok=True)
    text = P.load_text(text_path)
    added = sync_text(text_path, text)
    text = P.load_text(text_path)
    checks = readiness(ctx, text)
    paths = {"slides": os.path.join(out_dir, "slides.md"), "readiness": os.path.join(out_dir, "readiness.txt"), "text": text_path}
    with open(paths["slides"], "w", encoding="utf-8", newline="\n") as f:
        f.write(render(ctx, text, out_dir, fmt))
    with open(paths["readiness"], "w", encoding="utf-8", newline="\n") as f:
        f.write(render_readiness(checks, text, ctx))
    orphans = sorted(k for k, v in text.items() if k not in SLIDE_TEXT and v.strip())
    n = len(build_slides(ctx, text, checks, out_dir))
    return {"paths": paths, "checks": checks, "added": added, "orphans": orphans, "slides": n + 1,
            "ready": not any(c["level"] == "blocking" and not c["ok"] for c in checks)}


# ---------------------------------------------------------------- from a release folder
def collect_from_release(release_dir: str, root: str = None, results_dir: str = None) -> dict:
    """The context of a release folder: the grid, benchmark, freeze and extraction runs named by its recipe, with the tables and the case
    studies of the folder itself, so the slides show exactly what the release holds. Raises R.ReleaseError."""
    from capp import config
    root = os.path.abspath(root or str(config.ROOT))
    rd = os.path.abspath(release_dir)
    R.load_manifest(rd)
    recipe_path = os.path.join(rd, "inputs", "recipe.json")
    if not os.path.isfile(recipe_path):
        raise R.ReleaseError(f"{recipe_path} is missing: not a release folder this tool can read")
    rec = R.load_recipe(recipe_path)
    ins = R.inputs(rec, root)
    one = {i["what"]: i for i in ins}
    if not one["grid"]["exists"]:
        raise R.ReleaseError(f"the grid {one['grid']['path']} of the release is not there any more")
    missing = [f"{i['what']} {i['path']}" for i in ins if i["required"] and not i["exists"] and i["what"] != "case-study run"]
    if missing:
        raise R.ReleaseError("input(s) of the release not found: " + ", ".join(missing))
    case_dir = os.path.join(rd, "case_studies") if os.path.isdir(os.path.join(rd, "case_studies")) else None
    bench = one["benchmark"]["abs"] if "benchmark" in one else None
    fr = one["freeze manifest"]
    ctx = P.collect(one["grid"]["abs"], bench, case_dir, [i["abs"] for i in ins if i["what"] == "extraction run"],
                    fr["abs"] if (fr["exists"] or rec["freeze"]) else None, os.path.join(rd, "tables"),
                    results_dir or (R._abs(root, rec["results"]) if rec["results"] else None), root)
    ctx["slides_release"] = release_dir
    return ctx
