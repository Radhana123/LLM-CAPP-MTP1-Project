"""Step 28 (WP12): the presentation skeleton (capp/slides.py, scripts/slides.py).

Offline. The runs are made by the real harness with a deterministic fake planner (the helpers of the paper-draft tests), so the slides are
made from real records, real Table 4 code and real paired statistics. What the tests pin down: the facts on the slides are the facts of the
paper draft; each paired comparison lands on the slide of the research question it answers; on every slide the sentences come first and
at most one table or figure comes last (pandoc starts a new slide for whatever follows a table, which the real conversion test checks);
the author's sentences are put in place and never touched; the generated text makes no claim about what the numbers mean; and the
readiness check names what stands between the deck and a talk.
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capp import casestudy as C, paperdraft as P, release as R, slides as S      # noqa: E402
from test_paperdraft import bench_parts, grid, sealed, write_extraction_run           # noqa: E402,F401  (grid and sealed are fixtures)
from test_release import repo, run_folder, world                                      # noqa: E402,F401  (world and repo are fixtures)


@pytest.fixture(scope="module")
def ctx(grid):
    out, path = grid
    return P.collect(path, root=out)


def full_text(ctx=None, **over):
    t = {k: " ".join(["word"] * ((lo + hi) // 2 + 1)) for k, (title, req, (lo, hi), hint) in S.SLIDE_TEXT.items()}
    t["title"], t["authors"] = "Reliable plans from LLMs", "R. Author"
    t.update(over)
    return t


def render(ctx, text=None, fmt="pandoc", out_dir="."):
    return S.render(ctx, text if text is not None else {}, out_dir, fmt)


def slide_titles(md):
    return re.findall(r"^## (.+)$", md, flags=re.M)


def slide_bodies(md):
    """{title: body without the speaker notes} for the pandoc output."""
    parts = re.split(r"^## ", md, flags=re.M)[1:]
    out = {}
    for p in parts:
        title, _, body = p.partition("\n")
        out[title] = body.split("\n::: notes")[0]
    return out


def checks_of(ctx, text):
    return {c["id"]: c for c in S.readiness(ctx, text)}


# ---------------------------------------------------------------- the author's text
def test_every_slide_text_field_has_a_sane_budget_and_a_hint():
    for key, (title, req, (lo, hi), hint) in S.SLIDE_TEXT.items():
        assert title and hint and 0 <= lo <= hi <= 100 and isinstance(req, bool), key
    assert list(S.SLIDE_TEXT)[:2] == ["title", "authors"] and "cost" in S.SLIDE_TEXT and not S.SLIDE_TEXT["cost"][1]


def test_sync_text_adds_the_missing_entries_and_never_overwrites(tmp_path):
    p = tmp_path / "t" / "slides_text.json"
    assert S.sync_text(str(p), {}) == len(S.SLIDE_TEXT)
    data = json.load(open(p, encoding="utf-8"))
    assert list(data) == list(S.SLIDE_TEXT) and not any(data.values())
    data.update(problem="My words.", mine="kept")
    del data["conclusion"]
    json.dump(data, open(p, "w", encoding="utf-8"))
    assert S.sync_text(str(p), P.load_text(str(p))) == 1
    again = json.load(open(p, encoding="utf-8"))
    assert again["problem"] == "My words." and again["mine"] == "kept" and again["conclusion"] == ""
    before, mtime = p.read_bytes(), p.stat().st_mtime_ns
    assert S.sync_text(str(p), P.load_text(str(p))) == 0
    assert p.read_bytes() == before and p.stat().st_mtime_ns == mtime


def test_writing_status_states_and_the_benchmark_slide_only_when_there_is_a_benchmark(ctx):
    st = {s["key"]: s for s in S.writing_status({"problem": "one two", "foundation": " ".join(["w"] * 20), "method": " ".join(["w"] * 100)}, ctx)}
    assert st["problem"]["state"] == "short" and st["foundation"]["state"] == "ok" and st["method"]["state"] == "long" and st["protocol"]["state"] == "empty"
    assert "benchmark" not in st
    with_bench = dict(ctx, benchmark={"parts": [], "agreement": {"n": 0}})
    assert "benchmark" in {s["key"] for s in S.writing_status({}, with_bench)}
    assert S.required("benchmark", ctx) is False and S.required("benchmark", with_bench) is True and S.required("cost", ctx) is False


def test_a_missing_entry_shows_what_to_write_and_a_written_one_shows_the_text():
    assert S.body_text({}, "problem").startswith("_TO WRITE (15-70 words): Why an LLM process plan")
    assert S.body_text({"problem": "  Plans fail.  "}, "problem") == "Plans fail."


# ---------------------------------------------------------------- the deck
EXPECTED = ["Problem and gap", "Foundation: LLM-CAPP v1", "Research questions", "Components under test", "Data and protocol", "First results",
            "RQ1: Baseline failures", "RQ1: Failure types of the raw routes", "RQ2: Two-step planning and the hint", "RQ3: Localized repair",
            "RQ3: How much of the LLM route survives", "RQ4: Candidates and selection", "RQ5: Repair or constrain", "RQ6: Extraction and confirmation",
            "Failure case studies", "Limitations", "Reproducibility", "Conclusion"]


def test_the_slides_come_in_the_order_of_the_talk(ctx):
    assert slide_titles(render(ctx)) == EXPECTED


def test_the_benchmark_slide_follows_the_protocol_when_there_is_a_benchmark(grid, tmp_path):
    out, path = grid
    bp = tmp_path / "bench.json"
    json.dump(bench_parts(), open(bp, "w"))
    c = P.collect(path, benchmark=str(bp), root=out)
    md = render(c)
    titles = slide_titles(md)
    assert titles[titles.index("Data and protocol") + 1] == "Expert benchmark"
    body = slide_bodies(md)["Expert benchmark"]
    assert "5 parts; sources: Handbook example 4.2" in body and "Human-human agreement (M10) on 3 parts: exact match" in body


def test_the_title_slide_is_the_metadata_in_pandoc_and_the_first_slide_in_marp(ctx):
    md = render(ctx, {"title": 'A "quoted": title', "authors": "R. Author, Prof. S. Deb"})
    head = md.split("---\n")[1]
    lines = dict(ln.split(": ", 1) for ln in head.strip().splitlines())
    assert json.loads(lines["title"]) == 'A "quoted": title' and json.loads(lines["author"]) == "R. Author, Prof. S. Deb"
    assert json.loads(lines["subtitle"]) == "DEVELOPMENT RESULTS, NOT PAPER RESULTS"
    assert 'title: "TO WRITE: title"' in render(ctx) and md.count("\n---\n") == 1
    marp = render(ctx, {"title": "T", "authors": "A"}, "marp")
    assert marp.startswith("---\nmarp: true\npaginate: true\n---\n\n# T\n\nA\n\n**DEVELOPMENT RESULTS, NOT PAPER RESULTS**")


def test_pandoc_has_no_separators_between_slides_and_marp_has_one_before_each(ctx):
    md = render(ctx)
    assert [ln for ln in md.splitlines() if ln.strip() == "---"] == ["---", "---"]
    marp = render(ctx, fmt="marp")
    assert [ln for ln in marp.splitlines() if ln.strip() == "---"].__len__() == 2 + len(EXPECTED)
    assert "::: notes" not in marp and "<!--" in marp and "-->" in marp
    assert md.count("::: notes") == md.count("\n:::\n") and "<!--" not in md


def test_speaker_notes_are_prompts_not_findings(ctx):
    md = render(ctx)
    notes = re.findall(r"::: notes\n(.*?)\n:::", md, flags=re.S)
    assert len(notes) == len(EXPECTED) and all(n.startswith("- ") for n in notes)
    assert "- Name the rows that answer this question: A3 vs A2, A4 vs A2." in md
    assert "Read the interval, not the point estimate" in md


def test_unwritten_slides_show_what_to_write_and_written_ones_the_text(ctx):
    md = render(ctx, {"problem": "Plans from LLMs break precedence rules.", "rq3": "Repair kept most of the route."})
    bodies = slide_bodies(md)
    assert "Plans from LLMs break precedence rules." in bodies["Problem and gap"] and "_TO WRITE" not in bodies["Problem and gap"]
    assert "Repair kept most of the route." in bodies["RQ3: Localized repair"]
    assert "_TO WRITE (15-70 words): Two-step, hint, few-shot" in bodies["RQ2: Two-step planning and the hint"]


def assert_tables_last(md):
    tables = 0
    for title, body in slide_bodies(md).items():
        lines = [ln for ln in body.splitlines() if ln.strip()]
        table_idx = [i for i, ln in enumerate(lines) if ln.startswith("|")]
        if table_idx:
            tables += 1
            first, last = table_idx[0], table_idx[-1]
            assert table_idx == list(range(first, last + 1)), f"{title}: two tables or a gap in the table"
            assert last == len(lines) - 1, f"{title}: something after the table"
        assert sum(1 for ln in lines if ln.startswith("![")) <= 1, title
    return tables


def test_on_every_slide_the_sentences_come_first_and_one_table_or_figure_comes_last(ctx):
    assert assert_tables_last(render(ctx, full_text())) >= 6


def test_the_same_holds_with_case_studies_a_benchmark_and_an_extraction_report(release):
    repo, out = release
    c = S.collect_from_release(str(out), str(repo))
    md = render(c, full_text())
    bodies = slide_bodies(md)
    assert "| Type | Raw routes | Share | Parts |" in bodies["RQ1: Failure types of the raw routes"]              # the case-study table is in the deck
    assert assert_tables_last(md) >= 7


def test_the_deck_converts_to_one_pptx_slide_per_heading(ctx, tmp_path):
    if shutil.which("pandoc") is None:
        pytest.skip("pandoc is not installed")
    md = tmp_path / "slides.md"
    md.write_text(render(ctx, full_text()), encoding="utf-8")
    res = subprocess.run(["pandoc", str(md), "-o", str(tmp_path / "s.pptx")], capture_output=True, text=True)
    assert res.returncode == 0
    names = zipfile.ZipFile(tmp_path / "s.pptx").namelist()
    assert len([n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)]) == len(EXPECTED) + 1                # the title slide
    assert len([n for n in names if re.match(r"ppt/notesSlides/notesSlide\d+\.xml$", n)]) == len(EXPECTED)


# ---------------------------------------------------------------- the facts
def test_the_first_results_slide_has_the_headline_table_of_the_paper(ctx):
    body = slide_bodies(render(ctx))["First results"]
    assert P.headline_table(ctx) in body
    assert "_Development results (`dev_v2`), not paper results._" in body and body.index("_Development results") < body.index("| LLM |")


def test_the_components_table_says_which_question_each_row_answers(ctx):
    body = slide_bodies(render(ctx))["Components under test"]
    assert "| A1 | B0: five samples | samples | RQ4 |" in body and "| A2 | A1 + R1 repair | repair | RQ3 |" in body
    assert "| A3 | A2 + two-step with hint | two-step, hint | RQ2 |" in body and "| A6 | Constrained planner | repair, constrained | RQ5 |" in body
    assert "| A7 | Extended verifier | feasibility | further |" in body and "| A8 | not built yet | not built | – |" in body and "| A0 | B1: one call | – | – |" in body


def test_each_paired_comparison_is_on_the_slide_of_its_question(ctx):
    bodies = slide_bodies(render(ctx))
    rq = lambda n: next(b for t, b in bodies.items() if t.startswith(f"RQ{n}:"))
    assert "A3 vs A2" in rq(2) and "A4 vs A2" in rq(2) and "A1 vs A0" not in rq(2)
    assert "A2 vs A1" in rq(3) and "A1 vs A0" in rq(4) and "A5 vs A3" in rq(4) and "A6 vs A2" in rq(5)
    assert "A7 vs A5" not in "".join(bodies.values())                                                       # a further component: not a question
    for n in (2, 3, 4, 5):
        assert f"**{P.RQ[n]}**" in rq(n)
    assert "M5 (share of the LLM's operations that survive the fix" in bodies["RQ3: How much of the LLM route survives"]


def test_a_slide_holds_only_the_metrics_of_its_question(ctx):
    body = slide_bodies(render(ctx))["RQ2: Two-step planning and the hint"]
    metrics = set(re.findall(r"^\| A\d vs A\d \([^)]*\) \| ([^|]+) \|", body, flags=re.M))
    assert metrics == {S.T.PAIRED_LABELS["M1"], S.T.PAIRED_LABELS["M2p"]}
    assert body.index("Cells: difference row minus parent, 95% interval over parts; n.s. = the interval includes 0.") < body.index("| Comparison |")


def test_a_long_table_is_cut_and_says_where_the_rest_is():
    rows = "\n".join(f"| r{i} | x |" for i in range(12))
    t = "| a | b |\n|---|---|\n" + rows
    assert S._cap_table(t, 12) == t
    cut = S._cap_table(t, 5)
    assert cut.startswith("_(+7 more rows in the paper draft)_\n\n| a | b |") and cut.count("\n| r") == 5 and "| r4 |" in cut and "| r5 |" not in cut
    assert cut.splitlines()[-1] == "| r4 | x |"                                                                       # the table is last


def test_the_row_cap_applies_to_the_comparison_tables(ctx, monkeypatch):
    monkeypatch.setattr(S, "MAX_ROWS", 1)
    body = slide_bodies(render(ctx))["RQ4: Candidates and selection"]
    assert "more rows in the paper draft" in body and len(re.findall(r"^\| A\d vs A\d", body, flags=re.M)) == 1


def test_the_status_line_follows_the_grid_and_the_readiness(ctx):
    assert S.status_line(ctx, []) == "_Development results (`dev_v2`), not paper results._"
    bodies = slide_bodies(render(ctx))
    assert all("_Development results" in bodies[t] for t in bodies if t.startswith("RQ") or t == "First results")
    assert "_Development results" not in bodies["Research questions"] and "_Development results" not in bodies["Conclusion"]


def test_failure_types_and_case_counts_come_from_the_case_studies(grid, tmp_path):
    out, path = grid
    folder = os.path.join(out, next(d for d in sorted(os.listdir(out)) if d.endswith("_b0")))
    res = C.analyse([C.load_run(folder)])
    C.write_outputs(res, str(tmp_path / "cs"))
    c = P.collect(path, case_dir=str(tmp_path / "cs"), root=out)
    bodies = slide_bodies(render(c))
    types = bodies["RQ1: Failure types of the raw routes"]
    assert "| Type | Raw routes | Share | Parts |" in types and "raw LLM routes); " in types
    count = f"{len(res['cases'])} case(s) selected by a fixed rule; {res['n_without_commentary']} still without commentary."
    assert count in bodies["Failure case studies"] and count.replace("; ", ", ") in types
    none = slide_bodies(render(P.collect(path, root=out)))
    assert "_No case studies given._" in none["RQ1: Failure types of the raw routes"] and "_No case studies given._" in none["Failure case studies"]


def test_the_limitations_slide_shows_the_first_sentences_of_the_generated_list(ctx):
    body = slide_bodies(render(ctx))["Limitations"]
    bullets = [ln for ln in body.splitlines() if ln.startswith("- ")]
    assert 1 <= len(bullets) <= 5 and bullets[0].startswith("- The grid is on the development set `dev_v2`")
    assert all(len(b) <= 152 and (b.endswith(".") or b.endswith("…")) for b in bullets)
    assert S.first_sentence("One. Two.") == "One." and S.first_sentence("No stop") == "No stop" and len(S.first_sentence("x" * 400)) == 150


def test_the_limitations_slide_keeps_the_first_five_items(ctx, monkeypatch):
    monkeypatch.setattr(P, "limitations", lambda c: [f"Limit number {i}. More words." for i in range(8)])
    body = slide_bodies(render(ctx))["Limitations"]
    assert [ln for ln in body.splitlines() if ln.startswith("- ")] == [f"- Limit number {i}." for i in range(5)]


def test_more_case_study_runs_are_a_note_before_the_failure_type_table():
    def run(label):
        return {"label": label, "n_parsed_routes": 12, "table": [{"type": "precedence", "routes": 5, "share": 5 / 12, "parts": 3}, {"type": "differs", "parts": 2}]}
    lines = S.m3_compact({"cases": {"runs": [run("A"), run("B"), run("C")], "cases": [1, 2], "n_without_commentary": 2}}).splitlines()
    assert lines[0] == "Run `A` (12 raw LLM routes); 2 case(s) selected by a fixed rule, 2 still without commentary."
    assert lines[1] == "_(+2 more run(s) in the paper draft)_" and lines[2] == "" and lines[3] == "| Type | Raw routes | Share | Parts |"
    assert lines[-2:] == ["| precedence | 5 | 41.7% | 3 |", "| differs | – | – | 2 |"]
    single = S.m3_compact({"cases": {"runs": [run("A")], "cases": [], "n_without_commentary": 0}})
    assert "more run(s)" not in single


def test_a_second_extraction_run_is_a_note_before_the_report(grid, tmp_path):
    out, path = grid
    runs = []
    for name in ("fe1_extraction", "fe2_extraction"):
        run = str(tmp_path / name)
        write_extraction_run(run)
        runs.append(run)
    body = slide_bodies(render(P.collect(path, extraction=runs, root=out)))["RQ6: Extraction and confirmation"]
    lines = [ln for ln in body.splitlines() if ln.strip()]
    note = "_(+1 more run(s) in the paper draft)_"
    assert note in lines and lines.index(note) < lines.index("```") and lines[-1] == "```"
    assert "fe1_extraction" in body and "fe2_extraction" not in body


def test_the_extraction_report_is_on_the_rq6_slide(grid, tmp_path):
    out, path = grid
    run = str(tmp_path / "fe1_extraction")
    write_extraction_run(run)
    body = slide_bodies(render(P.collect(path, extraction=[run], root=out)))["RQ6: Extraction and confirmation"]
    assert "```\n=== M9 extraction quality - fe1_extraction: 3 of 3 drawings scored ===" in body and body.rstrip().endswith("```")
    assert "_No extraction run given: M9 is not available._" in slide_bodies(render(P.collect(path, root=out)))["RQ6: Extraction and confirmation"]


def test_figures_are_slides_of_their_own_and_only_when_the_files_exist(grid, tmp_path):
    out, path = grid
    tdir = tmp_path / "tables"
    tdir.mkdir()
    assert "First results: the ablation path" not in render(P.collect(path, tables_dir=str(tdir), root=out))
    (tdir / "fig_ablation.png").write_bytes(b"png")
    (tdir / "fig_cost.png").write_bytes(b"png")
    c = P.collect(path, tables_dir=str(tdir), root=out)
    md = render(c, out_dir=str(tmp_path / "slides"))
    titles = slide_titles(md)
    assert titles[titles.index("First results") + 1] == "First results: the ablation path" and "Cost and quality" in titles
    assert "![Ablation path per configuration](../tables/fig_ablation.png)" in md and "![Cost against plan quality](../tables/fig_cost.png)" in md
    assert "A cost line of mine." in render(c, {"cost": "A cost line of mine."})


def test_the_reproducibility_slide_names_the_freeze_and_the_commands(ctx, sealed):
    body = slide_bodies(render(ctx))["Reproducibility"]
    assert "No freeze manifest was found" in body and "python scripts/make_tables.py build <grid>" in body
    repo, cfg_path, man, _ = sealed
    sealed_body = slide_bodies(render(P.collect(cfg_path, root=repo)))["Reproducibility"]
    assert f"Freeze `{man['id']}`" in sealed_body and "the working tree now: **ok**" in sealed_body
    c = P.collect(cfg_path, root=repo)
    c["slides_release"] = "release/v1"
    assert "python scripts/release.py verify release/v1" in slide_bodies(render(c))["Reproducibility"]


def test_the_data_slide_names_the_grid_the_llms_and_the_statistics(ctx):
    body = slide_bodies(render(ctx))["Data and protocol"]
    assert "Grid `paper_test`: dataset `dev_v2` (development set), 6 parts, 2 repeats per cell." in body
    assert "LLMs: Model A (`qwen/qwen3.8-27b`), Model B (`openai/gpt-oss-120b`), Model C (`openai/gpt-oss-20b`)." in body
    assert "paired bootstrap, 95% interval over parts (200 resamples)" in body and "no correction for the number of comparisons" in body


def test_the_generated_text_makes_no_claim_about_what_the_numbers_mean(ctx):
    for fmt in S.FORMATS:
        md = render(ctx, fmt=fmt).lower()
        for q in P.RQ.values():
            md = md.replace(q.lower(), "")                                                       # the questions themselves say "improve"
        for pattern in (r"significant", r"outperform", r"demonstrat", r"\bprove[sd]?\b", r"\bwe show\b", r"superior", r"\bconfirms\b", r"\bshows that\b",
                        r"\bsupports\b", r"\bimprove[sd]?\b"):
            assert not re.search(pattern, md), (fmt, pattern)


def test_an_unknown_format_is_refused(ctx):
    with pytest.raises(P.PaperError, match="unknown format"):
        render(ctx, fmt="keynote")


# ---------------------------------------------------------------- readiness
def test_a_development_grid_with_all_the_sentences_is_blocked_only_by_the_dataset(ctx):
    blocking = [c["message"] for c in S.readiness(ctx, full_text()) if c["level"] == "blocking" and not c["ok"]]
    assert len(blocking) == 1 and "development set" in blocking[0]


def test_the_checks_about_the_papers_prose_are_not_slide_checks(ctx):
    ids = {c["id"] for c in S.readiness(ctx, {})}
    assert not any(i.startswith("text_") for i in ids) and {"slides_missing", "slides_placeholders", "slides_budget", "sealed", "llms"} <= ids


def test_missing_sentences_block_and_are_named(ctx):
    chk = checks_of(ctx, full_text(problem="", rq3=""))
    assert chk["slides_missing"]["ok"] is False and chk["slides_missing"]["level"] == "blocking"
    assert "2 slide(s) without your sentences: Problem and gap, RQ3" in chk["slides_missing"]["message"]
    assert checks_of(ctx, full_text(cost=""))["slides_missing"]["ok"]                                          # cost is optional
    assert "Expert benchmark" not in checks_of(ctx, {})["slides_missing"]["message"]


def test_the_benchmark_slide_needs_sentences_only_with_a_benchmark(grid, tmp_path):
    out, path = grid
    bp = tmp_path / "b.json"
    json.dump(bench_parts(), open(bp, "w"))
    c = P.collect(path, benchmark=str(bp), root=out)
    t = full_text()
    del t["benchmark"]
    assert "Expert benchmark" in checks_of(c, t)["slides_missing"]["message"]
    assert checks_of(c, full_text())["slides_missing"]["ok"]


@pytest.mark.parametrize("mark", ["TODO", "_TO WRITE", "[TO CONFIRM]"])
def test_a_placeholder_in_the_sentences_blocks(ctx, mark):
    chk = checks_of(ctx, full_text(problem=f"Plans fail {mark} here"))["slides_placeholders"]
    assert chk["ok"] is False and chk["level"] == "blocking" and "problem" in chk["message"]


def test_an_essay_on_a_slide_is_an_advisory(ctx):
    chk = checks_of(ctx, full_text(problem=" ".join(["w"] * 120)))["slides_budget"]
    assert chk["ok"] is False and chk["level"] == "advisory" and "Problem and gap (120 words, budget 15-70)" in chk["message"]
    assert checks_of(ctx, full_text())["slides_budget"]["ok"]


def test_a_sealed_grid_under_a_matching_freeze_with_all_sentences_is_ready(sealed):
    repo, cfg_path, man, _ = sealed
    c = P.collect(cfg_path, root=repo)
    checks = S.readiness(c, full_text())
    assert [x["message"] for x in checks if x["level"] == "blocking" and not x["ok"]] == []
    md = render(c, full_text())
    assert "_Development results" not in md and "_Not final" not in md
    drift = Path(repo) / "capp" / "planners.py"
    drift.write_text("x = 2\n")
    c2 = P.collect(cfg_path, root=repo)
    assert any("drifted" in x["message"] for x in S.readiness(c2, full_text()) if x["level"] == "blocking" and not x["ok"])
    assert "_Not final: 1 blocking item(s), see the readiness check._" in slide_bodies(render(c2, full_text()))["First results"]


# ---------------------------------------------------------------- writing the files
def test_write_all_writes_the_files_and_keeps_the_text(ctx, tmp_path):
    text_path = tmp_path / "mine" / "slides_text.json"
    res = S.write_all(ctx, str(text_path), str(tmp_path / "out"))
    assert set(res["paths"]) == {"slides", "readiness", "text"} and res["added"] == len(S.SLIDE_TEXT) and res["slides"] == len(EXPECTED) + 1
    assert res["ready"] is False and (tmp_path / "out" / "slides.md").is_file() and (tmp_path / "out" / "readiness.txt").read_text().startswith("NOT READY")
    data = json.load(open(text_path, encoding="utf-8"))
    data.update(problem="My words about the problem.", extra_note="not a slide")
    json.dump(data, open(text_path, "w", encoding="utf-8"))
    res = S.write_all(ctx, str(text_path), str(tmp_path / "out"))
    assert res["added"] == 0 and res["orphans"] == ["extra_note"]
    assert "My words about the problem." in (tmp_path / "out" / "slides.md").read_text()
    assert json.load(open(text_path, encoding="utf-8"))["problem"] == "My words about the problem."
    res = S.write_all(ctx, str(text_path), str(tmp_path / "out_marp"), "marp")
    assert (tmp_path / "out_marp" / "slides.md").read_text().startswith("---\nmarp: true")


def test_a_text_file_that_is_not_valid_is_refused(ctx, tmp_path):
    p = tmp_path / "t.json"
    p.write_text("[1, 2]")
    with pytest.raises(P.PaperError):
        S.write_all(ctx, str(p), str(tmp_path / "o"))


# ---------------------------------------------------------------- from a release folder
@pytest.fixture()
def release(repo):
    folder = f"results/{run_folder(repo, 'b0')}"
    (repo / "data").mkdir()
    json.dump(bench_parts(), open(repo / "data" / "bench.json", "w"))
    write_extraction_run(str(repo / "fe1_extraction"))
    rec = {"name": "t", "grid": "configs/grid.json", "results": "results", "benchmark": "data/bench.json", "case_studies": {"runs": [folder]},
           "extraction": ["fe1_extraction"]}
    path = repo / "configs" / "release.json"
    path.write_text(json.dumps(rec))
    out = repo / "release" / "t"
    R.build(str(path), str(out), root=str(repo), figures=False)
    return repo, out


def test_the_context_of_a_release_uses_its_tables_cases_and_inputs(release):
    repo, out = release
    c = S.collect_from_release(str(out), str(repo))
    assert c["tables_dir"] == str(out / "tables") and c["case_dir"] == str(out / "case_studies") and c["cases"]["cases"]
    assert c["benchmark"] and len(c["benchmark"]["parts"]) == 5 and len(c["extraction"]) == 1 and c["slides_release"] == str(out)
    md = render(c)
    assert "Expert benchmark" in slide_titles(md) and "=== M9 extraction quality - fe1_extraction" in md
    assert f"python scripts/release.py verify {out}" in md and "case(s) selected by a fixed rule" in md


def test_a_folder_that_is_not_a_release_is_refused(tmp_path, release):
    repo, out = release
    with pytest.raises(R.ReleaseError, match="not a release folder"):
        S.collect_from_release(str(tmp_path), str(repo))
    os.remove(out / "inputs" / "recipe.json")
    with pytest.raises(R.ReleaseError, match="recipe.json is missing"):
        S.collect_from_release(str(out), str(repo))


def test_a_release_whose_inputs_are_gone_is_refused_with_the_names(release):
    repo, out = release
    shutil.rmtree(repo / "fe1_extraction")
    with pytest.raises(R.ReleaseError, match="input\\(s\\) of the release not found: extraction run fe1_extraction"):
        S.collect_from_release(str(out), str(repo))
    os.remove(repo / "configs" / "grid.json")
    with pytest.raises(R.ReleaseError, match="grid configs/grid.json of the release is not there any more"):
        S.collect_from_release(str(out), str(repo))


def test_case_study_runs_may_be_gone_because_the_release_holds_the_cases(release):
    repo, out = release
    shutil.rmtree(repo / "results" / run_folder(repo, "b0"))
    assert S.collect_from_release(str(out), str(repo))["cases"]["cases"]


# ---------------------------------------------------------------- the command line
@pytest.fixture()
def cli():
    spec = importlib.util.spec_from_file_location("slides_cli", os.path.join(ROOT, "scripts", "slides.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_from_a_grid_config(cli, grid, tmp_path, capsys):
    out, path = grid
    o = tmp_path / "slides"
    assert cli.main([path, "--out", str(o)]) == 0
    text = capsys.readouterr().out
    assert "NOT READY" in text and f"Slides:    {o / 'slides.md'}  (19 slides, pandoc)" in text and "17 empty entries added" not in text
    assert f"({len(S.SLIDE_TEXT)} empty entries added)" in text
    assert (o / "slides.md").is_file() and (o / "readiness.txt").is_file() and (o / "slides_text.json").is_file()
    assert cli.main([path, "--out", str(o), "--format", "marp"]) == 0
    assert "(19 slides, marp)" in capsys.readouterr().out and (o / "slides.md").read_text().startswith("---\nmarp: true")
    assert cli.main([path, "--out", str(o), "--strict"]) == 1
    assert "--strict: blocking item(s) open." in capsys.readouterr().err and (o / "slides.md").is_file()


def test_cli_notes_entries_that_match_no_slide(cli, grid, tmp_path, capsys):
    out, path = grid
    t = tmp_path / "t.json"
    t.write_text(json.dumps({"mine": "something"}))
    assert cli.main([path, "--out", str(tmp_path / "o"), "--text", str(t)]) == 0
    assert "Note: 1 entry of the text file match no slide (mine); kept." in capsys.readouterr().out
    assert json.load(open(t, encoding="utf-8"))["mine"] == "something"


def test_cli_problems_exit_2(cli, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert cli.main([str(bad), "--out", str(tmp_path / "o")]) == 2
    assert cli.main([str(tmp_path / "missing.json"), "--out", str(tmp_path / "o")]) == 2
    assert cli.main([str(tmp_path), "--out", str(tmp_path / "o")]) == 2                              # a folder that is not a release
    assert "STOPPED:" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        cli.main([str(bad), "--format", "keynote"])
    assert exc.value.code == 2


def test_cli_from_a_release_folder(cli, world, tmp_path, capsys):
    """The command line works on the real repository root, so the recipe has absolute paths."""
    repo = tmp_path / "repo"
    shutil.copytree(world, repo)
    rec = {"name": "cli", "grid": str(repo / "configs" / "grid.json"), "results": str(repo / "results"), "freeze": str(tmp_path / "none.json")}
    (tmp_path / "recipe.json").write_text(json.dumps(rec))
    out = tmp_path / "rel"
    R.build(str(tmp_path / "recipe.json"), str(out), root=ROOT, figures=False)
    assert cli.main([str(out), "--out", str(tmp_path / "slides")]) == 0
    text = capsys.readouterr().out
    assert "(19 slides, pandoc)" in text
    assert f"python scripts/release.py verify {out}" in (tmp_path / "slides" / "slides.md").read_text()
    assert "freeze manifest" in (tmp_path / "slides" / "readiness.txt").read_text()


def test_the_script_runs_as_a_program(tmp_path):
    res = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "slides.py"), os.path.join("configs", "ablation_template.json"), "--out", str(tmp_path / "s")],
                         capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=180)
    assert res.returncode == 0 and "NOT READY" in res.stdout and (tmp_path / "s" / "slides.md").is_file()
