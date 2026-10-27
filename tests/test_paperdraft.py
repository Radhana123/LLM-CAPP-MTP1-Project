"""Step 26 (WP12): the paper draft generator (capp/paperdraft.py, scripts/paper_draft.py).

Offline. The runs are made by the real harness with a deterministic fake planner (as in the Table 4 tests), so the draft is generated from
real records, real Table 4 code and real paired statistics. What the tests pin down: the numbers in the draft are the numbers of the tables;
each paired comparison lands under the research question it answers; the author's prose is put in place and never touched; the generated
text makes no claim about what the numbers mean; and the readiness check names what stands between the draft and a paper.
"""
import copy
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capp import config, dataset, freeze as F, harness, paperdraft as P, tables as T      # noqa: E402
import route_builder as rb                                                                  # noqa: E402

N_PARTS, REPEATS = 6, 2
MODELS = (("a", "Model A", "qwen/qwen3.8-27b", "p_a", "A", 0.5, 0.5), ("b", "Model B", "openai/gpt-oss-120b", "p_b", "B", 0.4, 0.3),
          ("c", "Model C", "openai/gpt-oss-20b", "p_c", "C", 0.6, 0.6))
ROWS = (("A0", "B1: one call", "b1", None), ("A1", "B0: five samples", "b0", "A0"), ("A2", "A1 + R1 repair", "b0_r1", "A1"),
        ("A3", "A2 + two-step with hint", "b0_r1+ts+h1", "A2"), ("A4", "A2 + two-step without hint", "b0_r1+ts", "A2"),
        ("A5", "A3 + selection", "n5_r1_s3+ts+h1", "A3"), ("A6", "Constrained planner", "n5_r0_s1+cg1", "A2"),
        ("A7", "Extended verifier", "n5_r1_s3+ts+h1+fz2", "A5"), ("A8", "not built yet", None, "A5"))


def make_planner(salt, p_drop, p_swap):
    calls = {}

    def planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        routes = rb.generate_valid_routes(features, max_routes=15)
        if not routes:
            return {"success": False, "steps": [], "raw_response": ""}
        steps = list(routes[(sample_id * 3 + len(features)) % len(routes)]["steps"])
        key = (tuple(features), sample_id)
        calls[key] = calls.get(key, 0) + 1
        r = random.Random(f"{salt}-{','.join(features)}-{sample_id}-{calls[key]}")
        if r.random() < p_drop and len(steps) > 3:
            steps.pop(r.randrange(len(steps)))
        if r.random() < p_swap:
            i = r.randrange(len(steps) - 1)
            steps[i], steps[i + 1] = steps[i + 1], steps[i]
        return {"success": True, "steps": steps, "raw_response": "", "model": "fake",
                "meta": {"cached": True, "finish_reason": "stop", "prompt_tokens": 100, "completion_tokens": 20, "latency_s": 0.1}}
    return planner


def cfg_dict(results_dir, dataset_name="dev_v2", rows=ROWS, models=MODELS, limit=N_PARTS):
    return {"name": "paper_test", "dataset": dataset_name, "limit": limit, "repeats": REPEATS, "results_dir": results_dir, "resamples": 200, "seed": 7,
            "models": [{"id": mid, "label": label, "model": model, "tag": tag} for mid, label, model, tag, *_ in models],
            "rows": [{"id": i, "label": label, "system": system, "parent": parent} for i, label, system, parent in rows]}


def make_runs(out, dataset_name="dev_v2", systems=None, limit=N_PARTS):
    systems = systems or [r[2] for r in ROWS if r[2]]
    parts = dataset.load_dataset(dataset_name)[:limit]
    mp = pytest.MonkeyPatch()
    try:
        for mid, label, model, tag, salt, pd, ps in MODELS:
            mp.setenv("CAPP_MODEL_PRIMARY", model)
            planner = make_planner(salt, pd, ps)
            for system in systems:
                harness.run_experiment(system, parts, repeats=REPEATS, tag=tag, out_dir=out, dataset_name=dataset_name, progress=None,
                                       planner_factory=lambda part, role=None, pl=planner: pl, analysis=False)
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("paper_grid"))
    make_runs(out)
    path = os.path.join(out, "grid.json")
    json.dump(cfg_dict(out), open(path, "w"))
    return out, path


@pytest.fixture(scope="module")
def ctx(grid):
    out, path = grid
    return P.collect(path, root=out)                       # no FREEZE.json in the results folder


def full_text(**over):
    t = {k: " ".join(["word"] * ((lo + hi) // 2 + 1)) for k, (title, req, (lo, hi), hint) in P.TEXT_FIELDS.items()}
    t["title"], t["authors"] = "A study of planners", "R. Author"
    t.update(over)
    return t


def sections(md):
    """{heading: body} for the '## ' and '### ' sections of the draft."""
    out, cur = {}, None
    for line in md.splitlines():
        m = re.match(r"^#{2,4} (.+)$", line)
        if m:
            cur = m.group(1)
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def rq_section(md, n):
    key = next(k for k in sections(md) if k.startswith(f"RQ{n}:"))
    return sections(md)[key]


# ---------------------------------------------------------------- which comparison answers which question
@pytest.mark.parametrize("system,parent,labels,rq", [
    ("b0", "b1", ["samples"], 4), ("b0_r1", "b0", ["repair"], 3), ("b0_r1+ts+h1", "b0_r1", ["two-step", "hint"], 2),
    ("b0_r1+ts", "b0_r1", ["two-step"], 2), ("n5_r1_s3+ts+h1+fs2", "n5_r1_s3+ts+h1", ["few-shot"], 2),
    ("n5_r1_s3+ts+h1+fs2+fz2", "n5_r1_s3+ts+h1+fs2", ["feasibility"], None), ("n5_r1_s3+ts+h1+fs2+fz2+cas1", "n5_r1_s3+ts+h1+fs2+fz2", ["cascade"], None),
    ("n5_r0_s1+cg1", "b0_r1", ["repair", "constrained"], 5), ("n5_r1_s3", "b0_r1", ["selection"], 4),
    ("n5_r1_s3+ms", "n5_r1_s3", ["model"], None), ("b1", "b1", [], None)])
def test_components_and_research_questions(system, parent, labels, rq):
    got = P.components(system, parent)
    assert got == labels and P.rq_of(got) == rq


@pytest.mark.parametrize("labels,rq", [
    (["constrained", "two-step"], 5), (["repair", "constrained"], 5), (["two-step", "repair"], 2), (["hint", "samples"], 2),
    (["repair", "samples", "selection"], 3), (["selection", "model"], 4), (["feasibility", "cascade", "model"], None), ([], None)])
def test_a_row_that_changes_several_things_goes_to_the_highest_priority_question(labels, rq):
    assert P.rq_of(labels) == rq


def test_rb_has_no_components():
    assert P.components("rb", "b0") == [] and P.components("b0", "rb") == []


def test_the_ablation_template_maps_to_the_questions_of_the_report():
    cfg = T.load_config(os.path.join(ROOT, "configs", "ablation_template.json"))
    got = {c["row"]: c["rq"] for c in P.comparisons(cfg)}
    assert got == {"A1": 4, "A2": 3, "A3": 2, "A4": 2, "A5": 4, "A6": 2, "A7": None, "A8": None, "A9": 5}
    assert [c["parent"] for c in P.comparisons(cfg) if c["row"] == "A9"] == ["A2"]


def test_rows_that_are_not_built_make_no_comparison():
    cfg = T.validate_config(cfg_dict("x"))
    assert "A8" not in {c["row"] for c in P.comparisons(cfg)}
    assert {c["row"] for c in P.comparisons(cfg)} == {"A1", "A2", "A3", "A4", "A5", "A6", "A7"}


def test_depth_and_the_headline_rows(ctx):
    cfg = ctx["cfg"]
    assert [P.depth(cfg, r["id"]) for r in cfg["rows"]] == [0, 1, 2, 3, 3, 4, 3, 5, 5]
    for m in cfg["models"]:
        rows = P.headline_rows(cfg, ctx["res"], m["id"])
        assert [(t, r["id"]) for t, r in rows] == [("baseline", "A0"), ("deepest", "A7")]            # A8 is not built
    # the deepest row is chosen without looking at any result: remove the runs of A7 for one LLM and the rule moves to the next deepest
    res = copy.copy(ctx["res"])
    res["usable"] = {k: v for k, v in ctx["res"]["usable"].items() if k != ("A7", "a")}
    assert [r["id"] for _, r in P.headline_rows(cfg, res, "a")] == ["A0", "A5"]
    assert [r["id"] for _, r in P.headline_rows(cfg, res, "b")] == ["A0", "A7"]
    res["usable"] = {}
    assert P.headline_rows(cfg, res, "a") == []


# ---------------------------------------------------------------- the author's text
def test_text_file_is_created_with_every_section_and_never_overwritten(tmp_path):
    path = str(tmp_path / "t" / "paper_text.json")
    assert P.load_text(path) == {}
    assert P.sync_text(path, {}) == len(P.TEXT_FIELDS)
    data = json.load(open(path, encoding="utf-8"))
    assert list(data) == list(P.TEXT_FIELDS) and set(data.values()) == {""}
    data["abstract"] = "My abstract, ünïcode."
    data["my_notes"] = "kept although no section has this name"
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    before = open(path, "rb").read()
    assert P.sync_text(path, P.load_text(path)) == 0
    assert open(path, "rb").read() == before                                      # nothing to add: the file is not even rewritten
    del data["conclusion"]
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    assert P.sync_text(path, P.load_text(path)) == 1
    after = json.load(open(path, encoding="utf-8"))
    assert after["abstract"] == "My abstract, ünïcode." and after["my_notes"].startswith("kept") and after["conclusion"] == ""
    assert b"\r" not in open(path, "rb").read()


def test_bad_text_files_are_refused(tmp_path):
    p = tmp_path / "t.json"
    p.write_text("{nope")
    with pytest.raises(P.PaperError, match="not valid JSON"):
        P.load_text(str(p))
    p.write_text('["a"]')
    with pytest.raises(P.PaperError, match="JSON object"):
        P.load_text(str(p))
    p.write_text('{"abstract": 5}')
    with pytest.raises(P.PaperError, match="string values"):
        P.load_text(str(p))


def test_writing_status_and_word_counts():
    assert P.words("one two\nthree\tfour") == 4 and P.words("") == 0 and P.words(None) == 0
    st = {s["key"]: s for s in P.writing_status({"abstract": "x " * 10, "introduction": "y " * 500, "related_work": "z " * 400, "method": "w " * 1000})}
    assert st["abstract"]["state"] == "short" and st["introduction"]["state"] == "ok" and st["related_work"]["state"] == "ok"
    assert st["method"]["state"] == "long" and st["discussion"]["state"] == "empty" and st["acknowledgements"]["required"] is False


# ---------------------------------------------------------------- the draft
def draft(ctx, text=None, out_dir="."):
    return P.render(ctx, text if text is not None else {}, out_dir)


def test_the_draft_has_every_section_in_order(ctx):
    md = draft(ctx)
    heads = [k for k in sections(md) if re.match(r"^(Abstract|\d |Appendix)", k)]
    assert heads == ["Abstract", "1 Introduction", "2 Related work", "3 System and method", "4 Experimental setup", "5 Results", "6 Discussion",
                     "7 Limitations", "8 Conclusion", "Appendix A: readiness", "Appendix B: how to reproduce", "Appendix C: issues of the table build"]
    assert md.startswith("# _TO WRITE: title_")
    for n, q in P.RQ.items():
        assert f"- **RQ{n}**: {q}" in md and f"### RQ{n}: {q}" in md


def test_unwritten_sections_show_what_to_write_and_written_ones_show_the_text(ctx):
    text = {"abstract": "The abstract I wrote.", "rq3": "Repair kept more of the route.", "title": "My title", "authors": "Me"}
    md = draft(ctx, text)
    assert md.startswith("# My title\n\nMe\n")
    assert "The abstract I wrote." in md and "Repair kept more of the route." in md
    assert "_TO WRITE (150-250 words)" not in md and "_TO WRITE (80-250 words): Repair against" not in md
    assert "_TO WRITE (400-700 words)" in md and "_TO WRITE (80-250 words): Two-step" in md
    assert "## Acknowledgements" not in md                                                              # optional and empty: no heading


def test_table4_is_the_table_of_make_tables(ctx):
    md = draft(ctx)
    ref = T.render_markdown(ctx["res"]).splitlines()
    rows = [ln for ln in ref if ln.startswith("| A")]
    assert len(rows) == 3 * len(ROWS) and all(r in md for r in rows)
    assert md.count("Table 4: ablation grid") == 1 and "Table 4: ablation grid, paper_test" not in md
    for m in MODELS:
        assert f"#### {m[1]} (`{m[2]}`)" in md
    assert "A8 | not built yet | – | not built" in md


def test_the_headline_table_has_the_values_of_the_cells(ctx):
    res = ctx["res"]
    md = draft(ctx)
    block = sections(md)["Abstract"]
    for mid, label, *_ in MODELS:
        for tag, rid in (("baseline", "A0"), ("deepest", "A7")):
            c = res["usable"][(rid, mid)]["cell"]
            want = f"| {label} | {rid} ({tag}) | " + next(r[1] for r in ROWS if r[0] == rid) + f" | {T.cell_text(c['M1'], 'M1')} | {T.cell_text(c['M2p'], 'M2p')} | " \
                   f"{T.cell_text(c['M6'], 'M6')} | {T.cell_text(c['M7'], 'M7')} |"
            assert want in block


def test_comparisons_are_under_the_question_they_answer(ctx):
    md = draft(ctx)
    rq2, rq3, rq4, rq5 = (rq_section(md, n) for n in (2, 3, 4, 5))
    assert "A3 vs A2 (two-step, hint)" in rq2 and "A4 vs A2 (two-step)" in rq2 and "A5 vs A3" not in rq2
    assert "A2 vs A1 (repair)" in rq3 and "A3 vs A2" not in rq3
    assert "A1 vs A0 (samples)" in rq4 and "A5 vs A3 (selection)" in rq4 and "A2 vs A1" not in rq4
    assert "A6 vs A2 (repair, constrained)" in rq5
    other = sections(md)["Further components (objectives 5 and 7)"]
    assert "A7 vs A5 (feasibility)" in other and "A6" not in other
    assert "M5 (share of the LLM's operations" in rq3 and "| A1: B0: five samples |" in rq3 and "| A2: A1 + R1 repair |" in rq3
    assert "M5 (share" not in rq2


def test_cells_of_the_comparison_tables_are_the_paired_statistics(ctx):
    res = ctx["res"]
    rq4 = rq_section(draft(ctx), 4)
    # worked by hand: B0 makes five calls per part where B1 makes one, for every part and every repeat, so the difference is exactly +4
    assert rq4.count("+4.00 [+4.00, +4.00] costlier") == 3
    p = next(x for x in res["paired"] if x["model"] == "a" and x["row"] == "A1" and x["metric"] == "M6")
    assert f"{T._ci_text(p, 2)} {P._short(p['verdict'])}" in rq4
    cons = next(c for c in res["consistency"] if c["row"] == "A1" and c["metric"] == "M6")
    assert f"| {cons['label']} |" in rq4
    assert "n.s. = the interval includes 0" in rq4


def test_rq1_shows_the_baseline_rows(ctx):
    rq1 = rq_section(draft(ctx), 1)
    assert "Baseline rows" in rq1 and "| A0 | B1: one call | Model A |" in rq1 and "| A1 | B0: five samples | Model C |" in rq1
    assert "| A2 |" not in rq1 and "No case-study folder" in rq1


def test_a_question_without_a_row_says_so():
    import tempfile
    with tempfile.TemporaryDirectory() as out:
        make_runs(out, systems=["b1", "b0"])
        cfg = cfg_dict(out, rows=ROWS[:2])
        path = os.path.join(out, "g.json")
        json.dump(cfg, open(path, "w"))
        c = P.collect(path, root=out)
        md = draft(c)
        for n in (2, 3, 5):
            assert "No row of this grid isolates this component" in rq_section(md, n)
        assert "A1 vs A0 (samples)" in rq_section(md, 4) and "Further components" not in md


def test_the_generated_text_makes_no_claim_about_what_the_numbers_mean(ctx):
    md = draft(ctx)
    for pattern in (r"significant", r"outperform", r"demonstrat", r"\bprove[sd]?\b", r"\bwe show\b", r"superior", r"\bconfirms\b", r"\bshows that\b", r"\bsupports\b"):
        assert not re.search(pattern, md.lower()), pattern


def test_limitations_follow_from_the_run(ctx):
    lim = P.limitations(ctx)
    assert any("development set `dev_v2`" in x for x in lim) and any("proxy_rb" in x for x in lim)
    assert any("not corrected" in x for x in lim) and any("hosted" in x for x in lim)
    assert any("cannot invent an operation" in x for x in lim)                                 # a constrained row is in the grid
    assert not any("Only" in x and "LLM" in x for x in lim)
    two = copy.deepcopy(ctx)
    two["cfg"]["models"] = two["cfg"]["models"][:2]
    assert any(x.startswith("Only 2 LLM(s)") for x in P.limitations(two))
    broken = copy.deepcopy(ctx)
    broken["res"]["runs"] = {k: dict(v) for k, v in ctx["res"]["runs"].items()}
    broken["res"]["runs"][("A1", "b")]["status"] = "partial"
    assert any("A1/b" in x for x in P.limitations(broken))


def test_the_draft_is_deterministic_and_keeps_the_prose(ctx, tmp_path):
    a = draft(ctx, full_text())
    assert a == draft(ctx, full_text())
    paths = P.write_all(ctx, str(tmp_path / "t.json"), str(tmp_path / "out"))["paths"]
    first = open(paths["draft"], encoding="utf-8").read()
    t = json.load(open(paths["text"], encoding="utf-8"))
    t["abstract"] = "Written by hand."
    json.dump(t, open(paths["text"], "w", encoding="utf-8"))
    P.write_all(ctx, str(tmp_path / "t.json"), str(tmp_path / "out"))
    second = open(paths["draft"], encoding="utf-8").read()
    assert "Written by hand." in second and "Written by hand." not in first
    assert json.load(open(paths["text"], encoding="utf-8"))["abstract"] == "Written by hand."
    assert sections(second)["Abstract"].count("Written by hand.") == 1


def test_figures_are_linked_when_they_exist(ctx, tmp_path):
    tdir = tmp_path / "tables"
    tdir.mkdir()
    (tdir / "fig_ablation.png").write_bytes(b"png")
    c = dict(ctx, tables_dir=str(tdir))
    md = P.render(c, {}, str(tmp_path / "paper"))
    assert "![Ablation path per configuration](../tables/fig_ablation.png)" in md and "fig_cost" not in md
    assert "![" not in draft(ctx)


def test_appendix_a_is_the_readiness_file(ctx, tmp_path):
    out = P.write_all(ctx, str(tmp_path / "t.json"), str(tmp_path / "out"))
    md = open(out["paths"]["draft"], encoding="utf-8").read()
    ready = open(out["paths"]["readiness"], encoding="utf-8").read()
    assert ready.rstrip() in md and ready.startswith("NOT READY")


def test_configurations_table_and_system_legend(ctx):
    md = draft(ctx)
    assert "| A6 | Constrained planner | `n5_r0_s1+cg1` | A2 | repair, constrained | RQ5 |" in md
    assert "| A8 | not built yet | not built | A5 | – | – |" in md
    leg = P.option_legend(ctx["cfg"])
    for tok in ("`b1`", "`b0`", "`_r1`", "`+ts`", "`+h1`", "`+cg1`", "`+fz`", "`_s3`"):
        assert tok in leg
    assert "`+cas`" not in leg and "`+fs`" not in leg                                              # not in this grid


def test_llm_table_has_the_prices(ctx):
    t = P.llm_table(ctx["cfg"])
    assert "| a | Model A | `qwen/qwen3.8-27b` | 0.8 / 4 |" in t and "| c | Model C | `openai/gpt-oss-20b` | 0.075 / 0.3 |" in t


def test_data_facts(ctx):
    facts = "\n".join(P.data_facts(ctx))
    assert f"{N_PARTS} parts" in facts and "`dev_v2` (development set)" in facts and f"{REPEATS} repeats per cell" in facts and "200 resamples, seed 7" in facts


# ---------------------------------------------------------------- readiness
def status(checks, cid):
    return next(c for c in checks if c["id"] == cid)


def test_a_development_grid_with_all_the_prose_is_blocked_only_by_the_dataset(ctx):
    checks = P.readiness(ctx, full_text())
    open_blocking = [c["id"] for c in checks if c["level"] == "blocking" and not c["ok"]]
    assert open_blocking == ["sealed"]
    assert "development results, not paper results" in status(checks, "sealed")["message"]
    md = draft(ctx, full_text())
    assert "**DEVELOPMENT RESULTS, NOT PAPER RESULTS.**" in md and "1 blocking item(s)" in md


def test_missing_prose_and_leftovers_block(ctx):
    checks = P.readiness(ctx, {})
    assert status(checks, "text_missing")["ok"] is False and "title, authors, abstract" in status(checks, "text_missing")["message"]
    text = full_text(discussion="This is a TODO: fill in", related_work="see [TO CONFIRM] later", method=" _TO WRITE: x")
    c = status(P.readiness(ctx, text), "text_placeholders")
    assert c["ok"] is False and c["level"] == "blocking" and "discussion" in c["message"] and "related_work" in c["message"] and "method" in c["message"]
    assert status(P.readiness(ctx, full_text()), "text_placeholders")["ok"]


def test_word_budgets_are_advisory(ctx):
    c = status(P.readiness(ctx, full_text(abstract="too short")), "text_budget")
    assert c["level"] == "advisory" and c["ok"] is False and "abstract (2 words, 150-250)" in c["message"]
    assert status(P.readiness(ctx, full_text()), "text_budget")["ok"]
    assert status(P.readiness(ctx, {k: "w " * 600 for k in P.TEXT_FIELDS}), "text_total")["ok"] is False


def test_fewer_than_three_llms_block(ctx):
    c = copy.deepcopy(ctx)
    c["cfg"]["models"] = c["cfg"]["models"][:2]
    assert status(P.readiness(c, full_text()), "llms")["ok"] is False


def test_incomplete_runs_block_and_are_named(grid, tmp_path):
    out, path = grid
    import shutil
    dst = str(tmp_path / "res")
    shutil.copytree(out, dst)
    shutil.rmtree(os.path.join(dst, "p_b_b0_r1"))
    cfg = json.load(open(path))
    cfg["results_dir"] = dst
    p2 = str(tmp_path / "g.json")
    json.dump(cfg, open(p2, "w"))
    c = P.collect(p2, root=dst)
    chk = status(P.readiness(c, full_text()), "runs")
    assert chk["ok"] is False and "A2/b=missing" in chk["message"]
    md = draft(c)
    assert any("A2/b" in x for x in P.limitations(c))
    assert "Model B" in md                                                                       # the draft is still written


def test_the_unbuilt_row_is_an_advisory_item(ctx):
    c = status(P.readiness(ctx, full_text()), "unbuilt")
    assert c["level"] == "advisory" and "A8" in c["message"]


def test_case_studies_are_read_and_checked(grid, tmp_path):
    out, path = grid
    cases = {"types": ["order", "missing"], "per_type": 3, "n_without_commentary": 2, "orphan_commentary": [],
             "cases": [{"id": "c1"}, {"id": "c2"}, {"id": "c3"}],
             "runs": [{"label": "dev_b0", "system": "b0", "dataset": "dev_v2", "n_parts": 6, "n_parsed_routes": 60,
                       "table": [{"type": "order", "routes": 6, "share": 0.1, "parts": 3, "eligible": 3, "denominator": 60, "note": ""},
                                 {"type": "missing", "routes": 0, "share": 0.0, "parts": 0, "eligible": 0, "denominator": 60, "note": ""},
                                 {"type": "differs", "routes": None, "share": None, "parts": 4, "eligible": 4, "denominator": None, "note": "x"}]}]}
    d = tmp_path / "cs"
    d.mkdir()
    json.dump(cases, open(d / "cases.json", "w"))
    c = P.collect(path, case_dir=str(d), root=out)
    t = P.m3_tables(c)
    assert "| order | 6 | 10.0% | 3 |" in t and "| missing | 0 | 0.0% | 0 |" in t and "| differs | – | – | 4 |" in t
    assert "3 case(s) selected" in t and "2 still without commentary" in t
    chk = status(P.readiness(c, full_text()), "cases")
    assert chk["level"] == "advisory" and chk["ok"] is False and "2 case(s) without commentary" in chk["message"]
    cases["n_without_commentary"] = 0
    json.dump(cases, open(d / "cases.json", "w"))
    assert status(P.readiness(P.collect(path, case_dir=str(d), root=out), full_text()), "cases")["ok"]
    assert "| order | 6 | 10.0% | 3 |" in rq_section(draft(c), 1)
    assert "Appendix B" in draft(c) and f"--case-studies {d}" in draft(c)


def test_unreadable_inputs_are_named_not_fatal(grid, tmp_path):
    out, path = grid
    d = tmp_path / "cs"
    d.mkdir()
    (d / "cases.json").write_text("{nope")
    (tmp_path / "bench.json").write_text("{nope")
    c = P.collect(path, benchmark=str(tmp_path / "bench.json"), case_dir=str(d), extraction=[str(tmp_path / "none")],
                  freeze_path=str(tmp_path / "missing_freeze.json"), root=out)
    msgs = " | ".join(c["problems"])
    assert "cases.json could not be read" in msgs and "benchmark" in msgs and "extraction run" in msgs and "freeze manifest" in msgs
    inputs = [x for x in P.readiness(c, full_text()) if x["id"] == "input"]
    assert len(inputs) == len(c["problems"]) and all(x["level"] == "blocking" and not x["ok"] for x in inputs)
    assert draft(c)                                                                                  # still renders


def test_a_bad_grid_config_raises(tmp_path):
    p = tmp_path / "g.json"
    p.write_text('{"name": "x"}')
    with pytest.raises(T.TableError):
        P.collect(str(p))


# ---- the benchmark
def bench_parts(n=5, placeholder=False, second=3, reviewed=True):
    parts = []
    for i, p in enumerate(dataset.load_dataset("dev_v2")[:n]):
        p = copy.deepcopy(p)
        route = list(rb.generate_valid_routes(p["features"], max_routes=1)[0]["steps"])
        p["part_id"] = f"BENCH_{i:03d}"
        p["split"] = "bench"
        p["source"] = "synthetic test_v2 (placeholder)" if placeholder else "Handbook example 4.2"
        p["reference_routes"] = [route]
        p["reference_meta"] = {"annotator": "A. Expert", "review_status": "approved" if reviewed else ""}
        if i < second:
            p["second_annotator_routes"] = [route if i % 2 == 0 else route[:-2] + route[-2:][::-1]]
        parts.append(p)
    return parts


def test_benchmark_facts_and_checks(grid, tmp_path):
    out, path = grid
    bp = tmp_path / "bench.json"
    json.dump(bench_parts(), open(bp, "w"))
    c = P.collect(path, benchmark=str(bp), root=out)
    facts = "\n".join(P.data_facts(c))
    assert "Expert benchmark" in facts and "5 parts" in facts and "Handbook example 4.2" in facts
    assert "Human-human agreement (M10) on 3 parts: exact match" in facts
    chk = {x["id"]: x for x in P.readiness(c, full_text())}
    assert chk["bench_real"]["ok"] and chk["bench_size"]["ok"] is False and chk["bench_size"]["level"] == "advisory" and "5 parts" in chk["bench_size"]["message"]
    assert chk["bench_review"]["ok"] and chk["m10"]["ok"] is False and "3 parts" in chk["m10"]["message"]
    assert any("5 parts; every per-class statement on it is anecdotal" in x for x in P.limitations(c))
    assert any("3 parts: indicative" in x for x in P.limitations(c))


def test_placeholder_benchmark_parts_block(grid, tmp_path):
    out, path = grid
    bp = tmp_path / "bench.json"
    json.dump(bench_parts(placeholder=True, second=0, reviewed=False), open(bp, "w"))
    c = P.collect(path, benchmark=str(bp), root=out)
    chk = {x["id"]: x for x in P.readiness(c, full_text())}
    assert chk["bench_real"]["ok"] is False and chk["bench_real"]["level"] == "blocking" and "PLACEHOLDER" in chk["bench_real"]["message"]
    assert chk["bench_review"]["ok"] is False and "5 benchmark plan(s) without a supervisor review" in chk["bench_review"]["message"]
    assert "no second-annotator plans" in "\n".join(P.data_facts(c))


def test_a_grid_on_the_benchmark_needs_the_benchmark_file(tmp_path):
    cfg = cfg_dict(str(tmp_path), dataset_name="bench_nothing", rows=ROWS[:2])
    p = tmp_path / "g.json"
    json.dump(cfg, open(p, "w"))
    c = P.collect(str(p), root=str(tmp_path))
    chk = {x["id"]: x for x in P.readiness(c, full_text())}
    assert chk["bench_given"]["ok"] is False and chk["bench_given"]["level"] == "blocking"
    assert chk["m4"]["level"] == "blocking" and chk["sealed"]["ok"]                                  # a bench grid is on a sealed name


# ---- extraction
def write_extraction_run(path):
    from test_extraction_eval import rec          # noqa: F401  (the same hand-made records as the M9 tests)
    os.makedirs(path)
    recs = [rec("a.png", ["Hole", "Slot", "Thread"], ["Hole", "Slot"], ["Hole", "Thread", "Pocket"]), rec("b.png", ["Hole", "Boss"], ["Hole"], ["Hole"]),
            rec("c.png", ["Slot"], ["Slot", "Fillet"], ["Slot", "Fillet"])]
    with open(os.path.join(path, "records.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_m9_goes_under_rq6(grid, tmp_path):
    out, path = grid
    run = str(tmp_path / "fe1_extraction")
    write_extraction_run(run)
    c = P.collect(path, extraction=[run], root=out)
    rq6 = rq_section(draft(c), 6)
    assert "=== M9 extraction quality - fe1_extraction: 3 of 3 drawings scored ===" in rq6 and "```" in rq6
    chk = status(P.readiness(c, full_text()), "m9")
    assert chk["level"] == "advisory" and chk["ok"] is False and "3 drawings" in chk["message"]
    assert any("oracle user" in x for x in P.limitations(c))
    assert "No extraction run was given" in rq_section(draft(P.collect(path, root=out)), 6)


# ---- the sealed path: a freeze that matches
@pytest.fixture()
def sealed(tmp_path, monkeypatch):
    """A fake repository with a freeze, three LLMs on test_v2 under that freeze (the harness records the status), and a grid config."""
    repo = tmp_path / "repo"
    for rel, text in {"capp/planners.py": "x = 1\n", "capp/tables.py": "y = 1\n", "week1/route_builder.py": "rb = 1\n"}.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    (repo / "data").mkdir()
    (repo / "data" / "test_v2.json").write_text(json.dumps(dataset.load_dataset("test_v2")[:6]))
    (repo / "configs").mkdir()
    res_dir = str(repo / "results")
    cfg = cfg_dict(res_dir, dataset_name="test_v2", rows=ROWS[:3], limit=4)
    (repo / "configs" / "final.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(config, "ROOT", Path(str(repo)))
    for k in ("CAPP_MODEL_PRIMARY", "CAPP_STRUCTURED"):
        monkeypatch.delenv(k, raising=False)
    man = F.create(str(repo), grids=["configs/final.json"])
    make_runs(res_dir, dataset_name="test_v2", systems=["b1", "b0", "b0_r1"], limit=4)
    return str(repo), str(repo / "configs" / "final.json"), man, monkeypatch


def sealed_ctx(repo, cfg_path):
    return P.collect(cfg_path, root=repo)


def test_a_sealed_grid_under_a_matching_freeze_is_ready(sealed):
    repo, cfg_path, man, _ = sealed
    c = sealed_ctx(repo, cfg_path)
    checks = P.readiness(c, full_text())
    blocking = [x["message"] for x in checks if x["level"] == "blocking" and not x["ok"]]
    assert blocking == []
    assert status(checks, "freeze")["ok"] and status(checks, "freeze_runs")["ok"] and status(checks, "sealed")["ok"]
    assert [x["id"] for x in checks if x["level"] == "advisory" and not x["ok"]] == ["m4", "cases", "m9"]
    md = draft(c, full_text())
    assert f"Generated from the sealed set `test_v2` under freeze `{man['id']}`" in md and "DEVELOPMENT RESULTS" not in md
    repro = "\n".join(P.reproducibility(c))
    assert f"Freeze `{man['id']}`" in repro and f"the working tree now: **ok**" in repro and "Runs were made under freeze id(s): " + man["id"] in repro
    assert "qwen/qwen3.8-27b" in repro
    assert not any("development set" in x for x in P.limitations(c))


def test_drift_after_the_runs_blocks(sealed):
    repo, cfg_path, man, _ = sealed
    (Path(repo) / "capp" / "planners.py").write_text("x = 2\n")
    c = sealed_ctx(repo, cfg_path)
    chk = status(P.readiness(c, full_text()), "freeze")
    assert chk["ok"] is False and chk["level"] == "blocking" and "drifted" in chk["message"]
    assert "NOT READY" in draft(c, full_text())
    (Path(repo) / "capp" / "planners.py").write_text("x = 1\n")
    (Path(repo) / "capp" / "tables.py").write_text("y = 2\n")                                         # analysis code: allowed
    assert status(P.readiness(sealed_ctx(repo, cfg_path), full_text()), "freeze")["ok"]


def test_runs_made_under_another_freeze_block(sealed):
    repo, cfg_path, man, mp = sealed
    for folder in Path(repo, "results").iterdir():
        info = json.load(open(folder / "run.json"))
        info["freeze"]["id"] = "0" * 12
        json.dump(info, open(folder / "run.json", "w"))
    chk = status(P.readiness(sealed_ctx(repo, cfg_path), full_text()), "freeze_runs")
    assert chk["ok"] is False and "000000000000" in chk["message"] and man["id"] in chk["message"]


def test_a_sealed_grid_without_a_freeze_blocks(sealed):
    repo, cfg_path, man, _ = sealed
    os.remove(os.path.join(repo, "FREEZE.json"))
    chk = status(P.readiness(sealed_ctx(repo, cfg_path), full_text()), "freeze")
    assert chk["ok"] is False and "no freeze manifest" in chk["message"]


# ---------------------------------------------------------------- the command line
@pytest.fixture()
def cli():
    spec = importlib.util.spec_from_file_location("paper_cli", os.path.join(ROOT, "scripts", "paper_draft.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_writes_the_draft_and_the_text_file(cli, grid, tmp_path, capsys):
    out, path = grid
    o = tmp_path / "paper"
    assert cli.main([path, "--out", str(o)]) == 0
    text = capsys.readouterr().out
    assert text.startswith("NOT READY") and "Draft:" in text and f"{len(P.TEXT_FIELDS)} empty section(s) added" in text
    assert (o / "paper_draft.md").is_file() and (o / "readiness.txt").is_file() and (o / "paper_text.json").is_file()
    t = json.load(open(o / "paper_text.json"))
    t["abstract"] = "Mine."
    t["notes_to_self"] = "remember to thank the annotators"
    json.dump(t, open(o / "paper_text.json", "w"))
    assert cli.main([path, "--out", str(o)]) == 0
    text = capsys.readouterr().out
    assert "empty section(s) added" not in text and "match no section (notes_to_self)" in text
    assert "Mine." in open(o / "paper_draft.md", encoding="utf-8").read()
    assert json.load(open(o / "paper_text.json"))["notes_to_self"].startswith("remember")


def test_cli_strict_and_errors(cli, grid, tmp_path, capsys):
    out, path = grid
    assert cli.main([path, "--out", str(tmp_path / "p"), "--strict"]) == 1
    assert "blocking item(s) open" in capsys.readouterr().err
    assert (tmp_path / "p" / "paper_draft.md").is_file()                                         # still written
    assert cli.main([str(tmp_path / "none.json"), "--out", str(tmp_path / "q")]) == 2
    assert "STOPPED" in capsys.readouterr().err
    bad = tmp_path / "bad_text.json"
    bad.write_text("{nope")
    assert cli.main([path, "--out", str(tmp_path / "r"), "--text", str(bad)]) == 2
    assert "not valid JSON" in capsys.readouterr().err
    assert cli.main([path, "--out", str(tmp_path / "s"), "--text", str(tmp_path / "elsewhere" / "my_text.json")]) == 0
    assert (tmp_path / "elsewhere" / "my_text.json").is_file() and not (tmp_path / "s" / "paper_text.json").exists()


def test_cli_strict_passes_when_ready(cli, sealed, tmp_path, capsys):
    repo, cfg_path, man, _ = sealed
    text = tmp_path / "text.json"
    json.dump(full_text(), open(text, "w"))
    code = cli.main([cfg_path, "--out", str(tmp_path / "p"), "--text", str(text), "--strict", "--freeze", os.path.join(repo, "FREEZE.json")])
    out = capsys.readouterr().out
    assert code == 0 and out.startswith("READY")
