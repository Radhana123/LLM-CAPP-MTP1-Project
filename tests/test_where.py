"""Step 41: where the project stands and the command to run next (capp/where.py, scripts/where.py).

Offline. Every case builds a small repository folder in tmp_path (results/, configs/, .env) and reads what the tool says: which line is the first
TODO, which commands come under NEXT, that it never writes anything, never stops at a file it cannot read, and that a line fits a screenshot.
"""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from capp import config, dataset, diagnostics, harness, tables, verify as V, where as W      # noqa: E402
import where as where_script                                                                # noqa: E402
import route_builder as rb                                                                   # noqa: E402

QWEN = "qwen/qwen3.8-27b"
N_DEV = len(dataset.load_dataset("dev_v2"))
OK_CAND = {"sample_id": 0, "llm_ok": True, "llm_raw": "[]"}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("GROQ_API_KEY", "CAPP_QWEN_REASONING", "CAPP_MODEL_PRIMARY"):
        monkeypatch.delenv(name, raising=False)


def llm_info(truncated=0, params="now", model=QWEN, calls=100):
    info = {"model": model, "llm_client_summary": {"calls": calls, "truncated": truncated}}
    if params == "now":
        info["model_params"] = {model: config.model_params(model)}
    elif params is not None:
        info["model_params"] = {model: params}
    return info


def make_run(root, system, n, tag="dev", info=None, api=0):
    folder = root / "results" / f"{tag}_{system}"
    folder.mkdir(parents=True, exist_ok=True)
    if info is not None:
        (folder / "run.json").write_text(json.dumps(info), encoding="utf-8")
    api_cand = {"sample_id": 0, "llm_ok": False, "llm_raw": "API Error: HTTP 503: x [after 6 attempts]"}
    lines = [json.dumps({"part_id": f"P{i}", "repeat": 1, "candidates": [api_cand if i < api else OK_CAND]}) for i in range(n)]
    (folder / "records.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def g1_done(root):
    make_run(root, "rb", N_DEV)
    make_run(root, "b1", 20, info=llm_info())
    make_run(root, "b0", 20, info=llm_info())


def key(root):
    (root / ".env").write_text("GROQ_API_KEY=abc\n", encoding="utf-8")


def rows_of(root, **kw):
    return {r.label: r for r in W.build_rows(str(root), **kw)}


def first_todo(root, **kw):
    return next((r.label for r in W.build_rows(str(root), **kw) if r.state == "todo" and r.commands), None)


def nxt(root, **kw):
    return W.next_commands(W.build_rows(str(root), **kw))


def snapshot(root):
    return sorted(os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs)


def planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
    routes = rb.generate_valid_routes(features, max_routes=15)
    return {"success": True, "steps": V.clean_steps(routes[sample_id % len(routes)]["steps"]), "raw_response": "", "model": "fake",
            "meta": {"cached": True, "finish_reason": "stop", "prompt_tokens": 100, "completion_tokens": 20, "latency_s": 0.1}}


def small_grid(root, name="grid.json", dataset_name="dev_v2", **over):
    cfg = {"name": "wt", "dataset": dataset_name, "limit": 3, "repeats": 1, "results_dir": "results", "resamples": 100, "seed": 7,
           "models": [{"id": "a", "label": "Model A", "model": QWEN, "tag": "g_a"}],
           "rows": [{"id": "A0", "label": "B1", "system": "b1"}, {"id": "A1", "label": "B0", "system": "b0", "parent": "A0"}]}
    cfg.update(over)
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs" / name).write_text(json.dumps(cfg), encoding="utf-8")
    return "configs/" + name


def run_small_grid(root):
    parts = dataset.load_dataset("dev_v2")[:3]
    mp = pytest.MonkeyPatch()
    try:
        mp.setenv("CAPP_MODEL_PRIMARY", QWEN)
        for system in ("b1", "b0"):
            harness.run_experiment(system, parts, repeats=1, tag="g_a", out_dir=str(root / "results"), dataset_name="dev_v2", progress=None,
                                   planner=planner, analysis=False)
    finally:
        mp.undo()


# ================================================================ the first line that says TODO is where you are
def test_a_new_folder_wants_the_key_first(tmp_path):
    assert first_todo(tmp_path) == "key"
    assert "GROQ_API_KEY" in nxt(tmp_path)[0] and ".env" in nxt(tmp_path)[0]
    assert rows_of(tmp_path)["key"].state == "todo"


def test_with_the_key_the_smoke_test_comes_before_any_long_run(tmp_path):
    key(tmp_path)
    assert first_todo(tmp_path) == "smoke"
    assert nxt(tmp_path)[0].startswith("python scripts\\smoke_llm.py") and "RESULT: OK" in nxt(tmp_path)[0]
    assert rows_of(tmp_path)["key"].state == "ok" and rows_of(tmp_path)["key"].text == "GROQ_API_KEY found (.env)"


def test_the_key_in_the_environment_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "abc")
    assert rows_of(tmp_path)["key"].state == "ok" and rows_of(tmp_path)["key"].text == "GROQ_API_KEY found (environment variable)"


def test_a_run_made_with_todays_settings_is_the_evidence_so_the_smoke_test_goes_away_and_g1_is_next(tmp_path):
    key(tmp_path)
    make_run(tmp_path, "rb", N_DEV)
    make_run(tmp_path, "b1", 20, info=llm_info())
    rows = rows_of(tmp_path)
    assert "smoke" not in rows and rows["LLM runs"].text.startswith("1 in results\\: 1 made with today's settings and not cut off")
    assert rows["G1 rb"].state == "ok" and rows["G1 b1"].state == "ok" and rows["G1 b0"].text == "not made yet"
    assert first_todo(tmp_path) == "G1 b0" and nxt(tmp_path)[0].startswith("python scripts\\g1.py")
    assert W.report(str(tmp_path)).endswith("  (then  python scripts\\where.py  again)")


def test_the_key_is_not_asked_for_when_no_long_run_is_left(tmp_path, monkeypatch):
    g1_done(tmp_path)
    (tmp_path / "results" / "diagnostics.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(W, "g2_rows", lambda root, grid, g1_done: ([W.Row("G2 grid", "all 4 cells ok", "ok", ())], True))
    assert "key" not in rows_of(tmp_path)


def test_cut_off_and_old_runs_are_counted_and_g1_says_why_it_will_run_them_again(tmp_path):
    key(tmp_path)
    make_run(tmp_path, "rb", N_DEV)
    make_run(tmp_path, "b1", 20, info=llm_info(truncated=4))
    make_run(tmp_path, "b0", 20, info=llm_info(params=None))
    rows = rows_of(tmp_path)
    assert rows["LLM runs"].state == "warn" and "2 in results" in rows["LLM runs"].text
    assert "1 with answers cut off" in rows["LLM runs"].text and "1 with old model settings" in rows["LLM runs"].text
    assert rows["G1 b1"].text.startswith("to run again: 4 answer(s)") and rows["G1 b0"].text.startswith("to run again: it was made with other settings")
    assert "smoke" in rows                                                             # none of the two is evidence that today's settings work


def test_a_run_with_api_errors_or_too_few_records_is_to_run_again(tmp_path):
    key(tmp_path)
    make_run(tmp_path, "rb", N_DEV)
    make_run(tmp_path, "b1", 20, info=llm_info(), api=3)
    make_run(tmp_path, "b0", 7, info=llm_info())
    rows = rows_of(tmp_path)
    assert "API error(s)" in rows["G1 b1"].text and "7 record(s), 20 wanted" in rows["G1 b0"].text


def test_when_g1_is_done_the_diagnostics_file_is_next_and_then_the_grid(tmp_path):
    key(tmp_path)
    g1_done(tmp_path)
    assert first_todo(tmp_path) == "G1 diag" and nxt(tmp_path)[0] == "python scripts\\diagnostics.py"
    (tmp_path / "results" / "diagnostics.txt").write_text("x", encoding="utf-8")
    cmds = nxt(tmp_path, grid=small_grid(tmp_path))
    assert first_todo(tmp_path, grid=small_grid(tmp_path)) == "G2 grid"
    assert cmds[0].startswith("python scripts\\g2.py configs\\grid.json --dry-run")
    assert cmds[1].startswith("python scripts\\g2.py configs\\grid.json --only a ") and "--max-calls" not in cmds[1]


def test_the_grid_line_counts_cells_and_runs_and_waits_for_g1(tmp_path):
    key(tmp_path)
    grid = small_grid(tmp_path)
    r = rows_of(tmp_path, grid=grid)["G2 grid"]
    assert r.text == "0 of 2 cells ok, 2 run(s) to make or run again (configs\\grid.json)" and r.state == "info" and r.commands == ()


def test_a_complete_grid_asks_for_its_table_then_for_the_final_config(tmp_path, monkeypatch):
    key(tmp_path)
    g1_done(tmp_path)
    (tmp_path / "results" / "diagnostics.txt").write_text("x", encoding="utf-8")
    grid = small_grid(tmp_path)
    run_small_grid(tmp_path)
    r = rows_of(tmp_path, grid=grid)
    assert r["G2 grid"].state == "ok" and r["G2 grid"].text == "all 2 cells ok (configs\\grid.json)"
    assert first_todo(tmp_path, grid=grid) == "G2 table" and nxt(tmp_path, grid=grid)[0] == "python scripts\\make_tables.py build configs\\grid.json"
    table = tmp_path / "results" / "tables" / "wt" / "table4.md"
    table.parent.mkdir(parents=True)
    table.write_text("x", encoding="utf-8")
    assert first_todo(tmp_path, grid=grid) == "freeze"
    assert nxt(tmp_path, grid=grid)[0].startswith("copy configs\\ablation_template.json configs\\final.json")


def test_with_a_final_config_the_freeze_is_next(tmp_path):
    key(tmp_path)
    g1_done(tmp_path)
    (tmp_path / "results" / "diagnostics.txt").write_text("x", encoding="utf-8")
    grid = small_grid(tmp_path)
    run_small_grid(tmp_path)
    table = tmp_path / "results" / "tables" / "wt" / "table4.md"
    table.parent.mkdir(parents=True)
    table.write_text("x", encoding="utf-8")
    final = small_grid(tmp_path, name="final.json", dataset_name="test_v2", limit=None, repeats=5)
    cmds = nxt(tmp_path, grid=grid, final=final)
    assert cmds[0] == 'python scripts\\freeze.py create --grid configs\\final.json --run-tests --note "Tier 1"'
    assert cmds[1].startswith("python scripts\\freeze.py verify") and "git tag freeze-v1" in cmds[1]


def test_after_the_freeze_the_sealed_runs_are_made_from_the_commands_of_the_final_grid(tmp_path, monkeypatch):
    key(tmp_path)
    g1_done(tmp_path)
    (tmp_path / "results" / "diagnostics.txt").write_text("x", encoding="utf-8")
    grid = small_grid(tmp_path)
    run_small_grid(tmp_path)
    table = tmp_path / "results" / "tables" / "wt" / "table4.md"
    table.parent.mkdir(parents=True)
    table.write_text("x", encoding="utf-8")
    final = small_grid(tmp_path, name="final.json", dataset_name="test_v2", limit=None, repeats=5)
    monkeypatch.setattr(W.freeze, "status", lambda root, *a, **k: {"status": "ok", "id": "f1"})
    r = rows_of(tmp_path, grid=grid, final=final)
    assert r["freeze"].state == "ok" and "freeze f1" in r["freeze"].text
    assert r["G3/G4"].state == "todo" and r["G3/G4"].text.startswith("0 of 2 cells ok, 2 run(s) to make (configs\\final.json)")
    assert nxt(tmp_path, grid=grid, final=final) == ["python scripts\\make_tables.py commands configs\\final.json",
                                                     "python scripts\\make_tables.py status configs\\final.json"]


def test_a_freeze_file_before_g2_is_complete_is_a_warning_with_the_way_out(tmp_path, monkeypatch):
    key(tmp_path)
    monkeypatch.setattr(W.freeze, "status", lambda root, *a, **k: {"status": "ok", "id": "f1"})
    r = rows_of(tmp_path)["freeze"]
    assert r.state == "warn" and "BEFORE G2 is complete" in r.text and "git rm FREEZE.json" in r.text
    assert first_todo(tmp_path) == "smoke"                                              # a warning does not take the place of the next step


def test_a_drifted_freeze_after_g2_is_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "g2_rows", lambda root, grid, g1_done: ([W.Row("G2 grid", "all 4 cells ok", "ok", ())], True))
    monkeypatch.setattr(W.freeze, "status", lambda root, *a, **k: {"status": "drift", "id": "f1"})
    r = rows_of(tmp_path)["freeze"]
    assert r.state == "warn" and "drift" in r.text and "freeze.py verify" in r.text


def test_the_release_comes_after_the_final_grid(tmp_path, monkeypatch):
    assert rows_of(tmp_path)["release"].text == "not started (after G3/G4)"
    assert W.release_row(str(tmp_path), True).commands == ("copy configs\\release_template.json configs\\release_final.json",
                                                            "python scripts\\release.py check configs\\release_final.json")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "release_final.json").write_text("{}", encoding="utf-8")
    assert W.release_row(str(tmp_path), True).commands == ("python scripts\\release.py check configs\\release_final.json",)
    (tmp_path / "release" / "v1").mkdir(parents=True)
    r = W.release_row(str(tmp_path), True)
    assert r.state == "ok" and "release.py verify release\\v1" in r.text


# ================================================================ it never stops and never writes
def test_a_wrong_qwen_setting_is_the_first_thing_to_fix_and_nothing_crashes(tmp_path, monkeypatch):
    key(tmp_path)
    make_run(tmp_path, "b1", 20, info=llm_info())
    monkeypatch.setenv("CAPP_QWEN_REASONING", "xhigh")
    rows = rows_of(tmp_path)
    assert rows["setting"].state == "todo" and "CAPP_QWEN_REASONING='xhigh'" in rows["setting"].text
    assert first_todo(tmp_path) == "setting" and "Remove-Item Env:CAPP_QWEN_REASONING" in nxt(tmp_path)[0]
    assert "G1 b1" not in rows and rows["G1"].state == "warn"                           # g1 cannot judge a run either: said, not raised


def test_an_unreadable_grid_config_is_a_warning_line_not_a_crash(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "bad.json").write_text("{not json", encoding="utf-8")
    r = rows_of(tmp_path, grid="configs/bad.json")["G2 grid"]
    assert r.state == "warn" and "cannot be read (configs\\bad.json)" in r.text
    r = rows_of(tmp_path, grid="configs/missing.json")["G2 grid"]
    assert r.state == "warn" and "cannot be read (configs\\missing.json)" in r.text
    rows = rows_of(tmp_path, final="configs/bad.json")
    assert rows["G3/G4"].state == "warn"


def test_a_dataset_that_cannot_be_loaded_is_said_in_one_line(tmp_path):
    rows = rows_of(tmp_path, dataset="nope_v9")
    assert rows["G1"].state == "warn" and "cannot be worked out" in rows["G1"].text


def test_run_json_files_that_are_broken_are_skipped(tmp_path):
    for name, text in (("a", "{not json"), ("b", "[]"), ("c", '{"model": 5}')):
        (tmp_path / "results" / f"x_{name}").mkdir(parents=True)
        (tmp_path / "results" / f"x_{name}" / "run.json").write_text(text, encoding="utf-8")
    assert W.scan_runs(str(tmp_path)) == {"llm": 0, "usable": 0, "cut": 0, "old": 0}


def test_a_run_with_no_calls_is_not_evidence(tmp_path):
    make_run(tmp_path, "b1", 20, info=llm_info(calls=0))
    c = W.scan_runs(str(tmp_path))
    assert c == {"llm": 1, "usable": 0, "cut": 0, "old": 0}


def test_cascade_runs_count_for_every_model_they_used(tmp_path):
    info = {"model": "openai/gpt-oss-20b", "cascade_models": {"small": "openai/gpt-oss-20b", "large": "openai/gpt-oss-120b"},
            "llm_client_summary": {"calls": 9, "truncated": 0},
            "model_params": {"openai/gpt-oss-20b": config.model_params("openai/gpt-oss-20b")}}
    make_run(tmp_path, "cas", 20, info=info)
    assert W.scan_runs(str(tmp_path))["old"] == 1                                       # the large tier has no recorded settings


def test_it_writes_nothing_and_makes_no_folder(tmp_path):
    key(tmp_path)
    g1_done(tmp_path)
    grid = small_grid(tmp_path)
    before = snapshot(tmp_path)
    W.report(str(tmp_path), grid=grid)
    assert snapshot(tmp_path) == before
    empty = tmp_path / "empty"
    empty.mkdir()
    W.report(str(empty))
    assert snapshot(empty) == []


# ================================================================ the words
def test_every_line_fits_a_screenshot_and_is_plain_ascii(tmp_path):
    key(tmp_path)
    make_run(tmp_path, "b1", 20, info=llm_info(truncated=4))
    text = W.report(str(tmp_path), grid="configs/" + "x" * 150 + ".json")
    assert all(len(line) <= W.WIDTH for line in text.split("\n") if not line.startswith("  python") and not line.startswith("  copy"))
    assert text.isascii()
    assert any(line.endswith("...") for line in text.split("\n"))                       # the long grid error line is cut, not wrapped


def test_when_nothing_is_left_it_says_what_is_the_authors(tmp_path):
    rows = [W.Row("G1 rb", "done", "ok", ()), W.Row("release", "release\\v1 is there", "ok", ())]
    text = W.format_rows(rows, [], str(tmp_path))
    assert "nothing left that this tool can see" in text and "case-study commentary" in text and "Zenodo" in text


def test_a_warning_without_a_todo_says_to_read_it(tmp_path):
    text = W.format_rows([W.Row("freeze", "drifted", "warn", ())], [], str(tmp_path))
    assert "WARN freeze" in text and "read the WARN line(s) above" in text


def test_a_todo_without_a_command_is_not_the_next_step(tmp_path):
    rows = [W.Row("a", "x", "todo", ()), W.Row("b", "y", "todo", ("python scripts\\b.py",)), W.Row("c", "z", "todo", ("python scripts\\c.py",))]
    assert W.next_commands(rows) == ["python scripts\\b.py"]
    assert W.next_commands([W.Row("a", "x", "ok", ("python x",)), W.Row("b", "y", "info", ("python y",))]) == []


def test_the_labels_and_states_are_in_the_text(tmp_path):
    key(tmp_path)
    text = W.report(str(tmp_path), grid=small_grid(tmp_path))
    assert text.startswith("MTP-1: where you are") and "offline, nothing is run" in text
    for line in ("  ok   key", "  TODO smoke", "  TODO G1 rb", "       G2 grid", "  ok   freeze", "       release", "NEXT"):
        assert line in text


# ================================================================ scripts/where.py
def test_the_script_prints_the_report_and_exits_0(tmp_path, capsys):
    key(tmp_path)
    assert where_script.main(["--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "MTP-1: where you are" in out and "python scripts\\smoke_llm.py" in out


def test_the_script_hands_its_options_to_the_report(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(where_script.where, "report", lambda root, **kw: seen.update(root=root, **kw) or "x")
    assert where_script.main(["--tag", "t1", "--dataset", "d1", "--grid", "g.json", "--final", "f.json", "--root", str(tmp_path)]) == 0
    assert seen == {"root": str(tmp_path), "tag": "t1", "dataset": "d1", "grid": "g.json", "final": "f.json"}
    seen.clear()
    where_script.main([])
    assert seen == {"root": where_script.ROOT, "tag": diagnostics.DEFAULT_TAG, "dataset": diagnostics.DEFAULT_DATASET,
                    "grid": W.GRID, "final": W.FINAL}


def test_a_wrong_argument_exits_2(capsys):
    with pytest.raises(SystemExit) as e:
        where_script.main(["--nope"])
    assert e.value.code == 2


# ================================================================ details found by mutating the code (Step 41)
def test_a_folder_without_any_llm_run_says_so(tmp_path):
    r = rows_of(tmp_path)["LLM runs"]
    assert r.text == "none yet in results\\ (no run has called a model)" and r.state == "info"


def test_one_kind_of_trouble_is_enough_for_a_warning(tmp_path):
    make_run(tmp_path, "b1", 20, info=llm_info(truncated=2))
    r = rows_of(tmp_path)["LLM runs"]
    assert r.state == "warn" and r.text.endswith("1 with answers cut off at the token limit") and "old model settings" not in r.text
    other = tmp_path / "other"
    make_run(other, "b1", 20, info=llm_info(params=None))
    r = rows_of(other)["LLM runs"]
    assert r.state == "warn" and r.text.endswith("1 with old model settings (Steps 35/37)") and "with answers cut off" not in r.text
    clean = tmp_path / "clean"
    make_run(clean, "b1", 20, info=llm_info())
    assert rows_of(clean)["LLM runs"].state == "info"


def test_a_run_json_whose_summary_is_not_a_dict_is_counted_but_is_no_evidence(tmp_path):
    for name, summary in (("a", "x"), ("b", ["calls"]), ("c", None)):
        folder = tmp_path / "results" / f"x_{name}"
        folder.mkdir(parents=True)
        info = llm_info()
        info["llm_client_summary"] = summary
        (folder / "run.json").write_text(json.dumps(info), encoding="utf-8")
    assert W.scan_runs(str(tmp_path)) == {"llm": 3, "usable": 0, "cut": 0, "old": 0}


def test_no_smoke_test_and_no_key_when_no_long_run_is_left_even_without_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "g1_rows", lambda root, tag, dataset: ([W.Row("G1 rb", "done", "ok", ())], []))
    monkeypatch.setattr(W, "g2_rows", lambda root, grid, g1_done: ([W.Row("G2 grid", "all 4 cells ok", "ok", ())], True))
    rows = rows_of(tmp_path)
    assert "smoke" not in rows and "key" not in rows


def test_the_diagnostics_line_is_only_a_todo_once_g1_is_done(tmp_path):
    key(tmp_path)
    make_run(tmp_path, "rb", N_DEV)
    assert rows_of(tmp_path)["G1 diag"].state == "info" and rows_of(tmp_path)["G1 diag"].commands == ()
    g1_done(tmp_path)
    assert rows_of(tmp_path)["G1 diag"].state == "todo"


def test_a_row_that_is_not_built_is_not_a_cell(tmp_path):
    grid = small_grid(tmp_path, rows=[{"id": "A0", "label": "B1", "system": "b1"}, {"id": "A1", "label": "B0", "system": "b0", "parent": "A0"},
                                      {"id": "A2", "label": "later", "system": None, "parent": "A1"}])
    assert rows_of(tmp_path, grid=grid)["G2 grid"].text.startswith("0 of 2 cells ok, 2 run(s)")
    run_small_grid(tmp_path)
    assert rows_of(tmp_path, grid=grid)["G2 grid"].text.startswith("all 2 cells ok")


def test_the_sealed_runs_wait_for_the_freeze(tmp_path):
    final = small_grid(tmp_path, name="final.json", dataset_name="test_v2", limit=None, repeats=5)
    r = rows_of(tmp_path, final=final)["G3/G4"]
    assert r.state == "info" and r.commands == () and r.text.startswith("0 of 2 cells ok, 2 run(s) to make (configs\\final.json)")


def test_a_complete_final_grid_is_ok_and_opens_the_release(tmp_path, monkeypatch):
    final = small_grid(tmp_path, name="final.json", dataset_name="test_v2", limit=None, repeats=5)
    monkeypatch.setattr(W, "grid_counts", lambda cfg: {"cells": 2, "ok": 2, "pending": 0, "first_column": None, "columns": []})
    rows = rows_of(tmp_path, final=final)
    assert rows["G3/G4"].state == "ok" and rows["G3/G4"].text == "all 2 cells ok (configs\\final.json)"
    assert rows["release"].state == "todo" and rows["release"].text == "not built yet"


def test_the_key_stays_on_the_list_while_g1_is_unfinished_even_if_the_grid_is_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "g2_rows", lambda root, grid, g1_done: ([W.Row("G2 grid", "all 4 cells ok", "ok", ())], True))
    key(tmp_path)
    make_run(tmp_path, "rb", N_DEV)
    assert "key" in rows_of(tmp_path) and rows_of(tmp_path)["G1 b1"].state == "todo"


def test_when_g1_cannot_be_judged_the_grid_waits_and_the_key_is_asked_for(tmp_path):
    rows = rows_of(tmp_path, dataset="nope_v9", grid=small_grid(tmp_path))
    assert "key" in rows and rows["G2 grid"].state == "info" and rows["G2 grid"].commands == ()


def test_the_gates_are_listed_in_the_order_of_the_runbook(tmp_path):
    key(tmp_path)
    labels = [r.label for r in W.build_rows(str(tmp_path), grid=small_grid(tmp_path))]
    order = ["key", "LLM runs", "smoke", "G1 rb", "G1 b1", "G1 b0", "G1 diag", "G2 grid", "freeze", "G3/G4", "release"]
    assert labels == order
