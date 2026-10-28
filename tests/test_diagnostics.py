"""Step 31: one diagnostics file instead of ten pasted outputs (capp/diagnostics.py, scripts/diagnostics.py).

Offline. The pieces are tested on a small fake repository with a fake command runner (what was asked to run, and what came back); the real
runner is tested with real python processes, and one test runs the whole command line against the real repository. What they pin down:
the checks are a fixed list of offline scripts (never run_eval, never smoke_llm), one whose input is missing is skipped with the reason,
nothing but the one output file is written, a line that looks like a secret never reaches the file, the same inputs give the same file,
and the commands it recommends are the ones of RUNBOOK.md.
"""
import datetime
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capp import dataset as D, diagnostics as G, freeze as F      # noqa: E402

NOW = datetime.datetime(2026, 10, 28, 3, 0, tzinfo=datetime.timezone.utc)
KEY = "zz_notarealkey_0123456789abcdef"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("GROQ_API_KEY", "CAPP_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------- a small fake repository
def make_run(root, folder, system="b0", n=3, dataset="dev_v2", records=None, summary=True, strict=0.95, parse=1.0, llm=None, freeze="none", model="m1", mode="live",
             n_records=None):
    d = Path(root) / "results" / folder
    d.mkdir(parents=True, exist_ok=True)
    run = {"system": system, "dataset": dataset, "n_parts": n, "repeats": 1, "model": model, "llm_mode": mode, "freeze": {"status": freeze}}
    if llm is not False:
        run["llm_client_summary"] = llm if llm is not None else {"calls": 100, "cache_hits": 4, "prompt_tokens": 40000, "completion_tokens": 12000, "truncated": 0}
    (d / "run.json").write_text(json.dumps(run))
    nrec = n if records is None else records
    (d / "records.jsonl").write_text("".join(json.dumps({"part_id": f"P{i}"}) + "\n" for i in range(nrec)))
    if summary:
        (d / "summary.json").write_text(json.dumps({"n_records": nrec if n_records is None else n_records, "M1": {"parse_rate": parse}, "M2": {"final_strict_valid": strict}}))
    return d


def full_root(tmp_path, freeze_file=False):
    """A repository with the dataset, the grid config and the four baseline runs."""
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "data" / "dev_v2.json").write_text("[]")
    (root / "configs" / "ablation_template.json").write_text("{}")
    for s in G.RUN_SYSTEMS:
        make_run(root, f"dev_{s}", system=s)
    if freeze_file:
        (root / "FREEZE.json").write_text("{}")
    return root


class Fake:
    """A command runner that records what it was asked and answers by script name."""

    def __init__(self, answers=None, default=(0, "fine\n")):
        self.calls, self.answers, self.default = [], answers or {}, default

    def __call__(self, argv, cwd, timeout):
        self.calls.append((list(argv), cwd, timeout))
        return self.answers.get(os.path.basename(argv[1])[:-3], self.default)

    def scripts(self):
        return [os.path.basename(a[1])[:-3] for a, _, _ in self.calls]


def report(root, runner=None, **kw):
    kw.setdefault("now", NOW)
    return G.build_report(str(root), runner=runner or Fake(), **kw)


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@example.org", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)


def section(text, title):
    """The lines of one '======== title ====' section."""
    out, on = [], False
    for line in text.split("\n"):
        if line.startswith("========"):
            on = line.startswith(f"======== {title} ")
            continue
        if on:
            out.append(line)
    return out


# ---------------------------------------------------------------- small helpers
def test_clip_keeps_a_short_text_and_cuts_a_long_one_in_the_middle_saying_how_much():
    assert G._clip("a\nb\nc\n", 10) == "a\nb\nc"
    text = "\n".join(f"line {i}" for i in range(1000))
    out = G._clip(text, 100).split("\n")
    assert out[:75] == [f"line {i}" for i in range(75)] and out[-25:] == [f"line {i}" for i in range(975, 1000)]
    assert len(out) == 101 and "900 line(s) left out" in out[75] and "first 75 and the last 25" in out[75]
    assert len(G._clip(text, 100).split("\n")) == 101 and G._clip("x\n" * 100, 100) == "\n".join(["x"] * 100)       # exactly at the limit: untouched


def test_decode_makes_every_line_end_a_line_feed_and_survives_bytes_that_are_not_utf8():
    assert G._decode(b"a\r\nb\rc\n") == "a\nb\nc\n"
    assert "�" in G._decode(b"ok \xff\xfe")


def test_show_prints_the_command_the_way_it_is_typed_in_powershell():
    assert G._show(["/usr/bin/python3", "scripts/part_diff.py", "results/dev_rb", "results/dev_b0"]) == "python scripts\\part_diff.py results\\dev_rb results\\dev_b0"
    assert G._show(["x", "scripts/make_tables.py", "status", "configs/ablation_template.json"]) == "python scripts\\make_tables.py status configs\\ablation_template.json"
    assert G._show(["x", "scripts/physics_check.py", "--dataset", "dev_v2", "a/b"]) == "python scripts\\physics_check.py --dataset dev_v2 a/b"     # only our own folders


def test_pct_and_cell_show_a_dash_for_what_is_not_there_but_not_for_zero():
    assert G._pct(None) == "-" and G._pct("x") == "-" and G._pct(0.0) == "0.0%" and G._pct(0.9567) == "95.7%" and G._pct(1) == "100.0%"
    assert G._cell(None) == "-" and G._cell("") == "-" and G._cell(0) == "0" and G._cell("ab", 4) == "ab  "


def test_table_aligns_the_columns_and_copes_with_no_rows():
    t = G._table([["a", "1"], ["longer", "22"]], ["name", "n"])
    assert t == ["name    n", "------  --", "a       1", "longer  22"]
    assert G._table([], ["x", "yy"]) == ["x  yy", "-  --"]


# ---------------------------------------------------------------- running a command
def test_run_command_returns_the_exit_code_and_both_streams_with_stderr_marked():
    code, text = G.run_command([sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"], ROOT, 60)
    assert code == 3 and text.startswith("out\n") and text.endswith("[stderr]\nerr")
    assert G.run_command([sys.executable, "-c", "print('só ünï')"], ROOT, 60) == (0, "só ünï\n")                # utf-8 whatever the console code page is


def test_run_command_never_waits_for_a_keystroke():
    code, text = G.run_command([sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"], ROOT, 60)
    assert (code, text) == (0, "''\n")


def test_run_command_closes_standard_input_and_asks_for_utf8_whatever_the_console_is(monkeypatch):
    seen = {}

    def fake_run(argv, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")
    monkeypatch.setattr(G.subprocess, "run", fake_run)
    G.run_command(["x"], "somewhere", 12)
    assert seen["stdin"] is subprocess.DEVNULL and seen["cwd"] == "somewhere" and seen["timeout"] == 12 and seen["capture_output"] is True
    assert seen["env"]["PYTHONUTF8"] == "1" and seen["env"]["PYTHONIOENCODING"] == "utf-8"


def test_run_command_stops_at_the_time_limit_and_says_so():
    code, text = G.run_command([sys.executable, "-c", "import time; time.sleep(60)"], ROOT, 1)
    assert code is None and "[stopped after 1 s]" in text


def test_run_command_reports_a_program_that_cannot_start():
    code, text = G.run_command(["no-such-program-for-the-diagnostics-test"], ROOT, 5)
    assert code == 127 and text.startswith("[could not start:")


# ---------------------------------------------------------------- environment
def test_environment_outside_a_git_repository(tmp_path):
    env = G.environment(str(tmp_path))
    assert any("not a git repository" in l for l in env["lines"]) and env["notes"] == []


def test_environment_in_a_git_repository_counts_the_uncommitted_files_as_a_note_not_a_problem(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("x")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "first")
    clean = G.environment(str(tmp_path))
    assert any(l.startswith("git         commit ") and l.endswith("0 uncommitted file(s)") for l in clean["lines"]) and clean["notes"] == []
    (tmp_path / "b.txt").write_text("y")
    (tmp_path / "a.txt").write_text("changed")
    dirty = G.environment(str(tmp_path))
    assert any("2 uncommitted file(s)" in l for l in dirty["lines"]) and any("b.txt" in l for l in dirty["lines"])
    assert len(dirty["notes"]) == 1 and "2 uncommitted" in dirty["notes"][0] and not any("uncommitted" in p for p in dirty["problems"])


def test_environment_lists_ten_uncommitted_files_and_counts_the_rest(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("x")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "first")
    for i in range(13):
        (tmp_path / f"new{i:02d}.txt").write_text("x")
    lines = G.environment(str(tmp_path))["lines"]
    assert sum("new" in l for l in lines) == 10 and any("... and 3 more" in l for l in lines)


def test_environment_in_a_repository_without_a_commit_says_so_and_still_lists_the_files(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("x")
    env = G.environment(str(tmp_path))
    assert any(l.startswith("git         commit (none yet), 1 uncommitted file(s)") for l in env["lines"]) and len(env["notes"]) == 1


def test_environment_names_the_tag_on_the_commit(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("x")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "first")
    git(tmp_path, "tag", "baseline-v0")
    assert any(", tag baseline-v0," in l for l in G.environment(str(tmp_path))["lines"])


def test_the_api_key_is_reported_as_set_or_not_and_its_value_is_never_printed(tmp_path, monkeypatch):
    none = G.environment(str(tmp_path))
    assert any(l.startswith("GROQ_API_KEY  NO") for l in none["lines"]) and any("GROQ_API_KEY is not set" in p for p in none["problems"])
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    env = G.environment(str(tmp_path))
    assert any(l.startswith("GROQ_API_KEY  yes (environment variable)") for l in env["lines"]) and env["problems"] == []
    assert KEY not in "\n".join(env["lines"])


@pytest.mark.parametrize("content,expected", [
    (f"GROQ_API_KEY={KEY}\n", "yes (.env)"), (f'GROQ_API_KEY="{KEY}"\n', "yes (.env)"), (f"export GROQ_API_KEY={KEY}\r\n", "yes (.env)"),
    ("GROQ_API_KEY=\n", "NO"), ("# GROQ_API_KEY=abc\n", "NO"), ("OTHER_KEY=abcdefghijklmnopqrstuvwxyz\n", "NO"), ("", "NO")])
def test_the_api_key_in_the_dot_env_file(tmp_path, content, expected):
    (tmp_path / ".env").write_text(content)
    assert G._key_state(str(tmp_path)) == expected


def test_the_cache_is_counted_by_its_json_answers_only_and_the_env_variable_moves_it(tmp_path, monkeypatch):
    assert any("no cache folder yet" in l for l in G.environment(str(tmp_path))["lines"])
    cache = tmp_path / "cache" / "llm"
    (cache / "ab").mkdir(parents=True)
    (cache / "ab" / "x.json").write_text("{" + " " * 1_999_999 + "}")
    (cache / "ab" / "y.json").write_text("{}")
    (cache / "ab" / "notes.txt").write_text("not an answer")
    lines = G.environment(str(tmp_path))["lines"]
    assert any("LLM cache   2 answer(s), 2.0 MB" in l for l in lines) and any("(cache" in l and "llm)" in l for l in lines)
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "z.json").write_text("{}")
    monkeypatch.setenv("CAPP_CACHE_DIR", str(other))
    assert any("LLM cache   1 answer(s)" in l for l in G.environment(str(tmp_path))["lines"])


# ---------------------------------------------------------------- freeze
@pytest.mark.parametrize("status,problem", [("none", False), ("ok", False), ("analysis-changed", False), ("drift", True), ("unreadable", True)])
def test_freeze_state_makes_a_problem_only_of_drift_and_of_an_unreadable_manifest(tmp_path, monkeypatch, status, problem):
    monkeypatch.setattr(F, "status", lambda root: {"status": status, "id": "abc123" if status != "none" else None, "manifest": "FREEZE.json",
                                                   "differences": ["behaviour capp/x.py changed"], "why": "bad json" if status == "unreadable" else ""})
    res = G.freeze_state(str(tmp_path))
    assert res["lines"][0].startswith(f"status      {status}") and bool(res["problems"]) is problem
    if status == "none":
        assert "right until gate G2" in res["lines"][1]
    else:
        assert "freeze abc123" in res["lines"][0] and any("capp/x.py" in l for l in res["lines"])
    if problem:
        assert "leftover of a rehearsal" in res["problems"][0]
    if status == "unreadable":
        assert any("bad json" in l for l in res["lines"])


def test_freeze_state_without_a_manifest_uses_the_real_freeze_code(tmp_path):
    assert G.freeze_state(str(tmp_path))["lines"][0].startswith("status      none")


# ---------------------------------------------------------------- datasets
def test_datasets_say_which_are_there_which_are_sealed_and_only_a_missing_dev_set_is_a_problem(tmp_path):
    (tmp_path / "data").mkdir()
    parts = json.load(open(os.path.join(ROOT, "data", "dev_v2.json"), encoding="utf-8"))[:4]
    (tmp_path / "data" / "dev_v2.json").write_text(json.dumps(parts))
    res = G.datasets_state(str(tmp_path))
    rows = {l.split()[0]: l.split() for l in res["lines"][2:]}
    assert rows["dev_v2"][:4] == ["dev_v2", "open", "4", "ok"] and rows["test_v2"][1:3] == ["sealed", "missing"] and rows["bench_v1"][1:3] == ["sealed", "missing"]
    (tmp_path / "data" / "test_v2.json").write_text(json.dumps(parts[:2]))
    assert {l.split()[0]: l.split() for l in G.datasets_state(str(tmp_path))["lines"][2:]}["test_v2"][:4] == ["test_v2", "sealed", "2", "ok"]
    assert res["problems"] == []
    missing = G.datasets_state(str(tmp_path / "nowhere"))
    assert len(missing["problems"]) == 1 and "dev_v2.json is missing" in missing["problems"][0] and "make_dataset_v2" in missing["problems"][0]


def test_datasets_report_validation_problems_and_unreadable_files(tmp_path):
    (tmp_path / "data").mkdir()
    parts = json.load(open(os.path.join(ROOT, "data", "dev_v2.json"), encoding="utf-8"))[:2]
    parts[0]["material"] = "Unobtainium"
    (tmp_path / "data" / "dev_v2.json").write_text(json.dumps(parts))
    (tmp_path / "data" / "test_v2.json").write_text("{not json")
    res = G.datasets_state(str(tmp_path))
    assert any("dev_v2.json: 1 validation problem(s), first: DEV_001: material 'Unobtainium' invalid" in p for p in res["problems"])
    assert any("test_v2.json cannot be read" in p for p in res["problems"])
    rows = {l.split()[0]: l for l in res["lines"][2:]}
    assert "1 problem(s)" in rows["dev_v2"] and "unreadable" in rows["test_v2"]


# ---------------------------------------------------------------- runs
def test_runs_with_no_run_folder_and_folders_without_a_run_json_are_not_runs(tmp_path):
    assert G.runs_state(str(tmp_path))["lines"] == ["no run folder yet in results\\"]
    (tmp_path / "results" / "tables").mkdir(parents=True)
    (tmp_path / "results" / "diagnostics.txt").write_text("x")
    res = G.runs_state(str(tmp_path))
    assert res["n"] == 0 and res["problems"] == []


def test_a_run_row_has_the_figures_of_the_run_and_the_rows_are_sorted(tmp_path):
    make_run(tmp_path, "dev_b0", system="b0", n=20, strict=0.95, parse=0.98, freeze="ok", model="qwen/qwen3.8-27b")
    make_run(tmp_path, "dev_rb", system="rb", n=200, llm=False, strict=1.0, parse=None, model=None, mode="nocache")
    res = G.runs_state(str(tmp_path))
    assert res["n"] == 2 and res["lines"][0] == "2 run folder(s) in results\\ with a run.json" and res["problems"] == []
    rows = [l.split() for l in res["lines"][3:]]
    assert rows[0] == ["dev_b0", "b0", "dev_v2", "20", "1", "qwen/qwen3.8-27b", "live", "95.0%", "98.0%", "100", "0", "52k", "ok"]
    assert rows[1] == ["dev_rb", "rb", "dev_v2", "200", "1", "-", "nocache", "100.0%", "-", "-", "-", "-", "none"]


def test_the_run_rows_are_in_alphabetical_order_whatever_order_the_folders_were_made_in(tmp_path):
    for name in ("dev_m", "dev_z", "dev_a", "dev_q"):
        make_run(tmp_path, name)
    assert [l.split()[0] for l in G.runs_state(str(tmp_path))["lines"][3:]] == ["dev_a", "dev_m", "dev_q", "dev_z"]


def test_a_truncated_answer_a_missing_summary_and_a_record_count_that_disagrees_are_problems(tmp_path):
    make_run(tmp_path, "cut", llm={"calls": 10, "prompt_tokens": 1000, "completion_tokens": 1000, "truncated": 3})
    make_run(tmp_path, "nosum", summary=False)
    make_run(tmp_path, "short", n=5, records=4, n_records=5)
    res = G.runs_state(str(tmp_path))["problems"]
    assert any("results\\cut: 3 answer(s) hit the token limit" in p for p in res)
    assert any("results\\nosum: there is no summary.json" in p for p in res)
    assert any("results\\short: records.jsonl has 4 record(s), summary.json says 5" in p for p in res)
    assert len(res) == 3


def test_a_run_folder_without_records_or_with_a_broken_json_is_a_problem_and_still_a_row(tmp_path):
    d = make_run(tmp_path, "norec")
    (d / "records.jsonl").unlink()
    broken = make_run(tmp_path, "broken")
    (broken / "run.json").write_text("{oops")
    res = G.runs_state(str(tmp_path))
    assert any("results\\norec: there is no records.jsonl" in p for p in res["problems"])
    assert any("results\\broken: run.json or summary.json cannot be read" in p for p in res["problems"])
    rows = {l.split()[0]: l.split() for l in res["lines"][3:]}
    assert rows["broken"][1] == "unreadable" and len(rows["broken"]) == 13 and "norec" in rows


def test_blank_lines_in_records_are_not_records(tmp_path):
    d = make_run(tmp_path, "r", n=2)
    (d / "records.jsonl").write_text('{"a": 1}\n\n{"a": 2}\n   \n')
    assert G.runs_state(str(tmp_path))["problems"] == []


def test_a_long_list_of_runs_is_cut_and_says_how_many_more(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "MAX_RUN_ROWS", 3)
    for i in range(5):
        make_run(tmp_path, f"r{i}")
    lines = G.runs_state(str(tmp_path))["lines"]
    assert lines[0].startswith("5 run folder(s)") and lines[-1] == "... and 2 more" and len(lines) == 1 + 2 + 3 + 1


# ---------------------------------------------------------------- the G1 baselines
def test_the_baselines_to_make_come_with_the_exact_commands(tmp_path):
    lines = G.baselines_state(str(tmp_path), "dev", "dev_v2")["lines"]
    assert lines == ["to make  results\\dev_rb    python scripts\\run_eval.py --system rb --dataset dev_v2 --tag dev",
                     "to make  results\\dev_b1    python scripts\\run_eval.py --system b1 --dataset dev_v2 --tag dev --limit 20",
                     "to make  results\\dev_b0    python scripts\\run_eval.py --system b0 --dataset dev_v2 --tag dev --limit 20 --throttle 1"]


def test_a_baseline_counts_as_done_when_its_records_are_there_and_all_three_give_the_next_step(tmp_path):
    make_run(tmp_path, "dev_rb", system="rb")
    (tmp_path / "results" / "dev_b1").mkdir()                                  # a folder without records.jsonl is not a run
    lines = G.baselines_state(str(tmp_path), "dev", "dev_v2")["lines"]
    assert lines[0] == "done     results\\dev_rb" and lines[1].startswith("to make  results\\dev_b1") and lines[2].startswith("to make  results\\dev_b0")
    make_run(tmp_path, "dev_b1", system="b1")
    make_run(tmp_path, "dev_b0", system="b0")
    done = G.baselines_state(str(tmp_path), "dev", "dev_v2")["lines"]
    assert [l[:4] for l in done[:3]] == ["done"] * 3 and "all three are there" in done[3] and len(done) == 4


def test_the_tag_and_dataset_go_into_the_baseline_commands(tmp_path):
    lines = G.baselines_state(str(tmp_path), "fin", "test_v2")["lines"]
    assert "results\\fin_rb" in lines[0] and "--dataset test_v2 --tag fin" in lines[0]


def test_the_baseline_commands_are_the_lines_of_the_runbook():
    runbook = open(os.path.join(ROOT, "RUNBOOK.md"), encoding="utf-8").read().replace("\r\n", "\n").split("\n")
    for system, extra in G.BASELINES:
        assert f"python scripts\\run_eval.py --system {system} --dataset dev_v2 --tag dev{extra}" in runbook


# ---------------------------------------------------------------- the list of checks
def test_the_checks_are_a_fixed_list_of_offline_scripts(tmp_path):
    root = full_root(tmp_path)
    checks = G.check_list(str(root), "dev", "dev_v2", G.DEFAULT_GRID, "TMP")
    names = {os.path.basename(c["argv"][0])[:-3] for c in checks}
    assert names <= G.CHECK_SCRIPTS and names.isdisjoint({"run_eval", "smoke_llm", "benchmark", "release", "pack", "slides", "paper_draft", "show_plan"})
    assert not any(a in ("create", "build", "import", "commands") for c in checks for a in c["argv"])        # freeze create and make_tables build write files
    assert [c["title"] for c in checks] == ["physics_check", "feasibility_check", "objective_check", "feasrepair_check", "grid status", "compare_runs",
                                            "part_diff rb vs b0", "confidence_tune b0_r1", "risk_compare rb vs b0", "risk_compare rb vs b0_r1",
                                            "cascade_pareto b0_r1 b0", "freeze verify"]
    for name in names:                                                       # and every one of them exists
        assert os.path.isfile(os.path.join(ROOT, "scripts", name + ".py"))


def test_the_only_check_that_writes_files_is_pointed_at_the_temporary_folder(tmp_path):
    root = full_root(tmp_path)
    by = {c["title"]: c["argv"] for c in G.check_list(str(root), "dev", "dev_v2", G.DEFAULT_GRID, "THE-TMP")}
    assert by["objective_check"][-2:] == ["--out", "THE-TMP"]
    assert [t for t, a in by.items() if any(x in a for x in ("--out", "--csv", "--json", "--plot"))] == ["objective_check"]


def test_compare_runs_gets_the_baseline_folders_that_exist_and_needs_two(tmp_path):
    root = tmp_path / "r"
    (root / "results").mkdir(parents=True)
    get = lambda: next(c for c in G.check_list(str(root), "dev", "dev_v2", G.DEFAULT_GRID, "T") if c["title"] == "compare_runs")
    assert get()["argv"] == ["scripts/compare_runs.py"] and get()["needs"]
    make_run(root, "dev_b0")
    assert get()["needs"]
    make_run(root, "dev_rb", system="rb")
    assert get()["argv"] == ["scripts/compare_runs.py", "results/dev_rb", "results/dev_b0"] and get()["needs"] == []
    make_run(root, "dev_b1")
    make_run(root, "dev_b0_r1")
    assert get()["argv"][1:] == ["results/dev_rb", "results/dev_b1", "results/dev_b0", "results/dev_b0_r1"]


def test_the_tag_chooses_the_run_folders(tmp_path):
    root = full_root(tmp_path)
    for s in G.RUN_SYSTEMS:
        make_run(root, f"fin_{s}", system=s)
    by = {c["title"]: c["argv"] for c in G.check_list(str(root), "fin", "dev_v2", G.DEFAULT_GRID, "T")}
    assert by["part_diff rb vs b0"] == ["scripts/part_diff.py", "results/fin_rb", "results/fin_b0"]
    assert by["risk_compare rb vs b0_r1"] == ["scripts/risk_compare.py", "results/fin_rb", "results/fin_b0_r1", "--physics"]


# ---------------------------------------------------------------- running the checks
def test_every_check_runs_when_its_inputs_are_there(tmp_path):
    root = full_root(tmp_path, freeze_file=True)
    fake = Fake()
    done = G.run_checks(str(root), "dev", "dev_v2", G.DEFAULT_GRID, 77, 50, fake)
    assert [c["status"] for c in done] == ["ok"] * 12 and len(fake.calls) == 12
    assert fake.scripts() == ["physics_check", "feasibility_check", "objective_check", "feasrepair_check", "make_tables", "compare_runs", "part_diff",
                              "confidence_tune", "risk_compare", "risk_compare", "cascade_pareto", "freeze"]
    assert all(cwd == str(root) and timeout == 77 and argv[0] == sys.executable for argv, cwd, timeout in fake.calls)


def test_a_check_whose_input_is_missing_is_skipped_with_the_reason_and_not_started(tmp_path):
    root = tmp_path / "bare"
    root.mkdir()
    fake = Fake()
    done = G.run_checks(str(root), "dev", "dev_v2", G.DEFAULT_GRID, 60, 50, fake)
    assert fake.calls == [] and {c["status"] for c in done} == {"skipped"}
    why = {c["title"]: c["why"] for c in done}
    assert why["physics_check"] == "data\\dev_v2.json is not there (python scripts\\make_dataset_v2.py)"
    assert why["grid status"] == "configs\\ablation_template.json is not there"
    assert why["part_diff rb vs b0"] == "results\\dev_rb is not there yet" and why["freeze verify"].startswith("there is no FREEZE.json")
    assert why["compare_runs"].startswith("fewer than two of results\\dev_rb")


def test_part_diff_waits_for_both_folders_and_names_the_first_one_missing(tmp_path):
    root = full_root(tmp_path)
    import shutil
    shutil.rmtree(root / "results" / "dev_b0")
    done = {c["title"]: c for c in G.run_checks(str(root), "dev", "dev_v2", G.DEFAULT_GRID, 60, 50, Fake())}
    assert done["part_diff rb vs b0"]["why"] == "results\\dev_b0 is not there yet" and done["part_diff rb vs b0"]["status"] == "skipped"
    assert done["confidence_tune b0_r1"]["status"] == "ok"


def test_exit_codes_and_the_time_limit_become_statuses(tmp_path):
    root = full_root(tmp_path, freeze_file=True)
    fake = Fake({"physics_check": (2, "boom\n"), "part_diff": (None, "partial\n[stopped after 60 s]"), "freeze": (1, "Freeze: DRIFT\n")})
    done = {c["title"]: c for c in G.run_checks(str(root), "dev", "dev_v2", G.DEFAULT_GRID, 60, 50, fake)}
    assert (done["physics_check"]["status"], done["physics_check"]["code"]) == ("failed", 2)
    assert (done["part_diff rb vs b0"]["status"], done["part_diff rb vs b0"]["code"]) == ("timeout", None)
    assert (done["freeze verify"]["status"], done["freeze verify"]["code"]) == ("failed", 1) and done["feasibility_check"]["status"] == "ok"


def test_the_output_of_a_check_is_cut_and_made_lf_and_the_temporary_folder_is_not_shown(tmp_path):
    root = full_root(tmp_path)
    fake = Fake({"physics_check": (0, "\r\n".join(f"row {i}" for i in range(100)))})
    done = {c["title"]: c for c in G.run_checks(str(root), "dev", "dev_v2", G.DEFAULT_GRID, 60, 40, fake)}
    text = done["physics_check"]["text"]
    assert "\r" not in text and text.split("\n")[0] == "row 0" and text.split("\n")[-1] == "row 99" and "60 line(s) left out" in text
    assert done["objective_check"]["command"] == "python scripts\\objective_check.py --dataset dev_v2"
    assert "diagnostics-" not in "".join(c["command"] for c in done.values())


def test_the_temporary_folder_of_the_checks_exists_while_they_run_and_is_gone_after(tmp_path):
    root = full_root(tmp_path)
    seen = {}

    def runner(argv, cwd, timeout):
        if argv[1].endswith("objective_check.py"):
            seen["dir"] = argv[argv.index("--out") + 1]
            seen["there"] = os.path.isdir(seen["dir"])
        return 0, ""
    G.run_checks(str(root), "dev", "dev_v2", G.DEFAULT_GRID, 60, 50, runner)
    assert seen["there"] and not os.path.exists(seen["dir"])


def test_a_script_outside_the_offline_list_is_refused_before_anything_is_started(tmp_path, monkeypatch):
    root = full_root(tmp_path)
    fake = Fake()
    monkeypatch.setattr(G, "check_list", lambda *a: [{"title": "sneaky", "argv": ["scripts/run_eval.py", "--system", "b0"], "needs": []}])
    with pytest.raises(G.DiagnosticsError, match="run_eval is not one of the offline scripts"):
        G.run_checks(str(root), "dev", "dev_v2", G.DEFAULT_GRID, 60, 50, fake)
    assert fake.calls == []


def test_progress_is_announced_only_for_the_checks_that_run(tmp_path):
    root = full_root(tmp_path)
    messages = []
    G.run_checks(str(root), "dev", "dev_v2", G.DEFAULT_GRID, 60, 50, Fake(), progress=messages.append)
    assert messages[0] == "running physics_check ..." and "running freeze verify ..." not in messages and len(messages) == 11


# ---------------------------------------------------------------- the whole file
def test_the_file_has_the_problems_first_then_the_summary_of_the_checks_then_the_sections(tmp_path):
    rep = report(full_root(tmp_path))
    text = rep["text"]
    heads = [l for l in text.split("\n") if l.startswith("========")]
    assert [h.split()[1] for h in heads[:5]] == ["environment", "freeze", "datasets", "runs", "G1"] and heads[5].startswith("======== check: physics_check")
    assert text.startswith("DIAGNOSTICS  (2026-10-28 03:00 UTC)\nrepo repo   tag dev   dataset dev_v2   grid configs/ablation_template.json\n\nPROBLEMS")
    assert text.index("PROBLEMS") < text.index("CHECKS  ") < text.index("======== environment")
    assert "CHECKS  11 ok, 1 skipped" in text and "  skipped freeze verify   (there is no FREEZE.json" in text
    assert re.search(r"\n  ok      physics_check   exit 0, \d+\.\d s\n", text)
    assert "$ python scripts\\physics_check.py --dataset dev_v2\n(exit 0," in text and rep["counts"] == {"ok": 11, "failed": 0, "timeout": 0, "skipped": 1}


def test_a_failed_check_and_a_timeout_are_problems_but_a_freeze_drift_exit_is_the_freeze_sections_business(tmp_path, monkeypatch):
    root = full_root(tmp_path, freeze_file=True)
    monkeypatch.setattr(F, "status", lambda r: {"status": "ok", "id": "a1", "manifest": "FREEZE.json", "differences": []})
    fake = Fake({"physics_check": (2, "boom"), "part_diff": (None, "[stopped after 5 s]"), "freeze": (1, "Freeze a1: DRIFT")})
    rep = report(root, fake, timeout=5)
    assert "check 'physics_check' ended with exit code 2" in rep["problems"] and "check 'part_diff rb vs b0' did not finish in 5 s" in rep["problems"]
    assert not any("freeze verify" in p for p in rep["problems"]) and rep["counts"] == {"ok": 9, "failed": 2, "timeout": 1, "skipped": 0}


def test_a_dirty_tree_is_a_note_and_a_missing_key_is_a_problem(tmp_path):
    root = full_root(tmp_path)
    git(root, "init", "-q")
    git(root, "add", "data")
    git(root, "commit", "-q", "-m", "first")
    text = report(root)["text"]
    problems = text.split("PROBLEMS")[1].split("\n\n")[0]
    assert "GROQ_API_KEY is not set" in problems and "uncommitted" not in problems
    assert "\nNOTES\n  - git:" in text and text.index("NOTES") < text.index("CHECKS  ")


def test_with_no_problem_and_no_note_the_file_says_none(tmp_path, monkeypatch):
    root = full_root(tmp_path, freeze_file=True)
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    monkeypatch.setattr(F, "status", lambda r: {"status": "ok", "id": "a1", "manifest": "FREEZE.json", "differences": []})
    rep = report(root)
    assert rep["problems"] == [] and "PROBLEMS\n  none\n" in rep["text"] and "NOTES" not in rep["text"]


def test_no_checks_leaves_out_the_checks_and_starts_nothing(tmp_path):
    fake = Fake()
    rep = report(full_root(tmp_path), fake, checks=False)
    assert fake.calls == [] and "CHECKS" not in rep["text"] and "check:" not in rep["text"] and rep["counts"] == {"ok": 0, "failed": 0, "timeout": 0, "skipped": 0}
    assert "======== environment" in rep["text"] and "======== runs" in rep["text"]


def test_the_same_inputs_give_the_same_file_and_it_has_only_line_feeds(tmp_path):
    root = full_root(tmp_path)
    a, b = report(root)["text"], report(root)["text"]
    assert a == b and "\r" not in a and a.endswith("\n") and not a.endswith("\n\n")


def test_a_bad_tag_or_dataset_name_is_refused_before_anything_runs(tmp_path):
    fake, root = Fake(), full_root(tmp_path)
    for kw in ({"tag": "a/b"}, {"tag": ""}, {"tag": "x y"}, {"tag": "..\\x"}, {"dataset": "../data"}, {"dataset": ""}):
        with pytest.raises(G.DiagnosticsError):
            report(root, fake, **kw)
    assert fake.calls == []


def test_the_tag_and_the_dataset_show_in_the_header_and_the_commands(tmp_path):
    root = full_root(tmp_path)
    (root / "data" / "test_v2.json").write_text("[]")
    for s in G.RUN_SYSTEMS:
        make_run(root, f"x1_{s}", system=s)
    fake = Fake()
    text = report(root, fake, tag="x1", dataset="test_v2")["text"]
    assert "tag x1   dataset test_v2" in text and "--dataset test_v2" in text and "python scripts\\part_diff.py results\\x1_rb results\\x1_b0" in text


# ---------------------------------------------------------------- secrets
def test_a_line_of_check_output_that_looks_like_a_secret_never_reaches_the_file(tmp_path):
    secret = "gsk_" + "A1b2C3d4" * 4
    rep = report(full_root(tmp_path), Fake({"physics_check": (0, f"fine\nthe key is {secret} here\nfine too\n")}))
    assert secret not in rep["text"] and "gsk_" not in rep["text"]
    assert "[line removed: it looked like a secret (Groq API key)]" in rep["text"] and "fine too" in rep["text"]
    assert len(rep["removed"]) == 1 and "NOTE: 1 line(s) of this file looked like a secret and were removed." in rep["text"]


def test_the_value_of_the_key_of_this_machine_is_removed_wherever_it_shows_up(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    rep = report(full_root(tmp_path), Fake({"feasibility_check": (0, f"debug: {KEY}\n")}))
    assert KEY not in rep["text"] and any(w == "a value of your .env or environment" for _, w in rep["removed"])
    assert "yes (environment variable)" in rep["text"]


def test_the_value_in_the_dot_env_file_is_removed_too(tmp_path):
    root = full_root(tmp_path)
    (root / ".env").write_text(f"GROQ_API_KEY={KEY}\n")
    rep = report(root, Fake({"physics_check": (0, f"{KEY}\n")}))
    assert KEY not in rep["text"] and "yes (.env)" in rep["text"] and len(rep["removed"]) == 1


def test_a_file_without_a_secret_has_no_note_and_the_text_is_untouched(tmp_path):
    root = full_root(tmp_path)
    rep = report(root)
    assert rep["removed"] == [] and "looked like a secret" not in rep["text"]
    assert G.redact("a\nb\n", str(root)) == ("a\nb\n", [])


def test_redact_removes_only_the_lines_that_hold_a_secret_and_keeps_the_numbering(tmp_path):
    text = "one\nkey sk-" + "x" * 40 + "\nthree\napi_key = abcdefghijklmnopqrstuvwxyz\nfive\n"
    out, found = G.redact(text, str(tmp_path))
    assert out.split("\n") == ["one", "[line removed: it looked like a secret (API key (sk-...))]", "three",
                               "[line removed: it looked like a secret (key, secret or password assignment)]", "five", ""]
    assert [l for l, _ in found] == [2, 4]


# ---------------------------------------------------------------- writing the file
def test_write_text_writes_utf8_with_line_feeds_and_leaves_no_temporary_file(tmp_path):
    out = tmp_path / "results" / "diagnostics.txt"
    assert G.write_text("ü\nb\n", str(out)) == str(out)
    assert out.read_bytes() == "ü\nb\n".encode("utf-8") and [p.name for p in out.parent.iterdir()] == ["diagnostics.txt"]


def test_write_text_replaces_an_older_file(tmp_path):
    out = tmp_path / "d.txt"
    out.write_text("old")
    G.write_text("new\n", str(out))
    assert out.read_text() == "new\n" and [p.name for p in tmp_path.iterdir()] == ["d.txt"]


@pytest.mark.skipif(os.name == "nt", reason="file modes are not meaningful on Windows")
def test_write_text_makes_a_normal_readable_file(tmp_path):
    G.write_text("x\n", str(tmp_path / "d.txt"))
    assert (tmp_path / "d.txt").stat().st_mode & 0o777 == 0o644


def test_write_text_to_a_place_that_cannot_be_written_raises_and_leaves_nothing_behind(tmp_path):
    (tmp_path / "dir").mkdir()
    with pytest.raises(OSError):
        G.write_text("x\n", str(tmp_path / "dir"))
    assert [p.name for p in tmp_path.iterdir()] == ["dir"] and list((tmp_path / "dir").iterdir()) == []


# ---------------------------------------------------------------- the command line
@pytest.fixture()
def cli():
    spec = importlib.util.spec_from_file_location("diagnostics_cli", os.path.join(ROOT, "scripts", "diagnostics.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def snapshot(root):
    out = {}
    for r, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache", ".git")]
        for f in files:
            p = os.path.join(r, f)
            st = os.stat(p)
            out[os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns)
    return out


def test_the_command_line_writes_the_one_file_and_changes_nothing_else_in_the_repository(cli, tmp_path, capsys):
    before = snapshot(ROOT)
    out = tmp_path / "d.txt"
    assert cli.main(["--out", str(out), "--tag", "no_such_tag_for_the_test"]) == 0
    printed = capsys.readouterr()
    assert snapshot(ROOT) == before
    text = out.read_text(encoding="utf-8")
    for name in ("physics_check", "feasibility_check", "objective_check", "feasrepair_check", "grid status"):      # whatever FREEZE.json and the results of this machine are
        assert re.search(rf"\n  ok      {name}   exit 0, ", text)
    assert "\n  skipped part_diff rb vs b0   (results\\no_such_tag_for_the_test_rb is not there yet)" in text and "=== physics check:" in text and "=== feasibility check:" in text
    assert f"Written: {out}" in printed.out and "Attach this one file to the chat." in printed.out and "\nchecks: " in printed.out
    assert "running physics_check ..." in printed.err and sorted(p.name for p in tmp_path.iterdir()) == ["d.txt"]


def test_the_command_line_with_no_checks_is_quick_and_prints_the_problems(cli, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    out = tmp_path / "d.txt"
    assert cli.main(["--out", str(out), "--no-checks", "--tag", "no_such_tag_for_the_test"]) == 0
    printed = capsys.readouterr().out
    assert "checks:" not in printed and "CHECKS" not in out.read_text(encoding="utf-8")
    assert "problem(s)" in printed.split("\n")[0]


def test_the_command_line_passes_its_arguments_on_and_prints_the_problems_and_the_removed_lines(cli, monkeypatch, capsys):
    seen, written = {}, []

    def stub(root, **kw):
        seen.update(kw)
        return {"text": "t\n", "problems": ["first problem", "second problem"], "notes": [], "counts": {"ok": 3, "failed": 1, "timeout": 0, "skipped": 2}, "removed": [(3, "x")]}
    monkeypatch.setattr(cli.G, "build_report", stub)
    monkeypatch.setattr(cli.G, "write_text", lambda text, out: written.append((text, out)) or out)
    assert cli.main(["--out", "o.txt", "--tag", "t1", "--dataset", "d1", "--grid", "g.json", "--timeout", "7", "--max-lines", "20"]) == 0
    assert (seen["tag"], seen["dataset"], seen["grid"], seen["checks"], seen["timeout"], seen["max_lines"]) == ("t1", "d1", "g.json", True, 7, 20) and callable(seen["progress"])
    assert written == [("t\n", "o.txt")]
    out = capsys.readouterr().out.split("\n")
    assert out[:3] == ["2 problem(s):", "  - first problem", "  - second problem"]
    assert "1 line(s) that looked like a secret were removed from the file" in out and "checks: 3 ok, 1 failed, 2 skipped" in out
    assert cli.main(["--out", "o.txt", "--no-checks"]) == 0
    assert seen["checks"] is False and "checks:" not in capsys.readouterr().out


def test_the_command_line_refuses_wrong_arguments_with_exit_2_and_writes_nothing(cli, tmp_path, capsys):
    out = tmp_path / "d.txt"
    for args, word in ((["--timeout", "0"], "--timeout"), (["--max-lines", "3"], "--max-lines"), (["--tag", "a/b"], "--tag"), (["--dataset", "../x"], "--dataset")):
        assert cli.main(["--out", str(out), "--no-checks"] + args) == 2
        assert word in capsys.readouterr().err
    assert not out.exists()


def test_the_command_line_says_so_when_the_file_cannot_be_written(cli, tmp_path, capsys):
    (tmp_path / "isdir").mkdir()
    assert cli.main(["--out", str(tmp_path / "isdir"), "--no-checks"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("STOPPED: ") and "cannot be written" in err


def test_without_out_the_file_goes_to_results_diagnostics_txt_of_the_repository(cli, monkeypatch, capsys):
    written = []
    monkeypatch.setattr(cli.G, "write_text", lambda text, out: written.append(out) or out)
    assert cli.main(["--no-checks", "--tag", "no_such_tag_for_the_test"]) == 0
    assert written == [os.path.join(ROOT, "results", "diagnostics.txt")] and "Written: " + written[0] in capsys.readouterr().out


def test_the_default_output_is_results_diagnostics_txt_and_the_help_says_so():
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "diagnostics.py"), "--help"], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and "results/diagnostics.txt" in p.stdout.replace("\n", " ") and "--no-checks" in p.stdout
    assert G.DEFAULT_OUT == "results/diagnostics.txt"
