"""Step 34: the G1 baselines in one command (capp/g1.py, scripts/g1.py).

Offline. The sequence is driven with a runner that records the commands it is given (and, where a test needs it, writes the run folder the
real command would have written), so what is checked is the decision logic: which runs are skipped, which are repeated, what is asked,
where the sequence stops, and that the diagnostics file is still asked for at the end. The real commands are not run here.
"""
import builtins
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capp import config, dataset as D, diagnostics as DG, g1 as G           # noqa: E402

PARTS = D.load_dataset("dev_v2")[:30]               # a dataset of 30 parts: rb wants 30, b1 and b0 want 20
WANTED = {"rb": 30, "b1": 20, "b0": 20}
OK_CAND = {"sample_id": 0, "llm_ok": True, "llm_raw": "[]"}
API_CAND = {"sample_id": 0, "llm_ok": False, "llm_raw": "API Error: HTTP 503: service said no [after 6 attempts]"}


@pytest.fixture(autouse=True)
def no_key_in_the_environment(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def load_parts(name):
    assert name == "dev_v2"
    return PARTS


def make_run(root, system, n, tag="dev", api=0):
    folder = root / "results" / f"{tag}_{system}"
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        cand = API_CAND if i < api else OK_CAND
        lines.append(json.dumps({"part_id": f"P{i}", "repeat": 1, "candidates": [cand]}))
    (folder / "records.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


class Runner:
    """Records (argv, cwd). `make` = systems whose run folder it writes when asked to run them (as the real command would); `codes` = exit codes."""

    def __init__(self, root, make=("rb", "b1", "b0"), codes=None, interrupt=None):
        self.root, self.make, self.codes, self.interrupt, self.calls = root, make, codes or {}, interrupt, []

    def __call__(self, argv, cwd):
        self.calls.append((argv, cwd))
        script = os.path.basename(argv[0])
        if script == "diagnostics.py":
            return self.codes.get("diagnostics", 0)
        system = argv[argv.index("--system") + 1]
        if self.interrupt == system:
            raise KeyboardInterrupt
        code = self.codes.get(system, 0)
        if code == 0 and system in self.make:
            make_run(self.root, system, WANTED[system])
        return code

    @property
    def systems(self):
        return [c[0][c[0].index("--system") + 1] if "--system" in c[0] else "diagnostics" for c in self.calls]


class Say:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        self.lines.append(str(text))

    @property
    def text(self):
        return "\n".join(self.lines)


def go(root, runner=None, ask=None, key=True, **kw):
    if key:
        (root / ".env").write_text("GROQ_API_KEY=abc\n", encoding="utf-8")
    runner = runner or Runner(root)
    say = Say()
    asked = []

    def default_ask(question):
        asked.append(question)
        return True
    code = G.run_g1(str(root), runner=runner, ask=ask or default_ask, load_parts=load_parts, out=say, **kw)
    return code, runner, say, asked


# ---- what is already there ----------------------------------------------------------------------------------------
def test_wanted_parts_is_the_limit_or_the_whole_dataset():
    assert G.wanted_parts(" --limit 20", 200) == 20 and G.wanted_parts(" --limit 20 --throttle 1", 200) == 20
    assert G.wanted_parts("", 200) == 200 and G.wanted_parts(" --limit 500", 200) == 200


def test_a_run_folder_that_is_not_there_is_to_make(tmp_path):
    assert G.step_state(str(tmp_path), "dev", "b1", 20) == ("make", "")


def test_a_complete_run_is_done(tmp_path):
    make_run(tmp_path, "b1", 20)
    assert G.step_state(str(tmp_path), "dev", "b1", 20) == ("done", "")


def test_a_run_with_the_wrong_number_of_records_is_run_again(tmp_path):
    make_run(tmp_path, "b1", 5)
    assert G.step_state(str(tmp_path), "dev", "b1", 20) == ("redo", "it has 5 record(s), 20 wanted")
    make_run(tmp_path, "b0", 0)
    assert G.step_state(str(tmp_path), "dev", "b0", 20) == ("redo", "it has 0 record(s), 20 wanted")


def test_a_run_with_more_records_than_wanted_is_run_again_too(tmp_path):
    make_run(tmp_path, "b1", 25)                                    # for example made once without --limit
    assert G.step_state(str(tmp_path), "dev", "b1", 20) == ("redo", "it has 25 record(s), 20 wanted")


def test_a_run_with_api_errors_is_run_again_and_says_why(tmp_path):
    make_run(tmp_path, "b0", 20, api=3)
    state, why = G.step_state(str(tmp_path), "dev", "b0", 20)
    assert state == "redo" and why == "3 API error(s) in it (rate limit or outage, not the model)"


def test_records_that_cannot_be_read_mean_run_again(tmp_path):
    folder = tmp_path / "results" / "dev_b1"
    folder.mkdir(parents=True)
    (folder / "records.jsonl").write_text("{not json\n", encoding="utf-8")
    assert G.step_state(str(tmp_path), "dev", "b1", 20) == ("redo", "its records cannot be read")


def test_the_tag_is_part_of_the_folder(tmp_path):
    make_run(tmp_path, "b1", 20, tag="other")
    assert G.step_state(str(tmp_path), "dev", "b1", 20)[0] == "make" and G.step_state(str(tmp_path), "other", "b1", 20)[0] == "done"


# ---- the commands -----------------------------------------------------------------------------------------------------
def test_the_commands_are_the_ones_of_the_runbook():
    cmds = [" ".join(G.command(s, e, "dev", "dev_v2")) for s, e in DG.BASELINES]
    assert cmds == ["scripts/run_eval.py --system rb --dataset dev_v2 --tag dev",
                    "scripts/run_eval.py --system b1 --dataset dev_v2 --tag dev --limit 20",
                    "scripts/run_eval.py --system b0 --dataset dev_v2 --tag dev --limit 20 --throttle 1"]
    text = open(os.path.join(ROOT, "RUNBOOK.md"), encoding="utf-8").read().replace("\\", "/")
    for c in cmds:
        assert "python " + c in text, c


def test_the_dataset_and_tag_are_in_every_command():
    for s, e in DG.BASELINES:
        argv = G.command(s, e, "mytag", "other_set")
        assert argv[argv.index("--dataset") + 1] == "other_set" and argv[argv.index("--tag") + 1] == "mytag"


def test_the_throttle_of_a_command_is_read_from_its_options_and_is_zero_without():
    assert G._throttle_of("") == 0.0 and G._throttle_of(" --limit 20") == 0.0
    assert G._throttle_of(" --limit 20 --throttle 1") == 1.0 and G._throttle_of(" --throttle 2.5") == 2.5


def test_max_calls_goes_to_the_api_runs_only():
    assert "--max-calls" not in G.command("rb", "", "dev", "dev_v2", 50)
    assert G.command("b1", " --limit 20", "dev", "dev_v2", 50)[-2:] == ["--max-calls", "50"]
    assert "--max-calls" not in G.command("b1", " --limit 20", "dev", "dev_v2")


def test_show_writes_a_command_a_person_can_paste():
    assert G.show(["scripts/run_eval.py", "--system", "b0", "--tag", "my tag"]) == "python scripts/run_eval.py --system b0 --tag 'my tag'"


# ---- the sequence --------------------------------------------------------------------------------------------------------
def test_nothing_there_runs_rb_b1_b0_then_the_diagnostics_in_the_repository_folder(tmp_path):
    code, runner, say, asked = go(tmp_path, yes=True)
    assert code == 0 and runner.systems == ["rb", "b1", "b0", "diagnostics"]
    assert all(cwd == str(tmp_path) for _, cwd in runner.calls) and asked == []
    assert runner.calls[-1][0] == ["scripts/diagnostics.py", "--tag", "dev", "--dataset", "dev_v2"]
    assert "All three baselines are there." in say.text and "diagnostics failed" not in say.text and "diagnostics.py failed" not in say.text
    assert "API calls in this command: at most 120" in say.text


def test_runs_that_are_complete_are_not_run_again(tmp_path):
    make_run(tmp_path, "rb", 30)
    make_run(tmp_path, "b1", 20)
    code, runner, say, _ = go(tmp_path, yes=True)
    assert runner.systems == ["b0", "diagnostics"] and code == 0
    assert "rb : done, not run again" in say.text and "b1 : done, not run again" in say.text and "b0 : to make" in say.text
    assert "at most 100" in say.text


def test_a_run_with_api_errors_is_run_again_with_the_same_command(tmp_path):
    make_run(tmp_path, "rb", 30)
    make_run(tmp_path, "b1", 20)
    make_run(tmp_path, "b0", 20, api=2)
    code, runner, say, _ = go(tmp_path, yes=True)
    assert runner.systems == ["b0", "diagnostics"] and code == 0
    assert "b0 : to run again: 2 API error(s) in it" in say.text
    assert runner.calls[0][0] == G.command("b0", " --limit 20 --throttle 1", "dev", "dev_v2")


def test_everything_done_runs_only_the_diagnostics(tmp_path):
    for s in WANTED:
        make_run(tmp_path, s, WANTED[s])
    code, runner, say, _ = go(tmp_path, yes=True)
    assert runner.systems == ["diagnostics"] and code == 0 and "API calls in this command" not in say.text


def test_each_api_run_shows_its_plan_and_asks_unless_yes_is_given(tmp_path):
    code, runner, say, asked = go(tmp_path)
    assert code == 0 and len(asked) == 2 and "b1" in asked[0] and "b0" in asked[1] and all("calls the API" in q for q in asked)
    assert "Plan for b1: 20 parts" in say.text and "Plan for b0: 20 parts" in say.text and "Plan for rb" not in say.text
    assert say.text.index("Plan for b1") < say.text.index("=== b0")


def test_an_answer_of_no_stops_the_sequence_but_the_diagnostics_are_still_written(tmp_path):
    code, runner, say, asked = go(tmp_path, ask=lambda q: False)
    assert code == 1 and runner.systems == ["rb", "diagnostics"]
    assert "stopped before b1: not confirmed" in say.text and "Still to do: b1, b0" in say.text


def test_no_key_runs_rb_and_stops_before_the_first_api_run_with_the_place_for_the_key(tmp_path):
    code, runner, say, asked = go(tmp_path, key=False, yes=True)
    assert code == 1 and runner.systems == ["rb", "diagnostics"] and asked == []
    assert "GROQ_API_KEY: NOT set" in say.text and "GROQ_API_KEY is not set. Put GROQ_API_KEY=... in the .env file" in say.text
    assert "Still to do: b1, b0" in say.text


def test_the_key_in_the_environment_is_enough(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "abc")
    code, runner, say, _ = go(tmp_path, key=False, yes=True)
    assert code == 0 and "GROQ_API_KEY: set" in say.text


def test_a_run_that_fails_stops_the_sequence_and_says_what_to_do(tmp_path):
    runner = Runner(tmp_path, codes={"b1": 2})
    code, runner, say, _ = go(tmp_path, runner=runner, yes=True)
    assert code == 1 and runner.systems == ["rb", "b1", "diagnostics"]
    assert "b1 stopped (exit code 2); the reason is printed just above." in say.text and "run this command again" in say.text
    assert "Still to do: b1, b0" in say.text


def test_a_run_that_exits_zero_without_its_records_is_not_called_done(tmp_path):
    runner = Runner(tmp_path, make=("rb", "b0"))                  # b1 "succeeds" but writes nothing
    code, runner, say, _ = go(tmp_path, runner=runner, yes=True)
    assert code == 1 and "Still to do: b1" in say.text and "Still to do: b1, b0" not in say.text


def test_ctrl_c_stops_at_once_without_the_diagnostics_and_says_how_to_go_on(tmp_path):
    runner = Runner(tmp_path, interrupt="b1")
    code, runner, say, _ = go(tmp_path, runner=runner, yes=True)
    assert code == 1 and runner.systems == ["rb", "b1"]
    assert "Interrupted during b1" in say.text and "in the cache" in say.text and "diagnostics" not in say.text


def test_dry_run_runs_nothing_and_shows_the_plan_of_the_api_runs(tmp_path):
    code, runner, say, asked = go(tmp_path, dry_run=True)
    assert code == 0 and runner.calls == [] and asked == []
    assert "Dry run: nothing was run." in say.text and "Plan for b1" in say.text and "Plan for b0" in say.text
    assert "=== rb: python scripts/run_eval.py --system rb" in say.text and "Plan for rb" not in say.text
    assert "=== diagnostics" not in say.text


def test_dry_run_does_not_need_a_key(tmp_path):
    code, runner, say, _ = go(tmp_path, key=False, dry_run=True)
    assert code == 0 and "NOT set" in say.text and "is not set. Put" not in say.text


def test_the_diagnostics_can_be_left_out(tmp_path):
    code, runner, say, _ = go(tmp_path, yes=True, diag=False)
    assert runner.systems == ["rb", "b1", "b0"] and "=== diagnostics" not in say.text


def test_a_diagnostics_failure_is_said(tmp_path):
    runner = Runner(tmp_path, codes={"diagnostics": 2})
    code, runner, say, _ = go(tmp_path, runner=runner, yes=True)
    assert "diagnostics.py failed (exit code 2); its message is just above." in say.text


def test_max_calls_is_added_to_the_api_runs(tmp_path):
    code, runner, say, _ = go(tmp_path, yes=True, max_calls=40)
    flags = {runner.systems[i]: "--max-calls" in c[0] for i, c in enumerate(runner.calls)}
    assert flags == {"rb": False, "b1": True, "b0": True, "diagnostics": False}
    assert runner.calls[1][0][-2:] == ["--max-calls", "40"]


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "5"])
def test_a_bad_max_calls_is_refused_before_anything_runs(tmp_path, bad):
    runner = Runner(tmp_path)
    with pytest.raises(G.G1Error, match="--max-calls must be a whole number of 1 or more"):
        G.run_g1(str(tmp_path), max_calls=bad, runner=runner, load_parts=load_parts, out=Say())
    assert runner.calls == []


def test_a_dataset_that_cannot_be_loaded_is_a_g1_error(tmp_path):
    def nothing(name):
        raise FileNotFoundError("no such file")
    with pytest.raises(G.G1Error, match="dataset dev_v2: no such file"):
        G.run_g1(str(tmp_path), runner=Runner(tmp_path), load_parts=nothing, out=Say())


def test_the_tag_and_dataset_reach_the_commands(tmp_path):
    make_run(tmp_path, "rb", 30, tag="mine")
    (tmp_path / ".env").write_text("GROQ_API_KEY=abc\n", encoding="utf-8")
    runner, say = Runner(tmp_path), Say()
    G.run_g1(str(tmp_path), tag="mine", yes=True, runner=runner, load_parts=lambda n: PARTS, out=say)
    assert runner.calls[0][0][:7] == ["scripts/run_eval.py", "--system", "b1", "--dataset", "dev_v2", "--tag", "mine"]
    assert runner.calls[-1][0] == ["scripts/diagnostics.py", "--tag", "mine", "--dataset", "dev_v2"]


def test_the_estimate_in_the_plan_comes_from_the_runs_in_results(tmp_path):
    run = tmp_path / "results" / "old_b0"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"model": config.model("primary"), "llm_client_summary": {
        "calls": 100, "prompt_tokens": 50000, "completion_tokens": 10000, "latency_mean_s": 2.0}}), encoding="utf-8")
    code, runner, say, _ = go(tmp_path, dry_run=True)
    assert "10.0k in + 2.0k out tokens if every call is live" in say.text                 # b1: 20 calls x 500 in / 100 out


# ---- the default runner and the default question ----------------------------------------------------------------------
def test_the_default_runner_runs_python_in_the_given_folder_and_returns_its_exit_code(tmp_path):
    assert G.default_runner(["-c", "open('here.txt', 'w').write('1')"], str(tmp_path)) == 0
    assert (tmp_path / "here.txt").exists()
    assert G.default_runner(["-c", "import sys; sys.exit(7)"], str(tmp_path)) == 7


@pytest.mark.parametrize("typed,answer", [("y", True), ("Y", True), (" yes ", True), ("YES", True), ("n", False), ("", False), ("no", False), ("ja", False)])
def test_the_question_is_yes_only_for_y_or_yes(monkeypatch, typed, answer):
    monkeypatch.setattr(builtins, "input", lambda prompt="": typed)
    assert G.default_ask("Run?") is answer


def test_no_input_at_all_is_a_no(monkeypatch):
    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr(builtins, "input", eof)
    assert G.default_ask("Run?") is False


# ---- scripts/g1.py ------------------------------------------------------------------------------------------------------
def load_cli():
    spec = importlib.util.spec_from_file_location("g1_cli", os.path.join(ROOT, "scripts", "g1.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_the_command_line_dry_run_prints_the_sequence_and_exits_0(tmp_path, capsys):
    cli = load_cli()
    assert cli.main(["--dry-run", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "G1 baselines on dev_v2 (tag dev): rb, b1, b0" in out and "Dry run: nothing was run." in out
    assert list(tmp_path.iterdir()) == []                                                    # nothing was written


def test_the_command_line_refuses_a_sealed_dataset(capsys):
    cli = load_cli()
    for name in ("test_v2", "bench_v1"):
        assert cli.main(["--dataset", name, "--dry-run"]) == 2
        assert "is sealed" in capsys.readouterr().err


def test_the_command_line_says_a_wrong_dataset_or_max_calls_in_words(tmp_path, capsys):
    cli = load_cli()
    assert cli.main(["--dataset", str(tmp_path / "nowhere.json"), "--dry-run", "--root", str(tmp_path)]) == 2
    assert "STOPPED: dataset" in capsys.readouterr().err
    assert cli.main(["--max-calls", "0", "--dry-run", "--root", str(tmp_path)]) == 2
    assert "STOPPED: --max-calls must be a whole number of 1 or more" in capsys.readouterr().err


def test_the_command_line_passes_its_options_to_the_sequence(monkeypatch, tmp_path):
    cli = load_cli()
    seen = {}

    def fake(root, **kw):
        seen.update(root=root, **kw)
        return 1
    monkeypatch.setattr(cli.g1, "run_g1", fake)
    assert cli.main(["--tag", "t", "--yes", "--max-calls", "9", "--no-diagnostics", "--root", str(tmp_path)]) == 1
    assert seen == {"root": str(tmp_path), "tag": "t", "dataset": "dev_v2", "yes": True, "dry_run": False, "max_calls": 9, "diag": False}
    seen.clear()
    cli.main(["--dry-run", "--root", str(tmp_path)])
    assert seen["tag"] == "dev" and seen["yes"] is False and seen["dry_run"] is True and seen["max_calls"] is None and seen["diag"] is True
