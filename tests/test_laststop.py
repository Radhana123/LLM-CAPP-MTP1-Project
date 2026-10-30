"""Step 43: why the last run_eval stopped (capp/laststop.py, scripts/run_eval.py) and where scripts/g2.py prints it.

Offline. run_eval's harness call is replaced by one that raises; the G2 loop uses the fake runner of tests/test_g2.py.
"""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from capp import harness, laststop as L                                    # noqa: E402
import run_eval                                                            # noqa: E402
import test_g2 as TG                                                       # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("GROQ_API_KEY", "CAPP_QWEN_REASONING", "CAPP_MODEL_PRIMARY"):
        monkeypatch.delenv(name, raising=False)


# ================================================================ the module
def test_what_is_written_is_read_back_on_one_line(tmp_path):
    L.write(str(tmp_path), "STOPPED: the rate limit asks\nfor a wait of 80000 s.\n\n")
    assert L.read(str(tmp_path)) == "STOPPED: the rate limit asks for a wait of 80000 s."
    assert os.path.basename(L.path(str(tmp_path))) == "last_stop.txt"


def test_there_is_nothing_to_read_where_nothing_was_written(tmp_path):
    assert L.read(str(tmp_path)) == "" and L.read(str(tmp_path / "no_such_folder")) == ""


def test_clear_removes_the_note_and_is_quiet_when_there_is_none(tmp_path):
    L.write(str(tmp_path), "STOPPED: x")
    L.clear(str(tmp_path))
    assert L.read(str(tmp_path)) == "" and not os.path.exists(L.path(str(tmp_path)))
    L.clear(str(tmp_path))
    L.clear(str(tmp_path / "no_such_folder"))


def test_write_creates_the_folder_and_replaces_an_older_note(tmp_path):
    folder = str(tmp_path / "new" / "results")
    L.write(folder, "first")
    L.write(folder, "second")
    assert L.read(folder) == "second"


def test_a_note_that_cannot_be_written_never_raises(tmp_path):
    blocker = tmp_path / "a_file"
    blocker.write_text("x", encoding="utf-8")
    L.write(str(blocker / "inside"), "STOPPED: x")                       # the folder cannot be made
    assert L.read(str(blocker / "inside")) == ""


def test_the_text_is_plain_ascii_and_at_most_700_characters(tmp_path):
    L.write(str(tmp_path), "limit ≤ 600 → wait éé")
    got = L.read(str(tmp_path))
    assert got.isascii() and got.startswith("limit ? 600 ? wait") and "≤" not in got
    L.write(str(tmp_path), "w " * 900)
    got = L.read(str(tmp_path))
    assert len(got) == 700 and got.endswith("...")
    L.write(str(tmp_path), "x" * 700)
    assert L.read(str(tmp_path)) == "x" * 700                          # exactly at the limit: not cut


def test_an_unreadable_file_is_read_as_far_as_it_goes(tmp_path):
    with open(L.path(str(tmp_path)), "wb") as fh:
        fh.write(b"STOPPED: bad byte \xff here")
    assert L.read(str(tmp_path)).startswith("STOPPED: bad byte")


# ================================================================ scripts/run_eval.py
def run_main(monkeypatch, tmp_path, raises, extra=()):
    def fake(*a, **k):
        raise raises
    monkeypatch.setattr(run_eval.harness, "run_experiment", fake)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--system", "b0", "--limit", "2", "--out", str(tmp_path), *extra])
    return run_eval.main


def test_a_run_that_stops_writes_the_reason_and_still_exits_2(tmp_path, monkeypatch, capsys):
    main = run_main(monkeypatch, tmp_path, harness.HarnessError("the rate limit asks for a wait of 80000 s (about 1333 min)."))
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 2
    assert "STOPPED: the rate limit asks for a wait of 80000 s" in capsys.readouterr().err
    assert L.read(str(tmp_path)) == "STOPPED: the rate limit asks for a wait of 80000 s (about 1333 min)."


def test_an_error_in_the_run_is_written_too_and_the_traceback_still_comes(tmp_path, monkeypatch):
    main = run_main(monkeypatch, tmp_path, KeyError("operations"))
    with pytest.raises(KeyError):
        main()
    assert L.read(str(tmp_path)) == "KeyError: 'operations'"


def test_a_new_run_removes_the_note_of_an_earlier_stop_before_it_starts(tmp_path, monkeypatch):
    L.write(str(tmp_path), "STOPPED: old")
    seen = []

    def fake(*a, **k):
        seen.append(L.read(str(tmp_path)))
        raise harness.HarnessError("new")
    monkeypatch.setattr(run_eval.harness, "run_experiment", fake)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--system", "b0", "--limit", "2", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        run_eval.main()
    assert seen == [""] and L.read(str(tmp_path)) == "STOPPED: new"


def test_a_plan_leaves_the_note_alone(tmp_path, monkeypatch):
    L.write(str(tmp_path), "STOPPED: old")
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--system", "b0", "--limit", "2", "--out", str(tmp_path), "--plan"])
    run_eval.main()
    assert L.read(str(tmp_path)) == "STOPPED: old"


def test_a_run_that_goes_through_leaves_no_note(tmp_path, monkeypatch, capsys):
    L.write(str(tmp_path), "STOPPED: old")
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--system", "rb", "--limit", "2", "--out", str(tmp_path), "--tag", "t"])
    run_eval.main()
    capsys.readouterr()
    assert L.read(str(tmp_path)) == ""


# ================================================================ scripts/g2.py prints it under the grid
def writing(results, text, only_system=None):
    """A `after` hook of the fake runner: a run that exits with a code other than 0 leaves `text` as run_eval does."""
    def after(system, code):
        if code != 0 and (only_system is None or system == only_system):
            L.write(str(results), text)
    return after


def test_the_reason_is_printed_under_the_grid_just_before_still_to_do(tmp_path):
    results = tmp_path / "results"
    runner = TG.Runner(results, codes={"n5_r1_s3": 2}, after=writing(results, "STOPPED: the rate limit asks for a wait of 80000 s."))
    code, _, say, _, _ = TG.go(tmp_path, runner=runner, yes=True)
    assert code == 1
    assert "Why: STOPPED: the rate limit asks for a wait of 80000 s." in say.lines[-2]
    assert say.lines[-1].startswith("Still to do:")
    assert say.text.index("Why:") > say.text.index("Model A")                       # the grid (its header names the columns) comes first
    assert "(a) stopped (exit code 2)" in say.lines[-2] and "run this command again" in say.lines[-2]


def test_without_a_note_it_gives_the_command_that_starts_only_that_run(tmp_path):
    results = tmp_path / "results"
    code, runner, say, _, _ = TG.go(tmp_path, runner=TG.Runner(results, codes={"n5_r1_s3": 2}), yes=True)
    assert code == 1
    last = say.lines[-2]
    assert "Why:" not in last and "start it alone: python scripts/run_eval.py --system n5_r1_s3" in last
    assert "--tag g_a" in last


def test_a_note_of_an_earlier_stop_is_not_taken_for_this_one(tmp_path):
    results = tmp_path / "results"
    L.write(str(results), "STOPPED: from last week")
    code, _, say, _, _ = TG.go(tmp_path, runner=TG.Runner(results, codes={"n5_r1_s3": 2}), yes=True)
    assert code == 1 and "from last week" not in say.text and "start it alone" in say.lines[-2]


def test_a_note_written_by_an_earlier_run_of_the_same_sequence_is_cleared_before_the_next(tmp_path):
    results = tmp_path / "results"
    runner = TG.Runner(results, codes={"b1": 2}, after=lambda system, code: L.write(str(results), "STOPPED: only b1") if system == "b0" else None)
    code, _, say, _, _ = TG.go(tmp_path, runner=runner, yes=True)
    assert code == 1 and runner.systems[-1] == "b1"
    assert "only b1" not in say.text and "start it alone" in say.lines[-2]


def test_a_long_reason_is_cut_and_stays_ascii(tmp_path):
    results = tmp_path / "results"
    runner = TG.Runner(results, codes={"b0": 2}, after=writing(results, "STOPPED: wait ≥ 5 min. " + "x" * 2000))
    _, _, say, _, _ = TG.go(tmp_path, runner=runner, yes=True)
    why = say.lines[-2].split("Why: ")[1].split("\n")[0]
    assert why.isascii() and why.startswith("STOPPED: wait ? 5 min.") and len(why) == 700


def test_a_run_that_goes_through_prints_no_reason(tmp_path):
    code, _, say, _, _ = TG.go(tmp_path, yes=True)
    assert code == 0 and "Why:" not in say.text and "start it alone" not in say.text
