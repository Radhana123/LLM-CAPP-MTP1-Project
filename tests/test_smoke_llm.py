"""Step 38: scripts/smoke_llm.py says in one line whether three planner calls worked (OK / PROBLEM) and can show the old cache without an API call.

Offline: the model is a fake transport that counts its calls. What is pinned: the verdict for every outcome (all read, cut off, API error,
not a route, cache miss in --old), its exit code, that --old never reaches the transport and uses the requests of before Step 37 (so that the
cache of that time answers), that the settings of today are shown, and the arguments.
"""
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capp import config, llm_client      # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "week2"))
import llm_planner                        # noqa: E402

PRIMARY = config.model("primary")
ROUTE = ["Facing", "Center Drilling", "Drilling", "Slot Milling", "Inspection"]


@pytest.fixture()
def smoke(monkeypatch, tmp_path):
    for name in ("CAPP_QWEN_REASONING", "CAPP_LLM_MODE", "CAPP_RUN_TAG", "CAPP_MODEL_PRIMARY"):
        monkeypatch.setenv(name, "x")          # registered: the teardown puts back what was there (also what --old sets in os.environ)
        monkeypatch.delenv(name)
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GROQ_API_KEY", "zz_notarealkey_0123456789abcdef")
    monkeypatch.setattr(llm_client, "_sleep", lambda s: None)
    llm_client.reset_transport()
    llm_client.reset_call_log()
    spec = importlib.util.spec_from_file_location("smoke_llm_cli", os.path.join(ROOT, "scripts", "smoke_llm.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod
    llm_client.reset_transport()
    llm_client.reset_call_log()


class Fake:
    def __init__(self, text=None, finish="stop", exc=None):
        self.text, self.finish, self.exc, self.seen = json.dumps(ROUTE) if text is None else text, finish, exc, []

    def __call__(self, model, messages, params):
        self.seen.append((model, dict(params)))
        if self.exc:
            raise self.exc
        return {"text": self.text, "prompt_tokens": 100, "completion_tokens": 20, "finish_reason": self.finish}


def run(smoke, capsys, transport, *args):
    llm_client.set_transport(transport)
    llm_client.reset_call_log()
    code = smoke.main(list(args))
    return code, capsys.readouterr().out


# ---- the verdicts --------------------------------------------------------------------------------------------------------------
def test_three_good_answers_are_ok_and_exit_0_with_the_settings_shown(smoke, capsys):
    t = Fake()
    code, out = run(smoke, capsys, t)
    assert code == 0 and len(t.seen) == 3
    assert out.rstrip().split("\n")[-1] == "RESULT: OK  3 of 3 answers are a route of known operations and none was cut off"
    assert "settings  : reasoning_effort=none" in out and f"model     : {PRIMARY}" in out
    assert out.count("sample ") == 3 and "cached=False" in out and "finish=stop" in out


def test_a_second_run_is_all_from_the_cache_and_says_so(smoke, capsys):
    run(smoke, capsys, Fake())
    t = Fake()
    code, out = run(smoke, capsys, t)
    assert code == 0 and t.seen == [] and "cached=True" in out
    assert out.rstrip().split("\n")[-1].endswith("none was cut off (all from the cache, no API call was made)")


def test_cut_off_answers_are_a_problem_even_when_the_cut_text_still_parses(smoke, capsys):
    code, out = run(smoke, capsys, Fake(finish="length"))
    assert code == 1
    assert out.rstrip().split("\n")[-1].startswith("RESULT: PROBLEM  3 of 3 answers were cut off at the token limit (finish_reason length)")
    code, out = run(smoke, capsys, Fake(text='["Facing", "Cen', finish="length"), "--samples", "2")          # a different run tag is not needed: other text, new samples
    assert code == 1


def test_an_api_error_is_a_problem_with_the_message_and_a_hint_for_reasoning_effort(smoke, capsys):
    code, out = run(smoke, capsys, Fake(exc=ValueError("400 invalid reasoning_effort value")))
    last = out.rstrip().split("\n")[-1]
    assert code == 1 and last.startswith("RESULT: PROBLEM  3 of 3 calls failed with an API error: API Error:")
    assert "reasoning_effort" in last and last.endswith("send this line to Claude.")
    code, out = run(smoke, capsys, Fake(exc=ValueError("503 service unavailable")), "--samples", "1")
    last = out.rstrip().split("\n")[-1]
    assert code == 1 and "1 of 1 calls failed" in last and "Claude" not in last


def test_answers_that_are_not_a_route_are_a_problem_and_the_raw_text_is_shown(smoke, capsys):
    code, out = run(smoke, capsys, Fake(text="I cannot help with that."))
    assert code == 1 and "   raw  : I cannot help with that." in out
    assert out.rstrip().split("\n")[-1] == "RESULT: PROBLEM  3 of 3 answers are not a route of known operations (the raw text is printed above)"


def test_no_key_stops_before_any_call_with_exit_2(smoke, capsys, monkeypatch):
    monkeypatch.setattr(config, "get_api_key", lambda: None)
    t = Fake()
    code, out = run(smoke, capsys, t)
    assert code == 2 and t.seen == [] and "GROQ_API_KEY not found" in out and "RESULT" not in out


def test_the_number_of_samples_is_an_argument_and_zero_is_refused(smoke, capsys):
    t = Fake()
    code, out = run(smoke, capsys, t, "--samples", "5")
    assert code == 0 and len(t.seen) == 5 and "5 of 5 answers" in out
    assert smoke.main(["--samples", "0"]) == 2
    assert "--samples must be 1 or more" in capsys.readouterr().err


# ---- --old ---------------------------------------------------------------------------------------------------------------------
def test_old_shows_the_old_cached_answers_without_any_api_call(smoke, capsys, monkeypatch):
    # an old run: omit sends the requests of before Step 37, so the cache is filled the way it was then
    monkeypatch.setenv("CAPP_QWEN_REASONING", "omit")
    t = Fake(text="<think>let me think about it", finish="length")
    code, out = run(smoke, capsys, t)
    assert code == 1 and len(t.seen) == 3 and all("reasoning_effort" not in p for _, p in t.seen)
    monkeypatch.delenv("CAPP_QWEN_REASONING")                         # today's settings again
    never = Fake()
    code, out = run(smoke, capsys, never, "--old")
    assert never.seen == [] and code == 1                              # nothing was sent
    assert "settings  : nothing extra is sent" in out and "mode      : replay" in out and "cached=True" in out and "finish=length" in out
    assert out.rstrip().split("\n")[-1].startswith("RESULT: PROBLEM  3 of 3 answers were cut off at the token limit")
    assert os.environ["CAPP_QWEN_REASONING"] == "omit" and os.environ["CAPP_LLM_MODE"] == "replay"


def test_old_with_a_good_old_cache_is_ok(smoke, capsys, monkeypatch):
    monkeypatch.setenv("CAPP_QWEN_REASONING", "omit")
    run(smoke, capsys, Fake())
    monkeypatch.delenv("CAPP_QWEN_REASONING")
    code, out = run(smoke, capsys, Fake(), "--old")
    assert code == 0 and out.rstrip().split("\n")[-1].endswith("(all from the cache, no API call was made)")


def test_old_with_nothing_in_the_cache_says_so_and_does_not_call_the_api(smoke, capsys):
    t = Fake()
    code, out = run(smoke, capsys, t, "--old")
    assert code == 1 and t.seen == []
    assert out.rstrip().split("\n")[-1].startswith("RESULT: PROBLEM  none of the old calls is in the cache")


def test_old_with_some_of_the_calls_cached_names_the_others(smoke, capsys, monkeypatch):
    monkeypatch.setenv("CAPP_QWEN_REASONING", "omit")
    run(smoke, capsys, Fake(), "--samples", "2")
    monkeypatch.delenv("CAPP_QWEN_REASONING")
    code, out = run(smoke, capsys, Fake(), "--old", "--samples", "3")
    assert code == 1 and out.rstrip().split("\n")[-1].startswith("RESULT: PROBLEM  1 of 3 calls are not in the cache")


def test_old_does_not_need_a_key(smoke, capsys, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.setattr(config, "get_api_key", lambda: None)
    code, out = run(smoke, capsys, Fake(), "--old")
    assert "GROQ_API_KEY not found" not in out and "RESULT: PROBLEM  none of the old calls" in out


# ---- verdict() on its own -----------------------------------------------------------------------------------------------------
def test_verdict_prefers_an_api_error_over_cut_off_over_unreadable(smoke):
    ok = {"success": True, "steps": ROUTE, "raw_response": "[]"}
    bad = {"success": False, "steps": [], "raw_response": "nonsense"}
    err = {"success": False, "steps": [], "raw_response": "API Error: HTTP 429: slow down"}
    summ = {"truncated": 2, "cache_hits": 0, "calls": 3}
    assert smoke.verdict([ok, bad, err], summ)[1].startswith("RESULT: PROBLEM  1 of 3 calls failed with an API error")
    assert smoke.verdict([ok, bad, ok], summ)[1].startswith("RESULT: PROBLEM  2 of 3 answers were cut off")
    assert smoke.verdict([ok, bad, ok], {"truncated": 0, "cache_hits": 0, "calls": 3})[1].startswith("RESULT: PROBLEM  1 of 3 answers are not a route")
    assert smoke.verdict([ok, ok], {"truncated": 0, "cache_hits": 1, "calls": 2}) == (0, "RESULT: OK  2 of 2 answers are a route of known operations and none was cut off")


def test_a_text_the_planner_splits_into_lines_is_not_a_route_unless_every_line_is_a_known_operation(smoke, capsys):
    ok = {"success": True, "steps": ROUTE}
    assert smoke.is_route(ok)
    assert not smoke.is_route({"success": True, "steps": ROUTE + ["Hovering"]})                  # one invented name is enough
    assert not smoke.is_route({"success": True, "steps": ["I cannot help with that."]})
    assert not smoke.is_route({"success": True, "steps": []}) and not smoke.is_route({"success": False, "steps": ROUTE})
    assert smoke.is_route({"success": True, "steps": ["Facing"]}, valid={"Facing"})
    assert not smoke.is_route(ok, valid={"Facing"})
    code, out = run(smoke, capsys, Fake(text='["Facing", "Hovering", "Inspection"]'))
    assert code == 1 and "   raw  : [" in out and "3 of 3 answers are not a route of known operations" in out.rstrip().split("\n")[-1]


def test_the_known_operations_are_those_of_the_prompt_plus_inspection(smoke):
    valid = smoke._valid_ops()
    assert "Inspection" in valid and "Facing" in valid and "Drilling" in valid and "Slot Milling" in valid
    assert "Hovering" not in valid and len(valid) == len(set(llm_planner._get_valid_operations_list().split(", "))) + 1
    assert set(ROUTE) <= valid


def test_a_model_answer_that_starts_with_api_error_is_not_an_api_error(smoke, capsys):
    code, out = run(smoke, capsys, Fake(text="API Error: this is what the model said"))
    assert code == 1 and out.rstrip().split("\n")[-1].startswith("RESULT: PROBLEM  3 of 3 answers are not a route of known operations")
