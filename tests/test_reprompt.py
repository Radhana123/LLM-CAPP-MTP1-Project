import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import config, llm_client    # noqa: E402
from capp import reprompt as RP        # noqa: E402
from capp import verify as V           # noqa: E402
import llm_planner                     # noqa: E402

BAD = ["Facing", "Reaming", "Drilling", "Deburring", "Inspection"]
FEATS = ["Reamed_Hole"]


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CAPP_LLM_MODE", raising=False)
    monkeypatch.delenv("CAPP_RUN_TAG", raising=False)
    llm_client.reset_transport()
    llm_client.reset_call_log()
    yield
    llm_client.reset_transport()


def _viol():
    return V.check_route(BAD, FEATS).hard_violations


def _capture():
    seen = []

    def transport(model, messages, params):
        seen.append({"messages": messages, "params": params})
        return {"text": json.dumps(["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"]),
                "prompt_tokens": 400, "completion_tokens": 30, "finish_reason": "stop"}
    llm_client.set_transport(transport)
    return seen


def test_first_two_messages_are_exactly_what_the_v1_planner_sends():
    seen = _capture()
    llm_planner.generate_process_plan("Steel", FEATS, "0.02mm", 100, sample_id=0)
    RP.reprompt("Steel", FEATS, "0.02mm", 100, sample_id=0, previous_steps=BAD, violations=_viol())
    assert seen[0]["messages"] == seen[1]["messages"][:2]


def test_conversation_carries_the_rejected_plan_and_every_violation():
    seen = _capture()
    RP.reprompt("Steel", FEATS, "0.02mm", 100, previous_steps=BAD, violations=_viol())
    msgs = seen[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert json.loads(msgs[2]["content"]) == BAD
    for v in _viol():
        assert v.message in msgs[3]["content"]
    assert "JSON array" in msgs[3]["content"]


def test_feedback_is_capped_and_says_how_many_were_left_out():
    many = [V.Violation("order", f"problem {i}", ops=("a", "b")) for i in range(11)]
    text = RP.feedback_text(many)
    assert text.count("\n- problem") == RP.MAX_LISTED and "- problem 7" in text and "problem 8" not in text
    assert "and 3 more problem(s)" in text
    assert "more problem" not in RP.feedback_text(many[:3])
    RP.feedback_text(iter(many))               # a generator works too


def test_greedy_decoding_and_result_shape():
    seen = _capture()
    res = RP.reprompt("Steel", FEATS, "0.02mm", 100, previous_steps=BAD, violations=_viol())
    assert seen[0]["params"]["temperature"] == config.REPAIR_TEMPERATURE == 0.0
    assert seen[0]["params"]["max_tokens"] == config.REPAIR_MAX_TOKENS
    assert res["success"] and res["steps"][0] == "Facing" and res["meta"]["prompt_tokens"] == 400 and res["meta"]["cached"] is False


def test_cache_hit_on_the_same_report_and_new_call_for_a_different_one():
    seen = _capture()
    kw = dict(previous_steps=BAD, violations=_viol())
    RP.reprompt("Steel", FEATS, "0.02mm", 100, **kw)
    again = RP.reprompt("Steel", FEATS, "0.02mm", 100, **kw)
    assert len(seen) == 1 and again["meta"]["cached"] is True
    RP.reprompt("Steel", FEATS, "0.02mm", 100, previous_steps=BAD[:-1], violations=_viol())
    assert len(seen) == 2
    RP.reprompt("Steel", FEATS, "0.02mm", 100, sample_id=1, **kw)             # another sample of the same part: own entry
    assert len(seen) == 3


def test_missing_key_and_api_errors_are_reported_like_the_planner(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    res = RP.reprompt("Steel", FEATS, "0.02mm", 100, previous_steps=BAD, violations=_viol())
    assert res["success"] is False and res["raw_response"] == "GROQ_API_KEY not configured"

    def boom(model, messages, params):
        raise RuntimeError("503")
    llm_client.set_transport(boom)
    res = RP.reprompt("Steel", FEATS, "0.02mm", 100, previous_steps=BAD, violations=_viol())
    assert res["success"] is False and res["raw_response"].startswith("API Error")


def test_unparseable_answer_is_a_failure_not_a_crash():
    llm_client.set_transport(lambda m, msgs, p: {"text": "", "prompt_tokens": 1, "completion_tokens": 0, "finish_reason": "stop"})
    res = RP.reprompt("Steel", FEATS, "0.02mm", 100, previous_steps=BAD, violations=_viol())
    assert res["success"] is False and res["steps"] == []
