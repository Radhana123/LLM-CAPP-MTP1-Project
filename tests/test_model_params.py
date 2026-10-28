"""Steps 35 and 37: reasoning models get a reasoning setting and room for the answer (capp/config.py model_params / apply_model_params).

Why: with max_tokens = 200 for every model, a run of the small model (+ms, openai/gpt-oss-20b) had 96 of 100 answers cut off, because the
hidden reasoning of a gpt-oss model is counted in max_tokens (Step 35). The primary model, qwen3.8-27b, thinks by default too (Groq: thinking is
its default mode, "none" is the instruct mode), so since Step 37 it is sent reasoning_effort "none"; CAPP_QWEN_REASONING can change that, and
"omit" gives back the requests (and the cache) of before. What is pinned here: the models that are changed and only those, what is changed,
that the cache key follows what is really sent, that every kind of call (sample, re-prompt, two-step, constrained, cascade) gets it, because
they all go through llm_client.chat, and how the settings reach the API.
"""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capp import config, dataset as D, harness as H, llm_client             # noqa: E402

PRIMARY, SMALL, LARGE = config.model("primary"), config.model("small"), config.model("large")
ROUTE = ["Facing", "Center Drilling", "Drilling", "Chamfering", "Inspection"]


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.delenv("CAPP_QWEN_REASONING", raising=False)
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.delenv("CAPP_LLM_MODE", raising=False)
    monkeypatch.delenv("CAPP_RUN_TAG", raising=False)
    monkeypatch.setattr(llm_client, "_sleep", lambda s: None)
    llm_client.reset_transport()
    llm_client.reset_call_log()
    yield
    llm_client.reset_transport()


# ---- which models, and what ---------------------------------------------------------------------------------------------
def test_the_gpt_oss_models_get_low_effort_and_a_token_floor_and_the_others_get_nothing():
    for m in ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "openai/gpt-oss-whatever-next"):
        assert config.model_params(m) == {"reasoning_effort": "low", "max_tokens_at_least": 2048}
    for m in ("llama-3.3-70b-versatile", "openai/other", "openai/gpt-4o", "gpt-oss-20b", "", "x/openai/gpt-oss-20b", "x/qwen/qwen3.8-27b", "qwen/qwen3-32b",
              "qwen/qwen3.7-27b", "qwen3.8-27b"):
        assert config.model_params(m) == {}


def test_the_qwen_models_get_no_thinking_and_no_floor_by_default():
    for m in ("qwen/qwen3.8-27b", "qwen/qwen3.8-72b", "qwen/qwen3.8-whatever-next"):
        assert config.model_params(m) == {"reasoning_effort": "none"}


def test_the_default_roles_are_what_the_settings_were_written_for():
    assert PRIMARY == "qwen/qwen3.8-27b" and SMALL.startswith("openai/gpt-oss") and LARGE.startswith("openai/gpt-oss")
    assert config.model_params(PRIMARY) == {"reasoning_effort": "none"} and config.model_params(SMALL) and config.model_params(LARGE)


def test_a_model_that_needs_nothing_gets_its_own_dict_back():
    params = {"temperature": 0.3, "max_tokens": 200}
    assert config.apply_model_params("llama-3.3-70b-versatile", params) is params and params == {"temperature": 0.3, "max_tokens": 200}
    assert config.apply_model_params("qwen/qwen3.8-27b", params) is not params and params == {"temperature": 0.3, "max_tokens": 200}      # a copy, untouched


def test_the_primary_model_keeps_the_limit_of_the_caller_and_gets_the_setting():
    assert config.apply_model_params(PRIMARY, {"temperature": 0.3, "max_tokens": 200}) == {"temperature": 0.3, "max_tokens": 200, "reasoning_effort": "none"}
    assert config.apply_model_params(PRIMARY, {"max_tokens": 1})["max_tokens"] == 1                    # no floor: the answer is short without thinking
    assert config.apply_model_params(PRIMARY, {"temperature": 0.0}) == {"temperature": 0.0, "reasoning_effort": "none"}
    assert config.apply_model_params(PRIMARY, {"reasoning_effort": "high", "max_tokens": 10}) == {"reasoning_effort": "high", "max_tokens": 10}


# ---- CAPP_QWEN_REASONING ----------------------------------------------------------------------------------------------------
def test_the_qwen_setting_comes_from_the_environment_with_none_as_the_default(monkeypatch):
    assert config.qwen_reasoning() == "none" and config.QWEN_REASONING_DEFAULT == "none"
    for blank in ("", "   "):
        monkeypatch.setenv("CAPP_QWEN_REASONING", blank)
        assert config.qwen_reasoning() == "none"
    monkeypatch.setenv("CAPP_QWEN_REASONING", "  Low ")
    assert config.qwen_reasoning() == "low"


def test_omit_sends_nothing_so_the_requests_and_the_cache_of_before_step_37_are_back(monkeypatch):
    monkeypatch.setenv("CAPP_QWEN_REASONING", "omit")
    assert config.model_params(PRIMARY) == {}
    params = {"temperature": 0.3, "max_tokens": 200}
    assert config.apply_model_params(PRIMARY, params) is params


@pytest.mark.parametrize("value", ["default", "low", "medium", "high"])
def test_turning_thinking_on_sends_the_value_and_a_floor_for_the_limit(monkeypatch, value):
    monkeypatch.setenv("CAPP_QWEN_REASONING", value)
    assert config.model_params(PRIMARY) == {"reasoning_effort": value, "max_tokens_at_least": 4096}
    assert config.apply_model_params(PRIMARY, {"temperature": 0.3, "max_tokens": 200}) == {"temperature": 0.3, "max_tokens": 4096, "reasoning_effort": value}
    assert config.apply_model_params(PRIMARY, {"max_tokens": 9000})["max_tokens"] == 9000
    assert config.apply_model_params(PRIMARY, {"max_tokens": 4095})["max_tokens"] == 4096
    assert config.apply_model_params(PRIMARY, {"temperature": 0.0}) == {"temperature": 0.0, "reasoning_effort": value}


def test_a_wrong_value_is_refused_in_words_and_does_not_touch_the_gpt_oss_models(monkeypatch):
    monkeypatch.setenv("CAPP_QWEN_REASONING", "xhigh")
    with pytest.raises(ValueError, match=r"CAPP_QWEN_REASONING='xhigh', expected one of \('none', 'default', 'low', 'medium', 'high', 'omit'\)"):
        config.model_params(PRIMARY)
    with pytest.raises(ValueError, match="CAPP_QWEN_REASONING"):
        llm_client.chat(MSGS, temperature=0.3, max_tokens=200)
    assert config.model_params(SMALL) == {"reasoning_effort": "low", "max_tokens_at_least": 2048}
    assert config.model_params("llama-3.3-70b-versatile") == {}


def test_the_qwen_floor_is_one_place_and_can_be_changed(monkeypatch):
    monkeypatch.setenv("CAPP_QWEN_REASONING", "low")
    monkeypatch.setattr(config, "QWEN_THINKING_MIN_MAX_TOKENS", 8192)
    assert config.apply_model_params(PRIMARY, {"max_tokens": 200})["max_tokens"] == 8192
    monkeypatch.setattr(config, "QWEN_PREFIXES", ())
    assert config.model_params(PRIMARY) == {}


def test_a_reasoning_model_gets_the_effort_and_a_raised_limit_in_a_copy():
    params = {"temperature": 0.3, "max_tokens": 200}
    out = config.apply_model_params(SMALL, params)
    assert out == {"temperature": 0.3, "max_tokens": 2048, "reasoning_effort": "low"}
    assert params == {"temperature": 0.3, "max_tokens": 200}                          # the caller's dict is not touched
    assert out is not params


def test_a_limit_above_the_floor_stays_and_one_below_is_raised():
    assert config.apply_model_params(SMALL, {"max_tokens": 5000})["max_tokens"] == 5000
    assert config.apply_model_params(SMALL, {"max_tokens": 2048})["max_tokens"] == 2048
    assert config.apply_model_params(SMALL, {"max_tokens": 2047})["max_tokens"] == 2048
    assert config.apply_model_params(SMALL, {"max_tokens": 1})["max_tokens"] == 2048


def test_no_limit_from_the_caller_is_not_made_up():
    out = config.apply_model_params(SMALL, {"temperature": 0.0})
    assert out == {"temperature": 0.0, "reasoning_effort": "low"}


def test_an_effort_the_caller_chose_stays():
    assert config.apply_model_params(SMALL, {"reasoning_effort": "high", "max_tokens": 10})["reasoning_effort"] == "high"


def test_other_parameters_are_carried_over_as_they_are():
    rf = {"type": "json_schema", "json_schema": {"name": "x"}}
    out = config.apply_model_params(LARGE, {"temperature": 0.0, "max_tokens": 300, "response_format": rf})
    assert out["response_format"] == rf and out["temperature"] == 0.0 and out["max_tokens"] == 2048


def test_the_settings_can_be_changed_in_one_place(monkeypatch):
    monkeypatch.setattr(config, "REASONING_EFFORT", "medium")
    monkeypatch.setattr(config, "REASONING_MIN_MAX_TOKENS", 4096)
    assert config.apply_model_params(SMALL, {"max_tokens": 200}) == {"max_tokens": 4096, "reasoning_effort": "medium"}
    monkeypatch.setattr(config, "REASONING_MODEL_PREFIXES", ())
    assert config.model_params(SMALL) == {}


# ---- llm_client.chat: what is sent and what is cached ---------------------------------------------------------------------
class Spy:
    def __init__(self):
        self.seen = []

    def __call__(self, model, messages, params):
        self.seen.append((model, dict(params)))
        return {"text": json.dumps(ROUTE), "prompt_tokens": 10, "completion_tokens": 5, "finish_reason": "stop"}


MSGS = [{"role": "user", "content": "plan"}]


def test_the_request_to_a_reasoning_model_carries_the_settings():
    spy = Spy()
    llm_client.set_transport(spy)
    llm_client.chat(MSGS, model=SMALL, temperature=0.3, max_tokens=200)
    assert spy.seen == [(SMALL, {"temperature": 0.3, "max_tokens": 2048, "reasoning_effort": "low"})]


def test_the_request_to_the_primary_model_is_what_the_caller_asked_plus_no_thinking():
    spy = Spy()
    llm_client.set_transport(spy)
    llm_client.chat(MSGS, temperature=0.3, max_tokens=200)
    llm_client.chat(MSGS, model=PRIMARY, sample_id=1, temperature=0.0, max_tokens=700)
    assert spy.seen == [(PRIMARY, {"temperature": 0.3, "max_tokens": 200, "reasoning_effort": "none"}),
                        (PRIMARY, {"temperature": 0.0, "max_tokens": 700, "reasoning_effort": "none"})]


def test_with_omit_the_cache_key_of_the_primary_model_is_the_one_of_before_step_37(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPP_QWEN_REASONING", "omit")
    spy = Spy()
    llm_client.set_transport(spy)
    llm_client.chat(MSGS, sample_id=3, temperature=0.3, max_tokens=200)
    old_key = llm_client.make_key(PRIMARY, MSGS, {"temperature": 0.3, "max_tokens": 200}, 3)         # how a run before Step 37 made it
    assert (tmp_path / "cache" / old_key[:2] / (old_key + ".json")).exists()
    again = llm_client.chat(MSGS, sample_id=3, temperature=0.3, max_tokens=200)
    assert again.cached and len(spy.seen) == 1 and spy.seen == [(PRIMARY, {"temperature": 0.3, "max_tokens": 200})]


def test_by_default_the_primary_model_has_a_new_key_so_an_answer_made_while_it_was_thinking_is_not_served(tmp_path):
    old_key = llm_client.make_key(PRIMARY, MSGS, {"temperature": 0.3, "max_tokens": 200}, 3)
    cache = tmp_path / "cache" / old_key[:2]
    cache.mkdir(parents=True)
    (cache / (old_key + ".json")).write_text(json.dumps({"text": "<think>", "model": PRIMARY, "prompt_tokens": 1, "completion_tokens": 200,
                                                          "finish_reason": "length", "latency_s": 0.0}), encoding="utf-8")
    spy = Spy()
    llm_client.set_transport(spy)
    got = llm_client.chat(MSGS, sample_id=3, temperature=0.3, max_tokens=200)
    assert not got.cached and got.finish_reason == "stop" and len(spy.seen) == 1
    new_key = llm_client.make_key(PRIMARY, MSGS, {"temperature": 0.3, "max_tokens": 200, "reasoning_effort": "none"}, 3)
    assert new_key != old_key and (tmp_path / "cache" / new_key[:2] / (new_key + ".json")).exists()


def test_the_cache_key_of_a_reasoning_model_follows_what_is_sent(tmp_path):
    spy = Spy()
    llm_client.set_transport(spy)
    llm_client.chat(MSGS, model=SMALL, sample_id=3, temperature=0.3, max_tokens=200)
    sent = {"temperature": 0.3, "max_tokens": 2048, "reasoning_effort": "low"}
    new_key = llm_client.make_key(SMALL, MSGS, sent, 3)
    old_key = llm_client.make_key(SMALL, MSGS, {"temperature": 0.3, "max_tokens": 200}, 3)
    assert (tmp_path / "cache" / new_key[:2] / (new_key + ".json")).exists()
    assert not (tmp_path / "cache" / old_key[:2] / (old_key + ".json")).exists()


def test_an_answer_cut_off_under_the_old_limit_is_not_served_to_the_new_one():
    class Cut:
        def __call__(self, model, messages, params):
            full = params.get("max_tokens", 0) >= 2048
            return {"text": json.dumps(ROUTE) if full else "", "prompt_tokens": 10, "completion_tokens": params.get("max_tokens", 0),
                    "finish_reason": "stop" if full else "length"}
    llm_client.set_transport(Cut())
    r = llm_client.chat(MSGS, model=SMALL, temperature=0.3, max_tokens=200)
    assert r.finish_reason == "stop" and json.loads(r.text) == ROUTE


# ---- every kind of call goes through it ------------------------------------------------------------------------------------
def parts(n=2):
    base = D.load_dataset("dev_v2")[0]
    return [dict(base, part_id=f"M{i}", features=["Hole", "Chamfer"], batch_size=100 + i) for i in range(n)]


def run_with_spy(system, n=2):
    spy = Spy()
    llm_client.set_transport(spy)
    out = H.run_experiment(system, parts(n), tag="mp", progress=None, write=False)
    return spy, out


def test_sample_calls_of_the_small_model_carry_the_settings():
    spy, _ = run_with_spy("n2_r0_s1+ms")
    assert spy.seen and all(m == SMALL and p["max_tokens"] == 2048 and p["reasoning_effort"] == "low" for m, p in spy.seen)


def test_the_large_model_too():
    spy, _ = run_with_spy("n2_r0_s1+ml")
    assert spy.seen and all(m == LARGE and p["max_tokens"] == 2048 and p["reasoning_effort"] == "low" for m, p in spy.seen)


def test_the_primary_model_runs_are_sent_with_their_own_limit_and_no_thinking():
    spy, _ = run_with_spy("n2_r2_s1")
    assert spy.seen and all(m == PRIMARY and p["max_tokens"] == 200 and p["reasoning_effort"] == "none" for m, p in spy.seen)


def test_the_primary_model_runs_with_thinking_on_get_room_for_it_in_every_call(monkeypatch):
    monkeypatch.setenv("CAPP_QWEN_REASONING", "low")
    spy, _ = run_with_spy("n2_r2_s1")
    assert spy.seen and all(m == PRIMARY and p["max_tokens"] == 4096 and p["reasoning_effort"] == "low" for m, p in spy.seen)


class Wrong(Spy):
    """An answer that parses and is wrong, so that repair and the cascade's second model are needed."""

    def __call__(self, model, messages, params):
        super().__call__(model, messages, params)
        return {"text": json.dumps(["Facing", "Inspection"]), "prompt_tokens": 10, "completion_tokens": 5, "finish_reason": "stop"}


def test_a_cascade_changes_the_two_gpt_oss_tiers_and_nothing_else():            # (the primary model is not in a cascade)
    spy = Wrong()
    llm_client.set_transport(spy)
    H.run_experiment("n2_r2_s1+cas2", parts(2), tag="mp", progress=None, write=False)
    models = {m for m, _ in spy.seen}
    assert models == {SMALL, LARGE}
    assert all(p["max_tokens"] == 2048 and p["reasoning_effort"] == "low" for _, p in spy.seen)


def test_re_prompts_carry_the_settings_too():
    spy = Wrong()
    llm_client.set_transport(spy)
    H.run_experiment("n1_r2_s1+ms", parts(1), tag="mp", progress=None, write=False)
    assert len(spy.seen) == 2 and all(p["max_tokens"] == 2048 and p["reasoning_effort"] == "low" for _, p in spy.seen)      # sample + R2


def test_the_constrained_planner_carries_them():
    class Pick(Spy):
        def __call__(self, model, messages, params):
            super().__call__(model, messages, params)
            return {"text": "1", "prompt_tokens": 10, "completion_tokens": 1, "finish_reason": "stop"}
    spy = Pick()
    llm_client.set_transport(spy)
    H.run_experiment("n1_r0_s1+cg1+ms", D.load_dataset("dev_v2")[:3], tag="mp", progress=None, write=False)      # real parts: steps with a choice
    assert spy.seen and all(m == SMALL and p["max_tokens"] == 2048 and p["reasoning_effort"] == "low" for m, p in spy.seen)


def test_the_two_step_planner_keeps_its_structured_output_and_gets_the_settings():
    class TwoStep(Spy):
        def __call__(self, model, messages, params):
            super().__call__(model, messages, params)
            if "response_format" in params:
                return {"text": json.dumps({"operations": ROUTE}), "prompt_tokens": 10, "completion_tokens": 5, "finish_reason": "stop"}
            return {"text": "FINAL ORDER: " + " > ".join(ROUTE), "prompt_tokens": 10, "completion_tokens": 5, "finish_reason": "stop"}
    spy = TwoStep()
    llm_client.set_transport(spy)
    H.run_experiment("n1_r0_s1+ts+ms", parts(1), tag="mp", progress=None, write=False)
    assert len(spy.seen) == 2
    reason, formal = spy.seen[0][1], spy.seen[1][1]
    assert reason["max_tokens"] == 2048 and formal["max_tokens"] == 2048 and "response_format" in formal and "response_format" not in reason
    assert reason["reasoning_effort"] == formal["reasoning_effort"] == "low"


# ---- the run records it ---------------------------------------------------------------------------------------------------------
def test_the_run_records_the_settings_of_the_models_it_used():
    _, out = run_with_spy("n2_r0_s1+ms")
    assert out["run_info"]["model_params"] == {SMALL: {"reasoning_effort": "low", "max_tokens_at_least": 2048}}
    _, out = run_with_spy("n2_r0_s1+cas1")
    assert out["run_info"]["model_params"] == {SMALL: config.model_params(SMALL), LARGE: config.model_params(LARGE)}
    _, out = run_with_spy("n2_r0_s1")
    assert out["run_info"]["model_params"] == {PRIMARY: {"reasoning_effort": "none"}}
    assert H.run_experiment("rb", parts(1), tag="mp", progress=None, write=False)["run_info"]["model_params"] is None


def test_the_run_records_omit_as_no_settings(monkeypatch):
    monkeypatch.setenv("CAPP_QWEN_REASONING", "omit")
    _, out = run_with_spy("n2_r0_s1")
    assert out["run_info"]["model_params"] == {PRIMARY: {}}


# ---- how the setting reaches the API ----------------------------------------------------------------------------------------------
class FakeGroq:
    """Stands in for groq.Groq: records the arguments of chat.completions.create."""

    def __init__(self):
        self.calls = []
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kw):
        self.calls.append(kw)
        msg = type("M", (), {"content": "[]"})()
        choice = type("Ch", (), {"message": msg, "finish_reason": "stop"})()
        usage = type("U", (), {"prompt_tokens": 3, "completion_tokens": 2})()
        return type("R", (), {"choices": [choice], "usage": usage})()


def test_reasoning_effort_goes_to_the_api_in_the_body_and_the_other_arguments_as_they_are(monkeypatch):
    fake = FakeGroq()
    monkeypatch.setattr(llm_client, "_groq_client", fake)
    monkeypatch.setenv("GROQ_API_KEY", "zz_notarealkey_0123456789abcdef")
    monkeypatch.setattr(config, "get_api_key", lambda: "zz_notarealkey_0123456789abcdef")
    params = {"temperature": 0.3, "max_tokens": 200, "reasoning_effort": "none"}
    out = llm_client._groq_transport(PRIMARY, MSGS, params)
    assert fake.calls == [{"model": PRIMARY, "messages": MSGS, "temperature": 0.3, "max_tokens": 200, "extra_body": {"reasoning_effort": "none"}}]
    assert params == {"temperature": 0.3, "max_tokens": 200, "reasoning_effort": "none"}              # the caller's dict (it is the cache key) is not touched
    assert out == {"text": "[]", "prompt_tokens": 3, "completion_tokens": 2, "finish_reason": "stop"}
    llm_client._groq_transport(PRIMARY, MSGS, {"temperature": 0.0})                                   # nothing to move: no extra_body
    assert fake.calls[1] == {"model": PRIMARY, "messages": MSGS, "temperature": 0.0}
