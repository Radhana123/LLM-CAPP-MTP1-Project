"""Step 35: reasoning models get low reasoning effort and room for the answer (capp/config.py model_params / apply_model_params).

Why: with max_tokens = 200 for every model, a run of the small model (+ms, openai/gpt-oss-20b) had 96 of 100 answers cut off, because the
hidden reasoning of a gpt-oss model is counted in max_tokens. What is pinned here: the models that are changed and only those (the primary
model's requests, and with them its cache, stay byte for byte as they were), what is changed, that the cache key follows what is really sent,
and that every kind of call (sample, re-prompt, two-step, constrained, cascade) gets it, because they all go through llm_client.chat.
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
    for m in ("qwen/qwen3.8-27b", "llama-3.3-70b-versatile", "openai/other", "openai/gpt-4o", "gpt-oss-20b", "", "x/openai/gpt-oss-20b"):
        assert config.model_params(m) == {}


def test_the_default_roles_are_what_the_settings_were_written_for():
    assert PRIMARY == "qwen/qwen3.8-27b" and SMALL.startswith("openai/gpt-oss") and LARGE.startswith("openai/gpt-oss")
    assert config.model_params(PRIMARY) == {} and config.model_params(SMALL) and config.model_params(LARGE)


def test_a_model_that_needs_nothing_gets_its_own_dict_back():
    params = {"temperature": 0.3, "max_tokens": 200}
    assert config.apply_model_params(PRIMARY, params) is params and params == {"temperature": 0.3, "max_tokens": 200}


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


def test_the_request_to_the_primary_model_is_exactly_what_the_caller_asked():
    spy = Spy()
    llm_client.set_transport(spy)
    llm_client.chat(MSGS, temperature=0.3, max_tokens=200)
    llm_client.chat(MSGS, model=PRIMARY, sample_id=1, temperature=0.0, max_tokens=700)
    assert spy.seen == [(PRIMARY, {"temperature": 0.3, "max_tokens": 200}), (PRIMARY, {"temperature": 0.0, "max_tokens": 700})]


def test_the_cache_key_of_the_primary_model_is_unchanged_so_the_old_cache_still_answers(tmp_path):
    spy = Spy()
    llm_client.set_transport(spy)
    llm_client.chat(MSGS, sample_id=3, temperature=0.3, max_tokens=200)
    old_key = llm_client.make_key(PRIMARY, MSGS, {"temperature": 0.3, "max_tokens": 200}, 3)         # how a run before this step made it
    assert (tmp_path / "cache" / old_key[:2] / (old_key + ".json")).exists()
    again = llm_client.chat(MSGS, sample_id=3, temperature=0.3, max_tokens=200)
    assert again.cached and len(spy.seen) == 1


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


def test_the_primary_model_runs_are_sent_as_before():
    spy, _ = run_with_spy("n2_r2_s1")
    assert spy.seen and all(m == PRIMARY and p["max_tokens"] == 200 and "reasoning_effort" not in p for m, p in spy.seen)


class Wrong(Spy):
    """An answer that parses and is wrong, so that repair and the cascade's second model are needed."""

    def __call__(self, model, messages, params):
        super().__call__(model, messages, params)
        return {"text": json.dumps(["Facing", "Inspection"]), "prompt_tokens": 10, "completion_tokens": 5, "finish_reason": "stop"}


def test_a_cascade_changes_the_two_gpt_oss_tiers_and_nothing_else():
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
    assert out["run_info"]["model_params"] == {PRIMARY: {}}
    assert H.run_experiment("rb", parts(1), tag="mp", progress=None, write=False)["run_info"]["model_params"] is None
