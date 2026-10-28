import json
import os
import re
import subprocess
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import compare                # noqa: E402
from capp import config                 # noqa: E402
from capp import dataset as D           # noqa: E402
from capp import harness as H           # noqa: E402
from capp import llm_client             # noqa: E402
from capp import planners as P          # noqa: E402
from capp import plannerspec as PS      # noqa: E402
from capp import summary as SM          # noqa: E402
from capp import systems as SY          # noqa: E402
from capp import twostep as T           # noqa: E402
from capp import verify as V            # noqa: E402
from harness_clock import count_sleeps  # noqa: E402
import llm_planner                      # noqa: E402
import route_builder as rb              # noqa: E402

FS = PS.PlannerSpec
PART = {"part_id": "T1", "material": "Steel", "features": ["Hole", "Reamed_Hole"], "tolerance": "0.02mm", "batch_size": 100}


@pytest.fixture(scope="module")
def dev():
    return D.load_dataset("dev_v2")


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GROQ_API_KEY", "")                       # a test can never reach the real API
    monkeypatch.delenv("CAPP_LLM_MODE", raising=False)
    monkeypatch.delenv("CAPP_RUN_TAG", raising=False)
    monkeypatch.delenv("CAPP_STRUCTURED", raising=False)
    monkeypatch.setattr(llm_client, "_sleep", lambda s: None)    # no waiting between retries
    llm_client.reset_transport()
    llm_client.reset_call_log()
    T.reset_refusals()
    yield
    llm_client.reset_transport()
    T.reset_refusals()


def kind(messages):
    sys_msg = messages[0]["content"]
    if sys_msg.startswith("You are a manufacturing process planning expert. Your job is to plan the machining"):
        return "reason"
    if sys_msg.startswith("You convert a machining plan"):
        return "formalise"
    return "v1"


def features_of(messages):
    return re.search(r"Features: (.*)\n", messages[-1]["content"]).group(1).split(", ")


def rb_route(features):
    return V.clean_steps(rb.generate_valid_routes(features, max_routes=1)[0]["steps"])


def fake(seen=None, bad_first=False, reasoning_text=None, formal_text=None, finish="stop"):
    """Transport playing both steps. Reasoning ends with FINAL ORDER = the Route Builder's route (reversed when bad_first and
    the sample is a first one); formalisation copies the FINAL ORDER line like the prompt asks."""
    seen = [] if seen is None else seen

    def transport(model, messages, params):
        seen.append({"kind": kind(messages), "messages": messages, "params": params})
        k = kind(messages)
        if k == "reason":
            route = rb_route(features_of(messages))
            if bad_first and len(seen) <= 2:
                route = list(reversed(route))
            text = reasoning_text if reasoning_text is not None else \
                "Facing first, inspection last; drilling before reaming.\nFINAL ORDER: " + " > ".join(route)
            return {"text": text, "prompt_tokens": 300, "completion_tokens": 120, "finish_reason": finish}
        if k == "formalise":
            order = T.reasoning_order(messages[1]["content"]) or []
            text = formal_text if formal_text is not None else json.dumps({"operations": order})
            return {"text": text, "prompt_tokens": 200, "completion_tokens": 40, "finish_reason": "stop"}
        if messages[-1]["content"].startswith("A verifier rejected"):          # R2 re-prompt
            return {"text": json.dumps(rb_route(features_of([messages[1]]))), "prompt_tokens": 100,
                    "completion_tokens": 30, "finish_reason": "stop"}
        return {"text": json.dumps(rb_route(features_of(messages))), "prompt_tokens": 100, "completion_tokens": 30,
                "finish_reason": "stop"}
    llm_client.set_transport(transport)
    return seen


# ---- names ------------------------------------------------------------------------------------------------
def test_ts_option_parses_and_combines():
    assert PS.split_name("b0+ts") == ("b0", FS(two_step=True))
    assert PS.split_name("n5_r1_s3+ts+fs2x+h1")[1] == FS(few_shot=2, loo="features", hint=True, two_step=True)
    assert PS.split_name("b0+ts")[1].is_default is False and FS().is_default
    assert PS.split_name("b0+h1+ts+fs4")[1].name == "fs4+h1+ts"
    assert SY.parse_system("n5_r1_s3+ts") == (5, "r1", "s3") and SY.planner_spec("b0+ts").two_step
    for bad in ("b0+ts+ts", "b0+ts1", "b0+TS", "rb+ts"):
        with pytest.raises(ValueError):
            SY.parse_system(bad)


# ---- prompts and schema ----------------------------------------------------------------------------------------
def test_closed_vocabulary_contains_inspection_and_the_schema_is_an_enum_of_it():
    ops = T.valid_operations()
    v1_ops = llm_planner._get_valid_operations_list().split(", ")
    assert "Inspection" in ops and "Facing" in ops and ops == sorted(set(ops)) and len(ops) == len(v1_ops) + 1
    assert "Inspection" not in v1_ops                          # v1's prompt demands Inspection but does not list it
    assert set(llm_planner._get_valid_operations_list().split(", ")) <= set(ops)        # v1's list plus Inspection
    assert set(ops) <= V.CANONICAL_OPS                                                  # nothing the verifier does not know
    rf = T.response_format("json_schema")
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    sch = rf["json_schema"]["schema"]
    assert sch["properties"]["operations"]["items"]["enum"] == ops and sch["additionalProperties"] is False
    assert T.response_format("json_object") == {"type": "json_object"} and T.response_format("none") is None
    json.dumps(rf)                                                                       # serialisable: it goes into the cache key


def test_reasoning_prompt_keeps_the_v1_rules_and_asks_for_no_json():
    msgs = T.reason_messages("Steel", ["Hole", "Face"], "0.02mm", 250)
    assert [m["role"] for m in msgs] == ["system", "user"]
    system = msgs[0]["content"]
    assert 'MUST start with "Facing" and end with "Inspection"' in system and "Drilling must come before Tapping/Reaming" in system
    assert "{valid_ops}" not in system and "FINAL ORDER:" in system and "Do NOT write JSON" in system
    assert "Material: Steel\nFeatures: Hole, Face\nTolerance: 0.02mm\nBatch Size: 250" in msgs[1]["content"]
    assert "Validated plans" not in msgs[1]["content"] and "rule-based" not in msgs[1]["content"]
    assert msgs[1]["content"].endswith("end with the FINAL ORDER line.")


def test_reasoning_prompt_carries_examples_least_similar_first_and_the_hint():
    pool = P.Pool([P.Example("A", "Steel", ("Hole",), "0.05mm", 1, ("Facing", "Drilling", "Inspection")),
                   P.Example("B", "Steel", ("Hole", "Face"), "0.05mm", 1, ("Facing", "Face Milling", "Drilling", "Inspection"))])
    exs = pool.retrieve({"part_id": "Q", "material": "Steel", "features": ["Hole", "Face"]}, 2)      # B (most similar), A
    msgs = T.reason_messages("Steel", ["Hole", "Face"], "0.05mm", 1, exs, ["Facing", "Drilling", "Inspection"])
    user = msgs[1]["content"]
    assert user.index("Features: Hole; plan: Facing > Drilling > Inspection") < user.index("Features: Hole, Face; plan: Facing > Face")
    assert 'A rule-based planner proposes this route:\n["Facing", "Drilling", "Inspection"]' in user
    assert user.index("Validated plans") < user.index("rule-based") < user.index("FINAL ORDER line")


def test_formalisation_prompt_is_a_faithful_copy_instruction():
    msgs = T.formal_messages("Drill, then ream.\nFINAL ORDER: Facing > Drilling > Reaming > Inspection")
    s = msgs[0]["content"]
    assert 'Output ONLY a JSON object of the form {"operations": [' in s and "Do NOT add, remove, reorder or correct" in s
    assert "{valid_ops}" not in s and "Reaming" in s
    assert "FINAL ORDER: Facing > Drilling" in msgs[1]["content"]


@pytest.mark.parametrize("text,expected", [
    ("blah\nFINAL ORDER: Facing > Center Drilling > Drilling > Inspection", ["Facing", "Center Drilling", "Drilling", "Inspection"]),
    ("**FINAL ORDER:** Facing > **Drilling** > Inspection.", ["Facing", "Drilling", "Inspection"]),
    ("Final order: Facing -> Drilling -> Inspection", ["Facing", "Drilling", "Inspection"]),
    ("FINAL ORDER: Facing, Drilling, Inspection", ["Facing", "Drilling", "Inspection"]),
    ("FINAL ORDER: Facing > Reaming\nmore text\nFINAL ORDER: Facing > Drilling", ["Facing", "Drilling"]),
    ("no such line", None), ("", None), (None, None),
])
def test_reasoning_order(text, expected):
    assert T.reasoning_order(text) == expected


@pytest.mark.parametrize("raw,steps,parser", [
    ('{"operations": ["Facing", "Inspection"]}', ["Facing", "Inspection"], "structured"),
    (' {"operations": []} ', [], "structured"),
    ('["Facing", "Inspection"]', ["Facing", "Inspection"], "array"),
    ('<think>maybe [1, 2]</think>{"operations": ["Facing"]}', ["Facing"], "structured"),
    ('Sure! {"operations": ["Facing", "Drilling"]} Hope it helps', ["Facing", "Drilling"], "fallback"),
    ('Here you go: ["Facing", "Drilling"]', ["Facing", "Drilling"], "fallback"),
    ('{"operations": "Facing"}', [], "none"),                       # an object without an operations list is no route
    ("", [], "none"),
])
def test_parse_formalised(raw, steps, parser):
    assert T.parse_formalised(raw) == (steps, parser)


def test_strip_think():
    assert T.strip_think("<think>a</think> plan") == "plan"
    assert T.strip_think("<think>only thoughts") == "only thoughts"                  # cut off inside the tag: keep what there is
    assert T.strip_think("<think>only</think>") == "only" and T.strip_think("<think></think>") == ""
    assert T.strip_think("") == "" and T.strip_think(None) == ""


# ---- the planner -------------------------------------------------------------------------------------------------
def run_planner(sample_id=0, part=PART, **kw):
    planner = T.make_planner(**kw)
    return planner(part["material"], part["features"], part["tolerance"], part["batch_size"], sample_id=sample_id)


def test_two_calls_with_the_right_parameters_and_a_complete_meta():
    seen = fake()
    res = run_planner()
    assert [s["kind"] for s in seen] == ["reason", "formalise"]
    r, f = seen
    assert r["params"] == {"temperature": config.REASON_TEMPERATURE, "max_tokens": config.REASON_MAX_TOKENS, "reasoning_effort": "none"}      # Step 37
    assert f["params"]["temperature"] == 0.0 and f["params"]["max_tokens"] == config.FORMAL_MAX_TOKENS
    assert f["params"]["response_format"] == T.response_format("json_schema")
    assert res["success"] and res["steps"] == rb_route(PART["features"]) and res["model"] == llm_planner.MODEL
    m = res["meta"]
    assert m["n_calls"] == 2 and m["cached"] is False and m["prompt_tokens"] == 500 and m["completion_tokens"] == 160
    assert [c["role"] for c in m["calls"]] == ["reason", "formalise"] and m["calls"][1]["structured"] == "json_schema"
    assert m["structured_used"] == "json_schema" and m["parser"] == "structured"
    assert m["reasoning"].startswith("Facing first") and m["reasoning_order"] == res["steps"] and m["faithful"] is True
    assert m["prompt_context"] == {"examples": [], "hint": False, "two_step": True}
    assert m["reasoning_truncated"] is False and m["finish_reason"] == "stop"
    json.dumps(res)                                              # everything in the result can go into records.jsonl


def test_second_call_is_free_and_another_sample_is_a_new_draw():
    seen = fake()
    run_planner(sample_id=0)
    again = run_planner(sample_id=0)
    assert len(seen) == 2 and again["meta"]["cached"] is True and again["meta"]["n_calls"] == 2
    run_planner(sample_id=1)
    assert len(seen) == 4


def test_formalisation_that_changes_the_stated_order_is_flagged_not_hidden():
    fake(formal_text=json.dumps({"operations": ["Facing", "Reaming", "Drilling", "Inspection"]}))
    res = run_planner()
    assert res["steps"] == ["Facing", "Reaming", "Drilling", "Inspection"] and res["meta"]["faithful"] is False
    assert res["meta"]["reasoning_order"] == rb_route(PART["features"])


def test_no_final_order_line_means_faithful_is_unknown_and_truncation_is_recorded():
    fake(reasoning_text="I would drill first and ream later, with facing first.", finish="length")
    res = run_planner()
    assert res["meta"]["faithful"] is None and res["meta"]["reasoning_order"] is None and res["meta"]["reasoning_truncated"] is True


def test_think_tags_in_the_reasoning_are_removed_before_step_two():
    seen = fake(reasoning_text="<think>hmm</think>Plan.\nFINAL ORDER: Facing > Inspection")
    res = run_planner()
    assert "<think>" not in res["meta"]["reasoning"] and "hmm" not in seen[1]["messages"][1]["content"]
    assert res["steps"] == ["Facing", "Inspection"]


def test_empty_reasoning_stops_after_one_call():
    seen = fake(reasoning_text="<think></think>")
    res = run_planner()
    assert len(seen) == 1 and res["success"] is False and res["raw_response"] == "empty reasoning" and res["meta"]["n_calls"] == 1


def test_unparsable_formalisation_is_a_failed_sample():
    fake(formal_text="")
    res = run_planner()
    assert res["success"] is False and res["steps"] == [] and res["meta"]["parser"] == "none"


def test_missing_key_gives_the_message_the_harness_guard_stops_on():
    res = run_planner()
    assert res == {"success": False, "steps": [], "raw_response": H.NO_KEY_MSG, "model": llm_planner.MODEL}


# ---- structured output: refused, downgraded, replayed -------------------------------------------------------------
def refusing(refuse, seen, message):
    inner = fake(seen)
    transport = llm_client._transport

    def t(model, messages, params):
        rf = params.get("response_format")
        mode = rf["type"] if rf else "none"
        if kind(messages) == "formalise":
            seen.append({"mode": mode})
            if mode in refuse:
                raise RuntimeError(message)
        return transport(model, messages, params)
    llm_client.set_transport(t)
    return inner


def test_a_refused_json_schema_falls_back_to_json_object_and_is_not_asked_again():
    seen = []
    refusing({"json_schema"}, seen, "400 response_format json_schema is not supported by this model")
    first = run_planner(sample_id=0)
    assert first["meta"]["structured_used"] == "json_object" and first["success"]
    assert first["meta"]["calls"][1]["skipped"][0][0] == "json_schema"
    n = len([s for s in seen if "mode" in s])
    second = run_planner(sample_id=1)
    modes = [s["mode"] for s in seen if "mode" in s][n:]
    assert modes == ["json_object"] and second["meta"]["structured_used"] == "json_object"      # json_schema not tried again


def test_both_structured_modes_refused_gives_plain_text_and_the_v1_parser_does_the_work():
    seen = []
    refusing({"json_schema", "json_object"}, seen, "response_format is unsupported")
    res = run_planner()
    assert res["meta"]["structured_used"] == "none" and res["success"] and res["meta"]["parser"] == "structured"


def test_an_ordinary_error_is_not_remembered_as_a_refusal():
    seen = []
    refusing({"json_schema"}, seen, "503 service unavailable")
    run_planner(sample_id=0)
    run_planner(sample_id=1)
    assert [s["mode"] for s in seen if "mode" in s].count("json_schema") >= 2 and not T._REFUSED


def modes_asked(seen):
    return sorted({s["mode"] for s in seen if "mode" in s})


def test_a_service_error_does_not_try_another_output_format():
    # an outage, a rate limit, a refused key say nothing about json_schema: asking again as json_object or plain text would cost another round of
    # waiting and would quietly make this sample a different experiment (meta.structured_used)
    for message in ("503 service unavailable", "429 rate limit reached", "Connection error."):
        seen = []
        refusing({"json_schema", "json_object", "none"}, seen, message)
        res = run_planner()
        assert res["success"] is False and res["raw_response"].startswith("API Error")
        assert modes_asked(seen) == ["json_schema"], message


def test_an_http_status_decides_between_a_refused_format_and_a_service_error(monkeypatch, tmp_path):
    cases = [(400, True), (422, True), (401, False), (403, False), (404, False), (429, False), (500, False), (503, False)]
    for status, falls_back in cases:
        monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / f"cache{status}"))      # an answer cached by the case before would hide the second ask
        seen = []
        inner = fake(seen)
        t0 = llm_client._transport

        def t(model, messages, params, status=status):
            rf = params.get("response_format")
            if kind(messages) == "formalise":
                seen.append({"mode": rf["type"] if rf else "none"})
                if rf and rf["type"] == "json_schema":
                    err = RuntimeError("bad request")
                    err.status_code = status
                    raise err
            return t0(model, messages, params)
        llm_client.set_transport(t)
        res = run_planner()
        assert (modes_asked(seen) != ["json_schema"]) == falls_back, status
        assert res["success"] is falls_back, status
        if falls_back:
            assert res["meta"]["structured_used"] == "json_object"
        llm_client.reset_transport()
        T.reset_refusals()


def test_a_refused_format_still_falls_back_when_the_error_has_no_status():
    seen = []
    refusing({"json_schema"}, seen, "response_format json_schema is not supported by this model")
    res = run_planner()
    assert res["success"] and res["meta"]["structured_used"] == "json_object" and modes_asked(seen) == ["json_object", "json_schema"]


def test_every_mode_failing_is_an_api_error_for_the_guard():
    fake()
    t0 = llm_client._transport

    def t(model, messages, params):
        if kind(messages) == "formalise":
            raise RuntimeError("503 down")
        return t0(model, messages, params)
    llm_client.set_transport(t)
    res = run_planner()
    assert res["success"] is False and res["raw_response"].startswith("API Error") and res["meta"]["n_calls"] == 1


def test_first_mode_can_be_set_from_the_environment(monkeypatch):
    monkeypatch.setenv("CAPP_STRUCTURED", "none")
    seen = fake()
    res = run_planner()
    assert "response_format" not in seen[1]["params"] and res["meta"]["structured_used"] == "none"
    monkeypatch.setenv("CAPP_STRUCTURED", "bogus")
    with pytest.raises(ValueError):
        config.structured_mode()


def test_replay_reproduces_a_run_and_finds_the_recorded_structured_mode(monkeypatch):
    monkeypatch.setenv("CAPP_STRUCTURED", "json_object")
    fake()
    first = run_planner()
    llm_client.reset_transport()
    monkeypatch.delenv("CAPP_STRUCTURED")
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    again = run_planner()                                         # asks json_schema first: not in the cache, then json_object: hit
    assert again["steps"] == first["steps"] and again["meta"]["cached"] is True
    assert again["meta"]["structured_used"] == "json_object"
    assert again["meta"]["calls"][1]["skipped"] == [["json_schema", "not in cache"]]


def test_replay_of_something_never_recorded_is_the_not_in_cache_error_the_guard_stops_on(monkeypatch):
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    res = run_planner()
    assert res["raw_response"].startswith("API Error: not in cache")
    fake()
    run_planner(sample_id=7)                                       # live mode: record the reasoning and its formalisation
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    assert run_planner(sample_id=8)["raw_response"].startswith("API Error: not in cache")


def test_response_format_reaches_the_groq_client(monkeypatch):
    got = {}

    def create(model, messages, **params):
        got.update(params)
        msg = types.SimpleNamespace(content='{"operations": ["Facing", "Inspection"]}')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
                                     usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=6))
    stub = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    monkeypatch.setattr(llm_client, "_groq_client", stub)
    monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")
    resp = llm_client.chat(T.formal_messages("x"), model="m", temperature=0.0, max_tokens=300,
                           response_format=T.response_format("json_schema"))
    assert got["response_format"]["type"] == "json_schema" and got["temperature"] == 0.0 and resp.text.startswith("{")


# ---- harness -------------------------------------------------------------------------------------------------------
def test_b0_ts_run_two_calls_per_sample_and_all_the_numbers(dev, tmp_path):
    seen = fake()
    out = H.run_experiment("b0+ts", dev[:3], tag="ts", out_dir=str(tmp_path), progress=None)
    assert len(seen) == 30 and [s["kind"] for s in seen[:2]] == ["reason", "formalise"]
    s = out["summary"]
    assert s["M7"]["calls_per_part"] == 5.0 and s["M7"]["api_calls_per_part"] == 10.0
    assert s["M7"]["prompt_tokens_per_part"] == 5 * 500 and s["M7"]["completion_tokens_per_part"] == 5 * 160
    ts = s["two_step"]
    assert ts["n"] == 15 and ts["structured_used"] == {"json_schema": 15} and ts["parser"] == {"structured": 15}
    assert ts["faithful_share"] == 1.0 and ts["with_final_order_line"] == 1.0 and ts["reasoning_truncated"] == 0
    assert ts["reasoning_tokens_mean"] == 120 and s["M1"]["strict_valid_of_parsed"] == 1.0
    assert s["planner_prompt"]["two_step_share"] == 1.0
    text = SM.format_summary("b0+ts", s)
    assert "two-step" in text and "samples/part (10.00 API calls incl. repair)" in text
    info = json.loads((tmp_path / "ts_b0+ts" / "run.json").read_text())
    assert info["two_step"] is True and info["structured_output_requested"] == "json_schema" and info["planner_options"] == "ts"
    rows = (tmp_path / "ts_b0+ts" / "per_candidate.csv").read_text().splitlines()
    assert {"api_calls", "structured", "parser", "faithful", "reasoning_chars"} <= set(rows[0].split(","))
    rec = out["records"][0]["candidates"][0]["meta"]
    assert rec["reasoning"] and rec["n_calls"] == 2                        # the reasoning is stored with the route


def test_one_step_runs_are_unchanged_by_the_new_counters(dev):
    fake()
    out = H.run_experiment("b0", dev[:2], tag="one", progress=None, write=False)
    s = out["summary"]
    assert s["two_step"] == {"n": 0} and s["M7"]["api_calls_per_part"] == s["M7"]["total_calls_per_part"] == 5.0
    text = SM.format_summary("b0", s)
    assert "calls/part" in text and "samples/part" not in text and "two-step" not in text


def test_plain_and_two_step_prompts_do_not_share_cache_but_each_is_free_the_second_time(dev):
    seen = fake()
    H.run_experiment("b1", dev[:3], tag="c", progress=None, write=False)
    assert len(seen) == 3
    H.run_experiment("b1+ts", dev[:3], tag="c", progress=None, write=False)
    assert len(seen) == 9
    H.run_experiment("b1+ts", dev[:3], tag="c", progress=None, write=False)
    H.run_experiment("b1", dev[:3], tag="c", progress=None, write=False)
    assert len(seen) == 9


def test_examples_and_hint_go_into_the_reasoning_prompt_only(dev):
    seen = fake()
    H.run_experiment("b1_r1+fs2x+h1+ts", dev[:3], tag="e", progress=None, write=False)
    reasons = [s for s in seen if s["kind"] == "reason"]
    forms = [s for s in seen if s["kind"] == "formalise"]
    assert len(reasons) == len(forms) == 3
    by_id = {p["part_id"]: p for p in dev}
    for s, part in zip(reasons, dev[:3]):
        user = s["messages"][1]["content"]
        assert "Validated plans of similar parts" in user and json.dumps(P.rb_plan(part["features"])) in user
        assert len(s["messages"]) == 2
    for s in forms:
        assert "Validated plans" not in s["messages"][1]["content"] and "rule-based" not in s["messages"][1]["content"]
    out = H.run_experiment("b1_r1+fs2x+h1+ts", dev[:3], tag="e", progress=None, write=False)
    for rec in out["records"]:
        ctx = rec["candidates"][0]["meta"]["prompt_context"]
        own = set(rec["part"]["features"])
        assert ctx["two_step"] and ctx["hint"] and len(ctx["examples"]) == 2
        assert all(set(by_id[e]["features"]) != own for e in ctx["examples"])


def test_r2_reprompt_after_a_two_step_sample_is_still_the_one_step_repair_conversation(dev):
    seen = fake(bad_first=True)
    part = dev[0]
    out = H.run_experiment("b1_r2+ts", [part], tag="r2", progress=None, write=False)
    kinds = [s["kind"] for s in seen]
    assert kinds[:2] == ["reason", "formalise"] and "v1" in kinds[2:]          # the re-prompt uses the v1 conversation
    repair = [s for s in seen if s["kind"] == "v1"][0]
    assert [m["role"] for m in repair["messages"]] == ["system", "user", "assistant", "user"]
    assert out["summary"]["M2"]["repaired_by_r2_total"] == 1 and out["summary"]["M2"]["final_strict_valid"] == 1.0
    assert out["summary"]["M7"]["api_calls_per_part"] == 3.0 and out["summary"]["M7"]["repair_calls_per_part"] == 1.0


def test_guards_stop_a_two_step_run(dev, monkeypatch):
    with pytest.raises(H.HarnessError, match="GROQ_API_KEY"):
        H.run_experiment("b0+ts", dev[:2], tag="nokey", progress=None, write=False)
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    with pytest.raises(H.HarnessError, match="not in cache"):
        H.run_experiment("b0+ts", dev[:2], tag="rp", progress=None, write=False)


def test_replay_reproduces_a_finished_two_step_run_offline(dev, monkeypatch):
    fake()
    first = H.run_experiment("n3_r1_s3+ts", dev[:3], tag="rp", progress=None, write=False)
    llm_client.reset_transport()
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    again = H.run_experiment("n3_r1_s3+ts", dev[:3], tag="rp", progress=None, write=False)
    assert [r["final"]["steps"] for r in again["records"]] == [r["final"]["steps"] for r in first["records"]]
    assert again["summary"]["two_step"]["n"] == 9 and again["summary"]["M7"]["cache_hit_rate"] == 1.0


def test_throttle_waits_only_after_live_two_step_parts(dev, monkeypatch):
    slept = []
    count_sleeps(monkeypatch, slept)       # only the harness's own pauses are recorded (see harness_clock.py)
    fake()
    H.run_experiment("b1+ts", dev[:2], tag="th", throttle=0.5, progress=None, write=False)
    assert slept == [0.5, 0.5]
    slept.clear()
    H.run_experiment("b1+ts", dev[:2], tag="th", throttle=0.5, progress=None, write=False)
    assert slept == []


def test_compare_table_has_the_new_rows(dev, tmp_path):
    fake()
    for name in ("b1", "b1+ts"):
        H.run_experiment(name, dev[:3], tag="cmp", out_dir=str(tmp_path), progress=None)
    text = compare.compare([str(tmp_path / "cmp_b1"), str(tmp_path / "cmp_b1+ts")])
    assert "API calls / part" in text and "stated order kept by step 2" in text and "b1+ts" in text


# ---- command line ----------------------------------------------------------------------------------------------
def test_cli_two_step_without_key_stops_and_bad_option_is_an_argument_error(tmp_path):
    env = dict(os.environ, GROQ_API_KEY="", PYTHONIOENCODING="utf-8", CAPP_CACHE_DIR=str(tmp_path / "c"))
    base = [sys.executable, os.path.join(ROOT, "scripts", "run_eval.py"), "--out", str(tmp_path)]
    res = subprocess.run(base + ["--system", "b0+ts", "--limit", "2", "--tag", "nokey"], capture_output=True, text=True,
                         cwd=ROOT, env=env, timeout=120)
    assert res.returncode == 2 and "STOPPED" in res.stderr and "GROQ_API_KEY" in res.stderr
    res = subprocess.run(base + ["--system", "b0+ts+ts"], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
    assert res.returncode == 2 and "ts" in res.stderr


# ---- show_plan ---------------------------------------------------------------------------------------------------------
def test_show_plan_prints_the_reasoning_the_verifier_and_the_repair(dev, tmp_path):
    fake(bad_first=True)
    H.run_experiment("b1_r1+ts", [dev[0]], tag="sp", out_dir=str(tmp_path), progress=None)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "show_plan.py"), str(tmp_path / "sp_b1_r1+ts"), dev[0]["part_id"]]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=60)
    assert res.returncode == 0, res.stderr
    out = res.stdout
    for token in ("reasoning:", "FINAL ORDER:", "structured output json_schema", "kept the stated order: yes", "LLM route:",
                  "verifier : INVALID", "repaired by r1:", "FINAL (repaired_llm"):
        assert token in out, token
    bad = subprocess.run(cmd[:-1] + ["NOPE"], capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert bad.returncode == 2 and "not in" in bad.stderr


def test_show_plan_works_on_one_step_runs_too(dev, tmp_path):
    fake()
    H.run_experiment("b1", [dev[1]], tag="sp1", out_dir=str(tmp_path), progress=None)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "show_plan.py"), str(tmp_path / "sp1_b1"), dev[1]["part_id"]]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=60)
    assert res.returncode == 0 and "reasoning:" not in res.stdout and "LLM route:" in res.stdout and "FINAL (" in res.stdout
