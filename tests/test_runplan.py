"""Step 33: what a run will ask of the API, before it starts (capp/runplan.py), and the cap that stops it (--max-calls).

Offline. The number that matters is the bound on live calls, so it is checked against the REAL planners, the real llm_client and the real
harness with a transport that always gives a wrong answer (the worst case: every sample slot needs its re-prompt and, in a cascade, its
second model): the number of calls the transport sees must never exceed the bound, and for the systems whose worst case can be reached it
must be equal to it. The estimates (tokens, cost, time) are arithmetic on the history of earlier runs and are checked on small run folders.
"""
import collections
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capp import config, dataset as D, harness as H, llm_client, runplan as RP, systems            # noqa: E402

WRONG = ["Facing", "Inspection"]            # parses, but covers no feature: strict-invalid for every dev part


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


def kind(messages):
    sys_msg = messages[0]["content"]
    if sys_msg.startswith("You are a manufacturing process planning expert. Your job is to plan the machining"):
        return "reason"
    if sys_msg.startswith("You convert a machining plan"):
        return "formalise"
    return "other"


class Worst:
    """A transport that always gives a wrong but parseable answer, and counts the requests that reach it, by model.
    style "ts": the reasoning is different every time (so the formalisation prompt is too and nothing is cached by accident)."""

    def __init__(self, style="route"):
        self.style, self.by_model, self.n = style, collections.Counter(), 0

    def __call__(self, model, messages, params):
        self.by_model[model] += 1
        self.n += 1
        k = kind(messages)
        if self.style == "cg":
            text = "1"
        elif self.style == "ts" and k == "reason":
            text = f"FINAL ORDER: Facing > Inspection ({self.n})"
        elif self.style == "ts" and k == "formalise":
            text = json.dumps({"operations": WRONG})
        else:
            text = json.dumps(WRONG)
        return {"text": text, "prompt_tokens": 50, "completion_tokens": 9, "finish_reason": "stop"}

    @property
    def total(self):
        return sum(self.by_model.values())


def dev(n):
    return D.load_dataset("dev_v2")[:n]


def run_worst(system, style="route", n=4, repeats=1):
    svc = Worst(style)
    llm_client.set_transport(svc)
    ps = dev(n)
    p = RP.plan(system, ps, repeats=repeats)
    H.run_experiment(system, ps, repeats=repeats, tag="w", progress=None, write=False)
    return p, svc


# ---- the bound against the real planners ---------------------------------------------------------------------
@pytest.mark.parametrize("system", ["b0", "b1", "b0_r2", "b0_r3", "n3_r2_s1+ms", "n3_r2_s1+ml", "n3_r2_s1+cas1", "n3_r2_s1+cas2",
                                    "n3_r2_s1+fs2+h1", "n3_r3_s1+fz1", "n2_r1_s3", "n4_r0_s2"])
def test_the_bound_is_reached_exactly_when_every_slot_is_wrong_and_never_exceeded(system):
    p, svc = run_worst(system, n=4, repeats=2)
    assert svc.total == p["max_calls"] > 0
    for t in p["tiers"]:
        assert svc.by_model[t["model"]] <= t["max_calls"]
    assert set(svc.by_model) <= {t["model"] for t in p["tiers"]}


@pytest.mark.parametrize("system", ["n3_r0_s1+ts", "n3_r2_s3+ts", "n2_r3_s1+ts+h1"])
def test_the_two_step_planner_is_two_calls_per_slot_and_the_bound_is_reached(system):
    p, svc = run_worst(system, style="ts", n=3, repeats=2)
    assert svc.total == p["max_calls"] and p["two_step"] is True


@pytest.mark.parametrize("system", ["n3_r0_s1+cg1", "n2_r2_s1+cg1", "n2_r2_s1+cas2+cg1"])
def test_the_constrained_planner_never_exceeds_its_bound(system):
    p, svc = run_worst(system, style="cg", n=4, repeats=2)
    assert 0 < svc.total <= p["max_calls"] and p["constrained"] is True
    for t in p["tiers"]:
        assert svc.by_model[t["model"]] <= t["max_calls"]


def test_the_constrained_bound_is_the_operations_minus_one_of_each_part():
    ps = dev(5)
    ops = [RP._ops_of(x) for x in ps]
    assert all(o >= 2 for o in ops)
    p = RP.plan("n3_r0_s1+cg1", ps)
    assert p["max_calls"] == 3 * sum(o - 1 for o in ops)
    p2 = RP.plan("n3_r2_s1+cg1", ps, repeats=2)                           # a re-prompt is one more call per slot
    assert p2["max_calls"] == 2 * 3 * sum(o - 1 + 1 for o in ops)


def test_the_graph_gives_the_operation_set_where_the_route_builder_has_none(monkeypatch):
    from capp import constrained as CG
    part = dev(1)[0]
    monkeypatch.setattr(CG, "required_ops", lambda features, pref="auto": [])
    assert RP._ops_of(part) == len(CG.graph_ops(systems.part_inputs(part)["features"])) > 0


def test_a_part_the_constrained_planner_cannot_build_costs_no_call():
    weird = dict(dev(1)[0], features=["NoSuchFeature"])
    assert RP._ops_of(weird) == 0
    assert RP.plan("n3_r0_s1+cg1", [weird])["max_calls"] == 0


# ---- the plan itself -------------------------------------------------------------------------------------------------
def test_plan_numbers_by_system():
    ps = dev(5)
    base = RP.plan("b0", ps)
    assert (base["uses_llm"], base["n_samples"], base["max_calls"], base["n_parts"], base["repeats"]) == (True, 5, 25, 5, 1)
    assert [(t["role"], t["model"], t["max_calls"]) for t in base["tiers"]] == [("primary", config.model("primary"), 25)]
    assert RP.plan("b1", ps)["max_calls"] == 5
    assert RP.plan("b0", ps, repeats=3)["max_calls"] == 75
    assert RP.plan("b0_r2", ps)["max_calls"] == 50 and RP.plan("b0_r3", ps)["max_calls"] == 50 and RP.plan("b0_r1", ps)["max_calls"] == 25
    assert RP.plan("b0+ts", ps)["max_calls"] == 50 and RP.plan("n5_r2_s1+ts", ps)["max_calls"] == 75
    assert RP.plan("n7_r0_s1", ps)["max_calls"] == 35
    ms, ml = RP.plan("b0+ms", ps), RP.plan("b0+ml", ps)
    assert ms["tiers"][0]["model"] == config.model("small") and ml["tiers"][0]["model"] == config.model("large")
    cas = RP.plan("n5_r2_s1+cas1", ps)
    assert [(t["role"], t["model"], t["max_calls"]) for t in cas["tiers"]] == [
        ("small", config.model("small"), 50), ("large", config.model("large"), 50)] and cas["max_calls"] == 100
    assert RP.plan("b0+fs2+h1", ps)["max_calls"] == 25 and RP.plan("n5_r1_s3+fz1", ps)["max_calls"] == 25


def test_the_throttle_is_carried_into_the_plan():
    ps = dev(3)
    assert RP.plan("b0", ps, throttle=1.5)["throttle_s"] == 1.5
    assert RP.plan("b0", ps)["throttle_s"] == 0.0 and RP.plan("b0", ps, throttle=None)["throttle_s"] == 0.0


def test_rb_makes_no_call_and_has_no_estimate():
    p = RP.plan("rb", dev(7), repeats=5)
    assert p["uses_llm"] is False and p["max_calls"] == 0 and p["tiers"] == [] and p["repeats"] == 1
    assert RP.format_plan(p) == "Plan for rb: 7 parts, no API call (the Route Builder needs none)."


def test_a_system_that_does_not_exist_is_a_value_error():
    for bad in ("b9", "n0_r0_s1", "n21_r0_s1", "rb+ts", "b0+zz", "b0_r0+fz1"):
        with pytest.raises(ValueError):
            RP.plan(bad, dev(2))


def test_the_plan_does_not_touch_the_api_or_the_cache(tmp_path):
    llm_client.set_transport(lambda *a: pytest.fail("a plan must not call the API"))
    RP.estimate(RP.plan("n5_r2_s1+cas2", dev(3)), {})
    assert llm_client.get_call_log() == [] and not (tmp_path / "cache").exists()


# ---- the history of earlier runs and the estimates -----------------------------------------------------------
def write_run(root, name, model="m1", calls=100, prompt=50000, completion=10000, latency=2.0, cascade=0, summary=True):
    d = root / name
    d.mkdir(parents=True)
    info = {"model": model, "cascade_level": cascade}
    if summary:
        info["llm_client_summary"] = {"calls": calls, "prompt_tokens": prompt, "completion_tokens": completion, "latency_mean_s": latency}
    (d / "run.json").write_text(json.dumps(info), encoding="utf-8")


def test_history_sums_the_runs_of_each_model(tmp_path):
    write_run(tmp_path, "dev_b0", "m1", calls=100, prompt=50000, completion=10000, latency=2.0)
    write_run(tmp_path, "dev_b1", "m1", calls=20, prompt=10000, completion=2000, latency=4.0)
    write_run(tmp_path, "dev_x", "m2", calls=10, prompt=900, completion=90, latency=1.0)
    h = RP.history(str(tmp_path))
    assert h["m1"] == {"calls": 120, "prompt": 60000, "completion": 12000, "latency_s": pytest.approx(280.0), "runs": 2}
    assert h["m2"] == {"calls": 10, "prompt": 900, "completion": 90, "latency_s": pytest.approx(10.0), "runs": 1}


def test_history_leaves_out_what_it_cannot_trust(tmp_path):
    write_run(tmp_path, "casc", "m1", cascade=1)                                  # the totals of a cascade mix two models
    write_run(tmp_path, "nocalls", "m1", calls=0)                                 # nothing to average (an rb run, a fully failed run)
    write_run(tmp_path, "nosummary", "m1", summary=False)                         # rb: llm_client_summary missing
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "run.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "list").mkdir()
    (tmp_path / "list" / "run.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / "nomodel").mkdir()
    (tmp_path / "nomodel" / "run.json").write_text(json.dumps({"llm_client_summary": {"calls": 5}}), encoding="utf-8")
    write_run(tmp_path, "good", "m1", calls=10, prompt=100, completion=10, latency=1.0)
    h = RP.history(str(tmp_path))
    assert list(h) == ["m1"] and h["m1"]["calls"] == 10 and h["m1"]["runs"] == 1


def test_history_of_a_folder_that_is_not_there_is_empty(tmp_path):
    assert RP.history(str(tmp_path / "nowhere")) == {}


def test_history_default_folder_is_results_of_the_repository(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT", str(tmp_path))
    write_run(tmp_path / "results", "dev_b0", "m1", calls=3, prompt=30, completion=3, latency=1.0)
    assert RP.history()["m1"]["calls"] == 3


def fake_plan(max_calls=40, model="m1", n_parts=10, repeats=1, throttle=0.0):
    return {"system": "b0", "uses_llm": True, "n_parts": n_parts, "repeats": repeats, "n_samples": 5, "reprompt": False, "two_step": False,
            "constrained": False, "cascade": 0, "tiers": [{"role": "primary", "model": model, "max_calls": max_calls}],
            "max_calls": max_calls, "throttle_s": throttle}


def test_the_estimate_is_the_mean_of_earlier_calls_times_the_bound(monkeypatch):
    monkeypatch.setitem(config.PRICE_USD_PER_MTOK, "m1", (0.5, 1.0))
    h = {"m1": {"calls": 100, "prompt": 50000, "completion": 10000, "latency_s": 200.0, "runs": 3}}
    p = RP.estimate(fake_plan(40), h)
    e = p["estimate"][0]
    assert (e["per_call_in"], e["per_call_out"]) == (500, 100)
    assert (e["tokens_in"], e["tokens_out"]) == (20000, 4000) and e["seconds"] == pytest.approx(80.0)
    assert e["usd"] == pytest.approx((20000 * 0.5 + 4000 * 1.0) / 1e6)
    assert (e["from_calls"], e["from_runs"]) == (100, 3) and p["estimate_seconds"] == pytest.approx(80.0)


def test_the_throttle_is_added_for_every_part_and_repeat():
    h = {"m1": {"calls": 10, "prompt": 100, "completion": 10, "latency_s": 10.0, "runs": 1}}
    p = RP.estimate(fake_plan(20, n_parts=10, repeats=3, throttle=1.5), h)
    assert p["estimate_seconds"] == pytest.approx(20 * 1.0 + 1.5 * 10 * 3)


def test_no_history_for_a_model_is_said_and_not_guessed():
    p = RP.estimate(fake_plan(40, model="other"), {"m1": {"calls": 1, "prompt": 1, "completion": 1, "latency_s": 1.0, "runs": 1}})
    assert p["estimate"] == [{"model": "other", "note": "no earlier run with this model in results/"}]
    text = RP.format_plan(p)
    assert "no earlier run with this model" in text and "min, one call at a time" not in text and "tokens" not in text.split("estimate")[1]


def test_a_model_without_a_price_has_no_cost_not_zero(monkeypatch):
    monkeypatch.delitem(config.PRICE_USD_PER_MTOK, "m1", raising=False)
    h = {"m1": {"calls": 10, "prompt": 100, "completion": 10, "latency_s": 10.0, "runs": 1}}
    p = RP.estimate(fake_plan(40), h)
    assert p["estimate"][0]["usd"] is None
    assert "n/a (no price in capp/config.py)" in RP.format_plan(p)


def test_a_cascade_is_estimated_per_model_and_the_time_adds_up():
    p = fake_plan(0)
    p["tiers"] = [{"role": "small", "model": "s", "max_calls": 10}, {"role": "large", "model": "l", "max_calls": 10}]
    p["max_calls"] = 20
    h = {"s": {"calls": 10, "prompt": 100, "completion": 10, "latency_s": 10.0, "runs": 1},
         "l": {"calls": 10, "prompt": 1000, "completion": 100, "latency_s": 50.0, "runs": 1}}
    RP.estimate(p, h)
    assert [e["tokens_in"] for e in p["estimate"]] == [100, 1000] and p["estimate_seconds"] == pytest.approx(10 + 50)


def test_token_counts_are_written_in_thousands_from_one_thousand():
    assert [RP._k(x) for x in (0, 999, 999.4, 1000, 1500, 20000)] == ["0", "999", "999", "1.0k", "1.5k", "20.0k"]


def test_the_text_of_a_plan(monkeypatch):
    monkeypatch.setitem(config.PRICE_USD_PER_MTOK, "m1", (0.5, 1.0))
    h = {"m1": {"calls": 100, "prompt": 50000, "completion": 10000, "latency_s": 200.0, "runs": 3}}
    text = RP.format_plan(RP.estimate(fake_plan(40, throttle=1.0), h))
    assert text.splitlines()[0] == "Plan for b0: 10 parts x 1 repeat(s), 5 sample(s) per part"
    assert "live calls, at most (primary)" in text and "40  (m1)" in text
    assert "live calls, at most (total)" in text and "a bound" in text
    assert "20.0k in + 4.0k out tokens if every call is live, $0.014 at list price [TO CONFIRM against the Groq console]" in text
    assert "from your 100 earlier calls in 3 run(s): 500 in + 100 out tokens per call" in text
    assert "about 1.5 min, one call at a time (includes --throttle 1 s per part)" in text     # 40 calls x 2 s + 10 parts x 1 s = 90 s
    assert "add --max-calls N (for example --max-calls 20)" in text


def test_the_text_says_what_a_system_does_per_slot():
    ps = dev(2)
    assert "calls per sample slot, at most  : 1\n" in RP.format_plan(RP.plan("b0", ps)) + "\n"
    assert "2 (two-step planner), plus 1 re-prompt" in RP.format_plan(RP.plan("b0_r2+ts", ps))
    assert "operations - 1" in RP.format_plan(RP.plan("b0+cg1", ps))
    t = RP.format_plan(RP.plan("b0_r2+cas1", ps))
    assert "the large model may be asked for every slot again" in t and "(small)" in t and "(large)" in t


# ---- --max-calls in the harness --------------------------------------------------------------------------------------
class Counting:
    def __init__(self):
        self.attempts = 0

    def __call__(self, model, messages, params):
        self.attempts += 1
        return {"text": json.dumps(["Facing", "Center Drilling", "Drilling", "Chamfering", "Inspection"]), "prompt_tokens": 10,
                "completion_tokens": 5, "finish_reason": "stop"}


def distinct_parts(n):
    base = D.load_dataset("dev_v2")[0]
    return [dict(base, part_id=f"C{i}", features=["Hole", "Chamfer"], batch_size=100 + i) for i in range(n)]


def go(ps, **kw):
    return H.run_experiment("b0", ps, tag="cap", progress=None, write=False, **kw)


def test_the_run_stops_between_parts_once_the_cap_is_reached_and_says_how_to_go_on():
    svc = Counting()
    llm_client.set_transport(svc)
    with pytest.raises(H.HarnessError) as ei:
        go(distinct_parts(4), max_calls=7)
    msg = str(ei.value)
    assert "stopped after 2 of 4 parts: 10 live calls reached --max-calls 7" in msg
    assert "run the same command again" in msg and "in the cache" in msg
    assert svc.attempts == 10


def test_running_the_same_command_again_makes_only_the_calls_that_are_missing():
    svc = Counting()
    llm_client.set_transport(svc)
    ps = distinct_parts(4)
    with pytest.raises(H.HarnessError):
        go(ps, max_calls=7)
    assert svc.attempts == 10
    out = go(ps)
    assert svc.attempts == 20 and len(out["records"]) == 4                      # the two finished parts came from the cache
    assert out["summary"]["M1"]["parsed_routes"] == 20


def test_a_cap_that_is_exactly_the_calls_needed_does_not_stop_the_last_part():
    svc = Counting()
    llm_client.set_transport(svc)
    out = go(distinct_parts(4), max_calls=20)
    assert len(out["records"]) == 4 and svc.attempts == 20
    svc2 = Counting()
    llm_client.set_transport(svc2)
    out2 = H.run_experiment("b0", distinct_parts(1), tag="cap-b", progress=None, write=False, max_calls=1)   # the cap is passed in the only part: nothing is left to stop
    assert len(out2["records"]) == 1 and svc2.attempts == 5


def test_cached_answers_do_not_count_against_the_cap():
    svc = Counting()
    llm_client.set_transport(svc)
    ps = distinct_parts(4)
    go(ps)
    assert svc.attempts == 20
    llm_client.reset_call_log()
    out = go(ps, max_calls=1)                                                      # every answer is in the cache: no live call at all
    assert len(out["records"]) == 4 and svc.attempts == 20


def test_calls_made_before_the_run_do_not_count_against_its_cap():
    svc = Counting()
    llm_client.set_transport(svc)
    go(distinct_parts(4))                                                          # 20 live calls in this process already
    with pytest.raises(H.HarnessError) as ei:
        H.run_experiment("b0", distinct_parts(6)[4:], tag="cap2", progress=None, write=False, max_calls=1)
    assert "stopped after 1 of 2 parts: 5 live calls reached --max-calls 1" in str(ei.value)


def test_repeats_count_as_parts_for_the_cap():
    svc = Counting()
    llm_client.set_transport(svc)
    with pytest.raises(H.HarnessError) as ei:
        H.run_experiment("b0", distinct_parts(2), repeats=2, tag="rep", progress=None, write=False, max_calls=5)
    assert "stopped after 1 of 4 parts: 5 live calls reached --max-calls 5" in str(ei.value)


def test_calls_that_failed_for_good_count_against_the_cap():
    class Down:
        def __call__(self, model, messages, params):
            raise RuntimeError("503 down")
    llm_client.set_transport(Down())
    with pytest.raises(H.HarnessError) as ei:
        go(distinct_parts(6), max_calls=2, max_consecutive_errors=1000)
    assert "live calls reached --max-calls 2" in str(ei.value)


@pytest.mark.parametrize("bad", [0, -3, True, 2.5, "5"])
def test_a_cap_that_is_not_a_positive_whole_number_is_refused_before_any_call(bad):
    svc = Counting()
    llm_client.set_transport(svc)
    with pytest.raises(H.HarnessError, match="max_calls must be a whole number of 1 or more"):
        go(distinct_parts(2), max_calls=bad)
    assert svc.attempts == 0


def test_no_cap_is_the_default():
    svc = Counting()
    llm_client.set_transport(svc)
    assert len(go(distinct_parts(3))["records"]) == 3 and svc.attempts == 15


def test_live_calls_counts_live_answers_and_failures_and_not_cache_hits():
    assert llm_client.live_calls() == 0
    svc = Counting()
    llm_client.set_transport(svc)
    msgs = [{"role": "user", "content": "x"}]
    llm_client.chat(msgs, sample_id=0)
    llm_client.chat(msgs, sample_id=1)
    llm_client.chat(msgs, sample_id=0)                                             # cache hit
    assert llm_client.live_calls() == 2

    def down(model, messages, params):
        raise RuntimeError("503")
    llm_client.set_transport(down)
    with pytest.raises(llm_client.ApiFailure):
        llm_client.chat(msgs, sample_id=9)
    assert llm_client.live_calls() == 3
    llm_client.reset_call_log()
    assert llm_client.live_calls() == 0


# ---- scripts/run_eval.py -----------------------------------------------------------------------------------------------
def load_cli():
    spec = importlib.util.spec_from_file_location("run_eval_plan_cli", os.path.join(ROOT, "scripts", "run_eval.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_plan_prints_the_plan_and_makes_no_call_and_writes_nothing(monkeypatch, capsys, tmp_path):
    cli = load_cli()
    llm_client.set_transport(lambda *a: pytest.fail("--plan must not call the API"))
    monkeypatch.setattr(cli.harness, "run_experiment", lambda *a, **k: pytest.fail("--plan must not run anything"))
    out = tmp_path / "res"
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--system", "n5_r2_s1+cas1", "--limit", "3", "--plan", "--out", str(out), "--throttle", "1"])
    cli.main()
    text = capsys.readouterr().out
    assert text.startswith("Plan for n5_r2_s1+cas1: 3 parts x 1 repeat(s), 5 sample(s) per part")
    assert "live calls, at most (total)     : 60" in text
    assert not out.exists()


def test_plan_uses_the_runs_in_the_out_folder_for_the_estimate(monkeypatch, capsys, tmp_path):
    cli = load_cli()
    write_run(tmp_path / "res", "dev_b0", config.model("primary"), calls=100, prompt=50000, completion=10000, latency=2.0)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--system", "b0", "--limit", "4", "--plan", "--out", str(tmp_path / "res")])
    cli.main()
    text = capsys.readouterr().out
    assert "live calls, at most (total)     : 20" in text and "10.0k in + 2.0k out tokens" in text and "from your 100 earlier calls in 1 run(s)" in text


def test_plan_takes_repeats_and_throttle_from_the_command_line(monkeypatch, capsys, tmp_path):
    cli = load_cli()
    write_run(tmp_path / "res", "dev_b0", config.model("primary"), calls=100, prompt=50000, completion=10000, latency=2.0)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--system", "b0", "--limit", "4", "--repeats", "2", "--throttle", "1", "--plan",
                                      "--out", str(tmp_path / "res")])
    cli.main()
    text = capsys.readouterr().out
    assert "4 parts x 2 repeat(s)" in text and "live calls, at most (total)     : 40" in text
    assert "about 1.5 min, one call at a time (includes --throttle 1 s per part)" in text         # 40 calls x 2 s + 4 parts x 2 repeats x 1 s = 88 s


def test_plan_of_rb_and_of_a_sealed_set_needs_no_freeze_and_stops_nothing(monkeypatch, capsys):
    cli = load_cli()
    for argv in (["--system", "rb"], ["--system", "b0", "--dataset", "test_v2", "--limit", "5"]):
        monkeypatch.setattr(sys, "argv", ["run_eval.py", *argv, "--plan"])
        cli.main()
        out = capsys.readouterr()
        assert out.out.startswith("Plan for ") and "STOPPED" not in out.err and "WARNING" not in out.err


def test_plan_of_a_system_that_does_not_exist_is_a_usage_error(monkeypatch, capsys):
    cli = load_cli()
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--system", "b9", "--plan"])
    with pytest.raises(SystemExit) as ei:
        cli.main()
    assert ei.value.code == 2 and "unknown system" in capsys.readouterr().err


def test_max_calls_is_passed_to_the_harness_and_is_none_by_default(monkeypatch, capsys):
    cli = load_cli()
    seen = []

    def fake_run(system, parts, **kw):
        seen.append(kw.get("max_calls", "missing"))
        return {"summary": {"M1": {}}, "out_dir": "somewhere"}
    monkeypatch.setattr(cli.harness, "run_experiment", fake_run)
    monkeypatch.setattr(cli.summary, "format_summary", lambda *a: "S")
    for extra in ([], ["--max-calls", "7"]):
        monkeypatch.setattr(sys, "argv", ["run_eval.py", "--system", "b0", "--limit", "2", *extra])
        cli.main()
    assert seen == [None, 7]


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_max_calls_below_one_is_a_usage_error(monkeypatch, capsys, bad):
    cli = load_cli()
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--system", "b0", "--max-calls", bad])
    with pytest.raises(SystemExit) as ei:
        cli.main()
    assert ei.value.code == 2 and "--max-calls must be 1 or more" in capsys.readouterr().err


def test_a_run_stopped_by_the_cap_exits_with_2_and_says_stopped(monkeypatch, capsys):
    cli = load_cli()

    def stop(*a, **k):
        raise H.HarnessError("stopped after 2 of 4 parts: ...")
    monkeypatch.setattr(cli.harness, "run_experiment", stop)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--system", "b0", "--limit", "2", "--max-calls", "3"])
    with pytest.raises(SystemExit) as ei:
        cli.main()
    assert ei.value.code == 2 and "STOPPED: stopped after 2 of 4 parts" in capsys.readouterr().err


def test_a_dataset_that_is_not_there_is_a_usage_error_not_a_traceback(monkeypatch, capsys, tmp_path):
    cli = load_cli()
    for extra in ([], ["--plan"]):
        monkeypatch.setattr(sys, "argv", ["run_eval.py", "--system", "b0", "--dataset", str(tmp_path / "nowhere.json"), *extra])
        with pytest.raises(SystemExit) as ei:
            cli.main()
        err = capsys.readouterr().err
        assert ei.value.code == 2 and "--dataset" in err and "nowhere.json" in err
