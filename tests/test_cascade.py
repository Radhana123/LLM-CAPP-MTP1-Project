"""Model cascade (capp/cascade.py, the +ms / +ml / +cas1 / +cas2 options, capp/cost.py, capp/pareto.py, scripts/cascade_pareto.py).
Offline: the LLM is a fake transport that answers by model id, so the models behave differently and every call is visible."""
import json
import os
import re
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from capp import cascade as CASC                                 # noqa: E402
from capp import compare, config, cost, dataset as D             # noqa: E402
from capp import harness as H                                    # noqa: E402
from capp import llm_client, pareto                              # noqa: E402
from capp import plannerspec as PS                               # noqa: E402
from capp import summary as SM                                   # noqa: E402
from capp import systems as SY                                   # noqa: E402
from capp import verify as V                                     # noqa: E402
from harness_clock import count_sleeps                           # noqa: E402
import cascade_pareto                                            # noqa: E402  (scripts/cascade_pareto.py)
import route_builder as rb                                       # noqa: E402

PRIMARY, SMALL, LARGE = "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"
FS = PS.PlannerSpec


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GROQ_API_KEY", "")                       # a test can never reach the real API
    for k in ("CAPP_LLM_MODE", "CAPP_RUN_TAG", "CAPP_STRUCTURED", "CAPP_MODEL_PRIMARY", "CAPP_MODEL_SMALL", "CAPP_MODEL_LARGE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(llm_client, "_sleep", lambda s: None)
    llm_client.reset_transport()
    llm_client.reset_call_log()
    yield
    llm_client.reset_transport()


# ---------------------------------------------------------------- parts and routes
FEATURE_SETS = [["Hole", "Chamfer"], ["Hole", "Slot"], ["Turning", "Thread"], ["Bore", "Groove"], ["Face", "Hole", "Chamfer"], ["Slot", "Pocket"]]


def make_parts(n):
    base = D.load_dataset("dev_v2")[0]
    return [dict(base, part_id=f"C{i}", features=list(FEATURE_SETS[i % len(FEATURE_SETS)]), batch_size=100 + i) for i in range(n)]


def good(features):
    return V.clean_steps(rb.generate_valid_routes(features, max_routes=1)[0]["steps"])


def swapped(features):          # Facing is no longer first: order violation, repairable by R1
    r = good(features)
    return [r[1], r[0]] + r[2:]


def valid(features, steps):
    return V.check_route(steps, features).valid


def result(steps, model="m", prompt_tokens=100, completion_tokens=20, cached=False):
    return {"success": bool(steps), "steps": list(steps), "raw_response": json.dumps(steps), "model": model,
            "meta": {"cached": cached, "finish_reason": "stop", "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                     "latency_s": 0.5, "model": model}}


class Tier:
    """A fake planner for one model: its answer to the n-th sample of a part is given by a script of behaviours."""
    def __init__(self, model, script, calls=None):
        self.model, self.script, self.calls = model, script, calls if calls is not None else []

    def __call__(self, material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        self.calls.append((tuple(features), sample_id))
        how = self.script[sample_id % len(self.script)] if isinstance(self.script, list) else self.script
        if how == "good":
            return result(good(features), self.model)
        if how == "swap":
            return result(swapped(features), self.model)
        if how == "empty":
            return {"success": False, "steps": [], "raw_response": "[]", "model": self.model, "meta": {"model": self.model, "cached": False}}
        if how == "invented":
            return result(good(features)[:1] + ["Warp Drive"] + good(features)[1:], self.model)
        if how == "fatal":
            raise SY.FatalRunError("large tier down")
        raise AssertionError(how)


def run(part, small, large, level, repair="r1", **kw):
    casc = CASC.Cascade(level, large, kw.pop("large_repair", None))
    return SY.run_llm_first(part, n_samples=kw.pop("n", 4), planner=small, repair=repair, cascade=casc, selection="s3", **kw)


# ---------------------------------------------------------------- options and config
def test_options_parse_and_combine():
    assert PS.split_name("n5_r1_s3+ms") == ("n5_r1_s3", FS(model="small"))
    assert PS.split_name("n5_r1_s3+ml")[1].model == "large" and PS.split_name("n5_r1_s3+mp")[1].model == "primary"
    assert PS.split_name("n5_r1_s3+cas1")[1].cascade == 1 and PS.split_name("n5_r1_s3+cas2")[1].cascade == 2
    assert PS.split_name("n5_r1_s3+cas0")[1].cascade == 0
    sp = PS.split_name("n5_r1_s3+fs2+ts+cas2+fz1")[1]
    assert (sp.few_shot, sp.two_step, sp.cascade, sp.feas, sp.model) == (2, True, 2, 1, "primary")
    assert PS.split_name("b0+fs2+ml+h1")[1] == FS(few_shot=2, hint=True, model="large")


def test_option_names_round_trip_and_defaults():
    for name in ("ms", "ml", "cas1", "cas2", "fs2+ts+ms", "h1+cas2", "fs4x+ts+cas1"):
        assert PS.split_name("n5_r1_s3+" + name)[1].name == name
    assert PS.split_name("b0")[1].is_default and PS.split_name("b0+mp")[1].is_default and PS.split_name("b0+fz1")[1].is_default
    assert not PS.split_name("b0+ms")[1].is_default and not PS.split_name("b0+cas1")[1].is_default


def test_option_errors():
    for bad, msg in (("b0+ms+ml", "two model options"), ("b0+cas1+cas2", "two cascade options"), ("b0+cas1+ms", "cannot be combined"),
                     ("b0+ml+cas2", "cannot be combined"), ("b0+cas3", "unknown planner option")):
        with pytest.raises(ValueError, match=msg):
            PS.split_name(bad)
    with pytest.raises(ValueError, match="rb has no planner"):
        SY.parse_system("rb+ms")
    assert SY.parse_system("n5_r1_s3+cas1") == (5, "r1", "s3") and SY.planner_spec("n5_r1_s3+cas1").cascade == 1
    assert SY.reprompts("n5_r2_s3+cas1") and not SY.reprompts("n5_r1_s3+cas1")


def test_prices_and_cost():
    assert config.price(SMALL) == (0.075, 0.30) and config.price(LARGE) == (0.15, 0.60) and config.price("nobody/else") is None
    assert config.call_cost_usd(SMALL, 1_000_000, 1_000_000) == pytest.approx(0.375)
    assert config.call_cost_usd(LARGE, 2000, 500) == pytest.approx((2000 * 0.15 + 500 * 0.60) / 1e6)
    assert config.call_cost_usd(LARGE, None, None) == 0.0 and config.call_cost_usd("nobody/else", 5, 5) is None


def test_model_roles_and_overrides(monkeypatch):
    assert (config.model("primary"), config.model("small"), config.model("large")) == (PRIMARY, SMALL, LARGE)
    monkeypatch.setenv("CAPP_MODEL_SMALL", "x/tiny")
    assert config.model("small") == "x/tiny" and config.MODEL_ROLES == ("primary", "small", "large")


def test_escalation_and_keep_rules():
    # cas1: only a slot that ended up with the Route Builder's route; cas2: whenever the raw route was not accepted as it was
    for prov, raw_ok, c1, c2 in (("raw_llm", True, False, False), ("repaired_llm", False, False, True), ("route_builder", False, True, True),
                                 ("raw_llm", False, False, True), ("route_builder", True, True, False)):
        assert CASC.escalate(1, prov, raw_ok) is c1 and CASC.escalate(2, prov, raw_ok) is c2
    assert CASC.keep_large("route_builder", "route_builder") and CASC.keep_large("repaired_llm", "raw_llm")
    assert CASC.keep_large("repaired_llm", "repaired_llm") and not CASC.keep_large("repaired_llm", "route_builder")
    assert not CASC.keep_large("raw_llm", "repaired_llm")
    with pytest.raises(ValueError):
        CASC.Cascade(3, lambda *a, **k: None)
    with pytest.raises(ValueError):
        CASC.Cascade(1, None)


# ---------------------------------------------------------------- the loop, with fake planners
def test_no_escalation_when_the_small_model_is_right():
    part = make_parts(1)[0]
    small, large = Tier(SMALL, "good"), Tier(LARGE, "good")
    for level in (1, 2):
        rec = run(part, small, large, level)
        assert large.calls == [] and rec.info["escalations"] == 0 and rec.info["cascade_level"] == level
        assert all(c.cascade["escalated"] is False and c.cascade["tier"] == "small" and c.cascade["small_raw_ok"] for c in rec.candidates)
        assert rec.final["provenance"] == "raw_llm"


def test_without_a_cascade_nothing_changes():
    part = make_parts(1)[0]
    rec = SY.run_llm_first(part, n_samples=3, planner=Tier(SMALL, "good"), repair="r1")
    assert all(c.cascade is None for c in rec.candidates) and "cascade_level" not in rec.info and "escalations" not in rec.info
    d = rec.to_dict()
    assert all("cascade" not in c for c in d["candidates"])                 # the record on disk has no new key without a cascade
    assert all("cascade" in c for c in run(make_parts(1)[0], Tier(SMALL, "empty"), Tier(LARGE, "good"), 1, n=1).to_dict()["candidates"] if c["sample_id"] >= 0)


def test_cas1_asks_the_large_model_only_where_the_small_one_cannot_be_validated_or_repaired():
    part = make_parts(1)[0]
    small, large = Tier(SMALL, ["good", "swap", "empty", "invented"]), Tier(LARGE, "good")
    rec = run(part, small, large, 1, repair="r1")
    # R1 repairs the swapped and the invented route, so only the empty answer (nothing to repair) goes up
    assert large.calls == [(tuple(part["features"]), 2)] and rec.info["escalations"] == 1
    by = {c.sample_id: c for c in rec.candidates if c.sample_id >= 0}
    assert by[2].cascade["escalated"] and by[2].cascade["tier"] == "large" and by[2].provenance == "raw_llm" and by[2].meta["model"] == LARGE
    assert by[2].cascade["small"]["llm_ok"] is False and by[2].cascade["trigger"] == CASC.TRIGGER[1]
    assert by[1].provenance == "repaired_llm" and by[1].cascade["escalated"] is False and by[1].meta["model"] == SMALL
    assert rec.info["attempts"] == rec.info["llm_calls"] == 4


def test_cas1_with_r0_escalates_every_invalid_sample():
    part = make_parts(1)[0]
    small, large = Tier(SMALL, ["good", "swap"]), Tier(LARGE, "good")
    rec = run(part, small, large, 1, repair="r0")
    assert [c[1] for c in large.calls] == [1, 3] and rec.info["escalations"] == 2
    esc = [c for c in rec.candidates if c.cascade and c.cascade["escalated"]]
    assert all(c.provenance == "raw_llm" and c.cascade["small"]["provenance"] == "route_builder" and c.cascade["small"]["replaced"] for c in esc)
    # the counters are those of the answers that were served: the replacements of the small model were set aside
    assert rec.info["replaced"] == 0 and all(c.cascade["set_aside_counters"]["replaced"] == 1 for c in esc)
    assert rec.info["raw_v1_valid"] == 4


def test_cas2_asks_whenever_the_raw_route_is_not_accepted_and_keeps_the_better_result():
    part = make_parts(1)[0]
    small, large = Tier(SMALL, ["good", "swap", "empty", "invented"]), Tier(LARGE, "good")
    rec = run(part, small, large, 2)
    assert sorted(c[1] for c in large.calls) == [1, 2, 3] and rec.info["escalations"] == 3
    served = {c.sample_id: c for c in rec.candidates if c.sample_id >= 0}
    assert served[0].cascade["escalated"] is False
    for i in (1, 2, 3):
        assert served[i].cascade["tier"] == "large" and served[i].provenance == "raw_llm" and served[i].cascade["small_raw_ok"] is False
        assert served[i].cascade["large_raw_ok"] is True
    assert served[1].cascade["small"]["provenance"] == "repaired_llm" and served[1].cascade["kept"] == "large"      # a tie or better: the large model


def test_cas2_keeps_the_small_result_when_the_large_one_is_worse():
    part = make_parts(1)[0]
    small, large = Tier(SMALL, ["swap"]), Tier(LARGE, ["empty"])
    rec = run(part, small, large, 2, n=2)
    for c in rec.candidates:
        if c.sample_id < 0:
            continue
        assert c.cascade["escalated"] and c.cascade["kept"] == "small" and c.cascade["tier"] == "small"
        assert c.provenance == "repaired_llm" and c.cascade["large"]["llm_ok"] is False
    assert rec.info["escalations"] == 2 and rec.info["repaired"] == 2


def test_cas2_tie_goes_to_the_large_model():
    part = make_parts(1)[0]
    rec = run(part, Tier(SMALL, ["swap"]), Tier(LARGE, ["swap"]), 2, n=1)
    c = rec.candidates[0]
    assert c.cascade["tier"] == "large" and c.provenance == "repaired_llm" and c.cascade["small"]["provenance"] == "repaired_llm"


def test_both_tiers_failing_ends_with_the_route_builder():
    part = make_parts(1)[0]
    rec = run(part, Tier(SMALL, "empty"), Tier(LARGE, "empty"), 1, n=3)
    assert rec.info["safety_net"] is True and rec.final["provenance"] == "route_builder" and rec.final["check"]["valid"]
    assert rec.info["escalations"] == 3 and rec.info["llm_failed"] == 3          # the counters of the answers served: the large model's failures


def test_a_fatal_error_of_the_large_tier_stops_the_run():
    part = make_parts(1)[0]
    with pytest.raises(SY.FatalRunError, match="large tier down"):
        run(part, Tier(SMALL, "empty"), Tier(LARGE, "fatal"), 1)


def test_an_exception_of_the_large_planner_is_a_failed_sample_not_a_crash():
    def boom(*a, **k):
        raise RuntimeError("503")
    rec = run(make_parts(1)[0], Tier(SMALL, ["empty", "good"]), boom, 1, n=2)
    assert rec.candidates[0].cascade["escalated"] and rec.candidates[0].llm_ok is False and rec.info["escalations"] == 1


def test_r2_reprompts_go_to_the_tier_that_made_the_sample():
    part = make_parts(1)[0]
    seen = []

    def repairer(tag):
        def f(material, features, tolerance="0.05mm", batch_size=100, sample_id=0, previous_steps=(), violations=()):
            seen.append((tag, sample_id))
            return result(good(features), tag)
        return f
    small, large = Tier(SMALL, ["swap", "swap", "empty", "good"]), Tier(LARGE, ["swap", "swap", "swap", "good"])
    rec = run(part, small, large, 1, repair="r2", repair_planner=repairer("small"), large_repair=repairer("large"), n=4)
    # slots 0, 1: the small model's re-prompt repairs them; slot 2: nothing to re-prompt, the large model answers a swap and its re-prompt repairs it
    assert [t for t, _ in seen] == ["small", "small", "large"] and [i for _, i in seen] == [0, 1, 2]
    assert rec.info["repair_calls"] == 3 and rec.info["escalations"] == 1 and rec.info["repaired_r2"] == 3


def test_the_repair_calls_of_a_set_aside_attempt_are_still_counted():
    part = make_parts(1)[0]

    def failing_repair(material, features, tolerance="0.05mm", batch_size=100, sample_id=0, previous_steps=(), violations=()):
        return {"success": False, "steps": [], "raw_response": "no", "model": "x", "meta": {}}
    rec = run(part, Tier(SMALL, ["swap"]), Tier(LARGE, ["good"]), 1, repair="r2", repair_planner=failing_repair, n=2)
    assert rec.info["escalations"] == 2 and rec.info["repair_calls"] == 2 and rec.info["r2_failed"] == 0
    assert all(c.cascade["set_aside_counters"]["r2_failed"] == 1 for c in rec.candidates if c.sample_id >= 0)


def test_the_default_repair_planner_of_the_large_tier_is_the_reprompt_module(monkeypatch):
    from capp import reprompt
    got = []

    def spy(material, features, tolerance="0.05mm", batch_size=100, sample_id=0, previous_steps=(), violations=(), model=None):
        got.append(model)
        return result(good(features), "spy")
    monkeypatch.setattr(reprompt, "reprompt", spy)
    rec = SY.run_llm_first(make_parts(1)[0], n_samples=1, planner=Tier(SMALL, "empty"), repair="r2",
                           cascade=CASC.Cascade(1, Tier(LARGE, "swap")))
    assert got == [None] and rec.info["repaired_r2"] == 1


def test_the_record_is_json_serialisable():
    rec = run(make_parts(1)[0], Tier(SMALL, ["good", "swap", "empty"]), Tier(LARGE, "good"), 2)
    d = json.loads(json.dumps(rec.to_dict()))
    esc = [c for c in d["candidates"] if (c.get("cascade") or {}).get("escalated")]
    assert esc and all("cascade" not in (c["cascade"].get("small") or c["cascade"].get("large")) for c in esc)


def test_cascade_with_the_feasibility_levels():
    # a small-model route that is strict-valid but needs a hole (fz1 hard finding) is "not accepted as it is" at cas2
    part = dict(make_parts(1)[0], features=["Hole", "Reamed_Hole"], tolerance="0.01mm")
    rec = SY.run_llm_first(part, n_samples=2, planner=Tier(SMALL, "good"), repair="r1", feas=1,
                           cascade=CASC.Cascade(2, Tier(LARGE, "good")))
    assert rec.info["fz_level"] == 1 and rec.info["cascade_level"] == 2 and rec.final["check"]["valid"]


# ---------------------------------------------------------------- the real planner path: model ids, cache, cost
def features_in(messages):
    for m in messages:
        mt = re.search(r"Features: (.*)\n", m["content"]) if m["role"] == "user" else None
        if mt:
            return mt.group(1).split(", ")
    raise AssertionError("no features in the prompt")


def model_transport(log, small=("good", "swap", "empty", "good", "swap"), large=("good",)):
    """The small model errs by script (per part: its n-th call), the large one is right; every call is logged with its model."""
    counts = {}

    def transport(model, messages, params):
        feats = features_in(messages)
        repair = messages[-1]["content"].startswith("A verifier rejected")
        key = (model, tuple(feats), repair)
        i = counts.get(key, 0)
        counts[key] = i + 1
        log.append((model, "repair" if repair else "sample", tuple(feats)))
        script = small if model == SMALL else large
        how = "good" if repair else script[i % len(script)]
        text = {"good": json.dumps(good(feats)), "swap": json.dumps(swapped(feats)), "empty": "[]"}[how]
        pt, ct = (200, 30) if model == SMALL else (210, 45) if model == LARGE else (220, 50)
        return {"text": text, "prompt_tokens": pt + (150 if repair else 0), "completion_tokens": ct, "finish_reason": "stop"}
    llm_client.set_transport(transport)
    return log


def go(system, parts, tag="cs", **kw):
    return H.run_experiment(system, parts, tag=tag, progress=None, write=False, **kw)


def test_model_options_route_every_call_to_their_model():
    parts = make_parts(3)
    for system, model in (("n3_r2_s1", PRIMARY), ("n3_r2_s1+ms", SMALL), ("n3_r2_s1+ml", LARGE), ("n3_r2_s1+mp", PRIMARY)):
        log = model_transport([], small=("swap", "good"), large=("swap", "good"))
        out = go(system, parts, tag="m-" + system)
        assert {m for m, _, _ in log} == {model}, system                 # samples and R2 re-prompts alike
        assert any(k == "repair" for _, k, _ in log)
        assert out["run_info"]["model"] == model and out["run_info"]["model_role"] == {PRIMARY: "primary", SMALL: "small", LARGE: "large"}[model]
        models = {c["meta"]["model"] for r in out["records"] for c in r["candidates"] if c["meta"]}
        assert models == {model}
        r2 = [c["repair"]["r2"]["meta"]["model"] for r in out["records"] for c in r["candidates"] if (c.get("repair") or {}).get("r2")]
        assert r2 and set(r2) == {model}
        llm_client.reset_transport()


def test_the_default_system_keeps_the_v1_model_and_cache_keys():
    log = model_transport([])
    go("b1", make_parts(1), tag="k1")
    key_default = llm_client.get_call_log()[0]["key"]
    go("b1+mp", make_parts(1), tag="k1")                          # +mp is the default: the same request, from the cache
    assert [c["cached"] for c in llm_client.get_call_log()] == [False, True] and len(log) == 1
    assert llm_client.get_call_log()[1]["key"] == key_default


def test_a_cascade_is_the_small_and_the_large_runs_put_together_from_the_cache(monkeypatch):
    parts = make_parts(4)
    log = model_transport([])
    ms = go("n4_r1_s3+ms", parts)
    ml = go("n4_r1_s3+ml", parts)
    n_before = len(log)
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")                        # any sample call that is not in the cache is an error
    for level in (1, 2):
        out = go(f"n4_r1_s3+cas{level}", parts)
        assert len(log) == n_before                                      # no API call at all
        assert out["summary"]["cascade"]["level"] == level
        assert out["summary"]["M7"]["cache_hit_rate"] == 1.0
    # and what the cascade serves is what the two runs said
    for rec in out["records"]:
        for c in rec["candidates"]:
            if c.get("cascade") and c["llm_ok"]:
                assert c["meta"]["model"] == (LARGE if c["cascade"]["tier"] == "large" else SMALL)
    assert ms["summary"]["cost"]["by_model"].keys() == {SMALL} and ml["summary"]["cost"]["by_model"].keys() == {LARGE}


def test_cost_by_model_is_tokens_times_list_price():
    parts = make_parts(3)
    model_transport([], small=("good",), large=("good",))
    out = go("n2_r1_s3+ms", parts)
    c = out["summary"]["cost"]
    assert c["by_model"][SMALL]["calls"] == 6 and c["by_model"][SMALL]["prompt_tokens"] == 6 * 200 and c["by_model"][SMALL]["completion_tokens"] == 6 * 30
    assert c["by_model"][SMALL]["usd"] == pytest.approx((1200 * 0.075 + 180 * 0.30) / 1e6)
    assert c["usd_per_part"] == pytest.approx(c["usd_total"] / 3) and c["tokens_per_part"] == pytest.approx(6 * 230 / 3)
    assert c["unpriced_models"] == [] and c["assumed_model_calls"] == 0
    out = go("n2_r1_s3+ml", parts)
    assert out["summary"]["cost"]["usd_per_part"] > c["usd_per_part"] * 1.5          # the large model costs more per token and answers longer


def test_cascade_cost_counts_both_attempts():
    parts = make_parts(4)
    log = model_transport([], small=("empty", "good"), large=("good",))
    out = go("n4_r1_s3+cas1", parts)
    s = out["summary"]
    n_small = sum(1 for m, k, _ in log if m == SMALL and k == "sample")
    n_large = sum(1 for m, k, _ in log if m == LARGE and k == "sample")
    assert n_small == 16 and n_large == 8                                 # half of the small model's answers are empty: those slots escalate
    assert s["cost"]["by_model"][SMALL]["calls"] == 16 and s["cost"]["by_model"][LARGE]["calls"] == 8
    exp = (16 * 200 * 0.075 + 16 * 30 * 0.30 + 8 * 210 * 0.15 + 8 * 45 * 0.60) / 1e6
    assert s["cost"]["usd_total"] == pytest.approx(exp)
    c = s["cascade"]
    assert (c["slots"], c["escalated"], c["rescued"]) == (16, 8, 8) and c["escalation_rate"] == 0.5 and c["rescue_rate"] == 1.0
    assert c["small_accepted_as_is_rate"] == 0.5 and c["served_by_large"] == 8 and c["kept_small_after_escalation"] == 0
    assert s["M7"]["total_calls_per_part"] == pytest.approx((16 + 8) / 4) and s["M7"]["api_calls_per_part"] == pytest.approx((16 + 8) / 4)
    assert s["M2"]["final_strict_valid"] == 1.0 and s["M2"]["provenance_share"].get("route_builder", 0) == 0


def test_cas2_on_the_real_path_replaces_repaired_routes_by_raw_ones():
    parts = make_parts(4)
    model_transport([], small=("swap",), large=("good",))
    r1 = go("n3_r1_s3+cas1", parts)
    r2 = go("n3_r1_s3+cas2", parts)
    assert r1["summary"]["cascade"]["escalated"] == 0 and r1["summary"]["M2"]["provenance_share"] == {"repaired_llm": 1.0}
    assert r2["summary"]["cascade"]["escalated"] == 12 and r2["summary"]["M2"]["provenance_share"] == {"raw_llm": 1.0}
    assert r2["summary"]["cascade"]["parts_with_final_from_large"] == 1.0


def test_two_step_and_few_shot_cascade_use_the_right_model_in_each_call():
    parts = make_parts(2)
    log = []

    def transport(model, messages, params):
        log.append(model)
        sys_msg = messages[0]["content"]
        if sys_msg.startswith("You convert a machining plan"):
            order = re.search(r"FINAL ORDER: (.*)", messages[1]["content"]).group(1).split(" > ")
            return {"text": json.dumps({"operations": order}), "prompt_tokens": 80, "completion_tokens": 20, "finish_reason": "stop"}
        feats = features_in(messages)
        route = swapped(feats) if model == SMALL else good(feats)
        return {"text": "ok.\nFINAL ORDER: " + " > ".join(route), "prompt_tokens": 90, "completion_tokens": 40, "finish_reason": "stop"}
    llm_client.set_transport(transport)
    out = go("n2_r1_s3+ts+cas2", parts, tag="ts")
    assert set(log) == {SMALL, LARGE} and log.count(SMALL) == log.count(LARGE) == 8       # reason + formalise, for the two samples of two parts
    assert out["summary"]["cascade"]["escalated"] == 4 and out["summary"]["two_step"]["n"] == 4
    assert out["summary"]["cost"]["by_model"][SMALL]["calls"] == 8 and out["summary"]["cost"]["by_model"][LARGE]["calls"] == 8


def test_few_shot_prompts_are_the_same_for_both_tiers():
    parts = D.load_dataset("dev_v2")[:3]
    seen = {}

    def transport(model, messages, params):
        seen.setdefault(json.dumps(messages, sort_keys=True), set()).add(model)
        feats = features_in(messages)
        return {"text": "[]" if model == SMALL else json.dumps(good(feats)), "prompt_tokens": 1, "completion_tokens": 1, "finish_reason": "stop"}
    llm_client.set_transport(transport)
    go("n1_r1_s1+fs2+cas1", parts, tag="fs")
    assert seen and all(m == {SMALL, LARGE} for m in seen.values())


def test_a_model_without_a_price_gives_no_dollar_figure(monkeypatch):
    monkeypatch.setenv("CAPP_MODEL_LARGE", "x/unpriced")
    model_transport([], small=("empty",), large=("good",))
    # the fake answers by the ids it knows: give the unknown one the large model's behaviour
    orig = llm_client._transport

    def mapped(model, messages, params):
        return orig(LARGE if model == "x/unpriced" else model, messages, params)
    llm_client.set_transport(mapped)
    out = go("n1_r1_s3+cas1", make_parts(2), tag="np")
    c = out["summary"]["cost"]
    assert c["usd_per_part"] is None and c["unpriced_models"] == ["x/unpriced"] and c["by_model"][SMALL]["usd"] is not None
    assert "n/a (no list price for x/unpriced)" in SM.format_summary("s", out["summary"])


def test_old_records_without_model_or_cascade_keys_still_summarise():
    model_transport([], small=("good",))
    out = go("n2_r1_s3", make_parts(2), tag="old")
    recs = json.loads(json.dumps(out["records"]))
    for r in recs:
        for c in r["candidates"]:
            c.pop("cascade", None)
            c["meta"].pop("model", None)
    s = SM.summarize(recs)
    assert s["cascade"] == {"n": 0}
    assert s["cost"]["assumed_model_calls"] == 4 and list(s["cost"]["by_model"]) == [PRIMARY]
    assert "assumed the primary model" in SM.format_summary("old", s)
    assert compare.table([("old", s)]).count("cascade") == 2          # two cascade rows, with "-" in them


def test_rows_and_csv_of_a_cascade_run(tmp_path):
    model_transport([], small=("empty", "good"))
    out = H.run_experiment("n3_r1_s3+cas1", make_parts(3), tag="csvrun", out_dir=str(tmp_path), progress=None)
    pr = SM.part_rows(out["records"])
    assert all(r["final_tier"] in ("small", "large", "route_builder") for r in pr)
    assert sum(r["escalations"] for r in pr) == out["summary"]["cascade"]["escalated"] > 0
    cr = SM.candidate_rows(out["records"])
    assert {r["cascade_tier"] for r in cr} <= {"small", "large", ""} and all("model" in r for r in cr)
    rows = open(os.path.join(out["out_dir"], "per_candidate.csv"), encoding="utf-8").read().splitlines()
    assert "cascade_tier" in rows[0] and "escalated" in rows[0]
    info = json.load(open(os.path.join(out["out_dir"], "run.json"), encoding="utf-8"))
    assert info["cascade_level"] == 1 and info["cascade_models"] == {"small": SMALL, "large": LARGE} and info["model"] == SMALL
    assert info["price_usd_per_mtok"][SMALL] == [0.075, 0.3]
    assert "cascade_tier" not in SM.candidate_rows(go("n1_r1_s3", make_parts(1), tag="plain")["records"])[0]


def test_printed_summary_and_comparison_table_have_the_new_rows():
    model_transport([], small=("empty", "good"))
    a = go("n3_r1_s3+cas1", make_parts(3), tag="p1")
    b = go("n3_r1_s3+ms", make_parts(3), tag="p1")
    text = SM.format_summary("n3_r1_s3+cas1", a["summary"])
    assert "cascade         : cas1 (small -> large model)" in text and "cost (list)" in text and "USD/part" in text
    assert "cascade" not in SM.format_summary("n3_r1_s3+ms", b["summary"]).replace("cost (list)", "")
    table = compare.table([("cas1", a["summary"]), ("ms", b["summary"])])
    assert "cost: list-price USD / part" in table and "models called" in table and "gpt-oss-120b+gpt-oss-20b" in table
    assert "cascade: escalated sample slots" in table and "cascade: final route from large model" in table


def test_throttle_counts_the_live_calls_of_a_set_aside_attempt(monkeypatch):
    slept = []
    count_sleeps(monkeypatch, slept)
    model_transport([], small=("empty",))
    go("n1_r1_s3+cas1", make_parts(2), tag="th", throttle=0.5)
    assert slept == [0.5, 0.5]
    slept.clear()
    go("n1_r1_s3+cas1", make_parts(2), tag="th", throttle=0.5)             # all cached now: no live call, no pause
    assert slept == []
    live = {"cached": False}
    rec = {"candidates": [{"meta": {"cached": True}, "cascade": {"small": {"meta": live}}}]}
    assert H._live_call_made(rec) is True
    assert H._live_call_made({"candidates": [{"meta": {"cached": True}, "cascade": {"large": {"meta": {"cached": True}}}}]}) is False


def test_the_cascade_harness_needs_a_factory_that_takes_a_role():
    with pytest.raises(H.HarnessError, match="planner_factory"):
        H.run_experiment("n1_r1_s3+cas1", make_parts(1), planner=Tier(SMALL, "good"), progress=None, write=False)
    seen = []

    def factory(part, role=None):
        seen.append(role)
        return Tier(LARGE if role == "large" else SMALL, "good" if role == "large" else "empty")
    out = H.run_experiment("n2_r1_s3+cas1", make_parts(1), planner_factory=factory, progress=None, write=False)
    assert seen == ["small", "large"] and out["summary"]["cascade"]["escalated"] == 2


def test_replay_of_a_cascade_run_is_identical(monkeypatch):
    parts = make_parts(3)
    model_transport([])
    first = go("n4_r1_s3+cas2", parts, tag="rp")
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    again = go("n4_r1_s3+cas2", parts, tag="rp")
    assert [r["final"]["steps"] for r in again["records"]] == [r["final"]["steps"] for r in first["records"]]
    assert again["summary"]["M7"]["cache_hit_rate"] == 1.0 and again["summary"]["cascade"]["escalated"] == first["summary"]["cascade"]["escalated"]


# ---------------------------------------------------------------- Pareto
def P(label, usd, share, gap=1.0):
    return {"label": label, "usd_per_part": usd, "llm_route_share": share, "efficiency_gap": gap}


def test_frontier_logic():
    pts = [P("a", 1.0, 0.5), P("b", 2.0, 0.9), P("c", 3.0, 0.8), P("d", 2.0, 0.9), P("e", 0.5, 0.5), P("f", None, 1.0)]
    f = pareto.frontier(pts, "usd_per_part", "llm_route_share", True)
    assert f == {"e", "b", "d"}                       # a is dominated by e, c by b; b and d tie; f has no price
    q = pareto.frontier([P("a", 1, 0, 3.0), P("b", 2, 0, 1.0), P("c", 3, 0, 2.0)], "usd_per_part", "efficiency_gap", False)
    assert q == {"a", "b"}
    assert pareto.frontier([], "usd_per_part", "llm_route_share", True) == set()


@pytest.fixture()
def runs_of_four():
    model_transport([], small=("good", "swap", "empty", "good", "swap"))
    parts = make_parts(6)
    res = {s: go(f"n4_r1_s3+{s}", parts, tag="par")["records"] for s in ("ms", "ml", "cas1", "cas2")}
    return [(f"n4_r1_s3+{s}", res[s]) for s in ("ms", "ml", "cas1", "cas2")]


def test_analyse_orders_the_systems_by_cost(runs_of_four):
    res = pareto.analyse(runs_of_four)
    pts = {p["label"].split("+")[1]: p for p in res["points"]}
    assert res["comparable"] and all(p["usd_per_part"] is not None for p in pts.values())
    assert pts["ms"]["usd_per_part"] < pts["cas1"]["usd_per_part"] <= pts["cas2"]["usd_per_part"] < pts["ml"]["usd_per_part"] + pts["ms"]["usd_per_part"]
    assert pts["ml"]["llm_route_share"] == 1.0 and pts["ms"]["llm_route_share"] == 1.0          # R1 repairs everything but an empty answer
    assert pts["cas1"]["escalation_rate"] > 0 and pts["ms"]["escalation_rate"] is None
    assert pts["cas2"]["escalation_rate"] >= pts["cas1"]["escalation_rate"]
    assert "n4_r1_s3+ms" in res["frontier_validity"]
    text = pareto.render(res)
    assert "frontier" in text and "n4_r1_s3+cas2" in text and "USD/part" in text


def test_price_of_validity_steps():
    res = {"points": [dict(P("s", 1.0, 0.6), on_validity_frontier=True), dict(P("c", 1.5, 0.9), on_validity_frontier=True),
                      dict(P("l", 4.0, 0.95), on_validity_frontier=True), dict(P("x", 5.0, 0.5), on_validity_frontier=False)]}
    pts = res["points"]
    for p in pts:
        p.update({"models": [], "unpriced": [], "tokens_per_part": 1, "latency_s_per_part": 1, "raw_strict_valid": 1, "m4_lcs": 1,
                  "escalation_rate": None, "final_from_large": None, "on_quality_frontier": False})
    res.update({"comparable": True, "price_of_validity": [], "frontier_validity": ["s", "c", "l"], "frontier_quality": []})
    assert "WARNING" not in pareto.render(res)
    res["comparable"] = False
    assert "not comparable" in pareto.render(res)


def test_analyse_gives_the_price_of_each_extra_point(runs_of_four):
    # a model that is wrong more often leaves Route Builder routes: make the small one fail its repair (r0 system)
    model_transport([], small=("swap", "good"))
    parts = make_parts(6)
    runs = [(f"n4_r0_s3+{s}", go(f"n4_r0_s3+{s}", parts, tag="r0")["records"]) for s in ("ms", "ml", "cas1")]
    res = pareto.analyse(runs)
    p = {x["label"].split("+")[1]: x for x in res["points"]}
    assert p["ms"]["llm_route_share"] < 1.0 and p["ml"]["llm_route_share"] == 1.0 and p["cas1"]["llm_route_share"] == 1.0
    assert p["ms"]["on_validity_frontier"] and p["cas1"]["on_validity_frontier"] and not p["ml"]["on_validity_frontier"]
    assert [(s["from"], s["to"]) for s in res["price_of_validity"]] == [("n4_r0_s3+ms", "n4_r0_s3+cas1")]
    st = res["price_of_validity"][0]
    assert st["usd_per_part_per_point"] == pytest.approx(st["extra_usd_per_part"] / st["extra_llm_route_points"]) and st["extra_usd_per_part"] > 0
    assert "Price of validity" in pareto.render(res)


def test_runs_over_different_parts_are_flagged(runs_of_four):
    a, b = runs_of_four[0], runs_of_four[1]
    res = pareto.analyse([a, (b[0], b[1][:3])])
    assert res["comparable"] is False and "not comparable" in pareto.render(res)


def test_cascade_pareto_cli(tmp_path, runs_of_four, capsys):
    dirs = []
    for label, recs in runs_of_four:
        d = tmp_path / label.replace("+", "_")
        d.mkdir()
        H.write_outputs(str(d), recs, SM.summarize(recs), {"system": label})
        dirs.append(str(d))
    out_csv, out_json = tmp_path / "p.csv", tmp_path / "p.json"
    assert cascade_pareto.main(dirs + ["--csv", str(out_csv), "--json", str(out_json)]) == 0
    text = capsys.readouterr().out
    assert "n4_r1_s3+cas1" in text and "USD/part" in text
    rows = out_csv.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("label,") and len(rows) == 5
    assert set(json.loads(out_json.read_text(encoding="utf-8"))) >= {"points", "comparable", "frontier_validity", "price_of_validity"}
    assert cascade_pareto.main([dirs[0], dirs[0]]) == 1                                  # the same system twice
    assert cascade_pareto.main([str(tmp_path / "nowhere")]) == 1


def test_cascade_pareto_plot(tmp_path, runs_of_four):
    pytest.importorskip("matplotlib")
    path = pareto.plot(pareto.analyse(runs_of_four), str(tmp_path / "pareto.png"))
    assert os.path.getsize(path) > 2000
