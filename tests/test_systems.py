import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import dataset as D          # noqa: E402
from capp import systems as S          # noqa: E402
from capp import verify as V           # noqa: E402
import agents                          # noqa: E402
import llm_planner                     # noqa: E402
import nsga2 as v1_nsga2               # noqa: E402
import route_builder as rb             # noqa: E402

MK = V.MARKER


def part(features, material="Steel", batch=100, dims=None, pid="P1", tol="0.02mm"):
    return {"part_id": pid, "material": material, "features": features, "tolerance": tol, "batch_size": batch,
            "dims": dims or dict(S.DEFAULT_DIMS), "machine_preference": "auto"}


def _meta():
    return {"cached": False, "finish_reason": "stop", "prompt_tokens": 300, "completion_tokens": 40, "latency_s": 0.4}


def fake_planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
    """Deterministic stand-in for the LLM: the output depends only on features and sample_id."""
    routes = rb.generate_valid_routes(features, max_routes=15)
    base = V.clean_steps(routes[0]["steps"]) if routes else ["Facing", "Inspection"]
    alt = V.clean_steps(routes[1]["steps"]) if len(routes) > 1 else base
    k = sample_id % 5
    if k == 0:
        steps = base
    elif k == 1:
        steps = list(reversed(base))                      # order violations
    elif k == 2:
        steps = base[:-1]                                 # Inspection missing
    elif k == 3:
        return {"success": False, "steps": [], "raw_response": "API Error: boom"}
    else:
        steps = alt
    return {"success": True, "steps": steps, "raw_response": str(steps), "model": "fake", "meta": _meta()}


def all_bad_planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
    steps = ["Inspection", "Facing"]
    return {"success": True, "steps": steps, "raw_response": str(steps), "model": "fake", "meta": _meta()}


def all_fail_planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
    return {"success": False, "steps": [], "raw_response": "API Error: down"}


# ---- RB ---------------------------------------------------------------------------------------------------
def test_rb_is_deterministic_valid_and_has_no_llm_cost():
    p = part(["Hole", "Keyway", "Taper"])
    a, b = S.run_rb(p).to_dict(), S.run_rb(p).to_dict()
    assert a == b
    assert a["final"]["check"]["valid"] and a["final"]["provenance"] == "route_builder"
    assert a["info"]["llm_calls"] == 0 and len(a["candidates"]) == 15


def test_rb_without_any_route_is_flagged_incomplete(monkeypatch):
    monkeypatch.setattr(rb, "generate_valid_routes", lambda *a, **k: [])
    rec = S.run_rb(part(["Hole"]))
    assert rec.info["fallback_incomplete"] and rec.final["steps"] == S.INCOMPLETE_FALLBACK
    assert not rec.final["check"]["valid"]


# ---- B0 / B1 behaviour ------------------------------------------------------------------------------------
def test_b0_provenance_and_counts():
    rec = S.run_b0(part(["Hole", "Chamfer"]), planner=fake_planner)
    c = rec.candidates
    by_name = {x.name: x for x in c}
    assert by_name["Route_LLM_1"].provenance == "raw_llm" and not by_name["Route_LLM_1"].replaced
    assert by_name["Route_LLM_2_corrected"].provenance == "route_builder" and by_name["Route_LLM_2_corrected"].replaced
    assert by_name["Route_LLM_3_corrected"].replaced
    assert by_name["Route_LLM_4_failed"].llm_ok is False and not by_name["Route_LLM_4_failed"].in_pool
    assert rec.info["attempts"] == 5 and rec.info["llm_calls"] == 5
    assert rec.info["raw_v1_valid"] == 2 and rec.info["replaced"] == 2 and rec.info["llm_failed"] == 1
    assert rec.info["safety_net"] is False and rec.final["check"]["valid"]


def test_deduplicated_samples_still_count_for_first_pass_validity():
    # sample 0 and 4 are both the Route Builder's routes; when they coincide the second is dropped from selection
    p = part(["Chamfer", "Face"])
    rec = S.run_b0(p, planner=fake_planner)
    parsed = [c for c in rec.candidates if c.llm_ok]
    assert len(parsed) == 4
    dropped = [c for c in rec.candidates if c.deduped]
    assert all(c.llm_ok and c.first_pass is not None and not c.in_pool for c in dropped)
    assert rec.info["deduped"] == len(dropped)
    assert rec.info["population"] == len([c for c in rec.candidates if c.in_pool])


def test_b1_makes_exactly_one_call_and_matches_b0_sample_0():
    calls = []

    def spy(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        calls.append(sample_id)
        return fake_planner(material, features, tolerance, batch_size, sample_id)

    p = part(["Hole", "Slot"])
    rec = S.run_b1(p, planner=spy)
    assert calls == [0] and rec.info["attempts"] == 1 and rec.system == "b1"
    b0 = S.run_b0(p, planner=fake_planner)
    first = [c for c in b0.candidates if c.sample_id == 0][0]
    assert rec.candidates[0].llm_steps == first.llm_steps


def test_safety_net_when_every_sample_fails_or_is_unusable():
    for planner in (all_fail_planner,):
        rec = S.run_b0(part(["Hole", "Chamfer"]), planner=planner)
        assert rec.info["safety_net"] and rec.info["llm_failed"] == 5
        assert all(c.provenance == "route_builder" for c in rec.candidates if c.in_pool)
        assert rec.final["check"]["valid"] and rec.final["name"].startswith("Route_Builder_Fallback_")


def test_all_invalid_samples_are_replaced_by_the_route_builder():
    rec = S.run_b0(part(["Hole", "Chamfer"]), planner=all_bad_planner)
    assert rec.info["raw_v1_valid"] == 0 and rec.info["replaced"] == 5
    assert rec.info["deduped"] == 4                     # five identical replacement routes collapse to one
    assert rec.final["provenance"] == "route_builder" and rec.final["replaced"]


def test_v1_replacement_can_find_no_route_on_complex_parts():
    """v1 asks the Route Builder for ONE route; for Valve Body it builds none, so replacement fails (a v1 weakness)."""
    feats = ["Bore", "Hole", "Counterbore", "Internal_Thread", "Face_Groove", "Face", "Slot", "Chamfer"]
    assert rb.generate_valid_routes(feats, max_routes=1) == []
    rec = S.run_b0(part(feats), planner=all_bad_planner)
    assert rec.info["replacement_failed"] == 5 and rec.info["safety_net"]
    assert rec.final["check"]["valid"]                  # the safety net (max_routes=30) rescues the part


def test_first_pass_uses_strict_check_and_v1_check_separately():
    def invented(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        steps = V.clean_steps(rb.generate_valid_routes(features, max_routes=1)[0]["steps"])
        steps = steps[:-1] + ["Deburring", steps[-1]]
        return {"success": True, "steps": steps, "raw_response": "x", "model": "f", "meta": _meta()}

    rec = S.run_b1(part(["Hole", "Chamfer"]), planner=invented)
    c = rec.candidates[0]
    assert c.v1_valid is True and c.first_pass["valid"] is False and c.first_pass["kinds"] == ["invented"]
    # v1 keeps it, so the final route is a raw LLM route that the strict verifier rejects
    assert rec.final["provenance"] == "raw_llm" and rec.final["check"]["valid"] is False


def test_scores_asis_vs_eval_expose_the_missing_changeover_charge():
    p = part(["Turning", "Slot"], material="Aluminum", dims={"diameter": 30, "length": 100, "depth": 10, "width": 24})
    llm = ["Facing", "Plain/Cylindrical Turning", "Slot Milling", "Inspection"]     # no marker, as an LLM writes it

    def planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        return {"success": True, "steps": llm, "raw_response": "x", "model": "f", "meta": _meta()}

    rec = S.run_b1(p, planner=planner)
    c = rec.candidates[0]
    assert c.provenance == "raw_llm"
    assert c.scores["time_min"] < c.scores_eval["time_min"]
    delta = c.scores_eval["time_min"] - c.scores["time_min"]
    # +15 min changeover, minus the one position change (1 min) the v1 agent skips for the step right after a marker
    assert delta == pytest.approx(agents.CHANGEOVER_TIME_MIN - agents.POSITION_CHANGE_TIME_MIN)
    assert c.scores_eval["cost_inr"] > c.scores["cost_inr"]
    assert c.scores_eval["efficiency_score"] < c.scores["efficiency_score"]


def test_final_selection_is_highest_efficiency_of_first_front():
    rec = S.run_b0(part(["Hole", "Keyway", "Taper"]), planner=fake_planner)
    pool = {c.name: c for c in rec.candidates if c.in_pool}
    front = [pool[n] for n in rec.front]
    best = max(front, key=lambda c: c.scores["efficiency_score"])
    assert rec.final["name"] == best.name


# ---- faithfulness to v1 ------------------------------------------------------------------------------------
V1_PARTS = [
    (["Hole", "Chamfer"], "Aluminum", 500), (["Thread", "Fillet"], "Steel", 50), (["Hole", "Keyway", "Taper"], "Steel", 50),
    (["Pocket", "Chamfer"], "Brass", 100), (["Step", "Keyway", "Chamfer", "Face"], "Cast Iron", 10),
    (["Bore", "Hole", "Counterbore", "Internal_Thread", "Face_Groove", "Face", "Slot", "Chamfer"], "Brass", 250),
    (["Chamfer", "Face"], "Titanium", 2500), (["Eccentric", "Turning", "Step", "Keyway", "Chamfer"], "Steel", 5),
]


@pytest.mark.parametrize("planner", [fake_planner, all_bad_planner, all_fail_planner])
@pytest.mark.parametrize("features, material, batch", V1_PARTS)
def test_b0_reproduces_v1_run_nsga2_and_ui_selection(monkeypatch, planner, features, material, batch):
    dims = {"diameter": 40.0, "length": 120.0, "depth": 15.0, "width": 30.0}
    monkeypatch.setattr(llm_planner, "generate_process_plan", planner)

    # v1: run_nsga2 (llm_first), then week6/app.py: highest efficiency among the first front, scored with the dimensions
    pareto = v1_nsga2.run_nsga2(material, batch, features=features, machine_preference="auto", max_routes=30,
                                tolerance="0.02mm", llm_attempts=5, route_generation_mode="llm_first")
    status = v1_nsga2.run_nsga2.last_llm_status
    best, best_eff = None, -1
    for ind in pareto:
        t = agents.time_agent(ind.steps, material, dims)
        c = agents.cost_agent(ind.steps, material, batch, dims)
        e = agents.energy_agent(ind.steps, material, dims)
        eff = agents.efficiency_agent(t, c, e)
        if eff > best_eff:
            best, best_eff = ind, eff

    rec = S.run_b0(part(features, material, batch, dims=dims, tol="0.02mm"), planner=planner)

    assert rec.front == [i.route_name for i in pareto]
    assert rec.final["name"] == best.route_name and rec.final["steps"] == best.steps
    assert rec.info["raw_v1_valid"] == status["llm_valid"]
    assert rec.info["replaced"] == status["llm_corrected"]
    assert rec.info["safety_net"] == status["route_builder_fallback_used"]
    assert rec.info["fallback_incomplete"] == status["fallback_incomplete"]


# ---- R1 variants (b0_r1, b1_r1) ---------------------------------------------------------------------------
def no_answer_repair(material, features, tolerance="0.05mm", batch_size=100, sample_id=0, previous_steps=(), violations=()):
    return {"success": False, "steps": [], "raw_response": "API Error: none", "model": "fake"}


def invented_planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
    """v1-valid but strict-invalid: a valid route with an invented operation and a repeated one."""
    base = V.clean_steps(rb.generate_valid_routes(features, max_routes=1)[0]["steps"])
    steps = base[:1] + ["Deburring"] + base[1:-1] + [base[1]] + base[-1:]
    return {"success": True, "steps": steps, "raw_response": str(steps), "model": "fake", "meta": _meta()}


def test_r1_variant_repairs_and_labels_provenance():
    rec = S.run_b0_r1(part(["Hole", "Chamfer"]), planner=fake_planner)
    by_name = {x.name: x for x in rec.candidates}
    c2 = by_name["Route_LLM_2_repaired"]
    assert c2.provenance == "repaired_llm" and c2.repaired and not c2.replaced
    assert c2.repair["strategy"] == "r1" and c2.repair["stage"] == "r1" and c2.repair["r1"]["ok"]
    assert V.check_route(c2.steps, ["Hole", "Chamfer"]).valid
    c3 = by_name["Route_LLM_3_repaired"]
    assert c3.repaired and "Inspection" in c3.repair["r1"]["inserted"]
    assert by_name["Route_LLM_1"].provenance == "raw_llm"
    assert rec.info["repaired"] == 2 and rec.info["replaced"] == 0 and rec.info["r1_failed"] == 0
    assert rec.info["raw_v1_valid"] == 2 and rec.final["check"]["valid"]


def test_r1_variant_uses_the_strict_criterion_but_b0_keeps_v1s():
    p = part(["Hole", "Chamfer"])
    b0 = S.run_b0(p, planner=invented_planner)
    r1 = S.run_b0_r1(p, planner=invented_planner)
    assert b0.info["replaced"] == 0 and b0.final["provenance"] == "raw_llm"        # v1 accepts the invented operation
    assert not b0.final["check"]["valid"] and b0.info["raw_strict_valid"] == 0
    assert r1.info["repaired"] >= 1 and r1.final["check"]["valid"]                  # R1 does not
    assert r1.final["provenance"] == "repaired_llm"
    assert "Deburring" not in r1.final["steps"]


def test_r1_falls_back_to_r0_when_r1_gives_up(monkeypatch):
    monkeypatch.setattr(S.R, "r1", lambda steps, features: S.R.Repair(ok=False, reason="forced"))
    rec = S.run_b0_r1(part(["Hole", "Chamfer"]), planner=fake_planner)
    assert rec.info["repaired"] == 0 and rec.info["r1_failed"] == 2 and rec.info["replaced"] == 2
    c2 = {x.name: x for x in rec.candidates}["Route_LLM_2_corrected"]
    assert c2.replaced and c2.provenance == "route_builder" and c2.repair["fallback"] == "r0" and c2.repair["r1"]["reason"] == "forced"
    assert rec.final["check"]["valid"]


def test_r1_rescues_the_parts_where_r0_finds_no_route():
    from capp import archetypes as AR
    feats = next(a["features"] for a in AR.ARCHETYPES if a["name"] == "Valve Body")
    assert rb.generate_valid_routes(feats, max_routes=1) == [] and rb.generate_valid_routes(feats, max_routes=30)
    p = part(feats)
    r0, r1 = S.run_b0(p, planner=all_bad_planner), S.run_b0_r1(p, planner=all_bad_planner)
    assert r0.info["replacement_failed"] >= 1
    assert r1.info["replacement_failed"] == 0 and r1.info["repaired"] >= 1
    assert r1.final["check"]["valid"] and r1.final["provenance"] == "repaired_llm"


def test_b0_r1_and_b0_see_the_same_llm_outputs_and_first_pass_numbers():
    p = part(["Hole", "Chamfer", "Slot"])
    a, b = S.run_b0(p, planner=fake_planner), S.run_b0_r1(p, planner=fake_planner)
    assert [c.llm_steps for c in a.candidates] == [c.llm_steps for c in b.candidates]
    assert a.info["raw_v1_valid"] == b.info["raw_v1_valid"] and a.info["raw_strict_valid"] == b.info["raw_strict_valid"]
    assert a.references == b.references


def test_unknown_repair_strategy_is_rejected():
    with pytest.raises(ValueError):
        S.run_llm_first(part(["Hole"]), planner=fake_planner, repair="r9")


def test_system_registry_has_the_repair_variants():
    names = {"rb", "b1", "b0", "b1_r1", "b0_r1", "b1_r2", "b0_r2", "b1_r3", "b0_r3"}
    assert names == set(S.SYSTEMS) and set(S.LLM_SYSTEMS) == names - {"rb"}
    assert set(S.REPROMPT_SYSTEMS) == {"b1_r2", "b0_r2", "b1_r3", "b0_r3"}
    for name, fn in S.SYSTEMS.items():
        if name != "rb":
            assert fn(part(["Hole"]), planner=fake_planner, repair_planner=no_answer_repair).system == name


# ---- R2 (re-prompt) and R3 (R2, then R1) ------------------------------------------------------------------
class Repairer:
    """Stand-in for capp.reprompt.reprompt that records what it was asked and answers from a callable."""
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def __call__(self, material, features, tolerance="0.05mm", batch_size=100, sample_id=0, previous_steps=(), violations=()):
        self.calls.append({"sample_id": sample_id, "previous": list(previous_steps), "violations": list(violations)})
        steps = self.answer(features, previous_steps)
        return {"success": bool(steps), "steps": steps, "raw_response": str(steps), "model": "fake", "meta": _meta()}


def fixes_it(features, previous):
    return V.clean_steps(rb.generate_valid_routes(features, max_routes=1)[0]["steps"])


def still_wrong(features, previous):
    return ["Inspection", "Facing"]


def test_r2_accepts_a_valid_answer_and_counts_the_extra_calls():
    rp = Repairer(fixes_it)
    rec = S.run_b0_r2(part(["Hole", "Chamfer"]), planner=fake_planner, repair_planner=rp)
    by_name = {x.name: x for x in rec.candidates}
    c2 = by_name["Route_LLM_2_repaired"]
    assert c2.provenance == "repaired_llm" and c2.repaired and c2.repair["stage"] == "r2" and c2.repair["tried"] == ["r2"]
    assert c2.repair["r2"]["ok"] and V.check_route(c2.steps, ["Hole", "Chamfer"]).valid
    assert rec.info["repair_calls"] == len(rp.calls) == 2 and rec.info["repaired_r2"] == 2 and rec.info["r2_failed"] == 0
    assert rec.info["llm_calls"] == 5 and rec.info["replaced"] == 0
    assert all(c["violations"] and all(v.hard for v in c["violations"]) for c in rp.calls)
    assert rp.calls[0]["previous"] == c2.llm_steps or rp.calls[1]["previous"] == c2.llm_steps


def test_r2_is_not_called_for_samples_that_are_already_valid():
    rp = Repairer(fixes_it)
    rec = S.run_b0_r2(part(["Hole"]), planner=lambda *a, **k: fake_planner(*a, **dict(k, sample_id=0)), repair_planner=rp)
    assert rp.calls == [] and rec.info["repair_calls"] == 0 and rec.info["repaired"] == 0


def test_r2_that_is_still_invalid_falls_back_to_r0():
    rp = Repairer(still_wrong)
    rec = S.run_b0_r2(part(["Hole", "Chamfer"]), planner=fake_planner, repair_planner=rp)
    c2 = {x.name: x for x in rec.candidates}["Route_LLM_2_corrected"]
    assert c2.replaced and c2.provenance == "route_builder" and c2.repair["fallback"] == "r0" and c2.repair["stage"] is None
    assert not c2.repair["r2"]["ok"] and "still invalid" in c2.repair["r2"]["reason"]
    assert rec.info["r2_failed"] == 2 and rec.info["replaced"] == 2 and rec.info["repaired"] == 0
    assert rec.final["check"]["valid"]


def test_r2_answer_with_no_route_or_an_exception_is_a_failure_not_a_crash():
    rec = S.run_b0_r2(part(["Hole", "Chamfer"]), planner=fake_planner, repair_planner=Repairer(lambda f, p: []))
    assert rec.info["r2_failed"] == 2 and rec.info["replaced"] == 2 and rec.final["check"]["valid"]

    def raising(*a, **k):
        raise RuntimeError("boom")
    rec = S.run_b0_r2(part(["Hole", "Chamfer"]), planner=fake_planner, repair_planner=raising)
    assert rec.info["r2_failed"] == 2 and rec.info["replaced"] == 2 and rec.final["check"]["valid"]


def test_a_fatal_error_from_the_repair_planner_stops_the_run():
    def fatal(*a, **k):
        raise S.FatalRunError("stop")
    with pytest.raises(S.FatalRunError):
        S.run_b0_r2(part(["Hole", "Chamfer"]), planner=fake_planner, repair_planner=fatal)


def test_r3_stops_at_r2_when_r2_works_and_never_runs_r1(monkeypatch):
    monkeypatch.setattr(S.R, "r1", lambda *a, **k: pytest.fail("R1 must not run when R2 succeeded"))
    rec = S.run_b0_r3(part(["Hole", "Chamfer"]), planner=fake_planner, repair_planner=Repairer(fixes_it))
    c2 = {x.name: x for x in rec.candidates}["Route_LLM_2_repaired"]
    assert c2.repair["strategy"] == "r3" and c2.repair["stage"] == "r2" and "r1" not in c2.repair
    assert rec.info["repaired_r2"] == 2 and rec.info["repaired_r1"] == 0


def test_r3_falls_back_to_r1_on_the_original_route_when_r2_fails():
    rp = Repairer(still_wrong)
    rec = S.run_b0_r3(part(["Hole", "Chamfer"]), planner=fake_planner, repair_planner=rp)
    c2 = {x.name: x for x in rec.candidates}["Route_LLM_2_repaired"]
    assert c2.repaired and c2.provenance == "repaired_llm" and c2.repair["stage"] == "r1"
    assert c2.repair["tried"] == ["r2", "r1"] and not c2.repair["r2"]["ok"] and c2.repair["r1"]["ok"]
    assert V.check_route(c2.steps, ["Hole", "Chamfer"]).valid
    assert rec.info["r2_failed"] == 2 and rec.info["repaired_r1"] == 2 and rec.info["replaced"] == 0
    assert rec.info["repair_calls"] == 2


def test_r3_uses_r0_only_as_the_last_resort(monkeypatch):
    monkeypatch.setattr(S.R, "r1", lambda *a, **k: S.R.Repair(ok=False, reason="forced"))
    rec = S.run_b0_r3(part(["Hole", "Chamfer"]), planner=fake_planner, repair_planner=Repairer(still_wrong))
    c2 = {x.name: x for x in rec.candidates}["Route_LLM_2_corrected"]
    assert c2.replaced and c2.repair["tried"] == ["r2", "r1"] and c2.repair["fallback"] == "r0"
    assert rec.info["replaced"] == 2 and rec.info["r1_failed"] == 2 and rec.info["r2_failed"] == 2


def test_r2_and_r3_see_the_same_samples_as_b0():
    p = part(["Hole", "Chamfer", "Slot"])
    base = S.run_b0(p, planner=fake_planner)
    for fn in (S.run_b0_r2, S.run_b0_r3):
        rec = fn(p, planner=fake_planner, repair_planner=Repairer(fixes_it))
        assert [c.llm_steps for c in rec.candidates] == [c.llm_steps for c in base.candidates]
        assert rec.info["raw_strict_valid"] == base.info["raw_strict_valid"]


def test_b1_variants_use_one_sample():
    rec = S.run_b1_r2(part(["Hole", "Chamfer"]), planner=lambda *a, **k: fake_planner(*a, **dict(k, sample_id=1)),
                      repair_planner=Repairer(fixes_it))
    assert rec.info["attempts"] == 1 and rec.info["repair_calls"] == 1 and rec.system == "b1_r2"


def test_r2_only_uses_hard_violations_in_the_report():
    rp = Repairer(fixes_it)
    # valid order but badly grouped machines (soft) AND an invented operation (hard)
    def planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        steps = ["Facing", "Plain/Cylindrical Turning", "Pocket Milling", "Grooving/Necking", "Slot Milling", "Deburring", "Inspection"]
        return {"success": True, "steps": steps, "raw_response": str(steps), "model": "fake", "meta": _meta()}
    S.run_b1_r2(part(["Turning", "Pocket", "Groove", "Slot"]), planner=planner, repair_planner=rp)
    assert rp.calls and {v.kind for v in rp.calls[0]["violations"]} == {"invented"}
