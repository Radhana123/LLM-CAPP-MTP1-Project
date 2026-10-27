import itertools
import json
import os
import random
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import dataset as D            # noqa: E402
from capp import feasibility as F        # noqa: E402
from capp import feasrepair as FR        # noqa: E402
from capp import harness as H            # noqa: E402
from capp import legacy                  # noqa: E402,F401
from capp import plannerspec as PS       # noqa: E402
from capp import repair as R             # noqa: E402
from capp import summary as SM           # noqa: E402
from capp import systems as S            # noqa: E402
from capp import verify as V             # noqa: E402
import precedence_graph as pg            # noqa: E402
import route_builder as rb               # noqa: E402

IG = "Internal Grooving"
DIMS = {"diameter": 30, "length": 60, "depth": 20, "width": 24}


def part(features=("Hole", "Groove"), tolerance="0.05mm", dims=None, material="Steel", pid="T1"):
    return {"part_id": pid, "material": material, "features": list(features), "tolerance": tolerance, "batch_size": 50,
            "dims": dict(DIMS if dims is None else dims), "required_ra_um": None, "machine_preference": "auto"}


def ids(steps, p):
    return sorted({v.rule_id[5:] for v in F.evaluate(steps, p).violations})


def hard(steps, p):
    return [v for v in FR.repairable_hard(F.evaluate(steps, p).violations)]


# ================================================================ order_repair with extra edges
def test_extra_edges_default_is_the_old_behaviour():
    ops = ["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"]
    pos = {op: i for i, op in enumerate(["Facing", "Reaming", "Drilling", "Center Drilling", "Inspection"])}
    assert R.order_repair(ops, pos) == R.order_repair(ops, pos, ()) == R.order_repair(ops, pos, [])


def test_extra_edge_is_obeyed_even_against_the_llm_order():
    ops = ["Facing", "Drilling", IG, "Inspection"]
    pos = {"Facing": 0, IG: 1, "Drilling": 2, "Inspection": 3}                 # the LLM put the groove first
    assert R.order_repair(ops, pos)[0].index(IG) < R.order_repair(ops, pos)[0].index("Drilling")      # no edge: the LLM order stays
    order = R.order_repair(ops, pos, [("Drilling", IG)])[0]
    assert order.index("Drilling") < order.index(IG)


def test_extra_edges_with_an_absent_operation_are_ignored():
    ops = ["Facing", "Drilling", "Inspection"]
    pos = {op: i for i, op in enumerate(ops)}
    assert R.order_repair(ops, pos, [("Boring", "Reaming")]) == R.order_repair(ops, pos)


def test_contradicting_extra_edge_gives_no_order():
    ops = ["Facing", "Center Drilling", "Drilling", "Inspection"]
    assert R.order_repair(ops, {}, [("Drilling", "Center Drilling")]) is None            # the graph has Center Drilling -> Drilling


def test_order_repair_with_extra_edges_is_exact_against_brute_force():
    rng = random.Random(11)
    ops = ["Facing", "Center Drilling", "Drilling", IG, "Chamfering", "Inspection"]
    extra = [("Drilling", IG), ("Chamfering", IG)]
    for _ in range(6):
        order0 = ops[:]
        rng.shuffle(order0)
        pos = {op: i for i, op in enumerate(order0)}
        res = R.order_repair(ops, pos, extra)
        assert res is not None
        best = None
        for perm in itertools.permutations(ops):
            idx = {op: i for i, op in enumerate(perm)}
            if idx["Facing"] != 0 or idx["Inspection"] != len(ops) - 1:
                continue
            if any(idx[a] > idx[b] for a, b in list(pg.PRECEDENCE_EDGES) + extra if a in idx and b in idx):
                continue
            disc = sum(1 for x, y in itertools.combinations(perm, 2) if pos[x] > pos[y])
            cost = (disc, V.implied_changeovers(list(perm)))
            best = cost if best is None or cost < best else best
        assert (res[1], res[2]) == best


# ================================================================ r1_fz: what it fixes
def test_precedence_valid_but_infeasible_route_is_repaired_by_reordering():
    p = part()
    route = ["Facing", IG, "Center Drilling", "Drilling", "Inspection"]
    assert V.check_route(route, p["features"]).valid and "HOLE-EXISTS" in ids(route, p)
    plain = R.r1(route, p["features"])
    assert plain.ok and plain.steps == route                                       # R1 alone leaves it
    rep = FR.r1_fz(route, p["features"], p)
    assert rep.ok and rep.steps == ["Facing", "Center Drilling", "Drilling", IG, "Inspection"]
    assert rep.fz_edges == [["Drilling", IG]] and rep.fz_inserted == [] and rep.fz_rounds == 1
    assert hard(rep.steps, p) == []


def test_missing_hole_is_inserted():
    p = part(("Groove",))
    rep = FR.r1_fz(["Facing", IG, "Inspection"], ["Groove"], p)
    assert rep.ok and rep.steps == ["Facing", "Drilling", IG, "Inspection"]
    assert rep.fz_inserted == ["Drilling"] and "Drilling" in rep.inserted
    assert V.check_route(rep.steps, ["Groove"]).valid


def test_an_existing_boring_counts_as_the_hole():
    p = part(("Groove", "Bore"))
    route = ["Facing", IG, "Plain/Cylindrical Turning", "Center Drilling", "Drilling", "Boring", "Inspection"]
    rep = FR.r1_fz(route, p["features"], p)
    assert rep.ok and hard(rep.steps, p) == []
    assert "Boring" in rep.steps and rep.steps.index(IG) > min(rep.steps.index("Drilling"), rep.steps.index("Boring"))


def test_lathe_operation_after_parting_off_is_moved_before_it():
    p = part(("Parting_Off", "Groove", "Hole"))
    route = ["Facing", "Center Drilling", "Drilling", "Parting-off", IG, "Inspection"]
    assert "SEQ-PARTOFF" in ids(route, p)
    rep = FR.r1_fz(route, p["features"], p)
    assert rep.ok and rep.steps.index(IG) < rep.steps.index("Parting-off") and hard(rep.steps, p) == []
    assert ["Internal Grooving", "Parting-off"] in rep.fz_edges


def test_a_route_without_findings_comes_back_exactly_as_r1_returns_it():
    p = part(("Hole",))
    route = ["Facing", "Center Drilling", "Drilling", "Inspection"]
    a, b = R.r1(route, p["features"]), FR.r1_fz(route, p["features"], p)
    assert (b.ok, b.steps, b.route, b.inserted, b.discordant_pairs) == (a.ok, a.steps, a.route, a.inserted, a.discordant_pairs)
    assert b.fz_inserted == [] and b.fz_edges == [] and b.fz_rounds == 0


def test_fz_never_deletes_a_valid_operation():
    p = part(("Hole", "Groove"))
    route = ["Facing", IG, "Chamfering", "Center Drilling", "Drilling", "Inspection"]
    rep = FR.r1_fz(route, p["features"], p)
    assert set(route) <= set(rep.steps)


def test_idempotent_and_deterministic():
    p = part()
    route = ["Facing", IG, "Center Drilling", "Drilling", "Inspection"]
    a = FR.r1_fz(route, p["features"], p, risks=True)
    b = FR.r1_fz(route, p["features"], p, risks=True)
    assert a.to_dict() == b.to_dict()
    again = FR.r1_fz(a.steps, p["features"], p, risks=True)
    assert again.steps == a.steps and again.fz_inserted == [] and again.fz_edges == [] and again.fz_rounds == 0


# ================================================================ levels: hard only, or also the risk insertions
def test_fz1_flags_risks_fz2_inserts_for_them():
    p = part(("Hole",), tolerance="0.02mm")
    route = ["Facing", "Center Drilling", "Drilling", "Inspection"]
    assert ids(route, p) == ["TOL-HOLE"]
    one = FR.r1_fz(route, p["features"], p, risks=False)
    two = FR.r1_fz(route, p["features"], p, risks=True)
    assert one.ok and one.steps == route and one.flagged == ["TOL-HOLE"]
    assert two.ok and two.steps == ["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"] and two.fz_inserted == ["Reaming"]
    assert ids(two.steps, p) == [] and two.remaining == []


def test_fz2_hole_pilot():
    # the v1 chain for Hole always starts with Center Drilling; a pilot finding needs a route that has Drilling for another reason
    p = part(("Groove",), tolerance="0.05mm")
    two = FR.r1_fz(["Facing", IG, "Inspection"], ["Groove"], p, risks=True)
    assert two.ok and two.steps[:3] == ["Facing", "Center Drilling", "Drilling"] and set(two.fz_inserted) == {"Drilling", "Center Drilling"}
    one = FR.r1_fz(["Facing", IG, "Inspection"], ["Groove"], p, risks=False)
    assert one.steps == ["Facing", "Drilling", IG, "Inspection"] and one.flagged == ["HOLE-PILOT"]


def test_parameter_and_support_findings_are_only_flagged():
    p = part(("Turning",), dims={"diameter": 20, "length": 200, "depth": 10, "width": 20})         # L/D = 10
    route = ["Facing", "Plain/Cylindrical Turning", "Inspection"]
    rep = FR.r1_fz(route, p["features"], p, risks=True)
    assert rep.ok and rep.steps == route and "HOLD-STEADY" in rep.flagged


def test_machine_capability_is_reported_and_does_not_block_the_repair():
    p = part(("Turning",), dims={"diameter": 400, "length": 100, "depth": 10, "width": 20})
    route = ["Facing", "Plain/Cylindrical Turning", "Inspection"]
    assert "CAP-LATHE-DIA" in ids(route, p)
    rep = FR.r1_fz(route, p["features"], p)
    assert rep.ok and rep.unrepairable == ["CAP-LATHE-DIA"] and rep.steps == R.r1(route, p["features"]).steps
    assert FR.repairable_hard(F.evaluate(route, p).violations) == []


def test_no_dimensions_the_dimension_free_rules_still_run():
    p = part(("Groove",), dims={})
    p["dims"] = None
    rep = FR.r1_fz(["Facing", IG, "Inspection"], ["Groove"], p)
    assert rep.ok and "Drilling" in rep.steps


def test_unorderable_constraint_reports_failure(monkeypatch):
    monkeypatch.setattr(pg, "PRECEDENCE_EDGES", list(pg.PRECEDENCE_EDGES) + [("Parting-off", IG)])       # contradicts SEQ-PARTOFF
    p = part(("Parting_Off", "Groove", "Hole"))
    rep = FR.r1_fz(["Facing", "Center Drilling", "Drilling", "Parting-off", IG, "Inspection"], p["features"], p)
    assert not rep.ok and "no valid ordering" in rep.reason


def test_a_rule_that_cannot_be_acted_on_is_not_an_infinite_loop(monkeypatch):
    # a finding whose action wants an operation that is already in the route and in place must end the loop
    bad = V.Violation("risk", "x", ops=("Drilling",), rule_id="PHYS:HOLE-PILOT", action="insert:Center Drilling")
    monkeypatch.setattr(F, "evaluate", lambda steps, part, machine=F.DEFAULT_MACHINE: F.Report(violations=[bad]))
    p = part(("Hole",))
    rep = FR.r1_fz(["Facing", "Center Drilling", "Drilling", "Inspection"], ["Hole"], p, risks=True)
    assert rep.ok and rep.fz_rounds == 0


# ================================================================ properties on the development set
def _dev():
    return D.load_dataset("dev_v2")


def _base(p):
    """The Route Builder's first route of a part, or None (a few feature sets have no route within max_routes=1; see Step 5)."""
    routes = rb.generate_valid_routes(p["features"], max_routes=15)
    return V.clean_steps(routes[0]["steps"]) if routes else None


def test_every_route_builder_variant_with_a_hard_finding_is_fixed():
    fixed = 0
    for p in _dev():
        for r in rb.generate_valid_routes(p["features"], max_routes=15):
            steps = V.clean_steps(r["steps"])
            if hard(steps, p):
                rep = FR.r1_fz(steps, p["features"], p)
                assert rep.ok and hard(rep.steps, p) == [], (p["part_id"], steps, rep.reason)
                assert V.check_route(rep.steps, p["features"]).valid
                fixed += 1
    assert fixed >= 40                                                           # BASELINE.md: 60 of 2637 variants carry HOLE-EXISTS


def test_random_damage_is_repaired_to_a_strict_valid_route_without_fixable_findings():
    rng = random.Random(5)
    n = 0
    for p in _dev()[:60]:
        base = _base(p)
        if base is None:
            continue
        for _ in range(3):
            bad = list(base)
            rng.shuffle(bad)
            if rng.random() < 0.5 and len(bad) > 3:
                bad.pop(rng.randrange(len(bad)))
            for risks in (False, True):
                rep = FR.r1_fz(bad, p["features"], p, risks=risks)
                assert rep.ok, (p["part_id"], bad, rep.reason)
                assert V.check_route(rep.steps, p["features"]).valid
                left = F.evaluate(rep.steps, p).violations
                assert FR.repairable_hard(left) == []
                if risks:
                    assert FR.insertable_risks(left) == []
                n += 1
    assert n >= 300


def test_fz_only_adds_operations_to_r1():
    """R1+ = R1's operation set plus the operations the rules asked for; nothing else changes the set."""
    rng = random.Random(9)
    for p in _dev()[:40]:
        base = _base(p)
        if base is None:
            continue
        bad = list(base)
        rng.shuffle(bad)
        a, b = R.r1(bad, p["features"]), FR.r1_fz(bad, p["features"], p, risks=True)
        assert set(b.steps) == set(a.steps) | set(b.fz_inserted)
        assert len(b.steps) == len(a.steps) + len(b.fz_inserted)
        assert b.inserted == a.inserted + b.fz_inserted


# ================================================================ the system names
def test_parse_system_fz():
    assert S.parse_system("n5_r1_s3+fz1") == (5, "r1", "s3") and S.planner_spec("n5_r1_s3+fz2").feas == 2
    assert S.planner_spec("n5_r1_s3").feas == 0 and S.planner_spec("n5_r1_s3+fz0").feas == 0
    assert S.parse_system("b0_r2+fz1") == (5, "r2", "s1") and S.parse_system("n5_r3_s3+fs2+h1+fz2") == (5, "r3", "s3")
    assert S.planner_spec("n5_r1_s3+fz1").is_default and S.planner_spec("n5_r1_s3+fz1").name == ""      # not a prompt option


@pytest.mark.parametrize("bad", ["b0+fz1", "n5_r0_s1+fz2", "rb+fz1", "n5_r1_s3+fz1+fz2", "n5_r1_s3+fz3", "n5_r1_s3+fz"])
def test_parse_system_rejects(bad):
    with pytest.raises(ValueError):
        S.parse_system(bad)


def test_run_llm_first_rejects_bad_levels():
    with pytest.raises(ValueError):
        S.run_llm_first(part(), planner=const(["Facing", "Inspection"]), repair="r0", feas=1)
    with pytest.raises(ValueError):
        S.run_llm_first(part(), planner=const(["Facing", "Inspection"]), repair="r1", feas=3)


# ================================================================ in a run
def _meta():
    return {"cached": False, "finish_reason": "stop", "prompt_tokens": 100, "completion_tokens": 20, "latency_s": 0.1}


def const(route):
    def planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        return {"success": True, "steps": list(route), "raw_response": str(route), "model": "fake", "meta": _meta()}
    return planner


def per_sample(routes):
    def planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        r = routes[sample_id % len(routes)]
        return {"success": True, "steps": list(r), "raw_response": str(r), "model": "fake", "meta": _meta()}
    return planner


BAD = ["Facing", IG, "Center Drilling", "Drilling", "Inspection"]      # strict-valid, HOLE-EXISTS


def run(p, planner, system="n1_r1_s1", **kw):
    spec = S.parse_system(system)
    return S.run_llm_first(p, n_samples=spec[0], planner=planner, system=system, repair=spec[1], selection=spec[2],
                           feas=S.planner_spec(system).feas, **kw)


def test_without_fz_a_precedence_valid_infeasible_route_is_kept():
    p = part()
    rec = run(p, const(BAD))
    assert rec.final["steps"] == BAD and rec.final["provenance"] == "raw_llm"
    assert "fz_level" not in rec.info and "fz_pool" not in rec.info


def test_fz1_repairs_it_and_records_what_it_did():
    p = part()
    rec = run(p, const(BAD), "n1_r1_s1+fz1")
    assert rec.final["steps"] == ["Facing", "Center Drilling", "Drilling", IG, "Inspection"] and rec.final["provenance"] == "repaired_llm"
    assert rec.info["fz_level"] == 1 and rec.info["fz_raw_hard"] == 1 and rec.info["fz_repaired"] == 1 and rec.info["repaired_r1"] == 1
    assert rec.info["fz_errors"] == 0 and rec.info["fz_pool"] == 1 and rec.info["fz_filtered"] == 0
    cand = rec.candidates[0]
    assert cand.repaired and cand.repair["strategy"] == "r1" and cand.repair["r1"]["fz_edges"] == [["Drilling", IG]]
    assert cand.first_pass["valid"]                                          # the first-pass check is still the precedence-level one
    assert H._feasibility(V.clean_steps(rec.final["steps"]), p)["route_infeasible"] == 0


def test_fz1_leaves_risks_alone_fz2_acts_on_them():
    p = part(("Hole",), tolerance="0.02mm")
    route = ["Facing", "Center Drilling", "Drilling", "Inspection"]
    one = run(p, const(route), "n1_r1_s1+fz1")
    two = run(p, const(route), "n1_r1_s1+fz2")
    assert one.final["steps"] == route and one.final["provenance"] == "raw_llm" and one.info["fz_raw_risk"] == 0
    assert "Reaming" in two.final["steps"] and two.final["provenance"] == "repaired_llm" and two.info["fz_raw_risk"] == 1


def test_a_feasible_route_is_untouched_at_every_level():
    p = part(("Hole",), tolerance="0.05mm")
    route = ["Facing", "Center Drilling", "Drilling", "Inspection"]
    base = run(p, const(route)).to_dict()
    for system in ("n1_r1_s1+fz1", "n1_r1_s1+fz2", "n1_r3_s1+fz2"):
        rec = run(p, const(route), system).to_dict()
        assert rec["final"] == base["final"]
        assert [c["steps"] for c in rec["candidates"]] == [c["steps"] for c in base["candidates"]]
        assert rec["info"]["fz_raw_hard"] == 0 and rec["info"]["fz_repaired"] == 0


def test_on_the_dev_set_fz1_changes_nothing_when_the_llm_route_is_the_route_builders():
    """The Route Builder's first routes have no hard findings (BASELINE.md Step 11), so fz1 must reproduce n1_r1_s1 exactly."""
    def rb_planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        steps = _base({"features": features}) or ["Facing", "Inspection"]
        return {"success": True, "steps": steps, "raw_response": "", "model": "fake", "meta": _meta()}
    for p in _dev()[:30]:
        a = run(p, rb_planner).to_dict()
        b = run(p, rb_planner, "n1_r1_s1+fz1").to_dict()
        assert a["final"] == b["final"] and a["front"] == b["front"], p["part_id"]


def test_hard_finding_that_cannot_be_repaired_falls_back_to_r0(monkeypatch):
    def failing(steps, features, part_, risks=False, machine=F.DEFAULT_MACHINE):
        return FR.FeasRepair(reason="forced")
    monkeypatch.setattr(FR, "r1_fz", failing)
    p = part()
    rec = run(p, const(BAD), "n1_r1_s1+fz1")
    assert rec.final["provenance"] == "route_builder" and rec.candidates[0].replaced
    assert rec.candidates[0].repair["fallback"] == "r0" and rec.info["r1_failed"] == 1


def test_a_risk_that_cannot_be_repaired_keeps_the_llm_route(monkeypatch):
    def failing(steps, features, part_, risks=False, machine=F.DEFAULT_MACHINE):
        return FR.FeasRepair(reason="forced")
    monkeypatch.setattr(FR, "r1_fz", failing)
    p = part(("Hole",), tolerance="0.02mm")
    route = ["Facing", "Center Drilling", "Drilling", "Inspection"]
    rec = run(p, const(route), "n1_r1_s1+fz2")
    assert rec.final["steps"] == route and rec.final["provenance"] == "raw_llm"
    assert rec.info["fz_risk_unfixed"] == 1 and rec.info["replaced"] == 0


def test_selection_prefers_the_candidate_without_risk(monkeypatch):
    """The Drilling-only route is cheaper, so S1 picks it; fz2 prefers the route that meets the tolerance."""
    def failing(steps, features, part_, risks=False, machine=F.DEFAULT_MACHINE):
        return FR.FeasRepair(reason="forced")
    monkeypatch.setattr(FR, "r1_fz", failing)
    p = part(("Hole",), tolerance="0.02mm")
    cheap = ["Facing", "Center Drilling", "Drilling", "Inspection"]
    careful = ["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"]
    planner = per_sample([cheap, careful])
    plain = run(p, planner, "n2_r1_s1+fz1")
    pref = run(p, planner, "n2_r1_s1+fz2")
    assert plain.final["steps"] == cheap                                     # fz1: risks do not count, v1 efficiency decides
    assert pref.final["steps"] == careful and pref.info["fz_pool"] == 1 and pref.info["fz_filtered"] == 1


def test_selection_prefers_the_candidate_without_hard_finding(monkeypatch):
    """A route R1 could not repair (hard finding left, R0 not used because it is a risk-level failure here) loses to a clean one."""
    p = part(("Hole", "Groove"))
    cands = [S.Candidate(0, "a", "raw_llm", BAD), S.Candidate(1, "b", "raw_llm", ["Facing", "Center Drilling", "Drilling", IG, "Inspection"])]
    info = {}
    keys = [S._fz_key(c, p, 1, info) for c in cands]
    assert keys == [(1, 0), (0, 0)]


def test_fz_key_ignores_what_cannot_tell_routes_apart():
    p = part(("Turning",), dims={"diameter": 400, "length": 100, "depth": 10, "width": 20}, tolerance="0.005mm")       # CAP-LATHE-DIA + TOL-SPEC
    c = S.Candidate(0, "a", "raw_llm", ["Facing", "Plain/Cylindrical Turning", "Inspection"])
    assert S._fz_key(c, p, 2, {}) == (0, 0)


def test_rule_error_is_counted_not_raised(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("rule bug")
    monkeypatch.setattr(F, "evaluate", boom)
    rec = run(part(), const(BAD), "n1_r1_s1+fz1")
    assert rec.info["fz_errors"] >= 1 and rec.final["steps"] == BAD          # nothing found, nothing changed, and the run goes on


def test_r2_report_carries_the_feasibility_finding_and_the_answer_must_clear_it():
    p = part()
    seen = {}

    def good(material, features, tolerance, batch_size, sample_id=0, previous_steps=(), violations=()):
        seen["msgs"] = [v.message for v in violations]
        steps = ["Facing", "Center Drilling", "Drilling", IG, "Inspection"]
        return {"success": True, "steps": steps, "raw_response": str(steps), "model": "fake", "meta": _meta()}

    rec = run(p, const(BAD), "n1_r2_s1+fz1", repair_planner=good)
    assert any("no hole to work on" in m for m in seen["msgs"])
    assert rec.final["provenance"] == "repaired_llm" and rec.candidates[0].repair["stage"] == "r2" and rec.info["repair_calls"] == 1

    def same_mistake(material, features, tolerance, batch_size, sample_id=0, previous_steps=(), violations=()):
        return {"success": True, "steps": list(BAD), "raw_response": str(BAD), "model": "fake", "meta": _meta()}

    rec = run(p, const(BAD), "n1_r2_s1+fz1", repair_planner=same_mistake)
    r2 = rec.candidates[0].repair["r2"]
    assert not r2["ok"] and "HOLE-EXISTS" in r2["reason"] and rec.final["provenance"] == "route_builder" and rec.info["r2_failed"] == 1


def test_r3_falls_back_to_the_feasibility_aware_r1():
    p = part()

    def same_mistake(material, features, tolerance, batch_size, sample_id=0, previous_steps=(), violations=()):
        return {"success": True, "steps": list(BAD), "raw_response": "", "model": "fake", "meta": _meta()}
    rec = run(p, const(BAD), "n1_r3_s1+fz1", repair_planner=same_mistake)
    assert rec.candidates[0].repair["tried"] == ["r2", "r1"] and rec.candidates[0].repair["stage"] == "r1"
    assert rec.final["steps"] == ["Facing", "Center Drilling", "Drilling", IG, "Inspection"]


def test_a_risk_alone_never_costs_an_llm_reprompt():
    p = part(("Hole",), tolerance="0.02mm")
    calls = []

    def spy(*a, **k):
        calls.append(1)
        return {"success": False, "steps": [], "raw_response": ""}
    rec = run(p, const(["Facing", "Center Drilling", "Drilling", "Inspection"]), "n1_r2_s1+fz2", repair_planner=spy)
    assert calls == [] and rec.info["repair_calls"] == 0 and rec.final["provenance"] == "raw_llm" and rec.info["fz_risk_unfixed"] == 1


# ================================================================ harness and summary
def test_harness_and_summary_for_an_fz_system():
    parts = _dev()[:8]
    out = H.run_experiment("n2_r1_s1+fz2", parts, tag="t", write=False, progress=None, planner=per_sample([BAD, ["Facing", "Inspection"]]))
    assert out["run_info"]["feasibility_repair"] == 2 and out["run_info"]["planner_options"] == "v1"
    fz = out["summary"]["feasibility_repair"]
    assert fz["n"] == 8 and fz["level"] == 2 and fz["rule_errors"] == 0
    assert "feasibility fix" in SM.format_summary("x", out["summary"])
    rows = SM.part_rows(out["records"])
    assert all("fz_pool" in r for r in rows)


def test_a_system_without_fz_has_no_fz_output():
    parts = _dev()[:4]
    out = H.run_experiment("n2_r1_s1", parts, tag="t", write=False, progress=None, planner=per_sample([BAD, ["Facing", "Inspection"]]))
    assert out["summary"]["feasibility_repair"] == {"n": 0} and out["run_info"]["feasibility_repair"] == 0
    assert "feasibility fix" not in SM.format_summary("x", out["summary"])
    assert all("fz_pool" not in r for r in SM.part_rows(out["records"]))


def test_fz_lowers_the_infeasible_share_on_damaged_routes():
    """The A7 effect in miniature: the same LLM answers, with and without +fz1, measured by the feasibility block of the summary."""
    def groove_first(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        steps = _base({"features": features}) or ["Facing", "Inspection"]
        if "Groove" in features:                                           # a plausible LLM habit: an internal groove right after Facing
            steps = [steps[0], IG] + [s for s in steps[1:] if s != IG]
        return {"success": True, "steps": steps, "raw_response": "", "model": "fake", "meta": _meta()}
    parts = [p for p in _dev() if "Groove" in p["features"]][:25]
    assert parts
    base = H.run_experiment("n1_r1_s1", parts, tag="t", write=False, progress=None, planner=groove_first)["summary"]["feasibility"]
    fixed = H.run_experiment("n1_r1_s1+fz1", parts, tag="t", write=False, progress=None, planner=groove_first)["summary"]["feasibility"]
    assert base["final_infeasible_share"] > 0
    assert fixed["final_infeasible_share"] == 0


def test_check_script_runs_offline_and_its_numbers_hold(tmp_path):
    out = tmp_path / "fzc.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "feasrepair_check.py"), "--dataset", "dev_v2", "--limit", "40",
                        "--damage", "2", "--json", str(out)], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    a, b = d["A"], d["B"]
    assert a["fz1_failed"] == 0 and a["fz1_fixed"] == a["with_fixable_hard"]
    assert a["fz2_fixed"] == a["with_insertable_risk"]
    assert b["fz1"]["still_hard_infeasible"] == 0 and b["fz2"]["still_hard_infeasible"] == 0
    assert b["fz2"]["still_insertable_risk"] == 0
    assert b["fz1"]["share_repaired"] == b["plain"]["share_repaired"] == 1.0
    assert "A. Route Builder routes" in r.stdout and "B. " in r.stdout
