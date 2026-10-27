import copy
import csv
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import compare as C            # noqa: E402
from capp import dataset as D            # noqa: E402
from capp import feasibility as F        # noqa: E402
from capp import harness as H            # noqa: E402
from capp import legacy                  # noqa: E402,F401
from capp import llm_client              # noqa: E402
from capp import physics                 # noqa: E402
from capp import summary as SM           # noqa: E402
from capp import systems                 # noqa: E402
from capp import verify as V             # noqa: E402

TURN = "Plain/Cylindrical Turning"
DIMS = {"diameter": 30, "length": 100, "depth": 10, "width": 24}


def part(features=("Turning",), tolerance="0.05mm", dims=None, material="Steel", ra=None, **kw):
    p = {"part_id": "T1", "material": material, "features": list(features), "tolerance": tolerance, "batch_size": 50,
         "dims": dict(DIMS if dims is None else dims), "required_ra_um": ra}
    p.update(kw)
    return p


def ids(rep):
    return [v.rule_id[5:] for v in rep.violations]


def one(rep, rid):
    found = [v for v in rep.violations if v.rule_id == f"PHYS:{rid}"]
    assert len(found) == 1, (rid, found)
    return found[0]


@pytest.fixture(scope="module")
def parts():
    return D.load_dataset("dev_v2")


@pytest.fixture(autouse=True)
def no_hook_left_behind():
    yield
    F.uninstall()


# ---- registry --------------------------------------------------------------------------------------------------------------
def test_registry_has_fifteen_documented_rules():
    assert len(F.RULES) == 15 and len({r.id for r in F.RULES}) == 15
    for r in F.RULES:
        assert r.kind in ("infeasible", "risk") and r.scope in ("route", "part")
        assert r.category in ("capability", "surface", "workholding", "hole", "finishing", "sequence")
        assert r.source_status in ("web", "derived", "model", "assumption")
        assert r.source and r.summary and r.action
        if r.source_status == "web":
            assert "http" in r.source, r.id                       # a web source names its page
    assert {r.category for r in F.RULES} == {"capability", "surface", "workholding", "hole", "finishing", "sequence"}


def test_no_rule_claims_a_handbook_source_and_hard_rules_have_a_basis():
    assert not any("handbook" == r.source_status for r in F.RULES)
    for r in F.RULES:
        if r.kind == "infeasible":
            assert r.source_status in ("derived", "assumption"), r.id         # a hard rule rests on logic of the vocabulary or on the machine data sheet
    assert F.NOT_ENCODED and all(F.NOT_ENCODED.values())


def test_rule_table_is_json_serialisable():
    t = F.rule_table()
    assert json.loads(json.dumps(t)) == t and len(t) == 15


def test_ra_floor_rule_is_labelled_an_assumption():
    assert F.RULE_BY_ID["RA-FLOOR"].source_status == "assumption" and F.RULE_BY_ID["RA-FLOOR"].kind == "risk"


# ---- violation format -------------------------------------------------------------------------------------------------------
def test_risk_is_a_soft_kind_and_infeasible_is_hard():
    assert V.Violation("risk", "x").hard is False and V.Violation("infeasible", "x").hard is True
    assert "risk" in V.KINDS and "risk" in V.SOFT_KINDS and "infeasible" in V.HARD_KINDS
    assert V.Violation("order", "x").action == ""


def test_findings_are_verify_violations_with_rule_ids_positions_and_actions():
    rep = F.evaluate(["Facing", "Parting-off", "Grooving/Necking", "Inspection"], part(("Groove", "Parting_Off")))
    v = one(rep, "SEQ-PARTOFF")
    assert isinstance(v, V.Violation) and v.kind == "infeasible" and v.hard
    assert v.ops == ("Grooving/Necking",) and v.positions == (2,) and v.action.startswith("reorder")


# ---- capability ---------------------------------------------------------------------------------------------------------------
def test_lathe_diameter_and_length_limits():
    steps = ["Facing", TURN, "Inspection"]
    assert "CAP-LATHE-DIA" not in ids(F.evaluate(steps, part(dims=dict(DIMS, diameter=250))))
    rep = F.evaluate(steps, part(dims=dict(DIMS, diameter=251)))
    assert one(rep, "CAP-LATHE-DIA").kind == "infeasible" and "251" in one(rep, "CAP-LATHE-DIA").message
    assert "CAP-LATHE-LEN" not in ids(F.evaluate(steps, part(dims=dict(DIMS, length=500))))
    assert one(F.evaluate(steps, part(dims=dict(DIMS, length=501))), "CAP-LATHE-LEN").kind == "infeasible"
    # no lathe operation: the lathe limits do not apply
    assert "CAP-LATHE-DIA" not in ids(F.evaluate(["Facing", "Slot Milling", "Inspection"], part(dims=dict(DIMS, diameter=400))))
    big = F.Machine(lathe_max_diameter_mm=500.0)
    assert "CAP-LATHE-DIA" not in ids(F.evaluate(steps, part(dims=dict(DIMS, diameter=400)), machine=big))


def test_mill_envelope_uses_the_sorted_dimensions():
    steps = ["Facing", "Slot Milling", "Inspection"]
    ok = F.evaluate(steps, part(dims={"diameter": 50, "length": 600, "width": 400, "depth": 400}))
    assert "CAP-MILL-ENV" not in ids(ok)
    for dims in ({"diameter": 50, "length": 700, "width": 100, "depth": 50}, {"diameter": 50, "length": 100, "width": 700, "depth": 50},
                 {"diameter": 50, "length": 500, "width": 450, "depth": 50}):
        assert one(F.evaluate(steps, part(dims=dims)), "CAP-MILL-ENV").kind == "infeasible", dims
    assert "CAP-MILL-ENV" not in ids(F.evaluate(["Facing", TURN, "Inspection"], part(dims={"diameter": 50, "length": 900, "width": 900, "depth": 900})))


def test_power_rule_uses_the_machine_power_and_the_physics_model():
    steps = ["Facing", TURN, "Inspection"]
    p = part()
    assert "CAP-POWER" not in ids(F.evaluate(steps, p))
    rep = F.evaluate(steps, p, machine=F.Machine(spindle_power_kw=1.0))
    found = [x for x in rep.violations if x.rule_id == "PHYS:CAP-POWER"]
    assert found and all(x.kind == "risk" and x.action.startswith("reduce") for x in found)
    rows = physics.analyse_route(steps, "Steel", DIMS)["ops"]
    for x in found:
        row = rows[x.positions[0]]
        assert row["op"] == x.ops[0] and row["spindle_kw"] > 1.0 and f"{row['spindle_kw']:.1f}" in x.message
    assert [x.ops[0] for x in found] == [r["op"] for r in rows if r["spindle_kw"] and r["spindle_kw"] > 1.0]


def test_power_rule_skips_hole_operations_without_a_hole_diameter():
    steps = ["Facing", "Center Drilling", "Drilling", "Inspection"]
    big = part(dims={"diameter": 200, "length": 100, "depth": 40, "width": 160})
    assert physics.analyse_route(steps, "Steel", big["dims"])["totals"]["ops_over_power"] == ["Drilling"]      # the Time agent's hole = part diameter
    rep = F.evaluate(steps, big)
    assert "CAP-POWER" not in ids(rep) and any(r == "CAP-POWER" and "Drilling" in n for r, n in rep.notes)
    with_hole = part(dims={"diameter": 200, "length": 100, "depth": 40, "width": 160, "hole_diameter": 200})
    assert one(F.evaluate(steps, with_hole), "CAP-POWER").ops == ("Drilling",)
    small_hole = part(dims={"diameter": 200, "length": 100, "depth": 40, "width": 160, "hole_diameter": 12})
    assert "CAP-POWER" not in ids(F.evaluate(steps, small_hole))


# ---- surface finish -------------------------------------------------------------------------------------------------------------
def test_ra_feed_suggests_the_feed_that_physics_gives():
    steps = ["Facing", TURN, "Inspection"]
    rep = F.evaluate(steps, part(ra=0.4))
    v = one(rep, "RA-FEED")
    fmax = physics.max_feed_for_ra(0.4, physics.NOSE_RADIUS_MM["turning"])
    assert v.kind == "risk" and f"{fmax:.3f}" in v.message and f"{fmax:.3f}" in v.action
    assert fmax < 0.15                                               # the v1 steel feed is too coarse for 0.4 um
    assert "RA-FEED" not in ids(F.evaluate(steps, part(ra=3.2)))
    assert "RA-FEED" not in ids(F.evaluate(steps, part(ra=None))) and "RA-FEED" in F.evaluate(steps, part(ra=None)).not_evaluated


def test_ra_floor_needs_a_finishing_process_and_polishing_clears_it():
    mill = ["Facing", "Slot Milling", "Inspection"]
    v = one(F.evaluate(mill, part(("Slot",), ra=0.4)), "RA-FLOOR")
    assert v.kind == "risk" and "finish" in v.action
    assert "RA-FLOOR" not in ids(F.evaluate(["Facing", "Slot Milling", "Polishing/Burnishing", "Inspection"], part(("Slot",), ra=0.4)))
    assert "RA-FLOOR" not in ids(F.evaluate(["Facing", TURN, "Inspection"], part(ra=0.4)))            # turning floor is 0.4 um: reachable
    assert "RA-FLOOR" not in ids(F.evaluate(mill, part(("Slot",), ra=0.8)))


# ---- workholding ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ld,expect", [(2.9, []), (3.0, ["HOLD-TAILSTOCK"]), (5.99, ["HOLD-TAILSTOCK"]), (6.0, ["HOLD-STEADY"]),
                                        (12.5, ["HOLD-STEADY"]), (20.0, ["HOLD-STEADY"]), (20.5, ["HOLD-SLENDER-MAX"])])
def test_slenderness_thresholds(ld, expect):
    steps = ["Facing", TURN, "Inspection"]
    rep = F.evaluate(steps, part(dims={"diameter": 20, "length": 20 * ld, "depth": 5}))
    got = [i for i in ids(rep) if i.startswith("HOLD-")]
    assert got == expect, (ld, got)
    for v in rep.violations:
        if v.rule_id.startswith("PHYS:HOLD-"):
            assert v.kind == "risk" and v.action.startswith(("support", "review")) and v.positions == (1,)


def test_steady_rest_message_asks_for_reduced_parameters_above_12():
    steps = ["Facing", TURN, "Inspection"]
    low = one(F.evaluate(steps, part(dims={"diameter": 20, "length": 160, "depth": 5})), "HOLD-STEADY")
    high = one(F.evaluate(steps, part(dims={"diameter": 20, "length": 300, "depth": 5})), "HOLD-STEADY")
    assert "reduced" not in low.message and "reduced" in high.message


def test_slenderness_only_for_turning_and_needs_dimensions():
    slender = part(dims={"diameter": 10, "length": 200, "depth": 5})
    assert not [i for i in ids(F.evaluate(["Facing", "Slot Milling", "Inspection"], slender)) if i.startswith("HOLD-")]
    rep = F.evaluate(["Facing", TURN, "Inspection"], part(dims=None) | {"dims": None})
    assert {"HOLD-TAILSTOCK", "HOLD-STEADY", "HOLD-SLENDER-MAX"} <= set(rep.not_evaluated) and "dims.length missing" in rep.not_evaluated["HOLD-STEADY"]
    rep = F.evaluate(["Facing", "Slot Milling", "Inspection"], part(dims=None) | {"dims": None})
    assert "HOLD-STEADY" in rep.evaluated                               # not applicable is not the same as not evaluable


# ---- hole making -------------------------------------------------------------------------------------------------------------------
def test_deep_hole_thresholds_and_missing_hole_diameter():
    steps = ["Facing", "Center Drilling", "Drilling", "Inspection"]

    def run(depth, d=10):
        return F.evaluate(steps, part(("Hole",), dims={"diameter": 50, "length": 80, "depth": depth, "hole_diameter": d}))
    assert "HOLE-DEEP" not in ids(run(30))                              # exactly 3 x D is not deeper than 3 x D
    v = one(run(35), "HOLE-DEEP")
    assert v.action == "drill:peck" and "peck drilling" in v.message and v.positions == (2,)
    assert one(run(60), "HOLE-DEEP").action == "drill:multi-peck"
    rep = F.evaluate(steps, part(("Hole",)))
    assert "HOLE-DEEP" in rep.not_evaluated and "hole_diameter" in rep.not_evaluated["HOLE-DEEP"]
    assert "HOLE-DEEP" in F.evaluate(["Facing", TURN, "Inspection"], part()).evaluated


def test_hole_finishing_operations_need_a_hole_before_them():
    cases = [(["Facing", "Center Drilling", "Reaming", "Inspection"], "Reaming"),
             (["Facing", "Internal Grooving", "Inspection"], "Internal Grooving"),
             (["Facing", "Center Drilling", "Tapping", "Inspection"], "Tapping"),
             (["Facing", "Counterboring", "Center Drilling", "Drilling", "Inspection"], "Counterboring")]      # the hole comes too late
    for steps, op in cases:
        v = one(F.evaluate(steps, part()), "HOLE-EXISTS")
        assert v.kind == "infeasible" and v.ops == (op,) and v.positions == (steps.index(op),) and v.action == "insert:Drilling"
    for steps in (["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"], ["Facing", "Boring", "Reaming", "Inspection"],
                  ["Facing", "Center Drilling", "Drilling", "Internal Grooving", "Inspection"]):
        assert "HOLE-EXISTS" not in ids(F.evaluate(steps, part()))


def test_drilling_without_a_centre_drill_is_a_risk():
    v = one(F.evaluate(["Facing", "Drilling", "Inspection"], part()), "HOLE-PILOT")
    assert v.kind == "risk" and v.action == "insert:Center Drilling" and v.positions == (1,)
    assert "HOLE-PILOT" not in ids(F.evaluate(["Facing", "Center Drilling", "Drilling", "Inspection"], part()))
    assert "HOLE-PILOT" in ids(F.evaluate(["Facing", "Drilling", "Center Drilling", "Inspection"], part()))       # the centre drill comes too late


# ---- tolerance --------------------------------------------------------------------------------------------------------------------------
def test_iso286_table_values():
    assert F.it_width_um(7, 25) == 21 and F.it_width_um(6, 25) == 13 and F.it_width_um(10, 25) == 84 and F.it_width_um(8, 60) == 46
    assert F.it_width_um(6, None) == 9 and F.it_width_um(6, 3) == 9                   # most favourable class; below 6 mm the 6-10 class
    assert F.it_width_um(7, 250) == 46 and F.it_width_um(7, 251) is None
    for g in range(4, 12):                                                          # widths grow with the size
        w = [row[g] for _, _, row in F.ISO286_UM]
        assert w == sorted(w)
    for _, _, row in F.ISO286_UM:                                                    # and with the grade
        assert [row[g] for g in range(4, 12)] == sorted(row[g] for g in range(4, 12))


def test_parse_tolerance():
    assert F.parse_tolerance_mm("0.05mm") == 0.05 and F.parse_tolerance_mm("+-0.01 mm") == 0.01 and F.parse_tolerance_mm(0.5) == 0.5
    assert F.parse_tolerance_mm(None) is None and F.parse_tolerance_mm("fine") is None and F.parse_tolerance_mm("0mm") is None


def test_tolerance_hole_asks_for_reaming_when_drilling_cannot_hold_it():
    drilled = ["Facing", "Center Drilling", "Drilling", "Inspection"]
    v = one(F.evaluate(drilled, part(("Hole",), tolerance="0.02mm")), "TOL-HOLE")                  # band 40 um < IT10 (58)
    assert v.kind == "risk" and v.action == "insert:Reaming" and v.ops == ("Drilling",) and "IT10" in v.message
    assert "TOL-HOLE" not in ids(F.evaluate(drilled, part(("Hole",), tolerance="0.05mm")))          # band 100 um holds IT10
    assert "TOL-HOLE" not in ids(F.evaluate(["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"], part(("Hole",), tolerance="0.02mm")))
    bored = ["Facing", "Center Drilling", "Drilling", "Boring", "Inspection"]
    assert "TOL-HOLE" not in ids(F.evaluate(bored, part(("Hole",), tolerance="0.05mm")))
    assert "IT8" in one(F.evaluate(bored, part(("Hole",), tolerance="0.01mm")), "TOL-HOLE").message    # band 20 um < IT8 (22 um at 6-10 mm)
    assert "TOL-HOLE" not in ids(F.evaluate(drilled, part(("Slot",), tolerance="0.01mm")))        # no hole feature
    assert "TOL-HOLE" not in ids(F.evaluate(["Facing", "Inspection"], part(("Hole",), tolerance="0.01mm")))   # no hole-making at all: coverage's job


def test_tolerance_hole_uses_the_hole_diameter_and_leaves_what_reaming_cannot_hold_to_tol_spec():
    drilled = ["Facing", "Center Drilling", "Drilling", "Inspection"]
    p = part(("Hole",), tolerance="0.005mm", dims={"diameter": 80, "length": 50, "depth": 20, "hole_diameter": 50})       # band 10 um < IT6 at 50 mm (19)
    rep = F.evaluate(drilled, p)
    assert "TOL-HOLE" not in ids(rep) and "TOL-SPEC" in ids(rep)
    fav = F.evaluate(drilled, part(("Hole",), tolerance="0.005mm"))                          # unknown size: 6-10 class, reaming holds 9 um
    assert "TOL-HOLE" in ids(fav) and "TOL-SPEC" not in ids(fav)


def test_tolerance_spec_is_a_part_level_finding():
    steps = ["Facing", TURN, "Inspection"]
    p = part(("Turning",), tolerance="0.01mm", dims={"diameter": 60, "length": 60, "depth": 5})     # band 20 um < IT7 at 60 mm (30)
    rep = F.evaluate(steps, p)
    v = one(rep, "TOL-SPEC")
    assert v.kind == "risk" and v.ops == () and v.positions == () and "IT7" in v.message
    assert rep.part_violations == [v] and v not in rep.route_violations
    assert "TOL-SPEC" not in ids(F.evaluate(steps, part(("Turning",), tolerance="0.02mm", dims={"diameter": 60, "length": 60, "depth": 5})))
    other = F.evaluate(["Facing", "Slot Milling", "Inspection"], part(("Slot",), tolerance="0.005mm"))
    assert "TOL-SPEC" not in ids(other)                                                       # milled features are not judged against IT grades
    assert "TOL-SPEC" in F.evaluate(steps, part(("Turning",), tolerance="x", dims={"diameter": 60})).not_evaluated
    assert "TOL-SPEC" in F.evaluate(steps, part(("Turning",), tolerance="0.01mm", dims={"diameter": 300, "length": 60})).not_evaluated


# ---- sequence ------------------------------------------------------------------------------------------------------------------------
def test_nothing_on_the_lathe_after_parting_off():
    bad = ["Facing", TURN, "Parting-off", "Internal Grooving", "Polishing/Burnishing", "Inspection"]
    rep = F.evaluate(bad, part(("Turning", "Parting_Off")))
    assert [v.ops for v in rep.violations if v.rule_id == "PHYS:SEQ-PARTOFF"] == [("Internal Grooving",), ("Polishing/Burnishing",)]
    ok = ["Facing", TURN, "Parting-off", "Chamfering", "Slot Milling", "Inspection"]
    assert "SEQ-PARTOFF" not in ids(F.evaluate(ok, part(("Turning", "Parting_Off"))))
    assert "SEQ-PARTOFF" not in ids(F.evaluate(["Facing", TURN, "Inspection"], part()))
    dup = ["Facing", TURN, "Parting-off", "Parting-off", "Inspection"]                 # a repeated operation is the duplicate rule's business
    assert "SEQ-PARTOFF" not in ids(F.evaluate(dup, part(("Turning", "Parting_Off"))))


# ---- evaluate() ----------------------------------------------------------------------------------------------------------------------
def test_a_zero_or_negative_number_counts_as_missing():
    steps = ["Facing", TURN, "Inspection"]
    assert "RA-FLOOR" in F.evaluate(steps, part(ra=0)).not_evaluated
    assert "HOLD-STEADY" in F.evaluate(steps, part(dims={"diameter": 0, "length": 100, "depth": 5})).not_evaluated


def test_missing_part_data_is_reported_not_passed():
    rep = F.evaluate(["Facing", TURN, "Inspection"], {})
    assert set(rep.not_evaluated) and not rep.violations
    assert set(rep.evaluated) | set(rep.not_evaluated) == {r.id for r in F.RULES}
    assert F.evaluate(["Facing", "Inspection"], None).violations == []


def test_markers_and_unknown_operations_are_tolerated_and_results_are_deterministic():
    steps = ["Facing", TURN, V.MARKER, "Slot Milling", "Telepathy", "Inspection"]
    a = F.evaluate(steps, part(ra=0.4))
    b = F.evaluate([s for s in steps if s != V.MARKER], part(ra=0.4))
    assert a.to_dict() == b.to_dict() == F.evaluate(steps, part(ra=0.4)).to_dict()
    assert F.evaluate([], part()).violations == []


def test_report_is_json_serialisable_and_consistent():
    rep = F.evaluate(["Facing", "Center Drilling", "Drilling", "Inspection"], part(("Hole", "Turning"), tolerance="0.005mm", ra=0.4,
                                                                                  dims={"diameter": 90, "length": 400, "depth": 30}))
    d = rep.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert d["n_infeasible"] == len(rep.infeasible) and d["n_risk"] == len(rep.risks) and len(d["violations"]) == len(rep.violations)
    assert d["route_risk"] + d["part_risk"] == d["n_risk"] and d["route_infeasible"] == d["n_infeasible"]
    assert d["evaluated"] == len(rep.evaluated) and set(d["not_evaluated"]) == set(rep.not_evaluated)
    assert all(v["scope"] in ("route", "part") and v["rule_id"] in F.RULE_BY_ID for v in d["violations"])


def test_rules_argument_runs_a_subset():
    rep = F.evaluate(["Facing", "Drilling", "Inspection"], part(), rules=[F.RULE_BY_ID["HOLE-PILOT"]])
    assert ids(rep) == ["HOLE-PILOT"] and rep.evaluated == ["HOLE-PILOT"]


def test_part_is_not_modified():
    p = part(ra=0.4)
    before = copy.deepcopy(p)
    F.evaluate(["Facing", TURN, "Inspection"], p)
    assert p == before


# ---- hook into the strict verifier ---------------------------------------------------------------------------------------------------
def test_baselines_are_untouched_unless_the_hook_is_installed():
    assert F.verify_rule not in V.PHYSICAL_RULES
    steps = ["Facing", "Internal Grooving", "Inspection"]
    c = V.check_route(steps, ["Groove"], part=part(("Groove",)))
    assert c.valid and "infeasible" not in c.kinds
    F.install()
    F.install()
    assert V.PHYSICAL_RULES.count(F.verify_rule) == 1
    c = V.check_route(steps, ["Groove"], part=part(("Groove",)))
    assert not c.valid and "infeasible" in c.kinds
    assert V.check_route(steps, ["Groove"]).valid                       # without a part nothing is evaluated
    F.uninstall()
    assert V.check_route(steps, ["Groove"], part=part(("Groove",))).valid


def test_risks_never_make_a_route_invalid():
    F.install()
    steps = ["Facing", "Center Drilling", "Drilling", "Inspection"]
    c = V.check_route(steps, ["Hole"], part=part(("Hole",), tolerance="0.02mm"))
    assert "risk" in c.kinds and c.valid and not c.hard_violations


def test_route_builder_routes_have_no_hard_infeasibility_on_dev(parts):
    hard = {}
    for p in parts:
        inp = systems.part_inputs(p)
        routes, _ = systems._rb_routes(inp["features"], inp["pref"], 15)
        rep = F.evaluate(routes[0]["steps"], p)
        for v in rep.infeasible:
            hard.setdefault(v.rule_id, []).append(p["part_id"])
    assert hard == {}, hard


# ---- harness, summary, compare --------------------------------------------------------------------------------------------------------------
@pytest.fixture
def clean_client(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CAPP_LLM_MODE", raising=False)
    monkeypatch.delenv("CAPP_RUN_TAG", raising=False)
    llm_client.reset_transport()
    llm_client.reset_call_log()
    yield
    llm_client.reset_transport()


def groove_part(base):
    p = copy.deepcopy(base)
    p.update({"features": ["Groove"], "tolerance": "0.05mm"})
    return p


def hole_less_planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
    """Strict-valid for a part whose only feature is Groove (Internal Grooving is a listed alternative) but physically infeasible: no hole."""
    steps = ["Facing", "Internal Grooving", "Inspection"] if sample_id % 2 == 0 else ["Facing", "Grooving/Necking", "Inspection"]
    meta = {"cached": False, "finish_reason": "stop", "prompt_tokens": 100, "completion_tokens": 20, "latency_s": 0.1}
    return {"success": True, "steps": steps, "raw_response": json.dumps(steps), "model": "fake", "meta": meta}


def test_harness_records_feasibility_for_final_and_raw_routes(parts, tmp_path, clean_client):
    ps = [groove_part(parts[0]), groove_part(parts[5])]
    out = H.run_experiment("b0", ps, repeats=1, tag="t", out_dir=str(tmp_path), planner=hole_less_planner, progress=None, dataset_name="dev_v2")
    recs = [json.loads(l) for l in (tmp_path / "t_b0" / "records.jsonl").read_text().splitlines()]
    for rec in recs:
        f = rec["final"]["feasibility"]
        assert set(f) >= {"violations", "n_infeasible", "n_risk", "evaluated", "not_evaluated"}
        assert f == json.loads(json.dumps(F.condense(F.evaluate(V.clean_steps(rec["final"]["steps"]), rec["part"]))))
        for c in rec["candidates"]:
            if c["llm_ok"]:
                assert c["raw_feasibility"]["route_infeasible"] == (1 if c["llm_steps"][1] == "Internal Grooving" else 0)
    fb = out["summary"]["feasibility"]
    raw = [c for rec in recs for c in rec["candidates"] if c["llm_ok"]]
    bad = sum(1 for c in raw if c["llm_steps"][1] == "Internal Grooving")
    assert 0 < bad < len(raw) and len(raw) == 10                                          # b0 samples 5 routes per part
    assert fb["n"] == 2 and fb["raw_n"] == 10 and fb["raw_strict_valid_n"] == 10         # every sample is strict-valid ...
    assert fb["raw_valid_infeasible_share"] == pytest.approx(bad / 10)                    # ... and some of them cannot be machined
    assert fb["raw_infeasible_share"] == pytest.approx(bad / 10)
    text = SM.format_summary("b0", out["summary"])
    assert "feasibility     :" in text and f"strict-valid but physically infeasible {bad / 10:.3f}" in text


def test_rb_run_has_final_feasibility_and_csv_columns(parts, tmp_path):
    out = H.run_experiment("rb", parts[:15], tag="t", out_dir=str(tmp_path), dataset_name="dev_v2", progress=None)
    recs = [json.loads(l) for l in (tmp_path / "t_rb" / "records.jsonl").read_text().splitlines()]
    assert all("feasibility" in r["final"] and all("raw_feasibility" not in c for c in r["candidates"]) for r in recs)
    rows = list(csv.DictReader(open(tmp_path / "t_rb" / "per_part.csv", encoding="utf-8")))
    for col in ("feas_infeasible", "feas_risk", "feas_spec", "feas_rules"):
        assert col in rows[0], col
    for row, rec in zip(rows, recs):
        f = rec["final"]["feasibility"]
        assert int(row["feas_infeasible"]) == f["route_infeasible"] and int(row["feas_risk"]) == f["route_risk"] and int(row["feas_spec"]) == f["part_risk"]
    cand = list(csv.DictReader(open(tmp_path / "t_rb" / "per_candidate.csv", encoding="utf-8")))
    assert "raw_feas_infeasible" in cand[0] and all(r["raw_feas_infeasible"] == "" for r in cand)
    fb = out["summary"]["feasibility"]
    assert fb["n"] == 15 and fb["final_infeasible_share"] == 0.0
    assert fb["final_risk_share"] == pytest.approx(sum(1 for r in recs if r["final"]["feasibility"]["route_risk"]) / 15)
    assert fb["raw_n"] == 0 and fb["raw_valid_infeasible_share"] is None
    assert sum(fb["by_rule_share"].values()) >= fb["final_risk_share"]
    run = json.loads((tmp_path / "t_rb" / "run.json").read_text())
    assert run["feasibility_analysis"] is True


def test_no_analysis_skips_feasibility(parts, tmp_path):
    out = H.run_experiment("rb", parts[:5], tag="t", out_dir=str(tmp_path), dataset_name="dev_v2", progress=None, analysis=False)
    recs = [json.loads(l) for l in (tmp_path / "t_rb" / "records.jsonl").read_text().splitlines()]
    assert all("feasibility" not in r["final"] for r in recs)
    assert out["summary"]["feasibility"] == {"n": 0}
    assert "feasibility     :" not in SM.format_summary("rb", out["summary"])
    rows = list(csv.DictReader(open(tmp_path / "t_rb" / "per_part.csv", encoding="utf-8")))
    assert all(r["feas_risk"] == "" and r["feas_rules"] == "" for r in rows)


def test_feasibility_does_not_change_routes_validity_or_scores(parts, tmp_path, clean_client):
    ps = [groove_part(parts[0]), parts[1], parts[2]]
    a = H.run_experiment("n2_r1_s3", ps, tag="a", out_dir=str(tmp_path / "a"), planner=hole_less_planner, progress=None)
    b = H.run_experiment("n2_r1_s3", ps, tag="a", out_dir=str(tmp_path / "b"), planner=hole_less_planner, progress=None, analysis=False)
    ra = [json.loads(l) for l in (tmp_path / "a" / "a_n2_r1_s3" / "records.jsonl").read_text().splitlines()]
    rb_ = [json.loads(l) for l in (tmp_path / "b" / "a_n2_r1_s3" / "records.jsonl").read_text().splitlines()]
    for x, y in zip(ra, rb_):
        x["final"].pop("physics"), x["final"].pop("feasibility")
        for c in x["candidates"]:
            c.pop("raw_feasibility", None)
        assert x["final"] == y["final"] and x["info"] == y["info"] and x["candidates"] == y["candidates"]
    for k in ("M1", "M2", "M3", "M4", "M6"):
        assert a["summary"][k] == b["summary"][k]


def test_feasibility_error_is_recorded_not_fatal(parts, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("rule broke")
    monkeypatch.setattr(F, "evaluate", boom)
    out = H.run_experiment("rb", parts[:4], tag="t", out_dir=str(tmp_path), dataset_name="dev_v2", progress=None)
    recs = [json.loads(l) for l in (tmp_path / "t_rb" / "records.jsonl").read_text().splitlines()]
    assert all("RuntimeError" in r["final"]["feasibility"]["error"] for r in recs)
    assert out["summary"]["feasibility"] == {"n": 0} and out["summary"]["M2"]["final_strict_valid"] == 1.0
    assert SM._feasibility_columns(recs[0]["final"]["feasibility"])["feas_risk"] == ""


def test_old_records_without_feasibility_still_summarise():
    assert SM._feasibility_block([{"final": {}, "candidates": []}]) == {"n": 0}
    assert SM._feasibility_block([]) == {"n": 0}
    assert SM._feas(None) is None and SM._feas({"error": "x"}) is None


def test_summary_survives_a_json_round_trip_and_compare_shows_rows(parts, tmp_path, clean_client):
    ps = [groove_part(parts[0]), groove_part(parts[5])]
    H.run_experiment("rb", parts[:6], tag="t", out_dir=str(tmp_path), dataset_name="dev_v2", progress=None)
    out = H.run_experiment("b0", ps, tag="t", out_dir=str(tmp_path), planner=hole_less_planner, progress=None)
    again = json.loads((tmp_path / "t_b0" / "summary.json").read_text())
    assert again["feasibility"] == json.loads(json.dumps(out["summary"]["feasibility"]))
    assert SM.format_summary("b0", again) == SM.format_summary("b0", out["summary"])
    text = C.compare(sorted(str(p) for p in tmp_path.iterdir() if p.is_dir()))
    for label in ("feasibility: finals with a hard infeasibility", "feasibility: finals with risks",
                  "feasibility: strict-valid raw LLM routes infeasible"):
        assert label in text, label


def test_cli_prints_the_feasibility_line_and_the_check_script_runs(tmp_path):
    env = dict(os.environ, CAPP_CACHE_DIR=str(tmp_path / "cache"))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_eval.py"), "--system", "rb", "--dataset", "dev_v2", "--limit", "8",
                        "--tag", "x", "--out", str(tmp_path / "o")], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "feasibility     :" in r.stdout
    js = tmp_path / "fc.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "feasibility_check.py"), "--dataset", "dev_v2", "--limit", "20",
                        "--json", str(js)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    for word in ("rules", "RB first route", "RB variants", "not evaluable", "sources"):
        assert word in r.stdout, word
    rep = json.loads(js.read_text())
    assert rep["parts"] == 20 and set(rep["rules"]) == {x.id for x in F.RULES}
