"""Step 19 (WP9), the data and the plan: capp/process_tables.py (tool rules, reference ranges with status and sources) and
capp/process_plan.py (route -> per-operation machine, tool, parameters, time, justification). No API calls, no network.

What is checked: the tables are complete and consistent (and honest about their status); the plan describes the route and never changes it;
its times add up to the v1 Time agent's; the flags that matter (a parameter above a reference range, an assumed machine, tool size or
hole diameter) are there; old records and odd routes do not break it.
"""
import copy

import pytest

from capp import analysis_view as AV
from capp import dataset, systems
from capp import process_plan as PP
from capp import process_tables as PT
from capp import verify as V

import agents

DIMS = {"diameter": 30, "length": 60, "depth": 20, "width": 20}
MARKER = V.MARKER


def part(material="Steel", features=("Hole",), dims=DIMS, hole=None, ra=None, batch=10):
    return AV.make_part(material, list(features), "0.05mm", batch, dims=dims, required_ra_um=ra, hole_diameter=hole)


def ops_of(plan):
    return [e["op"] for e in plan["operations"]]


def by_op(plan, op):
    return next(e for e in plan["operations"] if e["op"] == op)


# ---------------------------------------------------------------- the tables
def test_tables_are_consistent():
    assert PT.validate() == []


def test_one_row_per_operation_of_the_vocabulary():
    assert len(PT.OPERATIONS) == 41
    assert set(PT.OPERATIONS) == set(V.CANONICAL_OPS)


def test_every_operation_has_a_tool_for_every_material():
    for op in PT.OPERATIONS:
        for m in PT.MATERIALS:
            t = PT.tool_for(op, m)
            assert t["type"] and t["material"] and t["status"] == "indicative"


def test_flat_rows_count_and_keys():
    rows = PT.rows()
    n_tool = 41 * len(PT.MATERIALS) + len(PT.MACHINE_TOOL) * len(PT.MATERIALS)
    assert sum(r["table"] == "tool" for r in rows) == n_tool
    assert sum(r["table"] == "range" for r in rows) == 3 * len(PT.MATERIALS)
    assert len(rows) == n_tool + 21
    assert all(set(r) == {"table", "key", "value", "status", "sources", "note"} for r in rows)


def test_no_range_is_confirmed_and_the_status_counts_say_so():
    st = PT.status_counts(PT.RANGES.values())
    assert st["confirmed"] == 0                                    # no handbook page was checked: the tables must not claim otherwise
    assert sum(st.values()) == 21 and st["web"] and st["mixed"]
    assert all(r["status"] in PT.STATUSES for r in PT.RANGES.values())


def test_web_and_mixed_rows_name_a_source_that_exists():
    for r in PT.RANGES.values():
        if r["status"] in ("web", "mixed"):
            assert r["sources"] and all(s in PT.SOURCES for s in r["sources"])
    for what, url in PT.SOURCES.values():
        assert url.startswith("https://") and what


def test_a_confirmed_row_without_a_page_is_a_table_problem(monkeypatch):
    bad = dict(PT.RANGES[("drill", "Steel")], status="confirmed", source_note="")
    monkeypatch.setitem(PT.RANGES, ("drill", "Steel"), bad)
    assert any("needs the handbook page" in e for e in PT.validate())


def test_a_web_row_without_a_source_is_a_table_problem(monkeypatch):
    bad = dict(PT.RANGES[("mill", "Steel")], sources=[])
    monkeypatch.setitem(PT.RANGES, ("mill", "Steel"), bad)
    assert any("needs a source" in e for e in PT.validate())


def test_tool_rules_follow_the_workpiece_material():
    assert "coated carbide" in PT.tool_for("Plain/Cylindrical Turning", "Steel")["material"]
    assert "sharp" in PT.tool_for("Plain/Cylindrical Turning", "Aluminum")["material"]
    assert "coolant" in PT.tool_for("Drilling", "Titanium")["material"] and "coolant" not in PT.tool_for("Drilling", "Steel")["material"]
    assert PT.tool_for("Drilling", "Aluminum")["type"] == PT.tool_for("Drilling", "Steel")["type"]        # the tool type follows the operation only


def test_machine_specific_tools():
    f_mill = PT.tool_for("Facing", "Steel", "milling")
    f_lathe = PT.tool_for("Facing", "Steel", "lathe")
    assert f_mill["machine_override"] and "face mill" in f_mill["type"]
    assert not f_lathe["machine_override"] and "turning" in f_lathe["type"]
    assert PT.tool_for("Drilling", "Steel", "milling")["machine_override"] is False


def test_unknown_material_and_operation_raise():
    with pytest.raises(KeyError):
        PT.tool_for("Drilling", "Unobtainium")
    with pytest.raises(KeyError):
        PT.range_for("Drilling", "Unobtainium")
    with pytest.raises(KeyError):
        PT.tool_for("Whittling", "Steel")


def test_inspection_has_no_range_and_no_cutting_tool():
    assert PT.range_for("Inspection", "Steel") is None
    assert PT.check_range("Inspection", "Steel", 10, 0.1)["range"] is None


def test_check_range_within_below_above():
    lo, hi = PT.RANGES[("drill", "Steel")]["vc"]
    flo, fhi = PT.RANGES[("drill", "Steel")]["feed"]
    assert PT.check_range("Drilling", "Steel", (lo + hi) / 2, (flo + fhi) / 2)["attention"] is False
    c = PT.check_range("Drilling", "Steel", hi + 1, flo - 0.01)
    assert c["vc"] == "above" and c["feed"] == "below" and c["attention"] is True
    c = PT.check_range("Drilling", "Steel", lo - 1, fhi)
    assert c["vc"] == "below" and c["feed"] == "within" and c["attention"] is False       # below is conservative, not a warning
    assert PT.check_range("Drilling", "Steel", lo, None)["feed"] is None                   # a missing value is skipped on its own
    assert PT.check_range("Drilling", "Steel", None, fhi + 1)["vc"] is None


def test_check_range_edges_are_within():
    lo, hi = PT.RANGES[("mill", "Aluminum")]["vc"]
    assert PT.check_range("Slot Milling", "Aluminum", lo, None)["vc"] == "within"
    assert PT.check_range("Slot Milling", "Aluminum", hi, None)["vc"] == "within"


# ---------------------------------------------------------------- machines
def test_machine_class():
    assert PP.machine_class("Plain/Cylindrical Turning") == "lathe"
    assert PP.machine_class("Slot Milling") == "milling"
    assert PP.machine_class("Drilling") == "either"
    assert PP.machine_class("Inspection") == "either"


def test_assign_machines_by_segment():
    route = ["Facing", "Plain/Cylindrical Turning", MARKER, "Slot Milling", "Drilling", "Inspection"]
    assert PP.assign_machines(route) == [("lathe", False), ("lathe", False), ("milling", False), ("milling", False), ("milling", False)]
    # Facing is shared: it takes the machine of its segment (the lathe, because of the turning), and that is not an assumption


def test_a_segment_with_no_deciding_operation_is_assumed_a_lathe():
    got = PP.assign_machines(["Facing", "Drilling", "Inspection"])
    assert got == [("lathe", True)] * 3
    pref = PP.assign_machines(["Facing", "Drilling", "Inspection"], preference="milling")
    assert pref == [("milling", False)] * 3                                # the part's own preference decides, so it is not an assumption


def test_a_segment_with_both_kinds_has_no_usable_marker():
    got = PP.assign_machines(["Plain/Cylindrical Turning", "Slot Milling", "Drilling"])
    assert got[0] == ("lathe", False) and got[1] == ("milling", False)
    assert got[2] == ("lathe", True)                                       # shared operation of a mixed segment: lathe, assumed


# ---------------------------------------------------------------- the plan
def test_plan_describes_the_route_and_does_not_change_it():
    steps = ["Facing", "Center Drilling", "Drilling", "Inspection"]
    before = copy.deepcopy(steps)
    plan = PP.build_plan(steps, part())
    assert steps == before
    assert plan["route"] == V.normalize_markers(steps)
    assert ops_of(plan) == before
    assert [e["seq"] for e in plan["operations"]] == [10, 20, 30, 40]


def test_estimated_times_add_up_to_the_v1_time_agent():
    steps = ["Facing", "Plain/Cylindrical Turning", MARKER, "Slot Milling", "Center Drilling", "Drilling", "Inspection"]
    plan = PP.build_plan(steps, part("Steel", ("Turning", "Slot", "Hole")))
    t = plan["totals"]
    assert t["total_min"] == pytest.approx(t["v1_time_min"], abs=0.01)
    assert t["total_min"] == pytest.approx(sum(e["time_min"] for e in plan["operations"]))
    assert t["total_min"] == pytest.approx(t["cutting_min"] + t["tool_change_min"] + t["position_change_min"] + t["changeover_min"])
    assert t["changeover_min"] == agents.CHANGEOVER_TIME_MIN and t["n_changeovers"] == 1
    assert t["per_batch_hours"] == pytest.approx(t["total_min"] * 10 / 60)


def test_times_equal_v1_over_a_dataset_of_route_builder_routes():
    parts = dataset.load_dataset("dev_v2")[:12]
    for p in parts:
        rec = systems.run_rb(p).to_dict()
        plan = PP.build_plan(rec["final"]["steps"], dict(p, part_id=p["part_id"]))
        assert plan["totals"]["total_min"] == pytest.approx(plan["totals"]["v1_time_min"], abs=0.01), p["part_id"]
        assert plan["verifier"]["valid"] is True


def test_the_changeover_is_charged_to_the_first_operation_after_it():
    steps = ["Facing", "Plain/Cylindrical Turning", MARKER, "Slot Milling", "Inspection"]
    plan = PP.build_plan(steps, part("Steel", ("Turning", "Slot")))
    slot = by_op(plan, "Slot Milling")
    assert slot["changeover_before"] is True and slot["time"]["changeover_min"] == agents.CHANGEOVER_TIME_MIN
    assert slot["machine"] == "milling" and by_op(plan, "Facing")["changeover_before"] is False
    assert plan["changeovers"] == [{"before_seq": slot["seq"], "from": "lathe", "to": "milling"}]


def test_a_route_without_markers_gets_them_at_every_machine_switch():
    plan = PP.build_plan(["Facing", "Plain/Cylindrical Turning", "Slot Milling", "Inspection"], part("Steel", ("Turning", "Slot")))
    assert plan["totals"]["n_changeovers"] == 1 and MARKER in plan["route"]
    raw = PP.build_plan(["Facing", "Plain/Cylindrical Turning", "Slot Milling", "Inspection"], part("Steel", ("Turning", "Slot")), normalize=False)
    assert raw["totals"]["n_changeovers"] == 0


def test_every_operation_alone_builds_for_every_material():
    for op in PT.OPERATIONS:
        for m in PT.MATERIALS:
            plan = PP.build_plan([op], part(m, ()))
            e = plan["operations"][0]
            assert e["known"] and e["time_min"] >= 0
            if op == "Inspection":
                assert e["params"] is None and e["tool"]["size"] == ""


def test_parameters_are_the_v1_models_and_internally_consistent():
    plan = PP.build_plan(["Facing", "Slot Milling", "Drilling", "Inspection"], part("Aluminum", ("Slot", "Hole")))
    for e in plan["operations"]:
        p = e["params"]
        if not p:
            continue
        assert p["rpm"] == pytest.approx(1000 * p["vc_m_min"] / (3.141592653589793 * p["diameter_mm"]), rel=1e-6)
        assert p["feed_rate_mm_min"] > 0 and p["passes"] >= 1
        assert p["load"] is None or p["load"] >= 0


def test_milling_feed_is_per_tooth_and_the_feed_rate_uses_the_teeth():
    plan = PP.build_plan(["Slot Milling", "Inspection"], part("Steel", ("Slot",)))
    p = by_op(plan, "Slot Milling")["params"]
    assert p["feed_unit"] == "mm/tooth"
    assert p["feed_rate_mm_min"] == pytest.approx(p["feed"] * agents._CUTTER_TEETH["Slot Milling"] * p["rpm"])
    assert p["diameter_mm"] == agents._CUTTER_DIA["Slot Milling"]


def test_threading_feeds_by_pitch_and_is_not_checked_against_the_drilling_feed_range():
    plan = PP.build_plan(["Facing", "Tapping", "External Threading", "Inspection"], part("Steel", ("Thread",)))
    for op in PP.THREAD_FEED_OPS:
        p = by_op(plan, op)["params"]
        assert p["feed"] == agents._THREAD_PITCH[op] and "pitch" in p["feed_unit"]
        assert by_op(plan, op)["range_check"]["feed"] is None


def test_titanium_drilling_is_flagged_above_its_range():
    plan = PP.build_plan(["Facing", "Center Drilling", "Drilling", "Inspection"], part("Titanium", ("Hole",)))
    e = by_op(plan, "Drilling")
    assert e["range_check"]["vc"] == "above" and e["range_check"]["attention"]
    assert any("above the reference range" in f for f in e["flags"])


def test_a_parameter_below_the_range_is_called_conservative_not_a_warning():
    plan = PP.build_plan(["Facing", "Plain/Cylindrical Turning", "Inspection"], part("Aluminum", ("Turning",)))
    below = [f for e in plan["operations"] for f in e["flags"] if "below the reference range" in f]
    for f in below:
        assert "conservative" in f


def test_plan_counts_the_ranges_it_used_by_status():
    plan = PP.build_plan(["Facing", "Slot Milling", "Drilling", "Inspection"], part("Steel", ("Slot", "Hole")))
    t = plan["tables"]
    assert t["ranges_used"] == sum(t["status"].values()) > 0
    assert t["status"]["confirmed"] == 0 and t["tool_rules"] == "indicative"
    assert set(t["sources"]) <= set(PT.SOURCES)


def test_facing_on_the_milling_machine_says_its_parameters_are_turning_type():
    steps = ["Facing", "Plain/Cylindrical Turning", MARKER, "Slot Milling", "Chamfering", "Inspection"]
    plan = PP.build_plan(steps, part("Steel", ("Turning", "Slot", "Chamfer")))
    cham = by_op(plan, "Chamfering")
    assert cham["machine"] == "milling" and "chamfer mill" in cham["tool"]["type"]
    assert any("turning-type" in f for f in cham["flags"])
    assert not any("turning-type" in f for f in by_op(plan, "Facing")["flags"])          # on the lathe it is a turning tool


def test_hole_diameter_given_or_assumed():
    steps = ["Facing", "Center Drilling", "Drilling", "Inspection"]
    given = PP.build_plan(steps, part(hole=8))
    d = by_op(given, "Drilling")
    assert d["tool"]["size"] == "Ø8 mm" and d["tool"]["size_assumed"] is False
    assert not any("hole diameter" in a for a in given["assumptions"])
    assumed = PP.build_plan(steps, part())
    d = by_op(assumed, "Drilling")
    assert d["tool"]["size"] == "Ø30 mm" and d["tool"]["size_assumed"] is True
    assert any("No hole diameter was given" in a for a in assumed["assumptions"])


def test_hole_diameter_changes_spindle_speed_not_the_cutting_speed():
    steps = ["Facing", "Center Drilling", "Drilling", "Inspection"]
    small = by_op(PP.build_plan(steps, part(hole=6)), "Drilling")["params"]
    big = by_op(PP.build_plan(steps, part(hole=20)), "Drilling")["params"]
    assert small["vc_m_min"] == big["vc_m_min"] and small["rpm"] > big["rpm"]


def test_route_only_uses_the_reference_part_and_says_so():
    plan = PP.build_plan(["Facing", "Drilling", "Inspection"], part(dims=None))
    assert plan["part"]["dims"] is None and "reference part" in plan["dims_source"]
    assert any("No part dimensions were given" in a for a in plan["assumptions"])
    assert plan["totals"]["total_min"] == pytest.approx(plan["totals"]["v1_time_min"], abs=0.01)


def test_assumptions_always_name_what_is_not_known():
    plan = PP.build_plan(["Facing", "Inspection"], part())
    text = " ".join(plan["assumptions"])
    assert "v1 model" in text and "handbook" in text and "ASSUMED machine" in text and "No machining trial" in text


def test_unknown_operation_is_listed_without_tool_or_parameters():
    plan = PP.build_plan(["Facing", "Whittling", "Drilling", "Inspection"], part())
    e = by_op(plan, "Whittling")
    assert e["known"] is False and e["tool"] is None and e["params"] is None
    assert any("outside the vocabulary" in f for f in e["flags"])
    assert plan["verifier"]["valid"] is False and any("not a known operation" in m for m in plan["verifier"]["messages"])


def test_unknown_material_raises():
    with pytest.raises(KeyError):
        PP.build_plan(["Facing"], dict(part(), material="Unobtainium"))


def test_feasibility_findings_are_attached_to_the_operations_they_name():
    plan = PP.build_plan(["Facing", "Drilling", "Inspection"], part())              # Drilling without Center Drilling: a risk rule
    assert any("HOLE-PILOT" in f for f in by_op(plan, "Drilling")["flags"])
    assert plan["feasibility"]["risks"] and plan["feasibility"]["error"] is None


def test_a_rule_bug_does_not_stop_the_plan(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("rule bug")
    monkeypatch.setattr(PP.F, "evaluate", boom)
    plan = PP.build_plan(["Facing", "Inspection"], part())
    assert "feasibility check failed" in plan["feasibility"]["error"] and plan["operations"]


# ---------------------------------------------------------------- justification and repair
def test_first_and_last_operations_cite_their_rules():
    plan = PP.build_plan(["Facing", "Center Drilling", "Drilling", "Inspection"], part())
    assert "ORD-FIRST" in plan["operations"][0]["rules"] and "ORD-LAST" in plan["operations"][-1]["rules"]


def test_operations_name_the_features_they_serve_and_the_order_rule():
    plan = PP.build_plan(["Facing", "Center Drilling", "Drilling", "Inspection"], part())
    d = by_op(plan, "Drilling")
    assert d["serves"] == ["Hole"] and "serves Hole" in d["why"]
    assert any(r.startswith("ORD-EDGE:Center Drilling>Drilling") for r in d["rules"])


def test_a_support_operation_says_no_feature_needs_it():
    plan = PP.build_plan(["Facing", "Plain/Cylindrical Turning", "Center Drilling", "Drilling", "Inspection"], part())
    assert "support operation" in by_op(plan, "Plain/Cylindrical Turning")["why"]


def test_rule_ids_per_operation_are_capped():
    plan = PP.build_plan(["Facing", "Center Drilling", "Drilling", "Slot Milling", "Chamfering", "Inspection"], part("Steel", ("Hole", "Slot", "Chamfer")))
    for e in plan["operations"]:
        assert sum(r.startswith("ORD-EDGE") for r in e["rules"]) <= PP.MAX_RULES_SHOWN


def test_repair_view_marks_kept_inserted_and_removed():
    rv = PP.repair_view(["Facing", "Drilling", "Whittling", "Inspection"], ["Facing", "Center Drilling", "Drilling", "Inspection"])
    assert rv["identical"] is False and rv["removed"] == ["Whittling"] and rv["inserted"] == ["Center Drilling"]
    assert rv["marks"] == ["kept", "inserted", "kept", "kept"] and rv["reordered"] is False


def test_repair_view_sees_a_reordering_and_identity():
    assert PP.repair_view(["Facing", "Drilling", "Inspection"], ["Facing", "Drilling", "Inspection"])["identical"] is True
    rv = PP.repair_view(["Facing", "Chamfering", "Drilling", "Inspection"], ["Facing", "Drilling", "Chamfering", "Inspection"])
    assert rv["reordered"] is True and rv["removed"] == [] and rv["inserted"] == []


def test_repair_view_ignores_markers_and_handles_duplicates():
    rv = PP.repair_view(["Facing", "Drilling", "Drilling", MARKER, "Inspection"], ["Facing", "Drilling", MARKER, "Inspection"])
    assert rv["removed"] == ["Drilling"] and rv["raw"] == ["Facing", "Drilling", "Drilling", "Inspection"]


def test_plan_carries_the_repair_view_only_when_a_raw_route_is_known():
    steps = ["Facing", "Center Drilling", "Drilling", "Inspection"]
    assert "repair" not in PP.build_plan(steps, part())
    p = PP.build_plan(steps, part(), context={"raw_steps": ["Facing", "Drilling", "Inspection"]})
    assert p["repair"]["inserted"] == ["Center Drilling"]


# ---------------------------------------------------------------- run records
def _record(p=None, with_llm=True):
    p = p or dataset.load_dataset("dev_v2")[1]
    rec = systems.run_rb(p).to_dict()
    rec["part"], rec["repeat"] = p, 0
    if with_llm:
        final = rec["final"]
        raw = [s for s in final["steps"] if s != MARKER]
        raw = [s for s in raw if s != "Inspection"] + ["Whittling", "Inspection"]
        rec["candidates"] = [{"sample_id": 0, "name": final["name"], "llm_ok": True, "llm_steps": raw, "meta": {"reasoning": "Facing first, then the rest."}},
                             {"sample_id": 1, "name": "other", "llm_ok": False, "llm_steps": None, "meta": {}}]
        rec["info"] = dict(rec["info"], confidence=0.62, flagged=True, selection="s3")
        rec["final"] = dict(final, provenance="repaired_llm")
    return rec


def test_plan_from_record_carries_provenance_confidence_flag_repair_and_reasoning():
    rec = _record()
    plan = PP.plan_from_record(rec)
    c = plan["context"]
    assert c["provenance"] == "repaired_llm" and c["confidence"] == 0.62 and c["flagged"] is True and c["selection"] == "s3"
    assert c["n_samples"] == 2 and c["reasoning"].startswith("Facing first") and c["repeat"] == 0
    assert plan["repair"]["removed"] == ["Whittling"]
    assert plan["part"]["part_id"] == rec["part"]["part_id"]


def test_the_flag_is_recomputed_against_a_given_threshold():
    rec = _record()
    assert PP.plan_from_record(rec, flag_threshold=0.5)["context"]["flagged"] is False
    assert PP.plan_from_record(rec, flag_threshold=0.9)["context"]["flagged"] is True
    assert PP.plan_from_record(rec, flag_threshold=0.9)["context"]["confidence_flag"] == 0.9


def test_a_route_builder_record_has_no_confidence_and_no_repair():
    rec = _record(with_llm=False)
    plan = PP.plan_from_record(rec)
    assert "confidence" not in plan["context"] and "repair" not in plan
    assert plan["context"]["provenance"] == "route_builder"


def test_an_old_record_without_optional_fields_still_plans():
    rec = _record()
    del rec["info"]["confidence"], rec["info"]["flagged"], rec["info"]["selection"]
    for c in rec["candidates"]:
        c.pop("meta")
    plan = PP.plan_from_record(rec)
    assert plan["operations"] and "confidence" not in plan["context"] and "reasoning" not in plan["context"]


def test_a_record_whose_part_has_no_dims_is_planned_on_the_reference_part():
    rec = _record(with_llm=False)
    rec["part"] = {k: v for k, v in rec["part"].items() if k != "dims"}
    plan = PP.plan_from_record(rec)
    assert plan["part"]["dims"] is None


def test_plan_from_record_does_not_modify_the_record():
    rec = _record()
    before = copy.deepcopy(rec)
    PP.plan_from_record(rec)
    assert rec == before
