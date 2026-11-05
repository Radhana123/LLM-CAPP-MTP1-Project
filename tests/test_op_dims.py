"""Step 51: every operation has the dimensions of its own cut (capp/op_dims.py, dims["per_op"] in week3/agents.py, capp/physics.py and
capp/process_plan.py), and the app shows them: all operations in a dropdown, the ones the features need selected, the numbers of the
drawing filled in.

The thing that must never move is every result for an input WITHOUT per_op: report 6.7 and the frozen ablation depend on it. The golden
hashes below were computed with the code of Step 50 (git HEAD before this step) and are checked here on the same inputs.
No network: the page tests (Streamlit's AppTest) put an extraction result in the session state.
"""
import copy
import hashlib
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "week1"))

from capp import legacy  # noqa: E402,F401
import agents  # noqa: E402
import feature_vocab as fv  # noqa: E402
import route_builder as rb  # noqa: E402
from capp import analysis_view as AV, dataset, op_dims as OPD, physics, process_plan as PP, process_tables as PT, verify as V  # noqa: E402

PART = {"diameter": 25.0, "length": 50.0, "depth": 20.0, "width": 20.0}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# the seven materials of Step 50 (Step 55 added more to agents._CUTTING_PARAMS; the golden hash is over the old seven, whose numbers did not change)
OLD_MATERIALS = ("Aluminum", "Steel", "Brass", "Copper", "Titanium", "Plastic", "Cast Iron")


# ---------------------------------------------------------------- nothing moves without per_op
def test_every_result_without_per_op_is_what_step_50_gave():
    """Golden hashes of op times, physics, the three agents and whole process plans, over 40 dataset parts / 7 materials / 41 operations."""
    parts = dataset.load_dataset(os.path.join(ROOT, "data", "dev_v2.json"))[:40]
    times = [agents._op_time(op, m, p["dims"]) for p in parts for m in OLD_MATERIALS for op in sorted(PT.OPERATIONS)]
    assert digest(times) == "cae97fb70587326c"
    an, plans = [], []
    for p in parts[:12]:
        for r in rb.build_routes(p["features"], max_routes=2):
            steps = r["steps"]
            an.append(physics.analyse_route(steps, p["material"], p["dims"], p["batch_size"], 1.6))
            an.append((agents.time_agent(steps, p["material"], p["dims"]), agents.cost_agent(steps, p["material"], p["batch_size"], p["dims"]),
                       agents.energy_agent(steps, p["material"], p["dims"]), agents.time_breakdown(steps, p["material"], p["dims"])))
            pt = {"part_id": "g", "material": p["material"], "features": p["features"], "tolerance": p["tolerance"], "batch_size": p["batch_size"],
                  "dims": dict(p["dims"], hole_diameter=8.0) if len(plans) % 2 else p["dims"], "required_ra_um": 1.6}
            plans.append(PP.build_plan(steps, pt))
    assert digest(an) == "2ac0615163b86d63"
    assert (digest(plans), len(plans)) == ("155d057b21ade08a", 24)


def test_op_dims_hands_back_the_same_object_when_there_is_no_per_op():
    d = dict(PART)
    assert agents.op_dims("Drilling", d) is d and agents.op_dims("Drilling", None) is None and agents.op_dims("Drilling", {}) == {}


def test_an_empty_per_op_changes_no_time():
    d = dict(PART, per_op={})
    for m in agents._CUTTING_PARAMS:
        for op in PT.OPERATIONS:
            assert agents._op_time(op, m, d) == agents._op_time(op, m, PART)


def test_the_hole_operations_are_those_of_the_physics_model():
    assert tuple(agents.HOLE_OPS) == tuple(physics.HOLE_DIAMETER_OPS)


# ---------------------------------------------------------------- the schema: which dimension each operation reads
def test_the_operations_are_the_41_of_the_vocabulary_facing_first_inspection_last():
    assert len(OPD.ALL_OPS) == 41 and set(OPD.ALL_OPS) == set(V.CANONICAL_OPS) == set(OPD.USES)
    assert OPD.ALL_OPS[0] == "Facing" and OPD.ALL_OPS[-1] == "Inspection" and len(set(OPD.ALL_OPS)) == 41


def test_uses_is_exactly_what_the_time_model_reads():
    """Each dimension is varied on its own: the time of an operation changes with it if and only if USES lists it."""
    base = {"diameter": 40.0, "length": 70.0, "depth": 18.0, "width": 33.0}
    other = {"diameter": (7.0, 90.0), "length": (13.0, 300.0), "depth": (2.0, 90.0), "width": (9.0, 150.0)}
    for op in OPD.ALL_OPS:
        moves = set()
        for dim in base:
            for m in ("Steel", "Aluminum", "Titanium"):
                t0 = agents._op_time(op, m, base)
                if any(agents._op_time(op, m, dict(base, **{dim: v})) != t0 for v in other[dim]):
                    moves.add(dim)
        assert moves == set(OPD.USES[op]), (op, sorted(moves), OPD.USES[op])


def test_a_dimension_an_operation_does_not_use_has_no_box():
    c = OPD.cells(["Parting-off", "Inspection", "Center Drilling", "Pocket Milling"], [], None, PART)
    assert list(c["Parting-off"]) == ["diameter"] and c["Inspection"] == {} and c["Center Drilling"] == {}
    assert list(c["Pocket Milling"]) == ["length", "depth", "width"]


def test_the_limits_are_inside_what_the_boxes_and_the_dataset_accept():
    for dim, (lo, hi) in OPD.LIMITS.items():
        dlo, dhi = dataset.DIM_LIMITS[dim]
        assert lo <= dlo and hi == dhi, dim


# ---------------------------------------------------------------- the selection
def test_autoselect_is_facing_the_chain_of_every_feature_and_inspection_in_list_order():
    sel = OPD.autoselect(["Hole", "Slot", "Step"])
    assert sel == ["Facing", "Step Turning", "Center Drilling", "Drilling", "Slot Milling", "Inspection"]
    assert sel == [o for o in OPD.ALL_OPS if o in sel]


def test_autoselect_is_what_the_route_builder_needs_by_default():
    for feats in (["Hole"], ["Thread", "Groove"], ["Pocket", "Keyway", "Spline"], list(fv.get_all_features())):
        need = fv.get_required_operations_set(feats) | {"Facing", "Inspection"}
        assert set(OPD.autoselect(feats)) == need


def test_autoselect_with_no_feature_or_a_wrong_name():
    assert OPD.autoselect([]) == ["Facing", "Inspection"]
    assert OPD.autoselect(["NotAFeature", "Hole"]) == OPD.autoselect(["Hole"])


def test_every_operation_the_route_builder_puts_in_a_route_of_these_features_is_selected_or_a_support_operation():
    for feats in (["Hole"], ["Slot", "Step"], ["Boss", "Chamfer"]):
        sel = set(OPD.autoselect(feats))
        route = rb.build_routes(feats, max_routes=1)[0]["steps"]
        assert {s for s in route if s != V.MARKER} <= sel | set(PT.OPERATIONS)       # (never an unknown name)
        assert set(fv.get_required_operations_set(feats)) <= sel


def test_the_selection_follows_the_features_and_keeps_a_hand_edit_until_they_change():
    st = {}
    OPD.sync_selection(st, ["Hole"])
    assert st[OPD.SELECTION_KEY] == OPD.autoselect(["Hole"])
    st[OPD.SELECTION_KEY] = ["Reaming"]                       # the user picked something else
    OPD.sync_selection(st, ["Hole"])
    assert st[OPD.SELECTION_KEY] == ["Reaming"]
    OPD.sync_selection(st, ["Hole", "Slot"])                  # the features changed: set again
    assert st[OPD.SELECTION_KEY] == OPD.autoselect(["Hole", "Slot"])
    OPD.sync_selection(st, ["Slot", "Hole"])                  # the same features in another order are not a change
    st[OPD.SELECTION_KEY] = ["Reaming"]
    OPD.sync_selection(st, ["Slot", "Hole"])
    assert st[OPD.SELECTION_KEY] == ["Reaming"]
    del st[OPD.SELECTION_KEY]                                 # a lost widget state is put back
    OPD.sync_selection(st, ["Slot", "Hole"])
    assert st[OPD.SELECTION_KEY] == OPD.autoselect(["Slot", "Hole"])


# ---------------------------------------------------------------- the numbers of the drawing
def extraction(rows, ok=True, success=True):
    return {"success": success, "part_dimensions": {"read": {"ok": ok, "overall": {}, "features": rows, "hole_diameters": [], "ra_um": [], "dropped": []}}}


ROWS = [{"feature": "Hole", "text": "Ø8 THRU, 25 deep", "diameter_mm": 8.0, "depth_mm": 25.0},
        {"feature": "Slot", "text": "40 x 6 x 5", "length_mm": 40.0, "width_mm": 6.0, "depth_mm": 5.0},
        {"feature": "Step", "text": "Ø30 x 20", "diameter_mm": 30.0, "length_mm": 20.0}]


def test_an_operation_gets_the_numbers_of_the_feature_it_makes():
    c = OPD.cells(OPD.autoselect(["Hole", "Slot", "Step"]), ["Hole", "Slot", "Step"], extraction(ROWS), PART)
    assert {d: (v["value"], v["source"]) for d, v in c["Drilling"].items()} == {"diameter": (8.0, "drawing"), "depth": (25.0, "drawing")}
    assert {d: (v["value"], v["source"]) for d, v in c["Slot Milling"].items()} == {"length": (40.0, "drawing"), "depth": (5.0, "drawing")}
    assert {d: (v["value"], v["source"]) for d, v in c["Step Turning"].items()} == {
        "diameter": (30.0, "drawing"), "length": (20.0, "drawing"), "depth": (20.0, "part")}
    assert all(v["source"] == "part" for v in c["Facing"].values()) and c["Facing"]["diameter"]["value"] == 25.0


def test_a_row_of_the_slot_width_goes_nowhere_if_the_slot_cutter_does_not_read_it():
    c = OPD.cells(["Slot Milling"], ["Slot"], extraction(ROWS), PART)
    assert "width" not in c["Slot Milling"]


def test_an_operation_that_only_prepares_a_feature_gets_no_number_of_its_own():
    rows = [{"feature": "Reamed_Hole", "text": "Ø12 H7", "diameter_mm": 12.0, "depth_mm": 30.0}]
    c = OPD.cells(["Drilling", "Reaming"], ["Reamed_Hole"], extraction(rows), PART)
    assert c["Reaming"]["diameter"]["value"] == 12.0 and c["Reaming"]["diameter"]["source"] == "drawing"
    assert c["Drilling"]["diameter"] == {"value": 25.0, "source": "part", "from": "whole part"}
    assert OPD.finishing(["Reamed_Hole", "Hole"]) == {"Reaming": ["Reamed_Hole"], "Drilling": ["Hole"]}


def test_several_rows_for_one_operation_give_the_largest_value_and_say_so():
    rows = [{"feature": "Step", "text": "a", "diameter_mm": 30.0, "length_mm": 20.0}, {"feature": "Step", "text": "b", "diameter_mm": 22.0, "length_mm": 35.0}]
    c = OPD.cells(["Step Turning"], ["Step"], extraction(rows), PART)["Step Turning"]
    assert (c["diameter"]["value"], c["length"]["value"]) == (30.0, 35.0) and "largest of 2" in c["diameter"]["from"]


def test_a_feature_the_user_removed_gives_nothing():
    c = OPD.cells(["Drilling", "Slot Milling"], ["Slot"], extraction(ROWS), PART)
    assert c["Drilling"]["diameter"]["source"] == "part" and c["Slot Milling"]["length"]["source"] == "drawing"


@pytest.mark.parametrize("ex", [None, {}, extraction(ROWS, success=False), extraction(ROWS, ok=False), {"success": True},
                                {"success": True, "part_dimensions": {"read": None, "derived": None}}])
def test_without_a_usable_dimension_read_everything_is_the_part_value(ex):
    c = OPD.cells(OPD.autoselect(["Hole", "Slot"]), ["Hole", "Slot"], ex, PART)
    assert all(v["source"] == "part" for row in c.values() for v in row.values())
    assert OPD.drawing_values(ex, ["Hole", "Slot"]) == {}


def test_numbers_outside_the_limits_are_dropped_not_clipped():
    rows = [{"feature": "Hole", "text": "x", "diameter_mm": 0.05, "depth_mm": 400.0}, {"feature": "Slot", "text": "y", "length_mm": 1500.0, "depth_mm": 0.5}]
    c = OPD.cells(["Drilling", "Slot Milling"], ["Hole", "Slot"], extraction(rows), PART)
    assert c["Drilling"]["diameter"]["source"] == "part" and c["Drilling"]["depth"]["source"] == "part"
    assert c["Slot Milling"]["length"]["source"] == "part" and c["Slot Milling"]["depth"] == {"value": 0.5, "source": "drawing", "from": c["Slot Milling"]["depth"]["from"]}


def test_a_value_that_is_not_a_number_is_ignored():
    rows = [{"feature": "Hole", "text": "x", "diameter_mm": True, "depth_mm": "25"}, "junk", None]
    assert OPD.drawing_values(extraction(rows), ["Hole"]) == {}


def test_a_hole_operation_without_a_drawing_number_takes_the_hole_diameter_of_the_part():
    c = OPD.cells(["Drilling", "Boring", "Facing", "Tapping"], [], None, PART, hole_diameter=8.0)
    assert [c[o]["diameter"]["value"] for o in ("Drilling", "Boring", "Tapping")] == [8.0, 8.0, 8.0]
    assert c["Drilling"]["diameter"]["from"] == "hole diameter of the part" and c["Facing"]["diameter"]["value"] == 25.0
    assert OPD.cells(["Drilling"], [], None, PART, hole_diameter=0.01)["Drilling"]["diameter"]["value"] == 25.0       # below the limit: not used
    assert OPD.cells(["Drilling"], [], None, PART, hole_diameter=None)["Drilling"]["diameter"]["value"] == 25.0


def test_every_cell_value_is_inside_the_limits_of_its_box():
    feats = list(fv.get_all_features())
    c = OPD.cells(OPD.ALL_OPS, feats, extraction(ROWS), PART, hole_diameter=8.0)
    for row in c.values():
        for dim, v in row.items():
            lo, hi = OPD.LIMITS[dim]
            assert lo <= v["value"] <= hi


# ---------------------------------------------------------------- the boxes and what goes into per_op
def test_the_boxes_are_filled_follow_the_part_and_keep_what_the_user_typed():
    st = {}
    cells = OPD.cells(["Drilling", "Facing"], ["Hole"], extraction(ROWS), PART)
    OPD.sync_state(st, cells)
    k_dia, k_face = OPD.key_of("Drilling", "diameter"), OPD.key_of("Facing", "diameter")
    assert (st[k_dia], st[k_face]) == (8.0, 25.0)
    st[k_dia] = 9.5                                            # the user types in the Drilling box
    part2 = dict(PART, diameter=60.0)
    OPD.sync_state(st, OPD.cells(["Drilling", "Facing"], ["Hole"], extraction(ROWS), part2))
    assert (st[k_dia], st[k_face]) == (9.5, 60.0)             # the typed value stays; the untouched box follows the part
    ROWS2 = [dict(ROWS[0], diameter_mm=6.0)]
    OPD.sync_state(st, OPD.cells(["Drilling", "Facing"], ["Hole"], extraction(ROWS2), part2))
    assert st[k_dia] == 9.5
    del st[k_dia]                                              # a box dropped because it was not drawn in a run is filled again
    OPD.sync_state(st, OPD.cells(["Drilling", "Facing"], ["Hole"], extraction(ROWS2), part2))
    assert st[k_dia] == 6.0


def test_a_new_drawing_is_applied_to_boxes_the_user_did_not_touch():
    st = {}
    OPD.sync_state(st, OPD.cells(["Drilling"], ["Hole"], extraction(ROWS), PART))
    OPD.sync_state(st, OPD.cells(["Drilling"], ["Hole"], extraction([dict(ROWS[0], diameter_mm=12.0, depth_mm=40.0)]), PART))
    assert (st[OPD.key_of("Drilling", "diameter")], st[OPD.key_of("Drilling", "depth")]) == (12.0, 40.0)


def values_of(st):
    return {k: v for k, v in st.items() if k.startswith("opdim|")}


def test_per_op_has_the_drawing_numbers_and_the_edits_and_nothing_else():
    st = {}
    cells = OPD.cells(["Facing", "Drilling", "Slot Milling"], ["Hole", "Slot"], extraction(ROWS), PART)
    OPD.sync_state(st, cells)
    assert OPD.per_op_from_values(cells, values_of(st)) == {"Drilling": {"diameter": 8.0, "depth": 25.0}, "Slot Milling": {"length": 40.0, "depth": 5.0}}
    st[OPD.key_of("Facing", "depth")] = 4.0                    # an edit of a box that showed the part's value
    got = OPD.per_op_from_values(cells, values_of(st))
    assert got["Facing"] == {"depth": 4.0} and "diameter" not in got["Facing"]


def test_with_no_drawing_number_and_no_edit_per_op_is_empty_so_nothing_changes():
    for feats in (["Hole"], ["Slot", "Step"], list(fv.get_all_features())):
        for ex in (None, extraction([])):
            st = {}
            cells = OPD.cells(OPD.autoselect(feats), feats, ex, PART, hole_diameter=8.0)
            OPD.sync_state(st, cells)
            assert OPD.per_op_from_values(cells, values_of(st)) == {}


def test_a_value_outside_the_limits_or_missing_is_not_written():
    cells = OPD.cells(["Drilling"], [], None, PART)
    assert OPD.per_op_from_values(cells, {OPD.key_of("Drilling", "diameter"): 9999.0}) == {}
    assert OPD.per_op_from_values(cells, {OPD.key_of("Drilling", "diameter"): None}) == {}
    assert OPD.per_op_from_values(cells, {}) == {}


def test_row_notes_say_where_the_numbers_come_from_dimension_by_dimension():
    cells = OPD.cells(["Drilling", "Facing", "Inspection", "External Threading"], ["Hole", "Thread"],
                      extraction(ROWS + [{"feature": "Thread", "text": "M10x1.25", "diameter_mm": 10.0, "pitch_mm": 1.25}]), PART)
    st = {}
    OPD.sync_state(st, cells)
    v = values_of(st)
    assert OPD.row_note(cells["Drilling"], v, "Drilling") == "\U0001F4D0 diameter, depth: Hole (Ø8, depth 25)"
    assert OPD.row_note(cells["Facing"], v, "Facing") == "whole part"
    assert OPD.row_note(cells["Inspection"], v, "Inspection") == "fixed time: no dimension used"
    assert OPD.row_note(cells["External Threading"], v, "External Threading") == "\U0001F4D0 diameter: Thread (Ø10, pitch 1.25) | whole part: length"
    v[OPD.key_of("Drilling", "depth")] = 30.0
    assert OPD.row_note(cells["Drilling"], v, "Drilling") == "\U0001F4D0 diameter: Hole (Ø8, depth 25) | \u270F\uFE0F edited: depth"
    v[OPD.key_of("Facing", "depth")] = 3.0
    assert OPD.row_note(cells["Facing"], v, "Facing") == "\u270F\uFE0F edited: depth | whole part: diameter"


# ---------------------------------------------------------------- the model with per_op
def test_an_operation_with_its_own_dimensions_is_timed_with_them_and_the_others_are_not():
    d = dict(PART, per_op={"Drilling": {"diameter": 8.0, "depth": 15.0}})
    for m in agents._CUTTING_PARAMS:
        assert agents._op_time("Drilling", m, d) == agents._op_time("Drilling", m, dict(PART, diameter=8.0, depth=15.0))
        assert agents._op_time("Drilling", m, d) != agents._op_time("Drilling", m, PART)
        for op in PT.OPERATIONS:
            if op != "Drilling":
                assert agents._op_time(op, m, d) == agents._op_time(op, m, PART)


def test_a_key_the_operation_does_not_list_comes_from_the_part():
    d = dict(PART, per_op={"Slot Milling": {"length": 80.0}})
    assert agents._op_time("Slot Milling", "Steel", d) == agents._op_time("Slot Milling", "Steel", dict(PART, length=80.0))


def test_the_route_time_changes_by_exactly_the_change_of_that_operation():
    route = ["Facing", "Center Drilling", "Drilling", "Slot Milling", "Inspection"]
    own = {"Drilling": {"diameter": 8.0, "depth": 15.0}}
    t0, t1 = agents.time_agent(route, "Steel", PART), agents.time_agent(route, "Steel", dict(PART, per_op=own))
    delta = agents._op_time("Drilling", "Steel", dict(PART, per_op=own)) - agents._op_time("Drilling", "Steel", PART)
    assert t1 - t0 == pytest.approx(delta, abs=0.01) and t1 != t0
    c0, c1 = agents.cost_agent(route, "Steel", 50, PART), agents.cost_agent(route, "Steel", 50, dict(PART, per_op=own))
    assert c1 != c0
    assert agents.energy_agent(route, "Steel", dict(PART, per_op=own)) == agents.energy_agent(route, "Steel", PART)    # energy has constants per op
    bd = agents.time_breakdown(route, "Steel", dict(PART, per_op=own))
    assert bd["machining_time"] == pytest.approx(agents.time_breakdown(route, "Steel", PART)["machining_time"] + delta, abs=0.01)


def test_per_op_does_not_change_the_dict_it_is_given():
    d = dict(PART, per_op={"Drilling": {"diameter": 8.0}})
    before = copy.deepcopy(d)
    agents._op_time("Drilling", "Steel", d)
    physics.analyse_route(["Facing", "Drilling"], "Steel", d)
    PP.build_plan(["Facing", "Drilling", "Inspection"], {"material": "Steel", "features": ["Hole"], "tolerance": "0.02mm", "batch_size": 5, "dims": d})
    assert d == before


def test_the_physics_of_an_operation_uses_its_own_dimensions_and_its_hole_diameter():
    own = {"Drilling": {"diameter": 8.0, "depth": 15.0}}
    a = physics.analyse_op("Drilling", "Steel", dict(PART, per_op=own))
    b = physics.analyse_op("Drilling", "Steel", dict(PART, diameter=8.0, depth=15.0, hole_diameter=8.0))
    assert a == b and a["t_cut_min"] == agents._op_time("Drilling", "Steel", dict(PART, diameter=8.0, depth=15.0))
    other = physics.analyse_op("Facing", "Steel", dict(PART, per_op=own))
    assert other == physics.analyse_op("Facing", "Steel", PART)


def test_a_hole_diameter_of_the_part_is_replaced_for_that_operation_only():
    d = dict(PART, hole_diameter=12.0, per_op={"Drilling": {"diameter": 8.0}})
    assert physics.analyse_op("Drilling", "Steel", d) == physics.analyse_op("Drilling", "Steel", dict(PART, diameter=8.0, hole_diameter=8.0))
    assert physics.analyse_op("Reaming", "Steel", d) == physics.analyse_op("Reaming", "Steel", dict(PART, hole_diameter=12.0))


def plan(own, extra=None):
    part = {"part_id": "t", "material": "Steel", "features": ["Hole"], "tolerance": "0.02mm", "batch_size": 5,
            "dims": dict(PART, **({"per_op": own} if own else {}), **(extra or {})), "required_ra_um": None}
    return PP.build_plan(["Facing", "Center Drilling", "Drilling", "Inspection"], part)


def test_the_process_plan_shows_the_own_dimensions_of_an_operation():
    base, p = plan(None), plan({"Drilling": {"diameter": 8.0, "depth": 15.0}})
    e = {x["op"]: x for x in p["operations"]}
    b = {x["op"]: x for x in base["operations"]}
    assert e["Drilling"]["own_dims"] == {"diameter": 8.0, "depth": 15.0}
    assert any("own dimensions for this operation: diameter 8 mm, depth 15 mm" in f for f in e["Drilling"]["flags"])
    assert e["Drilling"]["tool"]["size"] == "Ø8 mm" and e["Drilling"]["tool"]["size_assumed"] is False
    assert b["Drilling"]["tool"]["size_assumed"] is True and "own_dims" not in b["Drilling"]
    assert e["Drilling"]["params"]["passes"] == 1 and e["Drilling"]["params"]["diameter_mm"] == 8.0
    assert e["Drilling"]["time_min"] < b["Drilling"]["time_min"]
    for op in ("Facing", "Center Drilling", "Inspection"):
        assert e[op]["time_min"] == b[op]["time_min"] and "own_dims" not in e[op]
    assert any("1 operation(s) use dimensions of their own" in line for line in p["assumptions"])
    assert not any("dimensions of their own" in line for line in base["assumptions"])


def test_the_passes_shown_in_the_plan_follow_the_depth_of_the_operation():
    base, p = plan(None), plan({"Facing": {"depth": 4.0}})
    f0, f1 = (next(x for x in q["operations"] if x["op"] == "Facing") for q in (base, p))
    assert f0["params"]["passes"] == 4 and f1["params"]["passes"] == 1
    assert f1["params"]["passes"] == agents._num_passes("Facing", 4.0)


def test_the_total_of_the_plan_is_the_total_of_its_operations_with_per_op():
    p = plan({"Drilling": {"diameter": 8.0, "depth": 15.0}, "Facing": {"diameter": 40.0}})
    assert p["totals"]["total_min"] == pytest.approx(sum(e["time_min"] for e in p["operations"]))
    assert p["totals"]["v1_time_min"] is not None


def test_make_part_carries_per_op_as_floats_and_leaves_no_trace_without_it():
    part = AV.make_part("Steel", ["Hole"], "0.02mm", 5, dims=dict(PART, per_op={"Drilling": {"diameter": 8, "depth": 15}, "Facing": {}}), hole_diameter=6)
    assert part["dims"]["per_op"] == {"Drilling": {"diameter": 8.0, "depth": 15.0}} and part["dims"]["hole_diameter"] == 6.0
    assert "per_op" not in AV.make_part("Steel", ["Hole"], "0.02mm", 5, dims=dict(PART, per_op={}))["dims"]
    assert "per_op" not in AV.make_part("Steel", ["Hole"], "0.02mm", 5, dims=dict(PART))["dims"]
    assert AV.make_part("Steel", ["Hole"], "0.02mm", 5, dims=None)["dims"] is None


def test_the_feasibility_rules_and_the_analysis_run_on_a_part_with_per_op():
    part = AV.make_part("Steel", ["Hole"], "0.02mm", 5, dims=dict(PART, per_op={"Drilling": {"diameter": 8.0, "depth": 15.0}}), required_ra_um=1.6, hole_diameter=8)
    steps = ["Facing", "Center Drilling", "Drilling", "Inspection"]
    view = AV.analyse_route_view(steps, part)
    assert view is not None
    json.dumps(PP.build_plan(steps, part), default=str)                    # the plan, with per_op inside dims, is still serialisable


# ---------------------------------------------------------------- the page
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest      # noqa: E402

APP = os.path.join(ROOT, "week6", "app.py")


@pytest.fixture
def no_network(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    monkeypatch.setenv("CAPP_CORRECTIONS_LOG", str(tmp_path / "log" / "c.jsonl"))


def page_extraction(rows=None, overall=None):
    rows = ROWS[:1] + [{"feature": "Groove", "text": "3 x 2 deep", "width_mm": 3.0, "depth_mm": 2.0}] if rows is None else rows
    read = {"ok": True, "overall": overall if overall is not None else {"diameter": 50.0, "length": 120.0}, "features": rows,
            "hole_diameters": [8.0], "ra_um": [], "dropped": [], "note": ""}
    derived = {"diameter": 50.0, "length": 120.0, "width": None, "depth": 25.0, "hole_diameter": 8.0, "ra_um": None, "source": {}, "notes": []}
    return {"success": True, "confidence": "high", "notes": "", "rejected_features": [], "features": ["Hole", "Groove"], "hole_diameters": [8.0],
            "dimensions": {"success": True, "ok": True, "hole_diameters": [8.0], "dropped": [], "note": ""},
            "part_dimensions": {"read": read, "derived": derived}}


MODEL_V1 = "\U0001F4C8 v1 model (numbers of the report)"       # Step 52 made the calculator model the default of Full Analysis: these tests are about the v1 panel


def page(ex=None, features=("Hole", "Groove"), model=MODEL_V1):
    at = AppTest.from_file(APP, default_timeout=180)
    if ex is not None:
        at.session_state["extraction_result"] = ex
    at.session_state["confirmed_features"] = list(features)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if model is not None:
        at.radio(key="time_model").set_value(model).run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def box(at, op, dim):
    return at.number_input(key=OPD.key_of(op, dim))


def captions(at):
    return " | ".join(c.value for c in at.caption)


def test_the_dropdown_has_all_41_operations_and_the_ones_the_features_need_are_selected(no_network):
    at = page(page_extraction())
    ms = at.multiselect(key=OPD.SELECTION_KEY)
    assert list(ms.options) == list(OPD.ALL_OPS) and len(ms.options) == 41
    assert ms.value == OPD.autoselect(["Hole", "Groove"]) == ["Facing", "Grooving/Necking", "Center Drilling", "Drilling", "Inspection"]


def test_the_boxes_are_filled_from_the_drawing_per_operation(no_network):
    at = page(page_extraction())
    assert (box(at, "Drilling", "diameter").value, box(at, "Drilling", "depth").value) == (8.0, 25.0)
    assert box(at, "Grooving/Necking", "depth").value == 2.0
    assert box(at, "Grooving/Necking", "diameter").value == 50.0          # the groove row has no diameter: the part's value
    assert (box(at, "Facing", "diameter").value, box(at, "Facing", "depth").value) == (50.0, 25.0)
    text = captions(at)
    assert "diameter, depth: Hole (Ø8, depth 25)" in text and "2 of 5 selected operations use dimensions of their own (3 number(s) read from the drawing)" in text
    assert "whole part" in text and "fixed time: no dimension used" in text
    with pytest.raises(Exception):
        at.number_input(key=OPD.key_of("Inspection", "diameter"))       # no box for a dimension that is not used


def test_a_number_the_user_types_is_kept_and_a_new_drawing_does_not_overwrite_it(no_network):
    at = page(page_extraction())
    box(at, "Drilling", "depth").set_value(30.0).run()
    assert box(at, "Drilling", "depth").value == 30.0 and "edited: depth" in captions(at)
    at.run()
    assert box(at, "Drilling", "depth").value == 30.0
    at.session_state["extraction_result"] = page_extraction([dict(ROWS[0], diameter_mm=6.0, depth_mm=12.0)])
    at.run()
    assert box(at, "Drilling", "diameter").value == 6.0 and box(at, "Drilling", "depth").value == 30.0


def test_an_untouched_box_follows_the_part_dimensions_of_the_sidebar(no_network):
    at = page(page_extraction())
    at.number_input(key="dim_diameter").set_value(70.0).run()
    assert box(at, "Facing", "diameter").value == 70.0
    assert box(at, "Drilling", "diameter").value == 8.0                    # that one comes from the drawing


def test_an_operation_added_by_hand_gets_a_row_with_the_part_values(no_network):
    at = page(page_extraction())
    ms = at.multiselect(key=OPD.SELECTION_KEY)
    ms.set_value(list(ms.value) + ["Pocket Milling"]).run()
    assert not at.exception
    assert [box(at, "Pocket Milling", d).value for d in ("length", "depth", "width")] == [120.0, 25.0, 20.0]


def test_changing_the_features_selects_the_operations_again(no_network):
    at = page(page_extraction())
    at.multiselect(key="confirmed_features").set_value(["Hole", "Slot"]).run()
    assert at.multiselect(key=OPD.SELECTION_KEY).value == OPD.autoselect(["Hole", "Slot"])


def test_the_panel_works_without_a_drawing_in_manual_mode_and_claims_nothing(no_network):
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.radio(key="input_method").set_value("\u270d\ufe0f Select the features myself").run()
    assert not at.exception, [e.value for e in at.exception]
    at.radio(key="time_model").set_value(MODEL_V1).run()
    at.multiselect(key="manual_features").set_value(["Hole", "Slot"]).run()
    assert at.multiselect(key=OPD.SELECTION_KEY).value == OPD.autoselect(["Hole", "Slot"])
    assert box(at, "Slot Milling", "length").value == 50.0 and "no number from a drawing" in captions(at)
    assert not [c.value for c in at.caption if c.value.startswith("\U0001F4D0")] and "whole part" in captions(at)


def test_route_only_mode_has_no_panel(no_network):
    at = page(page_extraction())
    at.radio[0].set_value("\U0001f5fa️ Route Only").run()
    assert not at.exception
    assert all(m.key not in (OPD.SELECTION_KEY, "op_sel_calc") for m in at.multiselect)
    at.radio[0].set_value("\U0001f4ca Route + Full Analysis").run()
    assert at.multiselect(key="op_sel_calc").value == OPD.autoselect(["Hole", "Groove"])       # Step 52/54: the calculator model is the default, its dropdown has its own key
    at.radio(key="time_model").set_value(MODEL_V1).run()          # the model radio is not drawn in Route Only: it starts again at the default
    assert at.multiselect(key=OPD.SELECTION_KEY).value == OPD.autoselect(["Hole", "Groove"])
    assert box(at, "Drilling", "depth").value == 25.0


def test_generate_sends_the_per_operation_dimensions_to_the_analysis_and_says_which_were_used(no_network, monkeypatch):
    seen = []
    real = AV.make_part

    def spy(*a, **k):
        seen.append((a, k))
        return real(*a, **k)
    monkeypatch.setattr(AV, "make_part", spy)
    at = page(page_extraction())
    box(at, "Drilling", "depth").set_value(30.0).run()
    [b for b in at.button if "Generate" in b.label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    args, kw = seen[-1]
    dims = kw.get("dims") or args[4]
    assert dims["per_op"] == {"Drilling": {"diameter": 8.0, "depth": 30.0}, "Grooving/Necking": {"depth": 2.0}}
    assert (dims["diameter"], dims["length"], dims["depth"]) == (50.0, 120.0, 25.0)
    text = captions(at)
    assert "Dimensions of their own were used for: " in text and "Drilling (diameter 8, depth 30)" in text and "Grooving/Necking (depth 2)" in text
    assert box(at, "Drilling", "depth").value == 30.0                      # the boxes are still there after the click, with their values


def test_generate_without_any_drawing_number_sends_dims_exactly_as_before(no_network, monkeypatch):
    seen = []
    real = AV.make_part

    def spy(*a, **k):
        seen.append((a, k))
        return real(*a, **k)
    monkeypatch.setattr(AV, "make_part", spy)
    ex = page_extraction(rows=[], overall={})
    at = page(ex)
    [b for b in at.button if "Generate" in b.label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    args, kw = seen[-1]
    dims = kw.get("dims") or args[4]
    assert "per_op" not in dims
    assert "Dimensions of their own were used" not in captions(at)
