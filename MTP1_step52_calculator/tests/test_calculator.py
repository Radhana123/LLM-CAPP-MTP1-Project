"""Step 52: time and cost the way the Time-and-Cost-calculator (Radhana123/Time-and-Cost-calculator, streamlit_app.py) calculates them
(capp/calculator.py), the model switch and the calculator inputs in week6/app.py, and the home page numbers (capp/home_stats.py).

The oracle: `her_time` below is a copy, line by line, of the "Submit All" block of streamlit_app.py (lines 379-472 of the commit this step was
made against), with the `entry` dictionary of the calculator as its input. capp/calculator.py is compared with it on random inputs.
What the calculator does not have (energy, milling) is NOT tested against it: those tests check that the v1 model is used and said so.
No network: the page tests (Streamlit's AppTest) put an extraction result in the session state.
"""
import math
import os
import random
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "week1"))

from capp import legacy  # noqa: E402,F401
import agents  # noqa: E402
import precedence_graph as pg  # noqa: E402
from feature_vocab import GEOMETRY_FEATURES  # noqa: E402
from capp import calculator as CALC, home_stats as HS, op_dims as OPD  # noqa: E402

MARKER = "--- Machine Changeover ---"
MODEL_CALC = "\U0001F9EE Calculator model (your Time-and-Cost-calculator)"
MODEL_V1 = "\U0001F4C8 v1 model (numbers of the report)"


# ---------------------------------------------------------------- the oracle: the calculator's own code
def her_time(entry):
    """Copy of the formulas of streamlit_app.py ('Submit All'); returns the time of one machining entry in minutes."""
    process_type = entry["type"]
    time = 0.0
    if process_type == "Boring":
        try:
            a = math.radians((180 - entry["angle"]) / 2)
            extra_length = (entry["initial diameter"] / 2) * (math.tan(a))
            turn = (entry["final diameter"] - entry["initial diameter"]) / ((entry["depth of cut"]) * 2)
            time = (((entry["approach"] + entry["overrun"] + (entry["depth"])) / (entry["rpm"] * entry["feed"]) * (turn * 2)) + (((10 + extra_length) / (entry["feed"] * entry["rpm"])) * (turn * 2))) + (entry["job set time"] + entry["tool set time"])
        except ZeroDivisionError:
            time = 0
    elif process_type in ("Drilling - Center", "Drilling - Pilot", "Drilling - Main"):
        try:
            drill_length = ((entry["diameter"] / 2) * (math.tan(math.radians((180 - 118) / 2)))) + entry["depth"]
            time = ((entry["approach"] + entry["overrun"] + drill_length) / (entry["rpm"] * entry["feed"]) * (entry["turn"])) + (entry["job set time"] + entry["tool set time"])
        except ZeroDivisionError:
            time = 0
    elif process_type == "Facing":
        try:
            turn = (entry["length"] / entry["depth of cut"])
            time = ((((entry["approach"] + entry["overrun"] + (entry["diameter"] / 2)) / (entry["feed"] * entry["rpm"])) * turn)) + (entry["job set time"] + entry["tool set time"])
        except ZeroDivisionError:
            time = 0
    elif process_type == "Grooving":
        try:
            turn = (entry["length"]) / (entry["tool width"])
            time = ((((entry["approach"] + entry["overrun"] + ((entry["final diameter"] - entry["initial diameter"]) / 2)) / (entry["feed"] * entry["rpm"])) * turn)) + (entry["job set time"] + entry["tool set time"])
        except ZeroDivisionError:
            time = 0
    elif process_type == "Knurling":
        try:
            time = ((((entry["approach"] + entry["overrun"] + (entry["length"])) / (entry["feed"] * entry["rpm"])) * 2)) + (entry["job set time"] + entry["tool set time"])
        except ZeroDivisionError:
            time = 0
    elif process_type == "Reaming":
        try:
            turn = (entry["final diameter"] - entry["initial diameter"]) / (2 * entry["depth of cut"])
            time = ((((entry["approach"] + entry["overrun"] + (entry["length"])) / (entry["feed"] * entry["rpm"])) * turn)) + (entry["job set time"] + entry["tool set time"])
        except ZeroDivisionError:
            time = 0
    elif process_type == "Threading":
        try:
            turn = (0.6134 * entry["pitch"]) / (entry["depth of cut"])
            time = (((entry["approach"] + entry["overrun"] + (entry["length"])) / (entry["rpm"] * entry["feed"]) * (turn))) + (entry["job set time"] + entry["tool set time"])
        except ZeroDivisionError:
            time = 0
    elif process_type == "Turning - Concave":
        try:
            concave_length = entry["length"] * (math.radians(entry["angle"]))
            turn = (entry["diameter"]) / (entry["depth of cut"] * 4)
            time = ((((entry["approach"] + entry["overrun"] + concave_length) / (entry["feed"] * entry["rpm"])) * turn)) + (entry["job set time"] + entry["tool set time"])
        except ZeroDivisionError:
            time = 0
    elif process_type == "Turning - Convex":
        try:
            convex_length = entry["length"] * (math.radians(entry["angle"]))
            turn = (entry["diameter"]) / (entry["depth of cut"] * 4)
            time = ((((entry["approach"] + entry["overrun"] + convex_length) / (entry["feed"] * entry["rpm"])) * turn)) + (entry["job set time"] + entry["tool set time"])
        except ZeroDivisionError:
            time = 0
    elif process_type == "Turning - Straight":
        try:
            turn = ((entry["initial diameter"]) - (entry["final diameter"])) / ((entry["depth of cut"]) * 2)
            time = ((((entry["approach"] + entry["overrun"] + (entry["length"])) / (entry["feed"] * entry["rpm"])) * turn)) + (entry["job set time"] + entry["tool set time"])
        except ZeroDivisionError:
            time = 0
    elif process_type == "Turning - Taper":
        try:
            h_length = math.sqrt(((entry["length"]) ** 2) + (((entry["larger diameter"] - entry["smaller diameter"]) / 2) ** 2))
            turn = (((entry["larger diameter"]) - (entry["smaller diameter"])) / ((entry["depth of cut"]) * 2))
            time = ((((entry["approach"] + entry["overrun"] + h_length) / (entry["feed"] * entry["rpm"])) * turn)) + (entry["job set time"] + entry["tool set time"])
        except ZeroDivisionError:
            time = 0
    return time


def her_totals(times, tool_costs, extra_minutes, extra_costs, material_cost, labor_per_hour):
    """Copy of the totals of the same block."""
    total_time_min = sum(times) + sum(m + 5 for m in extra_minutes)
    labor = (total_time_min / 60) * labor_per_hour
    return total_time_min, labor, material_cost + labor + sum(extra_costs) + sum(tool_costs)


PROCESS_NAMES = list(CALC.PROCESSES)


def random_entry(process, rng):
    e = {"type": process}
    for key, _label, default in CALC.PROCESSES[process]:
        e[key] = max(0.0, default * rng.uniform(0.3, 2.5)) if key not in ("angle",) else rng.uniform(30, 150)
    return e


# ---------------------------------------------------------------- the formulas
def test_the_calculator_has_13_machining_processes_and_6_other_ones():
    assert sorted(PROCESS_NAMES) == sorted(["Boring", "Drilling - Center", "Drilling - Pilot", "Drilling - Main", "Facing", "Grooving", "Knurling",
                                            "Reaming", "Threading", "Turning - Concave", "Turning - Convex", "Turning - Straight", "Turning - Taper"])
    assert CALC.EXTRA == {"Custom": (5.0, 100.0), "Blanking": (15.0, 50.0), "Chamfering": (5.0, 10.0), "Parting": (20.0, 50.0),
                          "Resharpening": (35.0, 50.0), "Tool Change": (35.0, 50.0)}


@pytest.mark.parametrize("process", PROCESS_NAMES)
def test_every_formula_equals_the_code_of_the_calculator_on_random_inputs(process):
    rng = random.Random(52)
    checked = 0
    for _ in range(200):
        e = random_entry(process, rng)
        cut_free = her_time(e) - e["job set time"] - e["tool set time"]
        got = CALC.process_time(process, e)
        if cut_free >= 0:
            assert got == pytest.approx(her_time(e), rel=1e-12, abs=1e-12), (process, e)
            checked += 1
        else:                                                    # the one difference: a negative cutting time is taken as 0
            assert got == pytest.approx(e["job set time"] + e["tool set time"])
    assert checked > 100


@pytest.mark.parametrize("process", PROCESS_NAMES)
def test_the_default_inputs_give_the_time_the_calculator_shows(process):
    e = dict(CALC.DEFAULTS[process], type=process)
    assert CALC.process_time(process, e) == pytest.approx(her_time(e))


def test_default_times_by_hand():
    """Three defaults worked out on paper, not with either program."""
    assert CALC.process_time("Facing", CALC.DEFAULTS["Facing"]) == pytest.approx(5 * 45 / 27.2 + 5)                        # 5 passes of (10+10+25)/(0.16x170)
    assert CALC.process_time("Knurling", CALC.DEFAULTS["Knurling"]) == pytest.approx(2 * 28 / 12 + 5)                      # 2 passes of (10+18)/(0.2x60)
    assert CALC.process_time("Threading", CALC.DEFAULTS["Threading"]) == pytest.approx(0.6134 * 2 / 0.1 * 22 / 12 + 5)      # 12.268 passes of (10+12)/(0.2x60)


def test_the_defaults_are_the_ones_of_the_calculator():
    d = CALC.DEFAULTS
    assert (d["Facing"]["diameter"], d["Facing"]["length"], d["Facing"]["depth of cut"], d["Facing"]["overrun"]) == (50, 10, 2.0, 10.0)
    assert (d["Knurling"]["feed"], d["Knurling"]["rpm"]) == (0.20, 60) and (d["Threading"]["pitch"], d["Threading"]["rpm"]) == (2, 60)
    assert (d["Drilling - Main"]["turn"], d["Drilling - Main"]["diameter"], d["Drilling - Center"]["turn"]) == (7, 25, 4)
    assert all(d[p]["tool set time"] == 5.0 and d[p]["approach"] == 10.0 and d[p]["tool cost"] == 10.0 and d[p]["job set time"] == 0.0 for p in d)
    assert (CALC.LABOR_PER_HOUR, CALC.MATERIAL_COST, CALC.HANDLING_MIN) == (500.0, 1000.0, 5.0)


@pytest.mark.parametrize("zero", ["feed", "rpm"])
def test_a_zero_divisor_gives_time_0_as_in_the_calculator(zero):
    e = dict(CALC.DEFAULTS["Turning - Straight"], type="Turning - Straight", **{zero: 0.0})
    assert her_time(e) == 0 and CALC.process_time("Turning - Straight", e) == 0.0 and CALC.divides_by_zero("Turning - Straight", e)
    r = CALC.op_row("Plain/Cylindrical Turning", "Steel", None, CALC.Settings(edits={"Plain/Cylindrical Turning": {zero: 0.0}}))
    assert r["time_min"] == 0.0 and "divisor is 0" in r["note"]


def test_a_final_diameter_above_the_initial_one_is_not_subtracted_from_the_setting_time():
    e = dict(CALC.DEFAULTS["Turning - Straight"], **{"initial diameter": 38.0, "final diameter": 50.0})
    assert her_time(dict(e, type="Turning - Straight")) < 5.0                # the calculator would show a time below the setting time
    assert CALC.process_time("Turning - Straight", e) == 5.0                 # here: the setting time


# ---------------------------------------------------------------- cost and totals
def test_the_total_is_material_plus_labour_plus_tool_plus_extra_as_in_the_calculator():
    steps = ["Facing", "Plain/Cylindrical Turning", "Chamfering", "Drilling", "Parting-off"]
    s = CALC.Settings(labor_per_hour=420.0, material_cost=750.0)
    r = CALC.evaluate(steps, "Steel", {"diameter": 40.0, "length": 90.0, "depth": 30.0, "width": 20.0}, s)
    rows = {x["op"]: x for x in r["rows"]}
    machining = [(rows[o]["time_min"], rows[o]["tool_cost"]) for o in ("Facing", "Plain/Cylindrical Turning", "Drilling")]
    t, labor, total = her_totals([m[0] for m in machining], [m[1] for m in machining], [5.0, 20.0], [10.0, 50.0], 750.0, 420.0)
    assert r["time_min"] == pytest.approx(t) and r["labor_cost"] == pytest.approx(labor) and r["total_cost"] == pytest.approx(total)
    assert r["tool_cost"] == 30.0 and r["extra_cost"] == 60.0 and r["material_cost"] == 750.0


def test_each_machining_row_equals_the_calculator_for_the_inputs_it_shows():
    r = CALC.evaluate(["Facing", "Knurling", "Boring"], "Aluminum", {"diameter": 60.0, "length": 100.0, "depth": 20.0, "width": 20.0, "hole_diameter": 12.0})
    for row in r["rows"]:
        assert row["source"] == "calculator"
        assert row["time_min"] == pytest.approx(her_time(dict(row["fields"], type=row["process"])))
        assert row["labor_cost"] == pytest.approx(row["time_min"] * 500.0 / 60.0) and row["tool_cost"] == row["fields"]["tool cost"]


def test_an_other_process_costs_its_time_plus_5_min_and_its_extra_tool_cost():
    for op, proc in CALC.EXTRA_OF.items():
        t, c = CALC.EXTRA[proc]
        row = CALC.op_row(op, "Steel")
        assert (row["time_min"], row["extra_cost"], row["tool_cost"], row["source"]) == (t + 5.0, c, 0.0, "calculator (other process)")
    row = CALC.op_row("Chamfering", "Steel", None, CALC.Settings(edits={"Chamfering": {"time_min": 8.0, "extra_tool_cost": 25.0}}))
    assert (row["time_min"], row["extra_cost"]) == (13.0, 25.0)


def test_labour_and_material_cost_come_from_the_settings():
    steps = ["Facing", "Inspection"]
    a = CALC.evaluate(steps, "Steel", None, CALC.Settings(500, 1000))
    b = CALC.evaluate(steps, "Steel", None, CALC.Settings(1000, 2500))
    assert b["labor_cost"] == pytest.approx(2 * a["labor_cost"]) and b["total_cost"] - a["total_cost"] == pytest.approx(1500 + a["labor_cost"])
    assert b["time_min"] == a["time_min"]


# ---------------------------------------------------------------- what the calculator does not have
def test_the_operations_with_a_process_are_17_and_the_other_24_are_timed_by_v1_and_say_so():
    mapped = [o for o in OPD.ALL_OPS if CALC.has_process(o)]
    assert len(mapped) == 17 and len(OPD.ALL_OPS) - len(mapped) == 24
    assert set(CALC.PROCESS_OF.values()) <= set(CALC.PROCESSES) and set(CALC.EXTRA_OF.values()) <= set(CALC.EXTRA)
    assert not any(CALC.has_process(o) for o in OPD.ALL_OPS if "Milling" in o or o in ("Counterboring", "Countersinking", "Spotfacing", "Inspection"))
    dims = {"diameter": 40.0, "length": 90.0, "depth": 30.0, "width": 20.0}
    for op in ("Pocket Milling", "Slot Milling", "Counterboring", "Polishing/Burnishing"):
        row = CALC.op_row(op, "Steel", dims)
        assert row["source"] == CALC.NOT_IN_CALCULATOR and "v1 cutting time" in row["note"]
        assert row["time_min"] == pytest.approx(agents._op_time(op, "Steel", dims) + 5.0) and row["tool_cost"] == 10.0


def test_inspection_keeps_the_fixed_v1_time_and_costs_no_tool():
    d = {"diameter": 40.0, "length": 90.0, "depth": 30.0, "width": 20.0}
    row = CALC.op_row("Inspection", "Steel", d)
    assert row["time_min"] == agents._op_time("Inspection", "Steel", d) and row["tool_cost"] == 0.0 and row["source"] == CALC.NOT_IN_CALCULATOR


def test_energy_is_the_v1_energy_unchanged_and_does_not_depend_on_the_calculator_inputs():
    steps = ["Facing", "Plain/Cylindrical Turning", MARKER, "Slot Milling", "Inspection"]
    dims = {"diameter": 40.0, "length": 90.0, "depth": 30.0, "width": 20.0}
    base = CALC.evaluate(steps, "Brass", dims)
    edited = CALC.evaluate(steps, "Brass", dims, CALC.Settings(edits={"Facing": {"feed": 0.5}}, labor_per_hour=900))
    assert base["energy_kwh"] == edited["energy_kwh"] == agents.energy_agent(steps, "Brass", dims) == CALC.energy_agent(steps, "Brass", dims)


def test_a_machine_changeover_costs_the_v1_minutes_at_the_labour_rate_and_nothing_else():
    steps = ["Facing", MARKER, "Slot Milling"]
    r = CALC.evaluate(steps, "Steel", {"diameter": 40.0, "length": 90.0, "depth": 30.0, "width": 20.0})
    co = [x for x in r["rows"] if x["op"] == MARKER][0]
    assert co["time_min"] == agents.CHANGEOVER_TIME_MIN and co["labor_cost"] == pytest.approx(co["time_min"] * 500 / 60)
    assert co["tool_cost"] == co["extra_cost"] == 0.0 and r["n_ops"] == 2 and len(r["rows"]) == 3


def test_the_counts_of_the_summary():
    r = CALC.evaluate(["Facing", "Drilling", "Slot Milling", "Chamfering", "Inspection"], "Steel", {"diameter": 40.0, "length": 90.0, "depth": 30.0, "width": 20.0})
    assert (r["n_ops"], r["n_calculator"], r["n_v1"]) == (5, 3, 2)


def test_no_batch_discount_in_the_calculator_model():
    steps = ["Facing", "Drilling", "Inspection"]
    d = {"diameter": 40.0, "length": 90.0, "depth": 30.0, "width": 20.0}
    assert CALC.cost_agent(steps, "Steel", 1, d) == CALC.cost_agent(steps, "Steel", 10000, d)


def test_the_calculator_model_does_not_change_the_v1_agents_or_the_search():
    """NSGA-II, the frozen ablation and the report use week3/agents.py and week4/nsga2.py: neither reads the calculator."""
    dims = {"diameter": 40.0, "length": 90.0, "depth": 30.0, "width": 20.0}
    steps = ["Facing", "Plain/Cylindrical Turning", "Drilling", "Inspection"]
    before = (agents.time_agent(steps, "Steel", dims), agents.cost_agent(steps, "Steel", 500, dims), agents.energy_agent(steps, "Steel", dims))
    CALC.evaluate(steps, "Steel", dims, CALC.Settings(900, 5000, edits={"Facing": {"feed": 0.9}}))
    assert before == (agents.time_agent(steps, "Steel", dims), agents.cost_agent(steps, "Steel", 500, dims), agents.energy_agent(steps, "Steel", dims))
    for rel in ("week3/agents.py", "week4/nsga2.py", "week4/__init__.py"):
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            assert "calculator" not in open(path, encoding="utf-8").read().lower().replace("time-and-cost-calculator", "")


# ---------------------------------------------------------------- from the part to the inputs
PART = {"diameter": 50.0, "length": 120.0, "depth": 25.0, "width": 20.0}


def fields(op, dims=PART, own=None):
    return CALC.auto_fields_src(op, dims, own)


def test_without_any_number_of_the_part_every_input_is_the_default_of_the_calculator():
    for op in CALC.PROCESS_OF:
        f, src = fields(op, {})
        assert f == CALC.DEFAULTS[CALC.PROCESS_OF[op]] and set(src.values()) == {"default"}


def test_turning_takes_the_part_diameter_and_length_and_the_calculators_own_ratio_for_the_final_diameter():
    f, src = fields("Plain/Cylindrical Turning")
    assert (f["initial diameter"], f["length"]) == (50.0, 120.0) and f["final diameter"] == pytest.approx(50.0 * 38 / 50)
    assert (src["initial diameter"], src["final diameter"], src["length"], src["feed"]) == ("part", "part", "part", "default")


def test_a_smaller_diameter_of_the_drawing_is_the_final_diameter_of_a_turning_cut_and_a_larger_one_is_not():
    f, src = fields("Step Turning", own={"diameter": 30.0, "length": 20.0})
    assert (f["final diameter"], f["length"], src["final diameter"], src["length"]) == (30.0, 20.0, "drawing", "drawing")
    f, src = fields("Step Turning", own={"diameter": 70.0})
    assert f["final diameter"] == pytest.approx(38.0) and src["final diameter"] == "part"


def test_taper_and_contour_turning():
    f, _ = fields("Taper Turning", own={"diameter": 44.0, "length": 33.0})
    assert (f["larger diameter"], f["smaller diameter"], f["length"]) == (44.0, pytest.approx(44.0 * 38 / 50), 33.0)
    f, _ = fields("Contour Turning")
    assert (f["diameter"], f["length"], f["angle"]) == (50.0, 120.0, 20.0)


def test_facing_takes_the_diameter_and_the_depth_of_the_drawing_as_its_length():
    f, src = fields("Facing", own={"depth": 4.0})
    assert (f["diameter"], f["length"], src["diameter"], src["length"]) == (50.0, 4.0, "part", "drawing")
    assert fields("Facing")[0]["length"] == 10.0                                  # no number on the drawing: the default


def test_a_hole_operation_takes_the_hole_diameter_and_the_depth():
    dims = dict(PART, hole_diameter=8.0)
    f, src = fields("Drilling", dims)
    assert (f["diameter"], f["depth"], src["diameter"], src["depth"]) == (8.0, 25.0, "part", "part")
    f, src = fields("Drilling", dims, {"diameter": 10.0, "depth": 30.0})
    assert (f["diameter"], f["depth"], src["diameter"], src["depth"]) == (10.0, 30.0, "drawing", "drawing")
    f, src = fields("Drilling")                                                  # no hole diameter anywhere: the calculator's 25 mm stays
    assert f["diameter"] == 25.0 and src["diameter"] == "default"


def test_boring_and_reaming_scale_the_other_diameter_with_the_ratios_of_the_defaults():
    f, _ = fields("Boring", dict(PART, hole_diameter=20.0))
    assert (f["initial diameter"], f["final diameter"], f["depth"]) == (20.0, pytest.approx(20.0 * 28 / 25), 25.0)
    f, _ = fields("Reaming", dict(PART, hole_diameter=30.0))
    assert (f["final diameter"], f["initial diameter"], f["length"]) == (30.0, pytest.approx(28.0), 25.0)


def test_grooving_takes_the_depth_twice_for_the_final_diameter_and_the_width_as_the_length():
    f, src = fields("Grooving/Necking", own={"depth": 2.0, "width": 3.0})
    assert (f["initial diameter"], f["final diameter"], f["length"]) == (50.0, 54.0, 3.0)
    assert (src["final diameter"], src["length"], src["initial diameter"]) == ("drawing", "drawing", "part")
    f, _ = fields("Undercutting")
    assert f["final diameter"] == 50.5 and f["length"] == 2.0                      # the calculator's 0.5 mm and 2 mm


def test_threading_and_knurling():
    f, src = fields("External Threading", own={"pitch": 1.5, "length": 30.0})
    assert (f["pitch"], f["length"], src["pitch"]) == (1.5, 30.0, "drawing")
    assert fields("Tapping")[0]["pitch"] == 2.0 and fields("Tapping")[0]["length"] == 12.0           # the length of the part is not the length of a thread
    assert fields("Knurling")[0]["length"] == 18.0 and fields("Knurling", own={"length": 25.0})[0]["length"] == 25.0


def test_center_drilling_keeps_the_defaults():
    assert fields("Center Drilling")[0] == CALC.DEFAULTS["Drilling - Center"]


def test_the_drawing_gives_the_pitch_and_the_groove_width_to_the_calculator_inputs():
    rows = [{"feature": "Thread", "text": "M8 x 1.25", "diameter_mm": 8.0, "length_mm": 15.0, "pitch_mm": 1.25},
            {"feature": "Groove", "text": "3 x 2", "width_mm": 3.0, "depth_mm": 2.0}]
    ex = {"success": True, "part_dimensions": {"read": {"ok": True, "features": rows}}}
    got = OPD.drawing_values(ex, ["Thread", "Groove"])
    assert got["External Threading"]["pitch"]["value"] == 1.25 and got["Grooving/Necking"]["width"]["value"] == 3.0
    assert "pitch" not in OPD.cells(["External Threading"], ["Thread"], ex, PART)["External Threading"]       # the v1 boxes are as before


def test_a_pitch_outside_the_limits_is_dropped():
    ex = {"success": True, "part_dimensions": {"read": {"ok": True, "features": [{"feature": "Thread", "pitch_mm": 50.0, "length_mm": 15.0}]}}}
    assert "pitch" not in OPD.drawing_values(ex, ["Thread"]).get("External Threading", {})


def test_edits_win_over_the_drawing_and_the_part_and_only_for_the_inputs_that_exist():
    s = CALC.Settings(edits={"Drilling": {"feed": 0.3, "depth": 40.0, "nonsense": 9.0}}, own={"Drilling": {"depth": 30.0, "diameter": 10.0}})
    row = CALC.op_row("Drilling", "Steel", PART, s)
    assert (row["fields"]["feed"], row["fields"]["depth"], row["fields"]["diameter"]) == (0.3, 40.0, 10.0) and "nonsense" not in row["fields"]


def test_own_dimensions_of_step_51_count_as_drawing_numbers():
    row = CALC.op_row("Drilling", "Steel", dict(PART, per_op={"Drilling": {"diameter": 9.0, "depth": 33.0}}))
    assert (row["fields"]["diameter"], row["fields"]["depth"]) == (9.0, 33.0)


def test_describe_sources_names_the_origin_of_every_input_once():
    f, src = fields("Drilling", dict(PART, hole_diameter=8.0), {"depth": 30.0})
    text = CALC.describe_sources("Drilling", src, ["feed"])
    assert "from the drawing: drilling depth" in text and "from the part: drill diameter" in text and "edited: feed" in text
    assert "calculator default: " in text and text.count("feed") == 1


def test_the_state_of_the_boxes_follows_the_rules_of_the_dimension_panel():
    state = {}
    CALC.sync_state(state, {"Facing": {"diameter": 50.0, "length": 10.0}})
    assert state[CALC.key_of("Facing", "diameter")] == 50.0
    state[CALC.key_of("Facing", "length")] = 7.0                                        # the user typed 7
    CALC.sync_state(state, {"Facing": {"diameter": 60.0, "length": 12.0}})
    assert state[CALC.key_of("Facing", "diameter")] == 60.0 and state[CALC.key_of("Facing", "length")] == 7.0
    del state[CALC.key_of("Facing", "diameter")]                                        # a box dropped by the page is filled again
    CALC.sync_state(state, {"Facing": {"diameter": 60.0, "length": 12.0}})
    assert state[CALC.key_of("Facing", "diameter")] == 60.0


def test_edits_are_the_boxes_that_differ_from_the_offered_value():
    autos = {"Facing": {"diameter": 50.0, "length": 10.0}, "Drilling": {"feed": 0.16}}
    values = {CALC.key_of("Facing", "diameter"): 50.0, CALC.key_of("Facing", "length"): 4.0, CALC.key_of("Drilling", "feed"): 0.16}
    assert CALC.edits_from_values(autos, values) == {"Facing": {"length": 4.0}}


# ---------------------------------------------------------------- efficiency of routes compared with each other
def test_relative_efficiency_is_100_for_a_route_that_is_best_in_all_three_and_a_mean_of_ratios_otherwise():
    assert CALC.relative_efficiency([(10, 100, 1), (20, 50, 1), (10, 100, 2)]) == [83.33, 83.33, 66.67]
    assert CALC.relative_efficiency([(10, 100, 1), (20, 200, 2)]) == [100.0, 50.0]
    assert CALC.relative_efficiency([(7, 9, 3)]) == [100.0] and CALC.relative_efficiency([]) == []
    assert CALC.relative_scores([(10, 100, 1), (20, 50, 1)]) == [(100.0, 50.0, 100.0), (50.0, 100.0, 100.0)]


def test_relative_efficiency_survives_a_zero_value():
    assert CALC.relative_efficiency([(0, 100, 1), (10, 100, 1)]) == [100.0, 100.0]
    assert CALC.relative_efficiency([(0, 100, 1), (0, 100, 1)]) == [100.0, 100.0]


def test_relative_efficiency_does_not_depend_on_the_scale_of_the_numbers_and_the_v1_one_saturates():
    base = [(60.0, 2500.0, 1.5), (45.0, 2600.0, 1.2), (80.0, 2400.0, 1.0)]
    assert CALC.relative_efficiency(base) == CALC.relative_efficiency([(t * 10, c * 3, e * 0.2) for t, c, e in base])
    assert agents.efficiency_agent(100.0, 2500.0, 1.5) == agents.efficiency_agent(1000.0, 2500.0, 1.5)        # the v1 one cannot tell 100 min from 1000 min
    assert CALC.relative_efficiency([(100.0, 2500.0, 1.5), (1000.0, 2500.0, 1.5)])[0] > CALC.relative_efficiency([(100.0, 2500.0, 1.5), (1000.0, 2500.0, 1.5)])[1]


# ---------------------------------------------------------------- the home page
def test_the_cards_follow_the_code_and_not_a_copy_of_its_numbers(monkeypatch):
    monkeypatch.setattr(OPD, "ALL_OPS", ("Facing", "Inspection"))
    monkeypatch.setattr(HS, "GEOMETRY_FEATURES", ["Hole"] * 3)
    monkeypatch.setattr(HS.pg, "PRECEDENCE_EDGES", [("a", "b")])
    monkeypatch.setattr(agents, "USD_TO_INR", 99.5)
    cards = dict((c[1], c[0]) for c in HS.cards())
    assert (cards["Operations"], cards["Features"], cards["Precedence Rules"], cards["Per USD"]) == ("2", "3", "1", "\u20b999.5")
    assert "1 precedence rules" in " ".join(b[1] for b in HS.blurbs()) and "all 2 operations" in " ".join(b[1] for b in HS.blurbs()).lower()


def test_the_cards_are_read_from_the_code():
    cards = HS.cards()
    assert [c[1] for c in cards] == ["Features", "Operations", "Dataset Parts", "Precedence Rules", "Per USD"]
    assert cards[0][0] == str(len(GEOMETRY_FEATURES)) == "36" and cards[1][0] == "41" == str(len(OPD.ALL_OPS))
    assert cards[3][0] == str(len(pg.PRECEDENCE_EDGES)) == "44" and cards[4][0] == "₹96.095"
    assert cards[2][0] == "320" and cards[2][2] == "200 dev + 120 test"


def test_the_exchange_rate_on_the_page_is_the_one_of_the_cost_model():
    assert HS.usd_text() == f"₹{agents.USD_TO_INR:g}"
    text = " ".join(b[1] for b in HS.blurbs()) + " ".join(c[0] for c in HS.cards())
    assert "95.595" not in text and f"{agents.USD_TO_INR:g}" in text


def test_the_dataset_card_counts_the_files_and_survives_missing_ones(tmp_path):
    import json
    (tmp_path / "dev_v2.json").write_text(json.dumps([{}] * 7))
    assert HS.dataset_sizes(str(tmp_path)) == {"dev": 7}
    assert HS.cards(str(tmp_path))[2][:3:2] == ("7", "7 dev")
    assert HS.cards(str(tmp_path / "nothing"))[2][0] == "–"
    (tmp_path / "test_v2.json").write_text("not json")
    assert HS.dataset_sizes(str(tmp_path)) == {"dev": 7}


def test_the_blurbs_describe_the_current_models():
    titles = [b[0] for b in HS.blurbs()]
    text = " ".join(b[1] for b in HS.blurbs())
    assert any("Calculator" in t for t in titles) and any("v1" in t for t in titles)
    assert "13 machining processes" in text and "6 other processes" in text and "no energy formula" in text
    assert f"+{agents.CHANGEOVER_TIME_MIN} min" in text and "44 precedence rules" in text and "all 41 operations" in text.lower()


# ---------------------------------------------------------------- the page
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest      # noqa: E402

APP = os.path.join(ROOT, "week6", "app.py")
ROWS = [{"feature": "Hole", "text": "Ø8 THRU, 25 deep", "diameter_mm": 8.0, "depth_mm": 25.0},
        {"feature": "Groove", "text": "3 x 2 deep", "width_mm": 3.0, "depth_mm": 2.0}]


@pytest.fixture(autouse=True)
def no_network(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    monkeypatch.setenv("CAPP_CORRECTIONS_LOG", str(tmp_path / "log" / "c.jsonl"))


def extraction():
    read = {"ok": True, "overall": {"diameter": 50.0, "length": 120.0}, "features": ROWS, "hole_diameters": [8.0], "ra_um": [], "dropped": [], "note": ""}
    derived = {"diameter": 50.0, "length": 120.0, "width": None, "depth": 25.0, "hole_diameter": 8.0, "ra_um": None, "source": {}, "notes": []}
    return {"success": True, "confidence": "high", "notes": "", "rejected_features": [], "features": ["Hole", "Groove"], "hole_diameters": [8.0],
            "dimensions": {"success": True, "ok": True, "hole_diameters": [8.0], "dropped": [], "note": ""},
            "part_dimensions": {"read": read, "derived": derived}}


def page(with_drawing=True, features=("Hole", "Groove")):
    at = AppTest.from_file(APP, default_timeout=180)
    if with_drawing:
        at.session_state["extraction_result"] = extraction()
    at.session_state["confirmed_features"] = list(features)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def box(at, op, name):
    return at.number_input(key=CALC.key_of(op, name))


def generate(at):
    [b for b in at.button if "Generate" in b.label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def captions(at):
    return " | ".join(c.value for c in at.caption)


def metric(at, label):
    return [m for m in at.metric if label in m.label][0]


@pytest.fixture
def spy(monkeypatch):
    calls = []
    real = CALC.evaluate

    def wrapper(steps, material, dims=None, settings=None):
        r = real(steps, material, dims, settings)
        calls.append({"steps": list(steps), "material": material, "dims": dims, "settings": settings, "result": r})
        return r
    monkeypatch.setattr(CALC, "evaluate", wrapper)
    return calls


def test_the_calculator_model_is_the_default_of_full_analysis_with_the_calculators_defaults():
    at = page()
    assert at.radio(key="time_model").value == MODEL_CALC and list(at.radio(key="time_model").options) == [MODEL_CALC, MODEL_V1]
    assert at.number_input(key="calc_labor").value == 500.0 and at.number_input(key="calc_material").value == 1000.0
    assert not at.number_input(key="calc_labor").disabled


def test_the_panel_shows_the_inputs_of_the_calculator_process_of_each_operation_filled_from_the_drawing():
    at = page()
    assert (box(at, "Drilling", "diameter").value, box(at, "Drilling", "depth").value) == (8.0, 25.0)       # the hole of the drawing
    assert (box(at, "Drilling", "feed").value, box(at, "Drilling", "rpm").value, box(at, "Drilling", "turn").value) == (0.16, 170.0, 7.0)
    assert (box(at, "Grooving/Necking", "initial diameter").value, box(at, "Grooving/Necking", "final diameter").value,
            box(at, "Grooving/Necking", "length").value) == (50.0, 54.0, 3.0)
    assert (box(at, "Facing", "diameter").value, box(at, "Facing", "length").value) == (50.0, 10.0)
    assert box(at, "Center Drilling", "tool set time").value == 5.0
    text = captions(at)
    assert "from the drawing: drilling depth, drill diameter" in text and "calculator default: " in text and "calculator process" in " ".join(m.value for m in at.markdown)
    with pytest.raises(Exception):
        at.number_input(key=OPD.key_of("Drilling", "diameter"))              # the four v1 boxes are only for the operations without a process


def test_an_operation_without_a_process_keeps_the_four_boxes_and_says_how_it_is_timed():
    at = page()
    ms = at.multiselect(key=OPD.SELECTION_KEY)
    ms.set_value(list(ms.value) + ["Pocket Milling"]).run()
    assert not at.exception
    assert [at.number_input(key=OPD.key_of("Pocket Milling", d)).value for d in ("length", "depth", "width")] == [120.0, 25.0, 20.0]
    assert "Operations the calculator has no process for" in " ".join(m.value for m in at.markdown)


def test_editing_a_calculator_input_changes_the_time_shown_for_that_operation_and_is_kept():
    at = page()
    before = [c.value for c in at.caption if c.value.startswith("⏱") and "min" in c.value]
    box(at, "Drilling", "turn").set_value(14.0).run()
    after = [c.value for c in at.caption if c.value.startswith("⏱")]
    assert before != after and "edited: num of times drilled" in captions(at)
    assert box(at, "Drilling", "turn").value == 14.0
    at.run()
    assert box(at, "Drilling", "turn").value == 14.0


def test_the_untouched_boxes_follow_the_part_dimensions_but_the_edited_one_does_not():
    at = page()
    box(at, "Facing", "length").set_value(4.0).run()
    at.number_input(key="dim_diameter").set_value(70.0).run()
    assert box(at, "Facing", "diameter").value == 70.0 and box(at, "Facing", "length").value == 4.0
    assert box(at, "Drilling", "diameter").value == 8.0                       # from the drawing


def test_generate_evaluates_every_route_with_the_calculator_and_shows_its_numbers(spy):
    at = page()
    box(at, "Drilling", "feed").set_value(0.2).run()
    spy.clear()
    generate(at)
    assert spy, "the calculator model was not used"
    first = spy[0]
    s = first["settings"]
    assert (s.labor_per_hour, s.material_cost) == (500.0, 1000.0)
    assert s.edits == {"Drilling": {"feed": 0.2}}
    assert s.own["Drilling"] == {"diameter": 8.0, "depth": 25.0} and s.own["Grooving/Necking"]["depth"] == 2.0
    assert first["dims"]["hole_diameter"] == 8.0 and first["dims"]["diameter"] == 50.0
    df = at.dataframe[0].value
    assert list(df["Time (min)"]) == [round(c["result"]["time_min"], 2) for c in spy[:len(df)]]
    assert list(df["Cost (₹)"]) == [round(c["result"]["total_cost"], 2) for c in spy[:len(df)]]
    assert list(df["Energy (kWh)"]) == [agents.energy_agent(c["steps"], "Aluminum", first["dims"]) for c in spy[:len(df)]]
    assert "Calculator model: time and cost are computed the way your Time-and-Cost-calculator does" in captions(at)


def test_the_efficiency_column_and_the_radar_use_the_relative_scores_of_the_routes(spy):
    import json
    at = generate(page())
    df = at.dataframe[0].value
    triples = [(round(c["result"]["time_min"], 2), round(c["result"]["total_cost"], 2), c["result"]["energy_kwh"]) for c in spy[:len(df)]]
    assert list(df["Efficiency"]) == CALC.relative_efficiency(triples)
    best = list(df["Efficiency"]).index(max(df["Efficiency"]))
    radar = [e for e in at.get("plotly_chart") if e.key == "fa_radar"][0]
    r = json.loads(radar.proto.spec)["data"][0]["r"]
    want = CALC.relative_scores(triples)[best]
    assert r[:3] == pytest.approx(list(want)) and r[3] == pytest.approx(df["Efficiency"].max()) and min(r) > 0


def test_the_result_of_the_selected_route_is_shown_the_way_the_calculators_result_page_shows_it(spy):
    at = generate(page())
    labels = [m.label for m in at.metric]
    for want in ("Total Time", "Labour Cost", "Material Cost", "Tool Cost", "Extra Process Cost", "TOTAL COST"):
        assert any(want in x for x in labels), want
    best_total = metric(at, "TOTAL COST").value
    assert any(f"₹{c['result']['total_cost']:,.2f}" == best_total for c in spy)
    assert metric(at, "Material Cost").value == "₹1,000.00"
    assert any("Per-Operation Time and Cost" in e.label for e in at.expander)
    text = captions(at)
    assert "the calculator has no energy formula" in text and "Batch size does not change the cost" in text
    heads = " ".join(m.value for m in at.markdown)
    assert "Time Breakdown" not in heads and "Cost Breakdown (" not in heads and "Time & Cost — as in your Time-and-Cost-calculator" in heads
    assert "route sheet below is the physics model's plan" in text


def test_the_table_of_the_calculator_rows_has_the_inputs_and_the_source_of_every_operation(spy):
    at = generate(page())
    tables = [d.value for d in at.dataframe]
    ops = [t for t in tables if "Inputs / note" in list(t.columns)][0]
    assert {"Operation", "Process", "Time (min)", "Labour (₹)", "Tool (₹)", "Extra (₹)", "Source", "Inputs / note"} <= set(ops.columns)
    assert set(ops["Source"]) <= {"calculator", "calculator (other process)", CALC.NOT_IN_CALCULATOR, "v1 model (not in the calculator)"}
    assert any("feed 0.16" in x for x in ops["Inputs / note"])
    assert any(abs(ops["Time (min)"].sum() - c["result"]["time_min"]) < 0.06 for c in spy)         # it is the table of one of the routes evaluated


def test_labour_cost_and_material_cost_of_the_sidebar_change_the_result(spy):
    at = page()
    at.number_input(key="calc_labor").set_value(1000.0).run()
    at.number_input(key="calc_material").set_value(2500.0).run()
    spy.clear()
    generate(at)
    assert (spy[0]["settings"].labor_per_hour, spy[0]["settings"].material_cost) == (1000.0, 2500.0)
    assert metric(at, "Material Cost").value == "₹2,500.00" and "₹1000/h" in metric(at, "Labour Cost").delta


def test_a_calculator_input_set_to_0_gives_a_zero_time_like_the_calculator(spy):
    at = page()
    box(at, "Drilling", "rpm").set_value(0.0).run()
    assert not at.exception
    assert "a divisor is 0" in captions(at)
    generate(at)
    assert not at.exception


def test_the_v1_model_is_still_there_and_does_not_call_the_calculator(spy):
    at = page()
    at.radio(key="time_model").set_value(MODEL_V1).run()
    assert not at.exception
    spy.clear()
    generate(at)
    assert spy == []
    heads = " ".join(m.value for m in at.markdown)
    assert "Time Breakdown" in heads and "Cost Breakdown" in heads and "Time & Cost — as in your" not in heads
    assert not [m for m in at.metric if "TOTAL COST" in m.label]
    assert "1 USD = ₹96.095" in captions(at)
    from agents import efficiency_agent
    df = at.dataframe[0].value
    for _, row in df.iterrows():
        assert row["Efficiency"] == efficiency_agent(row["Time (min)"], row["Cost (₹)"], row["Energy (kWh)"])


def test_the_v1_numbers_of_the_page_are_the_v1_agents_numbers(spy):
    at = page()
    at.radio(key="time_model").set_value(MODEL_V1).run()
    generate(at)
    df = at.dataframe[0].value
    assert len(df) >= 1 and (df["Time (min)"] > 0).all()
    assert (df["Efficiency"] > 0).any()


def test_the_calculator_model_needs_no_drawing_and_works_in_manual_mode():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.radio(key="input_method").set_value("✍️ Select the features myself").run()
    at.multiselect(key="manual_features").set_value(["Hole", "Slot"]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert box(at, "Drilling", "diameter").value == 25.0                      # no hole diameter, no drawing: the calculator's default stays
    assert box(at, "Facing", "diameter").value == 25.0                        # the part diameter of the sidebar
    generate(at)
    assert metric(at, "TOTAL COST").value.startswith("₹")


def test_route_only_mode_has_no_calculator_panel_and_no_model_switch():
    at = page()
    at.radio[0].set_value("\U0001f5fa️ Route Only").run()
    assert not at.exception
    assert all(r.key != "time_model" for r in at.radio) and all(m.key != OPD.SELECTION_KEY for m in at.multiselect)
    generate(at)
    assert not [m for m in at.metric if "TOTAL COST" in m.label]


def test_the_model_radio_survives_a_rerun_and_the_two_cost_inputs_are_greyed_out_in_the_v1_model():
    at = page()
    at.radio(key="time_model").set_value(MODEL_V1).run()
    assert at.radio(key="time_model").value == MODEL_V1 and at.number_input(key="calc_labor").disabled
    at.run()
    assert at.radio(key="time_model").value == MODEL_V1


def test_the_home_page_shows_the_current_numbers_and_no_stale_exchange_rate():
    at = page(with_drawing=False)
    html = " ".join(m.value for m in at.markdown)
    for number, label, _tip in HS.cards():
        assert f'<div class="stat-num">{number}</div><div class="stat-lbl">{label}</div>' in html.replace("\n", "")
    assert "320" in html and "₹96.095" in html and "95.595" not in html and "₹95" not in html.replace("₹96", "")
    assert "Calculator Time & Cost Model" in html and "v1 Time Model" in html and "Dimensions of every operation" in html
    assert "Machine-Aware Routing" in html and "INR Costing, Physics and Route Sheet" in html


def test_the_mapped_operations_that_are_not_selected_still_use_the_numbers_of_the_drawing(spy):
    at = page()
    ms = at.multiselect(key=OPD.SELECTION_KEY)
    ms.set_value([o for o in ms.value if o != "Drilling"]).run()
    spy.clear()
    generate(at)
    assert spy[0]["settings"].own["Drilling"] == {"diameter": 8.0, "depth": 25.0}
    drill = [r for r in spy[0]["result"]["rows"] if r["op"] == "Drilling"]
    assert drill and drill[0]["fields"]["diameter"] == 8.0
