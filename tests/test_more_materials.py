"""Step 55: twelve more materials (19 in all). The seven old ones must not move; the new ones must be in every table, run through every part of the
program, and carry the numbers that the comments of the tables say they come from."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "week1", "week3", "week6"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from capp import legacy  # noqa: E402,F401
import agents  # noqa: E402
import tokenizer  # noqa: E402
from material_tokens import MATERIALS  # noqa: E402
from capp import calculator as CALC, materials as M, physics, planners, process_plan as PP, process_tables as PT  # noqa: E402

OLD = ("Aluminum", "Steel", "Brass", "Copper", "Titanium", "Plastic", "Cast Iron")
NEW = ("Mild Steel", "Alloy Steel", "Tool Steel", "Hardened Steel", "Austenitic Stainless Steel", "Martensitic Stainless Steel", "Nodular Cast Iron",
       "Bronze", "Hardened Aluminum Alloy", "Inconel", "Pure Titanium", "Magnesium")
PART = {"diameter": 50.0, "length": 120.0, "depth": 25.0, "width": 20.0}
ROUTE = ["Facing", "Plain/Cylindrical Turning", "Center Drilling", "Drilling", "--- Machine Changeover ---", "Slot Milling", "Inspection"]


# ---------------------------------------------------------------- the old seven did not move
OLD_PARAMS = {"Aluminum": (300, 0.25, 120, 0.20, 250, 0.10), "Steel": (120, 0.15, 40, 0.10, 80, 0.06), "Brass": (200, 0.20, 80, 0.15, 150, 0.08),
              "Copper": (180, 0.20, 70, 0.15, 120, 0.08), "Titanium": (60, 0.10, 20, 0.06, 40, 0.04), "Plastic": (400, 0.30, 150, 0.25, 300, 0.12),
              "Cast Iron": (100, 0.20, 35, 0.12, 70, 0.06)}
OLD_COST = {"Aluminum": 1.0, "Steel": 1.5, "Brass": 1.2, "Copper": 1.3, "Titanium": 2.5, "Plastic": 0.6, "Cast Iron": 1.4}
OLD_ENERGY = {"Aluminum": 1.0, "Steel": 1.6, "Brass": 1.1, "Copper": 0.95, "Titanium": 2.2, "Plastic": 0.5, "Cast Iron": 1.5}
OLD_KC = {"Aluminum": (700, 0.25), "Steel": (1500, 0.22), "Brass": (700, 0.27), "Copper": (700, 0.27), "Titanium": (2110, 0.22), "Plastic": (200, 0.25),
          "Cast Iron": (1150, 0.20)}


def test_the_old_seven_keep_every_number_and_their_place_at_the_front():
    assert tuple(MATERIALS)[:7] == OLD and tuple(PT.MATERIALS)[:7] == OLD
    for m in OLD:
        assert agents._CUTTING_PARAMS[m] == OLD_PARAMS[m]
        assert agents._MATERIAL_COST_FACTOR[m] == OLD_COST[m] and agents._MATERIAL_ENERGY_FACTOR[m] == OLD_ENERGY[m]
        assert (M.props(m)["kc1_1"], M.props(m)["mc"]) == OLD_KC[m]
    assert [json.load(open(os.path.join(ROOT, "week1", "token_map.json")))["material"][m] for m in OLD] == list(range(201, 208))


# ---------------------------------------------------------------- every table knows all 19
def test_every_table_has_the_same_nineteen_materials():
    names = set(OLD) | set(NEW)
    assert len(MATERIALS) == len(PT.MATERIALS) == 19
    assert set(MATERIALS) == set(PT.MATERIALS) == names
    assert set(M.MATERIAL_PROPS) == set(agents._CUTTING_PARAMS) == set(agents._MATERIAL_COST_FACTOR) == set(agents._MATERIAL_ENERGY_FACTOR) == names
    assert set(planners.MATERIAL_CLASS) == names
    assert all(MATERIALS[m] for m in names)                                               # a description each


def test_the_token_map_gives_each_new_material_the_next_free_token():
    tm = json.load(open(os.path.join(ROOT, "week1", "token_map.json")))["material"]
    assert [tm[m] for m in NEW] == list(range(208, 220)) and len(set(tm.values())) == 19
    others = {v for k, d in json.load(open(os.path.join(ROOT, "week1", "token_map.json"))).items() if k != "material" and isinstance(d, dict) for v in d.values()}
    assert not others & set(tm.values())                                                  # no clash with a feature / tolerance / batch token
    for m in NEW:
        out = tokenizer.tokenize({"material": m, "features": ["Hole"], "tolerance": "0.02mm", "batch_size": 50})
        assert out["success"] and out["tokens"][0] == tm[m] and out["token_labels"][0] == m


# ---------------------------------------------------------------- where the numbers come from
# rows of the kc table of machiningdoctor.com (kc1.1 N/mm2, mc), as read on the page
KC_TABLE = {"Mild Steel": (1350, 0.21), "Alloy Steel": (1725, 0.24), "Tool Steel": (2450, 0.23), "Hardened Steel": (4600, 0.25),
            "Austenitic Stainless Steel": (2150, 0.20), "Martensitic Stainless Steel": (1875, 0.21), "Nodular Cast Iron": (1225, 0.25),
            "Bronze": (700, 0.27), "Hardened Aluminum Alloy": (800, 0.25), "Inconel": (3300, 0.24), "Pure Titanium": (1700, 0.23)}


@pytest.mark.parametrize("m", sorted(KC_TABLE))
def test_kc_is_the_row_of_the_table(m):
    assert (M.props(m)["kc1_1"], M.props(m)["mc"]) == KC_TABLE[m] and "ASSUMED" not in M.props(m)["kc_group"]


def test_magnesium_is_the_one_new_material_whose_kc_is_an_assumption_and_says_so():
    assert M.ASSUMED_MATERIALS_KC == ("Plastic", "Magnesium") and "ASSUMED" in M.props("Magnesium")["kc_group"]
    assert M.props("Magnesium")["kc1_1"] < M.props("Aluminum")["kc1_1"]                   # the lowest specific cutting energy of the structural metals


RATIO = {  # (new material, v1 row it is scaled from, ratio): the comment above _CUTTING_PARAMS
    "Mild Steel": ("Steel", 78 / 59), "Alloy Steel": ("Steel", 0.70), "Tool Steel": ("Steel", 42 / 59), "Hardened Steel": ("Steel", 0.40),
    "Austenitic Stainless Steel": ("Steel", 45 / 59), "Martensitic Stainless Steel": ("Steel", 54 / 59), "Nodular Cast Iron": ("Cast Iron", 0.80),
    "Bronze": ("Brass", 0.75), "Hardened Aluminum Alloy": ("Aluminum", 0.85), "Inconel": ("Steel", 0.25), "Pure Titanium": ("Titanium", 1.25),
    "Magnesium": ("Aluminum", 480 / 360)}


@pytest.mark.parametrize("m", NEW)
def test_the_speeds_are_the_v1_row_times_the_ratio_of_the_comment(m):
    base, r = RATIO[m]
    got = agents._CUTTING_PARAMS[m]
    for i in (0, 2, 4):                                                                    # turning, drilling, milling speed
        assert got[i] == pytest.approx(OLD_PARAMS[base][i] * r, abs=1.0), (m, i)
    assert all(v > 0 for v in got)


def test_energy_factor_is_interpolated_from_the_old_ones_against_kc():
    anch = [(700, 1.0), (1150, 1.5), (1500, 1.6), (2110, 2.2)]

    def want(k):
        a, b = (anch[0], anch[1]) if k <= 700 else (anch[-2], anch[-1]) if k >= 2110 else next((anch[i], anch[i + 1]) for i in range(3) if anch[i][0] <= k <= anch[i + 1][0])
        return a[1] + (b[1] - a[1]) * (k - a[0]) / (b[0] - a[0])
    for m in NEW:
        assert agents._MATERIAL_ENERGY_FACTOR[m] == pytest.approx(want(M.props(m)["kc1_1"]), abs=0.006), m


def test_the_materials_come_in_a_sensible_order():
    vt = lambda m: agents._CUTTING_PARAMS[m][0]
    assert vt("Hardened Steel") < vt("Alloy Steel") < vt("Steel") < vt("Mild Steel")
    assert vt("Inconel") < vt("Hardened Steel") < vt("Pure Titanium") and vt("Inconel") == min(vt(m) for m in NEW)
    assert vt("Magnesium") == max(vt(m) for m in NEW) and vt("Magnesium") > vt("Aluminum")
    assert vt("Austenitic Stainless Steel") < vt("Steel") and vt("Nodular Cast Iron") < vt("Cast Iron")
    cf = agents._MATERIAL_COST_FACTOR
    assert cf["Inconel"] > cf["Titanium"] > cf["Steel"] > cf["Mild Steel"] > cf["Magnesium"]
    ef = agents._MATERIAL_ENERGY_FACTOR
    assert ef["Hardened Steel"] > ef["Inconel"] > ef["Tool Steel"] > ef["Steel"] > ef["Magnesium"]
    life = lambda m: M.props(m)["t_ref_min"]
    assert life("Inconel") < life("Hardened Steel") < life("Steel") and life("Magnesium") > life("Aluminum")


# ---------------------------------------------------------------- the handbook tables are complete for them
@pytest.mark.parametrize("m", NEW)
def test_process_tables_have_tools_and_a_range_for_each_new_material(m):
    for op in PT.OPERATIONS:
        t = PT.tool_for(op, m)
        assert t["type"] and t["material"], (op, m)
    for cls in ("turn", "drill", "mill"):
        r = PT.RANGES[(cls, m)]
        assert r["vc"][0] < r["vc"][1] and r["feed"][0] < r["feed"][1] and r["status"] in PT.STATUSES and r["status"] != "confirmed"
        if r["status"] in ("web", "mixed"):
            assert r["sources"] and all(s in PT.SOURCES for s in r["sources"])


def test_the_new_source_is_s5_and_the_copied_rows_say_which_old_row_they_are():
    assert "S5" in PT.SOURCES and PT.SOURCES["S5"][1].startswith("https://")
    for new, old in (("Mild Steel", "Steel"), ("Bronze", "Copper"), ("Hardened Aluminum Alloy", "Aluminum"), ("Pure Titanium", "Titanium")):
        for cls in ("turn", "drill", "mill"):
            assert PT.RANGES[(cls, new)]["vc"] == PT.RANGES[(cls, old)]["vc"] and PT.RANGES[(cls, new)]["note"].startswith(f"same row as {old}")
    assert all(PT.RANGES[("turn", m)]["sources"] == ["S5"] for m in ("Alloy Steel", "Tool Steel", "Austenitic Stainless Steel", "Inconel", "Nodular Cast Iron"))
    assert PT.RANGES[("turn", "Inconel")]["vc"] == (12, 45) and PT.RANGES[("turn", "Austenitic Stainless Steel")]["vc"] == (45, 135)


def test_the_v1_speed_of_each_new_material_sits_inside_its_turning_range_or_is_flagged_not_hidden():
    for m in NEW:
        lo, hi = PT.RANGES[("turn", m)]["vc"]
        assert 0.5 * lo <= agents._CUTTING_PARAMS[m][0] <= 2.5 * hi, m                     # a gross mismatch would mean a typing slip in one of the two tables


# ---------------------------------------------------------------- they run through the program
@pytest.mark.parametrize("m", NEW)
def test_the_v1_agents_and_the_physics_run_and_give_sane_numbers(m):
    t = agents.time_agent(ROUTE, m, PART)
    c = agents.cost_agent(ROUTE, m, 50, PART)
    e = agents.energy_agent(ROUTE, m, PART)
    assert t > 0 and c > 0 and e > 0
    a = physics.analyse_route(ROUTE, m, PART, 50, 1.6)
    assert a["material"] == m and a["ops"] and a["totals"]


def test_a_harder_material_takes_longer_and_costs_more_than_a_softer_one():
    t = lambda m: agents.time_agent(ROUTE, m, PART)
    assert t("Inconel") > t("Austenitic Stainless Steel") > t("Steel") > t("Mild Steel") > t("Aluminum") > t("Magnesium")
    assert agents.cost_agent(ROUTE, "Inconel", 50, PART) > agents.cost_agent(ROUTE, "Steel", 50, PART) > agents.cost_agent(ROUTE, "Magnesium", 50, PART)


@pytest.mark.parametrize("m", NEW)
def test_a_process_plan_and_the_calculator_run_for_each_new_material(m):
    part = {"part_id": "x", "material": m, "features": ["Hole", "Slot"], "tolerance": "0.02mm", "batch_size": 50, "dims": dict(PART), "required_ra_um": 1.6}
    plan = PP.build_plan(ROUTE, part)
    assert plan["part"]["material"] == m and plan["operations"]
    assert plan["totals"]["v1_time_min"] == pytest.approx(agents.time_agent(ROUTE, m, PART))          # the plan's time is the v1 Time agent's (report 6.7)
    res = CALC.evaluate(ROUTE, m, dict(PART, hole_diameter=8.0), CALC.Settings())
    assert res["time_min"] > 0 and res["total_cost"] > res["material_cost"] and res["energy_kwh"] > 0
    # the milling defaults of the calculator follow the material: Vc and fz are the v1 cutting data of that material
    d = CALC.milling_defaults("Slot Milling", m)
    assert d["cutting speed"] == agents._CUTTING_PARAMS[m][4] and d["feed per tooth"] == agents._CUTTING_PARAMS[m][5]


def test_planner_similarity_knows_the_new_classes():
    assert planners.MATERIAL_CLASS["Mild Steel"] == planners.MATERIAL_CLASS["Steel"] == "ferrous"
    assert planners.MATERIAL_CLASS["Pure Titanium"] == planners.MATERIAL_CLASS["Titanium"]
    assert planners.MATERIAL_CLASS["Inconel"] not in (planners.MATERIAL_CLASS[m] for m in OLD)


# ---------------------------------------------------------------- the page
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(ROOT, "week6", "app.py")


@pytest.fixture
def no_network(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")


def _page(material, route_only=False):
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["extraction_result"] = {"success": True, "confidence": "high", "features": ["Hole", "Turning"], "notes": "", "rejected_features": []}
    at.session_state["confirmed_features"] = ["Hole", "Turning"]
    at.run()
    if route_only:
        at.radio[0].set_value("\U0001f5fa️ Route Only").run()
    sel = [s for s in at.selectbox if s.label.endswith("Material")][0]
    assert list(sel.options) == list(MATERIALS)                                           # 19 options, old seven first, in the old order
    sel.set_value(material).run()
    [b for b in at.button if "Generate" in b.label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("m", ["Mild Steel", "Tool Steel", "Austenitic Stainless Steel", "Inconel", "Magnesium", "Pure Titanium"])
def test_the_page_runs_end_to_end_with_a_new_material(m, no_network):
    at = _page(m)
    assert any("STEP 5" in x.value for x in at.markdown if "step-hdr" in x.value)
    assert any("TOTAL COST" in x.label for x in at.metric)


def test_the_page_runs_route_only_with_a_new_material(no_network):
    at = _page("Hardened Steel", route_only=True)
    assert any("Feasibility Check" in m.value for m in at.markdown)
