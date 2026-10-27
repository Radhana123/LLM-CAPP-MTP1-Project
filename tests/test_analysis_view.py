import copy
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import analysis_view as AV     # noqa: E402
from capp import feasibility as F        # noqa: E402
from capp import legacy                  # noqa: E402,F401
from capp import physics                 # noqa: E402

TURN = "Plain/Cylindrical Turning"
DIMS = {"diameter": 40, "length": 120, "depth": 15, "width": 32}
ROUTE = ["Facing", TURN, "Center Drilling", "Drilling", "Reaming", "Finish Turning"]


def full(ra=1.6, hole=8, tol="0.05mm", dims=None, features=("Hole", "Turning"), batch=100, material="Steel"):
    return AV.make_part(material, features, tol, batch, dims=dict(DIMS if dims is None else dims), required_ra_um=ra, hole_diameter=hole)


def route_only(tol="0.05mm", features=("Hole",), material="Steel"):
    return AV.make_part(material, features, tol, 100, dims=None)


# ---------------------------------------------------------------- make_part
def test_make_part_full_and_route_only():
    p = full()
    assert p["dims"]["hole_diameter"] == 8.0 and p["dims"]["diameter"] == 40.0 and p["required_ra_um"] == 1.6
    assert p["batch_size"] == 100 and p["features"] == ["Hole", "Turning"]
    r = route_only()
    assert r["dims"] is None and r["required_ra_um"] is None


def test_make_part_zero_means_unknown():
    p = AV.make_part("Steel", ["Hole"], "0.05mm", 10, dims=dict(DIMS), required_ra_um=0, hole_diameter=0)
    assert p["required_ra_um"] is None and "hole_diameter" not in p["dims"]


def test_make_part_does_not_mutate_input_dims():
    d = dict(DIMS)
    AV.make_part("Steel", ["Hole"], "0.05mm", 10, dims=d, hole_diameter=8)
    assert d == DIMS


# ---------------------------------------------------------------- full analysis
def test_full_view_has_all_sections():
    v = AV.analyse_route_view(ROUTE, full())
    assert v["mode"] == "full" and v["errors"] == []
    assert set(v["physics"]) == {"cards", "ops", "sensitivity", "flags", "other_flags", "ra"}
    assert v["feasibility"]["n_rules"] == len(F.RULES) == 15
    assert v["assumptions"]


def test_cards_match_physics_model_exactly():
    p = full()
    v = AV.analyse_route_view(ROUTE, p)
    ref = physics.analyse_route(ROUTE, p["material"], p["dims"], p["batch_size"], p["required_ra_um"])["totals"]
    c = v["physics"]["cards"]
    assert c["energy_kwh"] == ref["cutting_energy_kwh"]
    assert c["wear_cost_inr"] == ref["wear_cost_inr"]
    assert c["peak_spindle_kw"] == ref["peak_spindle_kw"]
    assert c["ra_risk"] == ref["ra"]["risk"]
    assert c["wear_cost_batch_inr"] == pytest.approx(c["wear_cost_inr"] * p["batch_size"])


def test_op_rows_cover_every_operation_in_order():
    v = AV.analyse_route_view(ROUTE, full())
    assert [r["Operation"] for r in v["physics"]["ops"]] == ROUTE
    assert all("Energy (kWh)" in r and "Tool wear (₹)" in r for r in v["physics"]["ops"])


def test_other_flags_leave_out_what_the_rules_report_again():
    v = AV.analyse_route_view(["Facing", "Not An Operation"], full(ra=0.1))
    p = v["physics"]
    assert any(f.startswith("Ra:") for f in p["flags"])                         # the model flags the Ra problem ...
    assert not any(f.startswith(AV.FLAGS_COVERED_BY_RULES) for f in p["other_flags"])    # ... and the panel leaves it to RA-FEED / RA-FLOOR
    assert any("unknown operation" in f for f in p["other_flags"])              # but still says that an operation could not be modelled
    assert set(p["other_flags"]) <= set(p["flags"])


def test_op_energy_sums_to_card():
    v = AV.analyse_route_view(ROUTE, full())
    s = sum(r["Energy (kWh)"] for r in v["physics"]["ops"] if r["Energy (kWh)"] is not None)
    assert s == pytest.approx(v["physics"]["cards"]["energy_kwh"])


def test_marker_is_ignored():
    p = full()
    a = AV.analyse_route_view(ROUTE, p)
    b = AV.analyse_route_view(ROUTE[:3] + [F.V.MARKER] + ROUTE[3:], p)
    assert a["physics"]["cards"] == b["physics"]["cards"]
    assert [r["Operation"] for r in b["physics"]["ops"]] == ROUTE


def test_sensitivity_consistent():
    sens = AV.analyse_route_view(ROUTE, full())["physics"]["sensitivity"]
    assert [s["vc_factor"] for s in sens] == list(AV.SPEED_FACTORS)
    slow, base, fast = sens
    assert slow["cutting_time_min"] > base["cutting_time_min"] > fast["cutting_time_min"]
    assert slow["energy_kwh"] == pytest.approx(base["energy_kwh"]) == pytest.approx(fast["energy_kwh"])     # energy of the cut is speed-independent
    assert slow["wear_cost_inr"] < base["wear_cost_inr"] < fast["wear_cost_inr"]
    assert slow["peak_spindle_kw"] < base["peak_spindle_kw"] < fast["peak_spindle_kw"]


def test_base_row_of_sensitivity_equals_cards():
    v = AV.analyse_route_view(ROUTE, full())
    base = [s for s in v["physics"]["sensitivity"] if s["vc_factor"] == 1.0][0]
    c = v["physics"]["cards"]
    assert base["energy_kwh"] == c["energy_kwh"] and base["wear_cost_inr"] == c["wear_cost_inr"] and base["cutting_time_min"] == c["cutting_time_min"]


def test_ra_high_when_required_below_floor():
    v = AV.analyse_route_view(ROUTE, full(ra=0.1))
    assert v["physics"]["cards"]["ra_risk"] == "high"
    assert any(x["rule_id"] == "RA-FLOOR" for x in v["feasibility"]["risks"])


def test_ra_unknown_without_requirement():
    v = AV.analyse_route_view(ROUTE, full(ra=None))
    assert v["physics"]["cards"]["ra_risk"] == "unknown"
    assert dict(v["feasibility"]["not_evaluated"]).get("RA-FLOOR")


def test_hole_diameter_changes_what_is_evaluated():
    with_hole = AV.analyse_route_view(ROUTE, full(hole=8))
    without = AV.analyse_route_view(ROUTE, full(hole=None))
    assert "HOLE-DEEP" not in dict(with_hole["feasibility"]["not_evaluated"])
    assert "HOLE-DEEP" in dict(without["feasibility"]["not_evaluated"])
    assert any("hole diameter" in a for a in without["assumptions"])
    assert not any("No hole diameter" in a for a in with_hole["assumptions"])


def test_violations_carry_action_and_source_status():
    v = AV.analyse_route_view(["Facing", "Drilling"], full(hole=8))
    risks = v["feasibility"]["risks"]
    pilot = [x for x in risks if x["rule_id"] == "HOLE-PILOT"]
    assert pilot and pilot[0]["action"] == "insert:Center Drilling" and pilot[0]["source_status"] == "web"
    assert all(x["kind"] == "risk" for x in risks)


def test_hard_finding_goes_to_infeasible():
    v = AV.analyse_route_view(["Facing", "Reaming"], full())                                 # reaming with nothing drilled before
    assert [x["rule_id"] for x in v["feasibility"]["infeasible"]] == ["HOLE-EXISTS"]
    assert v["feasibility"]["infeasible"][0]["kind"] == "infeasible"


def test_part_level_split_from_route_level():
    v = AV.analyse_route_view(ROUTE, full(tol="0.005mm"))
    assert [x["rule_id"] for x in v["feasibility"]["part_level"]] == ["TOL-SPEC"]
    assert all(x["scope"] == "route" for x in v["feasibility"]["risks"])


def test_machine_envelope_is_an_input():
    big = dict(DIMS, diameter=400)
    assert AV.analyse_route_view(ROUTE, full(dims=big))["feasibility"]["infeasible"][0]["rule_id"] == "CAP-LATHE-DIA"
    m = F.Machine(lathe_max_diameter_mm=500.0)
    assert AV.analyse_route_view(ROUTE, full(dims=big), machine=m)["feasibility"]["infeasible"] == []


def test_machine_power_is_used_in_cards():
    c = AV.analyse_route_view(ROUTE, full(), machine=F.Machine(spindle_power_kw=3.0))["physics"]["cards"]
    assert c["machine_power_kw"] == 3.0


def test_wear_and_energy_scale_with_batch():
    a = AV.analyse_route_view(ROUTE, full(batch=10))["physics"]["cards"]
    b = AV.analyse_route_view(ROUTE, full(batch=1000))["physics"]["cards"]
    assert a["energy_kwh"] == b["energy_kwh"] and a["wear_cost_inr"] == pytest.approx(b["wear_cost_inr"])    # per part
    assert b["wear_cost_batch_inr"] == pytest.approx(100 * a["wear_cost_batch_inr"])


def test_input_route_not_mutated():
    r = list(ROUTE)
    AV.analyse_route_view(r, full())
    assert r == ROUTE


def test_deterministic():
    assert AV.analyse_route_view(ROUTE, full()) == AV.analyse_route_view(ROUTE, copy.deepcopy(full()))


# ---------------------------------------------------------------- Route Only
def test_route_only_has_no_physics():
    v = AV.analyse_route_view(["Facing", "Drilling"], route_only())
    assert v["mode"] == "route_only" and v["physics"] is None and v["errors"] == []


def test_route_only_runs_dimension_free_rules_and_lists_the_rest():
    v = AV.analyse_route_view(["Facing", TURN, "Drilling"], route_only())
    f = v["feasibility"]
    assert [x["rule_id"] for x in f["risks"]] == ["HOLE-PILOT"]
    ne = dict(f["not_evaluated"])
    assert {"CAP-LATHE-DIA", "CAP-POWER", "HOLE-DEEP"} <= set(ne)
    assert all("missing" in why for why in ne.values())
    assert f["evaluated"] + len(ne) == f["n_rules"]


def test_route_only_still_catches_hard_order_defect():
    v = AV.analyse_route_view(["Facing", "Reaming"], route_only())
    assert [x["rule_id"] for x in v["feasibility"]["infeasible"]] == ["HOLE-EXISTS"]


def test_route_only_assumptions_do_not_claim_a_hole_diameter_note():
    v = AV.analyse_route_view(["Facing", "Drilling"], route_only())
    assert not any("No hole diameter" in a for a in v["assumptions"])


# ---------------------------------------------------------------- robustness
def test_bad_route_does_not_raise():
    v = AV.analyse_route_view(["Not An Operation", "Drilling"], full())
    assert v["mode"] == "full"
    assert v["physics"] is not None


def test_empty_route_does_not_raise():
    v = AV.analyse_route_view([], full())
    assert v["errors"] == [] or all("failed" in e for e in v["errors"])
    v = AV.analyse_route_view([], route_only())
    assert v["physics"] is None


def test_rule_bug_is_reported_not_raised(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(F, "evaluate", boom)
    v = AV.analyse_route_view(ROUTE, full())
    assert v["feasibility"] is None and any("feasibility check failed" in e for e in v["errors"])
    assert v["physics"] is not None                                                          # one failing part does not hide the other


def test_physics_bug_is_reported_not_raised(monkeypatch):
    def boom(*a, **k):
        raise ValueError("bad")
    monkeypatch.setattr(AV, "physics_view", boom)                # the feasibility rules call the physics model themselves, so patch the view's own call
    v = AV.analyse_route_view(ROUTE, full())
    assert v["physics"] is None and any("physics analysis failed" in e for e in v["errors"])
    assert v["feasibility"] is not None


# ---------------------------------------------------------------- table rows and rules
def test_route_row_columns():
    row = AV.route_row(ROUTE, full())
    assert set(row) == {"Cutting energy (kWh)", "kc (N/mm²)", "Tool wear (₹)", "Ra risk", "Hard issues", "Risks"}
    assert row["Hard issues"] == 0 and isinstance(row["Risks"], int)


def test_route_row_counts_match_view():
    p = full(ra=0.1, tol="0.005mm")
    v = AV.analyse_route_view(ROUTE, p)
    row = AV.route_row(ROUTE, p)
    f = v["feasibility"]
    assert row["Hard issues"] == len(f["infeasible"])
    assert row["Risks"] == len(f["risks"]) + len(f["part_level"])
    assert row["Cutting energy (kWh)"] == round(v["physics"]["cards"]["energy_kwh"], 3)


def test_route_row_survives_failure(monkeypatch):
    monkeypatch.setattr(physics, "analyse_route", lambda *a, **k: 1 / 0)
    row = AV.route_row(ROUTE, full())
    assert row["Ra risk"] == "n/a" and row["Hard issues"] is None


def test_rules_table_complete_and_honest():
    rows = AV.rules_table()
    assert len(rows) == len(F.RULES) == 15
    assert {r["Status"] for r in rows} <= {"web", "derived", "model", "assumption"}
    assert all(r["Source"] and r["What it checks"] and r["Action"] for r in rows)
    assert not any(r["Status"] == "handbook" for r in rows)


def test_assumptions_mention_what_is_not_modelled():
    text = " ".join(AV.assumptions(full()))
    assert "lower bound" in text and "ASSUMED" in text and "not the v1" in text


def test_risk_icons_cover_all_levels():
    assert set(AV.RISK_ICON) == set(AV.RISK_TEXT) == set(AV.RISK_SHORT) == {"low", "medium", "high", "unknown"}


def test_inspection_has_no_cutting_time():
    rows = {r["Operation"]: r for r in AV.analyse_route_view(ROUTE + ["Inspection"], full())["physics"]["ops"]}
    assert rows["Inspection"]["Cutting time (min)"] is None and rows["Drilling"]["Cutting time (min)"] > 0
    total = sum(r["Cutting time (min)"] or 0 for r in rows.values())
    assert total == pytest.approx(AV.analyse_route_view(ROUTE + ["Inspection"], full())["physics"]["cards"]["cutting_time_min"])


def test_route_row_ra_risk_label_is_short_when_unknown():
    assert AV.route_row(ROUTE, full(ra=None))["Ra risk"] == "⚪ n/a"


def test_module_does_not_import_streamlit():
    src = open(os.path.join(ROOT, "capp", "analysis_view.py"), encoding="utf-8").read()
    assert "import streamlit" not in src


def test_view_is_json_serialisable():
    import json
    json.dumps(AV.analyse_route_view(ROUTE, full()))
    json.dumps(AV.analyse_route_view(ROUTE, route_only()))
