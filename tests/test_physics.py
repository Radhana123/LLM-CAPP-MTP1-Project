import copy
import json
import math
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import dataset as D          # noqa: E402
from capp import harness as H          # noqa: E402
from capp import legacy                # noqa: E402,F401
from capp import llm_client            # noqa: E402
from capp import materials as MAT      # noqa: E402
from capp import physics as P          # noqa: E402
from capp import summary as SM         # noqa: E402
from capp import systems               # noqa: E402
from capp import verify as V           # noqa: E402
import agents                          # noqa: E402

DIMS = {"diameter": 50, "length": 100, "depth": 15, "width": 40}
TURN = "Plain/Cylindrical Turning"
MATERIALS = sorted(MAT.MATERIAL_PROPS)


@pytest.fixture(scope="module")
def parts():
    return D.load_dataset("dev_v2")


def rb_route(part):
    """The Route Builder's first route for a part, as the RB baseline makes it (machine preference included)."""
    p = systems.part_inputs(part)
    routes, _ = systems._rb_routes(p["features"], p["pref"], 15)
    return V.clean_steps(routes[0]["steps"])


# ---- the physics against hand calculations (formulas written out here, not imported) -------------------------------------
def test_turning_steel_hand_calculation():
    r = P.analyse_op(TURN, "Steel", DIMS)
    vc, f = 120.0, 0.15                                   # agents._CUTTING_PARAMS["Steel"] turning
    passes = agents._num_passes(TURN, 15)
    ap = 15 / passes
    q = vc * 1000 * f * ap                                # mm3/min
    kc = 1500 * f ** -0.22
    assert r["ap_mm"] == pytest.approx(ap) and passes == 5
    assert r["mrr_cm3_min"] == pytest.approx(q / 1000) == pytest.approx(54.0)
    assert r["kc_n_mm2"] == pytest.approx(kc) and kc == pytest.approx(2276.9, abs=0.1)
    assert r["power_kw"] == pytest.approx(kc * q / 60e6) == pytest.approx(2.0492, abs=1e-3)       # 1 N/mm2 = 1 mJ/mm3
    assert r["spindle_kw"] == pytest.approx(r["power_kw"] / 0.8)
    assert r["utilisation"] == pytest.approx(r["spindle_kw"] / 7.5)
    assert r["t_cut_min"] == pytest.approx(agents._op_time(TURN, "Steel", P._dims(DIMS)))
    assert r["energy_kwh"] == pytest.approx(r["power_kw"] * r["t_cut_min"] / 60)
    assert r["ra_um"] == pytest.approx(1000 * f ** 2 / (32 * 0.8))                                # 0.879 um


def test_specific_energy_of_steel_is_about_two_joules_per_mm3():
    a = P.analyse_route(["Facing", TURN, "Step Turning"], "Steel", DIMS)
    assert 1.5 < a["totals"]["specific_energy_j_mm3"] < 3.5


def test_drilling_uses_the_drilling_mrr_formula_and_the_point_angle():
    r = P.analyse_op("Drilling", "Steel", DIMS)                       # the part diameter is the hole diameter (as the Time agent)
    assert r["mrr_cm3_min"] == pytest.approx(250 * 50 * 0.1 * 40 / 1000)
    assert r["chip_mm"] == pytest.approx(0.1 / 2 * math.sin(math.radians(59)))
    assert r["ra_um"] == P.RA_FLOOR_UM["drill"]
    small = P.analyse_op("Drilling", "Steel", dict(DIMS, hole_diameter=10))
    assert small["mrr_cm3_min"] == pytest.approx(250 * 10 * 0.1 * 40 / 1000)
    assert small["t_cut_min"] < r["t_cut_min"] and small["power_kw"] < r["power_kw"]
    assert small["t_cut_min"] == pytest.approx(agents._op_time("Drilling", "Steel", dict(P._dims(DIMS), diameter=10)))


def test_hole_diameter_does_not_change_non_hole_operations():
    a = P.analyse_op(TURN, "Steel", DIMS)
    b = P.analyse_op(TURN, "Steel", dict(DIMS, hole_diameter=5))
    assert a == b


def test_milling_power_from_engagement_and_table_feed():
    r = P.analyse_op("Slot Milling", "Aluminum", DIMS)
    vc, fz = 250.0, 0.1                                               # Aluminum milling
    dc, z = agents._CUTTER_DIA["Slot Milling"], agents._CUTTER_TEETH["Slot Milling"]
    ae = 1.0 * dc
    n = vc * 1000 / (math.pi * dc)
    vf = fz * z * n
    ap = 15 / agents._num_passes("Slot Milling", 15)
    assert r["ae_mm"] == pytest.approx(ae)
    assert r["mrr_cm3_min"] == pytest.approx(ae * ap * vf / 1000)
    assert r["chip_mm"] == pytest.approx(fz * math.sqrt(1.0))
    assert r["ra_um"] >= P.RA_FLOOR_UM["mill"]


def test_thread_removal_rate_is_thread_volume_over_time():
    r = P.analyse_op("Tapping", "Steel", DIMS)
    pitch = agents._THREAD_PITCH["Tapping"]
    volume = 0.325 * math.pi * 50 * 100 * pitch
    assert r["mrr_cm3_min"] * 1000 * r["t_cut_min"] == pytest.approx(volume)


# ---- monotonic behaviour -------------------------------------------------------------------------------------------------
def test_kc_ordering_follows_the_material_table():
    kc = {m: P.analyse_op(TURN, m, DIMS)["kc_n_mm2"] for m in MATERIALS}
    assert kc["Titanium"] > kc["Steel"] > kc["Cast Iron"] > kc["Aluminum"] > kc["Plastic"]
    assert kc["Brass"] == pytest.approx(kc["Copper"])                       # same table row, same feed


def test_deeper_cut_needs_more_power_and_energy():
    shallow = P.analyse_op("Facing", "Steel", dict(DIMS, depth=2))
    deep = P.analyse_op("Facing", "Steel", dict(DIMS, depth=20))
    assert deep["power_kw"] >= shallow["power_kw"]
    assert deep["energy_kwh"] > shallow["energy_kwh"]


def test_higher_cutting_speed_shortens_tool_life_and_the_cut_by_the_taylor_law():
    base = P.analyse_op(TURN, "Steel", DIMS)
    fast = P.analyse_op(TURN, "Steel", DIMS, vc_factor=1.2)
    n = MAT.props("Steel")["taylor_n"]
    assert base["tool_life_min"] == pytest.approx(MAT.props("Steel")["t_ref_min"])
    assert fast["tool_life_min"] == pytest.approx(base["tool_life_min"] * 1.2 ** (-1 / n))
    assert fast["vc_m_min"] == pytest.approx(base["vc_m_min"] * 1.2)
    assert fast["t_cut_min"] == pytest.approx(base["t_cut_min"] / 1.2)                  # same feed and depth: a shorter cut
    assert fast["power_kw"] == pytest.approx(base["power_kw"] * 1.2)                    # MRR scales with the speed
    assert fast["energy_kwh"] == pytest.approx(base["energy_kwh"])                      # power x time: the same volume, the same energy
    assert fast["wear_fraction"] == pytest.approx(base["wear_fraction"] * 1.2 ** (1 / n - 1))
    assert fast["wear_cost_inr"] > base["wear_cost_inr"]
    slow = P.analyse_op(TURN, "Steel", DIMS, vc_factor=0.8)
    assert slow["wear_cost_inr"] < base["wear_cost_inr"] and slow["t_cut_min"] > base["t_cut_min"]
    assert slow["energy_kwh"] == pytest.approx(base["energy_kwh"])


def test_speed_factor_applies_to_every_cutting_process_and_not_to_forming():
    for op in ("Facing", "Drilling", "Center Drilling", "Boring", "Reaming", "Tapping", "Slot Milling", "Chamfering", "Parting-off"):
        b, f = P.analyse_op(op, "Steel", DIMS), P.analyse_op(op, "Steel", DIMS, vc_factor=1.25)
        assert f["t_cut_min"] == pytest.approx(b["t_cut_min"] / 1.25), op
        assert f["energy_kwh"] == pytest.approx(b["energy_kwh"]), op
        assert f["power_kw"] == pytest.approx(b["power_kw"] * 1.25), op
    b, f = P.analyse_op("Knurling", "Steel", DIMS), P.analyse_op("Knurling", "Steel", DIMS, vc_factor=1.25)
    assert b == f


def test_wear_cost_is_cutting_time_over_tool_life_times_edge_cost():
    r = P.analyse_op(TURN, "Titanium", DIMS)
    p = MAT.props("Titanium")
    assert r["wear_fraction"] == pytest.approx(r["t_cut_min"] / p["t_ref_min"])
    assert r["wear_cost_inr"] == pytest.approx(r["wear_fraction"] * p["tool_edge_cost_inr"])
    longer = P.analyse_op(TURN, "Titanium", dict(DIMS, length=200))
    assert longer["wear_cost_inr"] == pytest.approx(r["wear_cost_inr"] * longer["t_cut_min"] / r["t_cut_min"])


def test_titanium_wears_tools_faster_than_aluminium():
    t = P.analyse_route([TURN], "Titanium", DIMS)["totals"]["wear_cost_inr"]
    a = P.analyse_route([TURN], "Aluminum", DIMS)["totals"]["wear_cost_inr"]
    assert t > 5 * a


def test_ra_grows_with_the_square_of_the_feed_and_is_floored(monkeypatch):
    monkeypatch.setitem(agents._CUTTING_PARAMS, "Steel", (120, 0.30, 40, 0.1, 80, 0.06))
    coarse = P.analyse_op(TURN, "Steel", DIMS)["ra_um"]
    monkeypatch.setitem(agents._CUTTING_PARAMS, "Steel", (120, 0.15, 40, 0.1, 80, 0.06))
    fine = P.analyse_op(TURN, "Steel", DIMS)["ra_um"]
    assert coarse == pytest.approx(4 * fine)
    monkeypatch.setitem(agents._CUTTING_PARAMS, "Steel", (120, 0.02, 40, 0.1, 80, 0.06))
    r = P.analyse_op(TURN, "Steel", DIMS)
    assert r["ra_um"] == P.RA_FLOOR_UM["turning"] and "floored" in r["ra_model"]


@pytest.mark.parametrize("best,required,label", [(0.4, 1.6, "low"), (0.8, 1.6, "low"), (0.81, 1.6, "medium"), (1.6, 1.6, "medium"),
                                                 (1.61, 1.6, "high"), (3.2, 0.8, "high")])
def test_ra_risk_thresholds(best, required, label):
    ratio, got = P.ra_risk(best, required)
    assert got == label and ratio == pytest.approx(best / required)


def test_ra_risk_unknown_when_a_number_is_missing():
    assert P.ra_risk(None, 1.6) == (None, "unknown")
    assert P.ra_risk(0.8, None) == (None, "unknown")
    assert P.ra_risk(0.8, 0) == (None, "unknown")


def test_max_feed_for_ra_inverts_the_turning_formula():
    for req in (0.8, 1.6, 3.2):
        f = P.max_feed_for_ra(req)
        assert 1000 * f ** 2 / (32 * P.NOSE_RADIUS_MM["turning"]) == pytest.approx(req * P.RA_MEDIUM_RATIO)
    assert P.max_feed_for_ra(3.2) > P.max_feed_for_ra(0.8)
    assert P.max_feed_for_ra(1.6, nose_radius_mm=1.2) > P.max_feed_for_ra(1.6, nose_radius_mm=0.4)


# ---- classification and edge cases ----------------------------------------------------------------------------------------
def test_every_operation_of_the_route_builder_vocabulary_is_classified():
    for op in V.CANONICAL_OPS:
        assert P.op_process(op) != "unknown", op


def test_forming_and_inspection_have_time_but_no_energy_or_wear():
    for op in ("Knurling", "Forming", "Polishing/Burnishing"):
        r = P.analyse_op(op, "Steel", DIMS)
        assert r["process"] == "forming" and r["t_cut_min"] > 0
        assert r["energy_kwh"] is None and r["wear_cost_inr"] is None and r["ra_um"] is None and r["notes"]
    r = P.analyse_op("Inspection", "Steel", DIMS)
    assert r["process"] == "none" and r["energy_kwh"] is None


def test_unknown_operation_is_flagged_not_guessed():
    a = P.analyse_route(["Facing", "Telepathy", "Inspection"], "Steel", DIMS)
    assert a["ops"][1]["process"] == "unknown" and a["ops"][1]["energy_kwh"] is None
    assert any("Telepathy" in f for f in a["totals"]["flags"])
    assert a["totals"]["modelled_operations"] == 1


def test_unknown_material_and_bad_factor_raise():
    with pytest.raises(KeyError):
        P.analyse_op(TURN, "Unobtainium", DIMS)
    with pytest.raises(KeyError):
        P.analyse_route([TURN], "Unobtainium", DIMS)
    with pytest.raises(ValueError):
        P.analyse_op(TURN, "Steel", DIMS, vc_factor=0)


def test_changeover_markers_are_ignored():
    steps = ["Facing", TURN, "Drilling", "Inspection"]
    with_marker = ["Facing", TURN, P.MARKER, "Drilling", "Inspection"]
    a, b = P.analyse_route(steps, "Steel", DIMS), P.analyse_route(with_marker, "Steel", DIMS)
    assert a["totals"] == b["totals"] and len(b["ops"]) == len(a["ops"]) == 4


def test_missing_dims_use_the_time_agents_defaults():
    a = P.analyse_route([TURN], "Steel", None)
    b = P.analyse_route([TURN], "Steel", {"diameter": 25, "length": 50, "depth": 20})
    assert a["totals"]["cutting_time_min"] == pytest.approx(b["totals"]["cutting_time_min"])
    assert a["dims"]["width"] == pytest.approx(20.0)


def test_empty_route_has_zero_totals():
    t = P.analyse_route([], "Steel", DIMS)["totals"]
    assert t["cutting_energy_kwh"] == 0 and t["wear_cost_inr"] == 0 and t["kc_mean_n_mm2"] is None and t["ra"]["risk"] == "unknown"


def test_cutting_time_agrees_with_the_time_agent_per_operation(parts):
    for part in parts[:25]:
        route = rb_route(part)
        a = P.analyse_part_route(route, part)
        for row in a["ops"]:
            dims = P._dims(part["dims"])
            if row["op"] in P.HOLE_DIAMETER_OPS:
                dims = dict(dims, diameter=dims.get("hole_diameter") or dims["diameter"])
            assert row["t_cut_min"] == pytest.approx(agents._op_time(row["op"], part["material"], dims)), (part["part_id"], row["op"])


def test_power_flag_when_the_machine_is_too_small():
    big = P.analyse_route(["Drilling"], "Steel", dict(DIMS, diameter=200), machine_power_kw=7.5)["totals"]
    assert big["ops_over_power"] == ["Drilling"] and any("Drilling" in f for f in big["flags"])
    ok = P.analyse_route(["Drilling"], "Steel", dict(DIMS, diameter=200), machine_power_kw=100)["totals"]
    assert ok["ops_over_power"] == [] and ok["max_utilisation"] < 1


def test_totals_are_the_sums_of_the_rows():
    a = P.analyse_route(["Facing", TURN, "Drilling", "Slot Milling", "Knurling", "Inspection"], "Steel", DIMS, batch_size=40)
    rows = [r for r in a["ops"] if r["energy_kwh"] is not None]
    t = a["totals"]
    assert t["cutting_energy_kwh"] == pytest.approx(sum(r["energy_kwh"] for r in rows))
    assert t["wear_cost_inr"] == pytest.approx(sum(r["wear_cost_inr"] for r in rows))
    assert t["edges_per_batch"] == pytest.approx(40 * t["edges_per_part"])
    assert t["wear_cost_batch_inr"] == pytest.approx(40 * t["wear_cost_inr"])
    assert t["peak_power_kw"] == pytest.approx(max(r["power_kw"] for r in rows))
    vol = sum(r["mrr_cm3_min"] * r["t_cut_min"] for r in rows)
    assert t["removed_volume_cm3"] == pytest.approx(vol)
    assert t["kc_mean_n_mm2"] == pytest.approx(sum(r["kc_n_mm2"] * r["mrr_cm3_min"] * r["t_cut_min"] for r in rows) / vol)
    assert t["modelled_operations"] == 4 and t["operations"] == 6


def test_ra_summary_uses_the_finest_process_on_the_route():
    a = P.analyse_route(["Facing", "Drilling", "Reaming"], "Steel", DIMS, required_ra_um=1.6)["totals"]["ra"]
    rows = P.analyse_route(["Facing", "Drilling", "Reaming"], "Steel", DIMS)["ops"]
    assert a["best_um"] == pytest.approx(min(r["ra_um"] for r in rows if r["ra_um"] is not None))
    assert a["ratio"] == pytest.approx(a["best_um"] / 1.6) and a["risk"] in ("low", "medium")
    assert a["max_turning_feed_mm_rev"] == pytest.approx(P.max_feed_for_ra(1.6))
    no_req = P.analyse_route(["Facing"], "Steel", DIMS)["totals"]["ra"]
    assert no_req["risk"] == "unknown" and no_req["ratio"] is None and no_req["max_turning_feed_mm_rev"] is None


def test_a_rough_only_route_is_high_risk_for_a_fine_finish():
    a = P.analyse_route(["Drilling", "Tapping"], "Steel", DIMS, required_ra_um=0.4)["totals"]["ra"]
    assert a["risk"] == "high" and a["best_op"] == "Drilling"


def test_deterministic_and_json_serialisable(parts):
    part = parts[3]
    route = rb_route(part)
    a = P.analyse_part_route(route, part)
    b = P.analyse_part_route(route, part)
    assert a == b
    assert json.loads(json.dumps(a)) == a
    c = P.condense(a)
    assert json.loads(json.dumps(c)) == c and c["totals"] == a["totals"] and len(c["ops"]) == len(a["ops"])


def test_analysis_does_not_mutate_its_inputs():
    steps, dims = ["Facing", TURN, "Inspection"], dict(DIMS)
    s0, d0 = list(steps), copy.deepcopy(dims)
    P.analyse_route(steps, "Steel", dims)
    assert steps == s0 and dims == d0


def test_assumptions_are_listed():
    for key in ("SPINDLE_POWER_KW", "RA_FLOOR_UM", "tool diameter of hole operations"):
        assert key in P.ASSUMPTIONS


def test_physics_does_not_change_the_v1_agents():
    before = agents._op_time(TURN, "Steel", P._dims(DIMS))
    P.analyse_route(["Facing", TURN, "Drilling"], "Steel", DIMS, vc_factor=1.2)
    assert agents._op_time(TURN, "Steel", P._dims(DIMS)) == before
    assert agents._CUTTING_PARAMS["Steel"][:2] == (120, 0.15)


def test_every_dev_part_and_material_runs_through_the_route_builder_route(parts):
    seen = set()
    for part in parts:
        route = rb_route(part)
        t = P.analyse_part_route(route, part)["totals"]
        seen.add(part["material"])
        assert t["cutting_energy_kwh"] > 0 and t["wear_cost_inr"] > 0 and t["kc_mean_n_mm2"] > 0
        assert not any("unknown" in f for f in t["flags"])
        assert t["ra"]["risk"] in ("low", "medium", "high", "unknown")
    assert seen == set(MATERIALS)


# ---- harness, summary, compare ---------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clean_client(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CAPP_LLM_MODE", raising=False)
    monkeypatch.delenv("CAPP_RUN_TAG", raising=False)
    llm_client.reset_transport()
    llm_client.reset_call_log()
    yield
    llm_client.reset_transport()


def fake_planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
    routes, _ = systems._rb_routes(list(features), "auto", 15)
    base = V.clean_steps(routes[0]["steps"])
    steps = base if sample_id % 2 == 0 else list(reversed(base))
    meta = {"cached": False, "finish_reason": "stop", "prompt_tokens": 300, "completion_tokens": 40, "latency_s": 0.4}
    return {"success": True, "steps": steps, "raw_response": json.dumps(steps), "model": "fake", "meta": meta}


def test_harness_stores_physics_with_every_final_route(parts, tmp_path):
    out = H.run_experiment("rb", parts[:12], tag="t", out_dir=str(tmp_path), dataset_name="dev_v2", progress=None)
    recs = [json.loads(l) for l in (tmp_path / "t_rb" / "records.jsonl").read_text().splitlines()]
    assert len(recs) == 12
    for rec in recs:
        ph = rec["final"]["physics"]
        assert set(ph) == {"totals", "ops"} and ph["totals"]["cutting_energy_kwh"] > 0
        assert len(ph["ops"]) == len([s for s in rec["final"]["steps"] if s != P.MARKER])
        expected = P.analyse_part_route(rec["final"]["steps"], rec["part"])["totals"]
        assert ph["totals"] == json.loads(json.dumps(expected))
    run = json.loads((tmp_path / "t_rb" / "run.json").read_text())
    assert run["physics_analysis"] is True
    s = out["summary"]["physics"]
    assert s["n"] == 12 and s["energy_kwh_mean"] > 0 and 0 <= s["over_power_share"] <= 1
    assert sum(s["ra_risk_share"].values()) == pytest.approx(1.0)
    assert s["energy_kwh_mean"] == pytest.approx(sum(r["final"]["physics"]["totals"]["cutting_energy_kwh"] for r in recs) / 12)


def test_per_part_csv_has_physics_columns(parts, tmp_path):
    H.run_experiment("rb", parts[:6], tag="t", out_dir=str(tmp_path), dataset_name="dev_v2", progress=None)
    import csv
    rows = list(csv.DictReader(open(tmp_path / "t_rb" / "per_part.csv", encoding="utf-8")))
    assert len(rows) == 6
    for col in ("phys_energy_kwh", "phys_kc", "phys_wear_inr", "phys_peak_spindle_kw", "over_power", "ra_best_um", "ra_risk"):
        assert col in rows[0], col
    assert all(float(r["phys_energy_kwh"]) > 0 and r["ra_risk"] for r in rows)


def test_no_analysis_leaves_physics_out_everywhere(parts, tmp_path):
    out = H.run_experiment("rb", parts[:6], tag="t", out_dir=str(tmp_path), dataset_name="dev_v2", progress=None, analysis=False)
    recs = [json.loads(l) for l in (tmp_path / "t_rb" / "records.jsonl").read_text().splitlines()]
    assert all("physics" not in r["final"] for r in recs)
    assert out["summary"]["physics"] == {"n": 0}
    assert "physics         :" not in SM.format_summary("rb", out["summary"])
    run = json.loads((tmp_path / "t_rb" / "run.json").read_text())
    assert run["physics_analysis"] is False
    import csv
    rows = list(csv.DictReader(open(tmp_path / "t_rb" / "per_part.csv", encoding="utf-8")))
    assert all(r["phys_energy_kwh"] == "" and r["ra_risk"] == "" for r in rows)


def test_analysis_does_not_change_routes_or_scores(parts, tmp_path):
    a = H.run_experiment("n3_r1_s3", parts[:8], tag="a", out_dir=str(tmp_path / "a"), planner=fake_planner, progress=None)
    b = H.run_experiment("n3_r1_s3", parts[:8], tag="a", out_dir=str(tmp_path / "b"), planner=fake_planner, progress=None, analysis=False)
    ra = [json.loads(l) for l in (tmp_path / "a" / "a_n3_r1_s3" / "records.jsonl").read_text().splitlines()]
    rb_ = [json.loads(l) for l in (tmp_path / "b" / "a_n3_r1_s3" / "records.jsonl").read_text().splitlines()]
    for x, y in zip(ra, rb_):
        x["final"].pop("physics")
        x["final"].pop("feasibility")
        for c in x["candidates"]:
            c.pop("raw_feasibility", None)
        assert x["final"] == y["final"] and x["info"] == y["info"] and x["candidates"] == y["candidates"]
    for k in ("M1", "M2", "M3", "M4"):
        assert a["summary"][k] == b["summary"][k]


def test_physics_error_is_recorded_not_fatal(parts, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model broke")
    monkeypatch.setattr(P, "analyse_part_route", boom)
    out = H.run_experiment("rb", parts[:4], tag="t", out_dir=str(tmp_path), dataset_name="dev_v2", progress=None)
    recs = [json.loads(l) for l in (tmp_path / "t_rb" / "records.jsonl").read_text().splitlines()]
    assert len(recs) == 4 and all("RuntimeError" in r["final"]["physics"]["error"] for r in recs)
    assert out["summary"]["physics"] == {"n": 0}
    assert out["summary"]["M2"]["final_strict_valid"] == 1.0


def test_records_without_physics_still_summarise():
    assert SM._physics_block([{"final": {"scores_eval": {"energy_kwh": 1, "cost_inr": 1}}}]) == {"n": 0}
    assert SM._physics_block([]) == {"n": 0}
    assert SM._physics_columns(None)["ra_risk"] == "" and SM._physics_columns({"error": "x"})["phys_kc"] == ""


def test_summary_text_has_one_physics_line(parts, tmp_path):
    out = H.run_experiment("rb", parts[:6], tag="t", out_dir=str(tmp_path), dataset_name="dev_v2", progress=None)
    text = SM.format_summary("rb", out["summary"])
    lines = [l for l in text.splitlines() if l.startswith("physics")]
    assert len(lines) == 1 and "cutting energy" in lines[0] and "Ra risk" in lines[0] and "kWh/part" in lines[0]


def test_summary_survives_a_json_round_trip(parts, tmp_path):
    out = H.run_experiment("rb", parts[:5], tag="t", out_dir=str(tmp_path), dataset_name="dev_v2", progress=None)
    again = json.loads((tmp_path / "t_rb" / "summary.json").read_text())
    assert again["physics"] == json.loads(json.dumps(out["summary"]["physics"]))
    assert SM.format_summary("rb", again) == SM.format_summary("rb", out["summary"])


def test_compare_shows_physics_rows(parts, tmp_path):
    from capp import compare as C
    H.run_experiment("rb", parts[:6], tag="t", out_dir=str(tmp_path), dataset_name="dev_v2", progress=None)
    H.run_experiment("n3_r1_s3", parts[:6], tag="t", out_dir=str(tmp_path), planner=fake_planner, progress=None)
    dirs = sorted(str(p) for p in tmp_path.iterdir() if p.is_dir())
    text = C.compare(dirs)
    for label in ("physics: cutting energy kWh/part", "physics: tool wear INR/part", "physics: parts over machine power",
                  "physics: Ra risk high share"):
        assert label in text, label


def test_cli_no_analysis_flag(tmp_path):
    env = dict(os.environ, CAPP_CACHE_DIR=str(tmp_path / "cache"))
    base = [sys.executable, os.path.join(ROOT, "scripts", "run_eval.py"), "--system", "rb", "--dataset", "dev_v2", "--limit", "5",
            "--out", str(tmp_path / "o")]
    r = subprocess.run(base + ["--tag", "x"], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "physics         :" in r.stdout
    r = subprocess.run(base + ["--tag", "y", "--no-analysis"], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "physics         :" not in r.stdout


def test_physics_check_script_runs_and_reports(tmp_path):
    js = tmp_path / "pc.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "physics_check.py"), "--dataset", "dev_v2", "--limit", "12",
                        "--json", str(js)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    for word in ("coverage", "by material", "rank agreement", "machine power", "Ra risk", "speed what-if"):
        assert word in r.stdout, word
    rep = json.loads(js.read_text())
    assert rep["parts"] == 12 and rep["coverage"]["unknown"] == []
    sens = rep["speed_sensitivity"]
    assert sens["0.8"]["wear_cost_inr"] < sens["1.0"]["wear_cost_inr"] < sens["1.2"]["wear_cost_inr"]
