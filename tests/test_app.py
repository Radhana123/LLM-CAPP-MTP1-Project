"""Headless smoke tests of week6/app.py (Streamlit's AppTest): both modes run end to end without an exception and show what Step 12 added.

No API key and no network: the planner is switched to replay-only with an empty key, so the LLM route is skipped (as in the
no-key case of the app) and the Route Builder + NSGA-II routes are used.
"""
import os
import sys

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
APP = os.path.join(ROOT, "week6", "app.py")
FULL = "📊 Route + Full Analysis"
ROUTE_ONLY = "🗺️ Route Only"
GENERATE = "Generate"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")             # present but empty: load_dotenv() in the app does not override it
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")      # cache only; a miss is an error that the planner turns into "no LLM route"


def start(features=("Hole", "Turning"), mode=FULL, material=None):
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["extraction_result"] = {"success": True, "confidence": "high", "features": list(features), "notes": "",
                                             "rejected_features": []}
    at.session_state["confirmed_features"] = list(features)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if mode != FULL:
        at.radio[0].set_value(mode).run()
    if material:
        [s for s in at.selectbox if s.label.endswith("Material")][0].set_value(material).run()
    return at


def generate(at):
    [b for b in at.button if GENERATE in b.label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def headers(at):
    return [m.value for m in at.markdown if "step-hdr" in m.value]


def labels(at):
    return [m.label for m in at.metric]


def texts(elements):
    return [e.value for e in elements]


# ---------------------------------------------------------------- Route + Full Analysis
def test_full_analysis_runs_and_shows_step_5():
    at = generate(start())
    assert any("STEP 5" in h for h in headers(at))
    lab = labels(at)
    assert any("Cutting energy" in x for x in lab) and any("Tool wear" in x for x in lab) and any("Ra risk" in x for x in lab)
    assert any("Peak spindle" in x for x in lab) and any("Mean kc" in x for x in lab)


def test_agent_table_gets_the_physics_columns():
    at = generate(start())
    cols = at.dataframe[0].value.columns.tolist()
    assert cols[:7] == ["Route", "Machine Type", "Changeover", "Time (min)", "Cost (₹)", "Energy (kWh)", "Efficiency"]     # v1 columns first, unchanged
    assert cols[7:] == ["Cutting energy (kWh)", "kc (N/mm²)", "Tool wear (₹)", "Ra risk", "Hard issues", "Risks"]


def test_physics_columns_do_not_enter_the_efficiency_score():
    """Efficiency is still the v1 function of the three v1 numbers; the new columns only describe the routes."""
    from capp import legacy       # noqa: F401
    from agents import efficiency_agent
    df = generate(start()).dataframe[0].value
    for _, row in df.iterrows():
        assert row["Efficiency"] == efficiency_agent(row["Time (min)"], row["Cost (₹)"], row["Energy (kWh)"])


def test_required_ra_drives_the_ra_panel():
    at = start()
    ra = [s for s in at.selectbox if "Ra" in s.label][0]
    ra.set_value(0.4).run()
    generate(at)
    metric = [m for m in at.metric if "Ra risk" in m.label][0]
    assert "high" in metric.value
    assert any("RA-FEED" in w or "RA-FLOOR" in w for w in texts(at.warning))


def test_no_ra_means_not_assessed():
    at = generate(start())
    metric = [m for m in at.metric if "Ra risk" in m.label][0]
    assert "n/a" in metric.value and "no required Ra" in metric.delta


def test_hole_diameter_enables_the_deep_hole_check():
    at = start()
    [n for n in at.number_input if "Hole diameter" in n.label][0].set_value(5.0).run()
    [n for n in at.number_input if n.label == "Depth (mm)"][0].set_value(40.0).run()
    generate(at)
    assert any("HOLE-DEEP" in w for w in texts(at.warning))


def test_assumptions_and_sources_are_on_the_page():
    at = generate(start())
    labs = [e.label for e in at.expander]
    assert any("Assumptions" in x for x in labs) and any("15 feasibility rules" in x for x in labs)


def test_analysis_only_caption_is_shown():
    at = generate(start())
    assert any("does not change which route is chosen" in c.value or "do not change which route is chosen" in c.value for c in at.caption)


def test_every_material_runs():
    for mat in ("Aluminum", "Steel", "Brass", "Copper", "Titanium", "Plastic", "Cast Iron"):
        at = start(material=mat)
        generate(at)
        assert any("STEP 5" in h for h in headers(at)), mat


def test_several_feature_sets_run():
    for feats in (("Hole",), ("Turning",), ("Slot", "Pocket"), ("Hole", "Thread", "Chamfer"), ("Reamed_Hole", "Bore", "Face")):
        at = generate(start(features=feats))
        assert any("STEP 5" in h for h in headers(at)), feats


def test_step_5_adds_no_input_widget():
    """A widget inside the results would make Streamlit rerun and wipe them (the reason the speed table is not a slider)."""
    at = start()
    before = (len(at.selectbox), len(at.slider), len(at.radio), len(at.number_input), len(at.checkbox), len(at.multiselect))
    generate(at)
    after = (len(at.selectbox), len(at.slider), len(at.radio), len(at.number_input), len(at.checkbox), len(at.multiselect))
    assert before == after


# ---------------------------------------------------------------- Route Only
def test_route_only_runs_and_shows_feasibility_not_physics():
    at = generate(start(mode=ROUTE_ONLY))
    assert not any("STEP 5" in h for h in headers(at))
    assert any("Feasibility Check" in m.value for m in at.markdown)
    lab = labels(at)
    assert not any("Cutting energy" in x for x in lab) and not any("Ra risk" in x for x in lab)
    assert any("Without part dimensions" in c.value for c in at.caption)


def test_route_only_has_no_ra_or_hole_inputs():
    at = start(mode=ROUTE_ONLY)
    assert not any("Ra" in s.label for s in at.selectbox)
    assert not any("Hole diameter" in n.label for n in at.number_input)


def test_route_only_lists_rules_that_need_dimensions():
    at = generate(start(features=("Turning", "Hole"), mode=ROUTE_ONLY))
    labs = [e.label for e in at.expander]
    assert any("not evaluated" in x for x in labs)
    assert any("feasibility rules" in x for x in labs)


def test_route_only_every_material_runs():
    for mat in ("Aluminum", "Steel", "Titanium"):
        at = start(mode=ROUTE_ONLY, material=mat)
        generate(at)
        assert any("Feasibility Check" in m.value for m in at.markdown), mat


# ---------------------------------------------------------------- unchanged behaviour
def test_no_features_still_asks_for_one():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    [b for b in at.button if GENERATE in b.label][0].click().run()
    assert not at.exception
    assert any("select at least one feature" in e.value for e in at.error)


def test_start_page_renders_without_a_run():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception and not at.error
