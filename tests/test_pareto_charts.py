"""Step 57: the Pareto front graphs (3D and 2D, Step 48) are not in the app any more, in either mode. This file keeps them out (it was the test of the
charts; the name stays so that a zip of the next step does not leave a dead test file behind).

What stays: the NSGA-II result line ("Found N Pareto-optimal route(s)"), the table of the routes of the search and the list of all valid routes.
Headless (Streamlit's AppTest), no key, no network: the planner is replay-only with an empty key, so the routes are the Route Builder's.
"""
import os
import sys

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
APP = os.path.join(ROOT, "week6", "app.py")
ROUTE_ONLY = "\U0001f5fa️ Route Only"
FEATURES = ["Groove", "Hole", "Turning"]


def run(route_only=False, v1=False):
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["extraction_result"] = {"success": True, "confidence": "high", "features": FEATURES, "notes": "", "rejected_features": []}
    at.session_state["confirmed_features"] = FEATURES
    at.run()
    if route_only:
        at.radio[0].set_value(ROUTE_ONLY).run()
    elif v1:
        at.radio(key="time_model").set_value("\U0001F4C8 v1 model (numbers of the report)").run()
    [b for b in at.button if "Generate" in b.label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.fixture(scope="module")
def pages():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GROQ_API_KEY", "")
        mp.setenv("CAPP_LLM_MODE", "replay")
        return {"ro": run(route_only=True), "calc": run(), "v1": run(v1=True)}


def test_there_is_no_pareto_front_graph_in_either_mode(pages):
    for mode in ("ro", "calc", "v1"):
        at = pages[mode]
        assert not [e for e in at.get("plotly_chart") if "pareto" in str(e.proto.id).lower()], mode
        assert not any("Pareto Front" in m.value for m in at.markdown), mode
        assert not any("Pareto chart" in i.value for i in at.info), mode
        assert not any("Pareto Front" in c.value for c in at.caption), mode


def test_the_pareto_optimal_routes_of_the_search_are_still_reported_and_listed(pages):
    """What is taken out is the graph. The NSGA-II result line, the table of the search's routes and the list of all valid routes stay."""
    for mode in ("ro", "calc", "v1"):
        at = pages[mode]
        assert any("Pareto-optimal route" in s.value for s in at.success), mode
        assert [e for e in at.expander if "valid routes" in e.label], mode
    assert len(pages["calc"].dataframe[0].value) >= 2


def test_no_pareto_chart_code_is_left_in_the_app():
    src = open(APP, encoding="utf-8").read()
    for dead in ("make_pareto_charts", "alternative_routes", "ro_pareto", "fa_pareto", "Route_Alt_"):
        assert dead not in src, dead
    assert "Pareto Front" not in src
