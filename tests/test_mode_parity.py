"""Step 56: Route + Full Analysis shows everything that Route Only shows (Route Only is a part of Full Analysis, with the dimension-free numbers).

Before: Full Analysis in the calculator model had no tool / position / machine change section ("Where changes occurred", the split chart) and no list
of all the valid routes, which Route Only has. Step 57 then took the Pareto front graphs out of the app, both modes.

Headless (Streamlit's AppTest), no key, no network (the planner is replay-only with an empty key: the routes are the Route Builder's).
"""
import json
import os
import re
import sys

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
APP = os.path.join(ROOT, "week6", "app.py")
ROUTE_ONLY = "\U0001f5fa️ Route Only"
MODEL_V1 = "\U0001F4C8 v1 model (numbers of the report)"
FEATURES = ["Groove", "Hole", "Turning"]          # 4 valid routes in the search

# What a section of Route Only is called in Full Analysis (the same content under the name of that page). A section of Route Only that is
# not in this table, or whose counterpart is missing from Full Analysis, fails the test below.
FULL_NAME_OF = {
    "Optimal Process Route": "Optimal Process Plan",
    "Feasibility Check": "Feasibility",
    "Route Sheet (process plan)": "Route Sheet (process plan)",
    "Time Breakdown (Reference Dimensions)": ("Time Breakdown", "Tool, Position and Machine Changes"),
    "Feature → Operation Mapping": "Feature → Operation Mapping",
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")


def run(route_only=False, v1=False, features=FEATURES):
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["extraction_result"] = {"success": True, "confidence": "high", "features": list(features), "notes": "", "rejected_features": []}
    at.session_state["confirmed_features"] = list(features)
    at.run()
    if route_only:
        at.radio[0].set_value(ROUTE_ONLY).run()
    elif v1:
        at.radio(key="time_model").set_value(MODEL_V1).run()
    [b for b in at.button if "Generate" in b.label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.fixture(scope="module")
def pages():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GROQ_API_KEY", "")
        mp.setenv("CAPP_LLM_MODE", "replay")
        return {"ro": run(route_only=True), "calc": run(), "v1": run(v1=True)}


def heads(at):
    """The section headings of a page: markdown lines that start with #, without the # signs and the emoji."""
    out = []
    for m in at.markdown:
        for line in m.value.split("\n"):
            if line.startswith("#"):
                out.append(re.sub(r"^[#\s]+", "", line).strip())
    return [re.sub(r"^[^\w(]+", "", h) for h in out]


def chart(at, key):
    return [e for e in at.get("plotly_chart") if getattr(e, "key", None) == key or str(e.proto.id).endswith("-" + key)]




def starts(heading_list, name):
    return any(h.startswith(name) for h in heading_list)


@pytest.mark.parametrize("mode", ["calc", "v1"])
def test_every_section_of_route_only_is_in_full_analysis(pages, mode):
    ro, full = heads(pages["ro"]), heads(pages[mode])
    for h in ro:
        if h.startswith(("LLM-Based", "Part Configuration")):
            continue
        name = next((k for k in FULL_NAME_OF if h.startswith(k)), None)
        assert name, f"Route Only has a section '{h}' that this test does not know: add its name in Full Analysis to FULL_NAME_OF"
        wanted = FULL_NAME_OF[name]
        wanted = (wanted,) if isinstance(wanted, str) else wanted
        assert any(starts(full, w) for w in wanted), f"'{h}' of Route Only is missing in Full Analysis ({mode}); looked for {wanted}"


@pytest.mark.parametrize("mode", ["calc", "v1"])
def test_the_expanders_and_downloads_of_route_only_are_in_full_analysis(pages, mode):
    ro = pages["ro"]
    full = pages[mode]
    assert [e for e in ro.expander if "valid routes" in e.label] and [e for e in full.expander if "valid routes" in e.label]
    n_ro = re.search(r"All (\d+) valid routes", [e.label for e in ro.expander if "valid routes" in e.label][0]).group(1)
    n_full = re.search(r"All (\d+) valid routes", [e.label for e in full.expander if "valid routes" in e.label][0]).group(1)
    assert n_ro == n_full and int(n_full) >= 2                    # the same routes of the search
    for part in ("Where changes occurred", "Download the route sheet", "feasibility rules and their sources", "not evaluated"):
        assert [e for e in ro.expander if part in e.label], part
        assert [e for e in full.expander if part in e.label], part
    assert len(full.get("download_button")) >= len(ro.get("download_button"))


def test_the_calculator_page_has_the_tool_position_and_machine_changes(pages):
    at = pages["calc"]
    labels = [m.label for m in at.metric]
    for want in ("Machining", "Tool Change", "Position", "Changeover", "TOTAL (v1 model)"):
        assert any(want in x for x in labels), want
    assert chart(at, "fa_split_bar")
    heads_text = " ".join(m.value for m in at.markdown)
    assert "Time Breakdown" not in heads_text                      # the calculator's table stays the time of the job: the v1 split is named for what it is
    assert any("different model" in c.value for c in at.caption)
    # the lines under "Where changes occurred" are the changes of the route of this page
    exp = [e for e in at.expander if "Where changes occurred" in e.label][0]
    assert any("Tool Change #1" in m.value for m in exp.markdown)


def test_the_v1_model_page_is_not_given_the_calculator_split_twice(pages):
    at = pages["v1"]
    assert not chart(at, "fa_split_bar") and chart(at, "fa_time_bar")
    assert len([e for e in at.expander if "Where changes occurred" in e.label]) == 1
    assert not any("TOTAL (v1 model)" in m.label for m in at.metric)






def test_the_table_and_the_choice_of_the_optimal_route_still_use_the_routes_of_the_search_only(pages):
    for mode in ("calc", "v1"):
        at = pages[mode]
        df = at.dataframe[0].value
        assert not any("Route_Alt" in r for r in df["Route"])
        n = int(re.search(r"All (\d+) valid routes", [e.label for e in at.expander if "valid routes" in e.label][0]).group(1))
        assert len(df) == n
    df = pages["v1"].dataframe[0].value
    eff = [m for m in pages["v1"].metric if "Efficiency" in m.label][0].value
    assert eff == f"{df['Efficiency'].max()}/100"




# ---- the numbers of the new parts ---------------------------------------------------------------------------------------








def test_the_split_of_the_calculator_page_adds_up_and_its_lines_are_the_changes_it_counts(pages):
    at = pages["calc"]
    val = lambda label: float([m for m in at.metric if label in m.label][0].value.split()[0])
    parts = [val(x) for x in ("Machining", "Tool Change", "Position", "Changeover")]
    assert sum(parts) == pytest.approx(val("TOTAL (v1 model)"), abs=0.06)
    exp = [e for e in at.expander if "Where changes occurred" in e.label][0]
    n_tool, n_pos = map(int, re.search(r"\((\d+) tool \+ (\d+) position\)", exp.label).groups())
    lines = [m.value for m in exp.markdown]
    # the lines are the explanation of the counts, written by the same rule as the Time Breakdown of the v1 model and Route Only (that rule leaves
    # out the position change after Facing, so a line count can be one less than the count in the title: BASELINE.md Step 56 says so)
    assert n_tool >= 1 and n_pos >= 1
    assert any("Tool Change #1" in x for x in lines) and any("Position Change #1" in x for x in lines)
    assert 1 <= sum("Tool Change #" in x for x in lines) <= n_tool and 1 <= sum("Position Change #" in x for x in lines) <= n_pos


def test_the_split_uses_the_dimensions_of_the_page():
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["extraction_result"] = {"success": True, "confidence": "high", "features": FEATURES, "notes": "", "rejected_features": []}
    at.session_state["confirmed_features"] = FEATURES
    at.run()
    at.number_input(key="dim_diameter").set_value(60.0).run()
    [b for b in at.button if "Generate" in b.label][0].click().run()
    assert not at.exception
    big = [m for m in at.metric if "Machining" in m.label][0].value
    small = [m for m in run().metric if "Machining" in m.label][0].value
    assert float(big.split()[0]) > float(small.split()[0])
    total = float([m for m in at.metric if "TOTAL (v1 model)" in m.label][0].value.split()[0])
    parts = sum(float([m for m in at.metric if x in m.label][0].value.split()[0]) for x in ("Machining", "Tool Change", "Position", "Changeover"))
    assert total == pytest.approx(parts, abs=0.06)                 # the total and the parts of the split use the same dimensions
