"""Step 14, the front end of week6/app.py: manual feature input (no image, no API key) and the hole diameter read from the drawing.

Headless (Streamlit's AppTest), no network: the planner is replay-only with an empty key, as in tests/test_app.py. The extraction
itself is not run here (it needs an image upload): the extraction result is put in the session state, which is what the app does
after the extractor returns. The extractor's dimension pass is tested in tests/test_dimensions.py.
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
UPLOAD = "🖼️ Upload a drawing (AI extraction)"
MANUAL = "✍️ Select the features myself"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")


def extraction(features=("Hole", "Turning"), holes=None, dim_ok=True, dim_error=""):
    r = {"success": True, "confidence": "high", "features": list(features), "notes": "", "rejected_features": []}
    if holes is not None:
        r["hole_diameters"] = list(holes)
        r["dimensions"] = {"success": dim_ok, "hole_diameters": list(holes), "dropped": [], "note": "", "error": dim_error}
    return r


def new_app():
    return AppTest.from_file(APP, default_timeout=180)


def with_extraction(features=("Hole", "Turning"), holes=None, **kw):
    at = new_app()
    at.session_state["extraction_result"] = extraction(features, holes, **kw)
    at.session_state["confirmed_features"] = list(features)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def radio(at, label_part):
    return [r for r in at.radio if label_part in r.label][0]


def mode(at, value):
    radio(at, "What do you want").set_value(value).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def manual(at):
    radio(at, "give the features").set_value(MANUAL).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def pick(at, features):
    [m for m in at.multiselect if m.label == "Features"][0].set_value(list(features)).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def hole_input(at):
    return [n for n in at.number_input if "Hole diameter" in n.label][0]


def generate(at):
    [b for b in at.button if "Generate" in b.label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def headers(at):
    return [m.value for m in at.markdown if "step-hdr" in m.value]


def texts(elements):
    return [e.value for e in elements]


# ---------------------------------------------------------------- manual feature input
def test_the_input_method_defaults_to_upload_and_offers_manual():
    at = new_app()
    at.run()
    r = radio(at, "give the features")
    assert r.value == UPLOAD and list(r.options) == [UPLOAD, MANUAL]
    assert not any(m.label == "Features" for m in at.multiselect)          # the editable list only appears after an extraction


def test_manual_mode_shows_the_feature_picker_without_any_extraction():
    at = manual(new_app().run())
    picker = [m for m in at.multiselect if m.label == "Features"]
    assert len(picker) == 1 and "Hole" in picker[0].options and "Turning" in picker[0].options
    assert not at.error and not any("Detected" in s.value for s in at.success)


def test_manual_mode_runs_the_whole_pipeline():
    at = pick(manual(new_app().run()), ["Hole", "Turning"])
    generate(at)
    assert any("STEP 5" in h for h in headers(at))
    assert any("STEP 1" in h for h in headers(at)) and any("Token sequence" in s.value for s in at.success)


def test_manual_mode_route_only_runs_too():
    at = mode(pick(manual(new_app().run()), ["Slot", "Pocket"]), ROUTE_ONLY)
    generate(at)
    assert not any("STEP 5" in h for h in headers(at)) and any("Feasibility Check" in m.value for m in at.markdown)


def test_manual_mode_without_a_selection_asks_for_one():
    at = manual(new_app().run())
    generate(at)
    assert any("select at least one feature" in e.value for e in at.error)


def test_an_empty_manual_selection_does_not_fall_back_to_an_earlier_extraction():
    at = with_extraction(features=("Hole", "Turning"))
    manual(at)
    generate(at)
    assert any("select at least one feature" in e.value for e in at.error)
    assert not any("STEP 1" in h for h in headers(at))


def test_the_manual_selection_is_used_not_the_extraction():
    at = with_extraction(features=("Hole", "Turning"))
    pick(manual(at), ["Slot"])
    generate(at)
    from capp import legacy       # noqa: F401
    from tokenizer import tokenize
    shown = [s.value for s in at.success if "Token sequence" in s.value][0]
    part = {"material": "Aluminum", "features": ["Slot"], "tolerance": "0.02mm", "batch_size": 500, "dims": {}}
    other = dict(part, features=["Hole", "Turning"])
    chips = lambda p: " ".join(f"`{t}`" for t in tokenize(p)["tokens"])       # noqa: E731
    assert chips(part) in shown and chips(other) not in shown
    assert at.session_state["confirmed_features"] == ["Hole", "Turning"]       # the extraction is kept, only not used


def test_switching_back_to_upload_uses_the_extraction_again():
    at = with_extraction(features=("Hole", "Turning"))
    pick(manual(at), ["Slot"])
    radio(at, "give the features").set_value(UPLOAD).run()
    assert not at.exception
    generate(at)
    assert any("STEP 5" in h for h in headers(at))


def test_the_manual_selection_is_kept_while_the_picker_is_hidden():
    at = pick(manual(new_app().run()), ["Slot", "Pocket"])
    radio(at, "give the features").set_value(UPLOAD).run()
    assert not at.exception and not any(m.label == "Features" for m in at.multiselect)
    manual(at)
    assert [m for m in at.multiselect if m.label == "Features"][0].value == ["Slot", "Pocket"]


def test_manual_mode_every_feature_can_be_picked():
    at = manual(new_app().run())
    picker = [m for m in at.multiselect if m.label == "Features"][0]
    from feature_vocab import GEOMETRY_FEATURES
    assert list(picker.options) == list(GEOMETRY_FEATURES)


def test_manual_mode_hole_diameter_is_still_typed_in_full_mode():
    at = pick(manual(new_app().run()), ["Hole"])
    assert hole_input(at).value == 0.0
    hole_input(at).set_value(8.0).run()
    assert hole_input(at).value == 8.0


# ---------------------------------------------------------------- the hole diameter read from the drawing
def test_the_largest_detected_hole_is_prefilled():
    at = with_extraction(holes=[10.0, 6.0])
    assert hole_input(at).value == 10.0


def test_the_prefill_names_every_detected_diameter_and_asks_for_a_check():
    at = with_extraction(holes=[10.0, 6.0])
    cap = " ".join(texts(at.caption))
    assert "Ø10" in cap and "Ø6" in cap and "largest" in cap and "check" in cap


def test_a_user_edit_survives_the_next_rerun():
    at = with_extraction(holes=[10.0, 6.0])
    hole_input(at).set_value(7.0).run()
    assert hole_input(at).value == 7.0
    at.selectbox[0].set_value(at.selectbox[0].options[1]).run()               # any other widget triggers a rerun
    assert not at.exception and hole_input(at).value == 7.0


def test_a_new_detection_is_prefilled_again():
    at = with_extraction(holes=[10.0, 6.0])
    hole_input(at).set_value(7.0).run()
    at.session_state["extraction_result"] = extraction(holes=[12.0])
    at.run()
    assert not at.exception and hole_input(at).value == 12.0


def test_an_extraction_without_a_hole_diameter_leaves_the_field_alone():
    at = with_extraction(holes=[])
    assert hole_input(at).value == 0.0
    hole_input(at).set_value(5.0).run()
    at.session_state["extraction_result"] = extraction(holes=[])
    at.run()
    assert hole_input(at).value == 5.0


def test_an_extraction_without_the_dimension_pass_leaves_the_field_alone():
    at = with_extraction(holes=None)
    assert hole_input(at).value == 0.0
    assert not any("Hole diameter" in c for c in texts(at.caption) if "AI read" in c)


def test_the_captions_for_nothing_read_and_for_a_failed_read():
    none_read = with_extraction(holes=[])
    assert any("No hole diameter is written" in c for c in texts(none_read.caption))
    failed = with_extraction(holes=[], dim_ok=False, dim_error="Vision API error: boom")
    assert any("not read" in c and "boom" in c for c in texts(failed.caption))


def test_route_only_mode_applies_the_prefill_when_full_mode_is_chosen_later():
    at = with_extraction(holes=[10.0])
    mode(at, ROUTE_ONLY)
    assert not [n for n in at.number_input if "Hole diameter" in n.label]
    mode(at, FULL)
    assert hole_input(at).value == 10.0


def test_the_prefilled_diameter_reaches_the_feasibility_check():
    """Ø10 and 40 mm deep: 4 x D, over the 3 x D limit of one pass: the deep-hole rule runs and warns. Without a diameter it cannot run."""
    at = with_extraction(holes=[10.0])
    [n for n in at.number_input if n.label == "Depth (mm)"][0].set_value(40.0).run()
    generate(at)
    assert any("HOLE-DEEP" in w for w in texts(at.warning))
    blank = with_extraction(holes=[])
    [n for n in blank.number_input if n.label == "Depth (mm)"][0].set_value(40.0).run()
    generate(blank)
    assert not any("HOLE-DEEP" in w for w in texts(blank.warning))


def test_the_prefill_adds_no_widget_to_the_results():
    at = with_extraction(holes=[10.0])
    before = (len(at.selectbox), len(at.slider), len(at.radio), len(at.number_input), len(at.checkbox), len(at.multiselect))
    generate(at)
    after = (len(at.selectbox), len(at.slider), len(at.radio), len(at.number_input), len(at.checkbox), len(at.multiselect))
    assert before == after


def test_a_failed_extraction_still_shows_the_error_and_no_dimension_caption():
    at = new_app()
    at.session_state["extraction_result"] = {"success": False, "features": [], "rejected_features": [], "confidence": None,
                                             "notes": "", "error": "GROQ_API_KEY not configured."}
    at.run()
    assert not at.exception and any("Extraction failed" in e.value for e in at.error)
    assert not any("🕳️" in c for c in texts(at.caption))
