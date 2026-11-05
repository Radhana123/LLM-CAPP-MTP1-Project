"""Step 50: the dimensions written on a drawing (capp/dimensions.py, the part-dimension pass of week1/image_feature_extractor.py) and what the
app does with them: a table beside the features, and the Part Dimensions of the sidebar filled from the drawing instead of the defaults.

No network: the extractor tests use a fake transport; the page tests (Streamlit's AppTest) put an extraction result in the session state.
"""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "week1"))

from capp import dimensions as D, llm_client      # noqa: E402
import image_feature_extractor as ife               # noqa: E402

REPLY = {"overall": {"diameter_mm": 50, "length_mm": "120 mm", "width_mm": None, "height_mm": None},
         "hole_diameters_mm": [10, 6.5],
         "features": [{"feature": "Groove", "text": "3 x 2 deep", "width_mm": 3, "depth_mm": 2},
                      {"feature": "Hole", "text": "2x Ø10 THRU, 25 deep", "diameter_mm": 10, "depth_mm": 25},
                      {"feature": "Thread", "text": "M10x1.25", "diameter_mm": 10, "pitch_mm": 1.25},
                      {"feature": "Chamfer", "text": "1x45", "length_mm": 1, "angle_deg": 45}],
         "surface_finish": ["Ra 1.6", "N8"], "note": "metric drawing"}


# ---------------------------------------------------------------- the parser
def test_the_limits_are_those_of_the_dataset_and_the_app():
    from capp import analysis_view as AV, dataset
    assert D.PART_LIMITS == dataset.DIM_LIMITS and D.HEIGHT_LIMITS[1] <= 500
    assert D.RA_CHOICES == tuple(c for c in AV.RA_CHOICES if c is not None)


def test_read_a_full_reply():
    r = D.read_part_dimensions(REPLY)
    assert r["ok"] and r["overall"] == {"diameter": 50.0, "length": 120.0} and r["hole_diameters"] == [10.0, 6.5]
    assert [f["feature"] for f in r["features"]] == ["Groove", "Hole", "Thread", "Chamfer"]
    assert r["features"][0] == {"feature": "Groove", "text": "3 x 2 deep", "width_mm": 3.0, "depth_mm": 2.0}
    assert r["features"][3]["angle_deg"] == 45.0 and r["ra_um"] == [1.6, 3.2] and r["note"] == "metric drawing" and r["dropped"] == []


@pytest.mark.parametrize("parsed", [None, [], "text", 5, {}, {"x": 1}, {"note": "no groups"}, {"features": []},
                                    {"features": ["Hole", "Slot"], "confidence": "low", "notes": "Could not parse model response"}])
def test_a_reply_without_any_group_is_not_ok(parsed):
    r = D.read_part_dimensions(parsed)
    assert r["ok"] is False and r["overall"] == {} and r["features"] == [] and r["hole_diameters"] == [] and r["note"]


def test_an_empty_but_valid_reply_is_ok_and_empty():
    r = D.read_part_dimensions({"overall": {"diameter_mm": None, "length_mm": None}, "hole_diameters_mm": [], "features": [], "surface_finish": []})
    assert r["ok"] and r["overall"] == {} and r["features"] == [] and r["ra_um"] == [] and r["dropped"] == []


def test_numbers_outside_the_range_of_the_app_are_dropped_and_listed_not_clipped():
    r = D.read_part_dimensions({"overall": {"diameter_mm": 900, "length_mm": 0, "width_mm": -5, "height_mm": 20}})
    assert r["overall"] == {"height": 20.0} and len(r["dropped"]) == 3 and any("900" in x for x in r["dropped"])


def test_words_and_booleans_are_not_numbers():
    r = D.read_part_dimensions({"overall": {"diameter_mm": "large", "length_mm": True}})
    assert r["overall"] == {} and len(r["dropped"]) == 2


def test_a_string_with_a_unit_or_a_symbol_is_read():
    r = D.read_part_dimensions({"overall": {"diameter_mm": "Ø50", "length_mm": "120 mm", "width_mm": "40 ±0.1"}})
    assert r["overall"] == {"diameter": 50.0, "length": 120.0, "width": 40.0}


def test_features_outside_the_vocabulary_are_dropped_and_listed():
    r = D.read_part_dimensions({"overall": {}, "features": [{"feature": "Gear", "text": "m2"}, {"feature": "Slot", "width_mm": 8}, "Hole", None]})
    assert [f["feature"] for f in r["features"]] == ["Slot"] and len(r["dropped"]) == 3


def test_a_feature_number_out_of_range_is_dropped_but_the_row_stays():
    r = D.read_part_dimensions({"overall": {}, "features": [{"feature": "Fillet", "text": "R5", "radius_mm": 5, "angle_deg": 400}]})
    assert r["features"] == [{"feature": "Fillet", "text": "R5", "radius_mm": 5.0}] and r["dropped"] == ["Fillet angle 400"]


def test_a_feature_with_neither_text_nor_numbers_is_not_a_row():
    assert D.read_part_dimensions({"overall": {}, "features": [{"feature": "Hole"}, {"feature": "Slot", "width_mm": None}]})["features"] == []


def test_a_single_feature_object_instead_of_a_list_is_accepted_and_the_table_is_capped():
    assert len(D.read_part_dimensions({"overall": {}, "features": {"feature": "Slot", "width_mm": 8}})["features"]) == 1
    many = [{"feature": "Hole", "diameter_mm": 5 + i * 0.1, "text": str(i)} for i in range(60)]
    assert len(D.read_part_dimensions({"overall": {}, "features": many})["features"]) == D.MAX_FEATURE_ROWS


def test_the_text_of_a_feature_is_one_short_line():
    r = D.read_part_dimensions({"overall": {}, "features": [{"feature": "Slot", "text": "  8 \n wide  " + "x" * 200}]})
    assert "\n" not in r["features"][0]["text"] and len(r["features"][0]["text"]) <= 60


@pytest.mark.parametrize("raw, ra", [("Ra 1.6", 1.6), ("Ra1.6", 1.6), ("ra = 0,8", 0.8), ("1.6", 1.6), (3.2, 3.2), ("Ra 3.2 µm", 3.2),
                                     ("N7", 1.6), ("N8", 3.2), ("N 6", 0.8), ("N12", 50.0), ("Ra 0.2", 0.2)])
def test_roughness_forms(raw, ra):
    assert D._ra_um(raw) == ra


@pytest.mark.parametrize("raw", ["Rz 10", "Rt 4", "Ra 40 µin", "N0", "N13", "", "smooth", None, True, 0, 500, [1.6]])
def test_roughness_that_is_not_an_ra_in_um(raw):
    assert D._ra_um(raw) is None


def test_roughness_is_sorted_unique_and_a_bare_string_is_accepted():
    r = D.read_part_dimensions({"surface_finish": ["Ra 3.2", "N8", "Ra 1.6", "Rz 8"]})
    assert r["ra_um"] == [1.6, 3.2] and r["dropped"] == ["Ra Rz 8"]
    assert D.read_part_dimensions({"surface_finish": "Ra 0.8"})["ra_um"] == [0.8]


def test_the_hole_diameters_are_cleaned_as_before():
    r = D.read_part_dimensions({"hole_diameters_mm": [6, "Ø10", "M8", 10.0, 300]})
    assert r["hole_diameters"] == [10.0, 6.0] and r["dropped"] == ["M8", "300"]


# ---------------------------------------------------------------- what the sidebar takes
def test_derive_takes_only_what_is_written():
    d = D.derive_part_dims(D.read_part_dimensions(REPLY))
    assert (d["diameter"], d["length"], d["depth"], d["hole_diameter"], d["ra_um"]) == (50.0, 120.0, 25.0, 10.0, 1.6)
    assert d["width"] is None                                              # a round part has no width written: no invention
    assert d["source"]["depth"] == "deepest depth written (Hole)" and "width" not in d["source"]


def test_derive_of_nothing_is_all_none():
    for read in (None, {}, D.read_part_dimensions({"overall": {}}), D.read_part_dimensions({})):
        d = D.derive_part_dims(read)
        assert all(d[k] is None for k in ("diameter", "length", "depth", "width", "hole_diameter", "ra_um")) and d["source"] == {}


def test_a_block_gives_length_and_width_and_a_height_is_not_a_depth():
    read = D.read_part_dimensions({"overall": {"length_mm": 80, "width_mm": 50, "height_mm": 20}})
    d = D.derive_part_dims(read)
    assert (d["length"], d["width"], d["diameter"], d["depth"]) == (80.0, 50.0, None, None)
    assert any("height of 20 mm" in n for n in d["notes"])


def test_the_depth_is_the_deepest_written_and_within_the_range_of_the_app():
    read = D.read_part_dimensions({"overall": {}, "features": [{"feature": "Pocket", "depth_mm": 8, "text": "8 deep"}, {"feature": "Hole", "depth_mm": 30},
                                                {"feature": "Groove", "depth_mm": 0.5}, {"feature": "Slot", "depth_mm": 500}]})
    d = D.derive_part_dims(read)
    assert d["depth"] == 30.0 and "Hole" in d["source"]["depth"]
    only_small = D.derive_part_dims(D.read_part_dimensions({"overall": {}, "features": [{"feature": "Groove", "depth_mm": 0.5}]}))
    assert only_small["depth"] is None


def test_the_strictest_ra_is_used_if_the_app_offers_it():
    assert D.derive_part_dims(D.read_part_dimensions({"surface_finish": ["Ra 3.2", "Ra 0.8"]}))["ra_um"] == 0.8
    d = D.derive_part_dims(D.read_part_dimensions({"surface_finish": ["Ra 0.2", "Ra 3.2"]}))
    assert d["ra_um"] is None and any("0.2" in n and "not one of the values" in n for n in d["notes"])
    assert D.derive_part_dims(D.read_part_dimensions({"surface_finish": ["Ra 1.25"]}))["ra_um"] is None


def test_the_largest_hole_is_the_hole_diameter():
    assert D.derive_part_dims(D.read_part_dimensions({"hole_diameters_mm": [6, 12, 8]}))["hole_diameter"] == 12.0


def test_describe_derived_lists_what_was_read_and_what_was_not():
    written, missing = D.describe_derived(D.derive_part_dims(D.read_part_dimensions(REPLY)))
    assert written == ["Diameter 50 mm", "Length 120 mm", "Depth 25 mm", "Hole Ø10 mm", "Ra 1.6 um"] and missing == ["Width"]
    assert D.describe_derived(None) == ([], ["Diameter", "Length", "Depth", "Width", "Hole diameter", "Ra"])


def test_feature_numbers_text():
    r = D.read_part_dimensions(REPLY)
    assert [D.feature_numbers_text(f) for f in r["features"]] == ["width 3, depth 2", "Ø10, depth 25", "Ø10, pitch 1.25",
                                                                  "length 1, angle 45 deg"]
    assert D.feature_numbers_text({"feature": "Fillet", "radius_mm": 5.0}) == "R5" and D.feature_numbers_text({"feature": "Slot"}) == ""


# ---------------------------------------------------------------- the pre-fill rule
def extraction_with(read_reply=REPLY):
    read = D.read_part_dimensions(read_reply)
    return {"success": True, "features": ["Hole"], "part_dimensions": {"read": read, "derived": D.derive_part_dims(read)}}


def test_prefill_part_applies_a_detection_once():
    ex = extraction_with()
    values, sig = D.prefill_part(ex, None)
    assert values == {"diameter": 50.0, "length": 120.0, "depth": 25.0, "ra_um": 1.6}          # no width, no hole (own rule)
    again, sig2 = D.prefill_part(ex, sig, {})
    assert again == {} and sig2 == sig                                         # the user's edit is not overwritten at the next rerun
    new, sig3 = D.prefill_part(extraction_with({"overall": {"diameter_mm": 30}}), sig)
    assert new == {"diameter": 30.0} and sig3 != sig                           # a new drawing is applied


def test_prefill_part_puts_back_only_the_fields_that_lost_their_value():
    ex = extraction_with()
    _, sig = D.prefill_part(ex, None)
    values, sig2 = D.prefill_part(ex, sig, {"diameter": True, "length": False, "depth": False, "ra_um": True})
    assert values == {"length": 120.0, "depth": 25.0} and sig2 == sig


def test_prefill_part_never_touches_a_field_without_a_detection():
    assert D.prefill_part(None, None) == ({}, None)
    assert D.prefill_part({"success": False, "part_dimensions": extraction_with()["part_dimensions"]}, None) == ({}, None)
    assert D.prefill_part({"success": True}, ("x",)) == ({}, ("x",))
    assert D.prefill_part({"success": True, "part_dimensions": {"read": None, "derived": None}}, None) == ({}, None)
    assert D.prefill_part(extraction_with({"overall": {}}), None) == ({}, None)


# ---------------------------------------------------------------- the dimension pass of the extractor
@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CAPP_LLM_MODE", "live")
    monkeypatch.setenv("CAPP_RUN_TAG", "t")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm_client, "_sleep", lambda s: None)
    llm_client.reset_call_log()
    yield
    llm_client.reset_transport()


def kind_of(messages):
    t = messages[0]["content"][0]["text"]
    if "against a checklist" in t:
        return "checklist"
    if "DIMENSION CALLOUTS" in t:
        return "part_dims" if "features found in this drawing" in t else "hole_dims"
    return "geometry" if "DRAWN GEOMETRY" in t else "annotation"


def transport(dim_reply=None, raises=()):
    seen = []

    def t(model, messages, params):
        k = kind_of(messages)
        seen.append({"kind": k, "prompt": messages[0]["content"][0]["text"], "params": params, "messages": messages})
        if k in raises:
            raise RuntimeError("boom")
        reply = {"geometry": {"features": ["Hole", "Groove"], "confidence": "high"}, "annotation": {"features": ["Hole", "Thread"]},
                 "checklist": {"checklist": {"Hole": "yes: circle", "Chamfer": "yes: 1x45"}},
                 "part_dims": dim_reply if dim_reply is not None else REPLY, "hole_dims": {D.KEY: [8]}}[k]
        return {"text": reply if isinstance(reply, str) else json.dumps(reply), "prompt_tokens": 1, "completion_tokens": 1, "finish_reason": "stop"}
    t.seen = seen
    return t


def extract(**kw):
    return ife.extract_features_from_image(b"not-a-real-image", "png", **kw)


def test_the_part_pass_replaces_the_hole_call_so_the_app_makes_four_calls(isolated):
    t = transport()
    llm_client.set_transport(t)
    r = extract(with_dimensions=True, with_checklist=True, with_part_dimensions=True)
    assert [c["kind"] for c in t.seen] == ["geometry", "annotation", "checklist", "part_dims"]
    assert r["success"] and r["hole_diameters"] == [10.0, 6.5] and r["dimensions"]["success"] is True
    assert r["part_dimensions"]["derived"]["diameter"] == 50.0 and r["part_dimensions"]["read"]["overall"]["length"] == 120.0


def test_the_prompt_names_the_features_that_were_found(isolated):
    t = transport()
    llm_client.set_transport(t)
    extract(with_checklist=True, with_part_dimensions=True)
    prompt = [c for c in t.seen if c["kind"] == "part_dims"][0]["prompt"]
    assert "are: Hole, Groove, Thread, Chamfer." in prompt and "Never estimate" in prompt and "WRITTEN" in prompt
    assert '"overall"' in prompt and '"surface_finish"' in prompt and "inches" in prompt


def test_the_prompt_formats_with_no_feature_and_keeps_its_json_braces():
    text = ife.PART_DIMENSION_SYSTEM_PROMPT.format(features="none (look for any dimension)")
    assert '{"overall": {"diameter_mm": 50' in text and "none (look for any dimension)" in text and "{features}" not in text


def test_without_the_option_nothing_changes(isolated):
    t = transport()
    llm_client.set_transport(t)
    r = extract(with_dimensions=True)
    assert [c["kind"] for c in t.seen] == ["geometry", "annotation", "hole_dims"] and "part_dimensions" not in r and r["hole_diameters"] == [8.0]
    t2 = transport()
    os.environ["CAPP_RUN_TAG"] = "other"
    llm_client.set_transport(t2)
    assert "part_dimensions" not in extract() and [c["kind"] for c in t2.seen] == ["geometry", "annotation"]


def test_the_feature_passes_are_untouched_by_the_part_pass(isolated, monkeypatch):
    t1 = transport()
    llm_client.set_transport(t1)
    extract()
    plain = [(c["messages"], c["params"]) for c in t1.seen]
    monkeypatch.setenv("CAPP_RUN_TAG", "second")
    t2 = transport()
    llm_client.set_transport(t2)
    extract(with_part_dimensions=True)
    assert [(c["messages"], c["params"]) for c in t2.seen if c["kind"] in ("geometry", "annotation")] == plain


def test_a_failed_part_pass_never_fails_the_extraction(isolated):
    llm_client.set_transport(transport(raises=("part_dims",)))
    r = extract(with_part_dimensions=True)
    assert r["success"] and r["features"] == ["Hole", "Groove", "Thread"]
    assert r["hole_diameters"] == [] and r["dimensions"]["success"] is False and "boom" in r["dimensions"]["error"]
    assert r["part_dimensions"] == {"read": None, "derived": None}


@pytest.mark.parametrize("reply", ["not json at all", "{}", '{"note": "x"}', "[1, 2]"])
def test_a_reply_without_dimensions_is_a_failed_dimension_read(isolated, reply):
    llm_client.set_transport(transport(dim_reply=reply))
    r = extract(with_part_dimensions=True)
    assert r["success"] and r["dimensions"]["success"] is False and r["part_dimensions"]["derived"] is None and r["hole_diameters"] == []


def test_no_api_key_is_reported_not_raised(isolated, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    r = ife._extract_part_dimensions("abc", "png", ["Hole"])
    assert r["success"] is False and r["read"] is None and r["derived"] is None and r["error"]


def test_the_dimension_call_is_deterministic_and_cached(isolated):
    t = transport()
    llm_client.set_transport(t)
    extract(with_checklist=True, with_part_dimensions=True)
    call = [c for c in t.seen if c["kind"] == "part_dims"][0]
    assert call["params"]["temperature"] == 0 and call["params"]["response_format"] == {"type": "json_object"}
    n = len(t.seen)
    extract(with_checklist=True, with_part_dimensions=True)
    assert len(t.seen) == n


# ---------------------------------------------------------------- the page
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest      # noqa: E402

APP = os.path.join(ROOT, "week6", "app.py")


@pytest.fixture
def no_network(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    monkeypatch.setenv("CAPP_CORRECTIONS_LOG", str(tmp_path / "log" / "c.jsonl"))


def page(ex, features=("Hole", "Groove")):
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["extraction_result"] = ex
    at.session_state["confirmed_features"] = list(features)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def full_extraction(reply=REPLY):
    ex = extraction_with(reply)
    ex.update({"confidence": "high", "notes": "", "rejected_features": [], "hole_diameters": ex["part_dimensions"]["read"]["hole_diameters"],
               "dimensions": {"success": True, "ok": True, "hole_diameters": ex["part_dimensions"]["read"]["hole_diameters"], "dropped": [], "note": ""},
               "features": ["Hole", "Groove"]})
    return ex


def field(at, key):
    return at.number_input(key=key).value


def captions(at):
    return " | ".join(c.value for c in at.caption)


def test_the_fields_take_what_the_drawing_says_and_the_rest_keeps_the_default(no_network):
    at = page(full_extraction())
    assert (field(at, "dim_diameter"), field(at, "dim_length"), field(at, "dim_depth")) == (50.0, 120.0, 25.0)
    assert field(at, "dim_width") == 20.0 and at.number_input(key="hole_dia_in").value == 10.0 and at.selectbox(key="dim_ra").value == 1.6
    text = captions(at)
    assert "Read from your drawing: Diameter 50 mm, Length 120 mm, Depth 25 mm, Hole Ø10 mm, Ra 1.6 um" in text
    assert "Not written on the drawing, so the default stays (set it yourself): Width" in text


def test_without_an_extraction_the_defaults_are_the_old_ones_and_nothing_is_claimed(no_network):
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert [field(at, k) for k in ("dim_diameter", "dim_length", "dim_depth", "dim_width")] == [25.0, 50.0, 20.0, 20.0]
    assert at.selectbox(key="dim_ra").value is None and "Read from your drawing" not in captions(at)


def test_an_extraction_without_part_dimensions_leaves_the_defaults_and_says_nothing(no_network):
    ex = {"success": True, "confidence": "high", "features": ["Hole"], "notes": "", "rejected_features": []}
    at = page(ex, ("Hole",))
    assert [field(at, k) for k in ("dim_diameter", "dim_length", "dim_depth", "dim_width")] == [25.0, 50.0, 20.0, 20.0]
    assert "Read from your drawing" not in captions(at) and "could not be read" not in captions(at)


def test_the_users_edit_is_kept_at_the_next_rerun_but_a_new_drawing_is_applied(no_network):
    at = page(full_extraction())
    at.number_input(key="dim_diameter").set_value(60.0).run()
    assert field(at, "dim_diameter") == 60.0
    at.run()
    assert field(at, "dim_diameter") == 60.0
    at.session_state["extraction_result"] = full_extraction({"overall": {"diameter_mm": 30, "length_mm": 70}})
    at.run()
    assert field(at, "dim_diameter") == 30.0 and field(at, "dim_length") == 70.0
    assert field(at, "dim_depth") == 25.0                  # that field is not in the new drawing: it keeps its value, it is not reset


def test_the_values_come_back_after_route_only_mode(no_network):
    at = page(full_extraction())
    at.radio[0].set_value("\U0001f5fa️ Route Only").run()
    assert not at.exception
    at.radio[0].set_value("\U0001f4ca Route + Full Analysis").run()
    assert (field(at, "dim_diameter"), field(at, "dim_length"), field(at, "dim_depth")) == (50.0, 120.0, 25.0)


def test_a_failed_dimension_read_says_so_and_keeps_the_defaults(no_network):
    ex = {"success": True, "confidence": "high", "features": ["Hole"], "notes": "", "rejected_features": [],
          "hole_diameters": [], "dimensions": {"success": False, "error": "timeout"}, "part_dimensions": {"read": None, "derived": None}}
    at = page(ex, ("Hole",))
    assert [field(at, k) for k in ("dim_diameter", "dim_length", "dim_depth", "dim_width")] == [25.0, 50.0, 20.0, 20.0]
    assert "could not be read from the drawing (timeout)" in captions(at)
    assert "The dimensions could not be read (timeout)" in captions(at)


def test_the_table_of_dimensions_is_shown_beside_the_features(no_network):
    at = page(full_extraction())
    tables = [d.value.to_dict("records") for d in at.dataframe if "Numbers (mm unless stated)" in list(d.value.columns)]
    assert len(tables) == 1
    rows = {r["Item"]: r for r in tables[0]}
    assert rows["Whole part"]["Numbers (mm unless stated)"] == "Ø50, length 120" and rows["Groove"]["Numbers (mm unless stated)"] == "width 3, depth 2"
    assert rows["Hole"]["As written"] == "2x Ø10 THRU, 25 deep" and rows["Surface finish"]["Numbers (mm unless stated)"] == "Ra 1.6, 3.2 um"
    assert "go to the operation that makes it" in captions(at)


def test_an_empty_dimension_read_says_that_nothing_is_written(no_network):
    at = page(full_extraction({"overall": {}, "features": [], "hole_diameters_mm": [], "surface_finish": []}))
    assert "No dimension is written on the drawing" in captions(at)
    assert [field(at, k) for k in ("dim_diameter", "dim_length", "dim_depth", "dim_width")] == [25.0, 50.0, 20.0, 20.0]
    assert "Not written on the drawing, so the default stays (set it yourself): Diameter, Length, Depth, Width, Hole diameter, Ra" in captions(at)


def test_what_was_dropped_is_listed(no_network):
    at = page(full_extraction({"overall": {"diameter_mm": 900, "length_mm": 100}}))
    assert "Not used (unreadable, or outside the range of the app): overall diameter 900" in captions(at)
    assert field(at, "dim_length") == 100.0 and field(at, "dim_diameter") == 25.0


def test_the_pipeline_gets_the_dimensions_of_the_drawing(no_network, monkeypatch):
    from capp import analysis_view as AV
    seen = []
    real = AV.make_part

    def spy(*a, **k):
        seen.append((a, k))
        return real(*a, **k)
    monkeypatch.setattr(AV, "make_part", spy)
    at = page(full_extraction())
    [b for b in at.button if "Generate" in b.label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert seen, "Full Analysis builds the part for the physics"
    args, kw = seen[-1]
    dims = kw.get("dims") or args[4]
    assert (dims["diameter"], dims["length"], dims["depth"], dims["width"]) == (50.0, 120.0, 25.0, 20.0)
    assert kw.get("required_ra_um") == 1.6 and kw.get("hole_diameter") == 10.0


def test_route_only_mode_does_not_show_the_fields_or_the_claim(no_network):
    at = page(full_extraction())
    at.radio[0].set_value("\U0001f5fa️ Route Only").run()
    assert not at.exception and "Read from your drawing" not in captions(at)
