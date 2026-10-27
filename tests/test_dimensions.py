"""Hole diameters read from a drawing (Step 14): the parser (capp/dimensions.py) and the optional dimension pass of the extractor."""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for d in ("week1",):
    sys.path.insert(0, os.path.join(ROOT, d))

from capp import dimensions as D, llm_client      # noqa: E402
import image_feature_extractor as ife               # noqa: E402


# ---------------------------------------------------------------- the parser
@pytest.mark.parametrize("raw, mm", [
    (8, 8.0), (6.5, 6.5), ("8", 8.0), ("8 mm", 8.0), ("Ø8", 8.0), ("⌀8", 8.0), ("dia 8", 8.0), ("Ø 10", 10.0),
    ("Ø12 H7", 12.0), ("12H7", 12.0), ("Ø8 ±0.1", 8.0), ("12 +0.02", 12.0), ("12 -0.02", 12.0), ("2x Ø6 THRU", 6.0),
    ("Ø10 thru", 10.0), ("8,5", 8.5), ("6.35 mm DEEP 12", 6.35), (" 9 ", 9.0), (D.HOLE_MIN_MM, D.HOLE_MIN_MM),
    (D.HOLE_MAX_MM, D.HOLE_MAX_MM),
])
def test_accepted_forms(raw, mm):
    assert D._to_mm(raw) == mm


@pytest.mark.parametrize("raw", [
    "M8", "m10", "M10x1.25", "10x1.25", "4 x 8", "5 in", '0.25"', "5-8", "abc", "", "  ", None, True, False, [8], {"d": 8},
    float("nan"), float("inf"), -3, 0, 0.1, D.HOLE_MIN_MM - 0.01, D.HOLE_MAX_MM + 0.01, 250, "Ø0.2",
])
def test_rejected_forms(raw):
    assert D._to_mm(raw) is None


def test_rounding_to_hundredths():
    assert D._to_mm(8.123456) == 8.12


def test_read_sorts_dedupes_and_reports_what_was_dropped():
    r = D.read_hole_diameters({D.KEY: [6, "Ø10", "M8", 10.0, 300, "x"], "note": "  two sizes "})
    assert r["ok"] and r["hole_diameters"] == [10.0, 6.0]
    assert r["dropped"] == ["M8", "300", "x"] and r["note"] == "two sizes"


def test_read_empty_list_is_ok_and_empty():
    for raw in ([], None):
        r = D.read_hole_diameters({D.KEY: raw})
        assert r["ok"] and r["hole_diameters"] == [] and r["dropped"] == []


def test_read_single_value_instead_of_a_list():
    assert D.read_hole_diameters({D.KEY: "Ø8"})["hole_diameters"] == [8.0]
    assert D.read_hole_diameters({D.KEY: 8})["hole_diameters"] == [8.0]


def test_read_missing_key_or_wrong_type_is_not_ok():
    for parsed in ({"features": ["Hole"]}, {}, [], None, "text", 5):
        r = D.read_hole_diameters(parsed)
        assert r["ok"] is False and r["hole_diameters"] == [] and r["note"]


def test_read_caps_the_list_and_reports_the_rest():
    many = list(range(1, 14))
    r = D.read_hole_diameters({D.KEY: many})
    assert len(r["hole_diameters"]) == D.MAX_LISTED and r["hole_diameters"][0] == 13.0
    assert len(r["dropped"]) == len(many) - D.MAX_LISTED


def test_read_note_is_optional_and_must_be_text():
    assert D.read_hole_diameters({D.KEY: [8]})["note"] == ""
    assert D.read_hole_diameters({D.KEY: [8], "note": 5})["note"] == ""
    assert D.read_hole_diameters({D.KEY: [8], "notes": "from notes"})["note"] == "from notes"


def test_choose_is_the_largest():
    assert D.choose_hole_diameter([6.0, 10.0, 8.0]) == 10.0
    assert D.choose_hole_diameter([]) is None and D.choose_hole_diameter(None) is None
    assert D.choose_hole_diameter([None, 4.0]) == 4.0


def test_prefill_applies_a_detection_once():
    ex = {"success": True, "hole_diameters": [6.0, 10.0]}
    value, sig = D.prefill(ex, None)
    assert value == 10.0 and sig == (6.0, 10.0)
    assert D.prefill(ex, sig) == (None, sig)                                    # the user's edit is not overwritten at the next rerun
    value2, sig2 = D.prefill({"success": True, "hole_diameters": [12.0]}, sig)  # a new drawing is applied
    assert value2 == 12.0 and sig2 == (12.0,)


def test_prefill_is_applied_again_when_the_field_lost_its_value():
    """Route Only mode does not show the field and Streamlit drops its state: back in Full mode the same detection is put back."""
    ex = {"success": True, "hole_diameters": [6.0, 10.0]}
    sig = (6.0, 10.0)
    assert D.prefill(ex, sig, present=False) == (10.0, sig)
    assert D.prefill(ex, sig, present=True) == (None, sig)
    assert D.prefill({"success": True, "hole_diameters": []}, sig, present=False) == (None, sig)      # nothing to put back


def test_prefill_never_touches_the_field_without_a_detection():
    assert D.prefill(None, None) == (None, None)
    assert D.prefill({"success": True, "hole_diameters": []}, (6.0,)) == (None, (6.0,))
    assert D.prefill({"success": True}, None) == (None, None)
    assert D.prefill({"success": False, "hole_diameters": [8.0]}, None) == (None, None)


# ---------------------------------------------------------------- the dimension pass of the extractor
@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CAPP_LLM_MODE", "live")
    monkeypatch.setenv("CAPP_RUN_TAG", "t")
    monkeypatch.setattr(llm_client, "_sleep", lambda s: None)
    llm_client.reset_call_log()
    yield
    llm_client.reset_transport()


FEATURE_REPLY = {"features": ["Hole", "Thread"], "confidence": "high", "notes": "ok"}


def is_dimension_call(messages):
    return messages[0]["content"][0]["text"].startswith("You are a manufacturing engineer reading the DIMENSION")


def transport(dim_reply=None, dim_raises=False):
    seen = []

    def t(model, messages, params):
        dim = is_dimension_call(messages)
        seen.append({"dim": dim, "messages": messages, "params": params, "model": model})
        if dim and dim_raises:
            raise RuntimeError("boom")
        text = dim_reply if dim else json.dumps(FEATURE_REPLY)
        return {"text": text, "prompt_tokens": 10, "completion_tokens": 5, "finish_reason": "stop"}
    t.seen = seen
    return t


def extract(**kw):
    return ife.extract_features_from_image(b"not-a-real-image", "png", **kw)


def test_default_is_unchanged_two_calls_and_no_dimension_keys():
    t = transport(json.dumps({D.KEY: [8]}))
    llm_client.set_transport(t)
    r = extract()
    assert r["success"] and "hole_diameters" not in r and "dimensions" not in r
    assert len(t.seen) == len(ife.EXTRACTION_PASSES) and not any(c["dim"] for c in t.seen)


def test_with_dimensions_adds_exactly_one_call_and_records_the_diameters():
    t = transport(json.dumps({D.KEY: [6, "Ø10", "M8"], "note": "two holes"}))
    llm_client.set_transport(t)
    r = extract(with_dimensions=True)
    assert r["success"] and r["features"] == ["Hole", "Thread"]
    assert r["hole_diameters"] == [10.0, 6.0]
    dim = r["dimensions"]
    assert dim["success"] and dim["dropped"] == ["M8"] and dim["note"] == "two holes"
    assert len(t.seen) == len(ife.EXTRACTION_PASSES) + 1 and [c["dim"] for c in t.seen].count(True) == 1


def test_the_feature_passes_are_untouched_by_the_dimension_pass(monkeypatch):
    """Same messages and params for the two feature passes with and without dimensions: their cache keys (and recall) do not change."""
    t1 = transport(json.dumps({D.KEY: []}))
    llm_client.set_transport(t1)
    extract()
    plain = [(c["messages"], c["params"]) for c in t1.seen]
    monkeypatch.setenv("CAPP_RUN_TAG", "second")                  # a new run tag: the cache of the first run is not used
    t2 = transport(json.dumps({D.KEY: []}))
    llm_client.set_transport(t2)
    extract(with_dimensions=True)
    with_dim = [(c["messages"], c["params"]) for c in t2.seen if not c["dim"]]
    assert len(plain) == len(with_dim) == len(ife.EXTRACTION_PASSES) and plain == with_dim
    # and the cache does serve the feature passes of the first run to a with_dimensions run of the same tag:
    monkeypatch.setenv("CAPP_RUN_TAG", "t")
    t3 = transport(json.dumps({D.KEY: [8]}))
    llm_client.set_transport(t3)
    extract(with_dimensions=True)
    assert [c["dim"] for c in t3.seen] == [True]


def test_the_dimension_prompt_is_separate_and_asks_only_for_what_is_written():
    p = ife.DIMENSION_SYSTEM_PROMPT
    assert p != ife.VISION_SYSTEM_PROMPT and D.KEY in p
    assert "WRITTEN" in p and "Never estimate" in p and "thread callout" in p
    assert "{" in p and "feature_guide" not in p


def test_dimension_call_uses_the_pinned_model_and_deterministic_settings():
    t = transport(json.dumps({D.KEY: [8]}))
    llm_client.set_transport(t)
    extract(with_dimensions=True)
    call = [c for c in t.seen if c["dim"]][0]
    assert call["model"] == ife.VISION_MODEL and call["params"]["temperature"] == 0
    assert call["params"]["response_format"] == {"type": "json_object"}


def test_second_extraction_of_the_same_image_is_fully_cached():
    t = transport(json.dumps({D.KEY: [8]}))
    llm_client.set_transport(t)
    a = extract(with_dimensions=True)
    n = len(t.seen)
    b = extract(with_dimensions=True)
    assert len(t.seen) == n and a["hole_diameters"] == b["hole_diameters"] == [8.0]


@pytest.mark.parametrize("reply", ["not json at all", json.dumps({"features": ["Hole"]}), json.dumps([1, 2]), ""])
def test_unusable_dimension_reply_leaves_the_extraction_intact(reply):
    llm_client.set_transport(transport(reply))
    r = extract(with_dimensions=True)
    assert r["success"] and r["features"] == ["Hole", "Thread"]
    assert r["hole_diameters"] == [] and r["dimensions"]["success"] is False and r["dimensions"]["error"]


def test_a_failing_dimension_call_does_not_fail_the_extraction():
    llm_client.set_transport(transport(dim_raises=True))
    r = extract(with_dimensions=True)
    assert r["success"] and r["features"] == ["Hole", "Thread"]
    assert r["hole_diameters"] == [] and r["dimensions"]["success"] is False and "boom" in r["dimensions"]["error"]


def test_empty_list_is_a_successful_answer_not_an_error():
    llm_client.set_transport(transport(json.dumps({D.KEY: [], "note": "no diameters written"})))
    r = extract(with_dimensions=True)
    assert r["dimensions"]["success"] is True and r["hole_diameters"] == [] and r["dimensions"]["note"] == "no diameters written"


def test_inch_drawing_reply_is_an_empty_list():
    llm_client.set_transport(transport(json.dumps({D.KEY: [], "note": "drawing is in inches"})))
    assert extract(with_dimensions=True)["hole_diameters"] == []


def test_out_of_range_and_thread_values_never_reach_the_result():
    llm_client.set_transport(transport(json.dumps({D.KEY: ["M12", 0.1, 999, "Ø7"]})))
    r = extract(with_dimensions=True)
    assert r["hole_diameters"] == [7.0] and r["dimensions"]["dropped"] == ["M12", "0.1", "999"]


def test_failed_feature_extraction_makes_no_dimension_call(monkeypatch):
    def t(model, messages, params):
        raise RuntimeError("down")
    llm_client.set_transport(t)
    r = extract(with_dimensions=True)
    assert r["success"] is False and "hole_diameters" not in r
