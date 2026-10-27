"""Step 20 (WP7): what the extractor (week1/image_feature_extractor.py) now reports per pass, and the opt-in evidence prompt.

A fake transport stands in for the API (as in tests/test_dimensions.py): no network. The point of several tests here is that the
DEFAULT call is the one of before Step 20: same prompt, same messages, same parameters, so the cache of earlier runs and the recall
measured with it stay valid.
"""
import hashlib
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "week1"))

from capp import extraction_review as ER, llm_client      # noqa: E402
import image_feature_extractor as ife                      # noqa: E402

# SHA-256 (first 16 hex digits) of VISION_SYSTEM_PROMPT as it was before Step 20. If this changes, every cached extraction of the
# earlier runs is lost and the recall figures of the report no longer describe the prompt that is used.
DEFAULT_PROMPT_SHA = "33c64a8869bd6e13"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CAPP_LLM_MODE", "live")
    monkeypatch.setenv("CAPP_RUN_TAG", "t")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm_client, "_sleep", lambda s: None)
    llm_client.reset_call_log()
    yield
    llm_client.reset_transport()


def pass_of(messages):
    text = messages[0]["content"][0]["text"]
    return "geometry" if "DRAWN GEOMETRY" in text else "annotation" if "TEXT AND SYMBOLS" in text else "other"


def transport(replies, raises=()):
    """replies: {pass name: dict or str}. Records every call."""
    seen = []

    def t(model, messages, params):
        name = pass_of(messages)
        seen.append({"pass": name, "messages": messages, "params": params, "model": model, "prompt": messages[0]["content"][0]["text"]})
        if name in raises:
            raise RuntimeError("boom")
        r = replies[name]
        return {"text": r if isinstance(r, str) else json.dumps(r), "prompt_tokens": 10, "completion_tokens": 5, "finish_reason": "stop"}
    t.seen = seen
    return t


GEOMETRY = {"features": ["Hole", "Slot"], "confidence": "high", "notes": "front view shows a slot",
            "evidence": {"Hole": "circle in the front view", "Slot": "long rectangle on top"}}
ANNOTATION = {"features": ["Hole", "Thread", "Counter Bore"], "confidence": "medium", "notes": "note M10 and a counter bore",
              "evidence": {"Hole": "dia 8 callout", "Thread": "note M10", "Counter Bore": "CBORE note"}}


def extract(**kw):
    return ife.extract_features_from_image(b"not-a-real-image", "png", **kw)


# ---------------------------------------------------------------- the default prompt did not change
def test_the_default_prompt_is_the_one_of_before_step_20():
    assert hashlib.sha256(ife.VISION_SYSTEM_PROMPT.encode()).hexdigest()[:16] == DEFAULT_PROMPT_SHA


def test_the_default_prompt_never_mentions_evidence():
    assert "evidence" not in ife.VISION_SYSTEM_PROMPT.lower()


def test_a_default_call_is_the_same_with_and_without_the_option_spelled_out(monkeypatch):
    t1 = transport({"geometry": GEOMETRY, "annotation": ANNOTATION})
    llm_client.set_transport(t1)
    extract()
    monkeypatch.setenv("CAPP_RUN_TAG", "second")                         # a new tag: the cache of the first call is not used
    t2 = transport({"geometry": GEOMETRY, "annotation": ANNOTATION})
    llm_client.set_transport(t2)
    extract(with_evidence=False)
    one = [(c["messages"], c["params"], c["model"]) for c in t1.seen]
    two = [(c["messages"], c["params"], c["model"]) for c in t2.seen]
    assert len(one) == len(two) == len(ife.EXTRACTION_PASSES) and one == two


def test_a_default_call_sends_the_default_prompt_text():
    t = transport({"geometry": GEOMETRY, "annotation": ANNOTATION})
    llm_client.set_transport(t)
    extract()
    assert {c["pass"] for c in t.seen} == {"geometry", "annotation"} and all("evidence" not in c["prompt"].lower() for c in t.seen)
    assert all("5. Output ONLY a JSON object" in c["prompt"] for c in t.seen)


# ---------------------------------------------------------------- what each pass reported
def test_the_result_carries_what_each_pass_reported():
    llm_client.set_transport(transport({"geometry": GEOMETRY, "annotation": ANNOTATION}))
    r = extract()
    assert r["success"] and r["passes_ok"] == 2
    assert r["features"] == ["Hole", "Slot", "Thread"] and r["rejected_features"] == ["Counter Bore"]
    g, a = r["passes"]["geometry"], r["passes"]["annotation"]
    assert g["success"] and g["features"] == ["Hole", "Slot"] and g["rejected_features"] == [] and g["confidence"] == "high"
    assert a["success"] and a["features"] == ["Hole", "Thread"] and a["rejected_features"] == ["Counter Bore"] and a["confidence"] == "medium"
    assert g["notes"] == "front view shows a slot"
    assert r["sample_agreement"] == {"Hole": 2, "Slot": 1, "Thread": 1}


def test_a_default_call_gives_no_evidence_even_if_the_model_volunteers_some():
    llm_client.set_transport(transport({"geometry": GEOMETRY, "annotation": ANNOTATION}))
    r = extract()
    assert r["passes"]["geometry"]["evidence"] == {} and r["passes"]["annotation"]["evidence"] == {}


def test_a_failed_pass_is_recorded_and_the_other_one_still_counts():
    llm_client.set_transport(transport({"geometry": GEOMETRY, "annotation": ANNOTATION}, raises=("annotation",)))
    r = extract()
    assert r["success"] and r["passes_ok"] == 1 and r["features"] == ["Hole", "Slot"]
    assert r["passes"]["geometry"]["success"] is True
    assert r["passes"]["annotation"]["success"] is False and "boom" in r["passes"]["annotation"]["error"]
    assert "features" not in r["passes"]["annotation"]


def test_both_passes_failing_is_a_failed_extraction():
    llm_client.set_transport(transport({"geometry": GEOMETRY, "annotation": ANNOTATION}, raises=("annotation", "geometry")))
    r = extract()
    assert r["success"] is False and r["features"] == []


def test_the_review_of_a_real_extraction_result():
    llm_client.set_transport(transport({"geometry": GEOMETRY, "annotation": ANNOTATION}))
    rv = ER.build_review(extract())
    assert rv["preaccepted"] == ["Hole"] and rv["flagged"] == ["Slot", "Thread"]
    assert [u["name"] for u in rv["unrecognised"]] == ["Counter Bore"] and rv["unrecognised"][0]["suggestion"] == "Counterbore"
    assert rv["unrecognised"][0]["seen_by"] == ["annotation"]


def test_the_review_after_one_pass_failed_flags_everything():
    llm_client.set_transport(transport({"geometry": GEOMETRY, "annotation": ANNOTATION}, raises=("annotation",)))
    rv = ER.build_review(extract())
    assert rv["preaccepted"] == [] and rv["flagged"] == ["Hole", "Slot"] and rv["notes"]


# ---------------------------------------------------------------- the evidence variant
def test_the_evidence_variant_is_a_separate_prompt_that_asks_for_evidence():
    p = ife.VISION_SYSTEM_PROMPT_EVIDENCE
    assert p != ife.VISION_SYSTEM_PROMPT and "evidence" in p
    assert p.startswith(ife.VISION_SYSTEM_PROMPT[:ife.VISION_SYSTEM_PROMPT.index("5. Output ONLY")])       # same task and rules 1-4
    assert "Never invent text" in p


def test_the_evidence_prompt_formats_and_keeps_its_json_braces():
    text = ife.VISION_SYSTEM_PROMPT_EVIDENCE.format(feature_guide="- Hole: a circle", focus="Focus on X")
    assert "- Hole: a circle" in text and "Focus on X" in text
    assert '{"features": ["Hole", "Thread", ...]' in text and '"evidence": {"Hole":' in text


def test_the_evidence_variant_sends_the_evidence_prompt_to_both_passes():
    t = transport({"geometry": GEOMETRY, "annotation": ANNOTATION})
    llm_client.set_transport(t)
    extract(with_evidence=True)
    assert len(t.seen) == 2 and all("ONE short evidence phrase" in c["prompt"] for c in t.seen)
    assert {c["pass"] for c in t.seen} == {"geometry", "annotation"}


def test_the_evidence_variant_returns_the_evidence_per_pass():
    llm_client.set_transport(transport({"geometry": GEOMETRY, "annotation": ANNOTATION}))
    r = extract(with_evidence=True)
    assert r["passes"]["geometry"]["evidence"] == {"Hole": "circle in the front view", "Slot": "long rectangle on top"}
    # "Counter Bore" is not a vocabulary feature: its evidence is dropped with it
    assert r["passes"]["annotation"]["evidence"] == {"Hole": "dia 8 callout", "Thread": "note M10"}
    rows = {x["feature"]: x for x in ER.build_review(r)["rows"]}
    assert rows["Hole"]["evidence_text"] == "geometry: circle in the front view; annotation: dia 8 callout"
    assert rows["Thread"]["evidence_text"] == "annotation: note M10"


def test_the_features_do_not_depend_on_the_evidence_option(monkeypatch):
    llm_client.set_transport(transport({"geometry": GEOMETRY, "annotation": ANNOTATION}))
    plain = extract()
    monkeypatch.setenv("CAPP_RUN_TAG", "evidence-run")
    ev = extract(with_evidence=True)
    assert plain["features"] == ev["features"] and plain["sample_agreement"] == ev["sample_agreement"]


def test_a_reply_without_evidence_to_the_evidence_prompt_still_works():
    llm_client.set_transport(transport({"geometry": {"features": ["Hole"]}, "annotation": {"features": ["Hole"]}}))
    r = extract(with_evidence=True)
    assert r["success"] and r["features"] == ["Hole"] and r["passes"]["geometry"]["evidence"] == {}


def test_the_two_prompts_have_separate_cache_entries():
    t = transport({"geometry": GEOMETRY, "annotation": ANNOTATION})
    llm_client.set_transport(t)
    extract()
    assert len(t.seen) == 2
    extract()
    assert len(t.seen) == 2                       # a second default call is served from the cache
    extract(with_evidence=True)
    assert len(t.seen) == 4                       # the evidence prompt is a new request
    extract(with_evidence=True)
    assert len(t.seen) == 4
    extract()
    assert len(t.seen) == 4                       # and the default entries are still there


# ---------------------------------------------------------------- cleaning the evidence
VALID = {"Hole", "Slot", "Thread"}


def test_clean_evidence_keeps_text_for_kept_features_only():
    out = ife._clean_evidence({"Hole": "circle", "Gear": "teeth", "Slot": "rectangle"}, VALID)
    assert out == {"Hole": "circle", "Slot": "rectangle"}


@pytest.mark.parametrize("raw", [None, "text", ["Hole"], 5, {}])
def test_clean_evidence_of_anything_that_is_not_a_mapping_is_empty(raw):
    assert ife._clean_evidence(raw, VALID) == {}


def test_clean_evidence_drops_empty_and_non_text_values():
    assert ife._clean_evidence({"Hole": "", "Slot": "   ", "Thread": 5}, VALID) == {}
    assert ife._clean_evidence({"Hole": None, "Slot": ["x"]}, VALID) == {}


def test_clean_evidence_collapses_white_space_and_truncates():
    out = ife._clean_evidence({"Hole": "  circle \n in   the\tview  ", "Slot": "x" * 500}, VALID)
    assert out["Hole"] == "circle in the view" and len(out["Slot"]) == ife.EVIDENCE_MAX_CHARS == 160
