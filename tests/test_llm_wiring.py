"""The planner, the extractor and NSGA-II must go through capp.llm_client (cache + sample ids)."""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for d in ("week1", "week2", "week4"):
    sys.path.insert(0, os.path.join(ROOT, d))

from capp import config, llm_client
import llm_planner
import image_feature_extractor as ife
from nsga2 import run_nsga2
from route_builder import generate_valid_routes_simple

FEATURES = ["Thread", "Fillet"]


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CAPP_LLM_MODE", "live")
    monkeypatch.setenv("CAPP_RUN_TAG", "t")
    monkeypatch.setattr(llm_client, "_sleep", lambda s: None)
    llm_client.reset_call_log()
    yield
    llm_client.reset_transport()


def transport_returning(text):
    seen = []

    def t(model, messages, params):
        seen.append({"model": model, "params": params})
        return {"text": text, "prompt_tokens": 100, "completion_tokens": 20, "finish_reason": "stop"}
    t.seen = seen
    return t


def test_planner_parses_reply_and_uses_pinned_model_and_v1_settings():
    t = transport_returning('["Facing", "Drilling", "Inspection"]')
    llm_client.set_transport(t)
    r = llm_planner.generate_process_plan("Aluminum", ["Hole"], "0.02mm", 500)
    assert r["success"] and r["steps"] == ["Facing", "Drilling", "Inspection"]
    assert t.seen[0]["model"] == config.model("primary") == llm_planner.MODEL
    assert t.seen[0]["params"] == {"temperature": 0.3, "max_tokens": 200}
    assert r["meta"]["cached"] is False and r["meta"]["completion_tokens"] == 20


def test_planner_same_sample_is_cached_other_sample_is_not():
    t = transport_returning('["Facing", "Inspection"]')
    llm_client.set_transport(t)
    args = ("Aluminum", ["Hole"], "0.02mm", 500)
    llm_planner.generate_process_plan(*args, sample_id=0)
    again = llm_planner.generate_process_plan(*args, sample_id=0)
    llm_planner.generate_process_plan(*args, sample_id=1)
    assert len(t.seen) == 2 and again["meta"]["cached"] is True


def test_planner_without_key_reports_not_configured(monkeypatch):
    llm_client.reset_transport()
    monkeypatch.setattr(config, "get_api_key", lambda: None)
    r = llm_planner.generate_process_plan("Aluminum", ["Hole"])
    assert r["success"] is False and "not configured" in r["raw_response"]


def test_nsga2_five_attempts_are_five_distinct_samples():
    valid = generate_valid_routes_simple(FEATURES, max_routes=1)[0]
    t = transport_returning(json.dumps(valid))
    llm_client.set_transport(t)
    run_nsga2("Aluminum", 500, features=FEATURES, llm_attempts=5)
    assert len(t.seen) == 5                       # not 1 live call + 4 cache hits
    assert run_nsga2.last_llm_status["llm_valid"] == 5


def test_extractor_goes_through_client_one_call_per_pass():
    t = transport_returning(json.dumps({"features": ["Hole", "NotAFeature"], "confidence": "high", "notes": ""}))
    llm_client.set_transport(t)
    r = ife.extract_features_from_image(b"not-a-real-image", "png")
    assert r["success"] and r["features"] == ["Hole"] and r["rejected_features"] == ["NotAFeature"]
    assert len(t.seen) == len(ife.EXTRACTION_PASSES)
    assert t.seen[0]["model"] == ife.VISION_MODEL
    assert t.seen[0]["params"]["temperature"] == 0
