import pytest

from capp import config


def test_default_models(monkeypatch):
    for role in ("PRIMARY", "SMALL", "LARGE"):
        monkeypatch.delenv("CAPP_MODEL_" + role, raising=False)
    assert config.model("primary") == "qwen/qwen3.8-27b"
    assert config.model("small") != config.model("large")


def test_model_override_from_env(monkeypatch):
    monkeypatch.setenv("CAPP_MODEL_PRIMARY", "some/other-model")
    assert config.model() == "some/other-model"


def test_unknown_model_role():
    with pytest.raises(KeyError):
        config.model("huge")


def test_mode_default_and_invalid(monkeypatch):
    monkeypatch.delenv("CAPP_LLM_MODE", raising=False)
    assert config.get_mode() == "live"
    monkeypatch.setenv("CAPP_LLM_MODE", "REPLAY")
    assert config.get_mode() == "replay"
    monkeypatch.setenv("CAPP_LLM_MODE", "banana")
    with pytest.raises(ValueError):
        config.get_mode()


def test_run_tag_and_cache_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("CAPP_RUN_TAG", raising=False)
    assert config.get_run_tag() == "default"
    monkeypatch.setenv("CAPP_RUN_TAG", "rep3")
    assert config.get_run_tag() == "rep3"
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path))
    assert config.get_cache_dir() == tmp_path


def test_planner_settings_match_v1():
    # baseline B0 depends on these; change them only on purpose
    assert config.PLANNER_TEMPERATURE == 0.3
    assert config.PLANNER_MAX_TOKENS == 200
