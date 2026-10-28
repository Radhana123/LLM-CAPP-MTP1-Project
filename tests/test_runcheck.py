"""Step 39: capp/runcheck.py, the two questions about a finished run (cut-off answers, model settings of today) shared by scripts/g1.py and the grid.

Offline, on small run.json dicts. What is pinned: what counts as a cut-off number, which models of a run are looked at (its own and both cascade
tiers), that a missing model_params means 'nothing extra', that today's settings decide (also with CAPP_QWEN_REASONING), and the wording.
"""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capp import config, runcheck as RC      # noqa: E402

QWEN, SMALL, LARGE = "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"
NOW_QWEN = {"reasoning_effort": "none"}
NOW_OSS = {"reasoning_effort": "low", "max_tokens_at_least": 2048}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("CAPP_QWEN_REASONING", raising=False)


# ---- truncated -----------------------------------------------------------------------------------------------------------------
def test_truncated_is_the_number_the_client_counted():
    assert RC.truncated({"llm_client_summary": {"truncated": 96}}) == 96
    assert RC.truncated({"llm_client_summary": {"truncated": 1}}) == 1
    assert RC.truncated({"llm_client_summary": {"truncated": 0}}) == 0


@pytest.mark.parametrize("info", [None, [], "x", {}, {"llm_client_summary": None}, {"llm_client_summary": []}, {"llm_client_summary": {}},
                                  {"llm_client_summary": {"truncated": None}}, {"llm_client_summary": {"truncated": True}},
                                  {"llm_client_summary": {"truncated": "7"}}, {"llm_client_summary": {"truncated": -2}},
                                  {"llm_client_summary": {"truncated": 2.5}}, {"llm_client_summary": {"truncated": [3]}}])
def test_anything_that_is_not_a_positive_whole_number_is_not_a_cut_off(info):
    assert RC.truncated(info) == 0


# ---- models_of -------------------------------------------------------------------------------------------------------------------
def test_the_models_of_a_run_are_its_own_and_both_cascade_tiers_once():
    assert RC.models_of({"model": QWEN}) == [QWEN]
    assert RC.models_of({"model": SMALL, "cascade_models": {"small": SMALL, "large": LARGE}}) == [SMALL, LARGE]
    assert RC.models_of({"model": QWEN, "cascade_models": {"small": SMALL, "large": LARGE}}) == [QWEN, SMALL, LARGE]
    assert RC.models_of({"model": SMALL, "cascade_models": {"small": SMALL, "large": SMALL}}) == [SMALL]


@pytest.mark.parametrize("info", [None, [], {}, {"model": None}, {"model": 5}, {"model": None, "cascade_models": {"small": SMALL}}])
def test_a_run_without_a_model_has_none_to_look_at(info):
    assert RC.models_of(info) == []


def test_a_cascade_entry_that_is_not_a_model_name_is_ignored():
    assert RC.models_of({"model": SMALL, "cascade_models": {"small": None, "large": 7}}) == [SMALL]
    assert RC.models_of({"model": SMALL, "cascade_models": "x"}) == [SMALL]


# ---- stale_settings ----------------------------------------------------------------------------------------------------------------
def test_a_run_made_with_the_settings_of_today_is_not_stale():
    assert RC.stale_settings({"model": QWEN, "model_params": {QWEN: NOW_QWEN}}) == []
    assert RC.stale_settings({"model": SMALL, "model_params": {SMALL: NOW_OSS}}) == []


def test_no_model_params_means_nothing_extra_which_is_stale_for_qwen_and_fine_for_a_model_that_needs_none():
    assert RC.stale_settings({"model": QWEN}) == [(QWEN, {}, NOW_QWEN)]
    assert RC.stale_settings({"model": QWEN, "model_params": {QWEN: {}}}) == [(QWEN, {}, NOW_QWEN)]
    assert RC.stale_settings({"model": QWEN, "model_params": None}) == [(QWEN, {}, NOW_QWEN)]
    assert RC.stale_settings({"model": QWEN, "model_params": "x"}) == [(QWEN, {}, NOW_QWEN)]
    assert RC.stale_settings({"model": "llama-3.3-70b-versatile"}) == []
    assert RC.stale_settings({"model": "llama-3.3-70b-versatile", "model_params": {"llama-3.3-70b-versatile": {}}}) == []


def test_the_settings_of_another_model_do_not_stand_in_for_this_one():
    assert RC.stale_settings({"model": QWEN, "model_params": {SMALL: NOW_OSS}}) == [(QWEN, {}, NOW_QWEN)]


def test_a_cascade_is_stale_per_tier():
    info = {"model": SMALL, "cascade_models": {"small": SMALL, "large": LARGE}, "model_params": {SMALL: NOW_OSS, LARGE: {}}}
    assert RC.stale_settings(info) == [(LARGE, {}, NOW_OSS)]
    info["model_params"] = {}
    assert [m for m, _, _ in RC.stale_settings(info)] == [SMALL, LARGE]
    info["model_params"] = {SMALL: NOW_OSS, LARGE: NOW_OSS}
    assert RC.stale_settings(info) == []


def test_a_run_without_a_model_is_never_stale():
    assert RC.stale_settings({"model": None, "model_params": None}) == []
    assert RC.stale_settings({}) == [] and RC.stale_settings(None) == []


def test_the_setting_of_today_decides(monkeypatch):
    info = {"model": QWEN, "model_params": {QWEN: NOW_QWEN}}
    monkeypatch.setenv("CAPP_QWEN_REASONING", "omit")
    assert RC.stale_settings(info) == [(QWEN, NOW_QWEN, {})]
    assert RC.stale_settings({"model": QWEN}) == []
    monkeypatch.setenv("CAPP_QWEN_REASONING", "low")
    assert RC.stale_settings(info) == [(QWEN, NOW_QWEN, {"reasoning_effort": "low", "max_tokens_at_least": 4096})]


def test_a_wrong_qwen_setting_raises_in_words_only_for_a_qwen_run(monkeypatch):
    monkeypatch.setenv("CAPP_QWEN_REASONING", "xhigh")
    with pytest.raises(ValueError, match="CAPP_QWEN_REASONING='xhigh'"):
        RC.stale_settings({"model": QWEN})
    assert RC.stale_settings({"model": SMALL, "model_params": {SMALL: NOW_OSS}}) == []


def test_the_settings_are_read_from_config_not_copied(monkeypatch):
    monkeypatch.setattr(config, "REASONING_EFFORT", "medium")
    assert RC.stale_settings({"model": SMALL, "model_params": {SMALL: NOW_OSS}}) == [(SMALL, NOW_OSS, {"reasoning_effort": "medium", "max_tokens_at_least": 2048})]


# ---- the words ----------------------------------------------------------------------------------------------------------------------
def test_settings_text_is_sorted_and_says_nothing_extra_for_none():
    assert RC.settings_text({}) == "nothing extra" and RC.settings_text(None) == "nothing extra"
    assert RC.settings_text({"reasoning_effort": "low", "max_tokens_at_least": 2048}) == "max_tokens_at_least=2048, reasoning_effort=low"


def test_stale_text_names_the_model_then_and_now_and_joins_several_with_semicolons():
    one = [(QWEN, {}, NOW_QWEN)]
    assert RC.stale_text(one) == "qwen/qwen3.8-27b: then nothing extra, now reasoning_effort=none"
    two = one + [(SMALL, {"reasoning_effort": "high"}, NOW_OSS)]
    assert RC.stale_text(two) == ("qwen/qwen3.8-27b: then nothing extra, now reasoning_effort=none; "
                                  "openai/gpt-oss-20b: then reasoning_effort=high, now max_tokens_at_least=2048, reasoning_effort=low")
    assert RC.stale_text([]) == ""
