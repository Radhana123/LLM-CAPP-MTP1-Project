import pytest

from capp import config, llm_client

MSGS = [{"role": "user", "content": "hello"}]


class FakeTransport:
    """Counts calls; the reply text is 'reply-<n>' so different live calls are distinguishable."""

    def __init__(self, fail_times=0, finish_reason="stop"):
        self.calls = 0
        self.fail_times = fail_times
        self.finish_reason = finish_reason

    def __call__(self, model, messages, params):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("boom")
        return {"text": "reply-%d" % self.calls, "prompt_tokens": 10, "completion_tokens": 5,
                "finish_reason": self.finish_reason}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CAPP_LLM_MODE", "live")
    monkeypatch.setenv("CAPP_RUN_TAG", "t")
    monkeypatch.setattr(llm_client, "_sleep", lambda s: None)
    llm_client.reset_call_log()
    yield
    llm_client.reset_transport()


def use(transport):
    llm_client.set_transport(transport)
    return transport


def test_second_identical_call_is_served_from_cache():
    t = use(FakeTransport())
    a = llm_client.chat(MSGS, temperature=0.3)
    b = llm_client.chat(MSGS, temperature=0.3)
    assert t.calls == 1
    assert (a.cached, b.cached) == (False, True)
    assert a.text == b.text == "reply-1"
    assert b.completion_tokens == 5


def test_sample_ids_are_different_samples_but_each_reproducible():
    t = use(FakeTransport())
    s0 = llm_client.chat(MSGS, sample_id=0)
    s1 = llm_client.chat(MSGS, sample_id=1)
    s0_again = llm_client.chat(MSGS, sample_id=0)
    assert t.calls == 2
    assert s0.text != s1.text
    assert s0_again.text == s0.text and s0_again.cached


def test_run_tag_and_params_change_the_key(monkeypatch):
    t = use(FakeTransport())
    llm_client.chat(MSGS, temperature=0.3)
    monkeypatch.setenv("CAPP_RUN_TAG", "rep2")
    llm_client.chat(MSGS, temperature=0.3)
    llm_client.chat(MSGS, temperature=0.0)
    assert t.calls == 3


def test_replay_mode_serves_cache_offline_and_raises_on_miss(monkeypatch):
    use(FakeTransport())
    llm_client.chat(MSGS)
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    use(FakeTransport(fail_times=99))          # network must not be touched
    assert llm_client.chat(MSGS).text == "reply-1"
    with pytest.raises(llm_client.CacheMiss):
        llm_client.chat([{"role": "user", "content": "never asked"}])


def test_nocache_mode_always_calls(monkeypatch):
    monkeypatch.setenv("CAPP_LLM_MODE", "nocache")
    t = use(FakeTransport())
    llm_client.chat(MSGS)
    llm_client.chat(MSGS)
    assert t.calls == 2
    assert not config.get_cache_dir().exists()


def test_retries_then_succeeds():
    t = use(FakeTransport(fail_times=2))
    assert llm_client.chat(MSGS).text == "reply-3"
    assert t.calls == 3


def test_failure_after_retries_raises_and_is_not_cached():
    t = use(FakeTransport(fail_times=99))
    with pytest.raises(ConnectionError):
        llm_client.chat(MSGS)
    assert t.calls == llm_client.MAX_ATTEMPTS
    t.fail_times = 0
    assert not llm_client.chat(MSGS).cached        # the failure left nothing in the cache


def test_missing_api_key_raises(monkeypatch):
    llm_client.reset_transport()
    monkeypatch.setattr(config, "get_api_key", lambda: None)
    with pytest.raises(llm_client.NoApiKey):
        llm_client.chat(MSGS)


def test_summary_counts_calls_tokens_and_truncation():
    use(FakeTransport(finish_reason="length"))
    llm_client.chat(MSGS, sample_id=0)
    llm_client.chat(MSGS, sample_id=0)       # cache hit
    llm_client.chat(MSGS, sample_id=1)
    s = llm_client.summary()
    assert s["calls"] == 3 and s["cache_hits"] == 1
    assert s["prompt_tokens"] == 30 and s["completion_tokens"] == 15
    assert s["truncated"] == 3
    assert s["latency_p95_s"] >= s["latency_p50_s"] >= 0
