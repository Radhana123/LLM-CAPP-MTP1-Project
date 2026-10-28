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
    with pytest.raises(llm_client.ApiFailure) as info:
        llm_client.chat(MSGS)
    assert isinstance(info.value.__cause__, ConnectionError) and t.calls == llm_client.MAX_ATTEMPTS
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


# ---- patient retries, and no retry where it cannot help (Step 32) ---------------------------------------------
class HttpError(Exception):
    """Looks like an exception of the Groq SDK: a status code, and a response with headers."""

    def __init__(self, status, message="service said no", retry_after=None, via_response=False):
        super().__init__(message)
        headers = {} if retry_after is None else {"retry-after": str(retry_after)}
        resp = type("Resp", (), {"status_code": status, "headers": headers})()
        self.response = resp
        if not via_response:
            self.status_code = status


class Script:
    """A transport that raises or answers as told, one item per call; the last item repeats."""

    def __init__(self, *items):
        self.items, self.calls = list(items), 0

    def __call__(self, model, messages, params):
        item = self.items[min(self.calls, len(self.items) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return {"text": item, "prompt_tokens": 10, "completion_tokens": 5, "finish_reason": "stop"}


@pytest.fixture()
def waits(monkeypatch):
    seen = []
    monkeypatch.setattr(llm_client, "_sleep", seen.append)
    return seen


def test_the_waits_grow_and_are_the_ones_of_the_table(waits):
    t = use(Script(ConnectionError("down"), ConnectionError("down"), ConnectionError("down"), "ok"))
    assert llm_client.chat(MSGS).text == "ok" and t.calls == 4
    assert waits == [2, 5, 10] == list(llm_client.RETRY_DELAYS[:3])
    assert llm_client.MAX_ATTEMPTS == len(llm_client.RETRY_DELAYS) + 1 == 6


def test_a_call_that_never_works_makes_every_attempt_and_waits_about_a_minute_in_all(waits):
    t = use(Script(HttpError(503)))
    with pytest.raises(llm_client.ApiFailure) as info:
        llm_client.chat(MSGS)
    assert t.calls == 6 and waits == [2, 5, 10, 20, 40] and sum(waits) == 77
    e = info.value
    assert (e.status, e.attempts, e.retry_after) == (503, 6, None) and str(e) == "HTTP 503: service said no [after 6 attempts]"
    assert isinstance(e.__cause__, HttpError)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_wrong_request_key_or_model_is_not_retried(waits, status):
    t = use(Script(HttpError(status, "The model `x` does not exist")))
    with pytest.raises(llm_client.ApiFailure) as info:
        llm_client.chat(MSGS)
    assert t.calls == 1 and waits == [] and info.value.attempts == 1 and info.value.status == status
    assert str(info.value) == f"HTTP {status}: The model `x` does not exist"


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 529])
def test_every_other_status_is_retried(waits, status):
    t = use(Script(HttpError(status), "ok"))
    assert llm_client.chat(MSGS).text == "ok" and t.calls == 2 and waits == [2]


def test_the_status_is_read_from_the_response_when_the_exception_has_none_of_its_own(waits):
    t = use(Script(HttpError(404, via_response=True)))
    with pytest.raises(llm_client.ApiFailure):
        llm_client.chat(MSGS)
    assert t.calls == 1 and llm_client._status(HttpError(429, via_response=True)) == 429 and llm_client._status(ConnectionError("x")) is None
    assert llm_client._status(type("E", (Exception,), {"status_code": "404"})()) is None                 # a status that is not an int is no status


def test_a_retry_after_longer_than_our_wait_is_obeyed_a_shorter_one_is_not(waits):
    t = use(Script(HttpError(429, retry_after=30), HttpError(429, retry_after=1), "ok"))
    assert llm_client.chat(MSGS).text == "ok" and t.calls == 3 and waits == [30, 5]


def test_a_retry_after_beyond_the_limit_is_a_quota_and_fails_at_once_naming_the_wait(waits):
    t = use(Script(HttpError(429, "Rate limit reached for tokens per day", retry_after=3000)))
    with pytest.raises(llm_client.ApiFailure) as info:
        llm_client.chat(MSGS)
    assert t.calls == 1 and waits == [] and info.value.retry_after == 3000
    assert str(info.value) == "HTTP 429 (retry after 3000 s): Rate limit reached for tokens per day"
    t2 = use(Script(HttpError(429, retry_after=llm_client.MAX_RETRY_AFTER), "ok"))                      # exactly the limit is still waited for
    assert llm_client.chat([{"role": "user", "content": "other"}]).text == "ok" and t2.calls == 2 and waits == [120]


@pytest.mark.parametrize("value", ["soon", "", "-5", None])
def test_a_retry_after_that_is_not_a_number_of_seconds_is_ignored(value):
    e = HttpError(429)
    e.response.headers["retry-after"] = value
    assert llm_client._retry_after(e) is None
    assert llm_client._retry_after(ConnectionError("x")) is None and llm_client._retry_after(HttpError(429, retry_after=0)) == 0.0


def test_a_failure_without_a_status_names_the_kind_of_error():
    f = llm_client._failure(ConnectionError("Connection error."), None, None, 1)
    assert str(f) == "ConnectionError: Connection error." and f.status is None and f.attempts == 1
    assert str(llm_client._failure(TimeoutError("slow"), None, None, 6)) == "TimeoutError: slow [after 6 attempts]"
    assert str(llm_client._failure(HttpError(429), 429, 12.0, 3)) == "HTTP 429 (retry after 12 s): service said no [after 3 attempts]"


def test_no_api_key_is_never_retried_and_stays_the_same_exception(waits):
    use(Script(llm_client.NoApiKey("GROQ_API_KEY not configured")))
    with pytest.raises(llm_client.NoApiKey):
        llm_client.chat(MSGS)
    assert waits == [] and llm_client.summary()["failures"] == 0


def test_the_summary_counts_the_retries_the_time_waited_and_the_calls_that_failed_for_good(waits):
    use(Script(HttpError(429, retry_after=30), "ok"))
    llm_client.chat(MSGS)
    s = llm_client.summary()
    assert (s["retries"], s["retry_wait_s"], s["failures"]) == (1, 30.0, 0)
    use(Script(HttpError(404)))
    with pytest.raises(llm_client.ApiFailure):
        llm_client.chat([{"role": "user", "content": "x"}])
    s = llm_client.summary()
    assert (s["retries"], s["retry_wait_s"], s["failures"]) == (1, 30.0, 1)
    llm_client.reset_call_log()
    s = llm_client.summary()
    assert (s["retries"], s["retry_wait_s"], s["failures"], s["calls"]) == (0, 0.0, 0, 0)


def test_a_call_that_succeeds_after_retries_is_cached_and_logged_once(waits):
    t = use(Script(HttpError(503), HttpError(503), "ok"))
    first = llm_client.chat(MSGS)
    again = llm_client.chat(MSGS)
    assert (first.cached, again.cached, again.text, t.calls) == (False, True, "ok", 3)
    assert llm_client.summary()["calls"] == 2 and llm_client.summary()["retries"] == 2
