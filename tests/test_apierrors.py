"""Step 32: a call that failed because of the service is not a model failure (capp/apierrors.py, capp/llm_client.py, capp/harness.py).

Offline. The pieces are checked on small dicts shaped like harness records; the whole path is checked through the real planners, the real
llm_client (retries, cache) and the real harness with a transport that fails the way Groq does: rate limits that clear and that do not, a
retired model, a refused key. What they pin down: an API error is counted and named (not mixed into "the model did not parse"), a run stops
at the first error that cannot be cured by waiting and says what to change, and the same command with the same tag makes only the calls
that are missing.
"""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capp import apierrors as A, dataset as D, harness as H, llm_client, summary as SM      # noqa: E402

ROUTE = ["Facing", "Center Drilling", "Drilling", "Chamfering", "Inspection"]


class HttpError(Exception):
    def __init__(self, status, message="service said no", retry_after=None):
        super().__init__(message)
        self.status_code = status
        self.response = type("Resp", (), {"status_code": status, "headers": {} if retry_after is None else {"retry-after": str(retry_after)}})()


# ---------------------------------------------------------------- the text of an API error
def test_an_api_error_is_a_text_that_starts_with_the_prefix_of_the_planners():
    assert A.is_api_error("API Error: HTTP 503: unavailable") and A.is_api_error("API Error: boom") and A.is_api_error("API Error")
    for not_one in ("api error: x", " API Error: x", "Exception: x", "GROQ_API_KEY not configured", "", None, 5, ["API Error"]):
        assert not A.is_api_error(not_one)


@pytest.mark.parametrize("raw,expected", [
    ("API Error: HTTP 404: The model `x` does not exist", (404, None)),
    ("API Error: HTTP 429 (retry after 3000 s): Rate limit reached", (429, 3000)),
    ("API Error: HTTP 503: unavailable [after 6 attempts]", (503, None)),
    ("API Error: ConnectionError: Connection error. [after 6 attempts]", (None, None)),
    ("API Error: boom", (None, None)), ("HTTP 404: no prefix", (None, None)), ("", (None, None)), (None, (None, None)), (404, (None, None))])
def test_the_status_and_the_wait_are_read_back_from_the_text(raw, expected):
    assert A.status_of(raw) == expected


def test_what_llm_client_writes_is_what_apierrors_reads(monkeypatch):
    for status, wait in ((404, None), (429, 3000), (401, None), (503, None)):
        f = llm_client._failure(HttpError(status, "m"), status, float(wait) if wait else None, 1)
        assert A.status_of("API Error: " + str(f)) == (status, wait)
    assert A.status_of("API Error: " + str(llm_client._failure(ConnectionError("x"), None, None, 6))) == (None, None)


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_key_stops_the_run_and_points_at_the_env_file(status):
    why = A.run_fatal(f"API Error: HTTP {status}: invalid key")
    assert f"HTTP {status}" in why and "GROQ_API_KEY" in why and ".env" in why and "rotated" in why


def test_a_model_that_does_not_exist_stops_the_run_and_points_at_the_config():
    why = A.run_fatal("API Error: HTTP 404: The model `qwen/x` does not exist")
    assert "HTTP 404" in why and "retired" in why and "capp/config.py" in why and "CAPP_MODEL_PRIMARY" in why


def test_a_quota_stops_the_run_and_names_the_wait_and_the_way_to_continue():
    why = A.run_fatal("API Error: HTTP 429 (retry after 3000 s): tokens per day")
    assert "3000 s (about 50 min)" in why and "same --tag" in why and "cache" in why
    assert "about 2 min" in A.run_fatal(f"API Error: HTTP 429 (retry after {int(llm_client.MAX_RETRY_AFTER) + 1} s): x")


@pytest.mark.parametrize("raw", [
    "API Error: HTTP 429 (retry after 120 s): x", "API Error: HTTP 429 (retry after 0 s): x", "API Error: HTTP 429: no wait given",
    "API Error: HTTP 400: bad request", "API Error: HTTP 422: unprocessable", "API Error: HTTP 500: oops", "API Error: HTTP 503: down",
    "API Error: ConnectionError: Connection error.", "API Error: boom", "Exception: x", "empty reasoning", "", None])
def test_whatever_waiting_or_the_next_prompt_can_cure_does_not_stop_the_run(raw):
    assert A.run_fatal(raw) is None


# ---------------------------------------------------------------- counting them in a record
def cand(raw="", r2=None, small=None, large=None):
    c = {"llm_raw": raw, "repair": {"r2": {"raw_response": r2}} if r2 is not None else None}
    if small is not None or large is not None:
        c["cascade"] = {"small": small, "large": large}
    return c


E = "API Error: HTTP 503: down"


def test_a_candidate_counts_its_sample_call_its_reprompt_and_the_other_attempt_of_a_cascade_slot():
    assert A.in_candidate(cand("[]")) == 0 and A.in_candidate({}) == 0 and A.in_candidate(cand(E)) == 1
    assert A.in_candidate(cand("[]", r2=E)) == 1 and A.in_candidate(cand(E, r2=E)) == 2
    assert A.in_candidate(cand("[]", small=cand(E))) == 1 and A.in_candidate(cand("[]", large=cand(E, r2=E))) == 2
    assert A.in_candidate(cand(E, r2=E, small=cand(E), large=cand(E, r2=E))) == 5
    assert A.in_candidate({"llm_raw": "[]", "cascade": {"small": None, "large": "not a dict", "kept": "small"}}) == 0
    assert A.in_candidate({"llm_raw": "[]", "repair": {"r2": None}}) == 0 and A.in_candidate({"llm_raw": "[]", "repair": {}}) == 0
    assert A.in_candidate(cand("[]", r2="Exception: x")) == 0 and A.in_candidate(cand("Exception: x")) == 0


def rec(part, repeat, *cands):
    return {"part_id": part, "repeat": repeat, "candidates": list(cands)}


def test_records_are_counted_and_the_affected_parts_are_named_in_order():
    recs = [rec("P1", 0, cand("[]"), cand(E)), rec("P2", 0, cand("[]")), rec("P2", 1, cand(E), cand(E, r2=E)), rec("P3", 0)]
    assert A.in_records(recs) == 4 and A.affected(recs) == [("P1", 0), ("P2", 1)]
    assert A.in_records([]) == 0 and A.affected([]) == [] and A.in_records([{"part_id": "x"}]) == 0 and A.affected([{"part_id": "x", "repeat": 0}]) == []


def test_the_advice_names_the_number_and_the_way_out():
    text = A.rerun_advice(3)
    assert text.startswith("3 call(s) failed because of the service") and "same --tag" in text and "cache" in text and "counted as failures" in text


# ---------------------------------------------------------------- through the real planners, llm_client and harness
@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CAPP_LLM_MODE", raising=False)
    monkeypatch.delenv("CAPP_RUN_TAG", raising=False)
    monkeypatch.setattr(llm_client, "_sleep", lambda s: None)
    llm_client.reset_transport()
    llm_client.reset_call_log()
    yield
    llm_client.reset_transport()


class Service:
    """A transport that raises on the attempts listed (1-based, over all attempts) and answers with a valid route otherwise."""

    def __init__(self, fail=(), error=None):
        self.attempts, self.fail, self.error = 0, set(fail), error or (lambda: HttpError(503))

    def __call__(self, model, messages, params):
        self.attempts += 1
        if self.attempts in self.fail:
            raise self.error()
        return {"text": json.dumps(ROUTE), "prompt_tokens": 300, "completion_tokens": 20, "finish_reason": "stop"}


def parts(n=2):
    base = D.load_dataset("dev_v2")[0]
    return [dict(base, part_id=f"X{i}", features=["Hole", "Chamfer"], batch_size=100 + i) for i in range(n)]       # distinct prompts, distinct cache keys


def run(ps, tag="api"):
    return H.run_experiment("b0", ps, tag=tag, progress=None, write=False)


def test_a_call_that_fails_for_good_is_counted_as_an_api_error_and_not_as_a_model_that_did_not_parse():
    svc = Service(fail=range(1, 7))                                    # the first call fails on all six attempts
    llm_client.set_transport(svc)
    out = run(parts())
    m1 = out["summary"]["M1"]
    assert (m1["llm_calls"], m1["parsed_routes"], m1["api_errors"]) == (10, 9, 1) and svc.attempts == 15
    first = out["records"][0]["candidates"][0]
    assert first["llm_ok"] is False and first["llm_raw"].startswith("API Error: HTTP 503: service said no [after 6 attempts]")
    assert A.status_of(first["llm_raw"]) == (503, None) and A.affected(out["records"]) == [("X0", 1)]
    assert out["run_info"]["llm_client_summary"]["failures"] == 1 and out["run_info"]["llm_client_summary"]["retries"] == 5


def test_a_clean_run_has_zero_api_errors_not_a_missing_key():
    llm_client.set_transport(Service())
    out = run(parts())
    assert out["summary"]["M1"]["api_errors"] == 0 and out["run_info"]["llm_client_summary"]["retries"] == 0
    assert "API errors" not in SM.format_summary("b0", out["summary"])


def test_the_summary_text_names_the_api_errors_and_the_way_out_only_when_there_are_some():
    llm_client.set_transport(Service(fail=range(1, 7)))
    text = SM.format_summary("b0", run(parts())["summary"])
    line = next(l for l in text.split("\n") if l.startswith("API errors"))
    assert line.startswith("API errors      : 1. 1 call(s) failed because of the service") and "same --tag" in line


def test_the_same_command_again_with_the_same_tag_makes_only_the_missing_call():
    svc = Service(fail=range(1, 7))
    llm_client.set_transport(svc)
    first = run(parts())
    assert first["summary"]["M1"]["api_errors"] == 1 and svc.attempts == 15
    again = Service()
    llm_client.set_transport(again)
    second = run(parts())
    assert again.attempts == 1 and second["summary"]["M1"]["api_errors"] == 0 and second["summary"]["M1"]["parsed_routes"] == 10
    assert second["summary"]["M7"]["cache_hit_rate"] == 0.9
    third = Service()
    llm_client.set_transport(third)
    assert run(parts())["summary"]["M7"]["cache_hit_rate"] == 1.0 and third.attempts == 0


def test_a_rate_limit_that_clears_inside_the_retries_is_no_error_at_all(monkeypatch):
    waits = []
    monkeypatch.setattr(llm_client, "_sleep", waits.append)
    llm_client.set_transport(Service(fail=(1, 2), error=lambda: HttpError(429, retry_after=3)))
    out = run(parts())
    assert out["summary"]["M1"]["api_errors"] == 0 and out["summary"]["M1"]["parsed_routes"] == 10
    assert waits == [3, 5] and out["run_info"]["llm_client_summary"]["retries"] == 2 and out["run_info"]["llm_client_summary"]["retry_wait_s"] == 8.0


def test_a_model_that_does_not_exist_stops_the_run_at_the_first_call(monkeypatch):
    svc = Service(fail=range(1, 1000), error=lambda: HttpError(404, "The model `x` does not exist"))
    llm_client.set_transport(svc)
    with pytest.raises(H.HarnessError) as info:
        run(parts(3))
    msg = str(info.value)
    assert svc.attempts == 1 and "the model was not found (HTTP 404)" in msg and "capp/config.py" in msg and "The answers made so far are in the cache." in msg
    assert "Last: API Error: HTTP 404: The model `x` does not exist" in msg


def test_a_refused_key_stops_the_run_at_the_first_call():
    svc = Service(fail=range(1, 1000), error=lambda: HttpError(401, "Invalid API Key"))
    llm_client.set_transport(svc)
    with pytest.raises(H.HarnessError, match="the API key was refused \\(HTTP 401\\)"):
        run(parts(3))
    assert svc.attempts == 1


def test_a_daily_quota_stops_the_run_at_once_and_names_the_wait():
    svc = Service(fail=range(1, 1000), error=lambda: HttpError(429, "tokens per day", retry_after=3000))
    llm_client.set_transport(svc)
    with pytest.raises(H.HarnessError) as info:
        run(parts(3))
    assert svc.attempts == 1 and "3000 s (about 50 min)" in str(info.value) and "same --tag" in str(info.value)


def test_a_request_the_service_refuses_once_is_an_api_error_and_the_run_goes_on():
    svc = Service(fail=(3,), error=lambda: HttpError(400, "bad request for this prompt"))
    llm_client.set_transport(svc)
    out = run(parts())
    assert out["summary"]["M1"]["api_errors"] == 1 and out["summary"]["M1"]["parsed_routes"] == 9 and svc.attempts == 10 and len(out["records"]) == 2
    bad = [c for r in out["records"] for c in r["candidates"] if A.is_api_error(c["llm_raw"])]
    assert len(bad) == 1 and bad[0]["llm_raw"].startswith("API Error: HTTP 400: bad request for this prompt") and "[after" not in bad[0]["llm_raw"]


def test_a_long_outage_stops_the_run_after_five_failed_calls_in_a_row_about_six_minutes_not_half_an_hour():
    svc = Service(fail=range(1, 10_000))                                # 503 on every attempt: nothing is fatal, but nothing works
    llm_client.set_transport(svc)
    with pytest.raises(H.HarnessError, match="5 API errors in a row"):
        run(parts(6))
    assert H.MAX_CONSECUTIVE_API_ERRORS == 5 and svc.attempts == 5 * llm_client.MAX_ATTEMPTS
    assert 5 * sum(llm_client.RETRY_DELAYS) < 7 * 60


def test_a_success_between_failures_starts_the_count_again():
    state = {"fail_next": True, "tries": 0, "attempts": 0}

    def alternate(model, messages, params):                              # one call fails on all its attempts, the next works, and so on
        state["attempts"] += 1
        if state["fail_next"]:
            state["tries"] += 1
            if state["tries"] == llm_client.MAX_ATTEMPTS:
                state["fail_next"], state["tries"] = False, 0
            raise HttpError(503)
        state["fail_next"] = True
        return {"text": json.dumps(ROUTE), "prompt_tokens": 300, "completion_tokens": 20, "finish_reason": "stop"}
    llm_client.set_transport(alternate)
    out = run(parts(4))                                                  # 20 calls: 10 fail for good, never five in a row
    assert len(out["records"]) == 4 and out["summary"]["M1"]["api_errors"] == 10 and out["summary"]["M1"]["parsed_routes"] == 10


def test_run_eval_says_to_run_the_same_command_again_only_when_the_run_has_api_errors(monkeypatch, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_eval_cli", os.path.join(ROOT, "scripts", "run_eval.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli.summary, "format_summary", lambda system, s: "THE SUMMARY")
    for errors in (2, 0):
        monkeypatch.setattr(cli.harness, "run_experiment", lambda *a, errors=errors, **k: {"summary": {"M1": {"api_errors": errors}}, "out_dir": "somewhere"})
        monkeypatch.setattr(sys, "argv", ["run_eval.py", "--system", "rb", "--limit", "2"])
        cli.main()
        out = capsys.readouterr()
        assert "THE SUMMARY" in out.out and "Files: somewhere" in out.out
        assert ("This run has API errors (see above): run the same command again, same --tag, to fill them." in out.err) == bool(errors)
