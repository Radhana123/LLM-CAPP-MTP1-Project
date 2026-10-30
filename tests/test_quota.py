"""Step 44: the Groq quota (tokens per day) on the free plan.

CAPP_QUOTA_WAIT lets a call sit out a quota 429 instead of failing (capp/llm_client.py, capp/config.py quota_wait); the stop message names the
limit Groq says was reached and how to wait (capp/apierrors.py); scripts/g2.py shows the tokens of a column and what they are in days of the
quota. Offline: the transport is a script, the sleep is recorded.
"""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from capp import apierrors as A, config, g2 as G, harness, llm_client, runplan        # noqa: E402
import run_eval                                                                         # noqa: E402
import test_g2 as TG                                                                    # noqa: E402
from test_llm_client import HttpError, Script, MSGS                                     # noqa: E402

QUOTA_429 = ("Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_x` service tier `on_demand` on tokens per day (TPD): "
             "Limit 200000, Used 199700, Requested 660. Please try again in 4m46s.")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    for name in ("GROQ_API_KEY", "CAPP_QWEN_REASONING", "CAPP_MODEL_PRIMARY", "CAPP_QUOTA_WAIT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CAPP_LLM_MODE", "live")
    monkeypatch.setenv("CAPP_RUN_TAG", "t")
    llm_client.reset_call_log()
    yield
    llm_client.reset_transport()


@pytest.fixture()
def waits(monkeypatch):
    seen = []
    monkeypatch.setattr(llm_client, "_sleep", seen.append)
    return seen


@pytest.fixture()
def notices(monkeypatch):
    seen = []
    monkeypatch.setattr(llm_client, "_notify", seen.append)
    return seen


def use(transport):
    llm_client.set_transport(transport)
    return transport


# ================================================================ config.quota_wait
def test_the_quota_wait_is_off_unless_the_variable_is_set(monkeypatch):
    assert config.quota_wait() == 0.0
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "  ")
    assert config.quota_wait() == 0.0
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "900")
    assert config.quota_wait() == 900.0
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "0")
    assert config.quota_wait() == 0.0
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "7200")
    assert config.quota_wait() == 7200.0                                                  # exactly the limit is allowed


@pytest.mark.parametrize("bad", ["soon", "-1", "7201", "nan", "inf", "15 min"])
def test_a_wrong_quota_wait_names_the_variable_and_what_is_expected(bad, monkeypatch):
    monkeypatch.setenv("CAPP_QUOTA_WAIT", bad)
    with pytest.raises(ValueError, match="CAPP_QUOTA_WAIT=.*number of seconds from 0 to 7200"):
        config.quota_wait()


def test_the_free_quota_figure_is_the_one_of_the_limits_page():
    assert config.QUOTA_TOKENS_PER_DAY == 200_000 and config.QUOTA_WAIT_LIMIT == 7200


# ================================================================ llm_client: sitting out a quota wait
def test_without_the_variable_a_quota_429_fails_at_once_as_before(waits, notices):
    t = use(Script(HttpError(429, QUOTA_429, retry_after=286)))
    with pytest.raises(llm_client.ApiFailure) as info:
        llm_client.chat(MSGS)
    assert t.calls == 1 and waits == [] and notices == [] and info.value.retry_after == 286


def test_with_the_variable_the_call_waits_the_hint_plus_a_margin_and_then_goes_through(waits, notices, monkeypatch):
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "900")
    t = use(Script(HttpError(429, QUOTA_429, retry_after=286), "ok"))
    assert llm_client.chat(MSGS).text == "ok" and t.calls == 2
    assert waits == [286 + llm_client.QUOTA_MARGIN_S] == [291.0]
    assert len(notices) == 1 and config.model("primary") in notices[0] and "291 s" in notices[0] and "1 wait(s)" in notices[0] and "Ctrl-C" in notices[0]


def test_the_notice_names_the_model(waits, notices, monkeypatch):
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "900")
    use(Script(HttpError(429, retry_after=200), "ok"))
    llm_client.chat(MSGS, model="openai/gpt-oss-20b")
    assert "openai/gpt-oss-20b" in notices[0]


def test_a_wait_longer_than_the_variable_is_not_waited_for(waits, notices, monkeypatch):
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "900")
    t = use(Script(HttpError(429, QUOTA_429, retry_after=901)))
    with pytest.raises(llm_client.ApiFailure) as info:
        llm_client.chat(MSGS)
    assert t.calls == 1 and waits == [] and notices == [] and info.value.retry_after == 901
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "901")
    t2 = use(Script(HttpError(429, retry_after=901), "ok"))
    assert llm_client.chat([{"role": "user", "content": "other"}]).text == "ok" and waits == [906.0]       # exactly the variable is waited for


def test_a_short_retry_after_is_the_old_rule_and_not_a_quota_wait(waits, notices, monkeypatch):
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "900")
    use(Script(HttpError(429, retry_after=30), HttpError(429, retry_after=120), "ok"))
    assert llm_client.chat(MSGS).text == "ok"
    assert waits == [30, 120] and notices == []                                             # no margin, no notice: the per-minute window


def test_a_quota_wait_does_not_use_up_an_attempt(waits, notices, monkeypatch):
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "900")
    items = [HttpError(503)] * 5 + [HttpError(429, retry_after=300)] + ["ok"]
    t = use(Script(*items))
    assert llm_client.chat(MSGS).text == "ok" and t.calls == 7
    assert waits == [2, 5, 10, 20, 40, 305.0]                                               # five ordinary retries, then the quota wait, then it works
    s = llm_client.summary()
    assert s["retries"] == 6 and s["retry_wait_s"] == 77 + 305 and s["failures"] == 0


def test_a_call_cannot_wait_for_ever(waits, notices, monkeypatch):
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "900")
    monkeypatch.setattr(llm_client, "MAX_QUOTA_WAITS", 3)
    t = use(Script(HttpError(429, QUOTA_429, retry_after=286)))
    with pytest.raises(llm_client.ApiFailure) as info:
        llm_client.chat(MSGS)
    assert t.calls == 4 and waits == [291.0] * 3 and len(notices) == 3
    assert info.value.retry_after == 286 and info.value.attempts == 4 and "[after 4 attempts]" in str(info.value)
    assert llm_client.summary()["failures"] == 1


def test_the_default_number_of_quota_waits_per_call():
    assert llm_client.MAX_QUOTA_WAITS == 60 and llm_client.QUOTA_MARGIN_S == 5.0


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_request_that_cannot_work_is_never_waited_for_whatever_the_header_says(status, waits, notices, monkeypatch):
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "900")
    t = use(Script(HttpError(status, retry_after=300)))
    with pytest.raises(llm_client.ApiFailure):
        llm_client.chat(MSGS)
    assert t.calls == 1 and waits == [] and notices == []


def test_a_503_with_a_long_retry_after_is_waited_for_too(waits, notices, monkeypatch):
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "900")
    use(Script(HttpError(503, retry_after=400), "ok"))
    assert llm_client.chat(MSGS).text == "ok" and waits == [405.0]


def test_the_answer_after_a_quota_wait_is_cached_and_its_latency_does_not_include_the_wait(waits, notices, monkeypatch):
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "900")
    t = use(Script(HttpError(429, retry_after=286), "ok"))
    first = llm_client.chat(MSGS)
    again = llm_client.chat(MSGS)
    assert (first.cached, again.cached, t.calls) == (False, True, 2) and first.latency_s < 5.0 and waits == [291.0]


def test_the_notice_goes_to_stderr_by_default(capsys):
    llm_client._notify("hello quota")
    cap = capsys.readouterr()
    assert cap.err == "hello quota\n" and cap.out == ""


# ================================================================ apierrors: which limit, and how to wait
def test_the_limit_clause_reads_which_limit_was_reached():
    raw = f"API Error: HTTP 429 (retry after 286 s): Error code: 429 - {{'error': {{'message': '{QUOTA_429}'}}}}"
    assert A.limit_clause(raw) == "Groq says: tokens per day (TPD): limit 200000, used 199700, requested 660."
    per_min = raw.replace("tokens per day (TPD)", "requests per minute (RPM)")
    assert A.limit_clause(per_min) == "Groq says: requests per minute (RPM): limit 200000, used 199700, requested 660."
    assert A.limit_clause("API Error: HTTP 429: something else") == "" and A.limit_clause(None) == "" and A.limit_clause("") == ""


def test_a_stop_for_the_quota_names_the_limit_and_says_how_to_wait_it_out():
    raw = f"API Error: HTTP 429 (retry after 286 s): {QUOTA_429}"
    why = A.run_fatal(raw)
    assert "wait of 286 s (about 5 min)" in why and "Groq says: tokens per day (TPD): limit 200000, used 199700, requested 660." in why
    assert "CAPP_QUOTA_WAIT to at least 286" in why and '$env:CAPP_QUOTA_WAIT="900"' in why


def test_no_tip_when_the_wait_is_already_allowed_or_cannot_be_allowed(monkeypatch):
    raw = "API Error: HTTP 429 (retry after 286 s): x"
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "900")
    assert "CAPP_QUOTA_WAIT" not in A.run_fatal(raw)                                          # it is on and the call still failed: more waiting is not the answer
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "286")
    assert "CAPP_QUOTA_WAIT" not in A.run_fatal(raw)                                          # exactly as long as the wait: allowed already
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "285")
    assert "CAPP_QUOTA_WAIT to at least 286" in A.run_fatal(raw)
    monkeypatch.delenv("CAPP_QUOTA_WAIT")
    assert "CAPP_QUOTA_WAIT" not in A.run_fatal("API Error: HTTP 429 (retry after 80000 s): x")     # a day: no setting waits that long
    assert "CAPP_QUOTA_WAIT to at least 7200" in A.run_fatal("API Error: HTTP 429 (retry after 7200 s): x")


def test_a_stop_without_the_limit_text_is_still_a_clean_message():
    why = A.run_fatal("API Error: HTTP 429 (retry after 3000 s): tokens per day")
    assert "Groq says" not in why and "wait of 3000 s" in why


# ================================================================ run_eval refuses a wrong CAPP_QUOTA_WAIT before it starts
def test_run_eval_stops_at_once_on_a_wrong_quota_wait(monkeypatch, capsys):
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "soon")
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--system", "rb", "--limit", "1"])
    with pytest.raises(SystemExit) as e:
        run_eval.main()
    assert e.value.code == 2 and "CAPP_QUOTA_WAIT='soon'" in capsys.readouterr().err


# ================================================================ g2: tokens and days of quota
def est(tin, tout, usd=0.5):
    return {"model": "m", "tokens_in": tin, "tokens_out": tout, "usd": usd}


def test_totals_add_the_tokens_of_every_plan_and_none_where_one_is_unknown():
    a, b = TG.fake_plan(100, [est(80_000, 20_000)], 60.0), TG.fake_plan(50, [est(30_000, 10_000)], 30.0)
    assert G.totals([a, b])["tokens"] == 140_000
    assert G.totals([a, TG.fake_plan(10, [{"model": "m", "note": "no earlier run"}])])["tokens"] is None
    assert G.totals([a, TG.fake_plan(10)])["tokens"] is None
    assert G.totals([a, TG.fake_plan(10, [{"usd": 1.0}])])["tokens"] is None                    # a row without token counts
    assert G.totals([a, TG.fake_plan(10, [{"note": "no earlier run", "tokens_in": 1, "tokens_out": 1}])])["tokens"] is None      # a note wins
    assert G.totals([a, TG.fake_plan(0, llm=False)])["tokens"] == 100_000                       # a run without LLM adds none and does not hide the rest


def test_the_tokens_are_known_even_where_there_is_no_price():
    a = TG.fake_plan(100, [est(80_000, 20_000, usd=None)], 60.0)
    t = G.totals([a])
    assert t["usd"] is None and t["tokens"] == 100_000
    assert "about 100K tokens" in G.totals_text(t) and "no estimate of cost" in G.totals_text(t)


def test_the_text_says_the_tokens_and_the_days_of_the_free_quota():
    t = {"calls": 600, "usd": 1.5, "seconds": 3600.0, "tokens": 600_000}
    text = G.totals_text(t, 200_000)
    assert "about 600K tokens = 3.0 day(s) of the free quota (200K tokens/day) [TO CONFIRM]" in text
    assert text.index("$1.50") < text.index("600K tokens")
    no_days = G.totals_text(t)
    assert "600K tokens" in no_days and "day(s)" not in no_days
    unknown = G.totals_text({"calls": 600, "usd": 1.5, "seconds": 3600.0, "tokens": None}, 200_000)
    assert "tokens" not in unknown and "day(s)" not in unknown
    assert G.totals_text({"calls": 0, "usd": 0.0, "seconds": 0.0, "tokens": 0.0}, 200_000) == "no API call"
    assert "0.5 day(s)" in G.totals_text({"calls": 5, "usd": None, "seconds": None, "tokens": 100_000}, 200_000)


def history(monkeypatch):
    one = {"calls": 10, "prompt": 10000, "completion": 1000, "latency_s": 20.0, "runs": 1}
    monkeypatch.setattr(runplan, "history", lambda d=None: {m: dict(one) for m in (TG.QWEN, TG.GPT, config.model("small"), config.model("large"))})


def line_of(say, name):
    return next(line for line in say.lines if line.strip().startswith(name + ":"))


def test_the_dry_run_shows_days_per_column_and_none_for_the_shared_one(tmp_path, monkeypatch):
    history(monkeypatch)
    monkeypatch.setattr(config, "price", lambda m: (1.0, 2.0))
    code, _, say, _, _ = TG.go(tmp_path, dry_run=True)
    a = line_of(say, "a")
    assert "K tokens = " in a and "day(s) of the free quota (200K tokens/day) [TO CONFIRM]" in a
    assert "day(s)" in line_of(say, "b")
    shared = line_of(say, "shared")
    assert "K tokens" in shared and "day(s)" not in shared
    assert "two-step (+ts) answers are longer" in say.text


def test_no_days_are_shown_without_a_quota_figure(tmp_path, monkeypatch):
    history(monkeypatch)
    monkeypatch.setattr(config, "price", lambda m: (1.0, 2.0))
    monkeypatch.setattr(config, "QUOTA_TOKENS_PER_DAY", None)
    _, _, say, _, _ = TG.go(tmp_path, dry_run=True)
    assert "K tokens" in line_of(say, "a") and "day(s)" not in line_of(say, "a")


def test_the_question_before_a_column_names_the_days_too(tmp_path, monkeypatch):
    history(monkeypatch)
    monkeypatch.setattr(config, "price", lambda m: (1.0, 2.0))
    _, _, _, asked, _ = TG.go(tmp_path)
    assert "day(s) of the free quota" in asked[0] and "day(s) of the free quota" in asked[1] and "day(s)" not in asked[2]


def test_the_days_are_the_tokens_over_the_quota(tmp_path, monkeypatch):
    history(monkeypatch)
    monkeypatch.setattr(config, "price", lambda m: (1.0, 2.0))
    monkeypatch.setattr(config, "QUOTA_TOKENS_PER_DAY", 1000)
    _, _, say, _, _ = TG.go(tmp_path, dry_run=True)
    a = line_of(say, "a")
    tokens = float(a.split("about ")[-1].split("K tokens")[0]) * 1000
    days = float(a.split("= ")[1].split(" day(s)")[0])
    assert abs(days - tokens / 1000) < 0.06 * max(1.0, tokens / 1000)                          # the text rounds both


def test_a_quota_wait_is_announced_and_a_wrong_one_refused(tmp_path, monkeypatch):
    code, _, say, _, _ = TG.go(tmp_path / "off", dry_run=True)
    assert "slow mode" not in say.text
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "900")
    code, _, say, _, _ = TG.go(tmp_path / "on", dry_run=True)
    assert "Quota slow mode is on (CAPP_QUOTA_WAIT=900)" in say.text and "Keep the laptop on" in say.text
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "soon")
    with pytest.raises(G.G2Error, match="CAPP_QUOTA_WAIT='soon'"):
        TG.go(tmp_path / "bad", dry_run=True)


def test_nothing_to_run_still_announces_the_slow_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPP_QUOTA_WAIT", "300")
    TG.go(tmp_path, yes=True)
    code, _, say, _, _ = TG.go(tmp_path, yes=True)
    assert code == 0 and "Nothing to run" in say.text and "CAPP_QUOTA_WAIT=300" in say.text


# ================================================================ configs/ablation_dev_free.json
def test_the_free_plan_grid_differs_from_the_template_in_three_repeats_only():
    import json
    from capp import tables as T
    a = json.load(open(os.path.join(ROOT, "configs", "ablation_template.json"), encoding="utf-8"))
    b = json.load(open(os.path.join(ROOT, "configs", "ablation_dev_free.json"), encoding="utf-8"))
    T.load_config(os.path.join(ROOT, "configs", "ablation_dev_free.json"))                       # it validates
    assert b["name"] == "ablation_dev_free" and "FREE plan" in b["description"]
    ra, rb = {r["id"]: r for r in a["rows"]}, {r["id"]: r for r in b["rows"]}
    assert list(ra) == list(rb)
    assert {k: v for k, v in rb.items() if v.get("repeats") == 1}.keys() == {"A5", "A6", "A7"}
    for rid in ra:
        assert {k: v for k, v in rb[rid].items() if k != "repeats"} == ra[rid]
    for key in ("dataset", "limit", "repeats", "results_dir", "pool", "resamples", "seed", "shared_tag", "models"):
        assert a[key] == b[key]                                                                  # same tags, so the same run folders
