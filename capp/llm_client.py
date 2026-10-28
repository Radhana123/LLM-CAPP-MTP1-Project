"""Single entry point for every LLM call in MTP-1 (planner, extractor, later the harness).

What it adds over calling Groq directly
  * response cache: the same request is never paid for twice, and a finished experiment can be replayed offline
  * sample_id + run tag in the cache key: the 5 samples of one part (T=0.3) stay 5 different samples, and repeat
    run r of an experiment is a different draw from run r+1, yet each is reproducible
  * patient retries (rate limit, outage, timeout), no retry where it cannot help (wrong request, key, model), and a call log (tokens,
    latency, finish_reason) for the cost metric M7
  * a pluggable transport, so tests never touch the network
"""
import hashlib
import json
import os
import statistics
import threading
import time
from dataclasses import dataclass

from . import config


class CacheMiss(RuntimeError):
    """replay mode and the request is not in the cache."""


class NoApiKey(RuntimeError):
    """A live call is needed but GROQ_API_KEY is not set."""


class ApiFailure(RuntimeError):
    """A live call that failed for good: every attempt failed, or the service said that trying again cannot help. str() starts with the
    HTTP status when there is one, so that the planners' 'API Error: ...' text carries it (capp/apierrors.py reads it back):
    'HTTP 404: The model ... does not exist', 'HTTP 429 (retry after 3000 s): ...', 'APIConnectionError: Connection error.'"""

    def __init__(self, message, status=None, retry_after=None, attempts=1):
        super().__init__(message)
        self.status, self.retry_after, self.attempts = status, retry_after, attempts


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    latency_s: float      # latency of the original live call (also when served from cache, so M7 stays meaningful)
    cached: bool
    key: str


_transport = None
_groq_client = None
_calls = []
_lock = threading.Lock()
_sleep = time.sleep
RETRY_DELAYS = (2, 5, 10, 20, 40)              # seconds waited before attempt 2, 3, ...: about a minute in all, the length of a rate-limit window
MAX_ATTEMPTS = len(RETRY_DELAYS) + 1
NO_RETRY_STATUS = frozenset({400, 401, 403, 404, 422})       # the request, the key or the model is wrong: the same call fails again
MAX_RETRY_AFTER = 120.0                        # a Retry-After beyond this is a quota (per day), not a window: do not wait for it inside a run
_stats = {"retries": 0, "retry_wait_s": 0.0, "failures": 0}


def set_transport(fn):
    """fn(model, messages, params) -> dict(text, prompt_tokens, completion_tokens, finish_reason). Used by tests."""
    global _transport
    _transport = fn


def reset_transport():
    global _transport
    _transport = None


def _groq_transport(model, messages, params):
    global _groq_client
    key = config.get_api_key()
    if not key:
        raise NoApiKey("GROQ_API_KEY not configured")
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=key)
    params = dict(params)
    if "reasoning_effort" in params:                       # sent in the body: an older version of the SDK has no such argument, the API does
        params["extra_body"] = {"reasoning_effort": params.pop("reasoning_effort")}
    resp = _groq_client.chat.completions.create(model=model, messages=messages, **params)
    choice = resp.choices[0]
    usage = getattr(resp, "usage", None)
    return {
        "text": choice.message.content or "",
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "finish_reason": getattr(choice, "finish_reason", "") or "",
    }


def _status(exc):
    """The HTTP status of an exception of the Groq SDK (or anything that carries one), else None."""
    for obj in (exc, getattr(exc, "response", None)):
        status = getattr(obj, "status_code", None)
        if isinstance(status, int):
            return status
    return None


def _retry_after(exc):
    """Seconds the service asks us to wait (Retry-After header), else None."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        value = float(headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _failure(exc, status, retry_after, attempts):
    if status:
        head = f"HTTP {status}" + (f" (retry after {retry_after:.0f} s)" if retry_after else "")
        msg = f"{head}: {exc}"
    else:
        msg = f"{type(exc).__name__}: {exc}"
    return ApiFailure(msg + (f" [after {attempts} attempts]" if attempts > 1 else ""), status, retry_after, attempts)


def make_key(model, messages, params, sample_id=0, run_tag=None):
    payload = {
        "model": model,
        "messages": messages,
        "params": params,
        "sample_id": sample_id,
        "run_tag": config.get_run_tag() if run_tag is None else run_tag,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_path(key):
    return config.get_cache_dir() / key[:2] / (key + ".json")


def _write_cache(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.%d" % os.getpid())
    tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)          # atomic: a crash never leaves a half-written cache file


def chat(messages, *, model=None, sample_id=0, **params):
    """Send a chat request. Extra keyword args (temperature, max_tokens, response_format, ...) go to the API and into the key."""
    model = model or config.model("primary")
    params = config.apply_model_params(model, params)           # reasoning models: low effort and room for the answer (capp/config.py)
    mode = config.get_mode()
    key = make_key(model, messages, params, sample_id)
    path = _cache_path(key)

    if mode != "nocache" and path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        resp = LLMResponse(rec["text"], rec["model"], rec["prompt_tokens"], rec["completion_tokens"],
                           rec["finish_reason"], rec["latency_s"], True, key)
        _log(resp, sample_id)
        return resp

    if mode == "replay":
        raise CacheMiss("not in cache (replay mode): key=%s model=%s sample_id=%s run_tag=%s"
                        % (key[:12], model, sample_id, config.get_run_tag()))

    transport = _transport or _groq_transport
    for attempt in range(MAX_ATTEMPTS):
        t0 = time.perf_counter()
        try:
            out = transport(model, messages, params)
            latency = time.perf_counter() - t0
            break
        except NoApiKey:
            raise
        except Exception as exc:        # network error, rate limit, 5xx ...
            status, wait_hint = _status(exc), _retry_after(exc)
            if status in NO_RETRY_STATUS or attempt == MAX_ATTEMPTS - 1 or (wait_hint or 0.0) > MAX_RETRY_AFTER:
                with _lock:
                    _stats["failures"] += 1
                raise _failure(exc, status, wait_hint, attempt + 1) from exc
            wait = max(RETRY_DELAYS[attempt], wait_hint or 0.0)
            with _lock:
                _stats["retries"] += 1
                _stats["retry_wait_s"] += wait
            _sleep(wait)

    resp = LLMResponse(out["text"], model, out.get("prompt_tokens", 0), out.get("completion_tokens", 0),
                       out.get("finish_reason", ""), latency, False, key)
    if mode != "nocache":
        _write_cache(path, {
            "text": resp.text, "model": model, "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens, "finish_reason": resp.finish_reason,
            "latency_s": resp.latency_s, "sample_id": sample_id, "run_tag": config.get_run_tag(),
            "params": params,
        })
    _log(resp, sample_id)
    return resp


def _log(resp, sample_id):
    with _lock:
        _calls.append({"key": resp.key, "model": resp.model, "cached": resp.cached, "sample_id": sample_id,
                       "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
                       "finish_reason": resp.finish_reason, "latency_s": resp.latency_s})


def get_call_log():
    with _lock:
        return list(_calls)


def reset_call_log():
    with _lock:
        _calls.clear()
        _stats.update(retries=0, retry_wait_s=0.0, failures=0)


def live_calls() -> int:
    """Calls that went to the service since the last reset: answered live, plus those that failed for good. A cache hit is free."""
    with _lock:
        return sum(not c["cached"] for c in _calls) + _stats["failures"]


def _pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))]


def summary():
    """Totals for the cost metric M7. 'truncated' counts finish_reason == 'length' (the answer hit the token limit). 'retries' and
    'retry_wait_s' say how often the service made us wait (rate limit, outage); 'failures' counts the calls that failed for good."""
    calls = get_call_log()
    lat = sorted(c["latency_s"] for c in calls)
    return {
        "calls": len(calls),
        "cache_hits": sum(c["cached"] for c in calls),
        "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
        "completion_tokens": sum(c["completion_tokens"] for c in calls),
        "truncated": sum(c["finish_reason"] == "length" for c in calls),
        "latency_mean_s": statistics.fmean(lat) if lat else 0.0,
        "latency_p50_s": _pct(lat, 0.50),
        "latency_p95_s": _pct(lat, 0.95),
        "retries": _stats["retries"],
        "retry_wait_s": round(_stats["retry_wait_s"], 1),
        "failures": _stats["failures"],
    }
