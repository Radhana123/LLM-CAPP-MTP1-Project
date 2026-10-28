"""API errors in the records (MTP-1 Step 32).

A call that failed because of the service (rate limit, outage, a retired model, a refused key) says nothing about the model. In the records it
is a candidate whose `llm_raw` starts with "API Error" (the planners write that text), so without this module it would count as a model that
"did not parse", lower the parse rate and be replaced by a Route Builder route. This module finds those calls, counts them, and decides when
a run must stop at the first one instead of making hundreds of calls that cannot work.

The text is written by capp/llm_client.py (ApiFailure) after the planners' prefix: `API Error: HTTP 404: ...`, `API Error: HTTP 429 (retry
after 3000 s): ...`, `API Error: APIConnectionError: Connection error. [after 6 attempts]`.

Re-running the same command with the same --tag fills the gaps: the answers that were received are in the cache and are not paid again; only the
calls that failed are made again. Nothing here changes a record.
"""
import re

from capp import llm_client

PREFIX = "API Error"
_HEAD = re.compile(r"^API Error: HTTP (\d{3})(?: \(retry after (\d+) s\))?")
FATAL_STATUS = (401, 403, 404)                 # the key or the model is wrong: every further call fails the same way


def is_api_error(raw) -> bool:
    return isinstance(raw, str) and raw.startswith(PREFIX)


def status_of(raw):
    """(HTTP status, seconds to wait) written in an API error text; None where it says nothing."""
    m = _HEAD.match(raw) if isinstance(raw, str) else None
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2)) if m.group(2) else None


def run_fatal(raw):
    """What to tell the author if this API error must stop the run (a refused key, a model that does not exist, a quota that needs a long
    wait), else None. A rate limit that clears in a minute, an outage and a request one prompt cannot make (400) are not fatal: the call is
    recorded as an API error and the run goes on."""
    status, wait = status_of(raw)
    if status in (401, 403):
        return (f"the API key was refused (HTTP {status}): check GROQ_API_KEY in .env. A key that was ever pasted into a chat should have been "
                "rotated in the Groq console.")
    if status == 404:
        return ("the model was not found (HTTP 404): Groq may have retired it. Check the model ids in capp/config.py (or CAPP_MODEL_PRIMARY, "
                "CAPP_MODEL_SMALL, CAPP_MODEL_LARGE) against the Groq console.")
    if status == 429 and wait and wait > llm_client.MAX_RETRY_AFTER:
        return (f"the rate limit asks for a wait of {wait} s (about {max(1, round(wait / 60))} min): this is the quota, not the per-minute limit. "
                "The answers made so far are in the cache: run the same command again later with the same --tag and only the missing calls are made.")
    return None


def in_candidate(cand: dict) -> int:
    """API errors in one candidate of a record: the sample call, the re-prompt (R2), and the other attempt of a cascade slot."""
    n = int(is_api_error(cand.get("llm_raw")))
    n += int(is_api_error(((cand.get("repair") or {}).get("r2") or {}).get("raw_response")))
    casc = cand.get("cascade") or {}
    for tier in ("small", "large"):
        other = casc.get(tier)
        if isinstance(other, dict):
            n += in_candidate(other)                        # the other attempt has no cascade entry of its own, so this ends
    return n


def in_records(records: list) -> int:
    return sum(in_candidate(c) for r in records for c in r.get("candidates", ()))


def affected(records: list) -> list:
    """[(part_id, repeat)] of the records with at least one API error, in the order of the records."""
    return [(r["part_id"], r["repeat"]) for r in records if any(in_candidate(c) for c in r.get("candidates", ()))]


def rerun_advice(n: int) -> str:
    return (f"{n} call(s) failed because of the service (rate limit or outage), not because of the model; they are counted as failures in these "
            "numbers. Run the same command again with the same --tag: the answers already received come from the cache and only these calls are made again.")
