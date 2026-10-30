"""Central configuration for MTP-1: model IDs, defaults, paths, cache mode.

Env-dependent values are functions (not constants) so tests and the harness can change them at runtime.

Environment variables
  GROQ_API_KEY        API key (read from .env, never committed)
  CAPP_LLM_MODE       live (default): use cache, call API on a miss | replay: cache only, a miss is an error | nocache: always call API
  CAPP_STRUCTURED     json_schema (default) | json_object | none: first structured-output mode of the two-step planner's 2nd call
  CAPP_RUN_TAG        label mixed into every cache key (e.g. rep1..rep5 for repeated runs); default "default"
  CAPP_CACHE_DIR      cache folder (default <repo>/cache/llm)
  CAPP_MODEL_PRIMARY / _SMALL / _LARGE   override model IDs (primary: default; small and large: the two tiers of the cascade)
  CAPP_QWEN_REASONING none (default) | default | low | medium | high | omit: reasoning of the qwen primary model (see model_params)
  CAPP_QUOTA_WAIT     seconds (default 0 = off): sit out a quota wait (HTTP 429 with a Retry-After above 120 s) of at most this long inside the
                      run instead of stopping; needs a laptop that stays on (see quota_wait)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_ENV_CANDIDATES = [ROOT / ".env", ROOT / "week6" / ".env", ROOT / "week1" / ".env"]


def _load_env():
    try:
        from dotenv import load_dotenv
    except ImportError:          # python-dotenv missing: fall back to real environment variables only
        return
    for p in _ENV_CANDIDATES:
        if p.exists():
            load_dotenv(p)
            return
    load_dotenv()


_load_env()

# Model IDs live here and nowhere else. Groq retired two models during the internship, so IDs are pinned in one place.
# "small"/"large" are only used by the optional model cascade (WP7); verify the IDs in the Groq console before using them.
_DEFAULT_MODELS = {
    "primary": "qwen/qwen3.8-27b",
    "small": "openai/gpt-oss-20b",
    "large": "openai/gpt-oss-120b",
}

# Model roles: "primary" is the model of every system without a model option; "small" and "large" are the two tiers of the model
# cascade (capp/cascade.py, system options +ms, +ml, +cas1, +cas2). Override with CAPP_MODEL_PRIMARY / _SMALL / _LARGE.
MODEL_ROLES = tuple(_DEFAULT_MODELS)

# List prices in USD per million tokens (input, output) on GroqCloud pay-as-you-go, used only to turn the token counts of a run into an
# "expected cost" for the Pareto comparison of the cascade. [TO CONFIRM] Source: a third-party price list dated 2026-10-13
# (morphllm.com/groq-pricing, citing console.groq.com/docs/models); Qwen3.8-27B was listed as a preview model. Prices change: check the
# Groq console before quoting a cost, and edit this table. A model that is not listed has no price (the cost is then "n/a", never 0).
PRICE_USD_PER_MTOK = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "qwen/qwen3.8-27b": (0.80, 4.00),
}

# v1 planner settings, kept identical so that baseline B0 is unchanged
PLANNER_TEMPERATURE = 0.3
PLANNER_MAX_TOKENS = 200

# R2 (re-prompt with the violation report, capp/reprompt.py): greedy decoding, so a repair is reproducible
REPAIR_TEMPERATURE = 0.0
REPAIR_MAX_TOKENS = 200

# Constrained planner (capp/constrained.py, option +cg1 / +cg2): one short call per step that has a choice, the answer is a number
CG_MAX_TOKENS = PLANNER_MAX_TOKENS

# Two-step planner (capp/twostep.py): a sampled reasoning call, then a greedy formalisation call under structured output
REASON_TEMPERATURE = PLANNER_TEMPERATURE
REASON_MAX_TOKENS = 700
FORMAL_TEMPERATURE = 0.0
FORMAL_MAX_TOKENS = 300
STRUCTURED_MODES = ("json_schema", "json_object", "none")      # tried in this order, each one only if the one before is refused

# Reasoning models think before they answer, and on Groq that hidden reasoning is counted in max_tokens, so the planner's 200 tokens can be used up
# before the first character of the route. Two families are known to do this, and both are handled here, in one place (model_params):
#   * gpt-oss (the small and large tiers of the cascade): with the planner's 200 a run of +ms had 96 of its 100 answers cut off (finish_reason
#     "length"). They get reasoning_effort "low" and a token limit of at least REASONING_MIN_MAX_TOKENS (Step 35).
#   * qwen3.8 (the primary model): Groq's own page says thinking is its DEFAULT mode (reasoning_effort "default"; "none" is the instruct mode).
#     So the primary model also got max_tokens 200 with thinking on. It now gets reasoning_effort "none" (Step 37): a plain answer, which is what
#     the baselines B0/B1 are (an LLM asked for a route, not an LLM that thinks about it first). CAPP_QWEN_REASONING changes that: low / medium /
#     high / default switch thinking on (with a token limit of at least QWEN_THINKING_MIN_MAX_TOKENS), omit sends nothing (the requests, and so the
#     cache, of before Step 37).
# [TO CONFIRM] against the first runs after the change: the "trunc" column of the diagnostics should be 0 for both families.
REASONING_MODEL_PREFIXES = ("openai/gpt-oss",)
REASONING_EFFORT = "low"                    # Groq: low, medium or high for gpt-oss
REASONING_MIN_MAX_TOKENS = 2048
QWEN_PREFIXES = ("qwen/qwen3.8",)
QWEN_REASONING_DEFAULT = "none"
QWEN_REASONING_VALUES = ("none", "default", "low", "medium", "high", "omit")
QWEN_THINKING_MIN_MAX_TOKENS = 4096

VALID_MODES = ("live", "replay", "nocache")


def model(role: str = "primary") -> str:
    if role not in _DEFAULT_MODELS:
        raise KeyError("unknown model role %r (use one of %s)" % (role, sorted(_DEFAULT_MODELS)))
    return os.getenv("CAPP_MODEL_" + role.upper(), _DEFAULT_MODELS[role])


def qwen_reasoning() -> str:
    """The reasoning setting of the qwen primary model: CAPP_QWEN_REASONING, else QWEN_REASONING_DEFAULT ('none')."""
    value = os.getenv("CAPP_QWEN_REASONING", "").strip().lower() or QWEN_REASONING_DEFAULT
    if value not in QWEN_REASONING_VALUES:
        raise ValueError("CAPP_QWEN_REASONING=%r, expected one of %s" % (value, QWEN_REASONING_VALUES))
    return value


# Tokens per day per model that the Groq free plan allows (Settings > Limits in the Groq console, read 2026-10-30: 200K for qwen/qwen3.8-27b,
# openai/gpt-oss-120b and openai/gpt-oss-20b). [TO CONFIRM] Groq changes limits: edit this after an upgrade. None: no figure is shown.
QUOTA_TOKENS_PER_DAY = 200_000
QUOTA_WAIT_LIMIT = 7200


def quota_wait() -> float:
    """CAPP_QUOTA_WAIT: the longest wait, in seconds, that one call sits out when Groq says the quota (tokens per day) is used up and when it
    will be free again (a Retry-After above llm_client.MAX_RETRY_AFTER). 0 (the default, also unset): do not wait, the call fails and the run
    stops, as before. The tokens-per-day quota refills a little all the time, so a run that waits goes on at the speed of the refill (about
    200K tokens per day on the free plan): it is slow, the answers it gets are cached, and it needs the laptop to stay on."""
    raw = os.getenv("CAPP_QUOTA_WAIT", "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not (0 <= value <= QUOTA_WAIT_LIMIT):                       # nan fails both
        raise ValueError("CAPP_QUOTA_WAIT=%r, expected a number of seconds from 0 to %d" % (raw, QUOTA_WAIT_LIMIT))
    return value


def model_params(model_id: str) -> dict:
    """The request settings a model needs beyond what the caller asks for ({} for most models; see the comment above). The keys:
    reasoning_effort (sent to the API unless the caller set one) and, where there is one, max_tokens_at_least (a floor for max_tokens,
    not an API parameter)."""
    if any(model_id.startswith(p) for p in REASONING_MODEL_PREFIXES):
        return {"reasoning_effort": REASONING_EFFORT, "max_tokens_at_least": REASONING_MIN_MAX_TOKENS}
    if any(model_id.startswith(p) for p in QWEN_PREFIXES):
        value = qwen_reasoning()
        if value == "omit":
            return {}
        if value == "none":
            return {"reasoning_effort": "none"}                 # no thinking: the answer is short, the limit of the caller is enough
        return {"reasoning_effort": value, "max_tokens_at_least": QWEN_THINKING_MIN_MAX_TOKENS}
    return {}


def apply_model_params(model_id: str, params: dict) -> dict:
    """The parameters of a request to `model_id`: `params` itself for a model without model_params, else a copy with reasoning_effort added
    (a value the caller set stays) and, where the model has a floor, max_tokens raised to it (only if the caller sent one). Called by
    llm_client.chat before the cache key is made, so the key follows what is really sent."""
    extra = model_params(model_id)
    if not extra:
        return params
    out = dict(params)
    out.setdefault("reasoning_effort", extra["reasoning_effort"])
    if "max_tokens" in out and "max_tokens_at_least" in extra:
        out["max_tokens"] = max(out["max_tokens"], extra["max_tokens_at_least"])
    return out


def price(model_id: str):
    """(input, output) USD per million tokens of a model id, or None if it is not in PRICE_USD_PER_MTOK."""
    return PRICE_USD_PER_MTOK.get(model_id)


def call_cost_usd(model_id: str, prompt_tokens, completion_tokens):
    """USD for one call (or a sum of calls) at list price; None if the model has no price. Missing token counts count as 0."""
    pr = price(model_id)
    if pr is None:
        return None
    return ((prompt_tokens or 0) * pr[0] + (completion_tokens or 0) * pr[1]) / 1e6


def structured_mode() -> str:
    """CAPP_STRUCTURED: the first structured-output mode the formalisation call asks for (default json_schema)."""
    mode = os.getenv("CAPP_STRUCTURED", "json_schema").strip().lower()
    if mode not in STRUCTURED_MODES:
        raise ValueError("CAPP_STRUCTURED=%r, expected one of %s" % (mode, STRUCTURED_MODES))
    return mode


def get_api_key():
    return os.getenv("GROQ_API_KEY") or None


def get_mode() -> str:
    mode = os.getenv("CAPP_LLM_MODE", "live").strip().lower()
    if mode not in VALID_MODES:
        raise ValueError("CAPP_LLM_MODE=%r, expected one of %s" % (mode, VALID_MODES))
    return mode


def get_run_tag() -> str:
    return os.getenv("CAPP_RUN_TAG", "default")


def get_cache_dir() -> Path:
    return Path(os.getenv("CAPP_CACHE_DIR", str(ROOT / "cache" / "llm")))
