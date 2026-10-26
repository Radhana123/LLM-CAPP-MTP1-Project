"""Central configuration for MTP-1: model IDs, defaults, paths, cache mode.

Env-dependent values are functions (not constants) so tests and the harness can change them at runtime.

Environment variables
  GROQ_API_KEY        API key (read from .env, never committed)
  CAPP_LLM_MODE       live (default): use cache, call API on a miss | replay: cache only, a miss is an error | nocache: always call API
  CAPP_RUN_TAG        label mixed into every cache key (e.g. rep1..rep5 for repeated runs); default "default"
  CAPP_CACHE_DIR      cache folder (default <repo>/cache/llm)
  CAPP_MODEL_PRIMARY / _SMALL / _LARGE   override model IDs
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

# v1 planner settings, kept identical so that baseline B0 is unchanged
PLANNER_TEMPERATURE = 0.3
PLANNER_MAX_TOKENS = 200

VALID_MODES = ("live", "replay", "nocache")


def model(role: str = "primary") -> str:
    if role not in _DEFAULT_MODELS:
        raise KeyError("unknown model role %r (use one of %s)" % (role, sorted(_DEFAULT_MODELS)))
    return os.getenv("CAPP_MODEL_" + role.upper(), _DEFAULT_MODELS[role])


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
