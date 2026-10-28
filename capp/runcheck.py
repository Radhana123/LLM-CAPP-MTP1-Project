"""Is a finished run still a valid run? Two questions about its run.json (MTP-1 Step 39), asked the same way by scripts/g1.py and by the grid
(capp/tables.py, `make_tables status/commands/build`):

  * truncated(info): how many of its answers were cut off at the token limit (finish_reason 'length'). Such a run measures the token limit,
    not the model.
  * stale_settings(info): was it made with the model settings of today (capp/config.py model_params)? Steps 35 and 37 changed what is sent to
    the gpt-oss models and to qwen3.8; a run made before that has answers under other settings, and the cache key is different too, so it
    cannot be completed from the cache: it has to be run again (same command, same --tag).

Both read only the run.json dict that capp/harness.py wrote (llm_client_summary, model, cascade_models, model_params); a run.json that lacks a field
says nothing about it, so a run folder without run.json is never called stale or cut off here (diagnostics lists such folders on its own).
"""
from capp import config


def truncated(info) -> int:
    """Number of cut-off answers recorded in run.json (0 when it says none or says nothing, or the number is not a positive whole number)."""
    summary = info.get("llm_client_summary") if isinstance(info, dict) else None
    cut = summary.get("truncated") if isinstance(summary, dict) else None
    return cut if isinstance(cut, int) and not isinstance(cut, bool) and cut > 0 else 0


def models_of(info) -> list:
    """The model ids an LLM run used: its own model and, for a cascade, both tiers. [] for a run without a model (the Route Builder)."""
    if not isinstance(info, dict) or not isinstance(info.get("model"), str):
        return []
    out = [info["model"]]
    tiers = info.get("cascade_models")
    if isinstance(tiers, dict):
        out += [m for m in (tiers.get("small"), tiers.get("large")) if isinstance(m, str) and m not in out]
    return out


def settings_text(params) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items())) if params else "nothing extra"


def stale_settings(info) -> list:
    """[(model id, settings it was run with, settings of today)] for every model whose settings differ; [] when all are the same.
    A run.json without model_params counts as 'nothing extra' (a run from before Step 35). Raises ValueError (in words) for a wrong
    CAPP_QWEN_REASONING."""
    recorded = info.get("model_params") if isinstance(info, dict) else None
    out = []
    for model in models_of(info):
        then = recorded.get(model) if isinstance(recorded, dict) else None
        now = config.model_params(model)
        if (then or {}) != now:
            out.append((model, then or {}, now))
    return out


def stale_text(stale: list) -> str:
    """'qwen/qwen3.8-27b: then nothing extra, now reasoning_effort=none' (several models separated by '; ')."""
    return "; ".join(f"{m}: then {settings_text(t)}, now {settings_text(n)}" for m, t, n in stale)
