"""Inference cost of a run (M7 in money): every LLM call in the records, by model, at list price.

The records hold, per sample slot, the tokens and the latency of the call(s) that produced it (capp/systems.py: Candidate.meta), of the R2
re-prompt, if any (Candidate.repair["r2"]["meta"]), and of the other attempt of a slot that a cascade escalated (Candidate.cascade).
Cached answers count in full: a cache hit saves the money on the second run, but the cost of the system is what a first run pays.
Prices are the list prices in capp/config.py (PRICE_USD_PER_MTOK, [TO CONFIRM]); a model without a price makes the dollar figures
None, never 0. Records written before Step 17 do not carry the model id: their calls are assigned to `default_model` (the primary
model) and counted in `assumed_model_calls`.
"""
from capp import config


def _usage_of_meta(meta, model, kind):
    if not meta:
        return None
    return {"kind": kind, "model": meta.get("model") or model, "assumed": not meta.get("model"),
            "calls": meta.get("n_calls", 1), "prompt_tokens": meta.get("prompt_tokens") or 0,
            "completion_tokens": meta.get("completion_tokens") or 0, "latency_s": meta.get("latency_s") or 0.0,
            "cached": bool(meta.get("cached"))}


def candidate_usage(c, default_model):
    """All LLM calls behind one candidate: its sample, its R2 re-prompt, and (cascade) the other attempt of the slot."""
    out = []
    u = _usage_of_meta(c.get("meta"), default_model, "sample")
    if u:
        out.append(u)
    r2 = (c.get("repair") or {}).get("r2")
    if r2:
        u = _usage_of_meta(r2.get("meta"), default_model, "repair")
        if u:
            out.append(u)
    casc = c.get("cascade") or {}
    for tier in ("small", "large"):
        if casc.get(tier):
            out += candidate_usage(casc[tier], default_model)
    return out


def record_usage(rec, default_model):
    return [u for c in rec["candidates"] for u in candidate_usage(c, default_model)]


def block(records, default_model=None):
    """Totals by model for a list of records: calls, tokens, latency and list-price dollars, per part and in total."""
    default_model = default_model or config.model("primary")
    n = len(records)
    by = {}
    for rec in records:
        for u in record_usage(rec, default_model):
            m = by.setdefault(u["model"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "assumed_calls": 0})
            m["calls"] += u["calls"]
            m["prompt_tokens"] += u["prompt_tokens"]
            m["completion_tokens"] += u["completion_tokens"]
            m["latency_s"] += u["latency_s"]
            m["assumed_calls"] += u["calls"] if u["assumed"] else 0
    for model, m in by.items():
        m["usd"] = config.call_cost_usd(model, m["prompt_tokens"], m["completion_tokens"])
    if not by:
        return {"n": 0}
    priced = all(m["usd"] is not None for m in by.values())
    total_usd = sum(m["usd"] for m in by.values()) if priced else None
    return {
        "n": n, "by_model": dict(sorted(by.items())),
        "calls_per_part": sum(m["calls"] for m in by.values()) / n,
        "tokens_per_part": sum(m["prompt_tokens"] + m["completion_tokens"] for m in by.values()) / n,
        "latency_s_per_part": sum(m["latency_s"] for m in by.values()) / n,
        "usd_per_part": None if total_usd is None else total_usd / n,
        "usd_total": total_usd,
        "unpriced_models": sorted(k for k, m in by.items() if m["usd"] is None),
        "assumed_model_calls": sum(m["assumed_calls"] for m in by.values()),
    }
