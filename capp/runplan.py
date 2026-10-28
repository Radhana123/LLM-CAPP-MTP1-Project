"""What a run will ask of the API, before it is started (MTP-1 Step 33).

    python scripts/run_eval.py --system n5_r1_s3+ts --dataset dev_v2 --limit 20 --plan

prints the plan and exits: no API call, nothing written. One number is a BOUND (it cannot be exceeded), the rest are ESTIMATES:

  * live calls, at most (bound). A sample slot costs the planner's calls (1; 2 for the two-step planner +ts; for the constrained planner
    +cg1 one call per step that has more than one option, so at most n_operations - 1) plus one re-prompt where the system re-prompts (R2,
    R3). A cascade (+cas1, +cas2) may ask the large model for the same slot again, so its bound is twice as large, split by model. A
    cached answer costs nothing, and how many answers are cached is not known before the run (a re-prompt depends on the answer before it),
    so this is the cost of an empty cache.
  * tokens, cost, time (estimates): the mean of your earlier runs with the same model, read from results/*/run.json. There is no number
    here that came from anywhere else: without an earlier run with that model the plan says so instead of guessing. The prices are
    capp/config.py's PRICE_USD_PER_MTOK, which says [TO CONFIRM] against the Groq console.

The bound is what `--max-calls N` enforces while the run goes on (capp/harness.py): the run stops between two parts when N live calls
have been made, and running the same command again continues from the cache.
"""
import glob
import json
import os

from capp import config, systems

SEP = "  "


def _ops_of(part: dict) -> int:
    """Operations the constrained planner would place for this part (it builds the set the same way); 0 if it cannot build one."""
    from capp import constrained as CG
    p = systems.part_inputs(part)
    try:
        ops = CG.required_ops(p["features"], part.get("machine_preference", "auto"))
        if not ops:
            ops = CG.graph_ops(p["features"])
    except ValueError:
        return 0
    return len(ops)


def _planner_calls(pspec, part: dict) -> int:
    """The most calls the planner of one sample slot can make."""
    if pspec.constrained:
        return max(0, _ops_of(part) - 1)             # a call per step with two or more options; the last operation is always forced
    return 2 if pspec.two_step else 1


def plan(system: str, parts: list, repeats: int = 1, throttle: float = 0.0) -> dict:
    """The bound on the live calls of `system` on `parts` (module docstring). Raises ValueError for a system name that does not exist."""
    spec = systems.parse_system(system)
    if spec is None:
        return {"system": system, "uses_llm": False, "n_parts": len(parts), "repeats": 1, "n_samples": 0, "reprompt": False,
                "tiers": [], "max_calls": 0, "throttle_s": 0.0}
    pspec = systems.planner_spec(system)
    n_samples = spec[0]
    reprompt = systems.reprompts(system)
    per_slot = [_planner_calls(pspec, part) + int(reprompt) for part in parts]
    one_tier = n_samples * sum(per_slot) * repeats                  # every slot of every part, every repeat, in the worst case
    if pspec.cascade:
        tiers = [{"role": "small", "model": config.model("small"), "max_calls": one_tier},
                 {"role": "large", "model": config.model("large"), "max_calls": one_tier}]     # every slot escalates
    else:
        tiers = [{"role": pspec.model, "model": config.model(pspec.model), "max_calls": one_tier}]
    return {"system": system, "uses_llm": True, "n_parts": len(parts), "repeats": repeats, "n_samples": n_samples,
            "reprompt": reprompt, "two_step": pspec.two_step, "constrained": bool(pspec.constrained), "cascade": pspec.cascade,
            "tiers": tiers, "max_calls": sum(t["max_calls"] for t in tiers), "throttle_s": float(throttle or 0.0)}


def history(results_dir: str = None) -> dict:
    """{model id: {"calls", "prompt", "completion", "latency_s", "runs"}} summed over the run folders of results/ whose run.json says
    what the client did. A cascade run is left out (its model field is only the first tier, the totals mix both)."""
    results_dir = results_dir or os.path.join(config.ROOT, "results")
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "run.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                info = json.load(f)
            cl, model = info.get("llm_client_summary"), info.get("model")
            if not isinstance(cl, dict) or not isinstance(model, str) or info.get("cascade_level"):
                continue
            calls = int(cl.get("calls") or 0)
            if calls <= 0:
                continue
            row = out.setdefault(model, {"calls": 0, "prompt": 0, "completion": 0, "latency_s": 0.0, "runs": 0})
            row["calls"] += calls
            row["prompt"] += int(cl.get("prompt_tokens") or 0)
            row["completion"] += int(cl.get("completion_tokens") or 0)
            row["latency_s"] += float(cl.get("latency_mean_s") or 0.0) * calls
            row["runs"] += 1
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return out


def estimate(p: dict, hist: dict) -> dict:
    """Adds p["estimate"] = per-tier {tokens in/out, USD, seconds} where `hist` has the model, else {"note": why not}. USD is None where
    config.PRICE_USD_PER_MTOK has no price for the model. Returns p."""
    rows, total_s = [], 0.0
    for t in p["tiers"]:
        h = hist.get(t["model"])
        if not h:
            rows.append({"model": t["model"], "note": "no earlier run with this model in results/"})
            continue
        n = t["max_calls"]
        tin, tout = n * h["prompt"] / h["calls"], n * h["completion"] / h["calls"]
        sec = n * h["latency_s"] / h["calls"]
        price = config.price(t["model"])
        usd = None if price is None else (tin * price[0] + tout * price[1]) / 1e6
        rows.append({"model": t["model"], "per_call_in": h["prompt"] / h["calls"], "per_call_out": h["completion"] / h["calls"],
                     "from_calls": h["calls"], "from_runs": h["runs"], "tokens_in": tin, "tokens_out": tout, "usd": usd,
                     "price": price, "seconds": sec})
        total_s += sec
    p["estimate"] = rows
    p["estimate_seconds"] = total_s + p["throttle_s"] * p["n_parts"] * p["repeats"]
    return p


def _k(x: float) -> str:
    return f"{x / 1000:.1f}k" if x >= 1000 else f"{x:.0f}"


def _row(label: str, text: str) -> str:
    return f"  {label:<32s}: {text}"


def format_plan(p: dict) -> str:
    s = p["system"]
    if not p["uses_llm"]:
        return f"Plan for {s}: {p['n_parts']} parts, no API call (the Route Builder needs none)."
    L = [f"Plan for {s}: {p['n_parts']} parts x {p['repeats']} repeat(s), {p['n_samples']} sample(s) per part"]
    if p["constrained"]:
        how = "a call per step with 2+ options (at most operations - 1; about 2 per route on the development set)"
    else:
        how = "2 (two-step planner)" if p["two_step"] else "1"
    L.append(_row("calls per sample slot, at most", how + (", plus 1 re-prompt" if p["reprompt"] else "")))
    if p["cascade"]:
        L.append(_row("cascade", "the large model may be asked for every slot again, and the bound counts that"))
    for t in p["tiers"]:
        L.append(_row(f"live calls, at most ({t['role']})", f"{t['max_calls']}  ({t['model']})"))
    L.append(_row("live calls, at most (total)", f"{p['max_calls']}  <- a bound. Cached answers cost nothing; how many are cached is not known before the run"))
    for e in p.get("estimate", ()):
        if "note" in e:
            L.append(_row("estimate", f"{e['model']}: {e['note']} (run a few parts first)"))
            continue
        usd = "n/a (no price in capp/config.py)" if e["usd"] is None else f"${e['usd']:.3f} at list price [TO CONFIRM against the Groq console]"
        L.append(_row("estimate", f"{e['model']}: about {_k(e['tokens_in'])} in + {_k(e['tokens_out'])} out tokens if every call is live, {usd}"))
        L.append(_row("", f"from your {e['from_calls']} earlier calls in {e['from_runs']} run(s): {e['per_call_in']:.0f} in + {e['per_call_out']:.0f} out tokens per call"))
    if p.get("estimate") and all("note" not in e for e in p["estimate"]):
        L.append(_row("time, if every call is live", f"about {p['estimate_seconds'] / 60:.1f} min, one call at a time"
                      + (f" (includes --throttle {p['throttle_s']:g} s per part)" if p["throttle_s"] else "")))
    L.append(_row("not included", "waiting after a rate limit (up to about 77 s for a call that keeps failing)"))
    L.append(_row("to stop early", f"add --max-calls N (for example --max-calls {max(1, p['max_calls'] // 2)}): the run stops between two parts, "
                  "and the same command later continues from the cache"))
    return "\n".join(L)
