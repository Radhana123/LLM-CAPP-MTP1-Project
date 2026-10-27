"""Headless evaluation harness: run a reference system over a dataset, write tables and the metric summary.

    from capp import harness, dataset
    harness.run_experiment("b0", dataset.load_dataset("dev_v2")[:20], repeats=1, tag="dev")

Writes to <out_dir>/<tag>_<system>/
    records.jsonl      every run in full (all candidates, checks, scores): the source for everything else
    per_part.csv       one row per part and repeat (final route, validity, provenance, scores, M4)
    per_candidate.csv  one row per LLM sample and Route Builder route (first-pass validity, failure kinds, tokens, latency)
    summary.json       metrics M1..M8, overall and per part class
    run.json           what was run: system, dataset, model, settings, git commit, freeze status (capp/freeze.py), time

Repeats use the cache run tag "<tag>-r<k>", so repeat k is a different, reproducible draw. Use the SAME tag for B1 and B0 if
you want them paired: B1's single call is then B0's first sample (same cache entry, no extra API call).
"""
import csv
import functools
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

from capp import verify as V
from capp import cascade as CASC
from capp import config, feasibility, freeze, llm_client, physics, summary as S, systems

NO_KEY_MSG = "GROQ_API_KEY not configured"


class HarnessError(systems.FatalRunError):
    pass


def _guarded(planner, max_consecutive_errors):
    """Wrap the planner so that a missing key, a cache miss in replay mode or a long API outage stop the run
    instead of silently turning into 'LLM failed, Route Builder used' results."""
    state = {"errors": 0}

    def call(*args, **kwargs):
        res = planner(*args, **kwargs)
        if not res.get("success"):
            raw = res.get("raw_response", "")
            if raw == NO_KEY_MSG:
                raise HarnessError("GROQ_API_KEY is not set and the response is not in the cache. Put the key in .env, or "
                                   "use CAPP_LLM_MODE=replay with a filled cache.")
            if raw.startswith("API Error: not in cache"):
                raise HarnessError("replay mode: " + raw)
            if raw.startswith("API Error"):
                state["errors"] += 1
                if state["errors"] >= max_consecutive_errors:
                    raise HarnessError(f"{state['errors']} API errors in a row, stopping. Last: {raw[:200]}")
                return res
        state["errors"] = 0
        return res
    return call


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _write_csv(path, rows):
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_outputs(out_dir, records, summary, run_info):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "records.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _write_csv(os.path.join(out_dir, "per_part.csv"), S.part_rows(records))
    _write_csv(os.path.join(out_dir, "per_candidate.csv"), S.candidate_rows(records))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)


def load_records(out_dir):
    with open(os.path.join(out_dir, "records.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _live_call_made(rec) -> bool:
    """True if the run made at least one LLM call that was not answered from the cache (sample calls, R2 re-prompts, and the other
    attempt of a sample slot that a model cascade escalated)."""
    def live(c):
        if c["meta"] and not c["meta"].get("cached"):
            return True
        r2 = (c.get("repair") or {}).get("r2")
        if r2 and r2.get("meta") and not r2["meta"].get("cached"):
            return True
        casc = c.get("cascade") or {}
        return any(live(casc[t]) for t in ("small", "large") if casc.get(t))
    return any(live(c) for c in rec["candidates"])


def _resolve_pool(pool, pspec):
    """The retrieval pool of a few-shot system: a Pool, a dataset name (default dev_v2) or None when no examples are used."""
    if not pspec.few_shot:
        return None
    from capp import dataset, planners
    if isinstance(pool, planners.Pool):
        return pool
    name = pool or "dev_v2"
    if dataset.is_sealed(name):
        raise HarnessError("the retrieval pool must not be the sealed test set or the expert benchmark: examples come from the development data only")
    return planners.Pool.from_dataset(dataset.load_dataset(name), name=name)


def _physics(final: dict, part: dict):
    """Physics analysis of the final route (capp/physics.py); never allowed to stop a run."""
    if not final.get("steps"):
        return None
    try:
        return physics.condense(physics.analyse_part_route(final["steps"], part))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _feasibility(steps, part):
    """Physical-feasibility check of a route (capp/feasibility.py); never allowed to stop a run."""
    if not steps:
        return None
    try:
        return feasibility.condense(feasibility.evaluate(steps, part))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_experiment(system: str, parts: list, repeats: int = 1, tag: str = "exp", out_dir: str = "results",
                   dataset_name: str = "", planner=None, throttle: float = 0.0, max_consecutive_errors: int = 20,
                   front_uses_dims: bool = False, progress=print, write: bool = True, repair_planner=None,
                   weights=None, pool=None, planner_factory=None, analysis: bool = True) -> dict:
    """Run `system` on every part, `repeats` times (rb is deterministic: one run). `system` is rb, b0, b1, a repair variant
    (b0_r1 ... b1_r3) or a generic name n<N>_r<0-3>_s<0-3> (see systems.parse_system), optionally with planner options
    (+fs2, +fs4x, +h1: retrieval few-shot, hint; +fz1, +fz2: feasibility-aware repair; see capp/plannerspec.py).
    `planner` produces the LLM samples, `repair_planner` the R2 re-prompts (only R2/R3 systems call it); both default to the real
    ones and both are guarded. `planner_factory(part) -> planner` replaces `planner` for planner options; by default it is built
    from the system name. `pool` (a planners.Pool or a dataset name, default dev_v2) is where few-shot examples come from.
    `weights` (time, cost, energy) are the priorities of selection S2. `analysis` adds the physics analysis (cutting energy kc,
    Taylor tool-wear cost, Ra risk) and the physical-feasibility check (capp/feasibility.py) of every final route and of every raw LLM
    route to its record; it needs no API call and does not change any route or any validity decision."""
    try:
        spec = systems.parse_system(system)
        pspec = systems.planner_spec(system)
    except ValueError as exc:
        raise HarnessError(str(exc)) from None
    uses_llm = spec is not None
    if planner is not None and (planner_factory is not None or not pspec.is_default):
        raise HarnessError("`planner` replaces the whole prompt, so it cannot be combined with planner options in the system "
                           "name or with `planner_factory`; pass planner_factory instead")
    pool_obj = _resolve_pool(pool, pspec) if uses_llm else None
    if uses_llm and planner_factory is None and not pspec.is_default:
        from capp import planners
        planner_factory = planners.make_factory(pspec, pool_obj)
    casc_level = pspec.cascade if uses_llm else 0
    tier1_role = "small" if casc_level else (pspec.model if uses_llm else "primary")     # the model role of the samples (cascade: its first tier)
    if weights is not None:
        try:
            weights = systems.normalize_weights(weights)
        except ValueError as exc:
            raise HarnessError(str(exc)) from None
    if not uses_llm:
        repeats = 1
    if uses_llm and planner is None:
        import llm_planner
        planner = llm_planner.generate_process_plan
    holder = {"planner": planner}                     # the planner of the current part (differs per part with options)
    wrapped = _guarded(lambda *a, **k: holder["planner"](*a, **k), max_consecutive_errors) if uses_llm else None
    wrapped_repair = wrapped_large = wrapped_large_repair = None
    if systems.reprompts(system):
        if repair_planner is None:
            from capp import reprompt
            repair_planner = reprompt.reprompt if tier1_role == "primary" else functools.partial(reprompt.reprompt, model=config.model(tier1_role))
        wrapped_repair = _guarded(repair_planner, max_consecutive_errors)
        if casc_level:                                  # the re-prompts of a sample go to the model that produced it
            from capp import reprompt
            wrapped_large_repair = _guarded(functools.partial(reprompt.reprompt, model=config.model("large")), max_consecutive_errors)
    if casc_level:
        wrapped_large = _guarded(lambda *a, **k: holder["large"](*a, **k), max_consecutive_errors)
    run_fn = systems.resolve(system)

    records = []
    old_tag = os.environ.get("CAPP_RUN_TAG")
    t_start = time.time()
    try:
        for r in range(1, repeats + 1):
            os.environ["CAPP_RUN_TAG"] = f"{tag}-r{r}"
            for i, part in enumerate(parts, 1):
                if uses_llm and planner_factory is not None:
                    holder["planner"] = planner_factory(part, role="small") if casc_level else planner_factory(part)
                    if casc_level:
                        holder["large"] = planner_factory(part, role="large")
                kw = {"planner": wrapped, "front_uses_dims": front_uses_dims} if uses_llm else {}
                if casc_level:
                    kw["cascade"] = CASC.Cascade(casc_level, wrapped_large, wrapped_large_repair)
                if wrapped_repair:
                    kw["repair_planner"] = wrapped_repair
                if uses_llm and weights is not None:
                    kw["weights"] = weights
                rec = run_fn(part, **kw).to_dict()
                rec["repeat"] = r
                if analysis:
                    rec["final"]["physics"] = _physics(rec["final"], part)
                    rec["final"]["feasibility"] = _feasibility(V.clean_steps(rec["final"]["steps"]), part)
                    for cand in rec["candidates"]:                         # the raw LLM route, as sampled: is a strict-valid route feasible?
                        if cand.get("llm_ok") and cand.get("llm_steps"):
                            cand["raw_feasibility"] = _feasibility(V.clean_steps(cand["llm_steps"]), part)
                rec["run_tag"] = os.environ["CAPP_RUN_TAG"]
                rec["part"] = part
                records.append(rec)
                if progress and (i % 10 == 0 or i == len(parts)):
                    progress(f"[{system} r{r}/{repeats}] {i}/{len(parts)} parts, {time.time() - t_start:.0f}s")
                if throttle and _live_call_made(rec):
                    time.sleep(throttle)
    finally:
        if old_tag is None:
            os.environ.pop("CAPP_RUN_TAG", None)
        else:
            os.environ["CAPP_RUN_TAG"] = old_tag

    summary = S.summarize(records)
    run_info = {
        "system": system, "tag": tag, "dataset": dataset_name, "n_parts": len(parts), "repeats": repeats,
        "model": config.model(tier1_role) if uses_llm else None,
        "model_role": tier1_role if uses_llm else None,
        "cascade_level": casc_level if uses_llm else None,
        "cascade_models": {"small": config.model("small"), "large": config.model("large")} if casc_level else None,
        "price_usd_per_mtok": {m: config.price(m) for m in sorted({config.model(r) for r in
                               (("small", "large") if casc_level else (tier1_role,))})} if uses_llm else None,
        "temperature": config.PLANNER_TEMPERATURE if uses_llm else None,
        "repair_temperature": config.REPAIR_TEMPERATURE if systems.reprompts(system) else None,
        "planner_options": (pspec.name or "v1") if uses_llm else None,
        "few_shot_k": pspec.few_shot if uses_llm else None, "few_shot_leave_out": pspec.loo if pspec.few_shot else None,
        "hint": pspec.hint if uses_llm else None,
        "two_step": pspec.two_step if uses_llm else None, "feasibility_repair": pspec.feas if uses_llm else None,
        "constrained": pspec.constrained if uses_llm else None,
        "physics_analysis": analysis, "feasibility_analysis": analysis,
        "structured_output_requested": config.structured_mode() if uses_llm and pspec.two_step else None,
        "pool": [pool_obj.name, len(pool_obj)] if pool_obj is not None else None,
        "n_samples": spec[0] if spec else None, "repair": spec[1] if spec else None, "selection": spec[2] if spec else None,
        "weights": list(weights or systems.DEFAULT_WEIGHTS) if spec and spec[2] == "s2" else None,
        "confidence_flag_threshold": systems.CONFIDENCE_FLAG if spec else None,
        "max_tokens": (config.CG_MAX_TOKENS if pspec.constrained else config.PLANNER_MAX_TOKENS) if uses_llm else None,
        "llm_mode": config.get_mode(), "front_uses_dims": front_uses_dims,
        "git_commit": _git_commit(), "freeze": freeze.status(config.ROOT),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "started_utc": datetime.fromtimestamp(t_start, timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - t_start, 1),
        "llm_client_summary": llm_client.summary() if uses_llm else None,
    }
    path = os.path.join(out_dir, f"{tag}_{system}")
    if write:
        write_outputs(path, records, summary, run_info)
    return {"records": records, "summary": summary, "run_info": run_info, "out_dir": path}
