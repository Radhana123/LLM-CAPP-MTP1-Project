"""Headless evaluation harness: run a reference system over a dataset, write tables and the metric summary.

    from capp import harness, dataset
    harness.run_experiment("b0", dataset.load_dataset("dev_v2")[:20], repeats=1, tag="dev")

Writes to <out_dir>/<tag>_<system>/
    records.jsonl      every run in full (all candidates, checks, scores): the source for everything else
    per_part.csv       one row per part and repeat (final route, validity, provenance, scores, M4)
    per_candidate.csv  one row per LLM sample and Route Builder route (first-pass validity, failure kinds, tokens, latency)
    summary.json       metrics M1..M8, overall and per part class
    run.json           what was run: system, dataset, model, settings, git commit, time

Repeats use the cache run tag "<tag>-r<k>", so repeat k is a different, reproducible draw. Use the SAME tag for B1 and B0 if
you want them paired: B1's single call is then B0's first sample (same cache entry, no extra API call).
"""
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

from capp import config, llm_client, summary as S, systems

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
    """True if the run made at least one LLM call that was not answered from the cache (sample calls and R2 re-prompts)."""
    for c in rec["candidates"]:
        if c["meta"] and not c["meta"].get("cached"):
            return True
        r2 = (c.get("repair") or {}).get("r2")
        if r2 and r2.get("meta") and not r2["meta"].get("cached"):
            return True
    return False


def run_experiment(system: str, parts: list, repeats: int = 1, tag: str = "exp", out_dir: str = "results",
                   dataset_name: str = "", planner=None, throttle: float = 0.0, max_consecutive_errors: int = 20,
                   front_uses_dims: bool = False, progress=print, write: bool = True, repair_planner=None) -> dict:
    """Run `system` ('rb', 'b1', 'b0' or one of their repair variants) on every part, `repeats` times (rb is deterministic: one run).
    `planner` produces the LLM samples, `repair_planner` the R2 re-prompts (only systems in systems.REPROMPT_SYSTEMS call it);
    both default to the real ones and both are guarded."""
    if system not in systems.SYSTEMS:
        raise HarnessError(f"unknown system {system!r}, choose from {sorted(systems.SYSTEMS)}")
    uses_llm = system in systems.LLM_SYSTEMS
    if not uses_llm:
        repeats = 1
    if uses_llm and planner is None:
        import llm_planner
        planner = llm_planner.generate_process_plan
    wrapped = _guarded(planner, max_consecutive_errors) if uses_llm else None
    wrapped_repair = None
    if system in systems.REPROMPT_SYSTEMS:
        if repair_planner is None:
            from capp import reprompt
            repair_planner = reprompt.reprompt
        wrapped_repair = _guarded(repair_planner, max_consecutive_errors)
    run_fn = systems.SYSTEMS[system]

    records = []
    old_tag = os.environ.get("CAPP_RUN_TAG")
    t_start = time.time()
    try:
        for r in range(1, repeats + 1):
            os.environ["CAPP_RUN_TAG"] = f"{tag}-r{r}"
            for i, part in enumerate(parts, 1):
                kw = {"planner": wrapped, "front_uses_dims": front_uses_dims} if uses_llm else {}
                if wrapped_repair:
                    kw["repair_planner"] = wrapped_repair
                rec = run_fn(part, **kw).to_dict()
                rec["repeat"] = r
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
        "model": config.model("primary") if uses_llm else None,
        "temperature": config.PLANNER_TEMPERATURE if uses_llm else None,
        "repair_temperature": config.REPAIR_TEMPERATURE if system in systems.REPROMPT_SYSTEMS else None,
        "max_tokens": config.PLANNER_MAX_TOKENS if uses_llm else None,
        "llm_mode": config.get_mode(), "front_uses_dims": front_uses_dims,
        "git_commit": _git_commit(), "python": sys.version.split()[0], "platform": platform.platform(),
        "started_utc": datetime.fromtimestamp(t_start, timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - t_start, 1),
        "llm_client_summary": llm_client.summary() if uses_llm else None,
    }
    path = os.path.join(out_dir, f"{tag}_{system}")
    if write:
        write_outputs(path, records, summary, run_info)
    return {"records": records, "summary": summary, "run_info": run_info, "out_dir": path}
