"""Live smoke test of the LLM client (uses the real Groq API): three planner calls and one line that says whether they worked.

Run from the repo root:
    python scripts/smoke_llm.py            # 3 calls with the settings of today (the cache answers a repeat: run it twice, the second is instant)
    python scripts/smoke_llm.py --samples 5
    python scripts/smoke_llm.py --old      # NO API call: what the cache answered before Step 37 (qwen's old settings), if you ran this script then

The last lines are the verdict, so a screenshot of the end of the output is enough:
    RESULT: OK       every answer is a route of known operations and none was cut off
    RESULT: PROBLEM  an API error, answers cut off at the token limit (finish_reason 'length'), or answers that are not a route; the line says which
Exit code: 0 OK, 1 PROBLEM, 2 no API key (or a wrong argument).

Why: Qwen3.8 thinks by default and the hidden thinking counts in max_tokens (the v1 planner sends 200), so an answer can be cut off or empty and the
baseline then looks worse than it is. Since Step 37 the primary model is sent reasoning_effort "none" (capp/config.py, CAPP_QWEN_REASONING);
"settings" below shows what is sent. --old sets CAPP_QWEN_REASONING=omit and CAPP_LLM_MODE=replay for this one run: the requests are the old ones, a
cached answer is shown with its finish_reason, and a call that is not in the cache is reported, never made.
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for d in (ROOT, os.path.join(ROOT, "week1"), os.path.join(ROOT, "week2")):
    sys.path.insert(0, d)

from capp import config, llm_client  # noqa: E402
import llm_planner  # noqa: E402

MISS = "not in cache (replay mode)"


def _valid_ops() -> set:
    """The operation names of the planner's prompt, plus Inspection (rule 1 of that prompt makes every plan end with it, yet the list has none)."""
    return set(llm_planner._get_valid_operations_list().split(", ")) | {"Inspection"}


def is_route(result: dict, valid: set = None) -> bool:
    """The planner read something AND every step of it is a known operation. (The planner's parser falls back to one step per line of text,
    so a refusal or a cut-off piece of thinking would otherwise pass for a route.)"""
    valid = _valid_ops() if valid is None else valid
    return bool(result["success"] and result["steps"] and all(step in valid for step in result["steps"]))


def _settings(model: str) -> str:
    params = config.model_params(model)
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items())) if params else "nothing extra is sent"


def verdict(results: list, summary: dict, old: bool = False) -> tuple:
    """(exit code, 'RESULT: ...' line) from the planner results of the calls and llm_client.summary()."""
    n = len(results)
    raws = [r.get("raw_response", "") for r in results]
    misses = sum(1 for raw in raws if MISS in raw)
    api = [raw for r, raw in zip(results, raws) if not r["success"] and raw.startswith("API Error") and MISS not in raw]
    valid = _valid_ops()
    read = sum(1 for r in results if is_route(r, valid))
    cut = summary.get("truncated", 0)
    if old and misses == n:
        return 1, ("RESULT: PROBLEM  none of the old calls is in the cache, so there is nothing to look at (this script was not run with the old "
                   "settings, or the prompt changed since)")
    if api:
        first = api[0][:200]
        hint = "  If it names reasoning_effort, send this line to Claude." if "reasoning" in first.lower() else ""
        return 1, f"RESULT: PROBLEM  {len(api)} of {n} calls failed with an API error: {first}{hint}"
    if cut:
        return 1, (f"RESULT: PROBLEM  {cut} of {n} answers were cut off at the token limit (finish_reason length): the planner's max_tokens is "
                   "used up before the route is written")
    if misses:
        return 1, f"RESULT: PROBLEM  {misses} of {n} calls are not in the cache (with --old nothing is sent to the API; the others are shown above)"
    if read < n:
        return 1, f"RESULT: PROBLEM  {n - read} of {n} answers are not a route of known operations (the raw text is printed above)"
    where = " (all from the cache, no API call was made)" if n and summary.get("cache_hits", 0) == summary.get("calls", -1) else ""
    return 0, f"RESULT: OK  {read} of {n} answers are a route of known operations and none was cut off{where}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Three planner calls to the model and one line that says whether they worked.")
    ap.add_argument("--samples", type=int, default=3, help="how many calls (default 3)")
    ap.add_argument("--old", action="store_true", help="no API call: show what the cache answered with the settings from before Step 37")
    a = ap.parse_args(argv)
    if a.samples < 1:
        print("STOPPED: --samples must be 1 or more", file=sys.stderr)
        return 2
    if a.old:
        os.environ["CAPP_QWEN_REASONING"] = "omit"
        os.environ["CAPP_LLM_MODE"] = "replay"
    model = config.model("primary")
    print("model     :", model)
    print("settings  :", _settings(model))
    print("mode      :", config.get_mode(), "| run tag:", config.get_run_tag())
    print("cache dir :", config.get_cache_dir())
    if not config.get_api_key() and config.get_mode() != "replay":
        print("\nGROQ_API_KEY not found. Put it in .env in the repo root (GROQ_API_KEY=...) and run again.")
        return 2
    print()
    results = []
    for i in range(a.samples):
        r = llm_planner.generate_process_plan("Aluminum", ["Hole", "Slot"], "0.02mm", 500, sample_id=i)
        results.append(r)
        m = r.get("meta", {})
        print("sample %d | success=%s | cached=%s | finish=%s | tokens in/out=%s/%s | %.2fs"
              % (i, r["success"], m.get("cached"), m.get("finish_reason"),
                 m.get("prompt_tokens"), m.get("completion_tokens"), m.get("latency_s", 0.0)))
        print("   steps:", r["steps"])
        if not is_route(r):
            print("   raw  :", r["raw_response"][:200])
    s = llm_client.summary()
    print("\nsummary:", s)
    code, line = verdict(results, s, old=a.old)
    print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
