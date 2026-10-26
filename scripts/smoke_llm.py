"""Live smoke test of the LLM client (uses the real Groq API).

Run from the repo root:   python scripts/smoke_llm.py
Run it TWICE: the second run must show cached=True and finish instantly.

What to look at
  * finish_reason == 'length'  -> the answer was cut by max_tokens (v1 planner uses 200). Qwen3.8 can spend tokens on
                                  hidden reasoning, so a truncated or empty answer would make the baseline look worse than it is.
  * success False / steps []   -> parsing failed; raw_response shows what the model actually said.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for d in (ROOT, os.path.join(ROOT, "week1"), os.path.join(ROOT, "week2")):
    sys.path.insert(0, d)

from capp import config, llm_client  # noqa: E402
import llm_planner  # noqa: E402


def main():
    print("model     :", config.model("primary"))
    print("mode      :", config.get_mode(), "| run tag:", config.get_run_tag())
    print("cache dir :", config.get_cache_dir())
    if not config.get_api_key() and config.get_mode() != "replay":
        print("\nGROQ_API_KEY not found. Put it in .env in the repo root (GROQ_API_KEY=...) and run again.")
        return
    print()
    for i in range(3):
        r = llm_planner.generate_process_plan("Aluminum", ["Hole", "Slot"], "0.02mm", 500, sample_id=i)
        m = r.get("meta", {})
        print("sample %d | success=%s | cached=%s | finish=%s | tokens in/out=%s/%s | %.2fs"
              % (i, r["success"], m.get("cached"), m.get("finish_reason"),
                 m.get("prompt_tokens"), m.get("completion_tokens"), m.get("latency_s", 0.0)))
        print("   steps:", r["steps"])
        if not r["success"]:
            print("   raw  :", r["raw_response"][:200])
    s = llm_client.summary()
    print("\nsummary:", s)
    if s["truncated"]:
        print("WARNING: %d answer(s) were cut off by the token limit (finish_reason=length)." % s["truncated"])


if __name__ == "__main__":
    main()
