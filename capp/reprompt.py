"""R2: ask the LLM to correct its own route, with the verifier's violation report in the prompt (Section 6.3).

The conversation is the v1 planner's own (same system prompt, same user prompt, byte for byte), then the LLM's rejected plan
as its answer, then one more user turn that lists every hard violation of that plan and asks for a minimal correction:

    system     v1 planner system prompt (rules + valid operations)
    user       "Generate a machining process plan for: Material ... Features ... Tolerance ... Batch Size ..."
    assistant  the rejected plan as a JSON array
    user       "A verifier rejected your plan: - <violation> ... Correct it ..."

Same return shape as llm_planner.generate_process_plan, so the harness guard (missing key, replay miss, outage) works for both.
The call goes through capp.llm_client (cached). Temperature is config.REPAIR_TEMPERATURE (0.0): a repair should not be a lottery.
The planner's own sample_id and the run tag are part of the cache key, so a repair is reproduced from the cache like a sample.
"""
import json

from capp import legacy  # noqa: F401
from capp import config, llm_client

import llm_planner

MAX_LISTED = 8


def _original_user_prompt(material, features, tolerance, batch_size):
    # identical to llm_planner.generate_process_plan (tests/test_reprompt.py compares the two)
    feat_str = ", ".join(features)
    return (
        f"Generate a machining process plan for:\n"
        f"Material: {material}\nFeatures: {feat_str}\n"
        f"Tolerance: {tolerance}\nBatch Size: {batch_size}\n\n"
        f"Output ONLY a JSON array of operation names."
    )


def feedback_text(violations) -> str:
    vs = list(violations)
    lines = [f"- {v.message}" for v in vs[:MAX_LISTED]]
    more = len(vs) - MAX_LISTED
    if more > 0:
        lines.append(f"- ... and {more} more problem(s)")
    return (
        "A verifier rejected your plan for these reasons:\n" + "\n".join(lines) + "\n\n"
        "Correct the plan. Keep every operation that is correct and in a valid place, change only what is needed to fix the "
        "problems above, and do not add operations that no feature needs. "
        "Output ONLY a JSON array of operation names."
    )


def build_messages(material, features, tolerance, batch_size, previous_steps, violations) -> list:
    return [
        {"role": "system", "content": llm_planner.SYSTEM_PROMPT.format(valid_ops=llm_planner._get_valid_operations_list())},
        {"role": "user", "content": _original_user_prompt(material, features, tolerance, batch_size)},
        {"role": "assistant", "content": json.dumps(list(previous_steps))},
        {"role": "user", "content": feedback_text(violations)},
    ]


def reprompt(material: str, features: list, tolerance: str = "0.05mm", batch_size: int = 100, sample_id: int = 0,
             previous_steps: list = (), violations: list = ()) -> dict:
    """Returns {"success", "steps", "raw_response", "model", "meta"} like the planner."""
    model = config.model("primary")
    try:
        resp = llm_client.chat(
            build_messages(material, features, tolerance, batch_size, previous_steps, violations),
            model=model, sample_id=sample_id,
            temperature=config.REPAIR_TEMPERATURE, max_tokens=config.REPAIR_MAX_TOKENS,
        )
        raw = resp.text.strip()
        steps = llm_planner._parse_steps(raw)
        meta = {"cached": resp.cached, "finish_reason": resp.finish_reason, "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens, "latency_s": resp.latency_s}
        return {"success": len(steps) > 0, "steps": steps, "raw_response": raw, "model": model, "meta": meta}
    except llm_client.NoApiKey:
        return {"success": False, "steps": [], "raw_response": "GROQ_API_KEY not configured", "model": model}
    except Exception as e:
        return {"success": False, "steps": [], "raw_response": f"API Error: {str(e)}", "model": model}
