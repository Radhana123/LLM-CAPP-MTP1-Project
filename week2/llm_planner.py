# llm_planner.py
# LLM Process Planner — Week 2 | LLM-CAPP Project
# Groq API (Llama-3.1-8B) se manufacturing process plan generate karo

import os
import json
import sys
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week1")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repo root -> capp package

from feature_vocab import FEATURE_TO_OPERATIONS
from capp import config, llm_client      # MTP-1 WP0: model ID, .env loading and cache now live in capp/

GROQ_API_KEY = config.get_api_key()
MODEL = config.model("primary")           # qwen/qwen3.8-27b (set in capp/config.py, not here)


def _get_valid_operations_list() -> str:
    all_ops = set()
    for feat_data in FEATURE_TO_OPERATIONS.values():
        for alt in feat_data["alternatives"]:
            all_ops.update(alt)
    return ", ".join(sorted(all_ops))


SYSTEM_PROMPT = """You are a manufacturing process planning expert. Your job is to generate a step-by-step machining process plan for a given part.

RULES:
1. Every plan MUST start with "Facing" and end with "Inspection"
2. Use ONLY these valid operations: {valid_ops}
3. Respect machining precedence (e.g., Drilling must come before Tapping/Reaming)
4. Output ONLY a JSON array of operation names, nothing else
5. Do NOT include any explanation, just the JSON array

EXAMPLE:
Input: Material=Aluminum, Features=Hole,Thread, Tolerance=0.02mm, Batch=500
Output: ["Facing", "Center Drilling", "Drilling", "Tapping", "Inspection"]"""


def generate_process_plan(material: str, features: list,
                          tolerance: str = "0.05mm",
                          batch_size: int = 100,
                          sample_id: int = 0) -> dict:
    """
    LLM se manufacturing process plan generate karo (Groq, via capp.llm_client, cached).
    sample_id: ek hi part ke N alag samples ke liye 0..N-1 (cache key me jata hai, warna N baar same cached jawab milta).
    Returns: {"success": bool, "steps": [...], "raw_response": "...", "model": "...", "meta": {...}}
    """
    valid_ops  = _get_valid_operations_list()
    feat_str   = ", ".join(features)
    user_prompt = (
        f"Generate a machining process plan for:\n"
        f"Material: {material}\nFeatures: {feat_str}\n"
        f"Tolerance: {tolerance}\nBatch Size: {batch_size}\n\n"
        f"Output ONLY a JSON array of operation names."
    )

    try:
        resp = llm_client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT.format(valid_ops=valid_ops)},
                {"role": "user",   "content": user_prompt}
            ],
            model=MODEL,
            sample_id=sample_id,
            temperature=config.PLANNER_TEMPERATURE,
            max_tokens=config.PLANNER_MAX_TOKENS,
        )
        raw   = resp.text.strip()
        steps = _parse_steps(raw)
        meta = {"cached": resp.cached, "finish_reason": resp.finish_reason,
                "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
                "latency_s": resp.latency_s}
        return {"success": len(steps) > 0, "steps": steps, "raw_response": raw, "model": MODEL, "meta": meta}

    except llm_client.NoApiKey:
        return {"success": False, "steps": [], "raw_response": "GROQ_API_KEY not configured", "model": MODEL}
    except Exception as e:
        return {"success": False, "steps": [], "raw_response": f"API Error: {str(e)}", "model": MODEL}


def _parse_steps(raw_response: str) -> list:
    text = raw_response.strip()
    try:
        steps = json.loads(text)
        if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
            return steps
    except json.JSONDecodeError:
        pass
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            steps = json.loads(match.group())
            if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
                return steps
        except json.JSONDecodeError:
            pass
    lines = text.strip().split('\n')
    steps = []
    for line in lines:
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', line.strip()).strip('- ').strip()
        if cleaned and not cleaned.startswith('{') and not cleaned.startswith('['):
            steps.append(cleaned)
    return steps


if __name__ == "__main__":
    print("=== LLM Process Planner (Groq — Llama-3.1-8B) ===\n")

    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not set.")
        exit(1)

    print("─── Test 1: Aluminum Hole+Slot ───")
    r1 = generate_process_plan("Aluminum", ["Hole", "Slot"], "0.02mm", 500)
    print(f"  Success: {r1['success']}\n  Steps  : {r1['steps']}")

    print("\n─── Test 2: Thread+Fillet (asli bug case) ───")
    r2 = generate_process_plan("Steel", ["Thread", "Fillet"], "0.01mm", 50)
    print(f"  Success: {r2['success']}\n  Steps  : {r2['steps']}")