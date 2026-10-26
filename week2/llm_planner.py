# llm_planner.py
# LLM Process Planner — Week 2 | LLM-CAPP Project
# Groq API (Llama-3.1-8B) se manufacturing process plan generate karo

import os
import json
import sys
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week1")))

from dotenv import load_dotenv
from groq import Groq
from feature_vocab import FEATURE_TO_OPERATIONS

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("⚠️  GROQ_API_KEY not found in .env file!")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
MODEL  = "qwen/qwen3.8-27b"   # migrated from deprecated llama-3.1-8b-instant
                               # this model also supports vision (image input),
                               # used by image_feature_extractor.py below


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
                          batch_size: int = 100) -> dict:
    """
    Groq API (Llama-3.1-8B) se manufacturing process plan generate karo.
    Returns: {"success": bool, "steps": [...], "raw_response": "...", "model": "..."}
    """
    if client is None:
        return {
            "success": False,
            "steps": [],
            "raw_response": "GROQ_API_KEY not configured",
            "model": MODEL
        }

    valid_ops  = _get_valid_operations_list()
    feat_str   = ", ".join(features)
    user_prompt = (
        f"Generate a machining process plan for:\n"
        f"Material: {material}\nFeatures: {feat_str}\n"
        f"Tolerance: {tolerance}\nBatch Size: {batch_size}\n\n"
        f"Output ONLY a JSON array of operation names."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(valid_ops=valid_ops)},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        raw   = response.choices[0].message.content.strip()
        steps = _parse_steps(raw)
        return {"success": len(steps) > 0, "steps": steps, "raw_response": raw, "model": MODEL}

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