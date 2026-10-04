# llm_planner.py
# LLM Process Planner — Week 2 | LLM-CAPP Project
# Groq API se manufacturing process plan generate karo
# STEP 1 UPDATE (MTP1): 2-call Thinking->Decision pipeline
# STEP 1 FIX v3: Call 1 ab reasoning_format="raw" use karta hai (poora
# output content field me aata hai, <think> tags samet) — humne khud
# _strip_think_tags() se clean kar liya. "parsed" mode me content field
# khaali aa raha tha kyunki poora output "reasoning" field me chala jaata
# tha, jo hum padh nahi rahe the. max_tokens dono calls me badhaya gaya
# hai (hidden/raw reasoning mode me bhi model internally reasoning tokens
# consume karta hai, chhota budget truncate kar raha tha).

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
MODEL  = "qwen/qwen3.6-27b"   # migrated from deprecated llama-3.1-8b-instant
                               # NOTE: Groq lists this as a preview vision model
                               # (not marked production-stable) — worth revisiting
                               # model choice for the MTP thesis-final version.


def _get_valid_operations_list() -> str:
    all_ops = set()
    for feat_data in FEATURE_TO_OPERATIONS.values():
        for alt in feat_data["alternatives"]:
            all_ops.update(alt)
    return ", ".join(sorted(all_ops))


def _strip_think_tags(text: str) -> str:
    """reasoning_format='raw' se aane wale <think>...</think> block ko
    hata ke, agar tag ke baad kuch bacha hai toh wahi return karo, warna
    poora text (tags hata ke) return karo — kabhi khaali string na banaye
    jab tak actual content khaali na ho."""
    if not text:
        return text
    stripped = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if stripped:
        return stripped
    # Agar sirf tags hi the aur baad me kuch nahi bacha (tag close hi nahi
    # hua, truncate ho gaya) — tags ke andar ka content hi use karo, kuch
    # na hone se behtar.
    inner = re.sub(r'</?think>', '', text).strip()
    return inner


FEW_SHOT_EXAMPLES = """EXAMPLE 1:
Input: Material=Aluminum, Features=Hole,Slot, Tolerance=0.02mm, Batch=500
Output: ["Facing", "Center Drilling", "Drilling", "Milling", "Inspection"]

EXAMPLE 2:
Input: Material=Steel, Features=Thread,Fillet, Tolerance=0.01mm, Batch=50
Output: ["Facing", "Center Drilling", "Drilling", "Tapping", "Deburring", "Inspection"]

EXAMPLE 3:
Input: Material=Brass, Features=Bore,Chamfer, Tolerance=0.03mm, Batch=200
Output: ["Facing", "Center Drilling", "Boring", "Chamfering", "Inspection"]

EXAMPLE 4:
Input: Material=Aluminum, Features=Turning,Thread, Tolerance=0.02mm, Batch=1000
Output: ["Facing", "Plain/Cylindrical Turning", "Center Drilling", "Drilling", "Tapping", "Inspection"]

EXAMPLE 5:
Input: Material=Steel, Features=Hole,Reaming, Tolerance=0.005mm, Batch=100
Output: ["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"]"""


THINKING_SYSTEM_PROMPT = """You are a manufacturing process planning expert.
Given a part's material, features, tolerance and batch size, THINK OUT LOUD
about the correct machining process plan. Do NOT output JSON yet.

Cover, in plain text:
1. Which operations each requested feature needs
2. What precedence/ordering constraints apply (e.g. Drilling before Tapping/Reaming, Center Drilling before Drilling)
3. If a rule-based reference route is given below, whether it looks correct, and what (if anything) is wrong or missing in it
4. The final ordered list of operations you intend to output, in prose

Valid operations you may use: {valid_ops}

Every plan MUST start with "Facing" and end with "Inspection".
"""


def _build_thinking_user_prompt(material, features, tolerance, batch_size, route_builder_suggestion):
    feat_str = ", ".join(features)
    prompt = (
        f"Material: {material}\nFeatures: {feat_str}\n"
        f"Tolerance: {tolerance}\nBatch Size: {batch_size}\n\n"
        f"{FEW_SHOT_EXAMPLES}\n"
    )
    if route_builder_suggestion:
        rb_str = " -> ".join(route_builder_suggestion)
        prompt += (
            f"\nA rule-based Route Builder has suggested this candidate route "
            f"for this part:\n[{rb_str}]\n"
            f"Verify this candidate against precedence rules and the requested "
            f"features. Refine it if it has an error, or confirm it if it's "
            f"correct — don't discard it without reason.\n"
        )
    prompt += "\nNow think through the correct plan for THIS part."
    return prompt


DECISION_SYSTEM_PROMPT = """You will be given reasoning text about a manufacturing
process plan. Extract ONLY the final ordered list of operations as a JSON array
of strings. Output ONLY the JSON array — no explanation, no markdown fences."""


def generate_process_plan(material: str, features: list,
                          tolerance: str = "0.05mm",
                          batch_size: int = 100,
                          route_builder_suggestion: list = None) -> dict:
    """
    Groq API se manufacturing process plan generate karo — 2-call pipeline:
    Call 1 (Thinking) = reasoning_format="raw" — poora output content me
    aata hai, phir _strip_think_tags() se clean karte hain.
    Call 2 (Decision) = reasoning_format="hidden" — sirf clean JSON.

    Returns: {"success": bool, "steps": [...], "reasoning": "...",
              "raw_response": "...", "model": "..."}
    """
    if client is None:
        return {
            "success": False,
            "steps": [],
            "reasoning": "",
            "raw_response": "GROQ_API_KEY not configured",
            "model": MODEL
        }

    valid_ops = _get_valid_operations_list()

    # ---- Call 1: Thinking ----
    try:
        thinking_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": THINKING_SYSTEM_PROMPT.format(valid_ops=valid_ops)},
                {"role": "user", "content": _build_thinking_user_prompt(
                    material, features, tolerance, batch_size, route_builder_suggestion
                )}
            ],
            temperature=0.3,
            max_tokens=1200,
            reasoning_format="raw"
        )
        raw_thinking = thinking_response.choices[0].message.content.strip()
        reasoning = _strip_think_tags(raw_thinking)
    except Exception as e:
        return {
            "success": False,
            "steps": [],
            "reasoning": "",
            "raw_response": f"API Error (thinking call): {str(e)}",
            "model": MODEL
        }

    # ---- Call 2: Decision ----
    try:
        decision_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Reasoning:\n{reasoning}\n\nExtract the final JSON array now."}
            ],
            temperature=0.0,
            max_tokens=400,
            reasoning_format="hidden"
        )
        raw = decision_response.choices[0].message.content.strip()
        steps = _parse_steps(raw)
        return {
            "success": len(steps) > 0,
            "steps": steps,
            "reasoning": reasoning,
            "raw_response": raw,
            "model": MODEL
        }
    except Exception as e:
        return {
            "success": False,
            "steps": [],
            "reasoning": reasoning,
            "raw_response": f"API Error (decision call): {str(e)}",
            "model": MODEL
        }


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
    print("=== LLM Process Planner (Groq — Thinking+Decision 2-call pipeline) ===\n")

    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not set.")
        exit(1)

    print("─── Test 1: Aluminum Hole+Slot ───")
    r1 = generate_process_plan("Aluminum", ["Hole", "Slot"], "0.02mm", 500)
    print(f"  Success  : {r1['success']}")
    print(f"  Steps    : {r1['steps']}")
    print(f"  Reasoning: {r1['reasoning'][:200]}...")
    if not r1['success']:
        print(f"  RAW      : {r1['raw_response']}")

    print("\n─── Test 2: Thread+Fillet (asli bug case) ───")
    r2 = generate_process_plan("Steel", ["Thread", "Fillet"], "0.01mm", 50)
    print(f"  Success  : {r2['success']}")
    print(f"  Steps    : {r2['steps']}")
    if not r2['success']:
        print(f"  RAW      : {r2['raw_response']}")

    print("\n─── Test 3: With Route Builder suggestion ───")
    r3 = generate_process_plan(
        "Aluminum", ["Hole", "Slot"], "0.02mm", 500,
        route_builder_suggestion=["Facing", "Drilling", "Center Drilling", "Milling", "Inspection"]
    )
    print(f"  Success  : {r3['success']}")
    print(f"  Steps    : {r3['steps']}")
    print(f"  Reasoning: {r3['reasoning'][:300]}...")
    if not r3['success']:
        print(f"  RAW      : {r3['raw_response']}")