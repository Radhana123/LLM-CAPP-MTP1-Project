"""Two-step planner (report 6.2): reason first, then formalise. Option `+ts` of a system name.

    Call 1  reasoning     The LLM gets the part and the closed operation list and writes a short plan in plain language: for
                          each operation the feature it serves, for each ordering decision the precedence relation it relies
                          on, and a last line "FINAL ORDER: a > b > c". Sampled (T = config.REASON_TEMPERATURE), so the N samples
                          of a part are N different reasoning paths. No JSON in this step.
    Call 2  formalisation The reasoning text is turned into the operation sequence by a greedy call (T = 0) under the
                          platform's structured output: a JSON schema whose operation names are an enum of the closed
                          vocabulary, so the model cannot invent an operation name. It is told to copy the stated order, not to
                          correct it: a mistake in the reasoning must stay visible to the verifier.

Structured output is requested as json_schema, and if the platform or the model refuses it, as json_object, and last as plain
text (config.STRUCTURED_MODES; CAPP_STRUCTURED sets the first one). Whatever happened is recorded per sample (meta.structured_used,
meta.parser), because a run that quietly fell back to plain text is not the same experiment. The v1 multi-strategy parser is the
last-resort fallback only (meta.parser == "fallback").

The reasoning text is stored with the route (meta.reasoning), together with the order the reasoning itself states
(meta.reasoning_order) and whether the formalised sequence is exactly that order (meta.faithful).

Result shape = llm_planner.generate_process_plan's; meta adds n_calls (2), per-call tokens and latency in meta.calls, and sums the
two calls in prompt_tokens / completion_tokens / latency_s, so M7 counts the full price of a sample.
"""
import json
import re

from capp import legacy  # noqa: F401
from capp import config, llm_client
from capp.planners import HINT_TEXT

import llm_planner

SCHEMA_NAME = "process_plan"
FINAL_ORDER = re.compile(r"FINAL\s+ORDER\s*:?\s*(.+)", re.IGNORECASE)

REASON_SYSTEM = """You are a manufacturing process planning expert. Your job is to plan the machining of a given part and to explain the plan briefly.

RULES:
1. Every plan MUST start with "Facing" and end with "Inspection"
2. Use ONLY these valid operations: {valid_ops}
3. Respect machining precedence (e.g., Drilling must come before Tapping/Reaming)
4. Write a short plan in plain language, at most about 150 words: for each operation say which feature it serves, and for each ordering decision name the precedence relation it relies on
5. Finish with one line that starts with "FINAL ORDER:" followed by all operations in order, separated by " > "
6. Do NOT write JSON"""

FORMAL_SYSTEM = """You convert a machining plan written in plain language into a JSON operation sequence.

RULES:
1. Use ONLY these valid operations: {valid_ops}
2. Take the operations of the plan's final order, in exactly that order. If there is no final order line, take the operations in the order the plan gives them
3. Write each operation with its exact name from the list above
4. Do NOT add, remove, reorder or correct anything
5. Output ONLY a JSON object of the form {{"operations": ["Facing", "...", "Inspection"]}}"""


def valid_operations() -> list:
    """The closed vocabulary of the prompt: v1's list, plus Inspection (v1's rule 1 demands it, its list does not name it)."""
    ops = set(llm_planner._get_valid_operations_list().split(", "))
    ops.add("Inspection")
    return sorted(ops)


def schema() -> dict:
    return {"type": "object",
            "properties": {"operations": {"type": "array", "items": {"type": "string", "enum": valid_operations()}}},
            "required": ["operations"], "additionalProperties": False}


def response_format(mode: str):
    if mode == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": schema()}}
    if mode == "json_object":
        return {"type": "json_object"}
    return None


def strip_think(text: str) -> str:
    """Remove <think>...</think>; if nothing is left (cut off inside the tag) keep the inside rather than nothing."""
    if not text:
        return ""
    without_blocks = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"</?think>", "", without_blocks).strip() or re.sub(r"</?think>", "", text).strip()


def reason_messages(material, features, tolerance, batch_size, examples=(), hint=None) -> list:
    lines = [f"Plan the machining of this part:\nMaterial: {material}\nFeatures: {', '.join(features)}\n"
             f"Tolerance: {tolerance}\nBatch Size: {batch_size}\n"]
    if examples:                                                    # least similar first, most similar right before the question
        lines.append("Validated plans of similar parts, for reference:")
        for ex in reversed(list(examples)):
            lines.append(f"- Material: {ex.material}; Features: {', '.join(ex.features)}; plan: {' > '.join(ex.plan)}")
        lines.append("")
    if hint:
        lines.append(HINT_TEXT.format(route=json.dumps(list(hint))).rstrip("\n") + "\n")
    lines.append("Explain the plan briefly and end with the FINAL ORDER line.")
    return [{"role": "system", "content": REASON_SYSTEM.format(valid_ops=", ".join(valid_operations()))},
            {"role": "user", "content": "\n".join(lines)}]


def formal_messages(reasoning: str) -> list:
    return [{"role": "system", "content": FORMAL_SYSTEM.format(valid_ops=", ".join(valid_operations()))},
            {"role": "user", "content": f"Plan:\n{reasoning}\n\nReturn the operation sequence now."}]


def reasoning_order(reasoning: str):
    """The operations on the last FINAL ORDER line of the reasoning, or None if there is none."""
    found = FINAL_ORDER.findall(reasoning or "")
    if not found:
        return None
    parts = [p.strip(" \t*`'\".,;:[]") for p in re.split(r"\s*(?:>|->|→|,)\s*", found[-1])]
    parts = [p for p in parts if p]
    return parts or None


def parse_formalised(raw: str):
    """(steps, parser): 'structured' = a JSON object with an operations list, 'array' = a bare JSON array,
    'fallback' = v1's heuristics, 'none' = nothing found."""
    text = strip_think(raw)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("operations"), list) and all(isinstance(s, str) for s in obj["operations"]):
            return list(obj["operations"]), "structured"
        if isinstance(obj, list) and all(isinstance(s, str) for s in obj):
            return list(obj), "array"
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*?"operations"\s*:\s*(\[.*?\])', text, re.DOTALL)
    if m:
        try:
            ops = json.loads(m.group(1))
            if isinstance(ops, list) and all(isinstance(s, str) for s in ops):
                return ops, "fallback"
        except json.JSONDecodeError:
            pass
    steps = llm_planner._parse_steps(text)
    return steps, ("fallback" if steps else "none")


# Modes the platform has refused in this process (model, mode): later calls do not ask again.
_REFUSED = set()
_REFUSAL = re.compile(r"response_format|json_schema|structured|does not support|not supported|unsupported", re.IGNORECASE)


def reset_refusals():
    _REFUSED.clear()


def _formalise_call(messages, model, sample_id):
    """Ask for structured output, falling back mode by mode. Returns (LLMResponse, mode used, [[mode, why it was skipped]])."""
    modes = list(config.STRUCTURED_MODES)
    modes = modes[modes.index(config.structured_mode()):]
    skipped, last = [], None
    for mode in modes:
        if (model, mode) in _REFUSED:
            skipped.append([mode, "refused earlier in this run"])
            continue
        params = dict(temperature=config.FORMAL_TEMPERATURE, max_tokens=config.FORMAL_MAX_TOKENS)
        rf = response_format(mode)
        if rf:
            params["response_format"] = rf
        try:
            return llm_client.chat(messages, model=model, sample_id=sample_id, **params), mode, skipped
        except llm_client.NoApiKey:
            raise
        except llm_client.CacheMiss as exc:                         # replay: this mode was not the one that was recorded
            last = exc
            skipped.append([mode, "not in cache"])
        except Exception as exc:
            last = exc
            if mode == "none":
                raise
            if _REFUSAL.search(str(exc)):
                _REFUSED.add((model, mode))
            skipped.append([mode, str(exc)[:120]])
    raise last


def make_planner(examples=(), hint=None, context=None, model=None):
    """planner(material, features, tolerance, batch_size, sample_id) -> the v1 result dict, with the two-step meta.
    `model` is a model id for both calls (default: the primary model, llm_planner.MODEL)."""
    context = dict(context) if context is not None else {"examples": [e.part_id for e in examples], "hint": hint is not None}
    context["two_step"] = True

    def planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0, _model=model):
        model = _model or llm_planner.MODEL
        # ---- call 1: reasoning
        try:
            r1 = llm_client.chat(reason_messages(material, features, tolerance, batch_size, examples, hint), model=model,
                                 sample_id=sample_id, temperature=config.REASON_TEMPERATURE, max_tokens=config.REASON_MAX_TOKENS)
        except llm_client.NoApiKey:
            return {"success": False, "steps": [], "raw_response": "GROQ_API_KEY not configured", "model": model}
        except Exception as exc:
            return {"success": False, "steps": [], "raw_response": f"API Error: {exc}", "model": model}
        reasoning = strip_think(r1.text)
        calls = [{"role": "reason", "cached": r1.cached, "finish_reason": r1.finish_reason, "prompt_tokens": r1.prompt_tokens,
                  "completion_tokens": r1.completion_tokens, "latency_s": r1.latency_s}]
        meta = {"cached": r1.cached, "finish_reason": r1.finish_reason, "prompt_tokens": r1.prompt_tokens,
                "completion_tokens": r1.completion_tokens, "latency_s": r1.latency_s, "n_calls": 1, "calls": calls,
                "reasoning": reasoning, "reasoning_truncated": r1.finish_reason == "length", "prompt_context": context,
                "model": model}
        if not reasoning:
            return {"success": False, "steps": [], "raw_response": "empty reasoning", "model": model, "meta": meta}
        # ---- call 2: formalisation
        try:
            r2, mode, skipped = _formalise_call(formal_messages(reasoning), model, sample_id)
        except llm_client.NoApiKey:
            return {"success": False, "steps": [], "raw_response": "GROQ_API_KEY not configured", "model": model}
        except Exception as exc:
            return {"success": False, "steps": [], "raw_response": f"API Error: {exc}", "model": model, "meta": meta}
        steps, parser = parse_formalised(r2.text)
        stated = reasoning_order(reasoning)
        calls.append({"role": "formalise", "cached": r2.cached, "finish_reason": r2.finish_reason, "prompt_tokens": r2.prompt_tokens,
                      "completion_tokens": r2.completion_tokens, "latency_s": r2.latency_s, "structured": mode, "parser": parser,
                      "skipped": skipped})
        meta.update({"cached": r1.cached and r2.cached, "finish_reason": r2.finish_reason, "n_calls": 2,
                     "prompt_tokens": r1.prompt_tokens + r2.prompt_tokens,
                     "completion_tokens": r1.completion_tokens + r2.completion_tokens,
                     "latency_s": r1.latency_s + r2.latency_s, "structured_used": mode, "parser": parser,
                     "reasoning_order": stated, "faithful": None if stated is None else stated == steps})
        raw = r2.text.strip()
        return {"success": len(steps) > 0, "steps": steps, "raw_response": raw, "model": model, "meta": meta}
    return planner
