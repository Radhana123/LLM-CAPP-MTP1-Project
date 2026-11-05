# image_feature_extractor.py
# Week 1 (extension) | LLM-CAPP Project
#
# 2D image (engineering sketch / drawing / photo) leke, Groq ke vision-capable
# model (qwen/qwen3.8-27b) se manufacturing features extract karta hai.
#
# Design principle (existing llm_planner.py se consistent):
#   - Model ko sirf EXISTING feature vocabulary di jaati hai — koi naya
#     naam invent nahi kar sakta.
#   - Output feature_vocab.is_valid_feature() se validate hota hai — agar
#     model kuch galat de, wo silently reject hota hai (drop, warning nahi crash).
#   - "AI is verified, not trusted" — extraction ka result UI mein user ko
#     confirm/edit karne ke liye dikhaya jaata hai, blindly pipeline mein
#     aage nahi bhej diya jaata.
#   - Run-to-run consistency: same image ko baar baar extract karne pe kabhi
#     1-2 feature miss ho jaate the (LLM sampling variance + hidden
#     reasoning token budget se). Fix: ab har extraction N=2 independent
#     calls karta hai aur unke UNION ko final result maanta hai (self-
#     consistency / ensemble pattern).

import os
import sys
import json
import base64
import re

from dotenv import load_dotenv
from feature_vocab import GEOMETRY_FEATURES, is_valid_feature

# .env dhoondne ke liye bare load_dotenv() bharosemand nahi tha -- wo current
# working directory se search karta hai, aur agar Streamlit kisi aur folder
# se start ho (ya kabhi CWD change ho), ye silently .env miss kar deta tha,
# jisse "GROQ_API_KEY not configured" error baar baar wapas aata tha. Ab
# explicitly dono jagah check karte hain jahan .env ho sakti hai.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_ENV_PATHS = [
    os.path.join(_THIS_DIR, ".env"),                      # week1/.env
    os.path.join(_THIS_DIR, "..", "week6", ".env"),        # week6/.env (app.py isi se load karta hai)
    os.path.join(_THIS_DIR, "..", ".env"),                 # project root .env
]
for _env_path in _CANDIDATE_ENV_PATHS:
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        break
else:
    load_dotenv()   # last resort -- purana CWD-based default behaviour

sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, "..")))   # repo root -> capp package
from capp import config, dimensions, llm_client      # MTP-1 WP0: model ID, .env loading and cache now live in capp/; Step 14: dimensions

GROQ_API_KEY = config.get_api_key()

VISION_MODEL = config.model("primary")   # qwen/qwen3.8-27b -- same model as llm_planner.py, also accepts image input

# NOTE: qwen/qwen3.8-27b is a dual-mode (thinking/non-thinking) model. By
# default it can spend completion tokens on internal reasoning before
# producing the final answer -- with a small token budget this consumes the
# whole budget and leaves nothing for the actual JSON, which is the likely
# cause of earlier "could not parse model response" failures. The
# reasoning_effort="none" + reasoning_format="hidden" params below disable
# that, response_format={"type": "json_object"} forces valid JSON, and the
# larger max_completion_tokens gives room even if reasoning can't be fully
# disabled. See https://console.groq.com/docs/reasoning for details.

MAX_IMAGE_MB = 20   # Groq's documented per-image limit
# Extraction ek hi image ko 2 ALAG nazariye se padhta hai aur dono ka UNION leta hai:
#   "geometry"   -> shapes/views (circles, cutouts, steps, tapers ...)
#   "annotation" -> text callouts, symbols, notes, title block (M10, R5, C1, 45 deg ...)
# Pehle same prompt 2 baar temperature=0 pe chalta tha -- dono ka jawab lagbhag
# identical aata tha, isliye union se koi extra feature nahi milta tha.
EXTRACTION_PASSES = ["geometry", "annotation"]

# Thinking mode off ("none") hone se dense drawings mein features miss ho jaate
# the. "low" thoda sochne deta hai (recall behtar). Latency/token quota zyada
# lagta hai -- agar quota ki dikkat ho ya response khaali aaye to "none" kar do.
REASONING_EFFORT = "low"
MAX_COMPLETION_TOKENS = 4000   # thinking ke liye extra jagah (JSON chhota hi hota hai)

# Har feature drawing mein kaisa dikhta hai -- model ko sirf naam dene se wo
# Boss/Fillet/Counterbore/Step jaise features ka matlab guess karta tha.
FEATURE_GUIDE = {
    "Hole": "a visible circle (usually with a centerline cross) seen end-on, or two parallel dashed lines in a side view; diameter callouts like Ø10, R6 on a circle. A centerline alone, with no circle or bore, is NOT a hole",
    "Slot": "long narrow cut (length much larger than width) with parallel sides: obround/rounded-end slot, U-shaped channel, or open-ended groove milled into a face",
    "Pocket": "recessed area enclosed on ALL sides, milled to a depth (rectangular, circular or irregular window). A cut that is open at the sides or ends is a Slot, not a Pocket",
    "Boss": "raised cylindrical or rectangular projection standing above the surrounding surface (e.g. round boss or lug on a base block)",
    "Thread": "callouts like M10, M12x1.5, UNC/UNF, G1/4; thin partial arc in the end view or fine lines along a cylinder",
    "Chamfer": "small angled edge break; notes like C1, 1x45 deg, 2.5x45",
    "Fillet": "rounded corner or edge with radius callouts like R5, R2.5 (external or internal corners)",
    "Groove": "narrow rectangular notch cut around a shaft (relief groove / undercut / necking)",
    "Step": "stepped SHAFT: round stock with several different diameters (turned shoulders). Do NOT use for flat shoulders on a block or plate (that is Shoulder)",
    "Face": "flat machined end surface or flat mounting surface; surface-finish symbol on a flat face",
    "Taper": "round CONICAL surface whose diameter changes linearly, with a taper ratio or angle callout. A flat sloped face on a block is NOT Taper (that is Inclined_Face)",
    "Knurl": "cross-hatched/diamond or straight pattern on a cylindrical surface; word KNURL",
    "Counterbore": "hole with a larger flat-bottomed step at the mouth; two concentric circles; CBORE symbol",
    "Countersink": "conical enlargement at a hole mouth; CSK or 90 deg countersink symbol",
    "Keyway": "rectangular groove along a shaft or bore for a key; keyway width x depth callout",
    "Spline": "many parallel axial grooves/teeth around a shaft or bore",
    "Gear_Teeth": "gear tooth profile; module or tooth-count callout",
    "Contour_3D": "free-form curved 3D sculpted surface",
    "Engraved_Mark": "engraved text, logo or marking",
    "Turning": "plain cylindrical outer surface of a ROUND SHAFT or rod made on a lathe. Do NOT use for a round boss standing on a block, plate or bracket",
    "Bore": "large precision internal cylinder (hollow cylinder, sleeve, H7-type tolerance bore) finished by boring, distinct from a small drilled Hole",
    "Center_Drill": "center holes at shaft ends; callouts like center drill / 60 deg center",
    "Shoulder": "flat step or shoulder milled on a prismatic part (block, plate, bracket): a raised level or ledge with a vertical wall and a flat top, not on round stock",
    "Inclined_Face": "flat sloped/angled face on a block or wedge (inclined plane, angled cut, triangular wedge side); angle or slope dimension; NOT round/conical",
    "Rib": "thin triangular or straight web/gusset joining two faces of a bracket (e.g. the diagonal support behind an upright plate)",
    "T_Slot": "groove with a T-shaped cross-section (narrow neck, wider bottom), as on machine tables and fixtures",
    "Dovetail": "angled-sided slide groove or tongue with a dovetail (trapezoid) cross-section",
    "Spotface": "shallow flat circular seat machined around a hole for a bolt head; SF symbol or 'spotface' note",
    "Helical_Groove": "spiral groove or flute winding around a cylinder or bore, with a lead/pitch or helix angle callout",
    "Parting_Off": "cut-off groove that separates the finished part from the bar; 'parting' / 'cut off' note at a shaft end",
    "Undercut": "small relief groove next to a thread or shoulder (thread relief, grinding relief)",
    "Contour_Turn": "curved profile turned on a shaft: arc, radius or concave/convex contour along the axis",
    "Eccentric": "cylinder whose axis is offset from the main axis (cam, crank pin); eccentricity dimension",
    "Reamed_Hole": "precision hole with a tight tolerance (H7, H8, +0.02) or 'ream' note; finished beyond drilling",
    "Internal_Thread": "tapped hole: M-callout on a hole (M8 tapped, M10x1.25 thread depth), partial dashed circle in the hole's end view",
    "Face_Groove": "ring or channel cut into a flat face (O-ring groove, face groove); not on a cylinder's outer surface",
}

PASS_FOCUS = {
    "geometry": "Focus on the DRAWN GEOMETRY in every view (front, side, top, section, detail): shapes, outlines, hidden (dashed) lines, centerlines.",
    "annotation": "Focus on the TEXT AND SYMBOLS: dimension callouts (Ø, R, M, x45), notes, section labels, surface-finish and tolerance symbols, title block, general notes.",
}


VISION_SYSTEM_PROMPT = """You are a manufacturing engineer analyzing a 2D
engineering drawing, CAD view or sketch of a mechanical part.

TASK: List EVERY manufacturing feature that is present. Use ONLY these exact
feature names (with how each usually appears in a drawing):
{feature_guide}

{focus}

RULES:
1. Check ALL views and ALL text/notes in the image. A feature counts if the
   geometry OR an annotation indicates it (e.g. a note "R5" means Fillet,
   "1x45" means Chamfer, "M10" means Thread).
2. Recall matters more than caution: the user reviews your list afterwards and
   removes wrong items, but cannot see items you left out. If a feature is
   plausible from the drawing, include it.
3. Go through the feature list above one by one before answering; do not stop
   after the first few.
   First decide what kind of part it is. If it is a block, plate or bracket
   (prismatic, made by milling), do NOT report the lathe-only features Turning,
   Step, Taper, Groove, Knurl, Bore, Parting_Off, Undercut, Contour_Turn or
   Eccentric; use Shoulder, Inclined_Face, Boss, Rib
   and Slot for such shapes instead.
4. If the image is unclear or low-resolution, still answer but set
   "confidence" to "low".
5. Output ONLY a JSON object in exactly this format, nothing else:
   {{"features": ["Hole", "Thread", ...], "confidence": "high/medium/low",
     "notes": "ONE short sentence, max 20 words"}}
"""


# Step 20 (WP7): optional evidence variant (with_evidence=True). The same prompt, except that the model also gives ONE short evidence
# phrase per feature (the annotation text it read, or the view where it saw the shape), shown to the user next to the feature (report 6.5).
# It is a separate prompt, so the default prompt above (its messages, its cache keys, its recall) is untouched; the two are compared in
# scripts/extraction_eval.py (--evidence).
_OUTPUT_RULE_AT = VISION_SYSTEM_PROMPT.index("5. Output ONLY")
VISION_SYSTEM_PROMPT_EVIDENCE = VISION_SYSTEM_PROMPT[:_OUTPUT_RULE_AT] + """5. For every feature you report give ONE short evidence phrase
   (max 12 words): the annotation text you read, or the view where you saw the shape.
   Never invent text that is not on the drawing.
6. Output ONLY a JSON object in exactly this format, nothing else:
   {{"features": ["Hole", "Thread", ...],
     "evidence": {{"Hole": "circle with diameter callout in the front view", "Thread": "note M10 next to the hole"}},
     "confidence": "high/medium/low", "notes": "ONE short sentence, max 20 words"}}
"""
EVIDENCE_MAX_CHARS = 160

# Step 49: optional THIRD feature pass (with_checklist=True, the app uses it): the same drawing is checked against ALL the vocabulary
# features one by one (yes + the sign it is based on, or no). The two passes above write a free list and stop when the obvious features
# are named; a per-feature answer forces a look at every feature, and the "yes" must name its evidence, so a guess is easy to spot in
# the review table. It has its own prompt, so the two passes above are untouched (same messages, same cache keys, same recall).
CHECKLIST_PASS = "checklist"
CHECKLIST_SYSTEM_PROMPT = """You are a manufacturing engineer checking a 2D engineering drawing, CAD view or sketch of a
mechanical part against a checklist.

CHECKLIST (the only feature names you may use, with how each usually appears in a drawing):
{feature_guide}

TASK: decide for EVERY feature of the checklist, one by one and in the order given, whether it is present in the
drawing. Look in ALL views (front, side, top, section, detail), at hidden (dashed) lines and centerlines, and read
ALL the text: dimension callouts, notes, symbols, section labels, title block.

RULES:
1. Answer "yes" only if you can point to a specific sign: a shape in a named view, or text that is written on the
   drawing. After "yes" give that sign in at most 12 words, e.g. "yes: circle with dia 10 callout in the front view".
   "Possible", "typical for this kind of part" and "not visible" all mean "no".
2. Do not stop after the first few features. A part drawing usually has several, and the easy ones to overlook are
   still features: a chamfer or fillet that is only a note (1x45, R5), a hole that shows only as a circle in the end
   view, a thread that shows only as a callout, a feature that appears in a section or detail view only.
3. First decide what kind of part it is. If it is a block, plate or bracket (prismatic, made by milling), answer "no"
   for the lathe-only features Turning, Step, Taper, Groove, Knurl, Bore, Parting_Off, Undercut, Contour_Turn and
   Eccentric; Shoulder, Inclined_Face, Boss, Rib and Slot are the features of such shapes.
4. If the image is unclear or low-resolution, still answer, and set "confidence" to "low".
5. Output ONLY a JSON object in exactly this format, nothing else, with one entry for each feature of the checklist:
   {{"checklist": {{"Hole": "yes: circle with dia 10 callout in the front view", "Slot": "no", "Thread": "no", ...}},
     "confidence": "high/medium/low", "notes": "ONE short sentence, max 20 words"}}
"""
_RECOVERED_NOTE = "Recovered via keyword match"       # what _parse_response writes when the reply was not JSON


def _says_yes(answer) -> bool:
    """True for a checklist answer that means present: True, or text starting with 'yes' ('yes: circle in the front view')."""
    if answer is True:
        return True
    return isinstance(answer, str) and re.match(r"\s*(yes|y|true|present)\b", answer, re.IGNORECASE) is not None


def _checklist_answers(parsed: dict):
    """(names, evidence) from a parsed checklist reply: the features answered 'yes', and the sign the model gave for each. (None, {}) if
    the reply has no checklist at all. A reply recovered by keyword match is NOT used: the checklist names every feature, so a
    keyword search would find them all."""
    checked = parsed.get("checklist")
    if isinstance(checked, dict):
        names, evidence = [], {}
        for name, answer in checked.items():
            if _says_yes(answer):
                names.append(name)
                if isinstance(answer, str):
                    sign = re.sub(r"^\s*(yes|y|true|present)\b[\s:;,.\-–—]*", "", answer, flags=re.IGNORECASE).strip()
                    if sign:
                        evidence[name] = sign
        return names, evidence
    if isinstance(parsed.get("features"), list) and not str(parsed.get("notes", "")).startswith(_RECOVERED_NOTE):
        return parsed["features"], {}                      # a model that gave a plain list instead: still a valid answer
    return None, {}

# Step 14: optional THIRD pass (with_dimensions=True): the hole diameters that are written on the drawing. It has its own prompt, so the
# two feature passes above are untouched (same messages, same cache keys, same recall) and one extra call is made per image.
# The reply is cleaned by capp/dimensions.py; the numbers only pre-fill a field the user edits (AI is verified, not trusted).
DIMENSION_SYSTEM_PROMPT = """You are a manufacturing engineer reading the DIMENSION CALLOUTS of a 2D engineering drawing.

TASK: list the diameter of every HOLE that is explicitly dimensioned in millimetres. A hole is a drilled, reamed,
counterbored or bored cylindrical cavity: a circle or two dashed lines with a diameter callout such as Ø10,
2x Ø6 THRU or Ø12 H7.

RULES:
1. Report only diameters that are WRITTEN on the drawing. Never estimate a size from how large a circle looks and
   never compute one.
2. Do NOT report the outside diameter of a shaft, boss or flange, a pitch-circle diameter (PCD), a thread callout
   (M8, M10x1.25) or a radius (R5). A hole that has only a thread callout and no diameter is left out.
3. Millimetres only. If the drawing is in inches, return an empty list and say so in the note.
4. If no hole diameter is written on the drawing, return an empty list.
5. Output ONLY a JSON object in exactly this format, nothing else:
   {"hole_diameters_mm": [10, 6.5], "note": "ONE short sentence, max 20 words"}
"""


def _extract_dimensions(b64_image: str, image_format: str) -> dict:
    """The dimension pass. Never raises and never makes the extraction fail: on any problem it reports success=False and an empty
    list, and the user types the hole diameter."""
    empty = {"success": False, "ok": False, "hole_diameters": [], "dropped": [], "note": ""}
    try:
        response = llm_client.chat(
            [{"role": "user", "content": [
                {"type": "text", "text": DIMENSION_SYSTEM_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{b64_image}"}},
            ]}],
            model=VISION_MODEL,
            sample_id=0,
            temperature=0,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            response_format={"type": "json_object"},
            reasoning_effort=REASONING_EFFORT,
            reasoning_format="hidden",
        )
        out = dimensions.read_hole_diameters(_parse_response(response.text.strip()))
        out["success"] = out["ok"]
        if not out["ok"]:
            out["error"] = out["note"]
        return out
    except llm_client.NoApiKey:
        return dict(empty, error="GROQ_API_KEY not configured.")
    except Exception as e:
        return dict(empty, error=f"Vision API error: {e}")


# Step 50: the dimension pass for the WHOLE part (with_part_dimensions=True, the app uses it). One call that replaces the hole-only call above
# (so the app still makes four calls): it reads the overall size, the dimensions written next to each feature that was found, the hole diameters
# and the surface roughness. Like the hole pass it only copies what is written (never estimates), and capp/dimensions.py cleans the reply;
# the numbers pre-fill fields the user edits (AI is verified, not trusted). The features found by the feature passes are named in the
# prompt, so the dimensions are read feature by feature, as the process calculator (Time-and-Cost-calculator) asks for them per process.
PART_DIMENSION_SYSTEM_PROMPT = """You are a manufacturing engineer reading the DIMENSION CALLOUTS of a 2D engineering drawing.

TASK: copy the dimensions that are WRITTEN on the drawing into the groups below. Millimetres only.

The manufacturing features found in this drawing are: {features}.

GROUPS:
1. "overall": the size of the whole part, as written.
   "diameter_mm": the largest OUTSIDE diameter of a round part (shaft, disc, flange). null for a block or plate.
   "length_mm": the overall length (round part: along the axis; block or plate: its longest side).
   "width_mm": block or plate only: the second side. null for a round part.
   "height_mm": block or plate only: the thickness or height. null for a round part.
2. "hole_diameters_mm": the diameter of every drilled, reamed, counterbored or bored hole that has a diameter callout
   (Ø10, 2x Ø6 THRU, Ø12 H7). Not a shaft or boss outside diameter, not a pitch-circle diameter, not a thread callout.
3. "features": one entry for each feature of the list above that has a dimension written next to it:
   {{"feature": "Groove", "text": "3 x 2 deep", "diameter_mm": null, "length_mm": null, "width_mm": 3, "depth_mm": 2,
     "pitch_mm": null, "radius_mm": null, "angle_deg": null}}
   Give only the numbers that are written: a groove 3 wide and 2 deep: width_mm 3, depth_mm 2; an M10x1.25 thread:
   diameter_mm 10, pitch_mm 1.25 (and length_mm if a thread length is written); an R5 fillet: radius_mm 5; a 1x45
   chamfer: length_mm 1, angle_deg 45; a hole Ø10 and 25 deep: diameter_mm 10, depth_mm 25 (a THRU hole has no depth).
   "text" is the callout as written, max 12 words. Use only feature names from the list above.
4. "surface_finish": the surface roughness values written on the drawing, as text, for example ["Ra 1.6", "N7"].
   An empty list if none is written.

RULES:
1. Report only numbers that are WRITTEN on the drawing. Never estimate a size from how large a shape looks, never
   measure it and never compute one (no sums, no differences).
2. Use null for anything that is not written, never 0. A group with nothing written is null or an empty list.
3. Millimetres only. If the drawing is in inches, give null and empty lists everywhere and say so in the note.
4. Output ONLY a JSON object in exactly this format, nothing else:
   {{"overall": {{"diameter_mm": 50, "length_mm": 120, "width_mm": null, "height_mm": null}},
     "hole_diameters_mm": [10, 6.5], "features": [...], "surface_finish": ["Ra 1.6"],
     "note": "ONE short sentence, max 20 words"}}
"""


def _extract_part_dimensions(b64_image: str, image_format: str, features) -> dict:
    """The part-dimension pass. Same contract as _extract_dimensions (never raises, never makes the extraction fail), and the same keys
    in the report (success, ok, hole_diameters, dropped, note, error), plus "read" (the cleaned dimensions) and "derived" (the values
    for the sidebar fields, capp/dimensions.derive_part_dims)."""
    empty = {"success": False, "ok": False, "hole_diameters": [], "dropped": [], "note": "", "read": None, "derived": None}
    names = ", ".join(f for f in (features or []) if is_valid_feature(f)) or "none (look for any dimension)"
    try:
        response = llm_client.chat(
            [{"role": "user", "content": [
                {"type": "text", "text": PART_DIMENSION_SYSTEM_PROMPT.format(features=names)},
                {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{b64_image}"}},
            ]}],
            model=VISION_MODEL,
            sample_id=0,
            temperature=0,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            response_format={"type": "json_object"},
            reasoning_effort=REASONING_EFFORT,
            reasoning_format="hidden",
        )
        read = dimensions.read_part_dimensions(_parse_response(response.text.strip()))
        out = {"success": read["ok"], "ok": read["ok"], "hole_diameters": read["hole_diameters"], "dropped": read["dropped"],
               "note": read["note"], "read": read, "derived": dimensions.derive_part_dims(read) if read["ok"] else None}
        if not read["ok"]:
            out["error"] = read["note"]
        return out
    except llm_client.NoApiKey:
        return dict(empty, error="GROQ_API_KEY not configured.")
    except Exception as e:
        return dict(empty, error=f"Vision API error: {e}")


def _encode_image(image_bytes: bytes) -> str:
    """Image bytes ko base64 string mein convert karo (Groq API ke liye)."""
    return base64.b64encode(image_bytes).decode("utf-8")


def _clean_evidence(raw, valid_features) -> dict:
    """{feature: short text} for the features that were kept; anything else the model wrote there is dropped. Never raises."""
    if not isinstance(raw, dict):
        return {}
    out = {}
    for f, text in raw.items():
        if f in valid_features and isinstance(text, str) and text.strip():
            out[f] = " ".join(text.split())[:EVIDENCE_MAX_CHARS]
    return out


def _extract_single(b64_image: str, image_format: str, feature_guide_str: str, focus_text: str, sample_id: int = 0,
                    with_evidence: bool = False, checklist: bool = False) -> dict:
    """Ek single Groq API call karke parse + validate karta hai. Internal
    helper -- extract_features_from_image() isse N baar call karke union
    leta hai consistency ke liye. checklist=True (Step 49): the per-feature yes/no prompt instead of the free list."""
    if checklist:
        prompt = CHECKLIST_SYSTEM_PROMPT.format(feature_guide=feature_guide_str)
    else:
        prompt = (VISION_SYSTEM_PROMPT_EVIDENCE if with_evidence else VISION_SYSTEM_PROMPT).format(feature_guide=feature_guide_str,
                                                                                                    focus=focus_text)
    try:
        response = llm_client.chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/{image_format};base64,{b64_image}"
                        }},
                    ],
                }
            ],
            model=VISION_MODEL,
            sample_id=sample_id,
            temperature=0,     # was 0.2 -- 0 minimizes run-to-run sampling
                               # variance, jo same image pe har baar alag
                               # result aane ki main wajah thi
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            response_format={"type": "json_object"},
            reasoning_effort=REASONING_EFFORT,
            reasoning_format="hidden",
        )
        raw = response.text.strip()
        parsed = _parse_response(raw)

        if checklist:
            names, evidence_raw = _checklist_answers(parsed)
            if names is None:                         # no checklist in the reply: this pass failed (the other passes still count)
                return {"success": False, "features": [], "rejected_features": [], "confidence": None, "notes": "",
                        "error": "The checklist reply could not be read.", "raw_response": raw}
        else:
            names, evidence_raw = parsed.get("features", []), parsed.get("evidence")

        valid_features = []
        rejected = []
        for f in names:
            if is_valid_feature(f):
                if f not in valid_features:
                    valid_features.append(f)
            else:
                rejected.append(f)

        return {
            "success": True,
            "features": valid_features,
            "rejected_features": rejected,
            "confidence": parsed.get("confidence", "medium"),
            "notes": parsed.get("notes", ""),
            "evidence": _clean_evidence(evidence_raw, valid_features) if (with_evidence or checklist) else {},
            "raw_response": raw,
        }
    except llm_client.NoApiKey:
        return {
            "success": False, "features": [], "rejected_features": [],
            "confidence": None, "notes": "",
            "error": "GROQ_API_KEY not configured.",
        }
    except Exception as e:
        return {
            "success": False, "features": [], "rejected_features": [],
            "confidence": None, "notes": "",
            "error": f"Vision API error: {str(e)}",
        }


def extract_features_from_image(image_bytes: bytes, image_format: str = "png", with_dimensions: bool = False,
                                with_evidence: bool = False, with_checklist: bool = False, with_part_dimensions: bool = False) -> dict:
    """
    Main entry point. 2D image (raw bytes) leke Groq vision model se
    manufacturing features extract karta hai.

    with_part_dimensions=True (Step 50, the app uses it) makes the dimension call read the whole part instead of the holes only (it replaces
    the with_dimensions call, so there is still ONE dimension call): the result then also has "hole_diameters" and "dimensions" as above, and
    "part_dimensions" = {"read": overall size, dimensions per feature, Ra (as written), "derived": the values for the sidebar fields}. A
    failure of that call never fails the extraction.

    with_checklist=True (Step 49, the app uses it) adds a third feature pass that checks the drawing against every vocabulary feature
    one by one (CHECKLIST_SYSTEM_PROMPT); result["passes"]["checklist"] then holds what it found, with the sign each "yes" is based on
    as evidence, and its features join the union. If that pass fails the other two still count. Default False: exactly the two passes,
    as before.

    with_dimensions=True (Step 14, the app uses it) adds a third call that reads the hole diameters written on the drawing:
    the result then has "hole_diameters" (mm, largest first, may be empty) and "dimensions" (the full report of that call:
    success, dropped values, note, error). A failure of that call never fails the extraction. Default False: exactly the
    two feature passes, as before.

    with_evidence=True (Step 20) uses the evidence variant of the prompt: every pass then also gives a short evidence phrase per feature
    (result["passes"][name]["evidence"]). Default False: the prompt, and so the cache keys, are exactly those of before.

    The result always carries "passes": what each pass reported on its own ({"success", "features", "rejected_features",
    "confidence", "notes", "evidence"}, or {"success": False, "error"} for a failed pass), and "passes_ok", the number of passes that
    worked: capp/extraction_review.py needs them to tell which features both passes saw (pre-accepted) from those one pass saw
    (to be confirmed), and capp/extraction_eval.py scores each pass alone.

    Recall ke liye EXTRACTION_PASSES (geometry + annotation) alag-alag padhta hai aur
    unke valid features ka UNION final result hota hai -- ek call kisi
    feature ko miss kare (sampling variance ya truncation ki wajah se) to
    doosri call usually usse pakad leti hai, isliye same image baar baar
    extract karne pe result stabilize ho jaata hai.

    Returns:
        {
            "success": bool,
            "features": [...],       # validated, union across all samples
            "rejected_features": [...],  # union of out-of-vocabulary terms seen
            "confidence": "high"/"medium"/"low",
            "notes": "...",
            "raw_response": "...",   # from the last successful sample
            "sample_agreement": {feature: count, ...}  # har feature kitne
                                                          # samples mein dikha
        }
    """
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_MB:
        return {
            "success": False, "features": [], "rejected_features": [],
            "confidence": None, "notes": "",
            "error": f"Image too large ({size_mb:.1f}MB) — Groq limit is {MAX_IMAGE_MB}MB.",
        }

    b64_image = _encode_image(image_bytes)
    feature_guide_str = "\n".join(
        f"- {f}: {FEATURE_GUIDE.get(f, 'no description')}" for f in GEOMETRY_FEATURES
    )

    pass_names = list(EXTRACTION_PASSES) + ([CHECKLIST_PASS] if with_checklist else [])
    samples = [
        _extract_single(b64_image, image_format, feature_guide_str, PASS_FOCUS.get(p, ""), with_evidence=with_evidence,
                        checklist=(p == CHECKLIST_PASS))
        for p in pass_names               # the passes differ in their prompt text, so they get different cache keys
    ]

    successful = [s for s in samples if s["success"]]
    if not successful:
        return samples[0]

    union_features = []
    union_rejected = []
    agreement = {}
    confidences = []
    for s in successful:
        for f in s["features"]:
            agreement[f] = agreement.get(f, 0) + 1
            if f not in union_features:
                union_features.append(f)
        for f in s["rejected_features"]:
            if f not in union_rejected:
                union_rejected.append(f)
        confidences.append(s["confidence"])

    conf_rank = {"high": 3, "medium": 2, "low": 1, None: 0}
    best_confidence = max(confidences, key=lambda c: conf_rank.get(c, 0)) if confidences else "low"

    combined_notes = successful[-1]["notes"]
    if len(successful) > 1 and any(agreement[f] < len(successful) for f in union_features):
        combined_notes += f" (combined from {len(successful)} passes; not every feature was seen in every pass)"

    passes = {}
    for name, smp in zip(pass_names, samples):
        if smp["success"]:
            passes[name] = {k: smp[k] for k in ("success", "features", "rejected_features", "confidence", "notes", "evidence")}
        else:
            passes[name] = {"success": False, "error": smp.get("error", "")}
    result = {
        "success": True,
        "features": union_features,
        "rejected_features": union_rejected,
        "confidence": best_confidence,
        "notes": combined_notes,
        "raw_response": successful[-1]["raw_response"],
        "sample_agreement": agreement,
        "passes": passes,
        "passes_ok": len(successful),
    }
    if with_part_dimensions:
        dim = _extract_part_dimensions(b64_image, image_format, union_features)
        result["hole_diameters"] = dim["hole_diameters"]
        result["part_dimensions"] = {"read": dim.pop("read"), "derived": dim.pop("derived")}
        result["dimensions"] = dim
    elif with_dimensions:
        dim = _extract_dimensions(b64_image, image_format)
        result["hole_diameters"] = dim["hole_diameters"]
        result["dimensions"] = dim
    return result


def _parse_response(raw_response: str) -> dict:
    """LLM ka JSON output parse karo — fallback strategies ke saath
    (llm_planner.py ke _parse_steps() jaisa hi pattern)."""
    text = raw_response.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_stripped = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.MULTILINE).strip()
    try:
        return json.loads(fence_stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', fence_stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    if match:
        try:
            fixed = match.group().replace("'", '"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    found = [f for f in GEOMETRY_FEATURES if re.search(rf'\b{re.escape(f)}\b', text, re.IGNORECASE)]
    if found:
        return {"features": found, "confidence": "low",
                "notes": "Recovered via keyword match — model response was not valid JSON"}

    return {"features": [], "confidence": "low", "notes": "Could not parse model response",
            "_debug_raw": text[:300]}


# ═══════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Image Feature Extractor — Self Test ===\n")

    if not GROQ_API_KEY:
        print("⚠️  GROQ_API_KEY not set — cannot run live test.")
        print("    (Module structure and validation logic can still be reviewed.)")
    else:
        test_image_path = "test_part_sketch.png"
        if os.path.exists(test_image_path):
            with open(test_image_path, "rb") as f:
                img_bytes = f.read()
            result = extract_features_from_image(img_bytes)
            print(f"Success     : {result['success']}")
            print(f"Features    : {result['features']}")
            print(f"Rejected    : {result['rejected_features']}")
            print(f"Confidence  : {result['confidence']}")
            print(f"Notes       : {result['notes']}")
            if "sample_agreement" in result:
                print(f"Agreement   : {result['sample_agreement']}")
        else:
            print(f"No test image found at '{test_image_path}' — place one there to test live.")

    print("\n=== Validation Layer Test (no API needed) ===")
    mock_parsed = {"features": ["Hole", "Thread", "FakeFeatureXYZ", "Slot"], "confidence": "high"}
    valid = [f for f in mock_parsed["features"] if is_valid_feature(f)]
    rejected = [f for f in mock_parsed["features"] if not is_valid_feature(f)]
    print(f"Input       : {mock_parsed['features']}")
    print(f"Valid       : {valid}")
    print(f"Rejected    : {rejected}  (correctly filtered out)")