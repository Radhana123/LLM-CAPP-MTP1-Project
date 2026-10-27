"""Hole diameters read from a drawing (Step 14): parsing, validation and the pre-fill rule. Pure functions, no network.

Why: dataset v2 has no hole diameter, and the physics and feasibility rules need one (HOLE-DEEP is not evaluated without it, and the
hole operations of CAP-POWER use the part diameter, an artefact BASELINE.md Step 11 points out). The image extractor (week1) now has an
optional third pass that asks the vision model for the diameters that are WRITTEN on the drawing; this module turns its JSON into a
clean list. AI is verified, not trusted: the numbers only pre-fill a field the user can edit, they never enter a plan silently.

What is accepted as a hole diameter (anything else is dropped and reported in `dropped`):
    a number                      8, 6.5
    a string with one number      "8", "8 mm", "Ø8", "⌀8", "dia 8", "Ø12 H7", "Ø8 ±0.1", "2x Ø6 THRU", "8,5" (comma decimal)
    in range                      HOLE_MIN_MM <= d <= HOLE_MAX_MM (the app's input bound; below 0.5 mm it is a misreading of a drawing)
Not accepted: a thread callout (M8, M10x1.25: that is not a drilled size), inches, a size like "10x1.25", a boolean, NaN/inf.

One number for the whole part: the physics and the rules take a single hole_diameter. When the drawing has several, the largest is
pre-filled (CAP-POWER, a hard rule, is worst for the largest hole; HOLE-DEEP is worst for the smallest, so a single value cannot be
conservative for both). All the detected sizes are listed to the user, who edits the field.

    read_hole_diameters(parsed)        {"ok", "hole_diameters" (largest first, unique), "dropped", "note"}   (parsed = the model's JSON)
    choose_hole_diameter(values)       the value to pre-fill, or None
    prefill(extraction, seen)          (value or None, new signature): what to put in the field and whether it was already applied
"""
import math
import re

HOLE_MIN_MM = 0.5
HOLE_MAX_MM = 200.0
MAX_LISTED = 8
KEY = "hole_diameters_mm"           # the key the dimension prompt asks the model for

_DIA_SIGNS = "Øø⌀ΦφÞ"
_NUMBER = re.compile(r"(\d+(?:[.,]\d+)?)")
_COUNT_PREFIX = re.compile(r"^\d+\s*[xX×]\s*(?=[" + _DIA_SIGNS + r"]|dia)", re.IGNORECASE)    # "2x Ø6" (but not "10x1.25", "4 x 8")
_LEADING = re.compile(r"^(?:[" + _DIA_SIGNS + r"]|dia(?:meter)?\.?|phi)?\s*", re.IGNORECASE)
_OK_AFTER = re.compile(r"^(?:mm\b|h\d|[±+(]|-\s*0[.,]|thru|through|deep|dp\b)", re.IGNORECASE)     # what may follow the number


def _to_mm(value):
    """The diameter in mm, or None if `value` is not an acceptable hole diameter."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        d = float(value)
    elif isinstance(value, str):
        s = value.strip()
        if not s or re.match(r"^[Mm]\s*\d", s):                  # thread callout
            return None
        s = _COUNT_PREFIX.sub("", s)
        s = _LEADING.sub("", s)
        m = _NUMBER.match(s)
        if not m:
            return None
        rest = s[m.end():].strip()
        if rest and not _OK_AFTER.match(rest):                   # inches, "10x1.25", "5 in", ...
            return None
        d = float(m.group(1).replace(",", "."))
    else:
        return None
    if not math.isfinite(d) or d < HOLE_MIN_MM or d > HOLE_MAX_MM:
        return None
    return round(d, 2)


def read_hole_diameters(parsed) -> dict:
    """Clean the model's reply (already JSON-parsed). ok=False if the reply does not have the expected key."""
    out = {"ok": False, "hole_diameters": [], "dropped": [], "note": ""}
    if not isinstance(parsed, dict) or KEY not in parsed:
        out["note"] = "the reply had no hole-diameter list"
        return out
    raw = parsed[KEY]
    note = parsed.get("note", parsed.get("notes", ""))
    out["note"] = note.strip() if isinstance(note, str) else ""
    if raw is None:
        raw = []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    seen = []
    for item in raw:
        d = _to_mm(item)
        if d is None:
            out["dropped"].append(str(item))
        elif d not in seen:
            seen.append(d)
    seen.sort(reverse=True)
    out["dropped"] += [str(d) for d in seen[MAX_LISTED:]]
    out["hole_diameters"] = seen[:MAX_LISTED]
    out["ok"] = True
    return out


def choose_hole_diameter(values):
    """The value to pre-fill: the largest detected diameter, None if there is none."""
    values = [v for v in (values or ()) if v is not None]
    return max(values) if values else None


def prefill(extraction, seen, present=True):
    """(value, signature). `value` is what to put in the hole-diameter field, None to leave the field alone.
    `seen` is the signature returned the last time (None at first); `present` says whether the field still has a value in the
    session. The same detection is applied once, so that the user's own edit is not overwritten at the next rerun; a different
    detection (a new drawing) is applied again; so is the same detection when the field has lost its value (Streamlit drops the
    state of a widget that was not shown in a run, e.g. while Route Only mode was selected). An extraction without a hole diameter
    never touches the field."""
    if not extraction or not extraction.get("success"):
        return None, seen
    sig = tuple(extraction.get("hole_diameters") or ())
    if not sig or (sig == seen and present):
        return None, seen
    return choose_hole_diameter(sig), sig
