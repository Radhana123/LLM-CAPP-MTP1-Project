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

Step 50 (second half of this module): the other dimensions that are written on the drawing. The extractor's dimension pass can ask for the
overall size, the dimensions written next to each detected feature (a groove 3 wide and 2 deep, an M10x1.25 thread, R5, 1x45, a hole
Ø10 x 25 deep) and the surface roughness; the app shows them as a table and pre-fills the Part Dimensions of the sidebar with what the
drawing says (diameter, length, depth, width, Ra, hole diameter). What the drawing does not say stays at the default and is marked as such.

    read_part_dimensions(parsed)       {"ok", "overall", "features", "ra_um", "hole_diameters", "dropped", "note"}   (parsed = the model's JSON)
    derive_part_dims(read)             {"diameter", "length", "depth", "width", "hole_diameter", "ra_um" (None = not written), "source", "notes"}
    prefill_part(extraction, seen, present)   ({field: value}, new signature); the hole diameter is pre-filled by prefill()
    describe_derived(derived)          (written, missing): the fields read from the drawing (with values) and those left at the default
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
    return _parse_mm(value, HOLE_MIN_MM, HOLE_MAX_MM)


def _parse_mm(value, lo, hi):
    """`value` (a number, or text such as 'Ø8', '120 mm', '2x Ø6 THRU') in mm rounded to 0.01, or None when it is not a number in
    [lo, hi] or is a thread callout."""
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
    if not math.isfinite(d) or d < lo or d > hi:
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


# ================================================================ Step 50: the other dimensions of the drawing
PART_KEYS = ("diameter", "length", "depth", "width")           # the part dimensions of the app and of dataset v2 (mm)
PART_LIMITS = {"diameter": (1.0, 500.0), "length": (1.0, 1000.0), "depth": (1.0, 300.0), "width": (1.0, 500.0)}   # = capp.dataset.DIM_LIMITS
HEIGHT_LIMITS = (1.0, 500.0)
RA_CHOICES = (0.4, 0.8, 1.6, 3.2, 6.3, 12.5)                   # um, the values the app can take (capp.analysis_view.RA_CHOICES without None)
# Ra of the ISO 1302 roughness grades N1..N12 (um), for a drawing that writes "N7" instead of "Ra 1.6"
N_GRADE_RA = {1: 0.025, 2: 0.05, 3: 0.1, 4: 0.2, 5: 0.4, 6: 0.8, 7: 1.6, 8: 3.2, 9: 6.3, 10: 12.5, 11: 25.0, 12: 50.0}
MAX_FEATURE_ROWS = 24
# the numbers the model may give for a feature: key -> (label shown, unit, (low, high))
FEATURE_NUMBERS = {"diameter_mm": ("diameter", "mm", (0.1, 1000.0)), "length_mm": ("length", "mm", (0.01, 1000.0)),
                   "width_mm": ("width", "mm", (0.01, 1000.0)), "depth_mm": ("depth", "mm", (0.01, 1000.0)),
                   "pitch_mm": ("pitch", "mm", (0.05, 20.0)), "radius_mm": ("radius", "mm", (0.05, 500.0)),
                   "angle_deg": ("angle", "deg", (0.5, 180.0))}
_OVERALL = (("diameter_mm", "diameter", PART_LIMITS["diameter"]), ("length_mm", "length", PART_LIMITS["length"]),
            ("width_mm", "width", PART_LIMITS["width"]), ("height_mm", "height", HEIGHT_LIMITS))
_RA = re.compile(r"(?:ra\s*[=:]?\s*)?(\d+(?:[.,]\d+)?)", re.IGNORECASE)
_NGRADE = re.compile(r"\bN\s*(\d{1,2})\b")


def _ra_um(value):
    """Ra in um from 'Ra 1.6', '1.6', 1.6, 'N7' ('Rz', inches and anything else: None)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
    elif isinstance(value, str):
        t = value.strip()
        m = _NGRADE.search(t)
        if m and not re.search(r"ra", t, re.IGNORECASE):
            return N_GRADE_RA.get(int(m.group(1)))
        if re.search(r"\brz\b|\brt\b|\bin\b|\u00b5in|microinch", t, re.IGNORECASE):
            return None
        m = _RA.search(t)
        if not m:
            return None
        v = float(m.group(1).replace(",", "."))
    else:
        return None
    return round(v, 3) if math.isfinite(v) and 0.01 <= v <= 100 else None


def _feature_row(item, dropped):
    """One row of the feature table from the model's entry, or None. Only vocabulary features are kept; numbers outside their range
    are dropped (and listed in `dropped`), never clipped."""
    from capp import legacy  # noqa: F401
    import feature_vocab as fv
    if not isinstance(item, dict):
        dropped.append(str(item)[:40])
        return None
    name = item.get("feature")
    if not isinstance(name, str) or not fv.is_valid_feature(name):
        dropped.append(str(name)[:40])
        return None
    nums = {}
    for key, (label, unit, (lo, hi)) in FEATURE_NUMBERS.items():
        raw = item.get(key)
        if raw is None:
            continue
        v = _parse_mm(raw, lo, hi) if unit == "mm" else _angle(raw, lo, hi)
        if v is None:
            dropped.append(f"{name} {label} {raw}")
        else:
            nums[key] = v
    text = item.get("text")
    text = " ".join(text.split())[:60] if isinstance(text, str) else ""
    if not nums and not text:
        return None
    return {"feature": name, "text": text, **nums}


def _angle(value, lo, hi):
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        m = _NUMBER.search(value)
        value = float(m.group(1).replace(",", ".")) if m else None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
        return None
    return round(float(value), 2)


def read_part_dimensions(parsed) -> dict:
    """Clean the reply of the part-dimension pass (already JSON-parsed). ok=False if the reply is not a mapping with at least one of the
    expected groups. Never raises. `hole_diameters` is cleaned exactly as read_hole_diameters does (largest first)."""
    out = {"ok": False, "overall": {}, "features": [], "ra_um": [], "hole_diameters": [], "dropped": [], "note": ""}
    # a reply is a dimension reply if it has one of the groups the prompt asks for; "features" alone does not count: the parser of the feature
    # passes answers an unreadable reply with a dict that has a "features" key (an empty list, or the names found by keyword search)
    if not isinstance(parsed, dict) or not ({"overall", KEY, "surface_finish"} & set(parsed)):
        out["note"] = "the reply had no dimensions"
        return out
    note = parsed.get("note", parsed.get("notes", ""))
    out["note"] = " ".join(note.split()) if isinstance(note, str) else ""
    overall = parsed.get("overall") if isinstance(parsed.get("overall"), dict) else {}
    for key, label, (lo, hi) in _OVERALL:
        if overall.get(key) is None:
            continue
        v = _parse_mm(overall[key], lo, hi)
        if v is None:
            out["dropped"].append(f"overall {label} {overall[key]}")
        else:
            out["overall"][label] = v
    holes = read_hole_diameters({KEY: parsed.get(KEY)}) if KEY in parsed else {"hole_diameters": [], "dropped": []}
    out["hole_diameters"] = holes["hole_diameters"]
    out["dropped"] += holes["dropped"]
    items = parsed.get("features")
    if isinstance(items, dict):
        items = [items]
    for item in (items if isinstance(items, (list, tuple)) else [])[:MAX_FEATURE_ROWS]:
        row = _feature_row(item, out["dropped"])
        if row:
            out["features"].append(row)
    finish = parsed.get("surface_finish")
    if isinstance(finish, (str, int, float)) and not isinstance(finish, bool):
        finish = [finish]
    for item in (finish if isinstance(finish, (list, tuple)) else []):
        v = _ra_um(item)
        if v is None:
            out["dropped"].append(f"Ra {item}")
        elif v not in out["ra_um"]:
            out["ra_um"].append(v)
    out["ra_um"].sort()
    out["ok"] = True
    return out


def derive_part_dims(read: dict) -> dict:
    """What the sidebar fields should take, from a read_part_dimensions result. A field is None when the drawing does not say it: the
    caller keeps its default and says so; nothing is invented.

        diameter       the largest OUTSIDE diameter written for the whole part (None for a block or plate)
        length         the overall length written
        width          the second side of a block or plate (None for a round part)
        depth          the deepest depth written next to any feature (a hole 25 deep, a groove 2 deep), within the app's range
        hole_diameter  the largest hole diameter written (choose_hole_diameter)
        ra_um          the strictest (smallest) Ra written, if it is one of the values the app offers
    "source" says in a few words where each value comes from; "notes" lists what could not be used."""
    read = read if isinstance(read, dict) else {}
    overall = read.get("overall") or {}
    out = {k: None for k in PART_KEYS}
    out.update({"hole_diameter": None, "ra_um": None, "source": {}, "notes": []})
    for key in ("diameter", "length", "width"):
        if overall.get(key) is not None:
            out[key] = overall[key]
            out["source"][key] = "overall " + {"diameter": "outside diameter", "length": "length", "width": "width"}[key] + " written on the drawing"
    lo, hi = PART_LIMITS["depth"]
    depths = [(f["depth_mm"], f["feature"]) for f in read.get("features") or [] if lo <= f.get("depth_mm", 0) <= hi]
    if depths:
        d, name = max(depths)
        out["depth"], out["source"]["depth"] = d, f"deepest depth written ({name})"
    elif overall.get("height") is not None:
        out["notes"].append(f"the drawing gives a height of {overall['height']:g} mm but no depth of cut: Depth stays at the default")
    holes = read.get("hole_diameters") or []
    if holes:
        out["hole_diameter"] = choose_hole_diameter(holes)
        out["source"]["hole_diameter"] = "largest hole diameter written" if len(holes) > 1 else "hole diameter written on the drawing"
    ras = read.get("ra_um") or []
    if ras:
        best = min(ras)
        match = [c for c in RA_CHOICES if abs(c - best) < 1e-6]
        if match:
            out["ra_um"], out["source"]["ra_um"] = match[0], "strictest Ra written" if len(ras) > 1 else "Ra written on the drawing"
        else:
            out["notes"].append(f"Ra {best:g} um is written on the drawing but is not one of the values the app offers: choose it yourself")
    return out


_FIELD_LABEL = {"diameter": "Diameter", "length": "Length", "depth": "Depth", "width": "Width", "hole_diameter": "Hole diameter", "ra_um": "Ra"}


def describe_derived(derived: dict):
    """(written, missing): the fields the drawing gave a value for, as text with the value ('Diameter 50 mm', 'Hole Ø10 mm', 'Ra 1.6 um'),
    and the labels of those it did not (they keep the default)."""
    derived = derived or {}
    written, missing = [], []
    for key, label in _FIELD_LABEL.items():
        v = derived.get(key)
        if v is None:
            missing.append(label)
        elif key == "ra_um":
            written.append(f"Ra {v:g} um")
        elif key == "hole_diameter":
            written.append(f"Hole \u00d8{v:g} mm")
        else:
            written.append(f"{label} {v:g} mm")
    return written, missing


def feature_numbers_text(row: dict) -> str:
    """'Ø10, depth 25' style text of the numbers of one feature row (for the table)."""
    parts = []
    for key, (label, unit, _) in FEATURE_NUMBERS.items():
        if key in row:
            v = f"{row[key]:g}"
            parts.append(f"Ø{v}" if key == "diameter_mm" else f"R{v}" if key == "radius_mm" else f"{label} {v}" + (" deg" if unit == "deg" else ""))
    return ", ".join(parts)


def prefill_part(extraction, seen, present=None):
    """({field: value}, signature): what to put in the sidebar fields (diameter, length, depth, width, ra_um) after a
    detection. Same rules as prefill(): one detection is applied once, so the user's edit is not overwritten at the next rerun; a new
    drawing is applied again; and a field that lost its value (Streamlit drops the state of a widget that was not drawn in a run, e.g. in
    Route Only mode) is put back. `present` = {field: still has a value in the session}; missing keys count as present. A field the
    drawing does not give is never touched."""
    if not extraction or not extraction.get("success"):
        return {}, seen
    derived = ((extraction.get("part_dimensions") or {}).get("derived")) or {}
    fields = PART_KEYS + ("ra_um",)            # the hole diameter has its own, older rule (prefill)
    values = {k: derived[k] for k in fields if derived.get(k) is not None}
    if not values:
        return {}, seen
    sig = tuple(derived.get(k) for k in fields)
    if sig == seen:
        present = present or {}
        return {k: v for k, v in values.items() if not present.get(k, True)}, seen
    return values, sig
