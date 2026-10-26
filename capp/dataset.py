"""Dataset v2: part schema, validation, load/save, summary.

A part record (JSON object):
  part_id            "DEV_001" / "TEST_001"
  split              "dev" | "test"
  archetype          archetype name (capp/archetypes.py)
  in_distribution    True for dev parts and for test parts whose archetype is also in the dev pool;
                     False for test parts from held-out archetypes (feature combination never seen in dev)
  material           one of the 7 materials
  features           list of vocabulary features (no duplicates)
  tolerance          "0.005mm" ... "0.5mm"
  batch_size         int, 1..10000 (UI slider range)
  dims               {"diameter","length","depth","width"} in mm, same keys and UI limits as the Streamlit app
  required_ra_um     required surface roughness Ra (um), derived from tolerance (see TOLERANCE_TO_RA_UM)
  machine_preference "auto" | "prefer_lathe" | "prefer_milling" (the UI's machine strategy)
  --- derived (stored for easy analysis, validated against the recomputed value) ---
  n_features, batch_category, machine_profile ("lathe" | "milling" | "mixed" | "flexible")
  --- optional ---
  reference_routes   list of expert-accepted routes (each a list of operations). Filled for the expert benchmark (WP2).

Everything here is plain Python (no I/O except load/save), so the evaluation harness can import it freely.
"""
import csv
import json
import os
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEEK1 = os.path.join(_ROOT, "week1")
if _WEEK1 not in sys.path:
    sys.path.insert(0, _WEEK1)

import feature_vocab as fv                     # noqa: E402  (week1 modules use flat imports)
from material_tokens import (                   # noqa: E402
    MATERIALS, TOLERANCE_CATEGORIES, get_batch_category,
)

DATA_DIR = os.path.join(_ROOT, "data")

SPLITS = ("dev", "test")
MACHINE_PREFERENCES = ("auto", "prefer_lathe", "prefer_milling")
DIM_KEYS = ("diameter", "length", "depth", "width")

# Same limits as the sliders / number inputs in week6/app.py.
DIM_LIMITS = {"diameter": (1.0, 500.0), "length": (1.0, 1000.0), "depth": (1.0, 300.0), "width": (1.0, 500.0)}
BATCH_LIMITS = (1, 10000)

# Required Ra from the tolerance (ISO 1302 preferred Ra values). A common rule of thumb, not a standard:
# tighter tolerance -> finer surface. Used by the Ra-risk verifier rule in Step 7; kept here so it is part of the data.
TOLERANCE_TO_RA_UM = {
    "0.005mm": 0.4,
    "0.01mm": 0.8,
    "0.02mm": 1.6,
    "0.05mm": 3.2,
    "0.1mm": 6.3,
    "0.5mm": 12.5,
}

REQUIRED_FIELDS = (
    "part_id", "split", "archetype", "in_distribution", "material", "features", "tolerance", "batch_size",
    "dims", "required_ra_um", "machine_preference", "n_features", "batch_category", "machine_profile",
)


def required_ra(tolerance: str) -> float:
    return TOLERANCE_TO_RA_UM[tolerance]


def machine_profile(features: list) -> str:
    """lathe / milling / mixed (needs both machines) / flexible (only features that can go on either)."""
    kinds = {fv.get_machine_type(f) for f in features}
    lathe, mill = "Lathe" in kinds, "Milling" in kinds
    if lathe and mill:
        return "mixed"
    if lathe:
        return "lathe"
    if mill:
        return "milling"
    return "flexible"


def derived_fields(part: dict) -> dict:
    return {
        "n_features": len(part["features"]),
        "batch_category": get_batch_category(part["batch_size"]),
        "machine_profile": machine_profile(part["features"]),
    }


def validate_part(part: dict) -> list:
    """Return a list of error strings (empty list = valid). Never raises on bad content."""
    errs = []
    for k in REQUIRED_FIELDS:
        if k not in part:
            errs.append(f"missing field '{k}'")
    if errs:
        return errs

    if part["split"] not in SPLITS:
        errs.append(f"split '{part['split']}' not in {SPLITS}")
    if part["material"] not in MATERIALS:
        errs.append(f"material '{part['material']}' invalid")

    feats = part["features"]
    if not isinstance(feats, list) or not feats:
        errs.append("features must be a non-empty list")
    else:
        bad = [f for f in feats if not fv.is_valid_feature(f)]
        if bad:
            errs.append(f"unknown features {bad}")
        if len(set(feats)) != len(feats):
            errs.append("duplicate features")

    if part["tolerance"] not in TOLERANCE_CATEGORIES:
        errs.append(f"tolerance '{part['tolerance']}' invalid")

    b = part["batch_size"]
    if not isinstance(b, int) or isinstance(b, bool) or not (BATCH_LIMITS[0] <= b <= BATCH_LIMITS[1]):
        errs.append(f"batch_size {b!r} must be an int in {BATCH_LIMITS}")

    dims = part["dims"]
    if not isinstance(dims, dict) or set(dims) != set(DIM_KEYS):
        errs.append(f"dims must have exactly the keys {DIM_KEYS}")
    else:
        for k in DIM_KEYS:
            lo, hi = DIM_LIMITS[k]
            v = dims[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not (lo <= v <= hi):
                errs.append(f"dims.{k}={v!r} outside [{lo}, {hi}]")

    if part["machine_preference"] not in MACHINE_PREFERENCES:
        errs.append(f"machine_preference '{part['machine_preference']}' invalid")

    if not errs:   # derived checks only make sense when the inputs above are valid
        if part["required_ra_um"] != required_ra(part["tolerance"]):
            errs.append(f"required_ra_um {part['required_ra_um']} != {required_ra(part['tolerance'])} for {part['tolerance']}")
        for k, v in derived_fields(part).items():
            if part[k] != v:
                errs.append(f"derived field {k}={part[k]!r}, expected {v!r}")

    ref = part.get("reference_routes")
    if ref is not None:
        if not isinstance(ref, list) or not all(isinstance(r, list) and r and all(isinstance(s, str) for s in r) for r in ref):
            errs.append("reference_routes must be a list of non-empty lists of operation names")
    return errs


def validate_dataset(parts: list) -> list:
    """Per-part errors plus dataset-level errors (duplicate ids). Returns list of 'part_id: error' strings."""
    errs = []
    ids = Counter(p.get("part_id") for p in parts)
    for pid, n in ids.items():
        if n > 1:
            errs.append(f"{pid}: duplicated {n} times")
    for p in parts:
        for e in validate_part(p):
            errs.append(f"{p.get('part_id')}: {e}")
    return errs


def feature_coverage(parts: list) -> dict:
    """{'covered': n, 'total': 36, 'missing': [...], 'counts': {feature: n_parts}}"""
    counts = Counter(f for p in parts for f in p["features"])
    allf = fv.get_all_features()
    return {
        "covered": sum(1 for f in allf if counts[f] > 0),
        "total": len(allf),
        "missing": [f for f in allf if counts[f] == 0],
        "counts": {f: counts[f] for f in allf},
    }


def summarize(parts: list) -> dict:
    cov = feature_coverage(parts)
    n_feat = Counter(p["n_features"] for p in parts)
    return {
        "n_parts": len(parts),
        "feature_coverage": f"{cov['covered']}/{cov['total']}",
        "missing_features": cov["missing"],
        "materials": dict(Counter(p["material"] for p in parts)),
        "tolerances": dict(Counter(p["tolerance"] for p in parts)),
        "batch_categories": dict(Counter(p["batch_category"] for p in parts)),
        "machine_profiles": dict(Counter(p["machine_profile"] for p in parts)),
        "n_features_hist": dict(sorted(n_feat.items())),
        "archetypes": len({p["archetype"] for p in parts}),
        "in_distribution": sum(1 for p in parts if p["in_distribution"]),
    }


# ---- I/O -----------------------------------------------------------------------------------------------
def dataset_path(name: str, ext: str = "json") -> str:
    return os.path.join(DATA_DIR, f"{name}.{ext}")


def save_dataset(parts: list, name_or_path: str) -> str:
    """Write JSON (stable formatting, so git diffs stay small) and a flat CSV next to it. Returns the JSON path."""
    path = name_or_path if name_or_path.endswith(".json") else dataset_path(name_or_path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(parts, f, indent=2, ensure_ascii=False)
        f.write("\n")
    csv_path = path[:-5] + ".csv"
    cols = ["part_id", "split", "archetype", "in_distribution", "material", "features", "tolerance",
            "batch_size", "diameter", "length", "depth", "width", "required_ra_um", "machine_preference",
            "n_features", "batch_category", "machine_profile"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in parts:
            row = dict(p, features=";".join(p["features"]), **p["dims"])
            w.writerow([row[c] for c in cols])
    return path


def load_dataset(name_or_path: str, validate: bool = True) -> list:
    """Load by short name ('dev_v2', 'test_v2') or by path. Raises ValueError if validation fails."""
    path = name_or_path if name_or_path.endswith(".json") else dataset_path(name_or_path)
    with open(path, encoding="utf-8") as f:
        parts = json.load(f)
    if validate:
        errs = validate_dataset(parts)
        if errs:
            raise ValueError(f"{path}: {len(errs)} validation errors, first: {errs[:3]}")
    return parts
