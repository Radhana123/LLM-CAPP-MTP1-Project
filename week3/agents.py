# agents.py
# Multi-Agent Evaluation System
# Week 3 | LLM-CAPP Project
#
# Time = Dimension-based machining time + Tool Change + Position Change + Changeover
# Cost = INR (1 USD = Rs.96.095)
# Energy = kWh

import math

# ── USD to INR conversion rate ─────────────────────
USD_TO_INR = 96.095   # 1 USD = Rs.96.095

# ═══════════════════════════════════════════════════
# PENALTY CONSTANTS
# ═══════════════════════════════════════════════════
CHANGEOVER_TIME_MIN    = 15
CHANGEOVER_COST_INR    = round(15 * USD_TO_INR)   # $15 → Rs.1441
CHANGEOVER_ENERGY_KWH  = 0.30

TOOL_CHANGE_TIME_MIN   = 2
TOOL_CHANGE_COST_INR   = round(2 * USD_TO_INR)    # $2  → Rs.192
TOOL_CHANGE_ENERGY_KWH = 0.05

POSITION_CHANGE_TIME_MIN   = 1
POSITION_CHANGE_COST_INR   = round(1 * USD_TO_INR) # $1  → Rs.96
POSITION_CHANGE_ENERGY_KWH = 0.02

# Operations that share same tool family
SAME_TOOL_GROUPS = [
    {"Center Drilling", "Drilling"},
    {"Counterboring", "Countersinking"},
    {"Facing", "Plain/Cylindrical Turning", "Step Turning", "Taper Turning"},
    {"Chamfering", "Corner Rounding/Filleting"},
]

# ═══════════════════════════════════════════════════
# CUTTING PARAMETERS (per material)
# Vc = cutting speed (m/min), f = feed (mm/rev)
# ═══════════════════════════════════════════════════
_CUTTING_PARAMS = {
    #            Vc_turn  f_turn  Vc_drill  f_drill  Vc_mill  fz_mill
    "Aluminum":  (300,    0.25,   120,      0.20,    250,     0.10),
    "Steel":     (120,    0.15,   40,       0.10,    80,      0.06),
    "Brass":     (200,    0.20,   80,       0.15,    150,     0.08),
    "Copper":    (180,    0.20,   70,       0.15,    120,     0.08),
    "Titanium":  (60,     0.10,   20,       0.06,    40,      0.04),
    "Plastic":   (400,    0.30,   150,      0.25,    300,     0.12),
    "Cast Iron": (100,    0.20,   35,       0.12,    70,      0.06),
    # Step 55: more materials (the rows above did not change). Speeds = a v1 row x a ratio (rounded), feeds = typical practice (ASSUMED):
    #   Mild Steel 78/59 of Steel, Tool Steel 42/59, Austenitic Stainless 45/59, Martensitic Stainless 54/59 (machinability ratings,
    #   Engineering ToolBox, 1112 = 100%, 1045 taken as 59%); Alloy Steel 0.70, Hardened Steel 0.40, Inconel 0.25 of Steel (carbide speed ranges of
    #   cnc-optimizations.pages.dev against 304); Nodular Cast Iron 0.80 of Cast Iron (same page, ductile vs grey iron); Bronze 0.75 of Brass,
    #   Hardened Aluminum Alloy 0.85 of Aluminum, Pure Titanium 1.25 of Titanium (ASSUMED); Magnesium 480/360 of Aluminum (ratings).
    "Mild Steel":                  (158,   0.18,   53,       0.12,    106,     0.07),
    "Alloy Steel":                 (84,    0.14,   28,       0.09,    56,      0.05),
    "Tool Steel":                  (85,    0.12,   28,       0.08,    57,      0.05),
    "Hardened Steel":              (48,    0.08,   16,       0.05,    32,      0.03),
    "Austenitic Stainless Steel":  (92,    0.12,   31,       0.08,    61,      0.05),
    "Martensitic Stainless Steel": (110,   0.14,   37,       0.09,    74,      0.05),
    "Nodular Cast Iron":           (80,    0.18,   28,       0.11,    56,      0.06),
    "Bronze":                      (150,   0.18,   60,       0.12,    113,     0.07),
    "Hardened Aluminum Alloy":     (255,   0.22,   102,      0.18,    213,     0.09),
    "Inconel":                     (30,    0.08,   10,       0.05,    20,      0.03),
    "Pure Titanium":               (75,    0.12,   25,       0.07,    50,      0.04),
    "Magnesium":                   (400,   0.30,   160,      0.22,    333,     0.12),
}

# Thread pitch (mm) per operation type
_THREAD_PITCH = {
    "External Threading": 1.5,
    "Tapping": 1.25,
    "Thread Milling": 1.5,
}

# Cutter diameters (mm) for milling ops
_CUTTER_DIA = {
    "Face Milling": 80, "Slab/Peripheral Milling": 63,
    "Slot Milling": 12, "T-Slot Milling": 16,
    "Dovetail Milling": 16, "Woodruff Keyway Milling": 10,
    "Pocket Milling": 16, "Profile Milling": 12,
    "Surface Contouring": 10, "Spotfacing": 25,
    "Corner Rounding/Filleting": 8, "Gear/Spline Milling": 50,
    "Thread Milling": 12, "Angular Milling": 25,
    "Gang Milling": 63, "Form Milling": 25,
    "Helical Milling": 12, "Engraving": 3,
}

# Number of teeth for milling cutters
_CUTTER_TEETH = {
    "Face Milling": 8, "Slab/Peripheral Milling": 6,
    "Slot Milling": 4, "T-Slot Milling": 4,
    "Dovetail Milling": 4, "Woodruff Keyway Milling": 4,
    "Pocket Milling": 4, "Profile Milling": 4,
    "Surface Contouring": 2, "Spotfacing": 4,
    "Corner Rounding/Filleting": 2, "Gear/Spline Milling": 12,
    "Thread Milling": 4, "Angular Milling": 6,
    "Gang Milling": 6, "Form Milling": 4,
    "Helical Milling": 4, "Engraving": 1,
}

# ═══════════════════════════════════════════════════
# MAX DEPTH PER PASS (mm) — per operation type
# Real machining me ek pass me itni hi depth cut hoti hai
# ═══════════════════════════════════════════════════
_MAX_DEPTH_PER_PASS = {
    # Turning
    "Plain/Cylindrical Turning": 3.0,   # 3mm radial depth per pass
    "Step Turning": 3.0,
    "Taper Turning": 2.5,
    "Contour Turning": 1.5,             # Finishing pass — shallow
    "Grooving/Necking": 1.0,            # Narrow grooving tool
    "Internal Grooving": 0.8,
    "Undercutting": 1.0,
    "Eccentric Turning": 2.0,
    "Forming": 1.5,
    "Knurling": 0.5,                    # Forming, not cutting
    # Drilling
    "Drilling": 40.0,                   # Peck drilling — full depth per peck cycle
    "Center Drilling": 5.0,             # Short depth
    "Boring": 2.0,                      # Fine boring — light passes
    "Reaming": 50.0,                    # Reaming — single pass (pre-drilled)
    "Tapping": 50.0,                    # Single pass
    "Counterboring": 5.0,
    "Countersinking": 5.0,
    "Spotfacing": 2.0,
    "External Threading": 50.0,         # Single pass (lathe threading)
    # Milling
    "Face Milling": 3.0,                # 3mm axial depth per pass
    "Slab/Peripheral Milling": 4.0,
    "Slot Milling": 5.0,                # End mill — 5mm per pass
    "T-Slot Milling": 8.0,              # T-slot cutter — full depth
    "Dovetail Milling": 5.0,
    "Woodruff Keyway Milling": 8.0,     # Woodruff — single plunge
    "Pocket Milling": 4.0,              # 4mm axial depth per pass
    "Profile Milling": 5.0,
    "Surface Contouring": 1.0,          # Finishing — very shallow
    "Gear/Spline Milling": 3.0,
    "Thread Milling": 1.5,
    "Angular Milling": 4.0,
    "Gang Milling": 5.0,
    "Form Milling": 3.0,
    "Helical Milling": 3.0,
    "Corner Rounding/Filleting": 3.0,
    "Engraving": 0.3,                   # Very shallow
}


def _num_passes(op: str, depth: float) -> int:
    """
    Depth ko max_depth_per_pass se divide karke passes nikalo.
    """
    max_dp = _MAX_DEPTH_PER_PASS.get(op, 5.0)
    return max(1, math.ceil(depth / max_dp))


# ═══════════════════════════════════════════════════
# DIMENSION-BASED MACHINING TIME (min) PER OPERATION
# ═══════════════════════════════════════════════════

# Step 51: an operation may have dimensions of its own (the Ø of ITS hole, the length of ITS slot). They sit in dims["per_op"] as
# {operation: {"diameter": mm, "length": mm, "depth": mm, "width": mm}}; a key an operation does not list is taken from the part's dims.
# Without "per_op" in dims nothing changes: op_dims returns the dims it was given, so every existing input gives the same time.
PER_OP = "per_op"
HOLE_OPS = ("Drilling", "Counterboring", "Countersinking", "Spotfacing", "Boring", "Reaming", "Tapping")   # = physics.HOLE_DIAMETER_OPS


def op_dims(op: str, dims: dict) -> dict:
    """The dims one operation works with: the part's dims, with the operation's own values (dims["per_op"][op]) laid over them.
    For a hole operation the own "diameter" is the diameter of its hole, so it is also set as "hole_diameter" (the physics model and the
    process plan read that key). dims without "per_op" (or None) are returned as they are, the same object."""
    if not dims or PER_OP not in dims:
        return dims
    mine = (dims[PER_OP] or {}).get(op) or {}
    d = {k: v for k, v in dims.items() if k != PER_OP}
    d.update({k: v for k, v in mine.items() if v is not None})
    if op in HOLE_OPS and mine.get("diameter") is not None:
        d["hole_diameter"] = mine["diameter"]
    return d


def _op_time(op: str, material: str, dims: dict) -> float:
    """
    Calculate machining time for one operation using standard formulas
    WITH multi-pass depth calculation.

    dims = {
        "diameter": mm,   # workpiece/hole diameter
        "length": mm,     # turning length / milling length
        "depth": mm,      # total depth to achieve (split into multiple passes)
        "width": mm,      # milling width of cut
    }
    (optionally dims["per_op"][op] = the same keys for this operation alone: see op_dims)
    """
    dims = op_dims(op, dims)
    p = _CUTTING_PARAMS.get(material, _CUTTING_PARAMS["Steel"])
    Vc_t, f_t, Vc_d, f_d, Vc_m, fz_m = p

    D  = dims.get("diameter", 25.0)   # mm
    L  = dims.get("length", 50.0)     # mm
    dp = dims.get("depth", 20.0)      # mm — TOTAL depth
    W  = dims.get("width", D * 0.8)   # mm

    passes = _num_passes(op, dp)      # how many passes needed

    # ── TURNING GROUP ──────────────────────────────
    if op in ("Facing",):
        N = (Vc_t * 1000) / (math.pi * max(D, 1))
        t = (D / 2) / (f_t * max(N, 1))
        return round(t * passes, 2)

    if op in ("Plain/Cylindrical Turning", "Step Turning", "Contour Turning",
              "Eccentric Turning", "Polishing/Burnishing"):
        N = (Vc_t * 1000) / (math.pi * max(D, 1))
        t = L / (f_t * max(N, 1))
        return round(t * passes, 2)

    if op in ("Taper Turning",):
        taper_length = L * 1.15
        N = (Vc_t * 1000) / (math.pi * max(D, 1))
        t = taper_length / (f_t * max(N, 1))
        return round(t * passes, 2)

    if op in ("Grooving/Necking", "Internal Grooving", "Undercutting"):
        N = (Vc_t * 1000) / (math.pi * max(D, 1))
        t = dp / (f_t * max(N, 1))
        return round(t, 2)   # depth already in formula, no extra × passes

    if op in ("Knurling",):
        N = (Vc_t * 1000) / (math.pi * max(D, 1))
        t = L / (f_t * max(N, 1)) * 1.5
        return round(t * passes, 2)

    if op in ("Parting-off",):
        N = (Vc_t * 1000) / (math.pi * max(D, 1))
        t = (D / 2) / (f_t * max(N, 1))
        return round(t, 2)   # single pass radial plunge

    if op in ("Forming",):
        N = (Vc_t * 1000) / (math.pi * max(D, 1))
        t = dp / (f_t * max(N, 1))
        return round(t * passes, 2)

    # ── DRILLING GROUP ────────────────────────────
    if op in ("Center Drilling",):
        N = (Vc_d * 1000) / (math.pi * max(3, 1))
        t = 5 / (f_d * max(N, 1))
        return round(t, 2)   # fixed short depth, single pass

    if op in ("Drilling", "Boring", "Reaming"):
        N = (Vc_d * 1000) / (math.pi * max(D, 1))
        # Peck drilling — each peck = max_depth_per_pass, retract & re-enter
        peck_depth = _MAX_DEPTH_PER_PASS.get(op, 40.0)
        pecks = max(1, math.ceil(dp / peck_depth))
        t_per_peck = peck_depth / (f_d * max(N, 1))
        retract_time = 0.1 * pecks  # 0.1 min retract per peck
        return round(t_per_peck * pecks + retract_time, 2)

    if op in ("Tapping", "External Threading"):
        pitch = _THREAD_PITCH.get(op, 1.5)
        N = (Vc_d * 1000) / (math.pi * max(D, 1))
        t = (L / pitch) / max(N, 1)
        return round(t, 2)   # single pass

    if op in ("Counterboring", "Countersinking", "Spotfacing"):
        N = (Vc_d * 1000) / (math.pi * max(D, 1))
        t = dp / (f_d * max(N, 1))
        return round(t * passes, 2)

    # ── MILLING GROUP ─────────────────────────────
    if op in _CUTTER_DIA:
        D_c = _CUTTER_DIA[op]
        z   = _CUTTER_TEETH[op]
        N   = (Vc_m * 1000) / (math.pi * max(D_c, 1))
        Vf  = fz_m * z * N  # table feed (mm/min)

        if op in ("Pocket Milling",):
            area = L * W
            step_over = D_c * 0.6
            t_per_pass = area / (Vf * max(step_over, 1))
            return round(t_per_pass * passes, 2)
        elif op in ("Face Milling", "Slab/Peripheral Milling", "Gang Milling"):
            t = (L + D_c) / max(Vf, 1)
            return round(t * passes, 2)
        elif op in ("Surface Contouring", "Profile Milling"):
            t = (L * 2 + W * 2) / max(Vf, 1)
            return round(t * passes, 2)
        elif op in ("Slot Milling", "T-Slot Milling", "Dovetail Milling",
                    "Woodruff Keyway Milling", "Angular Milling"):
            t = L / max(Vf, 1)
            return round(t * passes, 2)
        elif op in ("Thread Milling",):
            pitch = _THREAD_PITCH.get(op, 1.5)
            t = (L / pitch) * (math.pi * D / max(Vf, 1))
            return round(t, 2)   # single helical pass
        elif op in ("Helical Milling",):
            helix_length = math.sqrt((math.pi * D)**2 + dp**2)
            t = helix_length / max(Vf, 1)
            return round(t * passes, 2)
        elif op in ("Gear/Spline Milling",):
            t = (L / max(Vf, 1)) * passes  # each tooth = one pass
            return round(max(t, 0.5), 2)
        else:
            t = L / max(Vf, 1)
            return round(max(t * passes, 0.5), 2)

    # ── FIXED (non-geometric operations) ─────────
    fixed = {
        "Inspection": 5.0,
        "Chamfering": 1.0,
        "Corner Rounding/Filleting": 1.0,
        "Engraving": 3.0,
    }
    return fixed.get(op, 2.0)


# ═══════════════════════════════════════════════════
# BASE COST + ENERGY (1 USD = Rs.96.095)
# ═══════════════════════════════════════════════════
_BASE_COST = {
    "Facing": 961,               # $10
    "Center Drilling": 769,      # $8
    "Drilling": 1441,            # $15
    "Reaming": 1153,             # $12
    "Inspection": 480,           # $5
    "Boring": 1922,              # $20
    "Chamfering": 673,           # $7
    "External Threading": 1730,  # $18
    "Plain/Cylindrical Turning": 1538,  # $16
    "Taper Turning": 1634,              # $17
    "Step Turning": 1538,               # $16
    "Grooving/Necking": 1249,           # $13
    "Parting-off": 865,                 # $9
    "Knurling": 769,                    # $8
    "Forming": 1345,                    # $14
    "Internal Grooving": 1441,          # $15
    "Tapping": 1153,                    # $12
    "Counterboring": 961,               # $10
    "Countersinking": 865,              # $9
    "Contour Turning": 2114,            # $22
    "Undercutting": 961,                # $10
    "Eccentric Turning": 1922,          # $20
    "Polishing/Burnishing": 1057,       # $11
    "Face Milling": 1153,               # $12
    "Slab/Peripheral Milling": 1730,    # $18
    "Surface Contouring": 2691,         # $28
    "Slot Milling": 1345,               # $14
    "T-Slot Milling": 1634,             # $17
    "Dovetail Milling": 1538,           # $16
    "Woodruff Keyway Milling": 1057,    # $11
    "Pocket Milling": 1538,             # $16
    "Profile Milling": 1441,            # $15
    "Spotfacing": 769,                  # $8
    "Corner Rounding/Filleting": 865,   # $9
    "Gear/Spline Milling": 2498,        # $26
    "Thread Milling": 1634,             # $17
    "Angular Milling": 1345,            # $14
    "Gang Milling": 1249,               # $13
    "Form Milling": 1634,               # $17
    "Helical Milling": 1922,            # $20
    "Engraving": 961,                   # $10
}

_BASE_ENERGY = {
    "Facing": 0.30, "Center Drilling": 0.20, "Drilling": 0.50, "Reaming": 0.25,
    "Inspection": 0.05, "Boring": 0.60, "Chamfering": 0.15, "External Threading": 0.40,
    "Plain/Cylindrical Turning": 0.45, "Taper Turning": 0.48, "Step Turning": 0.45,
    "Grooving/Necking": 0.30, "Parting-off": 0.20, "Knurling": 0.15, "Forming": 0.30,
    "Internal Grooving": 0.35, "Tapping": 0.25, "Counterboring": 0.20,
    "Countersinking": 0.18, "Contour Turning": 0.55, "Undercutting": 0.20,
    "Eccentric Turning": 0.50, "Polishing/Burnishing": 0.20,
    "Face Milling": 0.35, "Slab/Peripheral Milling": 0.55, "Surface Contouring": 0.70,
    "Slot Milling": 0.35, "T-Slot Milling": 0.40, "Dovetail Milling": 0.38,
    "Woodruff Keyway Milling": 0.25, "Pocket Milling": 0.45, "Profile Milling": 0.40,
    "Spotfacing": 0.15, "Corner Rounding/Filleting": 0.18, "Gear/Spline Milling": 0.65,
    "Thread Milling": 0.40, "Angular Milling": 0.32, "Gang Milling": 0.50,
    "Form Milling": 0.42, "Helical Milling": 0.50, "Engraving": 0.15,
}

_MATERIAL_COST_FACTOR = {
    "Aluminum": 1.0, "Steel": 1.5, "Brass": 1.2, "Copper": 1.3,
    "Titanium": 2.5, "Plastic": 0.6, "Cast Iron": 1.4,
    # Step 55 (ASSUMED relative to the factors above: harder or dearer material, dearer tooling)
    "Mild Steel": 1.3, "Alloy Steel": 1.8, "Tool Steel": 2.0, "Hardened Steel": 2.4, "Austenitic Stainless Steel": 2.0,
    "Martensitic Stainless Steel": 1.8, "Nodular Cast Iron": 1.5, "Bronze": 1.4, "Hardened Aluminum Alloy": 1.2, "Inconel": 3.0,
    "Pure Titanium": 2.0, "Magnesium": 0.9,
}
_MATERIAL_ENERGY_FACTOR = {
    "Aluminum": 1.0, "Steel": 1.6, "Brass": 1.1, "Copper": 0.95,
    "Titanium": 2.2, "Plastic": 0.5, "Cast Iron": 1.5,
    # Step 55: interpolated (extrapolated outside 700..2110) from the factors above against kc1_1 (capp/materials.py); ASSUMED, not measured
    "Mild Steel": 1.56, "Alloy Steel": 1.82, "Tool Steel": 2.53, "Hardened Steel": 4.65, "Austenitic Stainless Steel": 2.24, "Martensitic Stainless Steel": 1.97, "Nodular Cast Iron": 1.52, "Bronze": 1.0, "Hardened Aluminum Alloy": 1.11, "Inconel": 3.37, "Pure Titanium": 1.8, "Magnesium": 0.72,
}


# ═══════════════════════════════════════════════════
# HELPER — setup events
# ═══════════════════════════════════════════════════
def _same_tool(a: str, b: str) -> bool:
    return any(a in g and b in g for g in SAME_TOOL_GROUPS)


def _count_setup_events(route_steps: list):
    tool_changes = 0
    position_changes = 0
    for i, step in enumerate(route_steps):
        if i == 0:
            continue
        prev = route_steps[i - 1]
        if step == "Inspection":
            pass
        elif prev == "--- Machine Changeover ---":
            pass
        elif _same_tool(prev, step):
            pass
        else:
            tool_changes += 1
        if step in ("Inspection", "--- Machine Changeover ---"):
            pass
        elif prev == "--- Machine Changeover ---":
            pass
        else:
            position_changes += 1
    return tool_changes, position_changes


# ═══════════════════════════════════════════════════
# TIME AGENT (dimension-based)
# ═══════════════════════════════════════════════════
def time_agent(route_steps: list, material: str,
               dims: dict = None) -> float:
    """
    Total time = Σ op_machining_time(dims) + setup penalties.
    dims = {"diameter": mm, "length": mm, "depth": mm, "width": mm}
    If dims is None, uses default reference dimensions (25mm dia, 50mm L, 20mm depth).
    """
    if dims is None:
        dims = {"diameter": 25.0, "length": 50.0, "depth": 20.0, "width": 20.0}

    machining = sum(
        _op_time(s, material, dims)
        for s in route_steps if s != "--- Machine Changeover ---"
    )
    changeovers = route_steps.count("--- Machine Changeover ---")
    tc, pc = _count_setup_events(route_steps)

    total = (machining
             + changeovers * CHANGEOVER_TIME_MIN
             + tc          * TOOL_CHANGE_TIME_MIN
             + pc          * POSITION_CHANGE_TIME_MIN)
    return round(total, 2)


# ═══════════════════════════════════════════════════
# MACHINE COST RATES (₹ per minute)
# = base_cost / base_time for each operation
# This gives cost proportional to actual machining time
# ═══════════════════════════════════════════════════
_COST_PER_MIN = {
    # Lathe operations (₹/min)
    "Facing": round(160 * 96.095/95.33),      "Center Drilling": round(220 * 96.095/95.33),
    "Drilling": round(179 * 96.095/95.33),    "Reaming": round(286 * 96.095/95.33),
    "Boring": round(191 * 96.095/95.33),      "Chamfering": round(222 * 96.095/95.33),
    "External Threading": round(245 * 96.095/95.33), "Plain/Cylindrical Turning": round(169 * 96.095/95.33),
    "Taper Turning": round(162 * 96.095/95.33), "Step Turning": round(169 * 96.095/95.33),
    "Grooving/Necking": round(207 * 96.095/95.33), "Parting-off": round(215 * 96.095/95.33),
    "Knurling": round(254 * 96.095/95.33),    "Forming": round(223 * 96.095/95.33),
    "Internal Grooving": round(204 * 96.095/95.33), "Tapping": round(229 * 96.095/95.33),
    "Counterboring": round(238 * 96.095/95.33), "Countersinking": round(286 * 96.095/95.33),
    "Contour Turning": round(175 * 96.095/95.33), "Undercutting": round(238 * 96.095/95.33),
    "Eccentric Turning": round(174 * 96.095/95.33), "Polishing/Burnishing": round(175 * 96.095/95.33),
    # Milling operations (₹/min)
    "Face Milling": round(191 * 96.095/95.33), "Slab/Peripheral Milling": round(172 * 96.095/95.33),
    "Surface Contouring": round(178 * 96.095/95.33), "Slot Milling": round(191 * 96.095/95.33),
    "T-Slot Milling": round(180 * 96.095/95.33), "Dovetail Milling": round(191 * 96.095/95.33),
    "Woodruff Keyway Milling": round(210 * 96.095/95.33), "Pocket Milling": round(169 * 96.095/95.33),
    "Profile Milling": round(179 * 96.095/95.33), "Spotfacing": round(254 * 96.095/95.33),
    "Corner Rounding/Filleting": round(215 * 96.095/95.33), "Gear/Spline Milling": round(177 * 96.095/95.33),
    "Thread Milling": round(203 * 96.095/95.33), "Angular Milling": round(191 * 96.095/95.33),
    "Gang Milling": round(207 * 96.095/95.33), "Form Milling": round(180 * 96.095/95.33),
    "Helical Milling": round(173 * 96.095/95.33), "Engraving": round(159 * 96.095/95.33),
    "Inspection": round(95 * 96.095/95.33),
}

# ═══════════════════════════════════════════════════
# COST AGENT (INR, dimension-based)
# cost = actual_machining_time × cost_per_min × material_factor
# ═══════════════════════════════════════════════════
def cost_agent(route_steps: list, material: str,
               batch_size: int, dims: dict = None) -> float:
    """
    Cost = Σ (op_machining_time × cost_per_min) × material_factor
           + setup costs (tool change, position change, changeover)
    Dimension-aware: bigger part = more time = more cost.
    """
    if dims is None:
        dims = {"diameter": 25.0, "length": 50.0, "depth": 20.0, "width": 20.0}

    factor = _MATERIAL_COST_FACTOR.get(material, 1.0)

    # Machining cost = time × rate per minute
    machining_cost = sum(
        _op_time(s, material, dims) * _COST_PER_MIN.get(s, 180)
        for s in route_steps
        if s != "--- Machine Changeover ---"
    )

    changeovers = route_steps.count("--- Machine Changeover ---")
    tc, pc = _count_setup_events(route_steps)

    total = (machining_cost * factor
             + changeovers * CHANGEOVER_COST_INR
             + tc          * TOOL_CHANGE_COST_INR
             + pc          * POSITION_CHANGE_COST_INR)

    if batch_size > 500:
        total *= 0.85
    elif batch_size > 100:
        total *= 0.92
    return round(total, 2)


# ═══════════════════════════════════════════════════
# ENERGY AGENT
# ═══════════════════════════════════════════════════
def energy_agent(route_steps: list, material: str,
                 dims: dict = None) -> float:
    factor = _MATERIAL_ENERGY_FACTOR.get(material, 1.0)
    machining = sum(
        _BASE_ENERGY.get(s, 0.3) for s in route_steps
        if s != "--- Machine Changeover ---"
    )
    changeovers = route_steps.count("--- Machine Changeover ---")
    tc, pc = _count_setup_events(route_steps)

    total = (machining
             + changeovers * CHANGEOVER_ENERGY_KWH
             + tc          * TOOL_CHANGE_ENERGY_KWH
             + pc          * POSITION_CHANGE_ENERGY_KWH) * factor
    return round(total, 2)


# ═══════════════════════════════════════════════════
# EFFICIENCY AGENT
# ═══════════════════════════════════════════════════
def efficiency_agent(time_val: float, cost_val: float,
                     energy_val: float) -> float:
    ts = max(0, 100 - time_val  * 1.2)
    cs = max(0, 100 - cost_val  * 0.005)
    es = max(0, 100 - energy_val * 15)
    return round((ts + cs + es) / 3, 2)


# ═══════════════════════════════════════════════════
# MASTER EVALUATE
# ═══════════════════════════════════════════════════
def evaluate_route(route_name: str, route_steps: list,
                   material: str, batch_size: int,
                   dims: dict = None) -> dict:
    t   = time_agent(route_steps, material, dims)
    c   = cost_agent(route_steps, material, batch_size, dims)
    e   = energy_agent(route_steps, material, dims)
    eff = efficiency_agent(t, c, e)
    return {
        "route_name": route_name, "steps": route_steps,
        "time_min": t, "cost_inr": c,
        "energy_kwh": e, "efficiency_score": eff,
    }


# ═══════════════════════════════════════════════════
# TIME BREAKDOWN (for UI)
# ═══════════════════════════════════════════════════
def time_breakdown(route_steps: list, material: str,
                   dims: dict = None) -> dict:
    if dims is None:
        dims = {"diameter": 25.0, "length": 50.0, "depth": 20.0, "width": 20.0}

    machining = sum(
        _op_time(s, material, dims)
        for s in route_steps if s != "--- Machine Changeover ---"
    )
    changeovers = route_steps.count("--- Machine Changeover ---")
    tc, pc = _count_setup_events(route_steps)

    return {
        "machining_time":       round(machining, 2),
        "tool_change_time":     round(tc * TOOL_CHANGE_TIME_MIN, 2),
        "position_change_time": round(pc * POSITION_CHANGE_TIME_MIN, 2),
        "changeover_time":      round(changeovers * CHANGEOVER_TIME_MIN, 2),
        "tool_changes":         tc,
        "position_changes":     pc,
        "changeovers":          changeovers,
        "material_factor":      1.0,
        "dims_used":            dims,
    }


# ═══════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Dimension-Based Time Test ===\n")

    route = ["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"]

    for mat in ["Aluminum", "Steel", "Titanium"]:
        for D, L, d in [(10, 30, 15), (25, 50, 20), (50, 100, 40)]:
            dims = {"diameter": D, "length": L, "depth": d, "width": D*0.8}
            t = time_agent(route, mat, dims)
            bd = time_breakdown(route, mat, dims)
            print(f"{mat:10} D={D}mm L={L}mm d={d}mm => "
                  f"Total={t}min "
                  f"(Mach={bd['machining_time']} "
                  f"TC={bd['tool_change_time']} "
                  f"PC={bd['position_change_time']})")
    print()
    print("=== Same route, different dimensions — time changes correctly ===")
    route2 = ["Facing", "Plain/Cylindrical Turning", "Grooving/Necking",
              "--- Machine Changeover ---", "Slot Milling", "Inspection"]
    for D, L, d in [(20, 40, 10), (50, 150, 25), (100, 300, 50)]:
        dims = {"diameter": D, "length": L, "depth": d, "width": D*0.5}
        t = time_agent(route2, "Steel", dims)
        print(f"  Steel D={D}mm L={L}mm depth={d}mm => {t} min")