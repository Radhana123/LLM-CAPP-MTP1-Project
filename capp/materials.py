"""Material constants for the physics-based scoring upgrade (used from Step 7: energy, tool wear, surface finish).

Kept separate from week3/agents.py on purpose: the v1 agents (baseline B0/B1/RB) must not change, the new
physics agents will import this table.

Where each number comes from (be honest about this in the thesis):
  kc1_1, mc   Specific cutting force at 1 mm chip thickness (N/mm^2) and Kienzle exponent, Sandvik-style table as
              published on machiningdoctor.com ("specific cutting force kc / kc1.1"). Group used is given in `kc_group`.
              Cutting force at chip thickness h:  kc = kc1_1 * h**(-mc).
              Plastic is NOT in that table: its value is an order-of-magnitude ASSUMPTION (flagged below). Magnesium (Step 55) is not in it either:
              its kc1_1 is an ASSUMPTION (magnesium has the lowest specific cutting energy of the structural metals, below aluminium).
              The table gives one group for "copper and Cu alloys (incl. brass)", so Brass and Copper share a value.
  taylor_n    Taylor exponent for V * T**n = C. Uncoated carbide is typically 0.20 to 0.25 (HSS 0.08 to 0.15,
              ceramics 0.40 to 0.60). The Time agent's cutting speeds are carbide-type speeds, so carbide range is used.
  t_ref_min   Tool life (min) at the Time agent's reference turning speed. ASSUMPTION (typical carbide edge life
              is tens of minutes); makes harder materials wear faster at their own reference speed. Calibrate or
              run a sensitivity analysis before quoting absolute wear costs.
  tool_edge_cost_inr   Cost of one cutting edge (INR). ASSUMPTION, same remark.

Taylor constant C is not stored: it depends on the reference speed, which lives in week3/agents.py
(`_CUTTING_PARAMS`). Use `taylor_constant(material, vc_ref)`.
"""

MATERIAL_PROPS = {
    "Aluminum": {
        "kc1_1": 700, "mc": 0.25, "kc_group": "wrought aluminium, not heat-treated",
        "taylor_n": 0.25, "t_ref_min": 90, "tool_edge_cost_inr": 150,
    },
    "Steel": {
        "kc1_1": 1500, "mc": 0.22, "kc_group": "non-alloyed steel, ~0.45% C, annealed",
        "taylor_n": 0.25, "t_ref_min": 45, "tool_edge_cost_inr": 250,
    },
    "Brass": {
        "kc1_1": 700, "mc": 0.27, "kc_group": "copper and Cu alloys (incl. brass)",
        "taylor_n": 0.25, "t_ref_min": 90, "tool_edge_cost_inr": 150,
    },
    "Copper": {
        "kc1_1": 700, "mc": 0.27, "kc_group": "copper and Cu alloys (incl. brass)",
        "taylor_n": 0.25, "t_ref_min": 60, "tool_edge_cost_inr": 150,
    },
    "Titanium": {
        "kc1_1": 2110, "mc": 0.22, "kc_group": "titanium alpha+beta alloys",
        "taylor_n": 0.20, "t_ref_min": 20, "tool_edge_cost_inr": 400,
    },
    "Plastic": {
        "kc1_1": 200, "mc": 0.25, "kc_group": "ASSUMED (not in the source table)",
        "taylor_n": 0.25, "t_ref_min": 120, "tool_edge_cost_inr": 120,
    },
    "Cast Iron": {
        "kc1_1": 1150, "mc": 0.20, "kc_group": "grey cast iron, pearlitic/ferritic",
        "taylor_n": 0.25, "t_ref_min": 45, "tool_edge_cost_inr": 250,
    },
    # ---- Step 55: twelve more materials. kc1_1 / mc / kc_group are rows of the same machiningdoctor.com table as above (except Magnesium: no row there).
    "Mild Steel": {
        "kc1_1": 1350, "mc": 0.21, "kc_group": "non-alloyed steel, ~0.15% C, annealed",
        "taylor_n": 0.25, "t_ref_min": 60, "tool_edge_cost_inr": 250,
    },
    "Alloy Steel": {
        "kc1_1": 1725, "mc": 0.24, "kc_group": "low-alloyed steel, quenched and tempered",
        "taylor_n": 0.25, "t_ref_min": 35, "tool_edge_cost_inr": 280,
    },
    "Tool Steel": {
        "kc1_1": 2450, "mc": 0.23, "kc_group": "high-alloyed steel, tool steel, annealed",
        "taylor_n": 0.22, "t_ref_min": 30, "tool_edge_cost_inr": 320,
    },
    "Hardened Steel": {
        "kc1_1": 4600, "mc": 0.25, "kc_group": "hardened steel",
        "taylor_n": 0.20, "t_ref_min": 15, "tool_edge_cost_inr": 450,
    },
    "Austenitic Stainless Steel": {
        "kc1_1": 2150, "mc": 0.20, "kc_group": "stainless steel, austenitic",
        "taylor_n": 0.22, "t_ref_min": 30, "tool_edge_cost_inr": 300,
    },
    "Martensitic Stainless Steel": {
        "kc1_1": 1875, "mc": 0.21, "kc_group": "stainless steel, ferritic / martensitic, annealed",
        "taylor_n": 0.22, "t_ref_min": 40, "tool_edge_cost_inr": 280,
    },
    "Nodular Cast Iron": {
        "kc1_1": 1225, "mc": 0.25, "kc_group": "nodular cast iron, ferritic",
        "taylor_n": 0.25, "t_ref_min": 45, "tool_edge_cost_inr": 250,
    },
    "Bronze": {
        "kc1_1": 700, "mc": 0.27, "kc_group": "copper and Cu alloys (CuSn, lead-free)",
        "taylor_n": 0.25, "t_ref_min": 70, "tool_edge_cost_inr": 150,
    },
    "Hardened Aluminum Alloy": {
        "kc1_1": 800, "mc": 0.25, "kc_group": "wrought aluminium, alloyed, curable, hardened",
        "taylor_n": 0.25, "t_ref_min": 70, "tool_edge_cost_inr": 150,
    },
    "Inconel": {
        "kc1_1": 3300, "mc": 0.24, "kc_group": "heat-resistant super alloys, Ni or Co based",
        "taylor_n": 0.20, "t_ref_min": 12, "tool_edge_cost_inr": 450,
    },
    "Pure Titanium": {
        "kc1_1": 1700, "mc": 0.23, "kc_group": "titanium alloys, pure titanium",
        "taylor_n": 0.20, "t_ref_min": 25, "tool_edge_cost_inr": 380,
    },
    "Magnesium": {
        "kc1_1": 450, "mc": 0.25, "kc_group": "ASSUMED (not in the source table)",
        "taylor_n": 0.25, "t_ref_min": 120, "tool_edge_cost_inr": 150,
    },
}

ASSUMED_FIELDS = ("t_ref_min", "tool_edge_cost_inr")   # not from a published table, for every material
ASSUMED_MATERIALS_KC = ("Plastic", "Magnesium")        # kc1_1 / mc assumed for these


def props(material: str) -> dict:
    """Constants for one material. Raises KeyError for an unknown material (no silent default)."""
    return MATERIAL_PROPS[material]


def kc(material: str, chip_thickness_mm: float) -> float:
    """Specific cutting force (N/mm^2) at a given chip thickness (mm), Kienzle: kc1_1 * h**(-mc)."""
    if chip_thickness_mm <= 0:
        raise ValueError("chip_thickness_mm must be > 0")
    p = MATERIAL_PROPS[material]
    return p["kc1_1"] * chip_thickness_mm ** (-p["mc"])


def taylor_constant(material: str, vc_ref_m_min: float) -> float:
    """C in V * T**n = C, chosen so that tool life equals t_ref_min at the reference speed vc_ref (m/min)."""
    p = MATERIAL_PROPS[material]
    return vc_ref_m_min * p["t_ref_min"] ** p["taylor_n"]


def tool_life_min(material: str, vc_m_min: float, vc_ref_m_min: float) -> float:
    """Tool life (min) at cutting speed vc, given the reference speed that defines C."""
    p = MATERIAL_PROPS[material]
    c = taylor_constant(material, vc_ref_m_min)
    return (c / vc_m_min) ** (1.0 / p["taylor_n"])
