"""Physics agents for "Route + Full Analysis": specific cutting energy kc, Taylor tool-wear cost, surface-roughness (Ra) risk.

    from capp import physics
    a = physics.analyse_route(steps, "Steel", {"diameter": 50, "length": 100, "depth": 15, "width": 40},
                              batch_size=100, required_ra_um=1.6)
    a["totals"]["cutting_energy_kwh"], a["totals"]["wear_cost_inr"], a["totals"]["ra"]["risk"]

This is an ANALYSIS layer. It never changes which route is chosen: the v1 Time/Cost/Energy/Efficiency agents (week3/agents.py)
stay the objectives of selection, so the baselines B0/B1/RB are untouched (report 6.7: "the route is selected first and the
parameters second"). It reuses the v1 cutting speeds, feeds, passes and machining times, so every number here is consistent
with the Time agent; what it adds is the physics that the v1 Energy and Cost agents replace by per-operation constants.

Model, per operation (units: mm, min, m/min, N/mm^2, kW, kWh, INR)
  Cutting time   t = agents._op_time(...), the Time agent's machining time.
  Chip thickness h from the feed and the tool geometry (turning h = f; drilling h = f/2 * sin(point half-angle); reaming
                 h = f / teeth; milling h = fz * sqrt(ae / D_cutter)).
  Specific cutting force   kc = kc1.1 * h**(-mc)  (Kienzle, capp/materials.py). 1 N/mm^2 = 1 mJ/mm^3, i.e. kc is the specific
                 cutting energy of the operation (steel ~ 2 J/mm^3).
  Material removal rate    Q (mm^3/min): turning Vc*1000*f*ap; plunging Vc*1000*f*w; drilling (pi/4) d^2 f N; milling ae*ap*Vf;
                 boring/reaming Vc*1000*f*(radial stock); threads: thread volume / t.
  Cutting power  Pc = kc*Q/6e7 kW;  spindle power Pc/eta;  utilisation = spindle power / machine power.
  Cutting energy E = Pc * t / 60 kWh.   This is the energy of the cut itself, a lower bound of the electrical energy: idle
                 power, spindle acceleration, coolant and auxiliaries are not modelled.
  Taylor   V * T**n = C. Each operation runs at the speed the Time agent uses for its process; the handbook speed of a process
                 is assumed to give the material's reference tool life t_ref (capp/materials.py). A speed factor
                 (vc_factor) changes T by vc_factor**(-1/n) and the cutting time by 1/vc_factor: a what-if / sensitivity analysis.
                 Wear fraction = t / T; wear cost = wear fraction * cost of one cutting edge. At a speed factor k the cut takes t/k
                 (same feed and depth), so the energy of the cut is unchanged, the power is k times higher and the wear fraction
                 grows with k**(1/n - 1) (Taylor, per part): faster is not free.
  Ra (best case)  Turning, facing, boring: Ra = f^2 / (32 r) (feed marks, corner radius r). Milling: fz^2 / (32 r) with the insert
                 corner radius (face milling) or the cutter radius D/2 (end milling). Practical roughness is worse (stability,
                 wear, built-up edge), so the result is floored at a typical best-practice value of the process (RA_FLOOR_UM).
                 Ra risk compares the finest Ra the route can produce with the required Ra (ratio <= 0.5 low, <= 1.0 medium,
                 above 1.0 high: the route's parameters cannot produce the required finish).

What is assumed, not sourced, is listed in ASSUMPTIONS and must be reported as such. The dataset has no hole diameter: like the Time
agent, hole-making operations use the part diameter unless dims has a "hole_diameter". Operations that form rather than cut
(Knurling, Forming, Polishing/Burnishing) and Inspection have no cutting energy or wear; their time is still returned.
"""
import math

from capp import legacy  # noqa: F401
from capp import materials as MAT

import agents

# ---- assumptions (every one of these is a parameter; none is a measurement) -------------------------------------------------
SPINDLE_POWER_KW = 7.5            # machine power available at the spindle (typical small CNC lathe / machining centre)
POWER_EFFICIENCY = 0.8            # share of spindle power that reaches the cut
POWER_WARN = 0.8                  # utilisation above this is "near the limit"
DRILL_HALF_POINT_ANGLE_DEG = 59.0
PLUNGE_TOOL_WIDTH_MM = {"Grooving/Necking": 3.0, "Internal Grooving": 2.0, "Undercutting": 3.0, "Parting-off": 3.0}
BORING_STOCK_MM = 1.0             # radial stock removed by a boring pass
REAMING_STOCK_MM = 0.10           # radial stock removed by a reamer
REAMER_TEETH = 6
THREAD_CHIP_MM = 0.05             # chip thickness used for kc in thread cutting
CHAMFER_DEPTH_MM = 1.0            # 1 x 45 degrees
NOSE_RADIUS_MM = {"turning": 0.8, "boring": 0.4, "face_mill": 0.8}
MILL_ENGAGEMENT = {"slot": 1.0, "face": 0.75, "other": 0.5}       # ae / D_cutter
RA_FLOOR_UM = {"turning": 0.4, "boring": 0.8, "mill": 0.8, "drill": 3.2, "ream": 0.8}
RA_MEDIUM_RATIO = 0.5
RA_HIGH_RATIO = 1.0
MIN_CHIP_MM = 0.005

ASSUMPTIONS = {
    "SPINDLE_POWER_KW": "typical machine size, not the machine of any real shop",
    "POWER_EFFICIENCY": "typical value, not measured",
    "PLUNGE_TOOL_WIDTH_MM": "typical grooving/parting insert widths",
    "BORING_STOCK_MM": "finish-bore allowance", "REAMING_STOCK_MM": "typical reaming allowance",
    "NOSE_RADIUS_MM": "ISO insert corner radii 0.4 and 0.8 mm",
    "MILL_ENGAGEMENT": "usual radial engagement of slotting, face and contour cuts",
    "RA_FLOOR_UM": "typical best-practice roughness of the process; check against a handbook before quoting",
    "t_ref_min, tool_edge_cost_inr (capp/materials.py)": "assumed for every material",
    "kc1_1, mc for Plastic (capp/materials.py)": "assumed, not in the source table",
    "tool diameter of hole operations": "the part diameter, as in the v1 Time agent (dataset v2 has no hole diameter)",
}

TURNING_OPS = ("Facing", "Plain/Cylindrical Turning", "Step Turning", "Taper Turning", "Contour Turning", "Eccentric Turning")
DRILL_OPS = ("Drilling", "Counterboring", "Countersinking", "Spotfacing")
THREAD_OPS = ("Tapping", "External Threading", "Thread Milling")
FORMING_OPS = ("Knurling", "Forming", "Polishing/Burnishing")
HOLE_DIAMETER_OPS = DRILL_OPS + ("Boring", "Reaming", "Tapping")
SLOT_LIKE = ("Slot Milling", "T-Slot Milling", "Dovetail Milling", "Woodruff Keyway Milling")
FACE_LIKE = ("Face Milling", "Slab/Peripheral Milling", "Gang Milling")
MARKER = "--- Machine Changeover ---"
DEFAULT_DIMS = {"diameter": 25.0, "length": 50.0, "depth": 20.0, "width": 20.0}      # the Time agent's defaults


def op_process(op: str) -> str:
    """Process class of an operation (how its physics is modelled)."""
    if op in TURNING_OPS:
        return "turning"
    if op == "Chamfering":
        return "chamfer"
    if op in PLUNGE_TOOL_WIDTH_MM:
        return "plunge"
    if op == "Center Drilling":
        return "center_drill"
    if op in DRILL_OPS:
        return "drill"
    if op == "Boring":
        return "bore"
    if op == "Reaming":
        return "ream"
    if op in THREAD_OPS:
        return "thread"
    if op in FORMING_OPS:
        return "forming"
    if op == "Inspection":
        return "none"
    if op in agents._CUTTER_DIA:
        return "mill"
    return "unknown"


def _dims(dims):
    d = dict(DEFAULT_DIMS)
    d.update({k: v for k, v in (dims or {}).items() if v is not None})
    if not dims or dims.get("width") is None:
        d["width"] = d["diameter"] * 0.8                    # as agents._op_time (0.8 * 25 = the default 20)
    return d


def _speed_feed(process: str, op: str, material: str):
    """(Vc, feed, v1 speed class) exactly as the Time agent picks them."""
    Vc_t, f_t, Vc_d, f_d, Vc_m, fz_m = agents._CUTTING_PARAMS[material]
    if process in ("turning", "chamfer", "plunge", "forming"):
        return Vc_t, f_t
    if process in ("drill", "center_drill", "bore", "ream", "thread") and op != "Thread Milling":
        return Vc_d, f_d
    return Vc_m, fz_m


def _kc(material, h):
    return MAT.kc(material, max(h, MIN_CHIP_MM))


def analyse_op(op: str, material: str, dims: dict = None, vc_factor: float = 1.0, machine_power_kw: float = SPINDLE_POWER_KW) -> dict:
    """One operation's physics. See the module docstring for the model."""
    if material not in MAT.MATERIAL_PROPS:
        raise KeyError(f"unknown material {material!r}")
    if vc_factor <= 0:
        raise ValueError("vc_factor must be > 0")
    d = _dims(dims)
    process = op_process(op)
    hole_d = d.get("hole_diameter") or d["diameter"]
    time_dims = dict(d, diameter=hole_d) if op in HOLE_DIAMETER_OPS else d
    row = {"op": op, "process": process, "t_cut_min": None, "vc_m_min": None, "feed": None, "ap_mm": None, "ae_mm": None,
           "chip_mm": None, "kc_n_mm2": None, "mrr_cm3_min": None, "power_kw": None, "spindle_kw": None, "utilisation": None,
           "energy_kwh": None, "tool_life_min": None, "wear_fraction": None, "wear_cost_inr": None, "ra_um": None,
           "ra_model": None, "notes": []}
    if process == "unknown":
        row["notes"].append("operation not known to the Time agent: nothing computed")
        return row
    t = agents._op_time(op, material, time_dims)
    row["t_cut_min"] = t
    if process in ("none", "forming"):
        row["notes"].append("no chip removal: no cutting energy, tool wear or Ra modelled" if process == "forming"
                            else "inspection: no cutting")
        return row

    vc0, feed = _speed_feed(process, op, material)
    vc = vc0 * vc_factor
    t = t / vc_factor                                          # same feed and depth at a higher speed: proportionally shorter cut
    row["t_cut_min"] = t
    dp = d["depth"]
    passes = agents._num_passes(op, dp)
    ap = dp / passes
    ae = None
    q = None                                                   # mm^3/min
    if process == "turning":
        h, q = feed, vc * 1000 * feed * ap
        r = NOSE_RADIUS_MM["turning"]
        ra = 1000 * feed ** 2 / (32 * r)
        floor, model = RA_FLOOR_UM["turning"], f"f^2/(32 r), r={r} mm"
    elif process == "chamfer":
        ap = CHAMFER_DEPTH_MM
        h, q = feed, vc * 1000 * feed * ap
        ra = floor = model = None
    elif process == "plunge":
        w = PLUNGE_TOOL_WIDTH_MM[op]
        ap, h, q = w, feed, vc * 1000 * feed * w
        ra = floor = model = None
    elif process in ("drill", "center_drill"):
        dia = 3.0 if process == "center_drill" else hole_d
        ap = None
        h = feed / 2 * math.sin(math.radians(DRILL_HALF_POINT_ANGLE_DEG))
        q = 250.0 * dia * feed * vc
        ra = floor = None
        if process == "drill":
            ra, floor, model = RA_FLOOR_UM["drill"], RA_FLOOR_UM["drill"], "process floor (drilling)"
        else:
            model = None
    elif process == "bore":
        ap = BORING_STOCK_MM
        h, q = feed, vc * 1000 * feed * ap
        r = NOSE_RADIUS_MM["boring"]
        ra = 1000 * feed ** 2 / (32 * r)
        floor, model = RA_FLOOR_UM["boring"], f"f^2/(32 r), r={r} mm"
    elif process == "ream":
        ap = REAMING_STOCK_MM
        h, q = feed / REAMER_TEETH, vc * 1000 * feed * ap
        ra, floor, model = RA_FLOOR_UM["ream"], RA_FLOOR_UM["ream"], "process floor (reaming)"
    elif process == "thread":
        pitch = agents._THREAD_PITCH[op]
        volume = 0.325 * math.pi * d["diameter"] * d["length"] * pitch          # mm^3 of thread groove: 0.325 p^2 * (L/p) * pi D
        h, q = THREAD_CHIP_MM, volume / t
        ap = None
        ra = floor = model = None
    else:                                                      # mill
        dc, z = agents._CUTTER_DIA[op], agents._CUTTER_TEETH[op]
        frac = MILL_ENGAGEMENT["slot"] if op in SLOT_LIKE else MILL_ENGAGEMENT["face"] if op in FACE_LIKE else MILL_ENGAGEMENT["other"]
        ae = frac * dc
        n = vc * 1000 / (math.pi * dc)
        vf = feed * z * n
        h, q = feed * math.sqrt(frac), ae * ap * vf
        r = NOSE_RADIUS_MM["face_mill"] if op in FACE_LIKE else dc / 2
        ra = 1000 * feed ** 2 / (32 * r)
        floor, model = RA_FLOOR_UM["mill"], f"fz^2/(32 r), r={r:g} mm"

    kc = _kc(material, h)
    power = kc * q / 6e7
    spindle = power / POWER_EFFICIENCY
    util = spindle / machine_power_kw
    n_exp = MAT.props(material)["taylor_n"]
    life = MAT.props(material)["t_ref_min"] * vc_factor ** (-1.0 / n_exp)
    wear = t / life
    row.update({"vc_m_min": vc, "feed": feed, "ap_mm": ap, "ae_mm": ae, "chip_mm": h, "kc_n_mm2": kc,
                "mrr_cm3_min": q / 1000.0, "power_kw": power, "spindle_kw": spindle, "utilisation": util,
                "energy_kwh": power * t / 60.0, "tool_life_min": life, "wear_fraction": wear,
                "wear_cost_inr": wear * MAT.props(material)["tool_edge_cost_inr"]})
    if ra is not None:
        row["ra_um"] = max(ra, floor)
        row["ra_model"] = model + (f" (floored at {floor} um)" if ra < floor else "")
    if util > 1.0:
        row["notes"].append(f"needs {spindle:.1f} kW at the spindle, machine has {machine_power_kw:g} kW")
    elif util > POWER_WARN:
        row["notes"].append(f"uses {util:.0%} of the machine power")
    return row


def ra_risk(best_ra, required_ra):
    """(ratio, label). label: low, medium, high, or unknown when either number is missing."""
    if best_ra is None or not required_ra:
        return None, "unknown"
    ratio = best_ra / required_ra
    return ratio, ("low" if ratio <= RA_MEDIUM_RATIO else "medium" if ratio <= RA_HIGH_RATIO else "high")


def max_feed_for_ra(required_ra_um: float, nose_radius_mm: float = NOSE_RADIUS_MM["turning"]) -> float:
    """Largest turning feed (mm/rev) whose theoretical Ra stays at or below the required Ra, with the safety ratio of RA_MEDIUM_RATIO."""
    return math.sqrt(32 * nose_radius_mm * RA_MEDIUM_RATIO * required_ra_um / 1000.0)


def analyse_route(steps: list, material: str, dims: dict = None, batch_size: int = 1, required_ra_um: float = None,
                  vc_factor: float = 1.0, machine_power_kw: float = SPINDLE_POWER_KW) -> dict:
    """Per-operation rows and route totals. Changeover markers are ignored. JSON-serialisable."""
    ops = [s for s in steps if s != MARKER]
    rows = [analyse_op(op, material, dims, vc_factor, machine_power_kw) for op in ops]
    cut = [r for r in rows if r["energy_kwh"] is not None]
    volume = sum(r["mrr_cm3_min"] * 1000 * r["t_cut_min"] for r in cut)                   # mm^3
    energy = sum(r["energy_kwh"] for r in cut)
    kc_mean = sum(r["kc_n_mm2"] * r["mrr_cm3_min"] * 1000 * r["t_cut_min"] for r in cut) / volume if volume > 0 else None
    wear_edges = sum(r["wear_fraction"] for r in cut)
    ra_rows = [r for r in cut if r["ra_um"] is not None]
    best = min(ra_rows, key=lambda r: r["ra_um"]) if ra_rows else None
    ratio, label = ra_risk(best["ra_um"] if best else None, required_ra_um)
    over = [r["op"] for r in cut if r["utilisation"] > 1.0]
    near = [r["op"] for r in cut if POWER_WARN < r["utilisation"] <= 1.0]
    flags = []
    if over:
        flags.append("power: " + ", ".join(sorted(set(over))) + f" need more than the {machine_power_kw:g} kW machine")
    if near:
        flags.append("power: " + ", ".join(sorted(set(near))) + " use more than 80% of the machine power")
    if label == "high":
        flags.append(f"Ra: the finest finish this route can produce is {best['ra_um']:.2g} um ({best['op']}), "
                     f"{required_ra_um:g} um is required")
    elif label == "medium":
        flags.append(f"Ra: predicted {best['ra_um']:.2g} um against {required_ra_um:g} um required, little margin")
    if any(r["process"] == "unknown" for r in rows):
        flags.append("unknown operation(s): " + ", ".join(r["op"] for r in rows if r["process"] == "unknown"))
    totals = {
        "operations": len(ops), "modelled_operations": len(cut), "cutting_time_min": sum(r["t_cut_min"] for r in cut),
        "cutting_energy_kwh": energy, "kc_mean_n_mm2": kc_mean, "specific_energy_j_mm3": None if kc_mean is None else kc_mean * 1e-3,
        "removed_volume_cm3": volume / 1000.0, "peak_power_kw": max((r["power_kw"] for r in cut), default=0.0),
        "peak_spindle_kw": max((r["spindle_kw"] for r in cut), default=0.0),
        "max_utilisation": max((r["utilisation"] for r in cut), default=0.0), "ops_over_power": sorted(set(over)),
        "ops_near_power": sorted(set(near)), "machine_power_kw": machine_power_kw,
        "wear_cost_inr": sum(r["wear_cost_inr"] for r in cut), "edges_per_part": wear_edges,
        "edges_per_batch": wear_edges * batch_size, "wear_cost_batch_inr": sum(r["wear_cost_inr"] for r in cut) * batch_size,
        "vc_factor": vc_factor,
        "ra": {"required_um": required_ra_um, "best_um": None if best is None else best["ra_um"],
               "best_op": None if best is None else best["op"], "ratio": ratio, "risk": label,
               "max_turning_feed_mm_rev": None if not required_ra_um else max_feed_for_ra(required_ra_um)},
        "flags": flags,
    }
    return {"material": material, "dims": _dims(dims), "batch_size": batch_size, "ops": rows, "totals": totals}


def analyse_part_route(steps: list, part: dict, **kw) -> dict:
    """analyse_route for a dataset part (dims, material, batch size and required Ra come from the part)."""
    return analyse_route(steps, part["material"], part.get("dims"), part.get("batch_size", 1), part.get("required_ra_um"), **kw)


def condense(analysis: dict) -> dict:
    """The totals plus a short row per operation: what the harness stores with a record."""
    keep = ("op", "process", "t_cut_min", "kc_n_mm2", "power_kw", "energy_kwh", "wear_cost_inr", "ra_um")
    return {"totals": analysis["totals"], "ops": [{k: r[k] for k in keep} for r in analysis["ops"]]}
