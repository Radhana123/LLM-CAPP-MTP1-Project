"""What the UI shows for a route: the numbers of "Route + Full Analysis" and the checks of both modes, as plain data.

    from capp import analysis_view as AV
    part = AV.make_part("Steel", ["Hole", "Turning"], "0.02mm", 100, dims={"diameter": 40, "length": 120, "depth": 15, "width": 32},
                        required_ra_um=1.6, hole_diameter=8)
    view = AV.analyse_route_view(steps, part)          # Route + Full Analysis
    view = AV.analyse_route_view(steps, AV.make_part(..., dims=None))     # Route Only: the dimension-free checks only

week6/app.py draws these dictionaries (Streamlit); nothing here imports Streamlit, so the numbers are testable without a browser.
This is analysis only: the route shown is the one the v1 pipeline selected (efficiency), the numbers describe it and flag problems, they
never change which route is chosen.

view = {
  "mode":        "full" | "route_only"
  "physics":     None (route only) or {"cards": {...}, "ops": [row per operation], "sensitivity": [...], "flags": [...],
                 "other_flags": the flags the feasibility rules do not report again, "ra": {...}}
  "feasibility": {"infeasible": [...], "risks": [...], "part_level": [...], "not_evaluated": [(rule, reason)], "notes": [...], "evaluated": n}
  "assumptions": [text lines that must stay next to the numbers]
}
"""
from capp import legacy  # noqa: F401
from capp import feasibility as F
from capp import physics

RA_CHOICES = (None, 0.4, 0.8, 1.6, 3.2, 6.3, 12.5)          # um; the values of the dataset (None = not specified)
SPEED_FACTORS = (0.8, 1.0, 1.2)
FLAGS_COVERED_BY_RULES = ("power:", "Ra:")      # physics flags that CAP-POWER and RA-FEED/RA-FLOOR report again as feasibility findings
RISK_ICON = {"low": "🟢", "medium": "🟡", "high": "🔴", "unknown": "⚪"}
RISK_SHORT = {"low": "low", "medium": "medium", "high": "high", "unknown": "n/a"}      # for narrow table cells and metric cards
RISK_TEXT = {"low": "low", "medium": "medium (little margin)", "high": "high (the standard feed cannot give it)", "unknown": "not assessed (no required Ra)"}


def make_part(material, features, tolerance, batch_size, dims=None, required_ra_um=None, hole_diameter=None) -> dict:
    """A part record as the rules and the physics model read it. dims=None = Route Only: no dimensions, so the rules that need
    dimensions are reported as not evaluable. hole_diameter (mm, optional) enables HOLE-DEEP and the hole operations of CAP-POWER."""
    d = None
    if dims is not None:
        d = {k: float(v) for k, v in dims.items() if v is not None}
        if hole_diameter:
            d["hole_diameter"] = float(hole_diameter)
    return {"part_id": "UI", "material": material, "features": list(features), "tolerance": tolerance, "batch_size": int(batch_size),
            "dims": d, "required_ra_um": required_ra_um or None}


def _clean(steps):
    return [s for s in steps if s != F.V.MARKER]


def _viol(v):
    r = F.RULE_BY_ID[v.rule_id[5:]]
    return {"rule_id": r.id, "kind": v.kind, "scope": r.scope, "category": r.category, "message": v.message, "action": v.action,
            "ops": list(v.ops), "source_status": r.source_status}


def feasibility_view(steps, part, machine=F.DEFAULT_MACHINE) -> dict:
    rep = F.evaluate(_clean(steps), part, machine)
    vs = [_viol(v) for v in rep.violations]
    return {"infeasible": [v for v in vs if v["kind"] == "infeasible"],
            "risks": [v for v in vs if v["kind"] == "risk" and v["scope"] == "route"],
            "part_level": [v for v in vs if v["scope"] == "part"],
            "not_evaluated": sorted(rep.not_evaluated.items()), "notes": [list(n) for n in rep.notes], "evaluated": len(rep.evaluated),
            "n_rules": len(F.RULES)}


def _card_values(t):
    ra = t["ra"]
    return {"energy_kwh": t["cutting_energy_kwh"], "kc_mean_n_mm2": t["kc_mean_n_mm2"], "specific_energy_j_mm3": t["specific_energy_j_mm3"],
            "wear_cost_inr": t["wear_cost_inr"], "edges_per_part": t["edges_per_part"], "edges_per_batch": t["edges_per_batch"],
            "wear_cost_batch_inr": t["wear_cost_batch_inr"], "peak_spindle_kw": t["peak_spindle_kw"], "max_utilisation": t["max_utilisation"],
            "machine_power_kw": t["machine_power_kw"], "removed_volume_cm3": t["removed_volume_cm3"], "cutting_time_min": t["cutting_time_min"],
            "ra_risk": ra["risk"], "ra_best_um": ra["best_um"], "ra_best_op": ra["best_op"], "ra_required_um": ra["required_um"],
            "ra_ratio": ra["ratio"], "max_turning_feed_mm_rev": ra["max_turning_feed_mm_rev"]}


def _op_rows(analysis):
    rows = []
    for r in analysis["ops"]:
        rows.append({"Operation": r["op"], "Process": r["process"],
                     "Cutting time (min)": None if r["process"] == "none" else r["t_cut_min"],       # inspection is time but not cutting
                     "kc (N/mm²)": r["kc_n_mm2"], "MRR (cm³/min)": r["mrr_cm3_min"], "Power (kW)": r["power_kw"],
                     "Spindle load (%)": None if r["utilisation"] is None else r["utilisation"] * 100.0,
                     "Energy (kWh)": r["energy_kwh"], "Tool wear (₹)": r["wear_cost_inr"], "Ra (µm)": r["ra_um"],
                     "Note": "; ".join(r["notes"])})
    return rows


def physics_view(steps, part, machine=F.DEFAULT_MACHINE, speed_factors=SPEED_FACTORS) -> dict:
    steps = _clean(steps)
    kw = dict(batch_size=part["batch_size"], required_ra_um=part.get("required_ra_um"), machine_power_kw=machine.spindle_power_kw)
    a = physics.analyse_route(steps, part["material"], part["dims"], **kw)
    sens = []
    for k in speed_factors:
        t = physics.analyse_route(steps, part["material"], part["dims"], vc_factor=k, **kw)["totals"]
        sens.append({"vc_factor": k, "cutting_time_min": t["cutting_time_min"], "energy_kwh": t["cutting_energy_kwh"],
                     "wear_cost_inr": t["wear_cost_inr"], "peak_spindle_kw": t["peak_spindle_kw"], "ops_over_power": t["ops_over_power"]})
    flags = list(a["totals"]["flags"])
    return {"cards": _card_values(a["totals"]), "ops": _op_rows(a), "sensitivity": sens, "flags": flags,
            "other_flags": [f for f in flags if not f.startswith(FLAGS_COVERED_BY_RULES)], "ra": a["totals"]["ra"]}


def assumptions(part, machine=F.DEFAULT_MACHINE) -> list:
    """The lines that belong next to the numbers: what is assumed, what is not modelled."""
    lines = [
        "Cutting energy is the energy of the cut only (a lower bound): idle power, spindle acceleration, coolant and auxiliaries are not modelled. "
        "It is not the v1 'Energy' score, which counts a fixed amount per operation.",
        f"kc is the Kienzle specific cutting force at the chip thickness of each operation (table of capp/materials.py); machine power "
        f"{machine.spindle_power_kw:g} kW and drive efficiency {physics.POWER_EFFICIENCY:g} are assumed, not those of a real machine.",
        "Tool wear: Taylor's law with the tool life at the handbook speed ASSUMED per material (capp/materials.py); the ₹ per cutting edge is assumed too. "
        "Compare routes and speeds with it; do not quote the absolute ₹.",
        "Ra risk is optimistic: it assumes the finest process of the route makes the critical surface. 'High' is a firm warning, 'low' is not a guarantee.",
        f"The machine envelope ({machine.lathe_max_diameter_mm:g} mm swing, {machine.lathe_max_length_mm:g} mm length, {machine.mill_travel_mm} mm travel) is assumed.",
    ]
    if part["dims"] is not None and not part["dims"].get("hole_diameter"):
        lines.append("No hole diameter was given: hole operations use the part diameter in the time and energy figures (as the v1 Time agent does) "
                     "and are not checked against the machine power.")
    lines.append(f"The tolerance is read as ± the value ({F.TOLERANCE_BAND_FACTOR:g} × the value as a band).")
    return lines


def analyse_route_view(steps, part, machine=F.DEFAULT_MACHINE) -> dict:
    """Everything the UI needs for one route. Never raises on a bad route: a failing part is reported inside the view."""
    full = part["dims"] is not None
    view = {"mode": "full" if full else "route_only", "physics": None, "feasibility": None, "assumptions": assumptions(part, machine),
            "errors": []}
    try:
        view["feasibility"] = feasibility_view(steps, part, machine)
    except Exception as exc:                                       # a rule bug must not take the page down
        view["errors"].append(f"feasibility check failed: {type(exc).__name__}: {exc}")
    if full:
        try:
            view["physics"] = physics_view(steps, part, machine)
        except Exception as exc:
            view["errors"].append(f"physics analysis failed: {type(exc).__name__}: {exc}")
    return view


def route_row(steps, part, machine=F.DEFAULT_MACHINE) -> dict:
    """The extra columns of the route table of Full Analysis: one route, a few numbers."""
    try:
        t = physics.analyse_route(_clean(steps), part["material"], part["dims"], part["batch_size"], part.get("required_ra_um"),
                                  machine_power_kw=machine.spindle_power_kw)["totals"]
        rep = F.evaluate(_clean(steps), part, machine)
    except Exception:
        return {"Cutting energy (kWh)": None, "kc (N/mm²)": None, "Tool wear (₹)": None, "Ra risk": "n/a", "Hard issues": None, "Risks": None}
    return {"Cutting energy (kWh)": round(t["cutting_energy_kwh"], 3), "kc (N/mm²)": None if t["kc_mean_n_mm2"] is None else round(t["kc_mean_n_mm2"]),
            "Tool wear (₹)": round(t["wear_cost_inr"], 1), "Ra risk": f"{RISK_ICON[t['ra']['risk']]} {RISK_SHORT[t['ra']['risk']]}",
            "Hard issues": len(rep.infeasible), "Risks": len(rep.risks)}


def rules_table() -> list:
    """Every rule with its source and status, for the 'sources' expander."""
    return [{"Rule": r["id"], "Category": r["category"], "Kind": r["kind"], "Scope": r["scope"], "Needs": ", ".join(r["needs"]) or "—",
             "What it checks": r["summary"], "Action": r["action"], "Source": r["source"], "Status": r["source_status"]}
            for r in F.rule_table()]
