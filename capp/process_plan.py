"""From a route to a process plan (report 6.7, WP9): per operation its number, machine, tool, cutting parameters, estimated time and
the feature and rule that justify it. capp/routesheet.py turns the result into a document.

    from capp import analysis_view as AV, process_plan as PP
    part = AV.make_part("Steel", ["Hole", "Chamfer"], "0.05mm", 100, dims={"diameter": 40, "length": 120, "depth": 15, "width": 32},
                        required_ra_um=1.6, hole_diameter=8)
    plan = PP.build_plan(steps, part, context={"provenance": "route_builder"})
    plan["operations"][0]   # {"seq": 10, "op": "Facing", "machine": "lathe", "tool": {...}, "params": {...}, "time_min": ..., ...}

The route is chosen first and the parameters second: nothing here changes the route (report 6.7). The cutting speeds and feeds are the
v1 model's per-material values (week3/agents.py, the numbers the Time agent already uses, via capp/physics.py), so the estimated times
add up to the v1 Time agent's total of the route (plan["totals"]["v1_time_min"], equal when the part has no hole diameter). The tool
rules and the reference ranges are in capp/process_tables.py with their sources and status: every number the plan prints can be traced
there. Parameters are compared with the ranges ("below" is conservative, "above" asks for a check) and with the machine's power
(capp/physics.py: utilisation of an ASSUMED machine, capp/feasibility.DEFAULT_MACHINE).

What is assumed, not known (the sheet says so): the stock is a plain bar of the part's diameter, hole operations use the part diameter when the
part has no hole diameter, the machine power, the tool sizes of milling (agents._CUTTER_DIA), no spindle-speed limit, the thread pitches of
week3/agents.py. Without dimensions (Route Only) the reference part of v1 is used (Ø25, L50, depth 20) and the plan says so.
"""
import math

from capp import legacy  # noqa: F401
from capp import feasibility as F
from capp import physics
from capp import process_tables as PT
from capp import verify as V

import agents
import precedence_graph as pg
import feature_vocab as fv
import route_builder as rb

MARKER = V.MARKER
REFERENCE_DIMS = {"diameter": 25.0, "length": 50.0, "depth": 20.0, "width": 20.0}
MAX_RULES_SHOWN = 3
THREAD_FEED_OPS = ("Tapping", "External Threading")       # fed by the pitch: the feed is not compared with a range


def machine_class(op: str) -> str:
    return "lathe" if op in rb.LATHE_ONLY_OPS else "milling" if op in rb.MILLING_ONLY_OPS else "either"


def assign_machines(route: list, preference: str = "auto") -> list:
    """The machine of every operation of a route with changeover markers: [(machine, assumed)], in route order, markers skipped.
    An operation that needs a machine (lathe-only, milling-only) has it; a shared operation takes the machine of its segment (between two
    markers); a segment with no operation that needs a machine takes the part's preference, else the lathe (assumed: nothing in the route
    decides it). A segment that holds both kinds (a route without usable markers) gives its shared operations the lathe, assumed."""
    segments, cur = [], []
    for s in route:
        if s == MARKER:
            segments.append(cur)
            cur = []
        else:
            cur.append(s)
    segments.append(cur)
    out = []
    for seg in segments:
        classes = {machine_class(op) for op in seg} - {"either"}
        if len(classes) == 1:
            m, seg_assumed = next(iter(classes)), False
        elif not classes:
            chosen = preference in ("lathe", "milling")
            m, seg_assumed = (preference if chosen else "lathe"), not chosen
        else:
            m, seg_assumed = None, True
        for op in seg:
            c = machine_class(op)
            out.append((c, False) if c != "either" else (m or "lathe", seg_assumed))
    return out


def _overheads(route: list) -> dict:
    """Index -> {tool_change_min, position_change_min, changeover_min}: v1's setup accounting (agents._count_setup_events), per position.
    A marker's own cost (changeover plus one tool change, as v1 counts it) is carried to the next operation."""
    out = {i: {"tool_change_min": 0.0, "position_change_min": 0.0, "changeover_min": 0.0} for i in range(len(route))}
    carry = 0
    for i in range(1, len(route)):
        step, prev = route[i], route[i - 1]
        tool = not (step == "Inspection" or prev == MARKER or agents._same_tool(prev, step))
        pos = not (step in ("Inspection", MARKER) or prev == MARKER)
        if step == MARKER:
            out[i]["changeover_min"] = float(agents.CHANGEOVER_TIME_MIN)
            out[i]["tool_change_min"] = float(agents.TOOL_CHANGE_TIME_MIN) if tool else 0.0
            continue
        out[i]["tool_change_min"] = float(agents.TOOL_CHANGE_TIME_MIN) if tool else 0.0
        out[i]["position_change_min"] = float(agents.POSITION_CHANGE_TIME_MIN) if pos else 0.0
    return out


def _tool_size(op, dims, hole_given):
    """(size text, assumed) of the tool where the part or the model fixes it."""
    d = dims
    hole = d.get("hole_diameter") or d["diameter"]
    assumed = not d.get("hole_diameter")
    if op in ("Drilling", "Boring", "Reaming", "Counterboring", "Countersinking"):
        return f"Ø{hole:g} mm", assumed
    if op == "Tapping":
        return f"M{hole:g} x {agents._THREAD_PITCH[op]:g}", assumed
    if op == "Center Drilling":
        return "Ø3 mm", False
    if op == "External Threading":
        return f"pitch {agents._THREAD_PITCH[op]:g} mm", False
    if op in agents._CUTTER_DIA:
        z = agents._CUTTER_TEETH.get(op)
        extra = f", pitch {agents._THREAD_PITCH[op]:g} mm" if op in agents._THREAD_PITCH else ""
        return f"Ø{agents._CUTTER_DIA[op]:g} mm" + (f", {z} teeth" if z else "") + extra, False
    return "", False


def _spindle(op, row, dims):
    """(rpm, feed, feed rate in mm/min, feed unit, diameter used) from a physics row; None where there is no cut. Threading operations feed
    by their pitch (the model's drill feed is not a thread lead), so their feed is the pitch."""
    vc, f = row["vc_m_min"], row["feed"]
    if vc is None:
        return None, None, None, None, None
    if row["process"] == "mill" or op == "Thread Milling":
        dc = agents._CUTTER_DIA[op]
        n = 1000 * vc / (math.pi * dc)
        return n, f, f * agents._CUTTER_TEETH[op] * n, "mm/tooth", dc
    if row["process"] == "center_drill":
        dia = 3.0
    elif row["process"] in ("drill", "bore", "ream") or op == "Tapping":
        dia = dims.get("hole_diameter") or dims["diameter"]
    else:
        dia = dims["diameter"]
    n = 1000 * vc / (math.pi * dia)
    if op in THREAD_FEED_OPS:
        pitch = agents._THREAD_PITCH[op]
        return n, pitch, pitch * n, "mm/rev (pitch)", dia
    return n, f, f * n, "mm/rev", dia


def _chains(route_ops, features):
    """{op: [feature]}: the features each operation serves, from the first alternative chain of each feature that the route contains in full."""
    have = set(route_ops)
    served = {}
    for feat in features:
        if not fv.is_valid_feature(feat):
            continue
        for chain in fv.get_operations_for_feature(feat):
            if set(chain) <= have:
                for op in chain:
                    served.setdefault(op, [])
                    if feat not in served[op]:
                        served[op].append(feat)
                break
    return served


def _justification(op, idx, ops, served, n_ops):
    """The reason text and the rule ids of an operation: the features it serves, the order rules it follows."""
    reasons, rules = [], []
    if op in served:
        reasons.append("serves " + ", ".join(served[op]))
    if op == pg.ALWAYS_FIRST and idx == 0:
        reasons.append("every route starts with Facing")
        rules.append("ORD-FIRST")
    if op == pg.ALWAYS_LAST and idx == n_ops - 1:
        reasons.append("every route ends with Inspection")
        rules.append("ORD-LAST")
    before = [a for a in pg.get_prerequisites(op) if a in ops[:idx]]
    for a in sorted(before)[:MAX_RULES_SHOWN]:
        rules.append(f"ORD-EDGE:{a}>{op}")
    if before:
        reasons.append("after " + ", ".join(sorted(before)[:MAX_RULES_SHOWN]) + (" ..." if len(before) > MAX_RULES_SHOWN else ""))
    if not reasons:
        reasons.append("no requested feature needs it (support operation)")
    return "; ".join(reasons), rules


def repair_view(raw: list, final: list) -> dict:
    """Before and after a repair or replacement: what the final route lacks of the raw one, what it adds, and the status of each final operation."""
    from collections import Counter
    r, f = V.clean_steps(raw), V.clean_steps(final)
    left = Counter(r)
    marks = []
    for op in f:
        if left[op] > 0:
            left[op] -= 1
            marks.append("kept")
        else:
            marks.append("inserted")
    gone = []
    count = Counter(f)
    for op in r:
        if count[op] > 0:
            count[op] -= 1
        else:
            gone.append(op)
    kept_in_order = [op for op, m in zip(f, marks) if m == "kept"]
    raw_kept = list(r)
    for op in gone:
        raw_kept.remove(op)
    return {"identical": r == f, "raw": r, "final": f, "marks": marks, "removed": gone, "inserted": [op for op, m in zip(f, marks) if m == "inserted"],
            "reordered": kept_in_order != raw_kept}


def build_plan(steps: list, part: dict, machine: F.Machine = F.DEFAULT_MACHINE, context: dict = None, normalize: bool = True) -> dict:
    """The process plan of one route. `part` as capp.analysis_view.make_part (material, features, tolerance, batch_size, dims or None,
    required_ra_um); `context` is what the pipeline knows about the route and is carried to the sheet unchanged
    (provenance, system, confidence, flagged, raw_steps, reasoning, ...). normalize=True puts one changeover marker at every
    lathe <-> milling switch first (capp.verify.normalize_markers), so the plan is charged for every machine change whatever made the route."""
    material = part["material"]
    if material not in PT.MATERIALS:
        raise KeyError(f"unknown material {material!r}")
    route = V.normalize_markers(steps) if normalize else list(steps)
    ops = [s for s in route if s != MARKER]
    reference = part.get("dims") is None
    dims = dict(REFERENCE_DIMS if reference else physics._dims(part["dims"]))
    hole_given = bool(dims.get("hole_diameter"))
    analysis = physics.analyse_route(ops, material, None if reference else part["dims"], part.get("batch_size", 1),
                                     part.get("required_ra_um"), machine_power_kw=machine.spindle_power_kw)
    rows = iter(analysis["ops"])
    machines = iter(assign_machines(route, part.get("machine_preference", "auto")))
    over = _overheads(route)
    served = _chains(ops, part.get("features", []))
    try:
        report = F.evaluate(ops, part, machine)
        found = report.violations
    except Exception as exc:                                      # a rule bug must not stop the sheet
        report, found = None, []
        fz_error = f"feasibility check failed: {type(exc).__name__}: {exc}"
    else:
        fz_error = None
    check = V.check_route(route, list(part.get("features", [])), None)

    entries, ranges_used, seq, idx = [], [], 0, -1
    pending_marker = None
    totals = {"cutting_min": 0.0, "tool_change_min": 0.0, "position_change_min": 0.0, "changeover_min": 0.0}
    n_ops = len(ops)
    for i, step in enumerate(route):
        if step == MARKER:
            pending_marker = {"tool_change_min": over[i]["tool_change_min"], "changeover_min": over[i]["changeover_min"]}
            continue
        idx += 1
        seq += 10
        prow = next(rows)
        mach, mach_assumed = next(machines)
        known = step in PT.OPERATIONS
        entry = {"seq": seq, "op": step, "known": known, "machine": mach, "machine_assumed": mach_assumed, "flags": []}
        if not known:
            entry.update({"description": "(not an operation of the vocabulary)", "tool": None, "params": None, "range_check": None})
            entry["flags"].append("operation outside the vocabulary: no tool, no parameters")
        else:
            info = PT.operation(step)
            tool = PT.tool_for(step, material, mach)
            size, size_assumed = _tool_size(step, dims, hole_given)
            entry.update({"description": info["description"], "tool": dict(tool, size=size, size_assumed=size_assumed)})
            if tool["machine_override"]:
                entry["flags"].append(f"on the {mach} machine this is done with a {tool['type']}; speed and feed shown are the v1 model's "
                                      "(turning-type) values: set the milling feed per tooth at the machine")
            n, feed_used, vf, unit, dia = _spindle(step, prow, dims)
            if prow["vc_m_min"] is None:
                entry["params"] = None
                entry["range_check"] = None
                if prow["notes"]:
                    entry["flags"].extend(prow["notes"])
            else:
                passes = agents._num_passes(step, dims["depth"])
                entry["params"] = {"vc_m_min": prow["vc_m_min"], "rpm": n, "feed": feed_used, "feed_unit": unit, "feed_rate_mm_min": vf,
                                   "ap_mm": prow["ap_mm"], "ae_mm": prow["ae_mm"], "passes": passes, "diameter_mm": dia,
                                   "mrr_cm3_min": prow["mrr_cm3_min"], "power_kw": prow["power_kw"], "spindle_kw": prow["spindle_kw"],
                                   "load": prow["utilisation"]}
                rc = PT.check_range(step, material, prow["vc_m_min"], None if step in THREAD_FEED_OPS else feed_used)
                entry["range_check"] = rc
                if rc["range"] is not None:
                    ranges_used.append(rc["range"])
                    for key, label in (("vc", "cutting speed"), ("feed", "feed")):
                        if rc[key] == "above":
                            entry["flags"].append(f"{label} above the reference range ({rc['range'][key][0]:g}-{rc['range'][key][1]:g}): check")
                        elif rc[key] == "below":
                            entry["flags"].append(f"{label} below the reference range: conservative (longer cut)")
                entry["flags"].extend(prow["notes"])
        t_cut = prow["t_cut_min"] or 0.0
        oh = over[i]
        marker_tc = pending_marker["tool_change_min"] if pending_marker else 0.0
        marker_co = pending_marker["changeover_min"] if pending_marker else 0.0
        entry["changeover_before"] = bool(pending_marker)
        entry["time"] = {"cutting_min": t_cut, "tool_change_min": oh["tool_change_min"] + marker_tc,
                         "position_change_min": oh["position_change_min"], "changeover_min": marker_co}
        entry["time_min"] = t_cut + entry["time"]["tool_change_min"] + oh["position_change_min"] + marker_co
        for k in totals:
            totals[k] += entry["time"][k]
        pending_marker = None
        why, rules = _justification(step, idx, ops, served, n_ops)
        entry["serves"] = served.get(step, [])
        entry["why"], entry["rules"] = why, rules
        for v in found:
            if step in v.ops:
                rid = v.rule_id
                entry["flags"].append(f"{'hard finding' if v.kind == 'infeasible' else 'risk'} {rid}: {v.message}")
        entries.append(entry)

    total = sum(e["time_min"] for e in entries)
    try:
        v1_time = agents.time_agent(route, material, None if reference else dims)
    except Exception:
        v1_time = None
    changeovers = []
    for k, e in enumerate(entries):
        if e["changeover_before"] and k:
            changeovers.append({"before_seq": e["seq"], "from": entries[k - 1]["machine"], "to": e["machine"]})
    plan = {
        "part": {"part_id": part.get("part_id", ""), "material": material, "features": list(part.get("features", [])),
                 "tolerance": part.get("tolerance", ""), "batch_size": part.get("batch_size", 1), "dims": None if reference else dims,
                 "required_ra_um": part.get("required_ra_um"), "hole_diameter_given": hole_given},
        "dims_source": "reference part of v1 (Ø25, L50, depth 20): no dimensions were given" if reference else "part dimensions",
        "route": route, "operations": entries, "changeovers": changeovers,
        "totals": {"cutting_min": totals["cutting_min"], "tool_change_min": totals["tool_change_min"],
                   "position_change_min": totals["position_change_min"], "changeover_min": totals["changeover_min"],
                   "total_min": total, "v1_time_min": v1_time, "n_operations": len(entries), "n_changeovers": len(changeovers),
                   "per_batch_hours": total * part.get("batch_size", 1) / 60.0},
        "physics": {k: analysis["totals"][k] for k in ("cutting_energy_kwh", "peak_spindle_kw", "max_utilisation", "machine_power_kw", "flags", "ra")},
        "verifier": {"valid": check.valid, "v1_valid": check.v1_valid, "kinds": list(check.kinds), "messages": [v.message for v in check.violations]},
        "feasibility": {"hard": [v.message for v in found if v.kind == "infeasible"], "risks": [v.message for v in found if v.kind == "risk"],
                        "findings": [{"rule_id": v.rule_id, "kind": v.kind, "message": v.message, "ops": list(v.ops)} for v in found],
                        "not_evaluated": sorted(report.not_evaluated.items()) if report else [], "error": fz_error},
        "machine": {"name": machine.name, "spindle_power_kw": machine.spindle_power_kw},
        "tables": {"ranges_used": len(ranges_used), "status": PT.status_counts(ranges_used), "tool_rules": "indicative",
                   "sources": sorted({s for r in ranges_used for s in r["sources"]})},
        "context": dict(context or {}),
        "assumptions": assumptions(dims, reference, hole_given, part, machine),
    }
    ctx = plan["context"]
    if ctx.get("raw_steps"):
        plan["repair"] = repair_view(ctx["raw_steps"], route)
    return plan


def assumptions(dims, reference, hole_given, part, machine) -> list:
    """The lines that must stay next to the numbers of a sheet."""
    L = ["Cutting speeds and feeds are the v1 model's per-material values (the ones its Time agent uses), not values chosen for this part's tool, "
         "grade or machine. They are compared with ranges from web pages (capp/process_tables.py), not with a handbook: no handbook page was checked.",
         "Tool types and tool materials come from a rule table of general machining practice (status: indicative). No machining trial was made.",
         f"Spindle load is that of an ASSUMED machine ({machine.spindle_power_kw:g} kW at the spindle, drive efficiency {physics.POWER_EFFICIENCY:g}); "
         "spindle-speed and torque limits are not modelled.",
         "Estimated time = cutting time of the v1 model + v1's set-up penalties (tool change, position change, machine changeover); it is not a "
         "shop-floor standard time."]
    if reference:
        L.append("No part dimensions were given: the reference part of v1 (Ø25 mm, L 50 mm, depth 20 mm) is used for every parameter and time.")
    elif not hole_given:
        L.append("No hole diameter was given: hole operations (drilling, boring, reaming, tapping, counterboring, countersinking) use the part "
                 "diameter, marked * in the tool column.")
    L.append("The stock is taken as a bar of the part's diameter; milling cutters are the v1 model's standard diameters and tooth numbers.")
    return L


def context_from_record(rec: dict, flag_threshold: float = None) -> dict:
    """What a run record knows about its final route, for the sheet: system, provenance, plan confidence and review flag (against
    `flag_threshold`, default the threshold the run used), selection, number of samples, the raw route of the chosen sample (so the sheet can show
    what a repair or replacement changed) and the planner's stated reasoning (two-step planners). Old records without a field give None for it."""
    fin, info = rec["final"], rec.get("info", {})
    cand = next((c for c in rec.get("candidates", []) if c.get("name") == fin.get("name")), None)
    thr = flag_threshold if flag_threshold is not None else rec.get("flag_threshold")
    conf = info.get("confidence")
    flagged = info.get("flagged") if thr is None else (None if conf is None else conf < thr)
    ctx = {"system": rec.get("system"), "provenance": fin.get("provenance"), "source": fin.get("provenance"), "confidence": conf,
           "flagged": flagged, "confidence_flag": thr, "selection": info.get("selection"),
           "n_samples": sum(1 for c in rec.get("candidates", []) if c.get("sample_id", -1) >= 0) or None,
           "repeat": rec.get("repeat")}
    if cand is not None:
        if cand.get("llm_ok") and cand.get("llm_steps"):
            ctx["raw_steps"] = list(cand["llm_steps"])
        reasoning = (cand.get("meta") or {}).get("reasoning")
        if reasoning:
            ctx["reasoning"] = reasoning
    return {k: v for k, v in ctx.items() if v is not None}


def plan_from_record(rec: dict, flag_threshold: float = None, machine: F.Machine = F.DEFAULT_MACHINE) -> dict:
    """The process plan of the final route of a run record (capp.harness.load_records). The part is rec["part"]; a record of an old run
    that lacks dimensions is planned on the reference part of v1 and the sheet says so."""
    part = dict(rec["part"])
    part.setdefault("part_id", rec.get("part_id", ""))
    return build_plan(rec["final"]["steps"], part, machine=machine, context=context_from_record(rec, flag_threshold))
