"""The time and cost model of the Time-and-Cost-calculator (Radhana123/Time-and-Cost-calculator, streamlit_app.py), for LLM-CAPP routes (Step 52).

    from capp import calculator as CALC
    s = CALC.Settings(labor_per_hour=500, material_cost=1000)
    r = CALC.evaluate(route_steps, "Steel", dims, s)     # rows per operation + the summary of the calculator's Result page
    r["time_min"], r["total_cost"]

What is taken over as it is written in the calculator
  * the time formula of each of its 13 machining processes (Boring, Drilling Center / Pilot / Main, Facing, Grooving, Knurling, Reaming,
    Threading, Turning Concave / Convex / Straight / Taper): (approach + overrun + length) / (feed x rpm) x number of passes
    + job setting time + tool setting time. A formula that divides by 0 gives time 0, as in the calculator;
  * the six "other processes" (Custom, Blanking, Chamfering, Parting, Resharpening, Tool Change): their time + 5 min handling, their extra tool cost;
  * the cost: operation cost = time x labour cost per hour / 60; total = material cost + labour + tool cost + extra process cost;
  * the default value of every input (feed 0.16 mm/rev, rpm 170, approach 10 mm, tool setting 5 min, tool cost Rs 10, ...).

What the calculator does not have, and what this module does about it (said on the page and in BASELINE as well)
  * ENERGY: no formula. capp/calculator.py uses week3/agents.energy_agent (the v1 energy model) unchanged.
  * MILLING (Step 53): the calculator has none. Eleven milling operations get five milling processes (MILLING, MILLING_OF) built the same way
    from the formulas of the milling sheets: Vc = pi D n / 1000, n = 1000 Vc / (pi D), Vf = n fz Z, face milling Tm = (L + D) / Vf, slab milling
    Tm = (L + sqrt(d (D - d))) / Vf; time = (approach + overrun + length term) / Vf x passes + job setting + tool setting. Marked "milling formulas (sheets)".
  * every other operation without a process (13 of the 41: Form / Helical / Thread / Gear milling, Engraving, Corner Rounding, Counterboring, Countersinking,
    Spotfacing, Forming, Polishing/Burnishing, Eccentric Turning, Inspection): the cutting time of the v1 model (week3/agents._op_time)
    + the calculator's default tool setting time (5 min) and tool cost (Rs 10); Inspection = the v1 fixed 5 min. Marked "v1 model" in every row.
  * the machine changeover between lathe and milling: the v1 time (15 min); its cost is the labour of those minutes. The v1 penalties for
    tool change and position change are NOT added: the calculator's tool setting time takes their place.
  * BATCH SIZE: the calculator prices one job; so does this module (no batch discount).

Which of the part's numbers go into which calculator input is auto_fields(): an input takes the number the drawing gives for the feature the
operation makes (dims["per_op"]) or the part's dimension where that is obviously the same thing (the part diameter is the initial diameter),
and keeps the calculator's default otherwise. Nothing is invented: the ratios used for the "final" diameters are those of the calculator's own defaults.
"""
import math

from capp import legacy  # noqa: F401
import agents

LABOR_PER_HOUR = 500.0           # the calculator's defaults for the two cost inputs
MATERIAL_COST = 1000.0
HANDLING_MIN = 5.0               # added to every "other process" by the calculator
FALLBACK_TOOL_SETTING_MIN = 5.0  # the calculator's default tool setting time, for an operation it has no process for
FALLBACK_TOOL_COST = 10.0        # ... and its default tool cost
MARKER = "--- Machine Changeover ---"
NOT_IN_CALCULATOR = "v1 model (no formula in the calculator)"

# the final / smaller / larger diameters of the calculator's own defaults, as ratios, so that they scale with the part
FINAL_TO_INITIAL_TURNING = 38.0 / 50.0     # Turning - Straight and Taper: 50 -> 38
BORE_FINAL_TO_INITIAL = 28.0 / 25.0        # Boring: 25 -> 28
REAM_INITIAL_TO_FINAL = 28.0 / 30.0        # Reaming: 28 -> 30
GROOVE_FINAL_MINUS_INITIAL = 0.5           # Grooving: 28 -> 28.5


def _f(key, label, default):
    return (key, label, float(default))


_COMMON = [_f("job set time", "Job Setting Time (min)", 0.0), _f("tool set time", "Tool Setting Time (min)", 5.0),
           _f("approach", "Approach (mm)", 10.0), _f("overrun", "Overrun (mm)", 0.0), _f("tool cost", "Tool Cost (Rs)", 10.0)]


def _drill(depth, turns, dia):
    return [_f("depth", "Drilling Depth (mm)", depth), _f("feed", "Feed (mm/rev)", 0.16), _f("rpm", "RPM", 170),
            _f("turn", "Num of times drilled", turns), _f("diameter", "Drill Diameter (mm)", dia)] + _COMMON


# process -> its inputs, in the order of the calculator, with its defaults
PROCESSES = {
    "Boring": [_f("initial diameter", "Initial Diameter (mm)", 25), _f("final diameter", "Final Diameter (mm)", 28), _f("depth", "Boring Length (mm)", 12),
               _f("feed", "Feed (mm/rev)", 0.16), _f("rpm", "RPM", 170), _f("angle", "Drill Angle (degree)", 118),
               _f("depth of cut", "Depth of cut (mm/pass)", 0.10)] + _COMMON,
    "Drilling - Center": _drill(5, 4, 6),
    "Drilling - Pilot": _drill(12, 5, 15),
    "Drilling - Main": _drill(12, 7, 25),
    "Facing": [_f("diameter", "Facing Diameter (mm)", 50), _f("length", "Facing Length (mm)", 10), _f("feed", "Feed (mm/rev)", 0.16), _f("rpm", "RPM", 170),
               _f("depth of cut", "Depth of cut (mm/pass)", 2.0)] + _COMMON[:3] + [_f("overrun", "Overrun (mm)", 10.0), _f("tool cost", "Tool Cost (Rs)", 10.0)],
    "Grooving": [_f("initial diameter", "Initial Diameter (mm)", 28), _f("final diameter", "Final Diameter (mm)", 28.5), _f("length", "Groove Length (mm)", 2),
                 _f("feed", "Feed (mm/rev)", 0.16), _f("rpm", "RPM", 170), _f("tool width", "Tool Width (mm)", 2),
                 _f("depth of cut", "Depth of Cut (mm/pass)", 0.10)] + _COMMON,
    "Knurling": [_f("length", "Knurling Length (mm)", 18), _f("feed", "Feed (mm/rev)", 0.20), _f("rpm", "RPM", 60),
                 _f("depth of cut", "Depth of cut (mm/pass)", 0.10)] + _COMMON,
    "Reaming": [_f("length", "Reaming Length (mm)", 20), _f("initial diameter", "Initial Diameter (mm)", 28), _f("final diameter", "Final Diameter (mm)", 30),
                _f("feed", "Feed (mm/rev)", 0.16), _f("rpm", "RPM", 170), _f("depth of cut", "Depth of cut (mm/pass)", 0.10)] + _COMMON,
    "Threading": [_f("pitch", "Thread Pitch (mm)", 2), _f("length", "Thread Length (mm)", 12), _f("feed", "Feed (mm/rev)", 0.20), _f("rpm", "RPM", 60),
                  _f("depth of cut", "Depth of cut (mm/pass)", 0.10)] + _COMMON,
    "Turning - Concave": [_f("diameter", "Diameter (mm)", 50), _f("angle", "Concave Angle (degree)", 20), _f("length", "Turning Length (mm)", 25),
                          _f("feed", "Feed (mm/rev)", 0.16), _f("rpm", "RPM", 170), _f("depth of cut", "Depth of cut (mm/pass)", 1.0)]
                         + _COMMON[:3] + [_f("overrun", "Overrun (mm)", 10.0), _f("tool cost", "Tool Cost (Rs)", 10.0)],
    "Turning - Convex": [_f("diameter", "Diameter (mm)", 50), _f("angle", "Convex Angle (degree)", 20), _f("length", "Turning Length (mm)", 25),
                         _f("feed", "Feed (mm/rev)", 0.16), _f("rpm", "RPM", 170), _f("depth of cut", "Depth of cut (mm/pass)", 1.0)]
                        + _COMMON[:3] + [_f("overrun", "Overrun (mm)", 10.0), _f("tool cost", "Tool Cost (Rs)", 10.0)],
    "Turning - Straight": [_f("initial diameter", "Initial Diameter (mm)", 50), _f("final diameter", "Final Diameter (mm)", 38),
                           _f("length", "Turning Length (mm)", 25), _f("feed", "Feed (mm/rev)", 0.16), _f("rpm", "RPM", 170),
                           _f("depth of cut", "Depth of cut (mm/pass)", 1.0)] + _COMMON,
    "Turning - Taper": [_f("larger diameter", "Larger Diameter (mm)", 50), _f("smaller diameter", "Smaller Diameter (mm)", 38),
                        _f("length", "Turning Length (mm)", 25), _f("feed", "Feed (mm/rev)", 0.16), _f("rpm", "RPM", 170),
                        _f("depth of cut", "Depth of cut (mm/pass)", 1.0)] + _COMMON,
}

# "other processes": name -> (time in min, extra tool cost in Rs), the calculator's defaults
EXTRA = {"Custom": (5.0, 100.0), "Blanking": (15.0, 50.0), "Chamfering": (5.0, 10.0), "Parting": (20.0, 50.0),
         "Resharpening": (35.0, 50.0), "Tool Change": (35.0, 50.0)}

# the operations of LLM-CAPP that have a process (or an "other process") in the calculator. The choice is a judgement, listed here so that it can be
# argued with; every other operation (all of milling, Counterboring, Countersinking, Spotfacing, Thread Milling, Forming, Polishing/Burnishing,
# Eccentric Turning, Inspection) has none and is timed by the v1 model.
PROCESS_OF = {
    "Facing": "Facing",
    "Plain/Cylindrical Turning": "Turning - Straight", "Step Turning": "Turning - Straight",
    "Taper Turning": "Turning - Taper",
    "Contour Turning": "Turning - Convex",                      # a curved outline: the calculator's arc formula is the closest
    "Grooving/Necking": "Grooving", "Internal Grooving": "Grooving", "Undercutting": "Grooving",
    "Knurling": "Knurling",
    "External Threading": "Threading", "Tapping": "Threading",  # a tap cuts a thread of a pitch, along a length
    "Boring": "Boring",
    "Center Drilling": "Drilling - Center", "Drilling": "Drilling - Main",
    "Reaming": "Reaming",
    "Drilling - Pilot": "Drilling - Pilot", "Turning - Concave": "Turning - Concave",      # (Step 54) processes of the calculator that no operation of the vocabulary maps to
}
EXTRA_OF = {"Chamfering": "Chamfering", "Parting-off": "Parting",
            "Custom": "Custom", "Blanking": "Blanking", "Resharpening": "Resharpening", "Tool Change": "Tool Change"}      # (Step 54) the other four 'other processes'
# the items of the calculator's list that are not operations of the vocabulary: they can be added to the dropdown of the page, never come from a route
CALCULATOR_ONLY = ("Drilling - Pilot", "Turning - Concave", "Custom", "Blanking", "Resharpening", "Tool Change")
SELECTION_KEY = "op_sel_calc"        # the dropdown of the calculator model (47 items); the v1 dropdown (41) keeps op_dims.SELECTION_KEY
SIGNATURE_KEY = "op_sel_calc_sig"
FIELD_KEYS = {p: [k for k, _, _ in fs] for p, fs in PROCESSES.items()}
DEFAULTS = {p: {k: d for k, _, d in fs} for p, fs in PROCESSES.items()}

# ---------------------------------------------------------------- milling (Step 53): the formulas of the sheets, in the style of the calculator
# The calculator has no milling. The sheets sent for this step give: Vc = pi D n / 1000, n = 1000 Vc / (pi D), Vf = n fz Z (table feed, mm/min),
# face milling Tm = (L + D) / Vf, slab milling Tm = (L + sqrt(d (D - d))) / Vf. The processes below use them the way the calculator uses its lathe
# formulas: (approach + overrun + length term) / feed rate x passes + job setting + tool setting. The default values of the cutter (diameter, teeth),
# of Vc and fz (for the material) and of the depth per pass are those the v1 model uses for the operation (week3/agents.py): marked "v1".
_MILL_COMMON = [("approach", "Approach (mm)", 0.0), ("overrun", "Overrun (mm)", 0.0), ("job set time", "Job Setting Time (min)", 0.0),
                ("tool set time", "Tool Setting Time (min)", 5.0), ("tool cost", "Tool Cost (Rs)", 10.0)]
_MILL_HEAD = [("cutter diameter", "Cutter Diameter (D, mm)", 0.0), ("teeth", "Number of Teeth (Z)", 0.0), ("cutting speed", "Cutting Speed (Vc, m/min)", 0.0),
              ("feed per tooth", "Feed per Tooth (fz, mm/tooth)", 0.0), ("length", "Cut Length (L, mm)", 0.0), ("total depth", "Total Depth (mm)", 0.0),
              ("depth of cut", "Depth of Cut per Pass (ap, mm)", 0.0)]
MILLING = {
    "Milling - Face": _MILL_HEAD + _MILL_COMMON,
    "Milling - Slab": _MILL_HEAD + _MILL_COMMON,
    "Milling - Slot": _MILL_HEAD + _MILL_COMMON,
    "Milling - Pocket": _MILL_HEAD + [("width", "Width (W, mm)", 0.0), ("stepover", "Stepover (ae, mm)", 0.0)] + _MILL_COMMON,
    "Milling - Profile": _MILL_HEAD + [("width", "Width (W, mm)", 0.0)] + _MILL_COMMON,
}
MILLING_OF = {
    "Face Milling": "Milling - Face",
    "Slab/Peripheral Milling": "Milling - Slab", "Gang Milling": "Milling - Slab",
    "Slot Milling": "Milling - Slot", "T-Slot Milling": "Milling - Slot", "Dovetail Milling": "Milling - Slot",
    "Woodruff Keyway Milling": "Milling - Slot", "Angular Milling": "Milling - Slot",
    "Pocket Milling": "Milling - Pocket",
    "Profile Milling": "Milling - Profile", "Surface Contouring": "Milling - Profile",
}
FIELD_KEYS_MILL = {p: [k for k, _, _ in fs] for p, fs in MILLING.items()}
MILLING_SOURCE = "milling formulas (sheets)"
STEPOVER_OF_CUTTER = 0.6        # v1: step-over = 0.6 x cutter diameter in pocket milling


def has_process(op: str) -> bool:
    return op in PROCESS_OF or op in EXTRA_OF or op in MILLING_OF


def process_of(op: str):
    """The calculator process (or milling process, or other process) of an operation; None if it has none."""
    return PROCESS_OF.get(op) or MILLING_OF.get(op) or EXTRA_OF.get(op)


def plan_job(steps: list, selected, auto) -> list:
    """The operations of the job in calculator mode (Step 54): the route, without the operations the user took out of the dropdown (`auto` = what the
    features selected by themselves, `selected` = what is selected now) and with the operations the user added and the route does not have, put before the
    last Inspection. Operations the route has and the features did not select (the planner's own choice) stay. With nothing removed or added the steps
    come back unchanged; otherwise the changeover markers are put again at every lathe <-> milling switch."""
    from capp import verify as V
    selected, auto = list(selected), list(auto)
    removed = set(auto) - set(selected)
    kept = [s for s in steps if s not in removed]
    present = set(kept)
    added = [o for o in selected if o not in set(auto) and o not in present]
    if not removed and not added:
        return list(steps)
    if added:
        kept = kept[:-1] + added + kept[-1:] if kept and kept[-1] == "Inspection" else kept + added
    return V.normalize_markers(kept)


def job_changes(steps: list, job: list):
    """(added, removed): the operations the job has and the route does not, and the other way round (changeover markers left out)."""
    route, now = [x for x in steps if x != MARKER], [x for x in job if x != MARKER]
    return [x for x in now if x not in route], [x for x in route if x not in now]


def fields_of(op: str):
    """[(key, label, default)] of the inputs of an operation: its process, or the two of an 'other process'. (The default of a milling input depends on the
    operation and the material: milling_defaults.)"""
    if op in PROCESS_OF:
        return list(PROCESSES[PROCESS_OF[op]])
    if op in MILLING_OF:
        return list(MILLING[MILLING_OF[op]])
    if op in EXTRA_OF:
        t, c = EXTRA[EXTRA_OF[op]]
        return [("time_min", "Time Taken (min)", t), ("extra_tool_cost", "Extra Tool Cost (Rs)", c)]
    return []


def milling_defaults(op: str, material: str = None) -> dict:
    """Starting values of the inputs of a milling operation: the v1 cutter of the operation, Vc and fz of the material, the v1 depth per pass,
    the calculator's defaults for the setting times and the tool cost, 0 for approach / overrun (the sheets' formulas have the cutter entry in them)."""
    prm = agents._CUTTING_PARAMS.get(material, agents._CUTTING_PARAMS["Steel"])
    d = float(agents._CUTTER_DIA[op])
    out = {"cutter diameter": d, "teeth": float(agents._CUTTER_TEETH[op]), "cutting speed": float(prm[4]), "feed per tooth": float(prm[5]),
           "length": REFERENCE_DIMS["length"], "total depth": REFERENCE_DIMS["depth"], "depth of cut": float(agents._MAX_DEPTH_PER_PASS.get(op, 5.0)),
           "width": REFERENCE_DIMS["width"], "stepover": STEPOVER_OF_CUTTER * d}
    out.update({k: dflt for k, _, dflt in _MILL_COMMON})
    return {k: out[k] for k, _, _ in fields_of(op)}


# ---------------------------------------------------------------- the formulas of the calculator
def _cut(process: str, p: dict) -> float:
    """The cutting time of one machining process, before the setting times: the formulas of streamlit_app.py ('Submit All'), as they are written.
    May raise ZeroDivisionError (the calculator then shows time 0). The milling processes (Step 53) use the formulas of the sheets (_mill_cut)."""
    if process in MILLING:
        return _mill_cut(process, p)
    ap, ov = p["approach"], p["overrun"]
    rpm, feed = p["rpm"], p["feed"]
    if process == "Boring":
        a = math.radians((180 - p["angle"]) / 2)
        extra_length = (p["initial diameter"] / 2) * math.tan(a)
        turn = (p["final diameter"] - p["initial diameter"]) / (p["depth of cut"] * 2)
        return (((ap + ov + p["depth"]) / (rpm * feed) * (turn * 2)) + (((10 + extra_length) / (feed * rpm)) * (turn * 2)))
    if process in ("Drilling - Center", "Drilling - Pilot", "Drilling - Main"):
        drill_length = ((p["diameter"] / 2) * math.tan(math.radians((180 - 118) / 2))) + p["depth"]
        return (ap + ov + drill_length) / (rpm * feed) * p["turn"]
    if process == "Facing":
        turn = p["length"] / p["depth of cut"]
        return ((ap + ov + (p["diameter"] / 2)) / (feed * rpm)) * turn
    if process == "Grooving":
        turn = p["length"] / p["tool width"]
        return ((ap + ov + ((p["final diameter"] - p["initial diameter"]) / 2)) / (feed * rpm)) * turn
    if process == "Knurling":
        return ((ap + ov + p["length"]) / (feed * rpm)) * 2
    if process == "Reaming":
        turn = (p["final diameter"] - p["initial diameter"]) / (2 * p["depth of cut"])
        return ((ap + ov + p["length"]) / (feed * rpm)) * turn
    if process == "Threading":
        turn = (0.6134 * p["pitch"]) / p["depth of cut"]
        return ((ap + ov + p["length"]) / (rpm * feed)) * turn
    if process in ("Turning - Concave", "Turning - Convex"):
        arc = p["length"] * math.radians(p["angle"])
        turn = p["diameter"] / (p["depth of cut"] * 4)
        return ((ap + ov + arc) / (feed * rpm)) * turn
    if process == "Turning - Straight":
        turn = (p["initial diameter"] - p["final diameter"]) / (p["depth of cut"] * 2)
        return ((ap + ov + p["length"]) / (feed * rpm)) * turn
    if process == "Turning - Taper":
        h = math.sqrt(p["length"] ** 2 + ((p["larger diameter"] - p["smaller diameter"]) / 2) ** 2)
        turn = (p["larger diameter"] - p["smaller diameter"]) / (p["depth of cut"] * 2)
        return ((ap + ov + h) / (feed * rpm)) * turn
    raise KeyError(f"not a process of the calculator: {process!r}")


def mill_rates(p: dict):
    """(n, Vf): spindle speed n = 1000 Vc / (pi D) in rpm and table feed Vf = n x fz x Z in mm/min (the formulas of the sheets). May raise ZeroDivisionError."""
    n = 1000.0 * p["cutting speed"] / (math.pi * p["cutter diameter"])
    return n, n * p["feed per tooth"] * p["teeth"]


def mill_passes(p: dict) -> int:
    """Passes in depth: total depth / depth of cut per pass, rounded up (a pass cannot be a fraction), at least 1. May raise ZeroDivisionError."""
    return max(1, math.ceil(p["total depth"] / p["depth of cut"]))


def _mill_cut(process: str, p: dict) -> float:
    n, vf = mill_rates(p)
    ap, ov, L, D = p["approach"], p["overrun"], p["length"], p["cutter diameter"]
    passes = mill_passes(p)
    if process == "Milling - Face":            # sheet: Tm = (L + D) / Vf
        return (ap + ov + L + D) / vf * passes
    if process == "Milling - Slab":            # sheet: Tm = (L + sqrt(d (D - d))) / Vf, d = depth of cut of the pass
        d = min(p["depth of cut"], p["total depth"])
        return (ap + ov + L + math.sqrt(max(d * (D - d), 0.0))) / vf * passes
    if process == "Milling - Slot":            # a straight cut of length L at the table feed, once per pass (as v1)
        return (ap + ov + L) / vf * passes
    if process == "Milling - Pocket":          # the cut of length L repeated across the width W in steps of ae
        return (ap + ov + L) / vf * math.ceil(p["width"] / p["stepover"]) * passes
    if process == "Milling - Profile":         # round the outline: 2 L + 2 W (as v1)
        return (ap + ov + 2 * L + 2 * p["width"]) / vf * passes
    raise KeyError(process)


def process_time(process: str, p: dict) -> float:
    """Time of one machining process in minutes = cutting time + job setting + tool setting, as in the calculator. 0 when a divisor is 0 (as there).
    One difference, for inputs that make no sense: a negative cutting time (a final diameter above the initial one) is taken as 0, not subtracted
    from the setting time."""
    try:
        cut = _cut(process, p)
    except ZeroDivisionError:
        return 0.0
    return max(cut, 0.0) + p["job set time"] + p["tool set time"]


def divides_by_zero(process: str, p: dict) -> bool:
    try:
        _cut(process, p)
    except ZeroDivisionError:
        return True
    return False


# ---------------------------------------------------------------- from the part to the inputs of the calculator
def auto_fields_src(op: str, dims: dict, own: dict = None, material: str = None):
    """(fields, source): the calculator inputs of `op` for a part (rules below) and, for each input, where its value comes from:
    "drawing" (a number the drawing gives for the feature this operation makes), "part" (a dimension of the part, or a ratio of the
    calculator's own defaults applied to one) or "default" (the calculator's default). `dims` = the part's dimensions (diameter, length, depth,
    width, optional hole_diameter, mm); `own` = {diameter, length, depth, width, pitch} of the drawing for this operation.

      Facing               diameter <- own diameter or part diameter;   facing length <- own depth
      Turning - Straight   initial diameter <- part diameter;  final diameter <- own diameter (if smaller) else 38/50 of the initial;
                           turning length <- own length or part length          (Plain/Cylindrical Turning, Step Turning)
      Turning - Taper      larger diameter <- own diameter or part diameter, smaller 38/50 of it;  length <- own length or part length
      Turning - Convex     diameter <- own diameter or part diameter;  length <- own length or part length          (Contour Turning)
      Grooving             initial diameter <- own diameter or part diameter;  final = initial + 2 x own depth (else + 0.5);  groove length <- own width
      Knurling             length <- own length                  (the part length is not the knurled length: the calculator's 18 mm stays)
      Threading            pitch <- own pitch;  length <- own length                          (External Threading, Tapping; the part length is not the thread length)
      Boring               initial diameter <- own diameter or hole diameter;  final 28/25 of it;  boring length <- own depth or part depth
      Drilling - Main      drill diameter <- own diameter or hole diameter;  drilling depth <- own depth or part depth
      Reaming              final diameter <- own diameter or hole diameter;  initial 28/30 of it;  length <- own depth or part depth
      Drilling - Center    nothing (the calculator's defaults)
    "hole diameter" is dims["hole_diameter"] when it is set; without it the default of the calculator stays.
    A milling operation (MILLING_OF, Step 53) takes: cut length <- own length or part length, total depth <- own depth or part depth, width <- own width
    or part width (as the v1 time model does for milling); cutter, teeth, Vc, fz, depth per pass, stepover = the v1 values for the operation and `material`
    (source "v1"); setting times and tool cost = the calculator's defaults."""
    if op in MILLING_OF:
        return _mill_fields(op, dims, own, material)
    proc = PROCESS_OF[op]
    f = dict(DEFAULTS[proc])
    src = {k: "default" for k in f}
    own = own or {}
    D, L, T, Hd = dims.get("diameter"), dims.get("length"), dims.get("depth"), dims.get("hole_diameter")

    def put(key, *candidates):
        """The first candidate (value, source) that has a value goes into `key`; none: the default stays."""
        for value, where in candidates:
            if value:
                f[key], src[key] = float(value), where
                return

    od, ol, ot, ow, opitch = (own.get(k) for k in ("diameter", "length", "depth", "width", "pitch"))
    if proc == "Facing":
        put("diameter", (od, "drawing"), (D, "part"))
        put("length", (ot, "drawing"))
    elif proc == "Turning - Straight":
        put("initial diameter", (D, "part"))
        init = f["initial diameter"]
        if od and od < init:
            put("final diameter", (od, "drawing"))
        else:
            f["final diameter"], src["final diameter"] = init * FINAL_TO_INITIAL_TURNING, ("part" if src["initial diameter"] != "default" else "default")
        put("length", (ol, "drawing"), (L, "part"))
    elif proc == "Turning - Taper":
        put("larger diameter", (od, "drawing"), (D, "part"))
        f["smaller diameter"], src["smaller diameter"] = f["larger diameter"] * FINAL_TO_INITIAL_TURNING, ("part" if src["larger diameter"] != "default" else "default")
        put("length", (ol, "drawing"), (L, "part"))
    elif proc in ("Turning - Convex", "Turning - Concave"):
        put("diameter", (od, "drawing"), (D, "part"))
        put("length", (ol, "drawing"), (L, "part"))
    elif proc == "Grooving":
        put("initial diameter", (od, "drawing"), (D, "part"))
        f["final diameter"] = f["initial diameter"] + (2 * ot if ot else GROOVE_FINAL_MINUS_INITIAL)
        src["final diameter"] = "drawing" if ot else src["initial diameter"]
        put("length", (ow, "drawing"))
    elif proc == "Knurling":
        put("length", (ol, "drawing"))
    elif proc == "Threading":
        put("pitch", (opitch, "drawing"))
        put("length", (ol, "drawing"))
    elif proc == "Boring":
        put("initial diameter", (od, "drawing"), (Hd, "part"))
        if src["initial diameter"] != "default":
            f["final diameter"], src["final diameter"] = f["initial diameter"] * BORE_FINAL_TO_INITIAL, src["initial diameter"]
        put("depth", (ot, "drawing"), (T, "part"))
    elif proc == "Drilling - Main":
        put("diameter", (od, "drawing"), (Hd, "part"))
        put("depth", (ot, "drawing"), (T, "part"))
    elif proc == "Reaming":
        put("final diameter", (od, "drawing"), (Hd, "part"))
        if src["final diameter"] != "default":
            f["initial diameter"], src["initial diameter"] = f["final diameter"] * REAM_INITIAL_TO_FINAL, src["final diameter"]
        put("length", (ot, "drawing"), (T, "part"))
    return f, src


def _mill_fields(op: str, dims: dict, own: dict, material: str):
    f = milling_defaults(op, material)
    src = {k: ("v1" if k in ("cutter diameter", "teeth", "cutting speed", "feed per tooth", "depth of cut", "stepover") else "default") for k in f}
    own = own or {}
    for key, own_key in (("length", "length"), ("total depth", "depth"), ("width", "width")):
        if key not in f:
            continue
        for value, where in ((own.get(own_key), "drawing"), ((dims or {}).get(own_key), "part")):
            if value:
                f[key], src[key] = float(value), where
                break
    return f, src


def auto_fields(op: str, dims: dict, own: dict = None, material: str = None) -> dict:
    """The inputs of auto_fields_src, without the sources."""
    return auto_fields_src(op, dims, own, material)[0]


def inputs_src(op: str, dims: dict, own: dict = None, material: str = None):
    """(values, sources) of the inputs of any operation that has a process, a milling process or an 'other process'."""
    if op in PROCESS_OF or op in MILLING_OF:
        return auto_fields_src(op, dims, own, material)
    if op in EXTRA_OF:
        values = {k: d for k, _, d in fields_of(op)}
        return values, {k: "default" for k in values}
    raise KeyError(f"{op!r} has no process in the calculator")


AUTO_KEY = "calc_auto"


def key_of(op: str, field: str) -> str:
    """The session / widget key of one calculator input."""
    return f"calc|{op}|{field}"


def sync_state(state, autos) -> None:
    """Fill the input boxes of the calculator panel (`autos` = {op: {input: value}}, from inputs_src). A box that still holds the value this
    function put there last time follows the new value (another part diameter, a new drawing); a box the user changed keeps what the user typed;
    a box that is not in the state yet (new, or dropped because it was not drawn in a run) is filled. `state` = st.session_state or a dict."""
    last = state.setdefault(AUTO_KEY, {})
    for op, fields in autos.items():
        for name, value in fields.items():
            k = key_of(op, name)
            if state.get(k) is None or state[k] == last.get(k):
                state[k] = float(value)
            last[k] = float(value)


def edits_from_values(autos, values) -> dict:
    """{op: {input: value}}: the boxes whose value differs from the one offered (Settings.edits). `values` = {key_of(op, input): number}."""
    out = {}
    for op, fields in autos.items():
        for name, value in fields.items():
            x = values.get(key_of(op, name))
            if x is not None and abs(float(x) - float(value)) > 1e-9:
                out.setdefault(op, {})[name] = float(x)
    return out


def describe_sources(op: str, sources: dict, edited) -> str:
    """A line for under the inputs of an operation: which come from the drawing, which from the part, which were edited, which are the calculator's defaults."""
    label = {k: lab.split(" (")[0].lower() for k, lab, _ in fields_of(op)}
    edited = set(edited or ())
    groups = {"drawing": [], "part": [], "v1": [], "default": []}
    for k, where in sources.items():
        if k not in edited:
            groups[where].append(label.get(k, k))
    out = []
    if groups["drawing"]:
        out.append("\U0001F4D0 from the drawing: " + ", ".join(groups["drawing"]))
    if groups["part"]:
        out.append("from the part: " + ", ".join(groups["part"]))
    if groups["v1"]:
        out.append("v1 cutter and cutting data: " + ", ".join(groups["v1"]))
    if edited:
        out.append("\u270F\uFE0F edited: " + ", ".join(label.get(k, k) for k in sources if k in edited))
    if groups["default"]:
        out.append("calculator default: " + ", ".join(groups["default"]))
    return " | ".join(out)


class Settings:
    """What the user types on the calculator's page: labour cost per hour, material cost, and the inputs the user changed
    (edits = {operation: {input: value}}; an input that is not in it takes auto_fields())."""

    def __init__(self, labor_per_hour=LABOR_PER_HOUR, material_cost=MATERIAL_COST, edits=None, own=None):
        self.labor_per_hour = float(labor_per_hour)
        self.material_cost = float(material_cost)
        self.edits = {op: dict(v) for op, v in (edits or {}).items() if v}
        self.own = {op: dict(v) for op, v in (own or {}).items() if v}      # drawing numbers per operation (OPD.drawing_values, as numbers)


REFERENCE_DIMS = {"diameter": 25.0, "length": 50.0, "depth": 20.0, "width": 20.0}


def _own_of(op, dims, settings):
    own = dict(settings.own.get(op) or {})
    for k, v in ((dims or {}).get(agents.PER_OP) or {}).get(op, {}).items():     # the own dimensions of Step 51 (v1) count as well
        own.setdefault(k, v)
    return own


def op_row(op: str, material: str, dims: dict = None, settings: Settings = None) -> dict:
    """One operation: {"op", "source", "process", "time_min", "labor_cost", "tool_cost", "extra_cost", "fields", "note"}.
    source = "calculator" (a process of the calculator), "calculator (other process)" or NOT_IN_CALCULATOR."""
    settings = settings or Settings()
    dims = dims if dims is not None else REFERENCE_DIMS
    row = {"op": op, "source": None, "process": None, "time_min": 0.0, "labor_cost": 0.0, "tool_cost": 0.0, "extra_cost": 0.0, "fields": {}, "note": ""}
    edits = settings.edits.get(op, {})
    if op in PROCESS_OF or op in MILLING_OF:
        proc = PROCESS_OF.get(op) or MILLING_OF[op]
        milling = op in MILLING_OF
        f = auto_fields(op, dims, _own_of(op, dims, settings), material)
        f.update({k: v for k, v in edits.items() if k in f})
        row.update({"source": MILLING_SOURCE if milling else "calculator", "process": proc, "fields": f, "time_min": process_time(proc, f),
                    "tool_cost": f["tool cost"]})
        if divides_by_zero(proc, f):
            row["note"] = ("a divisor is 0 (cutter diameter, teeth, Vc, fz, depth of cut or stepover): time 0 as in the calculator" if milling else
                           "a divisor is 0 (feed, rpm, depth of cut or tool width): the calculator gives time 0")
        elif milling:
            n, vf = mill_rates(f)
            row["note"] = f"n = {n:.0f} rpm, Vf = {vf:.0f} mm/min, {mill_passes(f)} pass(es)"
    elif op in EXTRA_OF:
        proc = EXTRA_OF[op]
        t, c = EXTRA[proc]
        f = {"time_min": edits.get("time_min", t), "extra_tool_cost": edits.get("extra_tool_cost", c)}
        row.update({"source": "calculator (other process)", "process": proc, "fields": f,
                    "time_min": f["time_min"] + HANDLING_MIN, "extra_cost": f["extra_tool_cost"], "note": f"+{HANDLING_MIN:g} min handling"})
    else:
        cut = agents._op_time(op, material, dims)
        if op == "Inspection":
            row.update({"source": NOT_IN_CALCULATOR, "time_min": cut, "note": "fixed time of the v1 model"})
        else:
            row.update({"source": NOT_IN_CALCULATOR, "time_min": cut + FALLBACK_TOOL_SETTING_MIN, "tool_cost": FALLBACK_TOOL_COST,
                        "note": f"v1 cutting time {cut:g} min + {FALLBACK_TOOL_SETTING_MIN:g} min tool setting, tool cost Rs {FALLBACK_TOOL_COST:g} (the calculator's defaults)"})
    row["labor_cost"] = row["time_min"] * settings.labor_per_hour / 60.0
    return row


def evaluate(steps: list, material: str, dims: dict = None, settings: Settings = None) -> dict:
    """The calculator's Result page for a route: one row per operation (a machine changeover is a row of its own), and the summary."""
    settings = settings or Settings()
    rows = []
    for s in steps:
        if s == MARKER:
            t = float(agents.CHANGEOVER_TIME_MIN)
            rows.append({"op": s, "source": "v1 model (not in the calculator)", "process": None, "time_min": t, "labor_cost": t * settings.labor_per_hour / 60.0,
                         "tool_cost": 0.0, "extra_cost": 0.0, "fields": {}, "note": "lathe <-> milling: the v1 changeover time; cost = labour of those minutes"})
        else:
            rows.append(op_row(s, material, dims, settings))
    total_time = sum(r["time_min"] for r in rows)
    labor = total_time / 60.0 * settings.labor_per_hour
    tool = sum(r["tool_cost"] for r in rows)
    extra = sum(r["extra_cost"] for r in rows)
    ops = [s for s in steps if s != MARKER]
    return {"rows": rows, "time_min": total_time, "labor_cost": labor, "material_cost": settings.material_cost, "tool_cost": tool,
            "extra_cost": extra, "total_cost": settings.material_cost + labor + tool + extra,
            "energy_kwh": agents.energy_agent([x for x in steps if x not in CALCULATOR_ONLY], material, dims),
            "n_calculator": sum(1 for r in rows if r["source"] and r["source"].startswith("calculator")),
            "n_milling": sum(1 for r in rows if r["source"] == MILLING_SOURCE),
            "n_v1": sum(1 for r in rows if r["op"] != MARKER and r["source"] == NOT_IN_CALCULATOR), "n_ops": len(ops)}


# the three agents, with the signatures of week3/agents.py plus the settings
def time_agent(steps, material, dims=None, settings=None):
    return round(evaluate(steps, material, dims, settings)["time_min"], 2)


def cost_agent(steps, material, batch_size=1, dims=None, settings=None):
    return round(evaluate(steps, material, dims, settings)["total_cost"], 2)


def energy_agent(steps, material, dims=None, settings=None):
    return agents.energy_agent([x for x in steps if x not in CALCULATOR_ONLY], material, dims)


def relative_scores(triples) -> list:
    """For routes that are compared with each other: per route (time, cost, energy) the score 100 x (best value among the routes / the
    route's value). 100 = the best of the routes in that figure. A value of 0 or less (a time of 0 from a divisor of 0) scores 100 and is not
    taken as the best, so one such route does not give every other route a score of 0."""
    triples = [tuple(t) for t in triples]
    if not triples:
        return []
    best = []
    for k in range(3):
        positive = [t[k] for t in triples if t[k] > 0]
        best.append(min(positive) if positive else 0.0)
    return [tuple(100.0 if t[k] <= best[k] or t[k] <= 0 else 100.0 * best[k] / t[k] for k in range(3)) for t in triples]


def relative_efficiency(triples) -> list:
    """Efficiency of routes that are compared with each other: the three relative scores averaged (relative_scores). 100 = best in all three.
    The v1 efficiency (agents.efficiency_agent) uses fixed scales made for the v1 times (a few minutes), where the calculator's times
    (setting times of 5 min per operation) would all score 0."""
    return [round(sum(sc) / 3, 2) for sc in relative_scores(triples)]
