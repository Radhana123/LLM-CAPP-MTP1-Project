"""Tool selection and handbook ranges for the process plan (report 6.7, WP9). Tables only; capp/process_plan.py uses them.

Report 6.7: "Tool selection assigns a tool type and tool material to each operation from a rule table keyed on the operation and the
workpiece material. Cutting parameters are retrieved per operation from tables curated from standard machining handbooks, and checked
against power and material-removal limits. The route is selected first and the parameters second."

What is in here and how far to trust it (say this in the thesis):
  OPERATIONS     one row per operation of the closed vocabulary (41): what it does, the tool it needs, the tool family.
  TOOL_MATERIAL  tool material by (tool family, workpiece material).
  RANGES         the cutting-speed and feed range that a source gives for (speed class, workpiece material), for the tool material of
                 TOOL_MATERIAL. It is the REFERENCE the plan's parameters are compared with (report: "comparison with handbook ranges"),
                 not a second source of parameters: the parameters of a plan are the v1 model's per-material speeds and feeds
                 (week3/agents.py, the numbers the Time agent already uses), so the route search is not touched.

Every row carries a `status`:
  web          every bound of the range lies within the numbers of the web pages listed in `sources` (SOURCES below), which were read
  mixed        at least one bound is from a read page, the other is a typical-practice figure that no page read here gives
  indicative   no page was read for it; typical general machining practice
  confirmed    checked against the handbook of the report ([8]) by the author or the supervisor and the page written in `source_note`
No row is "confirmed" yet: no handbook was available to the author of this module (as for capp/feasibility.py). The route sheet prints
the count of each status next to the parameters, and says so when nothing is confirmed. To confirm a row: check it, then set
status = "confirmed" and write the page in source_note. The table is meant to be edited; tests/test_process_plan.py keeps it complete.

Limits of the web sources: they are tool-maker and course pages, not a handbook; they disagree with each other by up to a factor of two
for carbide turning (course pages are conservative, tool-maker pages are not), so a range spans them; drilling feeds are given for 6-12 mm
drills and rise with the diameter; none of them is for the part's actual grade, hardness, tool or machine.
"""
SOURCES = {
    "S1": ("Aims Industrial, cutting speeds & feeds reference (drilling with HSS / HSS-Co / HSSE / coated HSS and solid carbide, slotting with "
           "carbide: Vc m/min, feed mm/rev for 6-12 mm drills, mm/tooth)",
           "https://aimsindustrial.com.au/blogs/product-guides/cutting-speeds-feeds-reference"),
    "S2": ("University of Florida, MAE design lab, 'Speeds and Feeds' (recommended surface speeds for HSS in ft/min, x2.5 for carbide; "
           "end-mill feed per tooth in inch)",
           "https://web.mae.ufl.edu/designlab/Advanced%20Manufacturing/Speeds%20and%20Feeds/Speeds%20and%20Feeds.htm"),
    "S3": ("University of Toronto, MIE, 'Module 1 The Drill Press' (HSS drilling speeds by hardness, drilling feed by drill diameter)",
           "https://www.mie.utoronto.ca/wp-content/uploads/2016/10/MODULE-1-The-Drill-Press.pdf"),
    "S4": ("incoherency.co.uk, lathe speeds (turning speeds for HSS and carbide in m/min; the page itself cites Chegg: a secondary source)",
           "https://incoherency.co.uk/jesref/lathe-speeds.html"),
}

MATERIALS = ("Aluminum", "Steel", "Brass", "Copper", "Titanium", "Plastic", "Cast Iron")
STATUSES = ("confirmed", "web", "mixed", "indicative")

# family -> what the operation is, for the tool-material rule and for the speed class
FAMILIES = ("turn", "bore", "groove", "thread_turn", "form", "drill", "center_drill", "tap", "ream", "mill", "engrave", "none")

# operation: (what it does, tool type, family)
OPERATIONS = {
    "Facing": ("Face the end of the stock to clean it up and set the length datum.", "facing / turning tool, indexable insert", "turn"),
    "Plain/Cylindrical Turning": ("Turn the outside diameter to size over the required length.", "turning tool, indexable insert", "turn"),
    "Step Turning": ("Turn stepped diameters (shoulders) to size.", "turning tool, indexable insert", "turn"),
    "Taper Turning": ("Turn a conical surface to the specified taper.", "turning tool, indexable insert", "turn"),
    "Contour Turning": ("Turn a curved or complex outline along the axis.", "profiling turning tool, indexable insert", "turn"),
    "Eccentric Turning": ("Turn a diameter whose axis is offset from the main axis (offset set-up).", "turning tool, indexable insert", "turn"),
    "Grooving/Necking": ("Cut an external groove or neck.", "external grooving insert", "groove"),
    "Internal Grooving": ("Cut a groove inside a bore.", "internal grooving tool (bar)", "groove"),
    "Undercutting": ("Cut a relief undercut at a shoulder or thread end.", "grooving / undercut insert", "groove"),
    "Parting-off": ("Part the finished component from the stock.", "parting-off blade, indexable insert", "groove"),
    "External Threading": ("Cut the external thread to the specified pitch.", "external threading insert", "thread_turn"),
    "Knurling": ("Roll a knurled pattern on the diameter.", "knurling tool (straight or diamond wheels)", "form"),
    "Forming": ("Form a special profile with a shaped tool.", "form tool", "form"),
    "Polishing/Burnishing": ("Burnish or polish the surface to the required finish.", "burnishing roller / polishing tool", "form"),
    "Boring": ("Enlarge and finish a drilled hole to size and position.", "boring bar with indexable insert", "bore"),
    "Center Drilling": ("Spot the hole centre so that the drill starts on position.", "center drill (drill and countersink combined)", "center_drill"),
    "Chamfering": ("Break the edge with a chamfer.", "chamfering tool / chamfer cutter", "turn"),
    "Counterboring": ("Enlarge the top of a hole with a flat bottom (screw head seat).", "counterbore (piloted)", "drill"),
    "Countersinking": ("Cone-shaped enlargement of the hole mouth.", "countersink", "drill"),
    "Drilling": ("Drill the hole to the required depth.", "twist drill", "drill"),
    "Reaming": ("Finish a drilled or bored hole to the tolerance and surface finish.", "machine reamer", "ream"),
    "Tapping": ("Cut the internal thread in the drilled hole.", "machine tap", "tap"),
    "Inspection": ("Inspect the dimensions and tolerance against the drawing.", "gauges / micrometer / CMM (no cutting tool)", "none"),
    "Angular Milling": ("Mill an inclined (angular) face.", "angle cutter / end mill", "mill"),
    "Corner Rounding/Filleting": ("Mill a radius at an edge or corner.", "corner-rounding end mill", "mill"),
    "Dovetail Milling": ("Mill a dovetail form.", "dovetail cutter", "mill"),
    "Engraving": ("Engrave an identification mark.", "engraving cutter", "engrave"),
    "Face Milling": ("Mill a flat face to size and finish.", "face mill, indexable inserts", "mill"),
    "Form Milling": ("Mill a special profile with a shaped cutter.", "form cutter", "mill"),
    "Gang Milling": ("Mill several surfaces at once with a gang of cutters on one arbor.", "side-and-face cutters (gang)", "mill"),
    "Gear/Spline Milling": ("Cut gear teeth or splines.", "gear / spline form cutter", "mill"),
    "Helical Milling": ("Mill a helical groove or an interpolated hole.", "end mill", "mill"),
    "Pocket Milling": ("Mill a recess (pocket) to depth.", "end mill", "mill"),
    "Profile Milling": ("Mill the outline of the part.", "end mill", "mill"),
    "Slab/Peripheral Milling": ("Mill a wide flat surface with the periphery of the cutter.", "slab / shell mill", "mill"),
    "Slot Milling": ("Mill a slot to width and depth.", "end mill (slot drill)", "mill"),
    "Spotfacing": ("Machine a flat seat around a hole.", "spotfacing cutter / end mill", "mill"),
    "Surface Contouring": ("Mill a 3D curved surface.", "ball-nose end mill", "mill"),
    "T-Slot Milling": ("Mill the T-slot after the plain slot.", "T-slot cutter", "mill"),
    "Thread Milling": ("Mill a thread by helical interpolation.", "thread mill", "mill"),
    "Woodruff Keyway Milling": ("Mill a Woodruff keyseat.", "Woodruff keyseat cutter", "mill"),
}

# tool family -> the tool-material group it follows
GROUP_OF_FAMILY = {"turn": "insert", "bore": "insert", "groove": "insert", "thread_turn": "insert", "mill": "mill",
                   "drill": "drill", "center_drill": "drill", "tap": "tap", "ream": "ream", "form": "form", "engrave": "engrave",
                   "none": "none"}

_INSERT = {
    "Steel": "coated carbide (CVD / PVD)",
    "Cast Iron": "coated carbide (cast-iron grade)",
    "Aluminum": "uncoated or polished carbide, sharp positive geometry (PCD for large batches)",
    "Brass": "uncoated carbide, positive or zero rake",
    "Copper": "uncoated carbide, sharp positive geometry",
    "Titanium": "uncoated or PVD-coated carbide, sharp edge, coolant",
    "Plastic": "uncoated polished carbide or HSS, sharp positive geometry",
}
_MILL = {
    "Steel": "coated carbide",
    "Cast Iron": "coated carbide",
    "Aluminum": "uncoated carbide, polished flutes (2-3 flutes)",
    "Brass": "uncoated carbide or HSS-Co",
    "Copper": "uncoated carbide, polished flutes",
    "Titanium": "coated carbide (AlCrN / TiAlN), coolant",
    "Plastic": "polished carbide or HSS, 1-2 flutes",
}
_DRILL = {
    "Steel": "HSS-Co (or TiN / TiAlN-coated HSS)",
    "Cast Iron": "HSS-Co (or TiN-coated HSS)",
    "Aluminum": "HSS-Co, polished flutes",
    "Brass": "HSS-Co, ground for brass (reduced rake)",
    "Copper": "HSS-Co",
    "Titanium": "HSS-Co, coolant (solid carbide for large batches)",
    "Plastic": "HSS, polished flutes, low-angle point",
}
_TAP = {m: "HSS-E" + (", TiN-coated" if m in ("Steel", "Cast Iron") else "") for m in MATERIALS}
_REAM = {m: ("carbide-tipped" if m in ("Cast Iron", "Titanium") else "HSS") for m in MATERIALS}

TOOL_MATERIAL = {("insert", m): v for m, v in _INSERT.items()}
TOOL_MATERIAL.update({("mill", m): v for m, v in _MILL.items()})
TOOL_MATERIAL.update({("drill", m): v for m, v in _DRILL.items()})
TOOL_MATERIAL.update({("tap", m): v for m, v in _TAP.items()})
TOOL_MATERIAL.update({("ream", m): v for m, v in _REAM.items()})
TOOL_MATERIAL.update({("form", m): "hardened steel wheels / HSS" for m in MATERIALS})
TOOL_MATERIAL.update({("engrave", m): "solid carbide, pointed" for m in MATERIALS})
TOOL_MATERIAL.update({("none", m): "-" for m in MATERIALS})

# the range class an operation is compared with, by tool family
# (the v1 model gives boring, threading, tapping and reaming the DRILL speed class of its table; here an operation is compared with the range of
# the tool it really uses: boring and threading inserts with the carbide turning range. Tapping, reaming, knurling-type forming and engraving have no
# range in the pages read: no reference, no verdict.)
SPEED_CLASS = {"turn": "turn", "bore": "turn", "groove": "turn", "thread_turn": "turn", "form": None, "drill": "drill",
               "center_drill": "drill", "tap": None, "ream": None, "mill": "mill", "engrave": None, "none": None}

# (speed class, material) -> range. vc in m/min; feed in mm/rev (turn, drill) or mm/tooth (mill).
# turn: carbide insert. drill: HSS-Co / HSSE / coated HSS, feed for 6-12 mm drills. mill: carbide, slotting.
_R = {}


def _row(cls, mat, vc, f, status, sources, note=""):
    _R[(cls, mat)] = {"vc": vc, "feed": f, "status": status, "sources": list(sources), "note": note, "source_note": ""}


# carbide turning. Lower ends: S2 (HSS ft/min x 2.5 for carbide, converted); S4 midpoints; upper ends: usual tool-maker practice
_row("turn", "Aluminum", (190, 600), (0.10, 0.50), "mixed", ["S2", "S4"], "lower end S2 (190 m/min), S4 335-365; upper end typical practice; feeds typical practice")
_row("turn", "Steel", (75, 250), (0.10, 0.40), "mixed", ["S2", "S4"], "lower end S2 (76 m/min), S4 mild steel 90-135; upper end typical practice; feeds typical practice")
_row("turn", "Brass", (150, 300), (0.10, 0.40), "mixed", ["S2", "S4"], "S2 190 m/min, S4 150-200; upper end typical practice; feeds typical practice")
_row("turn", "Copper", (60, 250), (0.10, 0.40), "mixed", ["S2", "S4"], "S4 copper / phosphor bronze 60-65, S2 bronze 76; upper end typical practice; feeds typical practice")
_row("turn", "Titanium", (15, 90), (0.10, 0.30), "mixed", ["S2"], "lower end S2 (15 m/min, conservative); upper end typical practice; feeds typical practice")
_row("turn", "Plastic", (150, 450), (0.10, 0.50), "mixed", ["S4"], "S4 wood and plastic, carbide 150-450; feeds typical practice")
_row("turn", "Cast Iron", (75, 200), (0.15, 0.50), "mixed", ["S2", "S4"], "S2 76 m/min, S4 class 20 iron 135; upper end typical practice; feeds typical practice")
# drilling, HSS-Co / HSSE / coated HSS (S1 columns), feed for 6-12 mm drills
_row("drill", "Steel", (18, 44), (0.10, 0.20), "web", ["S1", "S3"], "S1 mild and medium-carbon steel: HSS 18-30, HSS-Co 24-38, HSSE 28-44; feed 0.10-0.20; S3 25-37 for 120-150 HB")
_row("drill", "Cast Iron", (18, 35), (0.15, 0.25), "web", ["S1"], "S1 grey cast iron, HSS 18-25 up to HSSE 25-35; feed 0.15-0.25")
_row("drill", "Aluminum", (60, 160), (0.15, 0.30), "web", ["S1"], "S1 wrought aluminium, HSS 60-100 up to HSSE 100-160; feed 0.15-0.30")
_row("drill", "Brass", (50, 130), (0.10, 0.25), "web", ["S1"], "S1 brass / bronze / copper row, HSS 50-80 up to HSSE 80-130; feed 0.10-0.25")
_row("drill", "Copper", (50, 130), (0.10, 0.25), "web", ["S1"], "S1 brass / bronze / copper row (one row for the three)")
_row("drill", "Titanium", (6, 14), (0.04, 0.08), "web", ["S1"], "S1 Ti-6Al-4V with HSS-Co / HSSE / TiAlN-coated HSS 6-14; feed 0.04-0.08 (solid carbide 30-65 is a different tool)")
_row("drill", "Plastic", (50, 200), (0.10, 0.40), "indicative", [], "no page read: typical practice for polished HSS drills in plastics")
# carbide milling, slotting (S1) and end-mill feed per tooth (S2, inch converted)
_row("mill", "Steel", (76, 180), (0.04, 0.20), "web", ["S1", "S2"], "S1 mild steel VHM 100-180 (TiAlN / AlCrN), S2 76 m/min (HSS x2.5); fz S1 0.04-0.10, S2 0.05-0.20")
_row("mill", "Cast Iron", (61, 220), (0.05, 0.15), "web", ["S1", "S2"], "S1 grey iron VHM 120-220, S2 61-76 m/min (medium-hard / soft iron); fz S1 0.05-0.10, S2 0.05-0.15")
_row("mill", "Aluminum", (190, 400), (0.05, 0.30), "web", ["S1", "S2"], "S1 wrought aluminium VHM 200-400, S2 190 m/min; fz S1 0.05-0.15, S2 0.10-0.30")
_row("mill", "Brass", (100, 300), (0.05, 0.30), "mixed", ["S2"], "S2 brass 190 m/min and fz 0.10-0.30; S1 has no brass milling row; wider bounds typical practice")
_row("mill", "Copper", (76, 250), (0.05, 0.15), "mixed", ["S2"], "S2 bronze 76 m/min, fz 0.05-0.15; upper end typical practice")
_row("mill", "Titanium", (15, 65), (0.025, 0.10), "web", ["S1", "S2"], "S1 titanium VHM 30-65, S2 15 m/min; fz S1 0.025-0.05, S2 0.04-0.10")
_row("mill", "Plastic", (100, 500), (0.10, 0.80), "mixed", ["S2"], "S2 acetal: 190 m/min, fz 0.20-0.76; wider bounds typical practice")
RANGES = _R


# ---------------------------------------------------------------- lookups
def operation(op: str) -> dict:
    """What an operation is: description, tool type, family, tool-material group, speed class."""
    desc, tool, fam = OPERATIONS[op]
    return {"op": op, "description": desc, "tool_type": tool, "family": fam, "group": GROUP_OF_FAMILY[fam], "speed_class": SPEED_CLASS[fam]}


# shared operations are done with a different tool on the milling machine: (operation, machine) -> (tool type, tool-material group)
MACHINE_TOOL = {("Facing", "milling"): ("face mill, indexable inserts", "mill"),
                ("Chamfering", "milling"): ("chamfer mill", "mill"),
                ("Boring", "milling"): ("boring head with indexable insert", "insert")}


def tool_for(op: str, material: str, machine: str = None) -> dict:
    """The tool rule for an operation on a workpiece material (and the machine it runs on, for the shared operations whose tool differs):
    {"type", "material", "family", "status", "source_note", "machine_override"}."""
    info = operation(op)
    if material not in MATERIALS:
        raise KeyError(f"unknown material {material!r}")
    tool_type, group = info["tool_type"], info["group"]
    override = (op, machine) in MACHINE_TOOL
    if override:
        tool_type, group = MACHINE_TOOL[(op, machine)]
    return {"type": tool_type, "material": TOOL_MATERIAL[(group, material)], "family": info["family"],
            "status": "indicative", "source_note": "general machining practice; no handbook page checked", "machine_override": override}


def range_for(op: str, material: str):
    """The reference range for the speed class of an operation, or None (inspection)."""
    cls = operation(op)["speed_class"]
    if cls is None:
        return None
    if material not in MATERIALS:
        raise KeyError(f"unknown material {material!r}")
    return dict(RANGES[(cls, material)], speed_class=cls)


def check_range(op: str, material: str, vc, feed) -> dict:
    """Compare a plan's speed and feed with the reference range.
    Each of vc / feed is 'within', 'below' (conservative: a longer cut, less wear) or 'above' (check: more heat, more wear, more power)."""
    r = range_for(op, material)
    out = {"op": op, "range": r, "vc": None, "feed": None, "attention": False}
    if r is None:
        return out
    for key, val in (("vc", vc), ("feed", feed)):
        if val is None:
            continue
        lo, hi = r[key]
        out[key] = "below" if val < lo - 1e-12 else "above" if val > hi + 1e-12 else "within"
    out["attention"] = "above" in (out["vc"], out["feed"])
    return out


def rows() -> list:
    """Every row of the tool rule and the range tables, flat, for review (scripts/route_sheet.py tables)."""
    out = []
    for op, (desc, tool, fam) in sorted(OPERATIONS.items()):
        for m in MATERIALS:
            out.append({"table": "tool", "key": f"{op} | {m}", "value": f"{tool}; {TOOL_MATERIAL[(GROUP_OF_FAMILY[fam], m)]}",
                        "status": "indicative", "sources": "", "note": "general machining practice"})
    for (op, mach), (tool, group) in sorted(MACHINE_TOOL.items()):
        for m in MATERIALS:
            out.append({"table": "tool", "key": f"{op} on {mach} | {m}", "value": f"{tool}; {TOOL_MATERIAL[(group, m)]}",
                        "status": "indicative", "sources": "", "note": "general machining practice"})
    for (cls, m), r in sorted(RANGES.items()):
        unit = "mm/tooth" if cls == "mill" else "mm/rev"
        out.append({"table": "range", "key": f"{cls} | {m}",
                    "value": f"Vc {r['vc'][0]:g}-{r['vc'][1]:g} m/min; feed {r['feed'][0]:g}-{r['feed'][1]:g} {unit}",
                    "status": r["status"], "sources": ",".join(r["sources"]), "note": r["note"] + (" " + r["source_note"] if r["source_note"] else "")})
    return out


def status_counts(used_ranges) -> dict:
    """{status: n} over the ranges a plan used (the tool rules are all 'indicative')."""
    out = dict.fromkeys(STATUSES, 0)
    for r in used_ranges:
        out[r["status"]] += 1
    return out


def validate() -> list:
    """Problems of the tables themselves (empty list = consistent). The tests call this."""
    errs = []
    from capp import verify as V
    if set(OPERATIONS) != set(V.CANONICAL_OPS):
        errs.append(f"operations differ from the vocabulary: {sorted(set(OPERATIONS) ^ set(V.CANONICAL_OPS))}")
    for op in OPERATIONS:
        fam = OPERATIONS[op][2]
        if fam not in FAMILIES:
            errs.append(f"{op}: unknown family {fam}")
        for m in MATERIALS:
            if (GROUP_OF_FAMILY.get(fam), m) not in TOOL_MATERIAL:
                errs.append(f"no tool material for {op} on {m}")
    for cls in ("turn", "drill", "mill"):
        for m in MATERIALS:
            r = RANGES.get((cls, m))
            if r is None:
                errs.append(f"no range for {cls} | {m}")
                continue
            for key in ("vc", "feed"):
                lo, hi = r[key]
                if not 0 < lo < hi:
                    errs.append(f"{cls} | {m}: bad {key} range {lo}-{hi}")
            if r["status"] not in STATUSES:
                errs.append(f"{cls} | {m}: unknown status {r['status']}")
            if r["status"] in ("web", "mixed") and not r["sources"]:
                errs.append(f"{cls} | {m}: status {r['status']} needs a source")
            if r["status"] == "confirmed" and not r["source_note"]:
                errs.append(f"{cls} | {m}: 'confirmed' needs the handbook page in source_note")
            for s in r["sources"]:
                if s not in SOURCES:
                    errs.append(f"{cls} | {m}: unknown source {s}")
    return errs
