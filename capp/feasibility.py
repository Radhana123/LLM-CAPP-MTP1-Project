"""Extended verifier, physical-feasibility rules (report 6.6, WP6).

    from capp import feasibility
    rep = feasibility.evaluate(steps, part)             # part: a dataset record (material, features, tolerance, dims, ...)
    rep.infeasible, rep.risks                           # lists of verify.Violation
    rep.not_evaluated                                   # {rule id: why the rule could not be evaluated} (data missing)
    feasibility.rule_table()                            # id, category, kind, scope, source, source status of every rule

The v1 verifier (capp/verify.py) knows precedence and coverage only. These rules add what a precedence-valid route can still get
wrong: it cannot be machined on the machine, it needs a support, a hole has no hole to work on, the tolerance cannot be held.
Every finding is a verify.Violation (same format as precedence violations, rule ids "PHYS:<RULE>"), with an `action` that says what
the rule asks for ("insert:Reaming", "support:tailstock", "reduce:feed", ...), so that a later repair step or the user can act on it.

Two kinds, both from capp/verify.py
  infeasible (hard)   the route cannot be carried out as planned: the part does not fit the machine, a hole-finishing operation
                      has no hole, a lathe operation after parting-off.
  risk (soft)         needs a support, a smaller feed, a finishing operation, or rests on an assumption. Never makes a route invalid.
Scope: "route" rules judge the route; "part" rules judge the specification (route-independent; counted once per part).

Rules that need dimensions apply only when the part carries them (report 6.6). A rule that cannot be evaluated is reported under
`not_evaluated` with the missing field, never silently passed; a rule that does not apply (no turning operation in the route, say)
is evaluated and has no violation.

THIS MODULE IS ANALYSIS ONLY. Nothing here changes validity (M2), repair, selection or the baselines, unless install() is called.
install() adds the rules to verify.PHYSICAL_RULES, so that verify.check_route(route, features, part) reports them as well; "infeasible"
violations then make the strict check fail, "risk" ones do not.

SOURCES. Report 6.6 asks for a handbook citation for every rule. No handbook was available to the author of this module, so each rule
carries the source that WAS checked (`source`, with its URL) and a `source_status`:
  web         a web source that was read (vendor article, course notes, reference tables); to be replaced or confirmed by a handbook page
  derived     follows from the v1 vocabulary and precedence rules (no external source needed)
  model       follows from the cutting-force model of capp/physics.py (assumptions in physics.ASSUMPTIONS)
  assumption  rests on a number that has no source yet (machine envelope, Ra floors); the rule is listed so that the number can be
              replaced, and its findings are risks, never hard violations, unless the number is the machine's own data sheet
No rule is marked "handbook": none has been confirmed against one yet, and the supervisor's review is still to come.

Rules that have no source and are therefore NOT encoded: minimum chuck grip length, boring-bar overhang limits, thin-wall limits,
end-mill length-to-diameter limits, material-specific operation bans. They are listed in NOT_ENCODED.
"""
import math
import re
from dataclasses import dataclass, field

from capp import legacy  # noqa: F401
from capp import physics
from capp import verify as V

import route_builder as rb

# ---- assumptions (every one is a parameter; none is a measurement) -------------------------------------------------------
@dataclass(frozen=True)
class Machine:
    """Envelope and power of the machines the plan is made for. The defaults are an ASSUMED small CNC lathe and vertical machining
    centre, not the machines of the department: replace them with the data sheets."""
    name: str = "assumed small CNC lathe + vertical machining centre"
    lathe_max_diameter_mm: float = 250.0
    lathe_max_length_mm: float = 500.0
    mill_travel_mm: tuple = (600.0, 400.0, 400.0)         # X, Y, Z
    spindle_power_kw: float = physics.SPINDLE_POWER_KW


DEFAULT_MACHINE = Machine()

TOLERANCE_BAND_FACTOR = 2.0       # the dataset's tolerance string ("0.05mm") is read as +-0.05 mm, a band of 0.10 mm (less strict than reading it as the band)
CHUCK_ONLY_LD = 3.0               # source: CADEM, chuck only below L/D 3
TAILSTOCK_MAX_LD = 6.0            # tailstock for 3 <= L/D < 6; steady rest from 6
REDUCED_PARAMS_LD = 12.0          # above this L/D the source asks for reduced cutting parameters even with a steady rest
SOURCE_MAX_LD = 20.0              # the source gives nothing above L/D 20
DEEP_HOLE_LD = 3.0                # source: a deep hole is 3 x D (Mt Hood CC CNC course)
MULTI_PECK_LD = 5.0               # beyond 5 x D the multi-peck cycle (G83)

# ISO 286-1 standard tolerance grades, micrometres, by nominal size range (mm): Wikipedia "IT Grade", Table 1. Only the verified range 6-250 mm.
ISO286_UM = [
    (6, 10, {4: 4, 5: 6, 6: 9, 7: 15, 8: 22, 9: 36, 10: 58, 11: 90}),
    (10, 18, {4: 5, 5: 8, 6: 11, 7: 18, 8: 27, 9: 43, 10: 70, 11: 110}),
    (18, 30, {4: 6, 5: 9, 6: 13, 7: 21, 8: 33, 9: 52, 10: 84, 11: 130}),
    (30, 50, {4: 7, 5: 11, 6: 16, 7: 25, 8: 39, 9: 62, 10: 100, 11: 160}),
    (50, 80, {4: 8, 5: 13, 6: 19, 7: 30, 8: 46, 9: 74, 10: 120, 11: 190}),
    (80, 120, {4: 10, 5: 15, 6: 22, 7: 35, 8: 54, 9: 87, 10: 140, 11: 220}),
    (120, 180, {4: 12, 5: 18, 6: 25, 7: 40, 8: 63, 9: 100, 10: 160, 11: 250}),
    (180, 250, {4: 14, 5: 20, 6: 29, 7: 46, 8: 72, 9: 115, 10: 185, 11: 290}),
]
# IT grade a process attains: Engineering ToolBox "Machine processes, tolerance grades" (cites ANSI B4.1). Grinding, honing, lapping
# (IT4-5) are not operations of the 41-operation vocabulary.
PROCESS_IT = {"Drilling": 10, "Boring": 8, "Reaming": 6, "Turning": 7, "Milling": 10}
HOLE_FEATURES = ("Hole", "Reamed_Hole", "Bore")
TURNED_FEATURES = ("Turning", "Step", "Taper", "Contour_Turn", "Eccentric")
HOLE_IT_OPS = {op: PROCESS_IT[op] for op in ("Drilling", "Boring", "Reaming")}

SLENDER_OPS = ("Plain/Cylindrical Turning", "Step Turning", "Taper Turning", "Contour Turning", "Eccentric Turning", "Knurling",
               "External Threading", "Grooving/Necking", "Undercutting", "Polishing/Burnishing")     # operations along the workpiece length
HOLE_DEPENDENT = ("Reaming", "Tapping", "Thread Milling", "Counterboring", "Countersinking", "Internal Grooving")
HOLE_MAKERS = ("Drilling", "Boring")
RA_PROCESS_FLOOR_KEY = {"turning": "turning", "bore": "boring", "mill": "mill", "drill": "drill", "ream": "ream"}

NOT_ENCODED = {
    "minimum chuck grip length": "no source found",
    "boring-bar overhang (L/D) limits": "no source found (the pages found carry no numbers)",
    "thin walls / thin webs": "needs wall thickness, which the dataset does not have; no source",
    "end-mill length-to-diameter limits": "no source found",
    "operations not allowed for a material (e.g. burnishing cast iron)": "no source found",
    "Ra floor of a process": "web tables give ranges too wide to fix a floor; the floors of physics.RA_FLOOR_UM stay assumptions",
}


class NotEvaluable(Exception):
    """A rule needs data the part does not have."""


# ---- context -------------------------------------------------------------------------------------------------------------
def parse_tolerance_mm(value):
    """'0.05mm' -> 0.05; a number is taken as mm; None when it cannot be read."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    m = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(m.group(1)) if m and float(m.group(1)) > 0 else None


def it_width_um(grade: int, size_mm=None):
    """ISO 286-1 width (um) of IT<grade> at a nominal size. size None = the most favourable class (6-10 mm); below 6 mm the 6-10 class
    is used; above 250 mm None (not loaded)."""
    if size_mm is None or size_mm < 6:
        return ISO286_UM[0][2][grade]
    for lo, hi, row in ISO286_UM:
        if lo <= size_mm <= hi:
            return row[grade]
    return None


class Ctx:
    def __init__(self, steps, part, machine):
        self.steps = [s for s in steps if s != V.MARKER]
        self.part = part or {}
        self.machine = machine
        self.dims = self.part.get("dims") or {}
        self.features = list(self.part.get("features") or [])
        self.notes = []
        self._phys = None

    def need(self, what: str):
        """The value of 'dims.diameter', 'tolerance', 'required_ra_um', 'material', ...; NotEvaluable when it is missing."""
        if what.startswith("dims."):
            v = self.dims.get(what[5:])
        elif what == "tolerance":
            v = parse_tolerance_mm(self.part.get("tolerance"))
        else:
            v = self.part.get(what)
        if v is None or v == "" or (isinstance(v, (int, float)) and not isinstance(v, bool) and v <= 0):
            raise NotEvaluable(f"{what} missing")
        return v

    def positions(self, ops):
        s = set(ops)
        return tuple(i for i, o in enumerate(self.steps) if o in s)

    def first(self, op):
        return self.steps.index(op) if op in self.steps else None

    def physics(self):
        if self._phys is None:
            self.need("dims.diameter")
            mat = self.need("material")
            try:
                self._phys = physics.analyse_route(self.steps, mat, self.dims, self.part.get("batch_size", 1) or 1,
                                                   self.part.get("required_ra_um"), machine_power_kw=self.machine.spindle_power_kw)
            except KeyError as exc:
                raise NotEvaluable(str(exc))
        return self._phys


def viol(kind, rid, message, ops=(), positions=(), feature="", action=""):
    return V.Violation(kind, message, ops=tuple(ops), feature=feature, rule_id=f"PHYS:{rid}", positions=tuple(positions), action=action)


# ---- rule registry -------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    id: str
    category: str          # capability | surface | workholding | hole | finishing | sequence
    kind: str              # infeasible | risk
    scope: str             # route | part
    needs: tuple           # what the rule needs besides the route
    summary: str
    action: str
    source: str
    source_status: str     # web | derived | model | assumption
    fn: object = field(compare=False, repr=False, default=None)


RULES = []


def rule(id, category, kind, scope, needs, summary, action, source, source_status):
    def deco(fn):
        RULES.append(Rule(id, category, kind, scope, tuple(needs), summary, action, source, source_status, fn))
        return fn
    return deco


# ---- machine capability ----------------------------------------------------------------------------------------------------
@rule("CAP-LATHE-DIA", "capability", "infeasible", "route", ("dims.diameter",),
      "the part diameter must not exceed the lathe's turning capacity", "use a larger lathe",
      "machine envelope: ASSUMED (Machine defaults); replace with the department's data sheets", "assumption")
def cap_lathe_dia(c):
    ops = [o for o in c.steps if o in rb.LATHE_ONLY_OPS]
    if not ops:
        return []
    d = c.need("dims.diameter")
    if d <= c.machine.lathe_max_diameter_mm:
        return []
    return [viol("infeasible", "CAP-LATHE-DIA", f"part diameter {d:g} mm exceeds the lathe capacity of {c.machine.lathe_max_diameter_mm:g} mm",
                 dict.fromkeys(ops), c.positions(ops), action="machine:larger lathe")]


@rule("CAP-LATHE-LEN", "capability", "infeasible", "route", ("dims.length",),
      "the turned length must not exceed the lathe's maximum turning length", "use a longer lathe",
      "machine envelope: ASSUMED (Machine defaults); replace with the department's data sheets", "assumption")
def cap_lathe_len(c):
    ops = [o for o in c.steps if o in SLENDER_OPS]
    if not ops:
        return []
    n = c.need("dims.length")
    if n <= c.machine.lathe_max_length_mm:
        return []
    return [viol("infeasible", "CAP-LATHE-LEN", f"turned length {n:g} mm exceeds the lathe's {c.machine.lathe_max_length_mm:g} mm",
                 dict.fromkeys(ops), c.positions(ops), action="machine:longer lathe")]


@rule("CAP-MILL-ENV", "capability", "infeasible", "route", ("dims.length", "dims.depth"),
      "the milled length, width and depth must fit the machine's X, Y, Z travel", "use a larger milling machine",
      "machine envelope: ASSUMED (Machine defaults); replace with the department's data sheets", "assumption")
def cap_mill_env(c):
    ops = [o for o in c.steps if o in rb.MILLING_ONLY_OPS]
    if not ops:
        return []
    size = sorted((float(c.need("dims.length")), float(physics._dims(c.dims)["width"]), float(c.need("dims.depth"))), reverse=True)
    travel = sorted(c.machine.mill_travel_mm, reverse=True)
    over = [(s, t) for s, t in zip(size, travel) if s > t]
    if not over:
        return []
    return [viol("infeasible", "CAP-MILL-ENV",
                 f"work envelope {size[0]:g} x {size[1]:g} x {size[2]:g} mm does not fit the travel {travel[0]:g} x {travel[1]:g} x {travel[2]:g} mm",
                 dict.fromkeys(ops), c.positions(ops), action="machine:larger milling machine")]


@rule("CAP-POWER", "capability", "risk", "route", ("dims.diameter", "material"),
      "no operation may need more spindle power than the machine has at the standard speed and feed", "reduce:feed or depth, or a larger machine",
      "cutting-force model of capp/physics.py (Kienzle table of capp/materials.py); machine power ASSUMED", "model")
def cap_power(c):
    ph = c.physics()
    has_hole = c.dims.get("hole_diameter") is not None
    out, skipped = [], set()
    for i, row in enumerate(ph["ops"]):
        if row["utilisation"] is None or row["utilisation"] <= 1.0:
            continue
        if row["op"] in physics.HOLE_DIAMETER_OPS and not has_hole:
            skipped.add(row["op"])                      # the Time agent's hole diameter is the part diameter: not a real hole
            continue
        out.append(viol("risk", "CAP-POWER", f"{row['op']} needs {row['spindle_kw']:.1f} kW at the spindle, the machine has "
                        f"{c.machine.spindle_power_kw:g} kW", (row["op"],), (i,), action="reduce:feed or depth"))
    if skipped:
        c.notes.append(("CAP-POWER", "hole operations not checked (no hole_diameter): " + ", ".join(sorted(skipped))))
    return out


# ---- surface finish -------------------------------------------------------------------------------------------------------
def _floors(ph):
    """Smallest Ra floor among the cutting operations of the route, or None. Facing is left out when the route has any other operation
    with a Ra model: every route starts with it, so it would make every finish look reachable (the critical surface is rarely the end face)."""
    rows = [r for r in ph["ops"] if r["ra_um"] is not None and r["process"] in RA_PROCESS_FLOOR_KEY]
    others = [r for r in rows if r["op"] != "Facing"]
    rows = others or rows
    return min((physics.RA_FLOOR_UM[RA_PROCESS_FLOOR_KEY[r["process"]]] for r in rows), default=None)


@rule("RA-FEED", "surface", "risk", "route", ("required_ra_um", "dims.diameter", "material"),
      "the finest Ra the route's processes give at the standard feed must not exceed the required Ra (theoretical Ra = f^2 / (32 r))",
      "reduce:feed to the value in the message",
      "textbook formula for the ideal roughness of turning/milling (handbook page to be added), feed and nose radius of capp/physics.py", "model")
def ra_feed(c):
    req = c.need("required_ra_um")
    ph = c.physics()
    ra = ph["totals"]["ra"]
    floor = _floors(ph)
    if ra["best_um"] is None or floor is None or req < floor or ra["best_um"] <= req:
        return []
    i = c.first(ra["best_op"])
    row = ph["ops"][i]
    r = {"turning": physics.NOSE_RADIUS_MM["turning"], "bore": physics.NOSE_RADIUS_MM["boring"]}.get(row["process"])
    if r is not None:
        fmax = physics.max_feed_for_ra(req, r)
        hint, action = f"; a feed of at most {fmax:.3f} mm/rev gives a low Ra risk", f"reduce:feed to {fmax:.3f} mm/rev"
    else:
        hint, action = "", "reduce:feed"
    return [viol("risk", "RA-FEED", f"predicted Ra {ra['best_um']:.2g} um ({ra['best_op']}) exceeds the required {req:g} um at the standard feed{hint}",
                 (ra["best_op"],), (i,), action=action)]


@rule("RA-FLOOR", "surface", "risk", "route", ("required_ra_um", "dims.diameter", "material"),
      "the required Ra must not be below what the finest cutting process of the route can reach at all", "add a finishing process (grinding, honing, burnishing)",
      "Ra floors of capp/physics.py: ASSUMED (web ranges are too wide to fix a floor)", "assumption")
def ra_floor(c):
    req = c.need("required_ra_um")
    ph = c.physics()
    floor = _floors(ph)
    if floor is None or req >= floor or any(o == "Polishing/Burnishing" for o in c.steps):
        return []
    return [viol("risk", "RA-FLOOR", f"required Ra {req:g} um is below the best the route's cutting processes reach ({floor:g} um)",
                 (), (), action="finish:add a finishing process")]


# ---- workholding and support -----------------------------------------------------------------------------------------------
_CADEM = "CADEM, 'CNC lathe work holding: thumb rules' (L/D < 3 chuck only; 3-6 tailstock; 6-12 steady rest; to 20 with reduced parameters) https://cadem.com/cnc-turning-work-holding-thumb-rules/"


def _slenderness(c):
    ops = [o for o in c.steps if o in SLENDER_OPS]
    if not ops:
        return None, ops
    return c.need("dims.length") / c.need("dims.diameter"), ops


@rule("HOLD-TAILSTOCK", "workholding", "risk", "route", ("dims.length", "dims.diameter"),
      "turning with a length-to-diameter ratio from 3 to 6 needs tailstock support", "support:tailstock", _CADEM, "web")
def hold_tailstock(c):
    ld, ops = _slenderness(c)
    if ld is None or not (CHUCK_ONLY_LD <= ld < TAILSTOCK_MAX_LD):
        return []
    return [viol("risk", "HOLD-TAILSTOCK", f"L/D = {ld:.1f}: chuck alone is not enough, use the tailstock", dict.fromkeys(ops), c.positions(ops),
                 action="support:tailstock")]


@rule("HOLD-STEADY", "workholding", "risk", "route", ("dims.length", "dims.diameter"),
      "turning with a length-to-diameter ratio from 6 to 20 needs a steady rest (reduced cutting parameters above 12)", "support:steady rest", _CADEM, "web")
def hold_steady(c):
    ld, ops = _slenderness(c)
    if ld is None or not (TAILSTOCK_MAX_LD <= ld <= SOURCE_MAX_LD):
        return []
    extra = "; cutting parameters must be reduced" if ld > REDUCED_PARAMS_LD else ""
    return [viol("risk", "HOLD-STEADY", f"L/D = {ld:.1f}: use a steady rest{extra}", dict.fromkeys(ops), c.positions(ops), action="support:steady rest")]


@rule("HOLD-SLENDER-MAX", "workholding", "risk", "route", ("dims.length", "dims.diameter"),
      "turning with a length-to-diameter ratio above 20 is outside the range of the thumb rule", "review:slender part", _CADEM, "web")
def hold_slender_max(c):
    ld, ops = _slenderness(c)
    if ld is None or ld <= SOURCE_MAX_LD:
        return []
    return [viol("risk", "HOLD-SLENDER-MAX", f"L/D = {ld:.1f} is beyond the L/D 20 that the thumb rule covers", dict.fromkeys(ops), c.positions(ops),
                 action="review:slender part")]


# ---- hole making -----------------------------------------------------------------------------------------------------------
@rule("HOLE-DEEP", "hole", "risk", "route", ("dims.hole_diameter", "dims.depth"),
      "a hole deeper than 3 x D needs peck drilling, beyond 5 x D the multi-peck cycle", "drill:peck",
      "Mt Hood CC, CNC support course, 'Deep holes and G83 multi-peck drill cycle' (deep hole = 3 x D; G83 beyond 5 x D) https://mhcc.pressbooks.pub/supportcnc/?p=502", "web")
def hole_deep(c):
    if "Drilling" not in c.steps:
        return []
    d, depth = c.need("dims.hole_diameter"), c.need("dims.depth")
    ratio = depth / d
    if ratio <= DEEP_HOLE_LD:
        return []
    cycle = "multi-peck cycle (G83)" if ratio > MULTI_PECK_LD else "peck drilling"
    return [viol("risk", "HOLE-DEEP", f"hole depth/diameter = {ratio:.1f}: use {cycle}", ("Drilling",), c.positions(("Drilling",)),
                 action=f"drill:{'multi-peck' if ratio > MULTI_PECK_LD else 'peck'}")]


@rule("HOLE-EXISTS", "hole", "infeasible", "route", (),
      "an operation that works on an existing hole (reaming, tapping, thread milling, counterboring, countersinking, internal grooving) needs a Drilling or Boring before it",
      "insert:Drilling", "derived: v1 feature chains (feature_vocab) and VOCAB_CONSISTENT_EDGES of precedence_graph", "derived")
def hole_exists(c):
    out = []
    for op in HOLE_DEPENDENT:
        i = c.first(op)
        if i is None:
            continue
        if not any(m in c.steps[:i] for m in HOLE_MAKERS):
            out.append(viol("infeasible", "HOLE-EXISTS", f"'{op}' has no hole to work on: no Drilling or Boring before it", (op,), (i,),
                            action="insert:Drilling"))
    return out


@rule("HOLE-PILOT", "hole", "risk", "route", (),
      "a drilled hole is started with a centre drill so that it is located precisely", "insert:Center Drilling",
      "CIMCO, CNC programming handbook ch. 7 'Drilling': 'center drill all holes to ensure they are located precisely' https://www.cimco.com/docs/cncbook/en/Ch07_Drilling.html", "web")
def hole_pilot(c):
    i = c.first("Drilling")
    if i is None or "Center Drilling" in c.steps[:i]:
        return []
    return [viol("risk", "HOLE-PILOT", "Drilling without a Center Drilling before it", ("Drilling",), (i,), action="insert:Center Drilling")]


# ---- tolerance and finishing -------------------------------------------------------------------------------------------------
_IT = ("ISO 286-1 IT values (Wikipedia 'IT Grade', Table 1) https://en.wikipedia.org/wiki/IT_Grade; IT grade by process (Engineering ToolBox, "
       "citing ANSI B4.1) https://www.engineeringtoolbox.com/machine-processes-tolerance-grades-d_1367.html; the tolerance is read as +-T")


@rule("TOL-HOLE", "finishing", "risk", "route", ("tolerance",),
      "a hole tolerance finer than what drilling (IT10) or boring (IT8) holds needs a Reaming operation (IT6)", "insert:Reaming", _IT, "web")
def tol_hole(c):
    if not any(f in HOLE_FEATURES for f in c.features):
        return []
    makers = [o for o in c.steps if o in HOLE_IT_OPS]
    if not makers:
        return []                                            # no hole-making in the route: coverage (COV) reports that
    band = c.need("tolerance") * TOLERANCE_BAND_FACTOR * 1000.0
    size = c.dims.get("hole_diameter")
    best_op = min(makers, key=lambda o: HOLE_IT_OPS[o])
    w_best, w_ream = it_width_um(HOLE_IT_OPS[best_op], size), it_width_um(PROCESS_IT["Reaming"], size)
    if w_best is None or w_ream is None:
        raise NotEvaluable("hole diameter beyond 250 mm: IT values not loaded")
    if w_best <= band or w_ream > band:                     # held already, or beyond reaming too (then TOL-SPEC reports it)
        return []
    where = f"{size:g} mm" if size else "the most favourable size class (6-10 mm)"
    return [viol("risk", "TOL-HOLE", f"hole tolerance +-{c.need('tolerance'):g} mm ({band:g} um band) is finer than {best_op} holds "
                 f"(IT{HOLE_IT_OPS[best_op]}: {w_best} um at {where}); a Reaming operation (IT6: {w_ream} um) is needed", (best_op,),
                 c.positions((best_op,)), action="insert:Reaming")]


@rule("TOL-SPEC", "finishing", "risk", "part", ("tolerance", "dims.diameter"),
      "the tolerance must be reachable by some operation of the vocabulary (turned diameter: IT7, hole: IT6); grinding, honing and lapping are not in it",
      "outside the vocabulary: grinding / honing / lapping", _IT, "web")
def tol_spec(c):
    turned = any(f in TURNED_FEATURES for f in c.features)
    holes = any(f in HOLE_FEATURES for f in c.features)
    if not (turned or holes):
        return []
    tol = c.need("tolerance")
    band = tol * TOLERANCE_BAND_FACTOR * 1000.0
    out = []
    if turned:
        d = c.need("dims.diameter")
        w = it_width_um(PROCESS_IT["Turning"], d)
        if w is None:
            raise NotEvaluable("diameter beyond 250 mm: IT values not loaded")
        if w > band:
            out.append(viol("risk", "TOL-SPEC", f"turned-diameter tolerance +-{tol:g} mm ({band:g} um band) is finer than turning holds "
                            f"(IT{PROCESS_IT['Turning']}: {w} um at {d:g} mm); only grinding, honing or lapping reach it", action="outside the vocabulary"))
    if holes:
        size = c.dims.get("hole_diameter")
        w = it_width_um(PROCESS_IT["Reaming"], size)
        if w is None:
            raise NotEvaluable("hole diameter beyond 250 mm: IT values not loaded")
        if w > band:
            out.append(viol("risk", "TOL-SPEC", f"hole tolerance +-{tol:g} mm ({band:g} um band) is finer than reaming holds "
                            f"(IT6: {w} um at {size:g} mm)" if size else
                            f"hole tolerance +-{tol:g} mm ({band:g} um band) is finer than reaming holds in the most favourable size class "
                            f"(IT6: {w} um at 6-10 mm)", action="outside the vocabulary"))
    return out


# ---- sequence ---------------------------------------------------------------------------------------------------------------
@rule("SEQ-PARTOFF", "sequence", "infeasible", "route", (),
      "no lathe operation can follow Parting-off: the part is no longer held", "reorder:Parting-off after the lathe operations",
      "derived: rationale of NEW_FEATURE_EDGES in precedence_graph (extends it to every lathe-only operation)", "derived")
def seq_partoff(c):
    i = c.first("Parting-off")
    if i is None:
        return []
    return [viol("infeasible", "SEQ-PARTOFF", f"'{op}' is a lathe operation after Parting-off", (op,), (j,), action="reorder:before Parting-off")
            for j, op in enumerate(c.steps) if j > i and op in rb.LATHE_ONLY_OPS and op != "Parting-off"]


RULE_BY_ID = {r.id: r for r in RULES}
assert len(RULE_BY_ID) == len(RULES)


# ---- evaluation ----------------------------------------------------------------------------------------------------------------
@dataclass
class Report:
    violations: list = field(default_factory=list)          # verify.Violation, rule ids "PHYS:<RULE>"
    evaluated: list = field(default_factory=list)           # rule ids that could be evaluated
    not_evaluated: dict = field(default_factory=dict)       # rule id -> reason
    notes: list = field(default_factory=list)               # [(rule id, note)]: partial evaluation

    @property
    def infeasible(self):
        return [v for v in self.violations if v.kind == "infeasible"]

    @property
    def risks(self):
        return [v for v in self.violations if v.kind == "risk"]

    @property
    def route_violations(self):
        return [v for v in self.violations if RULE_BY_ID[v.rule_id[5:]].scope == "route"]

    @property
    def part_violations(self):
        return [v for v in self.violations if RULE_BY_ID[v.rule_id[5:]].scope == "part"]

    @property
    def rule_ids(self):
        return sorted({v.rule_id[5:] for v in self.violations})

    def to_dict(self) -> dict:
        return {"violations": [{"rule_id": v.rule_id[5:], "kind": v.kind, "scope": RULE_BY_ID[v.rule_id[5:]].scope, "message": v.message,
                                "ops": list(v.ops), "positions": list(v.positions), "action": v.action} for v in self.violations],
                "n_infeasible": len(self.infeasible), "n_risk": len(self.risks),
                "route_infeasible": sum(1 for v in self.route_violations if v.kind == "infeasible"),
                "route_risk": sum(1 for v in self.route_violations if v.kind == "risk"),
                "part_risk": len(self.part_violations),
                "evaluated": len(self.evaluated), "not_evaluated": dict(self.not_evaluated), "notes": [list(n) for n in self.notes]}


def evaluate(steps: list, part: dict, machine: Machine = DEFAULT_MACHINE, rules=None) -> Report:
    """Run the rules on a route (changeover markers allowed) for a part. Deterministic; no API call."""
    ctx = Ctx(steps, part, machine)
    rep = Report()
    for r in (RULES if rules is None else rules):
        try:
            found = r.fn(ctx)
        except NotEvaluable as exc:
            rep.not_evaluated[r.id] = str(exc)
            continue
        rep.evaluated.append(r.id)
        rep.violations.extend(found)
    rep.notes = list(ctx.notes)
    return rep


def condense(rep: Report) -> dict:
    """What the harness stores with a route."""
    return rep.to_dict()


def rule_table() -> list:
    return [{"id": r.id, "category": r.category, "kind": r.kind, "scope": r.scope, "needs": list(r.needs), "summary": r.summary,
             "action": r.action, "source": r.source, "source_status": r.source_status} for r in RULES]


# ---- optional hook into the strict verifier ----------------------------------------------------------------------------------------
def verify_rule(steps, part):
    """verify.PHYSICAL_RULES entry: the rules for a given part; nothing without a part."""
    return evaluate(steps, part).violations if part else []


def install():
    """Make verify.check_route(route, features, part) report these rules. 'infeasible' ones make the strict check fail, 'risk' ones do not."""
    if verify_rule not in V.PHYSICAL_RULES:
        V.PHYSICAL_RULES.append(verify_rule)


def uninstall():
    if verify_rule in V.PHYSICAL_RULES:
        V.PHYSICAL_RULES.remove(verify_rule)
