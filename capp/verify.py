"""Route verifier with structured violations (error taxonomy of metric M3).

v1 only answers yes/no ("FSM valid and complete"). Here every problem is a Violation object with a kind, the
operations involved, the positions in the route, the identifier of the rule that failed and, for missing coverage, the
feature concerned. The localized repair (capp/repair.py) works from these objects, and M3 is the distribution of their kinds.

Rule ids (stable strings, used in reports and by repair)
  ORD-FIRST / ORD-LAST    Facing first, Inspection last
  ORD-EDGE:<a>><b>        precedence edge "a before b" (week1/precedence_graph.py) broken
  COV:<feature>           the feature is not covered by any of its alternative operation chains
  SET-UNKNOWN             operation outside the 41 canonical operations
  SET-DUP                 operation repeated
  MACH-GROUP              more machine changeovers than needed (soft)
  PHYS:<name>             physical rules added in WP6 (they set rule_id themselves)

Kinds
  order             Facing not first, Inspection not last, or a precedence rule (a must come before b) broken
  missing           a requested feature is not covered by any of its alternative operation chains
  invented          an operation that is not one of the 41 canonical operations
  duplicate         the same operation appears more than once
  machine_grouping  more machine changeovers than the Route Builder needs for these features (SOFT: quality, not validity)
  infeasible        physical infeasibility; rules are added in WP6 (registry PHYSICAL_RULES below), none exist yet

Two validity notions, because they differ:
  v1_valid  what v1 calls valid: precedence/order rules + feature coverage (week1.precedence_graph + route_builder.is_complete)
  valid     strict: no HARD violation. Also rejects invented operations and duplicates, which v1 lets through.
"""
from dataclasses import dataclass, field
from functools import lru_cache

from capp import legacy  # noqa: F401  (puts week1..week5 on sys.path)

import feature_vocab as fv
import precedence_graph as pg
import route_builder as rb
from routes import ALL_OPERATIONS

MARKER = "--- Machine Changeover ---"       # Route Builder inserts this string into steps; it is not an operation
CANONICAL_OPS = frozenset(ALL_OPERATIONS)

KINDS = ("order", "missing", "invented", "duplicate", "machine_grouping", "infeasible")
SOFT_KINDS = ("machine_grouping",)
HARD_KINDS = tuple(k for k in KINDS if k not in SOFT_KINDS)

PHYSICAL_RULES = []     # WP6: callables f(clean_steps, part) -> list[Violation] (kind="infeasible")


@dataclass(frozen=True)
class Violation:
    kind: str
    message: str
    ops: tuple = ()          # order: (a, b) with a required before b, or ("Facing",) / ("Inspection",); missing: ops still absent
    feature: str = ""        # missing: the feature that is not covered
    rule_id: str = ""        # which rule failed (see module docstring)
    positions: tuple = ()    # 0-based indices in the route without markers: order (pos of a, pos of b); invented/duplicate: every occurrence

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown violation kind {self.kind!r}")

    @property
    def hard(self) -> bool:
        return self.kind in HARD_KINDS


@dataclass
class Check:
    steps: list                                   # route without changeover markers
    violations: list = field(default_factory=list)
    v1_valid: bool = False
    changeovers: int = 0                          # machine switches implied by the operations themselves

    @property
    def hard_violations(self):
        return [v for v in self.violations if v.hard]

    @property
    def valid(self) -> bool:
        return not self.hard_violations

    @property
    def kinds(self) -> list:
        return sorted({v.kind for v in self.violations}, key=KINDS.index)


def clean_steps(route: list) -> list:
    return [s for s in route if s != MARKER]


def implied_changeovers(steps: list) -> int:
    """Number of switches between lathe-only and milling-only operations (shared operations are ignored)."""
    seq = []
    for s in steps:
        if s in rb.LATHE_ONLY_OPS:
            seq.append("L")
        elif s in rb.MILLING_ONLY_OPS:
            seq.append("M")
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def normalize_markers(route: list) -> list:
    """Return the route with exactly one Machine-Changeover marker at every lathe<->milling switch.

    The v1 agents charge a changeover (15 min, Rs 1441, 0.30 kWh) only for marker strings that are IN the route. Route Builder
    routes carry markers, LLM routes never do, so v1 does not charge an LLM route for switching machines. Fair scoring (M6)
    therefore normalises every route first. Number of markers == implied_changeovers(route)."""
    out, cur = [], None
    for s in clean_steps(route):
        cls = "L" if s in rb.LATHE_ONLY_OPS else "M" if s in rb.MILLING_ONLY_OPS else None
        if cls:
            if cur is not None and cls != cur:
                out.append(MARKER)
            cur = cls
        out.append(s)
    return out


@lru_cache(maxsize=4096)
def _min_changeovers(features: tuple):
    routes = rb.generate_valid_routes(list(features), max_routes=15)
    return min(r["changeovers"] for r in routes) if routes else None


def check_route(route: list, features: list, part: dict = None) -> Check:
    """Verify a route (markers allowed) for the requested features. `part` is only needed by physical rules (WP6)."""
    steps = clean_steps(route)
    out = Check(steps=steps)
    if not steps:
        out.violations.append(Violation("missing", "empty route"))
        return out

    # ---- order -------------------------------------------------------------------------------------------
    if steps[0] != pg.ALWAYS_FIRST:
        pos = (steps.index(pg.ALWAYS_FIRST),) if pg.ALWAYS_FIRST in steps else ()
        out.violations.append(Violation("order", f"'{pg.ALWAYS_FIRST}' must be the first step, found '{steps[0]}'",
                                        ops=(pg.ALWAYS_FIRST,), rule_id="ORD-FIRST", positions=pos))
    if steps[-1] != pg.ALWAYS_LAST:
        pos = (steps.index(pg.ALWAYS_LAST),) if pg.ALWAYS_LAST in steps else ()
        out.violations.append(Violation("order", f"'{pg.ALWAYS_LAST}' must be the last step, found '{steps[-1]}'",
                                        ops=(pg.ALWAYS_LAST,), rule_id="ORD-LAST", positions=pos))
    for a, b in pg.PRECEDENCE_EDGES:                 # same rule and same index semantics as pg.validate_order
        if a in steps and b in steps and steps.index(a) > steps.index(b):
            out.violations.append(Violation("order", f"'{a}' must come before '{b}'", ops=(a, b),
                                            rule_id=f"ORD-EDGE:{a}>{b}", positions=(steps.index(a), steps.index(b))))

    # ---- missing coverage --------------------------------------------------------------------------------
    present = set(steps)
    for f in features:
        alts = fv.get_operations_for_feature(f)
        if not any(set(alt) <= present for alt in alts):
            best = min(alts, key=lambda alt: len(set(alt) - present))
            absent = tuple(op for op in best if op not in present)
            out.violations.append(Violation("missing", f"feature '{f}' is not covered (absent: {', '.join(absent)})",
                                            ops=absent, feature=f, rule_id=f"COV:{f}"))

    # v1's own notion of validity is exactly "no order violation and no missing coverage"
    out.v1_valid = not any(v.kind in ("order", "missing") for v in out.violations)

    # ---- invented, duplicate -----------------------------------------------------------------------------
    for op in dict.fromkeys(steps):
        if op not in CANONICAL_OPS:
            out.violations.append(Violation("invented", f"'{op}' is not a known operation", ops=(op,), rule_id="SET-UNKNOWN",
                                            positions=tuple(i for i, x in enumerate(steps) if x == op)))
    seen = {}
    for op in steps:
        seen[op] = seen.get(op, 0) + 1
    for op, n in seen.items():
        if n > 1:
            out.violations.append(Violation("duplicate", f"'{op}' appears {n} times", ops=(op,), rule_id="SET-DUP",
                                            positions=tuple(i for i, x in enumerate(steps) if x == op)))

    # ---- machine grouping (soft) -------------------------------------------------------------------------
    out.changeovers = implied_changeovers(steps)
    needed = _min_changeovers(tuple(features))
    if needed is not None and out.changeovers > needed:
        out.violations.append(Violation(
            "machine_grouping",
            f"{out.changeovers} machine changeovers implied, {needed} needed", ops=(), rule_id="MACH-GROUP"))

    # ---- physical feasibility (WP6) ----------------------------------------------------------------------
    for rule in PHYSICAL_RULES:
        out.violations.extend(rule(steps, part))
    return out
