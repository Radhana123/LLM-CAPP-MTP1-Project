"""Feasibility-aware R1 (row A7 of the ablation grid): the localized repair of capp/repair.py, extended with the actions of the
physical-feasibility rules of capp/feasibility.py. Used by the systems whose name ends in +fz1 or +fz2.

R1 repairs what the strict verifier calls invalid (order, missing, invented, duplicate). A route can pass all of that and still be
physically impossible, e.g. an Internal Grooving planned before any Drilling (HOLE-EXISTS) or a lathe operation after Parting-off
(SEQ-PARTOFF). The feasibility rules report such findings in the same Violation format, with an `action` that says what to do.
This module reads those actions:

    insert:<Operation>                  add the operation (must be one of the 41 canonical operations) to the route
    HOLE-EXISTS (action insert:Drilling) a hole-making operation (Drilling, else Boring) must come before the dependent operation:
                                        an ordering constraint on top of the precedence graph, plus the insertion if the route has none
    SEQ-PARTOFF (reorder:before ...)    every lathe operation must come before Parting-off: an ordering constraint
    machine:...                         machine capability (diameter, length, travel): no edit of the route can fix it, it is
                                        reported (`unrepairable`) and does not stop the repair
    everything else (support:, reduce:, drill:, finish:, review:, outside the vocabulary)   a parameter, a support or a different
                                        process: it cannot be expressed as a route edit here, so it is flagged only (`flagged`)

Two levels (the +fz option):
    fz1   act on the HARD feasibility violations only (kind "infeasible")
    fz2   fz1, and also on the RISK violations whose action is an insertion: HOLE-PILOT (insert Center Drilling) and TOL-HOLE
          (insert Reaming). A risk never makes a route invalid, so fz2 is the "improve" level, fz1 the "make feasible" level.

The loop: set repair + order repair (exactly R1), evaluate the rules on the result, turn the actionable findings into insertions and
ordering constraints, run the order repair again with the constraints; stop when nothing is actionable. Inserted operations and
constraints only accumulate, so it ends; a cap (MAX_ROUNDS) guards the rest. If no valid order exists the repair reports ok=False
and the caller falls back to R0 like for R1. A route without actionable findings is returned exactly as R1 returns it.

Deterministic, no API call. The feasibility check needs the part (dimensions only for the rules that use them; the hard rules
used here need none except the machine capability ones).

    rep = feasrepair.r1_fz(llm_steps, features, part, risks=False)
    rep.ok, rep.steps, rep.route, rep.inserted (every inserted operation), rep.fz_inserted (those inserted for feasibility),
    rep.fz_edges, rep.fz_rounds, rep.unrepairable, rep.flagged
"""
from dataclasses import dataclass, field

from capp import feasibility as F
from capp import legacy  # noqa: F401
from capp import repair as R
from capp import verify as V

MAX_ROUNDS = 6
LEVELS = (0, 1, 2)


@dataclass
class FeasRepair(R.Repair):
    strategy: str = "r1_fz"
    fz_inserted: list = field(default_factory=list)        # operations added because a feasibility rule asked for them
    fz_edges: list = field(default_factory=list)           # [a, b] ordering constraints added ("a before b")
    fz_rounds: int = 0                                     # extra order-repair rounds that were needed
    unrepairable: list = field(default_factory=list)       # rule ids of hard findings no route edit can fix (machine capability)
    flagged: list = field(default_factory=list)            # rule ids of findings that remain and are only flagged (supports, parameters ...)
    remaining: list = field(default_factory=list)          # rule ids still violated in the repaired route (hard or acted-on risks), "" if none


def is_machine(v) -> bool:
    return v.action.startswith("machine:")


def repairable_hard(violations) -> list:
    """The hard feasibility findings a route edit can fix (everything hard except machine capability)."""
    return [v for v in violations if v.kind == "infeasible" and not is_machine(v)]


def insertable_risks(violations) -> list:
    """The risk findings whose action is an insertion of a canonical operation."""
    return [v for v in violations if v.kind == "risk" and _insert_op(v) is not None]


def _insert_op(v):
    if v.action.startswith("insert:"):
        op = v.action[len("insert:"):]
        if op in V.CANONICAL_OPS:
            return op
    return None


def _hole_maker(ops):
    for m in F.HOLE_MAKERS:                                 # Drilling, then Boring
        if m in ops:
            return m
    return None


def plan_actions(violations, ops, risks: bool):
    """(insertions, edges, unrepairable, flagged) for the current route. `ops` is the operation set of the route."""
    inserts, edges, unrepairable, flagged = [], [], [], []
    for v in violations:
        rid = v.rule_id[5:] if v.rule_id.startswith("PHYS:") else v.rule_id
        if v.kind == "infeasible":
            if is_machine(v):
                unrepairable.append(rid)
            elif rid == "HOLE-EXISTS" and v.ops:
                maker = _hole_maker(ops)
                if maker is None:
                    inserts.append("Drilling")
                    maker = "Drilling"
                edges.append((maker, v.ops[0]))
            elif rid == "SEQ-PARTOFF" and v.ops:
                edges.append((v.ops[0], "Parting-off"))
            else:
                op = _insert_op(v)
                if op:
                    inserts.append(op)
                else:
                    flagged.append(rid)
        elif v.kind == "risk":
            op = _insert_op(v)
            if risks and op:
                inserts.append(op)
            else:
                flagged.append(rid)
    return inserts, edges, unrepairable, flagged


def r1_fz(steps: list, features: list, part: dict, risks: bool = False, machine: F.Machine = F.DEFAULT_MACHINE) -> FeasRepair:
    """R1 plus the feasibility actions. `risks=True` is level fz2 (also the insertion actions of risk rules)."""
    ops, unknown, dups, inserted = R.set_repair(steps, features)
    rep = FeasRepair(removed_unknown=unknown, removed_duplicates=dups, inserted=list(inserted))
    llm_pos = {}
    for op in V.clean_steps(steps):
        if op in V.CANONICAL_OPS and op not in llm_pos:
            llm_pos[op] = len(llm_pos)

    ops = list(ops)
    edges = []                                              # accumulated (a, b) constraints
    for rounds in range(MAX_ROUNDS + 1):
        res = R.order_repair(ops, llm_pos, edges)
        if res is None:
            rep.reason = "no valid ordering found (precedence conflict, feasibility constraint or search too large)"
            return rep
        order, disc, changeovers, states = res
        found = F.evaluate(order, part, machine).violations
        inserts, new_edges, unrepairable, flagged = plan_actions(found, ops, risks)
        fresh_ops = [op for op in dict.fromkeys(inserts) if op not in ops]
        fresh_edges = [e for e in dict.fromkeys(new_edges) if e not in edges]
        rep.unrepairable, rep.flagged = sorted(set(unrepairable)), sorted(set(flagged))
        if not fresh_ops and not fresh_edges:
            break
        if rounds == MAX_ROUNDS:
            rep.reason = "feasibility repair did not settle"
            return rep
        ops.extend(fresh_ops)
        edges.extend(fresh_edges)
        rep.fz_inserted.extend(fresh_ops)
        rep.inserted.extend(fresh_ops)
        rep.fz_rounds += 1
    rep.fz_edges = [list(e) for e in edges]
    rep.steps, rep.discordant_pairs, rep.changeovers, rep.states = order, disc, changeovers, states

    chk = V.check_route(order, features)
    if not chk.valid:
        rep.reason = "repaired route is still invalid: " + ", ".join(chk.kinds)
        return rep
    left = F.evaluate(order, part, machine).violations
    still = repairable_hard(left) + (insertable_risks(left) if risks else [])
    rep.remaining = sorted({v.rule_id[5:] for v in still})
    if repairable_hard(left):
        rep.reason = "a hard feasibility violation is left: " + ", ".join(rep.remaining)
        return rep
    rep.ok = True
    rep.route = V.normalize_markers(order)
    return rep
