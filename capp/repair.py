"""Localized minimal-edit repair of an LLM route (R1 of Section 6.3).

v1 repairs by throwing the LLM route away and using the Route Builder's first route (R0). R1 keeps what the LLM got right and
changes as little as possible, in two stages:

  set repair    remove unknown operations and repeated operations (the first occurrence stays), insert what is missing:
                the mandatory Facing and Inspection, and for every feature that no alternative chain covers, the chain that
                needs the fewest new operations (ties: the chain that does not bring a new machine type into the route).
                Valid operations that no feature asks for are kept; R1 never deletes a valid operation.
  order repair  among all orderings of the repaired set that satisfy the precedence graph (Facing first, Inspection last,
                every edge of week1/precedence_graph.py), take the one with the fewest discordant pairs against the LLM's
                order (Kendall-tau distance on the operations the LLM placed), then the fewest machine changeovers, and
                finally the canonical operation order, which only makes the result deterministic.

The search is a dynamic programme over the precedence-closed subsets of the operations, so it is exact (tests compare it
with brute force), and it stays small because Facing and Inspection pin both ends of the route. If no valid order exists, or
the search would exceed MAX_STATES, R1 reports ok=False and the caller falls back to R0.

R1 is deterministic, needs no API call, and its output is checked with the strict verifier before it is reported ok.

    rep = repair.r1(llm_steps, features)
    rep.ok, rep.steps, rep.route (with changeover markers), rep.inserted, rep.removed_unknown, rep.discordant_pairs
"""
from dataclasses import asdict, dataclass, field

from capp import legacy  # noqa: F401
from capp import verify as V

import feature_vocab as fv
import precedence_graph as pg
import route_builder as rb
from routes import ALL_OPERATIONS

MAX_STATES = 400_000
_RANK = {op: i for i, op in enumerate(ALL_OPERATIONS)}


@dataclass
class Repair:
    strategy: str = "r1"
    ok: bool = False
    reason: str = ""                 # why R1 gave up ("" when ok)
    steps: list = field(default_factory=list)       # repaired route, no markers
    route: list = field(default_factory=list)       # repaired route with one changeover marker per machine switch
    removed_unknown: list = field(default_factory=list)
    removed_duplicates: list = field(default_factory=list)
    inserted: list = field(default_factory=list)
    discordant_pairs: int = 0        # pairs of kept LLM operations whose order had to be reversed
    changeovers: int = 0
    states: int = 0                  # size of the search

    def to_dict(self):
        return asdict(self)


def _cls(op):
    return "L" if op in rb.LATHE_ONLY_OPS else "M" if op in rb.MILLING_ONLY_OPS else None


def set_repair(steps: list, features: list):
    """Returns (ops, removed_unknown, removed_duplicates, inserted): `ops` is the repaired set in the LLM's order
    (first occurrences), with inserted operations appended."""
    clean = V.clean_steps(steps)
    unknown = [op for op in dict.fromkeys(clean) if op not in V.CANONICAL_OPS]
    seen, ops, dups = set(), [], []
    for op in clean:
        if op not in V.CANONICAL_OPS:
            continue
        if op in seen:
            dups.append(op)
        else:
            seen.add(op)
            ops.append(op)
    inserted = []

    def add(op):
        if op not in seen:
            seen.add(op)
            ops.append(op)
            inserted.append(op)

    add(pg.ALWAYS_FIRST)
    add(pg.ALWAYS_LAST)
    for f in features:
        alts = fv.get_operations_for_feature(f)
        if any(set(alt) <= seen for alt in alts):
            continue
        classes = {_cls(o) for o in seen} - {None}

        def key(ia):
            i, alt = ia
            absent = [o for o in alt if o not in seen]
            new_types = {_cls(o) for o in absent} - {None} - classes
            return (len(absent), len(new_types), i)
        _, best = min(enumerate(alts), key=key)
        for op in best:
            add(op)
    return ops, unknown, dups, inserted


def order_repair(ops: list, llm_pos: dict, extra_edges=()):
    """Best valid ordering of `ops` (see module docstring). `llm_pos` maps an operation to its position in the LLM route
    (operations that were inserted have none). `extra_edges` are additional (a, b) = "a before b" constraints on top of the
    precedence graph; capp/feasrepair.py uses them for the physical-feasibility rules. Returns
    (order, discordant_pairs, changeovers, states) or None."""
    n = len(ops)
    idx = {op: i for i, op in enumerate(ops)}
    pred = [0] * n                                   # bitmask of operations that must come before i
    for a, b in list(pg.PRECEDENCE_EDGES) + list(extra_edges):
        if a in idx and b in idx:
            pred[idx[b]] |= 1 << idx[a]
    first, last = idx[pg.ALWAYS_FIRST], idx[pg.ALWAYS_LAST]
    for i in range(n):
        if i != first:
            pred[i] |= 1 << first
        if i != last:
            pred[last] |= 1 << i
    llm_before = [0] * n                             # operations the LLM placed before op i
    for op, i in idx.items():
        if op in llm_pos:
            for other, j in idx.items():
                if other in llm_pos and llm_pos[other] < llm_pos[op]:
                    llm_before[i] |= 1 << j
    cls = [_cls(op) for op in ops]
    rank = [_RANK.get(op, len(_RANK)) for op in ops]
    full = (1 << n) - 1

    # layers[d]: (set of placed operations, machine type of the last lathe/milling operation) -> (cost, (parent key, op placed))
    layers = [{(0, None): ((0, 0, 0), None)}]
    states = 1
    for depth in range(n):
        nxt = {}
        for (mask, last_cls), (cost, _) in layers[-1].items():
            for i in range(n):
                bit = 1 << i
                if mask & bit or pred[i] & ~mask:
                    continue
                disc = bin(llm_before[i] & ~mask & ~bit).count("1")
                c = cls[i]
                sw = 1 if (c and last_cls and c != last_cls) else 0
                new = (cost[0] + disc, cost[1] + sw, cost[2] + depth * rank[i])
                key = (mask | bit, c or last_cls)
                old = nxt.get(key)
                if old is None or new < old[0]:
                    nxt[key] = (new, ((mask, last_cls), i))
        layers.append(nxt)
        states += len(nxt)
        if not nxt or states > MAX_STATES:
            return None
    end = min(layers[n], key=lambda k: layers[n][k][0])
    cost = layers[n][end][0]
    order, key = [], end
    for depth in range(n, 0, -1):
        _, (prev_key, i) = layers[depth][key]
        order.append(ops[i])
        key = prev_key
    order.reverse()
    return order, cost[0], cost[1], states


def r1(steps: list, features: list) -> Repair:
    """R1: set repair, then order repair. Always returns a Repair; `ok` says whether it produced a strict-valid route."""
    ops, unknown, dups, inserted = set_repair(steps, features)
    rep = Repair(removed_unknown=unknown, removed_duplicates=dups, inserted=inserted)
    llm_pos = {}
    for op in V.clean_steps(steps):
        if op in V.CANONICAL_OPS and op not in llm_pos:
            llm_pos[op] = len(llm_pos)
    res = order_repair(ops, llm_pos)
    if res is None:
        rep.reason = "no valid ordering found (precedence conflict or search too large)"
        return rep
    order, disc, changeovers, states = res
    chk = V.check_route(order, features)
    rep.steps, rep.discordant_pairs, rep.changeovers, rep.states = order, disc, changeovers, states
    if not chk.valid:
        rep.reason = "repaired route is still invalid: " + ", ".join(chk.kinds)
        return rep
    rep.ok = True
    rep.route = V.normalize_markers(order)
    return rep
