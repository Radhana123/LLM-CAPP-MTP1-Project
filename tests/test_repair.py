import itertools
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import archetypes as A      # noqa: E402
from capp import metrics as M         # noqa: E402
from capp import repair as R          # noqa: E402
from capp import verify as V          # noqa: E402
import feature_vocab as fv            # noqa: E402
import precedence_graph as pg         # noqa: E402
import route_builder as rb            # noqa: E402

MK = V.MARKER


def test_valid_route_is_left_alone():
    route = ["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"]
    rep = R.r1(route, ["Reamed_Hole"])
    assert rep.ok and rep.steps == route and rep.discordant_pairs == 0
    assert rep.inserted == rep.removed_unknown == rep.removed_duplicates == []


def test_set_repair_removes_unknown_and_duplicates_and_inserts_missing():
    route = ["Facing", "Center Drilling", "Deburring", "Chamfering", "Chamfering"]
    rep = R.r1(route, ["Hole", "Chamfer"])
    assert rep.ok
    assert rep.removed_unknown == ["Deburring"] and rep.removed_duplicates == ["Chamfering"]
    assert set(rep.inserted) == {"Drilling", "Inspection"}
    assert rep.steps.count("Chamfering") == 1
    assert V.check_route(rep.steps, ["Hole", "Chamfer"]).valid


def test_order_repair_reverses_only_what_is_needed():
    # Reaming placed before Drilling; everything else is fine. Minimal fix: move Reaming after Drilling.
    route = ["Facing", "Center Drilling", "Reaming", "Drilling", "Inspection"]
    rep = R.r1(route, ["Reamed_Hole"])
    assert rep.ok and rep.steps == ["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"]
    assert rep.discordant_pairs == 1


def test_facing_and_inspection_are_forced_to_the_ends():
    rep = R.r1(["Drilling", "Center Drilling", "Facing", "Inspection", "Reaming"], ["Reamed_Hole"])
    assert rep.ok and rep.steps[0] == "Facing" and rep.steps[-1] == "Inspection"


def test_valid_operations_nobody_asked_for_are_kept():
    route = ["Facing", "Center Drilling", "Drilling", "Chamfering", "Inspection"]
    rep = R.r1(route, ["Hole"])
    assert rep.ok and "Chamfering" in rep.steps and rep.inserted == []


def test_empty_and_garbage_routes_are_repaired_to_a_valid_route():
    for route in ([], ["Banana"], [MK, MK], ["Inspection"]):
        rep = R.r1(route, ["Hole", "Slot"])
        assert rep.ok and V.check_route(rep.steps, ["Hole", "Slot"]).valid, route


def test_changeover_marker_in_result_matches_machine_switches():
    rep = R.r1(["Facing", "Slot Milling", "Plain/Cylindrical Turning", "Inspection"], ["Turning", "Slot"])
    assert rep.ok
    assert rep.route.count(MK) == V.implied_changeovers(rep.steps) == rep.changeovers
    assert V.clean_steps(rep.route) == rep.steps


def test_changeovers_break_ties_between_equally_close_orders():
    # Turning and Slot Milling have no precedence edge, so all orders keep the LLM's order of the others equally well;
    # the LLM order lathe, milling, lathe is kept only if it costs no extra discordance, otherwise fewest switches wins.
    route = ["Facing", "Plain/Cylindrical Turning", "Slot Milling", "Grooving/Necking", "Inspection"]
    rep = R.r1(route, ["Turning", "Slot", "Groove"])
    # moving Grooving before Slot costs 1 discordant pair but saves 1 changeover, tau comes first, so the order is kept
    assert rep.steps == route and rep.changeovers == 2 and rep.discordant_pairs == 0


def test_deterministic():
    route = ["Reaming", "Facing", "Slot Milling", "Drilling", "Plain/Cylindrical Turning", "Weird", "Drilling"]
    feats = ["Reamed_Hole", "Slot", "Turning"]
    first = R.r1(route, feats).to_dict()
    for _ in range(5):
        assert R.r1(route, feats).to_dict() == first


def _perturb(rng, steps):
    r = list(steps)
    for _ in range(rng.randint(1, 4)):
        kind = rng.choice(["swap", "drop", "dup", "invent", "reverse"])
        if kind == "swap" and len(r) > 1:
            i, j = rng.sample(range(len(r)), 2)
            r[i], r[j] = r[j], r[i]
        elif kind == "drop" and len(r) > 2:
            r.pop(rng.randrange(len(r)))
        elif kind == "dup" and r:
            r.insert(rng.randrange(len(r) + 1), rng.choice(r))
        elif kind == "invent":
            r.insert(rng.randrange(len(r) + 1), rng.choice(["Deburring", "Heat Treatment", "Polishing Deluxe"]))
        elif kind == "reverse":
            r.reverse()
    return r


def test_r1_always_yields_a_strict_valid_route_for_every_archetype():
    """Including the 8-feature Valve Body, where v1's single-route replacement (R0) can find nothing."""
    rng = random.Random(7)
    for a in A.ARCHETYPES:
        routes = rb.generate_valid_routes(a["features"], max_routes=3)
        for r in routes:
            base = V.clean_steps(r["steps"])
            for _ in range(15):
                bad = _perturb(rng, base)
                rep = R.r1(bad, a["features"])
                assert rep.ok, (a["name"], bad, rep.reason)
                chk = V.check_route(rep.route, a["features"])
                assert chk.valid and chk.v1_valid, (a["name"], bad, chk.violations)
                assert chk.changeovers == rep.changeovers


def _brute_force(ops, llm_pos, features):
    best = None
    for perm in itertools.permutations(ops):
        if any(v.kind == "order" for v in V.check_route(list(perm), features).violations):
            continue
        disc = sum(1 for x, y in itertools.combinations(perm, 2)
                   if x in llm_pos and y in llm_pos and llm_pos[x] > llm_pos[y])
        cost = (disc, V.implied_changeovers(list(perm)))
        if best is None or cost < best:
            best = cost
    return best


def test_order_repair_is_exact_against_brute_force():
    rng = random.Random(3)
    feats_all = fv.get_all_features()
    checked = 0
    while checked < 14:
        feats = rng.sample(feats_all, rng.randint(1, 3))
        routes = rb.generate_valid_routes(feats, max_routes=1)
        if not routes:
            continue
        bad = _perturb(rng, V.clean_steps(routes[0]["steps"]))
        ops, _, _, _ = R.set_repair(bad, feats)
        if len(ops) > 7:
            continue
        llm_pos = {}
        for op in V.clean_steps(bad):
            if op in V.CANONICAL_OPS and op not in llm_pos:
                llm_pos[op] = len(llm_pos)
        res = R.order_repair(ops, llm_pos)
        assert res is not None
        order, disc, co, _ = res
        assert (disc, co) == _brute_force(ops, llm_pos, feats), (bad, feats)
        assert sorted(order) == sorted(ops)
        checked += 1


def test_r1_survival_beats_replacement_on_a_typical_error():
    """One swapped pair: R1 keeps the whole route, R0 replaces it by the Route Builder's route."""
    feats = ["Turning", "Hole", "Slot"]
    base = V.clean_steps(rb.generate_valid_routes(feats, max_routes=1)[0]["steps"])
    bad = list(base)
    bad[2], bad[3] = bad[3], bad[2]
    rep = R.r1(bad, feats)
    assert rep.ok
    assert M.survived_share(bad, rep.steps) >= 0.75
    assert M.edit_distance(bad, rep.steps) <= 2


def test_precedence_conflict_makes_r1_give_up_instead_of_returning_garbage(monkeypatch):
    monkeypatch.setattr(pg, "PRECEDENCE_EDGES", list(pg.PRECEDENCE_EDGES) + [("Reaming", "Drilling")])
    rep = R.r1(["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"], ["Reamed_Hole"])
    assert rep.ok is False and "ordering" in rep.reason


def test_search_size_guard(monkeypatch):
    monkeypatch.setattr(R, "MAX_STATES", 3)
    rep = R.r1(["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"], ["Reamed_Hole"])
    assert rep.ok is False and rep.steps == []


def test_physical_rule_violation_makes_r1_report_failure():
    def rule(steps, part):
        return [V.Violation("infeasible", "no tool", ops=("Drilling",), rule_id="PHYS:test")]
    V.PHYSICAL_RULES.append(rule)
    try:
        rep = R.r1(["Facing", "Center Drilling", "Drilling", "Inspection"], ["Hole"])
        assert rep.ok is False and "infeasible" in rep.reason
    finally:
        V.PHYSICAL_RULES.remove(rule)
