import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import archetypes as A      # noqa: E402
from capp import verify as V          # noqa: E402
import feature_vocab as fv            # noqa: E402
import precedence_graph as pg         # noqa: E402
import route_builder as rb            # noqa: E402

M = V.MARKER


def kinds(route, features):
    return V.check_route(route, features).kinds


def test_valid_route_has_no_violations():
    c = V.check_route(["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"], ["Reamed_Hole"])
    assert c.valid and c.v1_valid and c.violations == [] and c.kinds == []


def test_order_violations_are_structured():
    c = V.check_route(["Facing", "Reaming", "Drilling", "Center Drilling", "Inspection"], ["Reamed_Hole"])
    assert c.kinds == ["order"] and not c.valid and not c.v1_valid
    pairs = {v.ops for v in c.violations}
    assert ("Drilling", "Reaming") in pairs and ("Center Drilling", "Drilling") in pairs


def test_facing_first_and_inspection_last():
    c = V.check_route(["Drilling", "Center Drilling", "Facing"], ["Hole"])
    ops = {v.ops for v in c.violations if v.kind == "order"}
    assert ("Facing",) in ops and ("Inspection",) in ops


def test_missing_feature_reports_feature_and_absent_ops():
    c = V.check_route(["Facing", "Center Drilling", "Inspection"], ["Hole"])
    v = [x for x in c.violations if x.kind == "missing"]
    assert len(v) == 1 and v[0].feature == "Hole" and v[0].ops == ("Drilling",)
    assert not c.v1_valid and not c.valid


def test_invented_and_duplicate_pass_v1_but_fail_strict():
    route = ["Facing", "Center Drilling", "Drilling", "Deburring", "Chamfering", "Chamfering", "Inspection"]
    c = V.check_route(route, ["Hole", "Chamfer"])
    assert c.v1_valid is True            # v1 never looks at unknown operations or repeats
    assert c.valid is False
    assert set(c.kinds) == {"invented", "duplicate"}
    assert [v.ops for v in c.violations if v.kind == "invented"] == [("Deburring",)]
    assert [v.ops for v in c.violations if v.kind == "duplicate"] == [("Chamfering",)]


def test_empty_route():
    c = V.check_route([], ["Hole"])
    assert not c.valid and c.kinds == ["missing"]


def test_markers_are_not_operations():
    route = ["Facing", "Plain/Cylindrical Turning", M, "Slot Milling", "Inspection"]
    c = V.check_route(route, ["Turning", "Slot"])
    assert c.valid and M not in c.steps
    assert V.clean_steps(route) == c.steps


def test_machine_grouping_is_soft():
    # lathe, milling, lathe, milling = 3 switches; the Route Builder needs 1
    route = ["Facing", "Plain/Cylindrical Turning", "Pocket Milling", "Grooving/Necking", "Slot Milling", "Inspection"]
    c = V.check_route(route, ["Turning", "Pocket", "Groove", "Slot"])
    assert "machine_grouping" in c.kinds
    assert c.valid, "grouping is a quality flag, not a validity failure"
    assert c.changeovers == 3


def test_unknown_violation_kind_rejected():
    with pytest.raises(ValueError):
        V.Violation("vibes", "x")


def test_physical_rules_hook_runs_and_is_hard():
    seen = []

    def rule(steps, part):
        seen.append(part)
        return [V.Violation("infeasible", "tool too big", ops=("Drilling",))]

    V.PHYSICAL_RULES.append(rule)
    try:
        c = V.check_route(["Facing", "Inspection"], [], part={"material": "Steel"})
        assert "infeasible" in c.kinds and not c.valid and seen == [{"material": "Steel"}]
    finally:
        V.PHYSICAL_RULES.remove(rule)


def test_v1_valid_equals_v1_functions_on_random_routes():
    """v1_valid must be exactly 'pg.validate_order and rb.is_complete' (the decision v1 takes)."""
    rng = random.Random(0)
    ops = sorted(V.CANONICAL_OPS)
    feats = fv.get_all_features()
    for _ in range(400):
        features = rng.sample(feats, rng.randint(1, 4))
        route = rng.sample(ops, rng.randint(2, 9))
        if rng.random() < 0.5:
            route = ["Facing"] + [o for o in route if o != "Facing"] + ["Inspection"]
        expected = pg.validate_order(route)[0] and rb.is_complete(route, features)
        assert V.check_route(route, features).v1_valid == expected, (route, features)


def test_route_builder_routes_pass_strict_check_for_every_archetype():
    for a in A.ARCHETYPES:
        for r in rb.generate_valid_routes(a["features"], max_routes=15):
            c = V.check_route(r["steps"], a["features"])
            assert c.valid and c.v1_valid, (a["name"], c.violations)
            assert c.changeovers == r["changeovers"]


def test_normalize_markers_inserts_one_marker_per_switch_and_is_idempotent():
    route = ["Facing", "Plain/Cylindrical Turning", "Slot Milling", "Chamfering", "Grooving/Necking", "Inspection"]
    n = V.normalize_markers(route)
    assert n.count(M) == V.implied_changeovers(route) == 2
    assert n == ["Facing", "Plain/Cylindrical Turning", M, "Slot Milling", "Chamfering", M, "Grooving/Necking", "Inspection"]
    assert V.normalize_markers(n) == n
    assert V.clean_steps(n) == route
    # a route that never switches machine gets no marker
    assert M not in V.normalize_markers(["Facing", "Center Drilling", "Drilling", "Inspection"])


def test_normalize_markers_matches_route_builder_changeover_count():
    for a in A.ARCHETYPES:
        for r in rb.generate_valid_routes(a["features"], max_routes=5):
            assert V.normalize_markers(r["steps"]).count(M) == r["changeovers"]


def test_violations_carry_rule_ids_and_positions():
    r = ["Facing", "Reaming", "Drilling", "Deburring", "Chamfering", "Chamfering", "Center Drilling"]
    c = V.check_route(r, ["Reamed_Hole", "Hole"])
    by_rule = {v.rule_id: v for v in c.violations}
    assert by_rule["ORD-LAST"].positions == ()            # Inspection is absent from the route
    e = by_rule["ORD-EDGE:Drilling>Reaming"]
    assert e.positions == (2, 1) and e.ops == ("Drilling", "Reaming")
    assert by_rule["SET-UNKNOWN"].positions == (3,)
    assert by_rule["SET-DUP"].positions == (4, 5)
    assert all(v.rule_id for v in c.violations)


def test_missing_feature_rule_id_names_the_feature():
    c = V.check_route(["Facing", "Inspection"], ["Hole"])
    assert [v.rule_id for v in c.violations if v.kind == "missing"] == ["COV:Hole"]


def test_first_and_last_positions_point_at_the_misplaced_operation():
    c = V.check_route(["Drilling", "Facing", "Inspection", "Center Drilling"], ["Hole"])
    ids = {v.rule_id: v.positions for v in c.violations}
    assert ids["ORD-FIRST"] == (1,) and ids["ORD-LAST"] == (2,)
