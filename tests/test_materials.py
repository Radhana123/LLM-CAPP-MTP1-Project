import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "week1"))
sys.path.insert(0, os.path.join(ROOT, "week3"))

from capp import materials as M          # noqa: E402
from material_tokens import MATERIALS     # noqa: E402
import agents                             # noqa: E402


def test_every_material_has_constants_and_nothing_extra():
    assert set(M.MATERIAL_PROPS) == set(MATERIALS)
    # the v1 agents know the same materials, so the physics agents can swap in without gaps
    assert set(M.MATERIAL_PROPS) == set(agents._CUTTING_PARAMS)


def test_values_are_positive_and_in_sane_ranges():
    for name, p in M.MATERIAL_PROPS.items():
        assert p["kc1_1"] > 0 and p["tool_edge_cost_inr"] > 0 and p["t_ref_min"] > 0, name
        assert 0.1 <= p["mc"] <= 0.35, name
        assert 0.15 <= p["taylor_n"] <= 0.30, name       # carbide range is 0.20 to 0.25 (a little slack)
        assert p["kc_group"], name


def test_specific_cutting_force_ordering():
    kc1 = {m: p["kc1_1"] for m, p in M.MATERIAL_PROPS.items()}
    assert kc1["Plastic"] < kc1["Aluminum"] <= kc1["Brass"] <= kc1["Copper"] < kc1["Cast Iron"]
    assert kc1["Cast Iron"] < kc1["Steel"] < kc1["Titanium"]


def test_kc_follows_kienzle_and_thins_out_with_chip_thickness():
    assert M.kc("Steel", 1.0) == pytest.approx(M.props("Steel")["kc1_1"])
    assert M.kc("Steel", 0.1) > M.kc("Steel", 0.2) > M.kc("Steel", 1.0)
    with pytest.raises(ValueError):
        M.kc("Steel", 0)


def test_unknown_material_raises_instead_of_defaulting():
    with pytest.raises(KeyError):
        M.props("Unobtainium")


def test_taylor_tool_life_matches_reference_and_scales_with_speed():
    for mat, p in M.MATERIAL_PROPS.items():
        vc = agents._CUTTING_PARAMS[mat][0]               # turning speed the Time agent uses
        assert M.tool_life_min(mat, vc, vc) == pytest.approx(p["t_ref_min"])
        # V * T**n = C  ->  doubling the speed divides tool life by 2**(1/n)
        assert M.tool_life_min(mat, 2 * vc, vc) == pytest.approx(p["t_ref_min"] / 2 ** (1 / p["taylor_n"]))
        assert M.taylor_constant(mat, vc) == pytest.approx(vc * p["t_ref_min"] ** p["taylor_n"])


def test_assumptions_are_flagged():
    assert "Plastic" in M.ASSUMED_MATERIALS_KC
    assert "ASSUMED" in M.MATERIAL_PROPS["Plastic"]["kc_group"]
    assert set(M.ASSUMED_FIELDS) == {"t_ref_min", "tool_edge_cost_inr"}
