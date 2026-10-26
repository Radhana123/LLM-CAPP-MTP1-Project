# test_planner.py
# pytest unit tests for planner (week2/planner.py, Route Builder based)
# Week 2 | LLM-CAPP Project
# UPDATED (MTP-1 WP0): purane tests "Route_A/B/C" aur select_route() assume karte the,
# jo ab exist nahi karte. Ab planner.plan() Route Builder se steps (list of str) deta hai.

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../week1")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../week4")))

from planner import plan
from fsm_validator import validate_sequence

PARTS = [
    ("Aluminum", ["Hole", "Slot"], "0.02mm", 500),
    ("Steel", ["Thread", "Pocket"], "0.01mm", 50),
    ("Brass", ["Groove"], "0.05mm", 10),
    ("Titanium", ["Thread", "Fillet"], "0.05mm", 100),
]


def _part(material, features, tolerance, batch_size):
    return {"material": material, "features": features, "tolerance": tolerance, "batch_size": batch_size}


@pytest.mark.parametrize("material,features,tolerance,batch", PARTS)
def test_plan_succeeds_and_is_a_step_list(material, features, tolerance, batch):
    result = plan(_part(material, features, tolerance, batch))
    assert result["success"] is True
    steps = result["process_steps"]
    assert isinstance(steps, list) and len(steps) > 0
    assert all(isinstance(s, str) for s in steps)


@pytest.mark.parametrize("material,features,tolerance,batch", PARTS)
def test_plan_is_precedence_valid(material, features, tolerance, batch):
    steps = plan(_part(material, features, tolerance, batch))["process_steps"]
    assert steps[0] == "Facing" and steps[-1] == "Inspection"
    assert validate_sequence(steps)["valid"] is True


def test_alternative_routes_are_step_lists():
    result = plan(_part("Aluminum", ["Hole", "Slot"], "0.02mm", 500))
    for alt in result["alternative_routes"]:
        assert isinstance(alt, list) and all(isinstance(s, str) for s in alt)


def test_invalid_material_fails():
    result = plan(_part("Kryptonite", ["Hole"], "0.02mm", 100))
    assert result["success"] is False
    assert result["errors"]


def test_tokens_in_result():
    result = plan(_part("Aluminum", ["Hole"], "0.02mm", 200))
    assert result["success"] is True
    assert len(result["tokens"]) > 0
