# test_week4.py
# pytest unit tests for Week 4 — FSM + NSGA-II
# Week 4 | LLM-CAPP Project

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../week3")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../week2")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../week1")))

import pytest
import llm_planner
from fsm_validator import validate_sequence, fix_sequence, fix_sequence_with_builder
from nsga2 import run_nsga2, dominates, Individual
from route_builder import generate_valid_routes_simple

# UPDATED (MTP-1 WP0): ROUTE_VARIANTS ab exist nahi karta, aur run_nsga2() default me LLM
# (network) call karta hai. Isliye har NSGA-II test me LLM ko fake kiya gaya hai (koi API call nahi).


# ── FSM Tests ─────────────────────────────────────

# Test 1: Valid sequence pass hoti hai
def test_valid_sequence():
    seq = ["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"]
    result = validate_sequence(seq)
    assert result["valid"] == True
    assert len(result["errors"]) == 0


# Test 2: Invalid sequence fail hoti hai
def test_invalid_sequence():
    seq = ["Reaming", "Drilling", "Inspection"]  # Facing missing, wrong order
    result = validate_sequence(seq)
    assert result["valid"] == False
    assert len(result["errors"]) > 0


# Test 3: Sequence Facing se start honi chahiye
def test_must_start_with_facing():
    seq = ["Drilling", "Reaming", "Inspection"]
    result = validate_sequence(seq)
    assert result["valid"] == False


# Test 4: Sequence Inspection pe khatam honi chahiye
def test_must_end_with_inspection():
    seq = ["Facing", "Drilling", "Reaming"]
    result = validate_sequence(seq)
    assert result["valid"] == False


# Test 5: Fix function Facing add karta hai
def test_fix_adds_facing():
    bad_seq = ["Drilling", "Inspection"]
    fixed = fix_sequence(bad_seq)
    assert fixed[0] == "Facing"


# Test 6: Fix function Inspection add karta hai
def test_fix_adds_inspection():
    bad_seq = ["Facing", "Drilling"]
    fixed = fix_sequence(bad_seq)
    assert fixed[-1] == "Inspection"


# Test 7: Fixed sequence valid hoti hai
def test_fixed_sequence_is_valid():
    bad_seq = ["Drilling", "Reaming"]
    fixed = fix_sequence(bad_seq)
    result = validate_sequence(fixed)
    assert result["valid"] == True


# ── NSGA-II Tests (LLM faked, no network) ────────

def _fake_llm(monkeypatch, steps=None, success=True):
    """llm_planner.generate_process_plan ko fake karo."""
    def fake(material, features, tolerance="0.05mm", batch_size=100):
        return {"success": success, "steps": list(steps) if steps else [], "error": "" if success else "fake failure"}
    monkeypatch.setattr(llm_planner, "generate_process_plan", fake)


SHORT = ["Facing", "Drilling", "Inspection"]
LONG = ["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"]


# Test 8: NSGA-II results return karta hai
def test_nsga2_returns_results(monkeypatch):
    _fake_llm(monkeypatch, success=False)
    pareto = run_nsga2("Aluminum", 500, features=["Hole", "Slot"], route_generation_mode="route_builder_first")
    assert len(pareto) > 0


# Test 9: Dominance check sahi kaam karta hai
def test_dominance_check():
    short = Individual("short", SHORT, "Aluminum", 500)
    long_ = Individual("long", LONG, "Aluminum", 500)
    # chhota route time, cost aur energy teeno me behtar hai
    assert dominates(short, long_)
    assert not dominates(long_, short)
    assert not dominates(short, short)


# Test 10: Pareto front mein koi ek dusre ko dominate nahi karta
def test_pareto_front_non_dominated(monkeypatch):
    _fake_llm(monkeypatch, success=False)
    pareto = run_nsga2("Steel", 100, features=["Thread", "Pocket"], route_generation_mode="route_builder_first")
    for i in range(len(pareto)):
        for j in range(len(pareto)):
            if i != j:
                assert not dominates(pareto[i], pareto[j])


# ── llm_first mode (v1 baseline B0 behaviour) ─────
# Ye tests v1 ka CURRENT behaviour pin karte hain (characterization), taaki MTP-1 me
# kuch badle to pata chale.

FEATURES = ["Thread", "Fillet"]


# Test 11: LLM ka valid + complete route as-is population me jata hai
def test_llm_first_keeps_valid_complete_route(monkeypatch):
    valid = generate_valid_routes_simple(FEATURES, max_routes=1)[0]
    _fake_llm(monkeypatch, steps=valid)
    pareto = run_nsga2("Aluminum", 500, features=FEATURES, llm_attempts=2)
    status = run_nsga2.last_llm_status
    assert status["llm_valid"] == 2 and status["llm_corrected"] == 0
    assert [p.route_name for p in pareto] == ["Route_LLM_1"]      # duplicate attempt dedup ho gaya
    assert pareto[0].steps == valid


# Test 12: Invalid LLM route POORA replace hota hai Route Builder ke route se (whole-route replacement)
def test_llm_first_invalid_route_replaced_by_route_builder(monkeypatch):
    _fake_llm(monkeypatch, steps=["Inspection", "Facing"])         # galat order
    pareto = run_nsga2("Aluminum", 500, features=FEATURES, llm_attempts=1)
    status = run_nsga2.last_llm_status
    assert status["llm_valid"] == 0 and status["llm_corrected"] == 1
    assert pareto[0].route_name == "Route_LLM_1_corrected"
    # Naam me "LLM" hai, par steps Route Builder ke hain -> MTP-1 me provenance alag track hoga
    assert pareto[0].steps == fix_sequence_with_builder(FEATURES)


# Test 13: Incomplete (lazy) LLM route reject hota hai, bhale FSM-valid ho
def test_llm_first_incomplete_route_not_accepted_as_is(monkeypatch):
    _fake_llm(monkeypatch, steps=["Facing", "Inspection"])         # FSM-valid par features cover nahi
    run_nsga2("Aluminum", 500, features=FEATURES, llm_attempts=1)
    status = run_nsga2.last_llm_status
    assert status["llm_valid"] == 0 and status["llm_corrected"] == 1


# Test 14: LLM fail ho to Route Builder fallback population deta hai
def test_llm_first_failure_falls_back_to_route_builder(monkeypatch):
    _fake_llm(monkeypatch, success=False)
    pareto = run_nsga2("Aluminum", 500, features=FEATURES, llm_attempts=2)
    status = run_nsga2.last_llm_status
    assert status["route_builder_fallback_used"] is True
    assert all(p.route_name.startswith("Route_Builder_Fallback") for p in pareto)
