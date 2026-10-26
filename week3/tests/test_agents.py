# test_agents.py
# pytest unit tests for multi-agent evaluation
# Week 3 | LLM-CAPP Project

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import time_agent, cost_agent, energy_agent, efficiency_agent, evaluate_route
from multi_agent_eval import evaluate_all_routes, find_best_route


# Test 1: Time agent positive value deta hai
def test_time_agent_positive():
    steps = ["Facing", "Drilling", "Inspection"]
    result = time_agent(steps, "Aluminum")
    assert result > 0


# Test 2: Steel zyada time leta hai Aluminum se (hard material)
def test_steel_slower_than_aluminum():
    steps = ["Facing", "Drilling"]
    al_time = time_agent(steps, "Aluminum")
    steel_time = time_agent(steps, "Steel")
    assert steel_time > al_time


# Test 3: Cost agent positive value deta hai
def test_cost_agent_positive():
    steps = ["Facing", "Drilling", "Inspection"]
    result = cost_agent(steps, "Aluminum", 100)
    assert result > 0


# Test 4: Bulk batch discount lagta hai
def test_bulk_discount():
    steps = ["Facing", "Drilling"]
    small_batch_cost = cost_agent(steps, "Aluminum", 10)
    large_batch_cost = cost_agent(steps, "Aluminum", 600)
    assert large_batch_cost < small_batch_cost


# Test 5: Energy agent positive value deta hai
def test_energy_agent_positive():
    steps = ["Facing", "Drilling", "Inspection"]
    result = energy_agent(steps, "Aluminum")
    assert result > 0


# Test 6: Efficiency score 0-100 ke beech hai
def test_efficiency_score_range():
    score = efficiency_agent(20, 40, 1.0)
    assert 0 <= score <= 100


# Test 7: evaluate_route complete dict deta hai
def test_evaluate_route_complete():
    steps = ["Facing", "Drilling", "Inspection"]
    result = evaluate_route("Route_A", steps, "Aluminum", 500)
    assert "time_min" in result
    assert "cost_inr" in result
    assert "energy_kwh" in result
    assert "efficiency_score" in result


# Test 8: Legacy mode (features nahi diye) saare fixed routes evaluate karta hai
def test_evaluate_all_routes_legacy():
    from routes import ALL_ROUTES
    results = evaluate_all_routes("Aluminum", 500)
    assert len(results) == len(ALL_ROUTES)


# Test 8b: Feature-aware mode sirf us part ke relevant routes evaluate karta hai
def test_evaluate_all_routes_feature_aware():
    results = evaluate_all_routes("Aluminum", 500, features=["Thread", "Fillet"])
    assert 1 <= len(results) <= 15
    for r in results:
        assert r["steps"][0] == "Facing" and r["steps"][-1] == "Inspection"
        assert "Corner Rounding/Filleting" in r["steps"]      # Fillet feature cover hona chahiye
        assert any(op in r["steps"] for op in ("Tapping", "Thread Milling", "External Threading"))   # Thread feature cover hona chahiye


# Test 9: Best route by efficiency milta hai
def test_find_best_route_efficiency():
    results = evaluate_all_routes("Aluminum", 500)
    best = find_best_route(results, "efficiency")
    assert best is not None
    assert "route_name" in best


# Test 10: Best route by cost sabse sasta hota hai
def test_find_best_route_cost():
    results = evaluate_all_routes("Steel", 50)
    best = find_best_route(results, "cost")
    min_cost = min(r["cost_inr"] for r in results)
    assert best["cost_inr"] == min_cost