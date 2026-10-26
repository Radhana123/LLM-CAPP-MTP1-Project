# multi_agent_eval.py
# Saare Routes ko evaluate karke compare karo
# Week 3 | LLM-CAPP Project (MAIN FILE)
# UPDATED: Purana evaluate_all_routes() features ko bilkul ignore karta tha —
# saare 77 fixed routes evaluate karta tha chahe part me koi bhi feature ho.
# Matlab find_best_route() kisi bhi part ke liye UNRELATED route bhi "best"
# bata sakta tha. Ab optional `features` param diya — agar diya jaye, seedha
# route_builder.py se sirf us part ke liye RELEVANT+COMPLETE routes evaluate
# honge. Features na diye jaayein toh purana (legacy, all-77) behavior chalta
# rahega — backward compatible.

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week2")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week1")))

from routes import ALL_ROUTES
from agents import evaluate_route


def evaluate_all_routes(material: str, batch_size: int, features: list = None) -> list:
    """
    Routes ko evaluate karke list return karo.

    features diya gaya (RECOMMENDED): route_builder.py se sirf us part ke
    liye relevant, COMPLETE, precedence-valid routes evaluate honge.

    features NAHI diya (LEGACY): purana behavior — saare 77 fixed routes
    evaluate honge, feature-relevance ka koi check nahi (sirf generic/demo
    comparison ke liye use karo, real part ke liye features zaroor do).
    """
    results = []

    if features:
        from route_builder import generate_valid_routes_simple
        candidate_routes = generate_valid_routes_simple(features, max_routes=15)   # step-lists (generate_valid_routes returns dicts)
        for i, steps in enumerate(candidate_routes):
            result = evaluate_route(f"Route_{i+1}", steps, material, batch_size)
            results.append(result)
    else:
        for route_name, steps in ALL_ROUTES.items():
            result = evaluate_route(route_name, steps, material, batch_size)
            results.append(result)

    return results


def print_comparison(results: list):
    """
    Saare routes ko ek table jaisa format mein print karo.
    """
    print(f"\n{'='*75}")
    print(f"  {'Route':<10} {'Time(min)':<12} {'Cost(₹)':<10} {'Energy(kWh)':<14} {'Efficiency':<10}")
    print(f"{'='*75}")
    for r in results:
        print(f"  {r['route_name']:<10} {r['time_min']:<12} {r['cost_inr']:<10} {r['energy_kwh']:<14} {r['efficiency_score']:<10}")
    print(f"{'='*75}")


def find_best_route(results: list, priority: str = "efficiency") -> dict:
    """
    Best route nikalo based on priority:
    'time', 'cost', 'energy', ya 'efficiency'
    """
    if priority == "time":
        return min(results, key=lambda r: r["time_min"])
    elif priority == "cost":
        return min(results, key=lambda r: r["cost_inr"])
    elif priority == "energy":
        return min(results, key=lambda r: r["energy_kwh"])
    else:  # efficiency — higher is better
        return max(results, key=lambda r: r["efficiency_score"])


if __name__ == "__main__":
    material = "Aluminum"
    batch_size = 500

    print("=== LEGACY MODE (features nahi diye — saare 77 routes, generic demo) ===")
    print(f"\n📦 Part: {material}, Batch Size: {batch_size}")
    results = evaluate_all_routes(material, batch_size)
    print_comparison(results)
    print("\n🏆 Best Routes by Priority (in-relevant ho sakta hai, kyunki features nahi diye):")
    print(f"  Fastest        : {find_best_route(results, 'time')['route_name']}")
    print(f"  Cheapest       : {find_best_route(results, 'cost')['route_name']}")
    print(f"  Most Energy-Efficient : {find_best_route(results, 'energy')['route_name']}")
    print(f"  Best Overall   : {find_best_route(results, 'efficiency')['route_name']}")

    print("\n\n=== FEATURE-AWARE MODE (recommended — Thread + Fillet asli bug case) ===")
    print(f"\n📦 Part: {material}, Batch Size: {batch_size}, Features: ['Thread', 'Fillet']")
    results2 = evaluate_all_routes(material, batch_size, features=["Thread", "Fillet"])
    print_comparison(results2)
    print("\n🏆 Best Routes by Priority (sab relevant + complete hain):")
    print(f"  Fastest        : {find_best_route(results2, 'time')['route_name']}")
    print(f"  Cheapest       : {find_best_route(results2, 'cost')['route_name']}")
    print(f"  Most Energy-Efficient : {find_best_route(results2, 'energy')['route_name']}")
    print(f"  Best Overall   : {find_best_route(results2, 'efficiency')['route_name']}")