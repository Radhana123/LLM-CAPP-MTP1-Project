# planner.py
# Main Planner — Week 2 | LLM-CAPP Project
# FIXED: select_route() poori tarah TOOTI HUI thi — "Route_A"/"Route_B"/"Route_C"
# reference karti thi jo ALL_ROUTES me exist hi nahi karte (KeyError crash).
# Ye woh original root-cause tha jiski wajah se pipeline hamesha ek hi (ya
# crash) route deta tha. Ab route_builder.py use hota hai — features se khud
# complete + precedence-valid route banta hai, koi hardcoded 3-route
# keyword-matching nahi.

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week1")))

from tokenizer import tokenize
from route_builder import generate_valid_routes


def plan(part_json: dict) -> dict:
    """
    Part JSON lekar complete process plan banao.
    """
    # Step 1: Tokenize karo
    token_result = tokenize(part_json)

    if not token_result["success"]:
        return {
            "success": False,
            "errors": token_result["errors"]
        }

    # Step 2: Route Builder se complete+valid routes banao (features se, seedha)
    features = part_json.get("features", [])
    candidate_routes = generate_valid_routes(features, max_routes=5)

    if not candidate_routes:
        return {
            "success": False,
            "errors": [f"Koi valid route nahi ban paya in features ke liye: {features}"]
        }

    selected_route = candidate_routes[0]["steps"]   # generate_valid_routes returns dicts

    return {
        "success": True,
        "tokens": token_result["tokens"],
        "token_labels": token_result.get("token_labels", features),
        "selected_route": "Route_1",
        "process_steps": selected_route,
        "alternative_routes": [r["steps"] for r in candidate_routes[1:]],
    }


def print_plan(result: dict):
    if not result["success"]:
        print("❌ FAILED:", result["errors"])
        return

    print(f"\n{'='*50}")
    print(f"  ✅ Process Plan Generated!")
    print(f"{'='*50}")
    print(f"  Tokens  : {result['tokens']}")
    print(f"  Labels  : {result['token_labels']}")
    print(f"\n  Process Steps:")
    for i, step in enumerate(result["process_steps"], 1):
        print(f"    {i}. {step}")
    if result.get("alternative_routes"):
        print(f"\n  ({len(result['alternative_routes'])} alternative routes bhi available hain)")


if __name__ == "__main__":
    part1 = {
        "material":   "Aluminum",
        "features":   ["Hole", "Slot"],
        "tolerance":  "0.02mm",
        "batch_size": 500
    }
    print("─── Part 1: Aluminum Hole+Slot ───")
    print_plan(plan(part1))

    part2 = {
        "material":   "Steel",
        "features":   ["Thread", "Pocket"],
        "tolerance":  "0.01mm",
        "batch_size": 50
    }
    print("\n─── Part 2: Steel Thread+Pocket ───")
    print_plan(plan(part2))

    # Asli bug case bhi test karte hain
    part3 = {
        "material":   "Titanium",
        "features":   ["Thread", "Fillet"],
        "tolerance":  "0.05mm",
        "batch_size": 100
    }
    print("\n─── Part 3: Titanium Thread+Fillet (asli bug case) ───")
    print_plan(plan(part3))