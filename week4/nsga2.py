# nsga2.py
# NSGA-II Multi-Objective Optimization
# Week 4 | LLM-CAPP Project
# UPDATED:
#   1. generate_valid_routes() ab list of dicts return karta hai
#      ({"steps": [...], "type": "LATHE_FIRST", "changeovers": 1})
#      — isliye population creation me r["steps"] use hota hai
#   2. Cost display INR me (₹)
#   3. Core algorithm (dominance, sorting, crowding) UNCHANGED

import random
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week3")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week1")))

from agents import time_agent, cost_agent, energy_agent
from route_builder import generate_valid_routes, is_complete

random.seed(42)


# ── Individual (ek solution) ──────────────────────
class Individual:
    def __init__(self, route_name: str, steps: list, material: str, batch_size: int):
        self.route_name  = route_name
        self.steps       = steps
        self.material    = material
        self.batch_size  = batch_size
        self.objectives  = self._evaluate()
        self.rank        = 0
        self.crowding    = 0.0

    def _evaluate(self) -> tuple:
        t = time_agent(self.steps, self.material)
        c = cost_agent(self.steps, self.material, self.batch_size)
        e = energy_agent(self.steps, self.material)
        return (t, c, e)


# ── Dominance check ───────────────────────────────
def dominates(a: Individual, b: Individual) -> bool:
    a_obj = a.objectives
    b_obj = b.objectives
    not_worse = all(a_obj[i] <= b_obj[i] for i in range(3))
    strictly_better = any(a_obj[i] < b_obj[i] for i in range(3))
    return not_worse and strictly_better


# ── Non-Dominated Sorting ─────────────────────────
def non_dominated_sort(population: list) -> list:
    fronts = [[]]
    dominated_by = {i: [] for i in range(len(population))}
    domination_count = {i: 0 for i in range(len(population))}

    for i, p in enumerate(population):
        for j, q in enumerate(population):
            if i == j:
                continue
            if dominates(p, q):
                dominated_by[i].append(j)
            elif dominates(q, p):
                domination_count[i] += 1

        if domination_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = current_front + 1
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    return fronts[:-1]


# ── Crowding Distance ─────────────────────────────
def crowding_distance(population: list, front: list) -> None:
    n = len(front)
    if n <= 2:
        for i in front:
            population[i].crowding = float('inf')
        return

    for i in front:
        population[i].crowding = 0.0

    for obj_idx in range(3):
        sorted_front = sorted(front, key=lambda i: population[i].objectives[obj_idx])
        population[sorted_front[0]].crowding = float('inf')
        population[sorted_front[-1]].crowding = float('inf')

        obj_min = population[sorted_front[0]].objectives[obj_idx]
        obj_max = population[sorted_front[-1]].objectives[obj_idx]
        obj_range = obj_max - obj_min if obj_max != obj_min else 1e-9

        for k in range(1, n - 1):
            prev_val = population[sorted_front[k-1]].objectives[obj_idx]
            next_val = population[sorted_front[k+1]].objectives[obj_idx]
            population[sorted_front[k]].crowding += (next_val - prev_val) / obj_range


# ── Main NSGA-II Function ─────────────────────────
def run_nsga2(material: str, batch_size: int, features: list = None, max_routes: int = 15,
              machine_preference: str = "auto", tolerance: str = "0.05mm",
              llm_attempts: int = 5, route_generation_mode: str = "llm_first") -> list:
    """
    NSGA-II run karo aur Pareto-optimal routes return karo.

    route_generation_mode="llm_first" (NEW DEFAULT):
        LLM Planner PRIMARY route-generator hai. llm_attempts baar LLM se
        candidate routes generate karwaye jaate hain (population diversity
        ke liye — ek single call se sirf 1 route milta, NSGA-II ko compare
        karne ke liye multiple chahiye). Har candidate FSM se validate hota
        hai; invalid ho toh Route Builder turant correct karta hai — Route
        Builder yahan ek INDEPENDENT generator nahi hai, sirf "corrector".
        Agar LLM completely fail ho jaaye (API down / sab candidates
        invalid+uncorrectable), Route Builder ek safety-net fallback deta
        hai taaki population kabhi khaali na rahe.

    route_generation_mode="route_builder_first" (legacy/backward-compatible):
        Purana behavior — Route Builder primary generator, LLM ek parallel
        extra candidate. Testing/comparison ke liye retained.
    """
    features = features or []
    population = []
    llm_status = {"mode": route_generation_mode, "attempts": 0, "llm_valid": 0,
                  "llm_corrected": 0, "llm_failed": 0,
                  "route_builder_fallback_used": False, "reason": "",
                  "machine_pref_overridden": False, "fallback_incomplete": False}

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week2")))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../week4")))

    if route_generation_mode == "llm_first" and features:
        try:
            from llm_planner import generate_process_plan
            from fsm_validator import validate_sequence, fix_sequence_with_builder

            for attempt in range(llm_attempts):
                llm_status["attempts"] += 1
                try:
                    llm_result = generate_process_plan(material, features, tolerance, batch_size, sample_id=attempt)
                    if not llm_result["success"]:
                        llm_status["llm_failed"] += 1
                        continue

                    llm_steps = llm_result["steps"]
                    fsm_check = validate_sequence(llm_steps)

                    # FSM sirf sequence-ORDER check karta hai (Facing pehle,
                    # Inspection last, precedence sahi) -- ye kabhi check
                    # nahi karta ki route mein requested features actually
                    # cover ho rahe hain. Isse ek lazy/minimal LLM route
                    # (jaise sirf "Facing, Inspection") FSM se trivially
                    # "valid" pass ho jaata, aur chunki uska time/cost/energy
                    # sabse kam hota (kuch kaam nahi kiya), NSGA-II use
                    # automatically "best" maan leta -- saare asli complete
                    # routes ko dominate karke Pareto front jeet leta. Ab
                    # completeness bhi explicitly check karte hain.
                    if fsm_check["valid"] and is_complete(llm_steps, features):
                        population.append(Individual(f"Route_LLM_{attempt+1}", llm_steps, material, batch_size))
                        llm_status["llm_valid"] += 1
                    else:
                        # Route Builder = CORRECTOR only here, not an independent generator
                        corrected = fix_sequence_with_builder(features)
                        if corrected and validate_sequence(corrected)["valid"] and is_complete(corrected, features):
                            population.append(Individual(f"Route_LLM_{attempt+1}_corrected", corrected, material, batch_size))
                            llm_status["llm_corrected"] += 1
                        else:
                            llm_status["llm_failed"] += 1
                except Exception:
                    llm_status["llm_failed"] += 1

            # Duplicate routes hatao (LLM baar-baar same route de sakta hai)
            seen, deduped = set(), []
            for ind in population:
                key = tuple(ind.steps)
                if key not in seen:
                    seen.add(key)
                    deduped.append(ind)
            population = deduped

            if population:
                llm_status["reason"] = (f"{llm_status['llm_valid']} valid, "
                                        f"{llm_status['llm_corrected']} corrected, "
                                        f"{llm_status['llm_failed']} failed — "
                                        f"{len(population)} unique routes in population")
            else:
                llm_status["reason"] = "All LLM attempts failed — falling back to Route Builder"

        except Exception as e:
            llm_status["reason"] = f"LLM path unavailable: {e}"

        # ── SAFETY NET — agar LLM se kuch bhi usable nahi mila ──
        if not population:
            llm_status["route_builder_fallback_used"] = True
            candidate_routes = generate_valid_routes(features, max_routes=max(3, max_routes),
                                                      machine_preference=machine_preference)

            # Agar restrictive machine_preference (prefer_lathe/prefer_milling)
            # ke saath koi route hi nahi ban paaya -- kyunki requested
            # features force karte hain doosre machine type ki zaroorat --
            # to "auto" try karo. Pehle ye silently trivial 2-step fallback
            # de deta tha, jisse lagta LLM ne khud hi ek incomplete route
            # "valid" maan liya, jabki asal wajah sirf machine-preference
            # constraint thi.
            if not candidate_routes and machine_preference != "auto":
                candidate_routes = generate_valid_routes(features, max_routes=max(3, max_routes),
                                                          machine_preference="auto")
                if candidate_routes:
                    llm_status["machine_pref_overridden"] = True
                    llm_status["reason"] = (
                        f"'{machine_preference}' se koi valid route nahi bana (requested "
                        f"features doosre machine type maangte hain) — Auto strategy use ki gayi."
                    )

            if not candidate_routes:
                # Genuinely koi bhi complete route nahi ban saka is feature
                # combination ke liye (kisi bhi machine strategy se). Ab bhi
                # ek fallback dena hai (crash se behtar), lekin explicitly
                # flag karte hain ki ye INCOMPLETE hai, taaki UI ise kabhi
                # "the answer" ki tarah na dikhaye.
                candidate_routes = [{"steps": ["Facing", "Inspection"], "type": "FALLBACK", "changeovers": 0}]
                llm_status["fallback_incomplete"] = True
                llm_status["reason"] = (
                    "Koi bhi complete, valid route is feature combination ke liye "
                    "nahi ban saka — dikhaya gaya route features cover NAHI karta."
                )

            population = [
                Individual(f"Route_Builder_Fallback_{i+1}", r["steps"], material, batch_size)
                for i, r in enumerate(candidate_routes)
            ]

    else:
        # ── LEGACY MODE: Route Builder primary, LLM parallel (backward-compatible) ──
        candidate_routes = generate_valid_routes(features, max_routes=max_routes, machine_preference=machine_preference)
        if not candidate_routes and machine_preference != "auto":
            candidate_routes = generate_valid_routes(features, max_routes=max_routes, machine_preference="auto")
            if candidate_routes:
                llm_status["machine_pref_overridden"] = True
                llm_status["reason"] = (
                    f"'{machine_preference}' se koi valid route nahi bana — Auto strategy use ki gayi."
                )
        if not candidate_routes:
            candidate_routes = [{"steps": ["Facing", "Inspection"], "type": "FALLBACK", "changeovers": 0}]
            llm_status["fallback_incomplete"] = True
            llm_status["reason"] = (
                "Koi bhi complete, valid route is feature combination ke liye nahi ban saka."
            )
        population = [
            Individual(f"Route_{i+1}", r["steps"], material, batch_size)
            for i, r in enumerate(candidate_routes)
        ]
        if features:
            try:
                from llm_planner import generate_process_plan
                from fsm_validator import validate_sequence, fix_sequence_with_builder
                llm_result = generate_process_plan(material, features, tolerance, batch_size)
                if llm_result["success"]:
                    fsm_check = validate_sequence(llm_result["steps"])
                    if fsm_check["valid"] and is_complete(llm_result["steps"], features):
                        population.append(Individual("Route_LLM", llm_result["steps"], material, batch_size))
                        llm_status["llm_valid"] = 1
                    else:
                        corrected = fix_sequence_with_builder(features)
                        if corrected and validate_sequence(corrected)["valid"] and is_complete(corrected, features):
                            population.append(Individual("Route_LLM_corrected", corrected, material, batch_size))
                            llm_status["llm_corrected"] = 1
            except Exception as e:
                llm_status["reason"] = f"LLM path skipped: {e}"

    fronts = non_dominated_sort(population)

    for front in fronts:
        crowding_distance(population, front)

    pareto_front = [population[i] for i in fronts[0]]
    pareto_front_indices = set(fronts[0])

    llm_status["reached_pareto_front"] = sum(
        1 for i in pareto_front_indices if population[i].route_name.startswith("Route_LLM")
    )
    llm_status["route_builder_in_pareto_front"] = sum(
        1 for i in pareto_front_indices if population[i].route_name.startswith("Route_Builder") or population[i].route_name.startswith("Route_") and not population[i].route_name.startswith("Route_LLM")
    )
    run_nsga2.last_llm_status = llm_status

    return pareto_front


def print_pareto(pareto: list, material: str, batch_size: int):
    print(f"\n{'='*70}")
    print(f"  NSGA-II Results — {material}, Batch: {batch_size}")
    print(f"{'='*70}")
    print(f"  {'Route':<20} {'Time(min)':<12} {'Cost(₹)':<12} {'Energy(kWh)':<14} {'Steps'}")
    print(f"  {'─'*65}")
    for ind in sorted(pareto, key=lambda x: x.objectives[0]):
        t, c, e = ind.objectives
        has_co = "⚙" if "--- Machine Changeover ---" in ind.steps else " "
        tag = "🤖" if ind.route_name.startswith("Route_LLM") else "  "
        clean_steps = [s for s in ind.steps if s != "--- Machine Changeover ---"]
        print(f"  {tag}{ind.route_name:<18} {t:<12} ₹{c:<11} {e:<14} {has_co} {' → '.join(clean_steps)}")
    print(f"{'='*70}")
    print(f"  Pareto-optimal routes: {len(pareto)}")

    status = getattr(run_nsga2, "last_llm_status", None)
    if status:
        print(f"\n  🤖 Route Generation Status (mode: {status['mode']}):")
        print(f"     LLM attempts         : {status['attempts']}")
        print(f"     LLM valid as-is      : {status['llm_valid']}")
        print(f"     LLM self-corrected   : {status['llm_corrected']}")
        print(f"     LLM failed           : {status['llm_failed']}")
        print(f"     Route Builder fallback used : {status['route_builder_fallback_used']}")
        print(f"     LLM routes in Pareto front  : {status['reached_pareto_front']}")
        print(f"     Detail               : {status['reason']}")


if __name__ == "__main__":
    print("=== NSGA-II Multi-Objective Optimization (₹ INR) ===")

    print("\nTest 1: Thread + Fillet (asli bug case)")
    pareto1 = run_nsga2("Aluminum", 500, ["Thread", "Fillet"])
    print_pareto(pareto1, "Aluminum", 500)

    print("\nTest 2: Hole + Keyway + Taper (mixed machines)")
    pareto2 = run_nsga2("Steel", 50, ["Hole", "Keyway", "Taper"])
    print_pareto(pareto2, "Steel", 50)

    print("\nTest 3: Pocket + Chamfer (milling only)")
    pareto3 = run_nsga2("Brass", 100, ["Pocket", "Chamfer"])
    print_pareto(pareto3, "Brass", 100)