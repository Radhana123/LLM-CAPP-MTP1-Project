# route_builder.py
# Dynamic Route Builder — Core Algorithm (Machine-Aware)
# Feature_vocab.py (Step 1) + precedence_graph.py (Step 2) use karke,
# kisi bhi feature-combination ke liye khud COMPLETE + VALID routes banata hai.
#
# MACHINE-AWARE: Lathe operations grouped together, Milling operations grouped
# together — unnecessary machine changeovers avoided. Shared operations (jo
# dono machines pe ho sakte hain) us group me jaate hain jaha unki zarurat hai.
#
# Guarantee: Route "incomplete" ho hi nahi sakta — completeness
# CONSTRUCTION se aati hai, filtering se nahi.

import random
import zlib
from itertools import product

import feature_vocab as fv
import precedence_graph as pg


# ════════════════════════════════════════════════════
# MACHINE CLASSIFICATION SETS
# ════════════════════════════════════════════════════
LATHE_ONLY_OPS = {
    "Plain/Cylindrical Turning", "Taper Turning", "Step Turning",
    "Grooving/Necking", "Parting-off", "Knurling", "Forming",
    "Internal Grooving", "External Threading", "Contour Turning",
    "Undercutting", "Eccentric Turning", "Polishing/Burnishing",
}

MILLING_ONLY_OPS = {
    "Face Milling", "Slab/Peripheral Milling", "Surface Contouring",
    "Slot Milling", "T-Slot Milling", "Dovetail Milling",
    "Woodruff Keyway Milling", "Pocket Milling", "Profile Milling",
    "Spotfacing", "Corner Rounding/Filleting", "Gear/Spline Milling",
    "Thread Milling", "Angular Milling", "Gang Milling",
    "Form Milling", "Helical Milling", "Engraving",
}

# Baaki sab shared: Center Drilling, Drilling, Boring, Reaming, Tapping,
# Counterboring, Countersinking, Chamfering


# ════════════════════════════════════════════════════
# STEP 1: Features se operation-combinations nikalo
# ════════════════════════════════════════════════════

def get_operation_combinations(features: list) -> list:
    """
    Har feature ke alternative chains ka cartesian product banao,
    phir har combination ka UNION nikaalo.
    Returns: list of sets — har set ek possible "required operations" combination hai.
    """
    if not features:
        return [set()]

    per_feature_alternatives = [fv.get_operations_for_feature(f) for f in features]

    combos = []
    seen = set()
    for combo in product(*per_feature_alternatives):
        merged = set()
        for chain in combo:
            merged.update(chain)
        key = frozenset(merged)
        if key not in seen:
            seen.add(key)
            combos.append(merged)

    return combos


# ════════════════════════════════════════════════════
# STEP 2: Topological ordering within a machine-phase
# ════════════════════════════════════════════════════

# Parting-off part ko bar se kaat deta hai -- us machine-phase mein iske baad
# koi aur kaam nahi ho sakta. Ye precedence graph mein sirf "sink" hai (iska koi
# successor nahi), isliye ise phase ke END mein khiskane se baaki rules nahi tootte.
_LAST_IN_PHASE = ("Parting-off",)


def _topological_orderings(operations: set, num_variants: int = 8, max_attempts: int = 60) -> list:
    """
    Randomized Kahn's algorithm — operations ke liye multiple DIFFERENT
    valid topological orderings nikalta hai.

    REPRODUCIBLE (MTP-1 WP0): randomness ab ek local RNG se aati hai jiska seed
    operations ke set se nikalta hai, aur set ko sorted karke iterate karte hain
    (string-set ka order har Python process mein alag hota hai). Isse same input
    hamesha same routes deta hai, global random state ya PYTHONHASHSEED se
    independent. Orderings ka distribution pehle jaisa hi hai.
    """
    if not operations:
        return [[]]

    rng = random.Random(zlib.crc32("|".join(sorted(operations)).encode("utf-8")))

    relevant_edges = [
        (a, b) for a, b in pg.PRECEDENCE_EDGES
        if a in operations and b in operations
    ]

    results = []
    attempts = 0
    while len(results) < num_variants and attempts < max_attempts:
        attempts += 1

        in_degree = {op: 0 for op in operations}
        graph = {op: [] for op in operations}
        for a, b in relevant_edges:
            graph[a].append(b)
            in_degree[b] += 1

        available = sorted(op for op in operations if in_degree[op] == 0)
        in_degree_copy = in_degree.copy()
        order = []

        while available:
            rng.shuffle(available)
            node = available.pop()
            order.append(node)
            for neighbor in graph[node]:
                in_degree_copy[neighbor] -= 1
                if in_degree_copy[neighbor] == 0:
                    available.append(neighbor)

        # Parting-off ko phase ke end mein rakho (agar uska koi successor nahi)
        movable = [op for op in _LAST_IN_PHASE if op in order
                   and not any(a == op for a, _b in relevant_edges)]
        if movable:
            order = [op for op in order if op not in movable] + movable

        if len(order) == len(operations) and order not in results:
            results.append(order)

    return results if results else [sorted(operations)]


# ════════════════════════════════════════════════════
# STEP 3: Machine-aware route construction
# ════════════════════════════════════════════════════

def _classify_and_assign(operations: set, machine_preference: str = "auto") -> dict:
    """
    Operations ko machine-type ke hisaab se classify karo aur
    shared ops ko best group me assign karo.

    machine_preference:
      "auto"           — system decides (minimize changeovers)
      "prefer_lathe"   — shared ops lathe pe jaayein jab possible
      "prefer_milling" — shared ops milling pe jaayein jab possible
    """
    ops = operations - {pg.ALWAYS_FIRST, pg.ALWAYS_LAST}

    lathe_ops = ops & LATHE_ONLY_OPS
    milling_ops = ops & MILLING_ONLY_OPS
    shared_ops = ops - lathe_ops - milling_ops

    has_lathe = len(lathe_ops) > 0
    has_milling = len(milling_ops) > 0

    # Warnings — jab user ki preference aur features clash kare
    warnings = []

    if machine_preference == "prefer_lathe" and has_milling:
        for op in milling_ops:
            warnings.append(f"'{op}' requires Milling machine — changeover unavoidable")
    elif machine_preference == "prefer_milling" and has_lathe:
        for op in lathe_ops:
            warnings.append(f"'{op}' requires Lathe machine — changeover unavoidable")

    if has_lathe and not has_milling and not shared_ops:
        lathe_ops = lathe_ops
        route_type = "LATHE_ONLY"

    elif has_milling and not has_lathe and not shared_ops:
        milling_ops = milling_ops
        route_type = "MILLING_ONLY"

    elif not has_lathe and not has_milling:
        # Sirf shared ops — preference decide karega
        if machine_preference == "prefer_milling":
            milling_ops = shared_ops
            route_type = "MILLING_ONLY"
        else:
            lathe_ops = shared_ops
            route_type = "LATHE_ONLY"
        shared_ops = set()

    elif has_lathe and not has_milling:
        # Lathe-only + shared → sab lathe pe
        if machine_preference == "prefer_milling" and shared_ops:
            # User prefers milling — shared ops milling pe, lathe ops lathe pe
            milling_ops = shared_ops
            shared_ops = set()
            route_type = "LATHE_FIRST" if len(lathe_ops) >= len(milling_ops) else "MILLING_FIRST"
        else:
            lathe_ops = lathe_ops | shared_ops
            shared_ops = set()
            route_type = "LATHE_ONLY"

    elif has_milling and not has_lathe:
        # Milling-only + shared → sab milling pe
        if machine_preference == "prefer_lathe" and shared_ops:
            lathe_ops = shared_ops
            shared_ops = set()
            route_type = "MILLING_FIRST" if len(milling_ops) >= len(lathe_ops) else "LATHE_FIRST"
        else:
            milling_ops = milling_ops | shared_ops
            shared_ops = set()
            route_type = "MILLING_ONLY"

    else:
        # Dono machines zaroori — shared ops preference ke hisaab se assign
        for op in sorted(shared_ops):      # sorted: the choice below depends on how many ops each machine already has,
            if machine_preference == "prefer_lathe":   # so a hash-order iteration made the routes differ between Python processes
                lathe_ops.add(op)
            elif machine_preference == "prefer_milling":
                milling_ops.add(op)
            else:
                # Auto — precedence-based smart assignment
                lathe_connected = any(
                    (op, l) in pg.PRECEDENCE_EDGES or (l, op) in pg.PRECEDENCE_EDGES
                    for l in lathe_ops
                )
                milling_connected = any(
                    (op, m) in pg.PRECEDENCE_EDGES or (m, op) in pg.PRECEDENCE_EDGES
                    for m in milling_ops
                )
                if lathe_connected and not milling_connected:
                    lathe_ops.add(op)
                elif milling_connected and not lathe_connected:
                    milling_ops.add(op)
                elif len(lathe_ops) >= len(milling_ops):
                    lathe_ops.add(op)
                else:
                    milling_ops.add(op)
        shared_ops = set()
        route_type = "LATHE_FIRST" if len(lathe_ops) >= len(milling_ops) else "MILLING_FIRST"

    return {
        "lathe": lathe_ops,
        "milling": milling_ops,
        "route_type": route_type,
        "warnings": warnings
    }


def build_routes(features: list, max_routes: int = 5, machine_preference: str = "auto") -> list:
    """
    Machine-aware route construction with preference support.
    """
    combos = get_operation_combinations(features)

    all_routes = []
    seen_routes = set()
    all_warnings = []

    for combo in combos:
        classification = _classify_and_assign(combo, machine_preference)
        lathe_ops = classification["lathe"]
        milling_ops = classification["milling"]
        route_type = classification["route_type"]
        all_warnings.extend(classification["warnings"])

        if route_type == "LATHE_ONLY":
            for ordering in _topological_orderings(lathe_ops, num_variants=8):
                route = [pg.ALWAYS_FIRST] + ordering + [pg.ALWAYS_LAST]
                key = tuple(route)
                if key not in seen_routes:
                    seen_routes.add(key)
                    all_routes.append({
                        "steps": route,
                        "type": "LATHE_ONLY",
                        "changeovers": 0
                    })

        elif route_type == "MILLING_ONLY":
            for ordering in _topological_orderings(milling_ops, num_variants=8):
                route = [pg.ALWAYS_FIRST] + ordering + [pg.ALWAYS_LAST]
                key = tuple(route)
                if key not in seen_routes:
                    seen_routes.add(key)
                    all_routes.append({
                        "steps": route,
                        "type": "MILLING_ONLY",
                        "changeovers": 0
                    })

        elif route_type == "LATHE_FIRST":
            for l_order in _topological_orderings(lathe_ops, num_variants=4):
                for m_order in _topological_orderings(milling_ops, num_variants=4):
                    route = [pg.ALWAYS_FIRST] + l_order + ["--- Machine Changeover ---"] + m_order + [pg.ALWAYS_LAST]
                    key = tuple(route)
                    if key not in seen_routes:
                        seen_routes.add(key)
                        all_routes.append({
                            "steps": route,
                            "type": "LATHE_FIRST",
                            "changeovers": 1
                        })

        elif route_type == "MILLING_FIRST":
            for m_order in _topological_orderings(milling_ops, num_variants=4):
                for l_order in _topological_orderings(lathe_ops, num_variants=4):
                    route = [pg.ALWAYS_FIRST] + m_order + ["--- Machine Changeover ---"] + l_order + [pg.ALWAYS_LAST]
                    key = tuple(route)
                    if key not in seen_routes:
                        seen_routes.add(key)
                        all_routes.append({
                            "steps": route,
                            "type": "MILLING_FIRST",
                            "changeovers": 1
                        })

        if len(all_routes) >= max_routes:
            break

    # Sort: 0-changeover routes first
    all_routes.sort(key=lambda x: x["changeovers"])
    # Attach warnings to each route
    unique_warnings = sorted(set(all_warnings))
    for r in all_routes:
        r["warnings"] = unique_warnings
    return all_routes[:max_routes]


# ════════════════════════════════════════════════════
# COMPLETENESS CHECK
# ════════════════════════════════════════════════════

def is_complete(route: list, features: list) -> bool:
    """
    Har feature ke liye check karo — uska kam se kam ek alternative chain
    poori tarah route me maujood hai.
    """
    route_set = set(route)
    for feature in features:
        alternatives = fv.get_operations_for_feature(feature)
        if not any(set(alt).issubset(route_set) for alt in alternatives):
            return False
    return True


# ════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════

def generate_valid_routes(features: list, max_routes: int = 5, machine_preference: str = "auto") -> list:
    """
    Main entry point — returns list of dicts:
    [{"steps": [...], "type": "LATHE_FIRST", "changeovers": 1}, ...]

    machine_preference: "auto", "prefer_lathe", "prefer_milling"
    Guarantee: har route (a) COMPLETE hai (b) precedence-VALID hai
    (c) machine-grouped hai (changeovers minimized/preference-aware)
    """
    candidates = build_routes(features, max_routes=max_routes * 3, machine_preference=machine_preference)

    final_routes = []
    for candidate in candidates:
        route = candidate["steps"]
        # Validate on clean steps (without changeover markers)
        clean = [s for s in route if s != "--- Machine Changeover ---"]
        if is_complete(clean, features):
            valid, _ = pg.validate_order(clean)
            if valid:
                final_routes.append(candidate)
        if len(final_routes) >= max_routes:
            break

    return final_routes


# Backward compatibility — kuch purani files list-of-lists expect karti hain
def generate_valid_routes_simple(features: list, max_routes: int = 5) -> list:
    """Same as generate_valid_routes but returns list of step-lists (not dicts)."""
    routes = generate_valid_routes(features, max_routes)
    return [r["steps"] for r in routes]


if __name__ == "__main__":
    print("=== Test 1: Lathe-only (Step + Groove) ===")
    routes1 = generate_valid_routes(["Step", "Groove"])
    for r in routes1:
        print(f"  [{r['type']}, {r['changeovers']} changeover] {' -> '.join(r['steps'])}")

    print("\n=== Test 2: Milling-only (Slot + Pocket) ===")
    routes2 = generate_valid_routes(["Slot", "Pocket"])
    for r in routes2:
        print(f"  [{r['type']}, {r['changeovers']} changeover] {' -> '.join(r['steps'])}")

    print("\n=== Test 3: Mixed — Thread(Lathe) + Fillet(Milling) ===")
    routes3 = generate_valid_routes(["Thread", "Fillet"])
    for r in routes3:
        print(f"  [{r['type']}, {r['changeovers']} changeover] {' -> '.join(r['steps'])}")

    print("\n=== Test 4: Complex mixed — Hole + Keyway + Taper ===")
    routes4 = generate_valid_routes(["Hole", "Keyway", "Taper"])
    for r in routes4:
        print(f"  [{r['type']}, {r['changeovers']} changeover] {' -> '.join(r['steps'])}")

    print("\n=== Test 5: Shared-only (Hole + Counterbore) ===")
    routes5 = generate_valid_routes(["Hole", "Counterbore"])
    for r in routes5:
        print(f"  [{r['type']}, {r['changeovers']} changeover] {' -> '.join(r['steps'])}")