# precedence_graph.py
# Operation ordering rules — Dynamic Route Builder ke liye
# Week 1 (foundation) → route_builder.py yahi rules use karega


from itertools import combinations

# ════════════════════════════════════════════════════
# 1. UNIVERSAL RULES (77/77 legacy routes confirm)
# ════════════════════════════════════════════════════
ALWAYS_FIRST = "Facing"
ALWAYS_LAST = "Inspection"


# ════════════════════════════════════════════════════
# 2. HIGH-CONFIDENCE RULES (legacy data-mined, 3+ routes support, canonical names)
# Format: (A, B) matlab A hamesha B se PEHLE aana chahiye
# ════════════════════════════════════════════════════
HIGH_CONFIDENCE_EDGES = [
    ("Plain/Cylindrical Turning", "Boring"),
    ("Center Drilling", "Drilling"),
    ("Drilling", "Reaming"),
    ("Plain/Cylindrical Turning", "Drilling"),
    ("Plain/Cylindrical Turning", "Grooving/Necking"),
    ("Plain/Cylindrical Turning", "Knurling"),
    ("Plain/Cylindrical Turning", "Pocket Milling"),
    ("Plain/Cylindrical Turning", "External Threading"),
    ("Grooving/Necking", "External Threading"),
    ("Pocket Milling", "Slot Milling"),
    ("Boring", "Reaming"),
    ("Boring", "External Threading"),
    ("Center Drilling", "Reaming"),
    ("Center Drilling", "External Threading"),
    ("Boring", "Chamfering"),
    ("Center Drilling", "Chamfering"),
    ("Drilling", "Chamfering"),
    ("Grooving/Necking", "Chamfering"),
    ("Pocket Milling", "Chamfering"),
    ("Reaming", "Chamfering"),
    ("Slot Milling", "Chamfering"),
    ("External Threading", "Chamfering"),
    ("Plain/Cylindrical Turning", "Chamfering"),
]

# ════════════════════════════════════════════════════
# 3. VOCAB-CONSISTENT RULES (feature_vocab.py ki apni chains se, self-consistent)
# ════════════════════════════════════════════════════
VOCAB_CONSISTENT_EDGES = [
    ("Drilling", "Boring"),           # Hole feature: [Center Drilling, Drilling, Boring]
    ("Drilling", "Tapping"),          # Thread_Internal: [..., Drilling, Tapping]
    ("Drilling", "Thread Milling"),   # Thread_Internal: [..., Drilling, Thread Milling]
    ("Drilling", "Counterboring"),    # Counterbore: [..., Drilling, Counterboring]
    ("Drilling", "Countersinking"),   # Countersink: [..., Drilling, Countersinking]
    ("Center Drilling", "Tapping"),
    ("Center Drilling", "Counterboring"),
    ("Center Drilling", "Countersinking"),
    ("Center Drilling", "Thread Milling"),
]

# ════════════════════════════════════════════════════
# 3b. NEW-FEATURE RULES (Parting_Off / Undercut ke liye physical ordering)
# Parting-off hamesha baaki lathe kaam ke BAAD (part kat jaane ke baad turning
# nahi ho sakti), aur Undercut (relief groove) turning ke baad, threading se pehle.
# ════════════════════════════════════════════════════
NEW_FEATURE_EDGES = [
    ("Plain/Cylindrical Turning", "Undercutting"),
    ("Undercutting", "External Threading"),
    ("Plain/Cylindrical Turning", "Parting-off"),
    ("Step Turning", "Parting-off"),
    ("Taper Turning", "Parting-off"),
    ("Contour Turning", "Parting-off"),
    ("Eccentric Turning", "Parting-off"),
    ("Undercutting", "Parting-off"),
    ("Grooving/Necking", "Parting-off"),
    ("External Threading", "Parting-off"),
    ("Knurling", "Parting-off"),
    ("Boring", "Parting-off"),
]

# Final live graph — ye hi route_builder.py use karega
PRECEDENCE_EDGES = HIGH_CONFIDENCE_EDGES + VOCAB_CONSISTENT_EDGES + NEW_FEATURE_EDGES


# ════════════════════════════════════════════════════
# 4. LOW-CONFIDENCE RULES — sirf 1 legacy route se, NOT enforced
# Review karke decide karo enable karni hai ya nahi
# ════════════════════════════════════════════════════
REVIEW_RULES = [
    ("Center Drilling", "Boring"),
    ("Boring", "Pocket Milling"),
    ("Center Drilling", "Pocket Milling"),
    ("Parting-off", "Chamfering"),
    ("Grooving/Necking", "Drilling"),
    ("Knurling", "Grooving/Necking"),
    ("Grooving/Necking", "Reaming"),
    ("Knurling", "External Threading"),
    ("Plain/Cylindrical Turning", "Parting-off"),
    ("External Threading", "Pocket Milling"),
    ("Reaming", "External Threading"),
    ("Plain/Cylindrical Turning", "Slot Milling"),
]


# ════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════

def get_prerequisites(op: str) -> set:
    """Diye gaye operation ke liye directly-required predecessor operations."""
    return {a for a, b in PRECEDENCE_EDGES if b == op}


def get_successors(op: str) -> set:
    """Diye gaye operation ke baad kaunse operations aa sakte hain (direct)."""
    return {b for a, b in PRECEDENCE_EDGES if a == op}


def validate_order(route: list) -> tuple:
    """
    Ek route (operations ki list) ke against saare precedence rules check karo.
    Returns: (is_valid: bool, violations: list of error strings)
    """
    violations = []

    if route and route[0] != ALWAYS_FIRST:
        violations.append(f"'{ALWAYS_FIRST}' route ka pehla step hona chahiye, mila: '{route[0]}'")

    if route and route[-1] != ALWAYS_LAST:
        violations.append(f"'{ALWAYS_LAST}' route ka last step hona chahiye, mila: '{route[-1]}'")

    for a, b in PRECEDENCE_EDGES:
        if a in route and b in route:
            if route.index(a) > route.index(b):
                violations.append(f"'{a}' ko '{b}' se pehle aana chahiye, par order ulta hai")

    return (len(violations) == 0, violations)


def has_cycle() -> bool:
    """Sanity check — PRECEDENCE_EDGES me koi circular dependency toh nahi (DFS-based)."""
    graph = {}
    for a, b in PRECEDENCE_EDGES:
        graph.setdefault(a, []).append(b)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {}

    def dfs(node):
        color[node] = GRAY
        for neighbor in graph.get(node, []):
            if color.get(neighbor, WHITE) == GRAY:
                return True  # back-edge mila = cycle
            if color.get(neighbor, WHITE) == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    for node in list(graph.keys()):
        if color.get(node, WHITE) == WHITE:
            if dfs(node):
                return True
    return False


if __name__ == "__main__":
    print("=== Sanity Check: Koi cycle toh nahi PRECEDENCE_EDGES me? ===")
    print("Cycle detected:", has_cycle())

    print(f"\nTotal HIGH-CONFIDENCE edges: {len(HIGH_CONFIDENCE_EDGES)}")
    print(f"Total VOCAB-CONSISTENT edges: {len(VOCAB_CONSISTENT_EDGES)}")
    print(f"Total live PRECEDENCE_EDGES: {len(PRECEDENCE_EDGES)}")
    print(f"Total REVIEW (not enforced) edges: {len(REVIEW_RULES)}")

    print("\n=== Test: validate_order() ===")
    good_route = ["Facing", "Center Drilling", "Drilling", "Tapping", "Chamfering", "Inspection"]
    bad_route = ["Facing", "Tapping", "Drilling", "Inspection"]  # Tapping before Drilling — invalid!

    valid, errors = validate_order(good_route)
    print(f"\nGood route: {good_route}\nValid: {valid}, Errors: {errors}")

    valid, errors = validate_order(bad_route)
    print(f"\nBad route: {bad_route}\nValid: {valid}, Errors: {errors}")