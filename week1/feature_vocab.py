# ============================================================
# 1. GEOMETRY FEATURES
# ============================================================
GEOMETRY_FEATURES = [
    # --- Original 10 (unchanged) ---
    "Hole",              # Round chhed — drilling se banta hai
    "Slot",              # Lambi cut — milling se
    "Pocket",            # Andar ki khudai — end mill se
    "Boss",              # Utha hua hissa
    "Thread",            # Pech wali cutting (external ya internal)
    "Chamfer",           # Edge pe angle cutting
    "Fillet",            # Curved/rounded edge
    "Groove",            # Nali
    "Step",              # Step down surface
    "Face",              # Flat surface
    # --- Extended 9 ---
    "Taper",             # Conical surface
    "Knurl",             # Grip pattern surface
    "Counterbore",       # Bolt head seating
    "Countersink",       # Screw head seating
    "Keyway",            # Shaft keyway slot
    "Spline",            # Splined shaft feature
    "Gear_Teeth",        # Gear tooth profile
    "Contour_3D",        # Free-form 3D surface
    "Engraved_Mark",     # Text/marking — "Engraving" operation se naam clash na ho isliye alag rakha
    # --- New 3 (added) ---
    "Turning",           # Plain cylindrical turning
    "Bore",              # Large precision bore (distinct from Hole)
    "Center_Drill",      # Explicit center drill callout
    # --- New 3 (prismatic/milled parts: bracket, block, clevis drawings) ---
    "Shoulder",          # Milled step/shoulder on a prismatic part (Step = LATHE shaft step, ye alag hai)
    "Inclined_Face",     # Flat angled/sloped face (wedge, inclined block) — Taper se alag: wo gol conical hota hai
    "Rib",               # Thin triangular/straight web joining two faces (bracket gusset)
    # --- New 11 (12 unreachable operations ko feature se jodne ke liye) ---
    # Milling
    "T_Slot",            # T-shaped groove (T-Slot Milling)
    "Dovetail",          # Dovetail slide groove (Dovetail Milling)
    "Spotface",          # Flat seat around a hole for a bolt head (Spotfacing)
    "Helical_Groove",    # Spiral groove / flute (Helical Milling)
    # Lathe
    "Parting_Off",       # Cut-off of the finished part from bar stock
    "Undercut",          # Relief groove next to a thread/shoulder
    "Contour_Turn",      # Curved/profiled turned contour (arc profile on a shaft)
    "Eccentric",         # Offset cylinder (cam-like) turned eccentrically
    # Both machines
    "Reamed_Hole",       # Precision hole (H7-type) — drill + ream
    "Internal_Thread",   # Tapped hole / internal thread (tap or thread mill)
    "Face_Groove",       # Groove cut in a flat face (lathe face-grooving or milled slot)
]

ALL_FEATURES = set(GEOMETRY_FEATURES)

# ============================================================
# 2. FEATURE -> OPERATIONS MAPPING (Dynamic Route Builder — Step 1)
# ============================================================
FEATURE_TO_OPERATIONS = {
    "Hole": {
        "machine": "Both",
        "alternatives": [
            ["Center Drilling", "Drilling"],
            ["Center Drilling", "Drilling", "Reaming"],
            ["Center Drilling", "Drilling", "Boring"],
        ],
    },
    "Slot": {
        "machine": "Milling",
        "alternatives": [
            ["Slot Milling"],
        ],
    },
    "Pocket": {
        "machine": "Milling",
        "alternatives": [
            ["Pocket Milling"],
        ],
    },
    "Boss": {
        "machine": "Both",
        "alternatives": [
            ["Profile Milling"],
            ["Step Turning"],
        ],
    },
    "Thread": {
        "machine": "Both",
        "alternatives": [
            ["External Threading"],
            ["Center Drilling", "Drilling", "Tapping"],
            ["Center Drilling", "Drilling", "Thread Milling"],
        ],
    },
    "Chamfer": {
        "machine": "Both",
        "alternatives": [
            ["Chamfering"],
        ],
    },
    "Fillet": {
        "machine": "Milling",
        "alternatives": [
            ["Corner Rounding/Filleting"],
        ],
    },
    "Groove": {
        "machine": "Lathe",
        "alternatives": [
            ["Grooving/Necking"],
            ["Internal Grooving"],
        ],
    },
    "Step": {
        "machine": "Lathe",
        "alternatives": [
            ["Step Turning"],
        ],
    },
    "Face": {
        "machine": "Both",
        "alternatives": [
            ["Facing"],
            ["Face Milling"],
        ],
    },
    "Taper": {
        "machine": "Lathe",
        "alternatives": [
            ["Taper Turning"],
        ],
    },
    "Knurl": {
        "machine": "Lathe",
        "alternatives": [
            ["Knurling"],
        ],
    },
    "Counterbore": {
        "machine": "Both",
        "alternatives": [
            ["Center Drilling", "Drilling", "Counterboring"],
        ],
    },
    "Countersink": {
        "machine": "Both",
        "alternatives": [
            ["Center Drilling", "Drilling", "Countersinking"],
        ],
    },
    "Keyway": {
        "machine": "Milling",
        "alternatives": [
            ["Woodruff Keyway Milling"],
            ["Slot Milling"],
        ],
    },
    "Spline": {
        "machine": "Milling",
        "alternatives": [
            ["Gear/Spline Milling"],
        ],
    },
    "Gear_Teeth": {
        "machine": "Milling",
        "alternatives": [
            ["Gear/Spline Milling"],
        ],
    },
    "Contour_3D": {
        "machine": "Milling",
        "alternatives": [
            ["Surface Contouring"],
            ["Profile Milling"],
        ],
    },
    "Engraved_Mark": {
        "machine": "Milling",
        "alternatives": [
            ["Engraving"],
        ],
    },
    "Turning": {
        "machine": "Lathe",
        "alternatives": [
            ["Plain/Cylindrical Turning"],
            ["Step Turning"],
        ],
    },
    "Bore": {
        "machine": "Lathe",
        "alternatives": [
            ["Boring"],
        ],
    },
    "Center_Drill": {
        "machine": "Both",
        "alternatives": [
            ["Center Drilling"],
        ],
    },
    "Shoulder": {
        "machine": "Milling",
        "alternatives": [
            ["Profile Milling"],
            ["Slab/Peripheral Milling"],
        ],
    },
    "Inclined_Face": {
        "machine": "Milling",
        "alternatives": [
            ["Angular Milling"],
        ],
    },
    "Rib": {
        "machine": "Milling",
        "alternatives": [
            ["Profile Milling"],
            ["Pocket Milling"],
        ],
    },
    "T_Slot": {
        "machine": "Milling",
        "alternatives": [
            ["T-Slot Milling"],
        ],
    },
    "Dovetail": {
        "machine": "Milling",
        "alternatives": [
            ["Dovetail Milling"],
        ],
    },
    "Spotface": {
        "machine": "Milling",
        "alternatives": [
            ["Spotfacing"],
        ],
    },
    "Helical_Groove": {
        "machine": "Milling",
        "alternatives": [
            ["Helical Milling"],
        ],
    },
    "Parting_Off": {
        "machine": "Lathe",
        "alternatives": [
            ["Parting-off"],
        ],
    },
    "Undercut": {
        "machine": "Lathe",
        "alternatives": [
            ["Undercutting"],
        ],
    },
    "Contour_Turn": {
        "machine": "Lathe",
        "alternatives": [
            ["Contour Turning"],
        ],
    },
    "Eccentric": {
        "machine": "Lathe",
        "alternatives": [
            ["Eccentric Turning"],
        ],
    },
    "Reamed_Hole": {
        "machine": "Both",
        "alternatives": [
            ["Center Drilling", "Drilling", "Reaming"],
        ],
    },
    "Internal_Thread": {
        "machine": "Both",
        "alternatives": [
            ["Center Drilling", "Drilling", "Tapping"],
            ["Center Drilling", "Drilling", "Thread Milling"],
        ],
    },
    "Face_Groove": {
        "machine": "Both",
        "alternatives": [
            ["Grooving/Necking"],
            ["Slot Milling"],
        ],
    },
}

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def is_valid_feature(feature: str) -> bool:
    """Check karta hai ki feature vocabulary me exist karta hai ya nahi."""
    return feature in ALL_FEATURES


def get_all_features() -> list:
    """Saari features ki list return karta hai."""
    return GEOMETRY_FEATURES.copy()


def get_operations_for_feature(feature: str) -> list:
    """
    Diye gaye feature ke liye saare alternative operation-sequences return karta hai.
    Route Builder isi function ko call karega (Step 1: Feature -> Operation Mapping).
    Returns: list of lists — har inner list ek valid alternative operation-chain hai.
    """
    if not is_valid_feature(feature):
        raise ValueError(
            f"'{feature}' vocabulary me nahi hai. Valid features: {get_all_features()}"
        )
    return FEATURE_TO_OPERATIONS[feature]["alternatives"]


def get_machine_type(feature: str) -> str:
    """Feature konsi machine (Lathe / Milling / Both) pe banta hai, ye batata hai."""
    if not is_valid_feature(feature):
        raise ValueError(f"'{feature}' vocabulary me nahi hai.")
    return FEATURE_TO_OPERATIONS[feature]["machine"]


def get_required_operations_set(features: list) -> set:
    """
    Diye gaye features ki list ke liye — default (pehla) alternative use karke
    saare zaroori operations ka UNION set nikalta hai. Completeness check
    (koi operation skip na ho) ke liye Route Builder isse use karega.
    """
    required = set()
    for feature in features:
        alternatives = get_operations_for_feature(feature)
        required.update(alternatives[0])  # default: pehla alternative
    return required


if __name__ == "__main__":
    print("=== Feature Vocabulary ===")
    for i, feat in enumerate(GEOMETRY_FEATURES, 1):
        default_ops = FEATURE_TO_OPERATIONS[feat]["alternatives"][0]
        machine = FEATURE_TO_OPERATIONS[feat]["machine"]
        print(f"  {i:2}. {feat:<14} [{machine:<7}] -> {default_ops}")
    print(f"\nTotal features: {len(GEOMETRY_FEATURES)}")