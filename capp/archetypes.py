"""Part archetypes for dataset v2.

An archetype fixes the feature set (what really occurs together on a real part) and the ranges for material,
tolerance, batch size and size family. The generator (capp/datagen.py) samples concrete parts from it.

pool = "dev"        used for the 200-part development set; also the retrieval pool for few-shot examples (WP4).
pool = "test_only"  feature combinations that never appear in the dev set. They go into the held-out test set only,
                    so the final numbers also measure generalisation to unseen part types, and retrieval cannot
                    simply copy a near-identical dev example.

family = "turned"     round bar stock, lathe-led  (diameter, length are the main sizes)
         "disc"       short, large-diameter parts (diameter large, length = thickness)
         "prismatic"  block / plate / bracket, mill-led (width, length, depth are the main sizes)

The first 19 entries are the v1 archetypes (same features, materials, tolerances, batches) so v1 results stay
comparable; the v1 set only covered 19 of the 36 features, the rest are new.
"""

_V1 = [
    ("Hex Bolt", "turned", ["Face", "Thread", "Chamfer"],
     ["Steel", "Titanium", "Brass"], ["0.02mm", "0.05mm", "0.1mm"], [500, 1000, 2500, 5000]),
    ("Keyed Shaft", "turned", ["Step", "Keyway", "Chamfer", "Face"],
     ["Steel", "Aluminum", "Cast Iron"], ["0.01mm", "0.02mm"], [10, 50, 100]),
    ("Flanged Bushing", "turned", ["Hole", "Face", "Chamfer", "Counterbore"],
     ["Brass", "Aluminum", "Copper"], ["0.02mm", "0.05mm"], [50, 100, 250]),
    ("Threaded Shaft with Relief Groove", "turned", ["Thread", "Fillet", "Step"],
     ["Steel", "Titanium"], ["0.01mm", "0.02mm"], [10, 50, 100]),
    ("Gear Blank", "disc", ["Gear_Teeth", "Hole", "Keyway", "Face"],
     ["Steel", "Cast Iron"], ["0.005mm", "0.01mm"], [10, 50]),
    ("V-Belt Pulley", "disc", ["Groove", "Hole", "Keyway", "Face"],
     ["Cast Iron", "Aluminum", "Steel"], ["0.02mm", "0.05mm"], [50, 100, 250]),
    ("Flange Plate", "disc", ["Hole", "Counterbore", "Countersink", "Chamfer", "Face"],
     ["Steel", "Aluminum", "Cast Iron"], ["0.05mm", "0.1mm"], [100, 250, 500]),
    ("Mounting Bracket", "prismatic", ["Slot", "Hole", "Chamfer", "Face"],
     ["Aluminum", "Steel"], ["0.05mm", "0.1mm", "0.5mm"], [250, 500, 1000]),
    ("Tapered Sleeve", "turned", ["Taper", "Hole", "Chamfer"],
     ["Brass", "Aluminum", "Steel"], ["0.02mm", "0.05mm"], [50, 100]),
    ("Knurled Grip Handle", "turned", ["Knurl", "Step", "Chamfer", "Face"],
     ["Aluminum", "Steel", "Plastic"], ["0.05mm", "0.1mm"], [100, 250, 500]),
    ("Pocketed Housing", "prismatic", ["Pocket", "Hole", "Counterbore", "Chamfer"],
     ["Aluminum", "Steel"], ["0.02mm", "0.05mm"], [10, 50, 100]),
    ("Splined Shaft", "turned", ["Spline", "Step", "Fillet", "Chamfer"],
     ["Steel", "Titanium"], ["0.005mm", "0.01mm"], [10, 50]),
    ("Cam Profile Disc", "disc", ["Contour_3D", "Hole", "Keyway"],
     ["Steel", "Cast Iron"], ["0.01mm", "0.02mm"], [10, 50]),
    ("Engraved ID Plate", "prismatic", ["Engraved_Mark", "Face", "Hole", "Chamfer"],
     ["Aluminum", "Brass", "Plastic"], ["0.1mm", "0.5mm"], [50, 100, 250]),
    ("Precision Dowel Pin", "turned", ["Chamfer", "Face"],
     ["Steel", "Titanium"], ["0.005mm", "0.01mm"], [500, 1000, 2500]),
    ("Stepped Shaft with Retaining Groove", "turned", ["Step", "Groove", "Fillet", "Chamfer"],
     ["Steel", "Aluminum"], ["0.01mm", "0.02mm"], [10, 50, 100]),
    ("Internally Threaded Bushing", "turned", ["Thread", "Hole", "Chamfer", "Face"],
     ["Brass", "Steel", "Copper"], ["0.02mm", "0.05mm"], [100, 250, 500]),
    ("Boss with Spotface", "disc", ["Boss", "Counterbore", "Hole"],
     ["Aluminum", "Steel", "Cast Iron"], ["0.02mm", "0.05mm"], [50, 100, 250]),
    ("Sealed Housing Cover", "disc", ["Groove", "Face", "Hole", "Chamfer"],
     ["Aluminum", "Steel", "Cast Iron"], ["0.05mm", "0.1mm"], [100, 250, 500]),
]

# New dev archetypes: bring the 17 features v1 never used into the set, plus three complex (7-8 feature) parts.
_NEW_DEV = [
    ("Turned Pin with Undercut", "turned", ["Turning", "Undercut", "Thread", "Chamfer", "Parting_Off"],
     ["Steel", "Brass", "Aluminum"], ["0.02mm", "0.05mm"], [1, 20, 100, 500]),
    ("Eccentric Cam Shaft", "turned", ["Eccentric", "Turning", "Step", "Keyway", "Chamfer"],
     ["Steel", "Cast Iron"], ["0.01mm", "0.02mm"], [5, 10, 50]),
    ("Bored Bearing Sleeve", "turned", ["Bore", "Center_Drill", "Turning", "Chamfer", "Face"],
     ["Steel", "Brass", "Cast Iron"], ["0.005mm", "0.01mm", "0.02mm"], [20, 50, 250]),
    ("Contoured Spindle Nose", "turned", ["Contour_Turn", "Turning", "Step", "Chamfer"],
     ["Steel", "Aluminum", "Titanium"], ["0.01mm", "0.02mm"], [1, 10, 50]),
    ("Wedge Clamp Block", "prismatic", ["Inclined_Face", "Hole", "Face", "Chamfer"],
     ["Steel", "Aluminum", "Cast Iron"], ["0.05mm", "0.1mm"], [20, 100, 500]),
    ("Gusseted Bracket", "prismatic", ["Shoulder", "Rib", "Hole", "Face", "Chamfer"],
     ["Aluminum", "Steel", "Cast Iron"], ["0.05mm", "0.1mm", "0.5mm"], [50, 250, 1000]),
    ("T-Slot Table Plate", "prismatic", ["T_Slot", "Face", "Hole", "Chamfer"],
     ["Cast Iron", "Steel", "Aluminum"], ["0.02mm", "0.05mm"], [1, 5, 20]),
    ("Dovetail Slide Block", "prismatic", ["Dovetail", "Face", "Shoulder", "Hole"],
     ["Steel", "Cast Iron", "Brass"], ["0.01mm", "0.02mm"], [5, 20, 100]),
    ("Spotfaced Mounting Flange", "disc", ["Spotface", "Reamed_Hole", "Hole", "Face"],
     ["Steel", "Aluminum", "Cast Iron"], ["0.02mm", "0.05mm"], [50, 250, 500]),
    ("Helical Flute Shaft", "turned", ["Helical_Groove", "Turning", "Step", "Chamfer"],
     ["Steel", "Aluminum", "Brass"], ["0.02mm", "0.05mm"], [5, 20, 100]),
    ("Precision Reamed Housing", "disc", ["Reamed_Hole", "Bore", "Face", "Face_Groove"],
     ["Aluminum", "Cast Iron", "Steel"], ["0.005mm", "0.01mm", "0.02mm"], [10, 50, 100]),
    ("Tapped Manifold Block", "prismatic", ["Internal_Thread", "Pocket", "Face", "Hole", "Chamfer"],
     ["Aluminum", "Brass", "Steel"], ["0.02mm", "0.05mm"], [10, 50, 250]),
    ("O-Ring Face Seal Cap", "disc", ["Face_Groove", "Face", "Hole", "Center_Drill"],
     ["Aluminum", "Steel", "Plastic"], ["0.02mm", "0.05mm"], [100, 500, 2500]),
    ("Center-Drilled Stud", "turned", ["Center_Drill", "Thread", "Turning", "Chamfer"],
     ["Steel", "Brass", "Titanium"], ["0.02mm", "0.05mm", "0.1mm"], [500, 1000, 5000]),
    # complex parts (7-8 features): where an LLM planner is most likely to drop an operation or break an order
    ("Gearbox Housing Half", "prismatic",
     ["Pocket", "Hole", "Counterbore", "Face", "Internal_Thread", "Spotface", "Reamed_Hole", "Chamfer"],
     ["Aluminum", "Cast Iron"], ["0.01mm", "0.02mm", "0.05mm"], [1, 10, 50]),
    ("Multi-Feature Drive Shaft", "turned",
     ["Turning", "Step", "Groove", "Thread", "Keyway", "Chamfer", "Undercut"],
     ["Steel", "Titanium"], ["0.005mm", "0.01mm", "0.02mm"], [1, 5, 10, 50]),
    ("Valve Body", "disc",
     ["Bore", "Hole", "Counterbore", "Internal_Thread", "Face_Groove", "Face", "Slot", "Chamfer"],
     ["Brass", "Steel", "Cast Iron"], ["0.01mm", "0.02mm", "0.05mm"], [10, 50, 250]),
]

# Held-out archetypes: none of these feature sets occurs in the dev pool (checked in tests).
_TEST_ONLY = [
    ("Cam Lobe Hub", "disc", ["Eccentric", "Spline", "Bore", "Chamfer"],
     ["Steel", "Cast Iron"], ["0.005mm", "0.01mm"], [5, 20]),
    ("Clevis Fork", "prismatic", ["Slot", "Shoulder", "Reamed_Hole", "Rib", "Face"],
     ["Steel", "Aluminum"], ["0.02mm", "0.05mm"], [10, 100, 500]),
    ("Taper Shank Holder", "turned", ["Taper", "Undercut", "Internal_Thread", "Face", "Chamfer"],
     ["Steel", "Titanium"], ["0.005mm", "0.01mm"], [10, 50]),
    ("Slide Guide Rail", "prismatic", ["Dovetail", "T_Slot", "Face", "Hole", "Countersink"],
     ["Steel", "Cast Iron"], ["0.01mm", "0.02mm"], [1, 5, 20]),
    ("Hydraulic Piston", "turned", ["Bore", "Face_Groove", "Groove", "Contour_Turn", "Chamfer", "Fillet"],
     ["Steel", "Aluminum"], ["0.005mm", "0.01mm"], [10, 50, 250]),
    ("Engraved Dial Knob", "turned", ["Knurl", "Engraved_Mark", "Turning", "Counterbore", "Internal_Thread"],
     ["Aluminum", "Brass", "Plastic"], ["0.05mm", "0.1mm"], [100, 500, 2500]),
    ("Helical Hub Pulley", "disc", ["Helical_Groove", "Groove", "Keyway", "Bore", "Face"],
     ["Cast Iron", "Aluminum", "Steel"], ["0.02mm", "0.05mm"], [20, 100, 250]),
]


def _build(rows, pool):
    return [
        {"name": n, "family": fam, "features": list(f), "materials": list(m),
         "tolerances": list(t), "batches": list(b), "pool": pool}
        for (n, fam, f, m, t, b) in rows
    ]


ARCHETYPES = _build(_V1, "dev") + _build(_NEW_DEV, "dev") + _build(_TEST_ONLY, "test_only")


def dev_archetypes():
    return [a for a in ARCHETYPES if a["pool"] == "dev"]


def heldout_archetypes():
    return [a for a in ARCHETYPES if a["pool"] == "test_only"]
