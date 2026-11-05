# material_tokens.py
# Materials aur Tolerances ki category definitions
# Task 2 — Week 1 | LLM-CAPP Project

MATERIALS = {
    "Aluminum":   "Soft metal, easy to machine, lightweight",
    "Steel":      "Hard metal, strong, needs carbide tools",
    "Brass":      "Medium hardness, good for threads",
    "Copper":     "Soft, electrical parts mein use hota hai",
    "Titanium":   "Very hard, aerospace grade, slow machining",
    "Plastic":    "ABS/POM, low-speed machining",
    "Cast Iron":  "Brittle, abrasive, needs special tools",
    # Step 55
    "Mild Steel":                  "Low-carbon steel, easy to machine, the most forgiving steel",
    "Alloy Steel":                 "Chromium / molybdenum steel (4140 type), tough, needs carbide",
    "Tool Steel":                  "High-alloy steel (A2 / D2 / O1), abrasive, slow machining",
    "Hardened Steel":              "Hardened above ~45 HRC, needs CBN / ceramic or light carbide cuts",
    "Austenitic Stainless Steel":  "304 / 316 type, work-hardens, gummy, needs sharp tools and coolant",
    "Martensitic Stainless Steel": "410 / 420 / 17-4PH type, harder than 304, moderate machinability",
    "Nodular Cast Iron":           "Ductile iron, tougher than grey iron, abrasive",
    "Bronze":                      "Copper-tin alloy, stringy chips, moderate speeds",
    "Hardened Aluminum Alloy":     "7075 / 2024 type, strong, still fast to machine",
    "Inconel":                     "Nickel super alloy, very slow machining, work-hardens, heat resistant",
    "Pure Titanium":               "Commercially pure titanium, easier than Ti-6Al-4V, still slow",
    "Magnesium":                   "Lightest structural metal, very fast to machine, chips are a fire risk",
}

TOLERANCE_CATEGORIES = {
    "0.005mm": "ultra_tight",
    "0.01mm":  "tight",
    "0.02mm":  "standard",
    "0.05mm":  "medium",
    "0.1mm":   "coarse",
    "0.5mm":   "rough",
}

def get_batch_category(batch_size: int) -> str:
    if batch_size <= 10:
        return "prototype"
    elif batch_size <= 99:
        return "small_batch"
    elif batch_size <= 999:
        return "medium_batch"
    else:
        return "mass_production"

def get_tolerance_category(tol: str) -> str:
    return TOLERANCE_CATEGORIES.get(tol, "unknown")

def is_valid_material(material: str) -> bool:
    return material in MATERIALS

if __name__ == "__main__":
    print("=== Materials ===")
    for mat, desc in MATERIALS.items():
        print(f"  {mat:12} → {desc}")

    print("\n=== Tolerance Categories ===")
    for tol, cat in TOLERANCE_CATEGORIES.items():
        print(f"  ±{tol:8} → {cat}")

    print("\n=== Batch Size Test ===")
    for size in [5, 50, 500, 5000]:
        print(f"  {size:5} units → {get_batch_category(size)}")