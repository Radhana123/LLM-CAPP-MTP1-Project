"""Deterministic generator for dataset v2 (dev set and held-out test set).

  generate_dev()   200 parts from the 36 dev archetypes, balanced per archetype  -> data/dev_v2.json
  generate_test()  120 parts: 60 from the dev archetypes (in-distribution) + 60 from the 7 held-out archetypes
                   (feature combinations never seen in dev)                       -> data/test_v2.json

The test set is sealed: do not tune prompts, rules or thresholds on it. It is used for the final runs only (WP11).
Each split uses its own random.Random(seed), so output never depends on global random state or call order.
"""
import random

from capp import archetypes as A
from capp import dataset as D

DEV_SEED = 42
TEST_SEED = 2026

# Size family -> how dims are sampled (integers in mm). Keys follow the Streamlit app: the Time agent reads these.
#   turned     D 10-60, L 40-250, depth 4..0.6*D (max 40), width 0.8*D (the agent's own default ratio)
#   disc       D 60-220, L (thickness) 15-60, depth 5..0.8*L (max 40), width 20..0.6*D
#   prismatic  W 30-160, L W..300, depth 10..min(60, L), D (hole / cutter size) 6-32


def sample_dims(rng: random.Random, family: str) -> dict:
    if family == "turned":
        d = rng.randint(10, 60)
        length = rng.randint(40, 250)
        depth = rng.randint(4, min(40, max(5, int(0.6 * d))))
        width = int(round(0.8 * d))
    elif family == "disc":
        d = rng.randint(60, 220)
        length = rng.randint(15, 60)
        depth = rng.randint(5, min(40, int(0.8 * length)))
        width = rng.randint(20, int(0.6 * d))
    elif family == "prismatic":
        width = rng.randint(30, 160)
        length = rng.randint(max(50, width), 300)
        depth = rng.randint(10, min(60, length))     # thickness / pocket depth never exceeds the length
        d = rng.randint(6, 32)
    else:
        raise ValueError(f"unknown family {family!r}")
    return {"diameter": d, "length": length, "depth": depth, "width": width}


def _key(part: dict) -> tuple:
    return (part["archetype"], part["material"], part["tolerance"], part["batch_size"],
            tuple(part["dims"][k] for k in D.DIM_KEYS))


def _sample_part(rng, arch, split, in_distribution, seen) -> dict:
    part = None
    for _ in range(100):          # resample on an (extremely unlikely) exact duplicate
        part = {
            "part_id": "",
            "split": split,
            "archetype": arch["name"],
            "in_distribution": in_distribution,
            "material": rng.choice(arch["materials"]),
            "features": list(arch["features"]),
            "tolerance": rng.choice(arch["tolerances"]),
            "batch_size": rng.choice(arch["batches"]),
            "dims": sample_dims(rng, arch["family"]),
            "machine_preference": "auto",
        }
        if _key(part) not in seen:
            break
    seen.add(_key(part))
    part["required_ra_um"] = D.required_ra(part["tolerance"])
    part.update(D.derived_fields(part))
    return part


def _allocate(rng, archs, n):
    """Balanced counts per archetype; the remainder goes to distinct randomly chosen archetypes."""
    per, extra = divmod(n, len(archs))
    chosen = [a for a in archs for _ in range(per)]
    if extra:
        chosen += rng.sample(archs, extra)
    return chosen


def _finish(parts, rng, prefix):
    rng.shuffle(parts)
    for i, p in enumerate(parts, 1):
        p["part_id"] = f"{prefix}_{i:03d}"
    return parts


def generate_dev(n: int = 200, seed: int = DEV_SEED) -> list:
    rng = random.Random(seed)
    seen = set()
    parts = [_sample_part(rng, a, "dev", True, seen) for a in _allocate(rng, A.dev_archetypes(), n)]
    return _finish(parts, rng, "DEV")


def generate_test(n_in: int = 60, n_ood: int = 60, seed: int = TEST_SEED, dev_parts: list = None) -> list:
    """Held-out test set. `dev_parts` (default: regenerated dev set) is used to guarantee no part is shared with dev."""
    rng = random.Random(seed)
    dev_parts = generate_dev() if dev_parts is None else dev_parts
    seen = {_key(p) for p in dev_parts}
    parts = [_sample_part(rng, a, "test", True, seen) for a in _allocate(rng, A.dev_archetypes(), n_in)]
    parts += [_sample_part(rng, a, "test", False, seen) for a in _allocate(rng, A.heldout_archetypes(), n_ood)]
    return _finish(parts, rng, "TEST")
