import copy
import json
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "week1"))

from capp import archetypes as A          # noqa: E402
from capp import datagen as G             # noqa: E402
from capp import dataset as D             # noqa: E402
import feature_vocab as fv                # noqa: E402
from material_tokens import MATERIALS, TOLERANCE_CATEGORIES   # noqa: E402
from route_builder import generate_valid_routes                # noqa: E402


@pytest.fixture(scope="module")
def dev():
    return D.load_dataset("dev_v2")


@pytest.fixture(scope="module")
def held_out():
    return D.load_dataset("test_v2")


def _good_part():
    p = {
        "part_id": "DEV_001", "split": "dev", "archetype": "Hex Bolt", "in_distribution": True,
        "material": "Steel", "features": ["Face", "Thread", "Chamfer"], "tolerance": "0.02mm",
        "batch_size": 500, "dims": {"diameter": 20, "length": 80, "depth": 8, "width": 16},
        "required_ra_um": 1.6, "machine_preference": "auto",
    }
    p.update(D.derived_fields(p))
    return p


# ---- archetypes -------------------------------------------------------------------------------------------
def test_archetypes_use_only_known_features_materials_tolerances():
    for a in A.ARCHETYPES:
        assert all(fv.is_valid_feature(f) for f in a["features"]), a["name"]
        assert len(set(a["features"])) == len(a["features"]), a["name"]
        assert set(a["materials"]) <= set(MATERIALS), a["name"]
        assert set(a["tolerances"]) <= set(TOLERANCE_CATEGORIES), a["name"]
        assert a["family"] in ("turned", "disc", "prismatic"), a["name"]
        assert a["pool"] in ("dev", "test_only"), a["name"]
    assert len({a["name"] for a in A.ARCHETYPES}) == len(A.ARCHETYPES)


def test_dev_archetypes_cover_all_36_features():
    covered = {f for a in A.dev_archetypes() for f in a["features"]}
    assert covered == set(fv.get_all_features()) and len(covered) == 36


def test_heldout_archetypes_have_feature_sets_unseen_in_dev():
    dev_sets = {frozenset(a["features"]) for a in A.dev_archetypes()}
    assert len(A.heldout_archetypes()) == 7
    for a in A.heldout_archetypes():
        assert frozenset(a["features"]) not in dev_sets, a["name"]


def test_every_archetype_has_a_complete_valid_route():
    # Otherwise a part could never be solved by the Route Builder and would distort every metric.
    for a in A.ARCHETYPES:
        assert generate_valid_routes(a["features"], max_routes=3), a["name"]


# ---- the committed data ----------------------------------------------------------------------------------
def test_committed_files_equal_regenerated_output(dev, held_out):
    assert dev == G.generate_dev()
    assert held_out == G.generate_test(dev_parts=dev)


def test_both_files_validate(dev, held_out):
    assert D.validate_dataset(dev) == []
    assert D.validate_dataset(held_out) == []


def test_dev_set_composition(dev):
    s = D.summarize(dev)
    assert s["n_parts"] == 200 and s["archetypes"] == 36
    assert s["feature_coverage"] == "36/36" and s["missing_features"] == []
    assert set(s["materials"]) == set(MATERIALS)
    assert set(s["tolerances"]) == set(TOLERANCE_CATEGORIES)
    assert set(s["batch_categories"]) == {"prototype", "small_batch", "medium_batch", "mass_production"}
    assert set(s["machine_profiles"]) == {"lathe", "milling", "mixed", "flexible"}
    assert sum(n for k, n in s["n_features_hist"].items() if k >= 6) >= 10      # enough complex parts
    assert all(p["split"] == "dev" and p["in_distribution"] for p in dev)
    assert all(p["part_id"].startswith("DEV_") for p in dev)
    # balanced: every dev archetype gets 5 or 6 parts (200 / 36)
    counts = list(__import__("collections").Counter(p["archetype"] for p in dev).values())
    assert min(counts) >= 5 and max(counts) <= 6


def test_held_out_composition_and_isolation(dev, held_out):
    assert len(held_out) == 120
    assert all(p["split"] == "test" and p["part_id"].startswith("TEST_") for p in held_out)
    in_d = [p for p in held_out if p["in_distribution"]]
    ood = [p for p in held_out if not p["in_distribution"]]
    assert len(in_d) == 60 and len(ood) == 60
    dev_sets = {frozenset(p["features"]) for p in dev}
    assert all(frozenset(p["features"]) in dev_sets for p in in_d)
    assert all(frozenset(p["features"]) not in dev_sets for p in ood)
    # no identical part in both splits
    assert not ({G._key(p) for p in dev} & {G._key(p) for p in held_out})
    assert D.feature_coverage(held_out)["missing"] == []


def test_dims_inside_ui_limits_and_ra_matches_tolerance(dev, held_out):
    for p in dev + held_out:
        for k, (lo, hi) in D.DIM_LIMITS.items():
            assert lo <= p["dims"][k] <= hi
        assert p["required_ra_um"] == D.TOLERANCE_TO_RA_UM[p["tolerance"]]


def test_ra_table_covers_every_tolerance_and_tightens_with_tolerance():
    assert set(D.TOLERANCE_TO_RA_UM) == set(TOLERANCE_CATEGORIES)
    vals = [D.TOLERANCE_TO_RA_UM[t] for t in sorted(D.TOLERANCE_TO_RA_UM, key=lambda s: float(s[:-2]))]
    assert vals == sorted(vals) and len(set(vals)) == len(vals)


# ---- generator behaviour ---------------------------------------------------------------------------------
def test_generator_is_deterministic_and_seed_sensitive():
    assert G.generate_dev(n=40, seed=1) == G.generate_dev(n=40, seed=1)
    assert G.generate_dev(n=40, seed=1) != G.generate_dev(n=40, seed=2)


def test_generator_does_not_touch_global_random_state():
    random.seed(123)
    a = random.random()
    random.seed(123)
    G.generate_dev(n=30)
    assert random.random() == a


def test_sample_dims_stay_in_family_ranges():
    rng = random.Random(0)
    for fam in ("turned", "disc", "prismatic"):
        for _ in range(500):
            d = G.sample_dims(rng, fam)
            assert set(d) == set(D.DIM_KEYS)
            for k, (lo, hi) in D.DIM_LIMITS.items():
                assert lo <= d[k] <= hi
            assert d["depth"] <= d["length"]
    with pytest.raises(ValueError):
        G.sample_dims(rng, "spherical")


# ---- validation ------------------------------------------------------------------------------------------
def test_good_part_is_valid():
    assert D.validate_part(_good_part()) == []


@pytest.mark.parametrize("mutate, fragment", [
    (lambda p: p.update(material="Diamond"), "material"),
    (lambda p: p.update(features=["Face", "Teleporter"]), "unknown features"),
    (lambda p: p.update(features=["Face", "Face"]), "duplicate"),
    (lambda p: p.update(features=[]), "non-empty"),
    (lambda p: p.update(tolerance="1mm"), "tolerance"),
    (lambda p: p.update(batch_size=0), "batch_size"),
    (lambda p: p.update(batch_size=10001), "batch_size"),
    (lambda p: p.update(batch_size=5.5), "batch_size"),
    (lambda p: p.update(batch_size=True), "batch_size"),
    (lambda p: p["dims"].pop("width"), "dims"),
    (lambda p: p["dims"].update(diameter=0), "dims.diameter"),
    (lambda p: p["dims"].update(length=5000), "dims.length"),
    (lambda p: p.update(split="val"), "split"),
    (lambda p: p.update(machine_preference="prefer_drill"), "machine_preference"),
    (lambda p: p.update(required_ra_um=0.1), "required_ra_um"),
    (lambda p: p.update(n_features=9), "n_features"),
    (lambda p: p.update(machine_profile="mixed"), "machine_profile"),
    (lambda p: p.update(reference_routes=[[]]), "reference_routes"),
    (lambda p: p.update(reference_routes="Facing"), "reference_routes"),
    (lambda p: p.pop("tolerance"), "missing field"),
])
def test_validate_part_catches_problems(mutate, fragment):
    p = copy.deepcopy(_good_part())
    mutate(p)
    errs = D.validate_part(p)
    assert errs and any(fragment in e for e in errs), errs


def test_reference_routes_are_optional_but_checked_when_present():
    p = _good_part()
    p["reference_routes"] = [["Facing", "Chamfering", "Inspection"]]
    assert D.validate_part(p) == []


def test_validate_dataset_flags_duplicate_ids():
    p = _good_part()
    errs = D.validate_dataset([p, copy.deepcopy(p)])
    assert any("duplicated" in e for e in errs)


# ---- I/O -------------------------------------------------------------------------------------------------
def test_save_and_load_roundtrip_writes_json_and_csv(tmp_path):
    parts = G.generate_dev(n=36, seed=7)
    path = D.save_dataset(parts, str(tmp_path / "mini.json"))
    assert D.load_dataset(path) == parts
    csv_lines = (tmp_path / "mini.csv").read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) == 37
    assert csv_lines[0].startswith("part_id,split,archetype") and "diameter" in csv_lines[0]


def test_load_dataset_rejects_corrupted_file(tmp_path):
    parts = G.generate_dev(n=36, seed=7)
    parts[0]["material"] = "Diamond"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(parts), encoding="utf-8")
    with pytest.raises(ValueError):
        D.load_dataset(str(path))
    assert D.load_dataset(str(path), validate=False)[0]["material"] == "Diamond"


def test_coverage_helper_reports_missing_features():
    cov = D.feature_coverage([_good_part()])
    assert cov["covered"] == 3 and cov["total"] == 36 and "Hole" in cov["missing"]
