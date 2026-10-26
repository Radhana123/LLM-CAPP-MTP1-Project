# test_feature_vocab.py
# Pytest suite — feature_vocab.py + token_map.json ke liye
# Week 1 | LLM-CAPP Project

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import feature_vocab as fv

TOKEN_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "token_map.json")


@pytest.fixture(scope="module")
def token_map():
    with open(TOKEN_MAP_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------
# Group 1: feature_vocab.py ke andar ki basic sanity
# ---------------------------------------------------------

def test_total_feature_count():
    """Total 36 features hone chahiye (10 original + 9 extended + 3 turning/bore/center + 3 prismatic + 11 milling/lathe/both)."""
    assert len(fv.GEOMETRY_FEATURES) == 36


def test_no_duplicate_features():
    """GEOMETRY_FEATURES list me koi duplicate naam nahi hona chahiye."""
    assert len(fv.GEOMETRY_FEATURES) == len(set(fv.GEOMETRY_FEATURES))


def test_every_feature_has_operations_entry():
    """Har feature ka FEATURE_TO_OPERATIONS me entry hona chahiye."""
    for feat in fv.GEOMETRY_FEATURES:
        assert feat in fv.FEATURE_TO_OPERATIONS, f"{feat} ka mapping missing hai"


def test_every_feature_has_at_least_one_alternative():
    """Har feature ke paas kam se kam 1 valid operation-chain honi chahiye."""
    for feat in fv.GEOMETRY_FEATURES:
        alternatives = fv.get_operations_for_feature(feat)
        assert len(alternatives) >= 1, f"{feat} ke paas koi alternative nahi hai"


def test_no_empty_operation_chains():
    """Koi alternative chain empty list nahi honi chahiye."""
    for feat, data in fv.FEATURE_TO_OPERATIONS.items():
        for alt in data["alternatives"]:
            assert len(alt) >= 1, f"{feat} ki ek chain empty hai"


def test_machine_type_is_valid():
    """Har feature ka 'machine' field 'Lathe', 'Milling', ya 'Both' hi hona chahiye."""
    valid_machines = {"Lathe", "Milling", "Both"}
    for feat in fv.GEOMETRY_FEATURES:
        machine = fv.get_machine_type(feat)
        assert machine in valid_machines, f"{feat} ka machine type invalid: {machine}"


# ---------------------------------------------------------
# Group 2: Helper functions ka behavior
# ---------------------------------------------------------

def test_is_valid_feature_true_case():
    assert fv.is_valid_feature("Hole") is True


def test_is_valid_feature_false_case():
    assert fv.is_valid_feature("NotAFeature") is False


def test_invalid_feature_raises_error():
    with pytest.raises(ValueError):
        fv.get_operations_for_feature("NotAFeature")


def test_required_operations_set_merges_correctly():
    """Multiple features ke operations ek set me merge hone chahiye, duplicates ke bina."""
    result = fv.get_required_operations_set(["Hole", "Counterbore"])
    # Dono me 'Center Drilling' aur 'Drilling' common hain
    assert "Center Drilling" in result
    assert "Drilling" in result
    assert "Counterboring" in result


# ---------------------------------------------------------
# Group 3: feature_vocab.py <-> token_map.json sync
# ---------------------------------------------------------

def test_all_geometry_features_in_token_map(token_map):
    """Har GEOMETRY_FEATURES entry ka token_map.json['geometry'] me match hona chahiye."""
    missing = [f for f in fv.GEOMETRY_FEATURES if f not in token_map["geometry"]]
    assert missing == [], f"token_map.json me missing features: {missing}"


def test_all_used_operations_in_token_map(token_map):
    """feature_vocab.py me use hui har operation ka token_map.json['operations'] me entry hona chahiye."""
    used_ops = set()
    for data in fv.FEATURE_TO_OPERATIONS.values():
        for alt in data["alternatives"]:
            used_ops.update(alt)
    missing = [op for op in used_ops if op not in token_map["operations"]]
    assert missing == [], f"token_map.json me missing operations: {missing}"


def test_no_duplicate_token_ids(token_map):
    """Kisi bhi section ke across, koi do tokens same ID share nahi karne chahiye."""
    all_ids = []
    for section in token_map.values():
        all_ids.extend(section.values())
    dupes = [t for t in set(all_ids) if all_ids.count(t) > 1]
    assert dupes == [], f"Duplicate token IDs mile: {dupes}"


def test_no_name_collision_between_geometry_and_operations(token_map):
    """Geometry aur Operations me koi ek naam dono jagah nahi hona chahiye (silent overwrite bug)."""
    collision = set(token_map["geometry"].keys()) & set(token_map["operations"].keys())
    assert collision == set(), f"Naam collision mila: {collision}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])