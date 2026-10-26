import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capp import metrics as M    # noqa: E402

A = ["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"]
MK = "--- Machine Changeover ---"


def test_exact_and_lcs():
    assert M.exact_match(A, list(A)) and not M.exact_match(A, A[:-1])
    assert M.lcs_length(A, A) == 5
    assert M.lcs_length(A, ["Facing", "Drilling", "Inspection"]) == 3
    assert M.lcs_ratio(A, A) == 1.0
    assert M.lcs_ratio(A, ["Facing", "Drilling", "Inspection"]) == pytest.approx(3 / 5)
    assert M.lcs_ratio([], []) == 1.0 and M.lcs_ratio(A, []) == 0.0


def test_markers_are_ignored_everywhere():
    withm = ["Facing", MK, "Center Drilling", "Drilling", "Reaming", "Inspection"]
    assert M.exact_match(withm, A) and M.lcs_ratio(withm, A) == 1.0
    assert M.edit_distance(withm, A) == 0 and M.kendall_tau_distance(withm, A) == 0.0


def test_kendall_tau_distance():
    assert M.kendall_tau_distance(A, A) == 0.0
    assert M.kendall_tau_distance(A, list(reversed(A))) == 1.0
    # one swap among 5 ops: 1 discordant pair out of 10
    sw = ["Facing", "Drilling", "Center Drilling", "Reaming", "Inspection"]
    assert M.kendall_tau_distance(A, sw) == pytest.approx(1 / 10)
    # only common operations count
    assert M.kendall_tau_distance(A, ["Facing", "Boring", "Inspection"]) == 0.0
    assert M.kendall_tau_distance(A, ["Boring", "Tapping"]) is None
    assert M.kendall_tau_distance(A, ["Facing"]) is None


def test_edit_distance():
    assert M.edit_distance(A, A) == 0
    assert M.edit_distance(A, A[:-1]) == 1                  # one deletion
    assert M.edit_distance(A, ["Facing", "Boring", "Drilling", "Reaming", "Inspection"]) == 1   # one substitution
    assert M.edit_distance([], A) == 5 and M.edit_distance(A, []) == 5


def test_survived_share():
    assert M.survived_share(A, A) == 1.0
    assert M.survived_share(A, ["Facing", "Inspection"]) == pytest.approx(2 / 5)
    assert M.survived_share([], A) == 1.0


def test_best_similarity_takes_the_closest_reference():
    refs = [["Facing", "Drilling", "Inspection"], A]
    s = M.best_similarity(A, refs)
    assert s["exact"] is True and s["lcs_ratio"] == 1.0 and s["kendall_tau"] == 0.0
    s2 = M.best_similarity(["Facing", "Reaming", "Drilling", "Inspection"], refs)
    assert s2["exact"] is False and 0 < s2["lcs_ratio"] < 1
    assert M.best_similarity(A, []) is None
    # tau undefined for every reference -> None, not a crash
    assert M.best_similarity(["Boring"], [["Tapping"]])["kendall_tau"] is None


def test_best_known_and_quality_gap():
    scores = [
        {"time_min": 20, "cost_inr": 2000, "energy_kwh": 2.0, "efficiency_score": 70},
        {"time_min": 25, "cost_inr": 1800, "energy_kwh": 2.5, "efficiency_score": 75},
    ]
    best = M.best_known(scores)
    assert best == {"time_min": 20, "cost_inr": 1800, "energy_kwh": 2.0, "efficiency_score": 75}
    gap = M.quality_gap(scores[0], best)
    assert gap["time_min_gap"] == 0 and gap["cost_inr_gap"] == pytest.approx(200 / 1800)
    assert gap["energy_kwh_gap"] == 0 and gap["efficiency_gap"] == 5
    assert all(v >= 0 for v in M.quality_gap(scores[1], best).values())
    assert M.best_known([]) == {}


def test_summary_stats_helpers():
    assert M.mean_std([1, 2, 3]) == (2.0, 1.0)
    assert M.mean_std([None, 4]) == (4.0, 0.0)
    assert M.mean_std([None]) == (None, None)
    assert M.percentile([1, 2, 3, 4, 5], 0.5) == 3 and M.percentile([], 0.5) is None
    assert M.rate([True, False, True, True]) == 0.75 and M.rate([]) is None


def test_spearman():
    assert M.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert M.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert M.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert M.spearman([1, 1, 2, 2], [5, 5, 9, 9]) == pytest.approx(1.0)           # ties get average ranks
    assert M.spearman([1, 1, 1], [1, 2, 3]) is None and M.spearman([1], [1]) is None
