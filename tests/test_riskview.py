"""Step 21: risk-aware comparison of two runs (capp/riskview.py, scripts/risk_compare.py).

Two kinds of test: pair rows written by hand, so that every number was computed on paper first, and two real runs of the harness with
the deterministic messy planner of tests/test_partdiff.py (no API call), checked against partdiff and against themselves.
"""
import csv
import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capp import dataset, harness, partdiff as PD, riskview as RV      # noqa: E402
import risk_compare                                                      # noqa: E402
import test_partdiff as TP                                               # noqa: E402


# ---------------------------------------------------------------- pair rows by hand
def row(pid, eff, base, other, rules_b="", rules_o="", extra=(), missing=(), repeat=0, over=(False, False), ra=("low", "low")):
    """A pair row with the fields riskview reads. base / other = (hard infeasibilities, route risks); None = not analysed."""
    r = {"part_id": pid, "repeat": repeat, "eff_diff": eff, "cause": "other has extra operations" if extra else "identical",
         "only_other": list(extra), "only_base": list(missing), "route_base": "Facing > Drilling", "route_other": "Facing > Drilling",
         "feas_rules_base": rules_b, "feas_rules_other": rules_o,
         "over_power_base": over[0], "over_power_other": over[1], "ra_risk_base": ra[0], "ra_risk_other": ra[1]}
    for side, v in (("base", base), ("other", other)):
        r["feas_infeasible_" + side], r["feas_risk_" + side] = ("", "") if v is None else v
    return r


# Six pairs worked out on paper (flags = hard + risks), plus two that must be left out.
#   pair  Efficiency  flags A -> B   rules A                  rules B       B adds
#   P1      -6         1 -> 0        TOL-HOLE                -             Reaming
#   P2      -2         1 -> 0        PHYS:HOLE-PILOT         -             Center Drilling
#   P3      -4         0 -> 0        -                       -             -
#   P4       0         0 -> 0        -                       -             -
#   P5      +3         0 -> 1        -                       TOL-HOLE      -
#   P6      +1         2 -> 1        PHYS:X;TOL-HOLE         TOL-HOLE      -
#   P7: not comparable (eff None).  P8: comparable, but the runs did not analyse feasibility.
# mean Efficiency -8/6 = -1.333; flags A 4/6, B 2/6; mean reduction 2/6; break-even P* = (8/6) / (2/6) = 4
def hand_rows():
    return [row("P1", -6.0, (0, 1), (0, 0), "TOL-HOLE", "", extra=["Reaming"]),
            row("P2", -2.0, (0, 1), (0, 0), "PHYS:HOLE-PILOT", "", extra=["Center Drilling"]),
            row("P3", -4.0, (0, 0), (0, 0)),
            row("P4", 0.0, (0, 0), (0, 0)),
            row("P5", 3.0, (0, 0), (0, 1), "", "TOL-HOLE"),
            row("P6", 1.0, (1, 1), (0, 1), "PHYS:X;TOL-HOLE", "TOL-HOLE"),
            row("P7", None, (0, 0), (0, 0)),
            row("P8", -1.0, None, None)]


# ---------------------------------------------------------------- flags of a route
def test_flags_are_hard_infeasibilities_plus_route_risks():
    r = row("P", 0.0, (2, 3), (0, 1))
    assert RV.route_flags(r, "base") == 5 and RV.route_flags(r, "other") == 1


def test_a_route_that_was_not_analysed_has_no_flags_number():
    r = row("P", 0.0, None, (0, 0))
    assert RV.route_flags(r, "base") is None and RV.route_flags(r, "other") == 0
    assert RV.route_flags({"feas_infeasible_base": 1}, "base") is None             # a field missing altogether
    assert RV.route_flags({"feas_infeasible_base": True, "feas_risk_base": 1}, "base") is None   # a bool is not a count


def test_physics_adds_over_power_and_high_ra_risk_only_when_asked():
    r = row("P", 0.0, (0, 1), (0, 0), over=(True, False), ra=("high", "medium"))
    assert RV.route_flags(r, "base") == 1 and RV.route_flags(r, "other") == 0
    assert RV.route_flags(r, "base", physics=True) == 3                              # 1 risk + over power + Ra high
    assert RV.route_flags(r, "other", physics=True) == 0                             # medium Ra risk is not counted
    assert RV.route_flags(row("P", 0.0, (0, 0), (0, 0), over=("", ""), ra=("", "")), "base", physics=True) is None


# ---------------------------------------------------------------- which pairs are analysed
def test_pairs_not_comparable_or_without_feasibility_are_left_out_and_counted():
    pairs, counts = RV.pairs_for_analysis(hand_rows())
    assert [p["part_id"] for p in pairs] == ["P1", "P2", "P3", "P4", "P5", "P6"]
    assert counts == {"comparable": 7, "no_risk_data": 1}
    p = pairs[5]
    assert (p["flags_base"], p["flags_other"], p["reduction"]) == (2, 1, 1)
    assert p["rules_cleared"] == ["PHYS:X"] and p["rules_added"] == [] and p["rules_base"] == ["PHYS:X", "TOL-HOLE"]


def test_physics_flags_change_which_pairs_have_data():
    rows = [row("A", -1.0, (0, 0), (0, 0), over=("", ""), ra=("", ""))]
    assert RV.analyze(rows)["n_pairs"] == 1 and RV.analyze(rows, physics=True)["n_pairs"] == 0


# ---------------------------------------------------------------- the analysis, by hand
@pytest.fixture(scope="module")
def hand():
    return RV.analyze(hand_rows())


def test_overall_numbers(hand):
    assert (hand["n_rows"], hand["n_comparable"], hand["n_no_risk_data"], hand["n_pairs"], hand["parts"]) == (8, 7, 1, 6, 6)
    o = hand["overall"]
    assert o["mean_eff"] == pytest.approx(-8 / 6) and o["mean_reduction"] == pytest.approx(2 / 6)
    assert (o["better"], o["equal"], o["worse"]) == (2, 1, 3)
    assert hand["flags_base"] == pytest.approx(4 / 6) and hand["flags_other"] == pytest.approx(2 / 6)


def test_the_efficiency_flags_table(hand):
    q = hand["quadrants"]
    counts = {k: v["n"] for k, v in q.items()}
    assert counts[("worse", "fewer")] == 2 and counts[("worse", "same")] == 1 and counts[("equal", "same")] == 1
    assert counts[("better", "more")] == 1 and counts[("better", "fewer")] == 1
    assert sum(counts.values()) == 6 and sum(1 for v in counts.values() if v) == 5
    assert q[("worse", "fewer")]["mean_eff"] == pytest.approx(-4.0) and q[("better", "more")]["mean_eff"] == pytest.approx(3.0)
    assert q[("worse", "more")]["mean_eff"] is None                                  # an empty cell has no mean


def test_what_the_lost_efficiency_bought(hand):
    w = hand["worse_fewer"]
    assert w["n"] == 2 and w["mean_eff"] == pytest.approx(-4.0) and w["mean_reduction"] == pytest.approx(1.0)
    assert dict(w["rules_cleared"]) == {"TOL-HOLE": 1, "PHYS:HOLE-PILOT": 1} and w["rules_added"] == []
    assert dict(w["ops_extra"]) == {"Reaming": 1, "Center Drilling": 1}
    assert [p["part_id"] for p in w["pairs"]] == ["P1", "P2"]                          # worst Efficiency first
    b = hand["better_more"]
    assert b["n"] == 1 and b["mean_eff"] == pytest.approx(3.0) and b["rules_added"] == [("TOL-HOLE", 1)] and b["rules_cleared"] == []


def test_pairs_split_by_which_side_has_flags(hand):
    sp = hand["split"]
    assert {k: v["n"] for k, v in sp.items()} == {"neither": 2, "only base": 2, "only other": 1, "both": 1}
    assert sp["neither"]["mean_eff"] == pytest.approx(-2.0) and (sp["neither"]["worse"], sp["neither"]["equal"]) == (1, 1)
    assert sp["only base"]["mean_eff"] == pytest.approx(-4.0) and sp["only base"]["worse"] == 2
    assert sp["only other"]["mean_eff"] == pytest.approx(3.0) and sp["both"]["mean_eff"] == pytest.approx(1.0)
    assert sum(v["n"] for v in sp.values()) == hand["n_pairs"]


def test_the_penalty_table(hand):
    # adjusted = Efficiency + P x reduction: P1 -6+P, P2 -2+P, P3 -4, P4 0, P5 3-P, P6 1+P
    expect = {0: (-8 / 6, 2, 1, 3), 1: (-6 / 6, 2, 1, 3), 2: (-4 / 6, 2, 2, 2), 5: (2 / 6, 2, 1, 3), 10: (12 / 6, 3, 1, 2)}
    assert [r["penalty"] for r in hand["penalties"]] == [0.0, 1.0, 2.0, 5.0, 10.0]
    for r in hand["penalties"]:
        mean, better, equal, worse = expect[int(r["penalty"])]
        assert r["mean"] == pytest.approx(mean) and (r["better"], r["equal"], r["worse"]) == (better, equal, worse), r["penalty"]


def test_the_break_even_penalty_closes_the_gap_exactly(hand):
    be = hand["break_even"]
    assert be["status"] == "value" and be["value"] == pytest.approx(4.0)
    adjusted = [p["eff_diff"] + be["value"] * p["reduction"] for p in hand["pairs"]]
    assert sum(adjusted) / len(adjusted) == pytest.approx(0.0, abs=1e-9)


def test_the_per_rule_table(hand):
    rules = {r["rule"]: r for r in hand["rules"]}
    assert [r["rule"] for r in hand["rules"]] == ["TOL-HOLE", "PHYS:HOLE-PILOT", "PHYS:X"]       # most cleared + added first
    t = rules["TOL-HOLE"]
    assert (t["base_has"], t["other_has"], t["cleared"], t["added"]) == (2, 2, 1, 1)             # P6 has it on both sides: neither
    assert t["mean_eff_when_cleared"] == pytest.approx(-6.0) and t["mean_eff_when_added"] == pytest.approx(3.0)
    assert rules["PHYS:X"]["cleared"] == 1 and rules["PHYS:X"]["mean_eff_when_cleared"] == pytest.approx(1.0)
    assert rules["PHYS:HOLE-PILOT"]["added"] == 0 and rules["PHYS:HOLE-PILOT"]["mean_eff_when_added"] is None


def test_custom_penalties(hand):
    s = RV.analyze(hand_rows(), penalties=(0.5, 4.0))
    assert [r["penalty"] for r in s["penalties"]] == [0.5, 4.0] and s["penalties"][1]["mean"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- break-even in the special cases
def test_break_even_cases():
    assert RV.break_even(-2.0, 0.5) == (pytest.approx(4.0), "value")
    assert RV.break_even(0.5, 0.2) == (0.0, "not worse")
    assert RV.break_even(0.0, 0.0) == (0.0, "not worse")
    assert RV.break_even(-2.0, 0.0) == (math.inf, "never")
    assert RV.break_even(-2.0, -1.0) == (math.inf, "never")                          # B has MORE flags: no P can close the gap
    assert RV.break_even(None, 0.3) == (None, "none") and RV.break_even(-1.0, None) == (None, "none")


def test_b_with_more_flags_and_lower_efficiency_has_no_break_even():
    rows = [row("A", -3.0, (0, 0), (0, 1)), row("B", -1.0, (0, 0), (0, 2)), row("C", 0.0, (0, 1), (0, 1))]
    s = RV.analyze(rows)
    assert s["break_even"]["status"] == "never" and s["break_even"]["value"] == math.inf
    assert "none" in RV.render(s, "rb", "llm") and "does not close the gap" in RV.render(s, "rb", "llm")


def test_when_b_is_not_worse_the_report_says_no_break_even_is_needed():
    rows = [row("A", 2.0, (0, 1), (0, 0)), row("B", 0.5, (0, 0), (0, 0))]
    s = RV.analyze(rows)
    assert s["break_even"]["status"] == "not worse"
    assert "not needed" in RV.render(s, "rb", "llm")


# ---------------------------------------------------------------- intervals
def test_the_intervals_resample_parts_not_pairs():
    """Part A has three repeats at -10, part B one repeat at 0. Resampling PARTS can only give the means -10 (A, A), -7.5 (A, B) and 0 (B, B);
    resampling the four pairs would also give -5 and -2.5."""
    rows = [row("A", -10.0, (0, 0), (0, 0), repeat=i) for i in range(3)] + [row("B", 0.0, (0, 0), (0, 0))]
    s = RV.analyze(rows, penalties=(0.0,), resamples=400)
    lo, hi = s["penalties"][0]["ci95"]
    assert lo == -10.0 and hi == 0.0
    assert s["parts"] == 2 and s["n_pairs"] == 4
    for x in RV._bootstrap(s["pairs"], (0.0,), 200, 3)[0][0.0]:
        assert x in (-10.0, -7.5, 0.0)


def test_the_intervals_are_reproducible_and_contain_the_estimates(hand):
    again = RV.analyze(hand_rows())
    for a, b in zip(hand["penalties"], again["penalties"]):
        assert a["ci95"] == b["ci95"]
        lo, hi = a["ci95"]
        assert lo <= a["mean"] <= hi
    assert hand["break_even"]["ci95"] == again["break_even"]["ci95"]
    other_seed = RV.analyze(hand_rows(), seed=99)
    assert [r["ci95"] for r in other_seed["penalties"]] != [r["ci95"] for r in hand["penalties"]]


def test_one_part_has_no_interval_but_a_report():
    s = RV.analyze([row("A", -2.0, (0, 1), (0, 0)), row("A", -4.0, (0, 1), (0, 0), repeat=1)])
    assert all(r["ci95"] is None for r in s["penalties"]) and s["break_even"]["ci95"] is None
    text = RV.render(s, "rb", "llm")
    assert "P = 0" in text and "Break-even P* = 3.00" in text


# ---------------------------------------------------------------- physics flags
def test_physics_flags_enter_the_analysis():
    rows = [row("A", -5.0, (0, 0), (0, 0), over=(True, False), ra=("low", "low")),
            row("B", -1.0, (0, 0), (0, 0), over=(False, False), ra=("high", "low"))]
    plain, phys = RV.analyze(rows), RV.analyze(rows, physics=True)
    assert plain["overall"]["mean_reduction"] == 0 and plain["break_even"]["status"] == "never"
    assert phys["overall"]["mean_reduction"] == pytest.approx(1.0) and phys["break_even"]["value"] == pytest.approx(3.0)
    assert "over power" in RV.render(phys, "rb", "llm") and "over power" not in RV.render(plain, "rb", "llm")


# ---------------------------------------------------------------- nothing to compare
def test_no_rows():
    s = RV.analyze([])
    assert s["n_pairs"] == 0 and "overall" not in s
    assert "Nothing to compare" in RV.render(s)


def test_no_comparable_pair_and_no_feasibility_data_say_why():
    s = RV.analyze([row("A", None, (0, 0), (0, 0))])
    assert "no pair is comparable" in RV.render(s)
    s = RV.analyze([row("A", -1.0, None, None)])
    assert "feasibility was not analysed" in RV.render(s) and s["n_no_risk_data"] == 1


def test_writing_a_csv_of_nothing_is_an_error(tmp_path):
    with pytest.raises(RV.RiskViewError):
        RV.write_csv(str(tmp_path / "x.csv"), RV.analyze([]))


# ---------------------------------------------------------------- the report
def test_the_report_names_the_runs_and_the_pairs(hand):
    text = RV.render(hand, "rb", "n5_r1_s3")
    assert "n5_r1_s3 minus rb" in text and "8 pairs, 7 comparable" in text and "6 with feasibility data" in text and "1 left out" in text
    assert "Mean Efficiency difference -1.33 pts over 6 pairs (6 parts)" in text
    assert "worse on Efficiency but with fewer flags: 2 pair(s)" in text
    assert "rules cleared: TOL-HOLE x1, PHYS:HOLE-PILOT x1" in text and "operations n5_r1_s3 adds: Reaming x1, Center Drilling x1" in text
    assert "P1/0" in text and "flags 1 -> 0" in text and "cleared TOL-HOLE" in text and "+Reaming" in text
    assert "better on Efficiency but with more flags: 1 pair(s)" in text and "added TOL-HOLE" in text
    assert "Break-even P* = 4.00" in text and "P = 10" in text
    assert "value judgement" in text and "indicative" in text


def test_top_limits_the_listed_pairs():
    rows = [row(f"P{i}", -1.0 - i, (0, 1), (0, 0)) for i in range(6)]
    text = RV.render(RV.analyze(rows), "rb", "llm", top=2)
    assert text.count("flags 1 -> 0") == 2 and "4 more" in text
    assert "P5/0" in text and "P0/0" not in text                                      # worst first


def test_csv_has_one_row_per_pair(hand, tmp_path):
    path = str(tmp_path / "pairs.csv")
    RV.write_csv(path, hand)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [r["part_id"] for r in rows] == ["P1", "P2", "P3", "P4", "P5", "P6"]
    assert rows[0]["rules_cleared"] == "TOL-HOLE" and rows[0]["ops_extra"] == "Reaming" and rows[0]["reduction"] == "1"
    assert rows[5]["flags_base"] == "2" and rows[5]["flags_other"] == "1"


# ---------------------------------------------------------------- mirror
def mirror(r):
    m = dict(r)
    for k in ("feas_infeasible", "feas_risk", "feas_rules", "over_power", "ra_risk"):
        m[k + "_base"], m[k + "_other"] = r[k + "_other"], r[k + "_base"]
    m["only_base"], m["only_other"] = r["only_other"], r["only_base"]
    m["eff_diff"] = None if r["eff_diff"] is None else -r["eff_diff"]
    return m


def test_swapping_the_runs_flips_efficiency_and_flags(hand):
    s = RV.analyze([mirror(r) for r in hand_rows()])
    assert s["overall"]["mean_eff"] == pytest.approx(-hand["overall"]["mean_eff"])
    assert s["overall"]["mean_reduction"] == pytest.approx(-hand["overall"]["mean_reduction"])
    assert s["flags_base"] == pytest.approx(hand["flags_other"]) and s["flags_other"] == pytest.approx(hand["flags_base"])
    assert s["worse_fewer"]["n"] == hand["better_more"]["n"] == 1                       # the old "better, more flags" pair is now "worse, fewer"
    assert dict(s["worse_fewer"]["rules_cleared"]) == {"TOL-HOLE": 1}


# ---------------------------------------------------------------- two real runs
@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("runs"))
    parts = dataset.load_dataset("dev_v2")[:20]
    dirs = {}
    for system in ("rb", "n5_r1_s3", "n5_r1_s3+fz2"):
        harness.run_experiment(system, parts, tag="t", out_dir=out, progress=None, planner=None if system == "rb" else TP.messy_planner)
        dirs[system] = os.path.join(out, "t_" + system)
    return {"dirs": dirs, "recs": {k: harness.load_records(v) for k, v in dirs.items()}}


def test_agrees_with_part_diff_on_the_pairs_and_the_means(runs):
    rows, _ = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["n5_r1_s3+fz2"])
    s = RV.analyze(rows)
    summ = PD.summarize(rows)
    assert s["n_comparable"] == summ["n_comparable"] and s["n_pairs"] == s["n_comparable"] - s["n_no_risk_data"]
    assert s["overall"]["mean_eff"] == pytest.approx(summ["overall"]["mean_diff"])
    assert (s["overall"]["better"], s["overall"]["equal"], s["overall"]["worse"]) == \
           (summ["overall"]["better"], summ["overall"]["equal"], summ["overall"]["worse"])
    comp = [r for r in rows if r["eff_diff"] is not None]
    assert s["flags_base"] == pytest.approx(sum(r["feas_infeasible_base"] + r["feas_risk_base"] for r in comp) / len(comp))
    assert s["flags_other"] == pytest.approx(sum(r["feas_infeasible_other"] + r["feas_risk_other"] for r in comp) / len(comp))


def test_the_tables_add_up_on_real_runs(runs):
    rows, _ = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["n5_r1_s3+fz2"])
    s = RV.analyze(rows)
    assert sum(q["n"] for q in s["quadrants"].values()) == sum(g["n"] for g in s["split"].values()) == s["n_pairs"]
    assert s["worse_fewer"]["n"] == s["quadrants"][("worse", "fewer")]["n"] and s["better_more"]["n"] == s["quadrants"][("better", "more")]["n"]
    for r in s["penalties"]:
        assert r["better"] + r["equal"] + r["worse"] == s["n_pairs"]
    assert s["penalties"][0]["mean"] == pytest.approx(s["overall"]["mean_eff"])
    for p in s["pairs"]:
        assert p["reduction"] == p["flags_base"] - p["flags_other"]
        assert set(p["rules_cleared"]) == set(p["rules_base"]) - set(p["rules_other"])


def test_a_run_against_itself_changes_nothing(runs):
    rows, _ = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["rb"])
    s = RV.analyze(rows)
    assert s["overall"]["mean_eff"] == 0 and s["overall"]["mean_reduction"] == 0
    assert {k for k, v in s["quadrants"].items() if v["n"]} == {("equal", "same")}
    assert s["break_even"]["status"] == "not worse" and s["worse_fewer"]["n"] == 0 and s["better_more"]["n"] == 0
    assert all(r["cleared"] == r["added"] == 0 and r["base_has"] == r["other_has"] for r in s["rules"])
    assert all(r["mean"] == 0 for r in s["penalties"])


def test_physics_flags_can_only_add_to_the_counts_on_real_runs(runs):
    rows, _ = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["n5_r1_s3"])
    plain, phys = RV.analyze(rows), RV.analyze(rows, physics=True)
    assert 0 < phys["n_pairs"] <= plain["n_pairs"]
    by_key = {(p["part_id"], p["repeat"]): p for p in plain["pairs"]}
    for p in phys["pairs"]:
        q = by_key[(p["part_id"], p["repeat"])]
        assert p["flags_base"] >= q["flags_base"] and p["flags_other"] >= q["flags_other"]
        assert p["flags_base"] - q["flags_base"] <= 2 and p["flags_other"] - q["flags_other"] <= 2     # over power and Ra high: at most two more
    assert any(p["flags_base"] > by_key[(p["part_id"], p["repeat"])]["flags_base"] for p in phys["pairs"])   # the physics flags do occur here


# ---------------------------------------------------------------- the command line
def test_cli_prints_the_report_and_writes_the_csv(runs, tmp_path, capsys):
    csv_path = str(tmp_path / "pairs.csv")
    code = risk_compare.main([runs["dirs"]["rb"], runs["dirs"]["n5_r1_s3+fz2"], "--csv", csv_path, "--top", "3", "--resamples", "50"])
    out = capsys.readouterr().out
    assert code == 0 and "Risk-aware comparison, t_n5_r1_s3+fz2 minus t_rb" in out and "Break-even P*" in out and f"wrote {csv_path}" in out
    assert os.path.exists(csv_path)


def test_cli_physics_and_penalties(runs, capsys):
    code = risk_compare.main([runs["dirs"]["rb"], runs["dirs"]["n5_r1_s3"], "--physics", "--penalties", "0,3", "--resamples", "50"])
    out = capsys.readouterr().out
    assert code == 0 and "over power" in out and "P = 0 " in out and "P = 3 " in out and "P = 10" not in out


def test_cli_errors(runs, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        risk_compare.main([str(tmp_path), runs["dirs"]["rb"]])
    assert e.value.code == 1 and "no records.jsonl" in capsys.readouterr().err
    for bad in ("x,y", "-1,2", ""):
        with pytest.raises(SystemExit) as e:
            risk_compare.main([runs["dirs"]["rb"], runs["dirs"]["n5_r1_s3"], f"--penalties={bad}"])
        assert e.value.code == 1
        assert "--penalties" in capsys.readouterr().err
