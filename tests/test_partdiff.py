"""Per-part comparison of two runs (capp/partdiff.py, scripts/part_diff.py). Offline: the runs are produced by the harness with a
deterministic fake planner that makes the kind of mistakes an LLM makes (a dropped operation, two swapped), no API call."""
import copy
import csv
import json
import os
import random
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from capp import compare, dataset, harness, partdiff as PD, verify as V      # noqa: E402
import route_builder as rb                                                    # noqa: E402
import part_diff                                                              # noqa: E402


# ---------------------------------------------------------------- the route delta
def test_identical_routes():
    d = PD.route_delta(["Facing", "Drilling", "Inspection"], ["Facing", "Drilling", "Inspection"])
    assert d["cause"] == "identical" and d["only_base"] == [] and d["only_other"] == []


def test_markers_are_ignored():
    d = PD.route_delta(["Facing", V.MARKER, "Face Milling"], ["Facing", "Face Milling"])
    assert d["cause"] == "identical"


def test_order_only():
    d = PD.route_delta(["Facing", "Drilling", "Inspection"], ["Drilling", "Facing", "Inspection"])
    assert d["cause"] == "order only" and d["only_base"] == [] and d["only_other"] == []


def test_other_has_extra_operations():
    d = PD.route_delta(["Facing", "Inspection"], ["Facing", "Reaming", "Inspection"])
    assert d["cause"] == "other has extra operations" and d["only_other"] == ["Reaming"] and d["only_base"] == []


def test_other_lacks_operations():
    d = PD.route_delta(["Facing", "Reaming", "Inspection"], ["Facing", "Inspection"])
    assert d["cause"] == "other lacks operations" and d["only_base"] == ["Reaming"] and d["only_other"] == []


def test_different_operations_in_route_order():
    d = PD.route_delta(["Facing", "External Threading", "Chamfering", "Inspection"],
                       ["Facing", "Tapping", "Chamfering", "Drilling", "Inspection"])
    assert d["cause"] == "different operations"
    assert d["only_base"] == ["External Threading"] and d["only_other"] == ["Tapping", "Drilling"]


def test_a_repeated_operation_counts_as_a_multiset():
    d = PD.route_delta(["Facing", "Drilling"], ["Facing", "Drilling", "Drilling"])
    assert d["cause"] == "other has extra operations" and d["only_other"] == ["Drilling"]


def test_changeovers_are_counted_per_route():
    lathe, mill = next(iter(rb.LATHE_ONLY_OPS)), next(iter(rb.MILLING_ONLY_OPS))
    d = PD.route_delta([lathe, lathe], [lathe, mill, lathe])
    assert d["changeovers_base"] == 0 and d["changeovers_other"] == 2


def test_swapping_the_runs_mirrors_the_delta():
    a, b = ["Facing", "Inspection"], ["Facing", "Reaming", "Inspection"]
    x, y = PD.route_delta(a, b), PD.route_delta(b, a)
    assert x["only_other"] == y["only_base"] and x["only_base"] == y["only_other"]
    assert (x["cause"], y["cause"]) == ("other has extra operations", "other lacks operations")


def test_small_helpers():
    assert PD._verdict(None) == "n/a" and PD._verdict(1.0) == "better" and PD._verdict(-1.0) == "worse"
    assert PD._verdict(1e-12) == "equal" and PD._verdict(0.0) == "equal"
    assert PD._rel(10.0, 15.0) == 0.5 and PD._rel(0.0, 5.0) is None and PD._rel(None, 5.0) is None


# ---------------------------------------------------------------- two real runs
def _meta():
    return {"cached": True, "finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.0}


def messy_planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
    routes = rb.generate_valid_routes(features, max_routes=15)
    if not routes:
        return {"success": False, "steps": [], "raw_response": ""}
    steps = V.clean_steps(routes[(sample_id * 3 + len(features)) % len(routes)]["steps"])
    r = random.Random(f"{','.join(features)}-{sample_id}")                       # deterministic across processes
    if r.random() < 0.4 and len(steps) > 3:
        steps = steps[:]
        steps.pop(r.randrange(len(steps)))
    if r.random() < 0.4:
        steps = steps[:]
        i = r.randrange(len(steps) - 1)
        steps[i], steps[i + 1] = steps[i + 1], steps[i]
    return {"success": True, "steps": steps, "raw_response": "", "model": "fake", "meta": _meta()}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("runs"))
    parts = dataset.load_dataset("dev_v2")[:14]
    dirs = {}
    for system in ("rb", "n5_r1_s3", "n5_r1_s3+fz2"):
        harness.run_experiment(system, parts, tag="t", out_dir=out, progress=None, planner=None if system == "rb" else messy_planner)
        dirs[system] = os.path.join(out, "t_" + system)
    recs = {k: harness.load_records(v) for k, v in dirs.items()}
    return {"dirs": dirs, "recs": recs, "parts": parts}


def test_one_row_per_part_and_repeat(runs):
    rows, un = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["n5_r1_s3"])
    assert len(rows) == len(runs["parts"]) and not any(un.values())
    assert [r["part_id"] for r in rows] == [p["part_id"] for p in runs["parts"]]


def test_agrees_with_compare_runs_paired_quality(runs):
    base, other = runs["recs"]["rb"], runs["recs"]["n5_r1_s3"]
    rows, _ = PD.pair_rows(base, other)
    q = compare.paired_quality(base, other)
    s = PD.summarize(rows)["overall"]
    assert (s["n"], s["better"], s["equal"], s["worse"]) == (q["n"], q["better"], q["equal"], q["worse"])
    assert s["mean_diff"] == pytest.approx(q["mean_diff"])


def test_a_run_against_itself_is_all_identical_and_equal(runs):
    rows, _ = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["rb"])
    assert {r["verdict"] for r in rows} == {"equal"} and {r["cause"] for r in rows} == {"identical"}
    assert all(r["eff_diff"] == 0 for r in rows)
    s = PD.summarize(rows)
    assert s["overall"]["mean_diff"] == 0 and s["overall"]["worse"] == 0 and s["operations"] == []


def test_swapping_the_runs_flips_every_pair(runs):
    a, b = runs["recs"]["rb"], runs["recs"]["n5_r1_s3"]
    fwd, _ = PD.pair_rows(a, b)
    bwd, _ = PD.pair_rows(b, a)
    mirror = {"better": "worse", "worse": "better", "equal": "equal", "n/a": "n/a", "identical": "identical", "order only": "order only",
              "other has extra operations": "other lacks operations", "other lacks operations": "other has extra operations",
              "different operations": "different operations"}
    fwd = {r["part_id"]: r for r in fwd}
    for r in bwd:
        f = fwd[r["part_id"]]
        assert r["verdict"] == mirror[f["verdict"]] and r["cause"] == mirror[f["cause"]]
        assert r["only_base"] == f["only_other"] and r["only_other"] == f["only_base"]
        if f["eff_diff"] is not None:
            assert r["eff_diff"] == pytest.approx(-f["eff_diff"])


def test_the_rows_describe_the_pair(runs):
    rows, _ = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["n5_r1_s3"])
    for r in rows:
        if r["eff_diff"] is not None:
            assert r["eff_diff"] == pytest.approx(r["eff_other"] - r["eff_base"])
        assert r["provenance_base"] == "route_builder" and r["provenance_other"] in ("raw_llm", "repaired_llm", "route_builder")
        assert r["n_ops_base"] == len(r["route_base"].split(" > ")) and r["n_ops_other"] == len(r["route_other"].split(" > "))
        assert r["cause"] in PD.CAUSES
        assert r["time_min_rel"] == pytest.approx((r["time_min_other"] - r["time_min_base"]) / r["time_min_base"])
    assert any(r["cause"] != "identical" for r in rows)                         # the fake planner does make mistakes


def test_physics_and_feasibility_columns_are_carried(runs):
    rows, _ = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["n5_r1_s3"])
    r = rows[0]
    for k in ("phys_energy_kwh", "phys_wear_inr", "phys_peak_spindle_kw", "over_power", "ra_risk", "feas_infeasible", "feas_risk", "feas_rules"):
        assert k + "_base" in r and k + "_other" in r
    assert isinstance(r["phys_energy_kwh_base"], float) and r["feas_infeasible_base"] != ""


def test_summary_tables_add_up(runs):
    rows, _ = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["n5_r1_s3"])
    s = PD.summarize(rows, ("machine_profile", "material", "archetype"))
    n = s["n_comparable"]
    assert n + s["n_not_comparable"] == s["n_pairs"] == len(rows)
    for g in ("machine_profile", "material", "archetype"):
        assert sum(t["n"] for t in s["groups"][g].values()) == n
    assert sum(t["n"] for t in s["causes"].values()) == n and sum(t["n"] for t in s["provenance_other"].values()) == n
    for t in list(s["groups"]["material"].values()) + list(s["causes"].values()):
        assert t["better"] + t["equal"] + t["worse"] == t["n"]
    assert s["overall"]["n"] == n


def test_operation_table_names_the_operations_that_differ(runs):
    rows, _ = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["n5_r1_s3"])
    s = PD.summarize(rows)
    seen = {(x["direction"], x["op"]) for x in s["operations"]}
    for r in rows:
        if r["eff_diff"] is not None:
            for op in r["only_other"]:
                assert ("other has", op) in seen
            for op in r["only_base"]:
                assert ("other lacks", op) in seen
    means = [x["mean_diff"] for x in s["operations"]]
    assert means == sorted(means)                                                # worst first


def test_unmatched_and_mismatched_pairs_are_reported_not_compared(runs):
    base, other = runs["recs"]["rb"], copy.deepcopy(runs["recs"]["n5_r1_s3"])
    other = other[2:]                                                            # two parts only in base
    base_short = base[:-1]                                                       # one part only in other
    other[0]["part"]["features"] = other[0]["part"]["features"] + ["Knurl"]      # same id, other features: another dataset
    rows, un = PD.pair_rows(base_short, other)
    assert len(un["only_base"]) == 2 and len(un["only_other"]) == 1 and un["mismatch"] == [(other[0]["part_id"], other[0]["repeat"])]
    assert len(rows) == len(other) - 1 - 1
    text = PD.render(rows, PD.summarize(rows), "rb", "x", unmatched=un)
    assert "Not paired" in text and "mismatch" in text


def test_an_invalid_final_is_not_comparable_and_is_listed_with_its_routes(runs):
    base, other = runs["recs"]["rb"], copy.deepcopy(runs["recs"]["n5_r1_s3"])
    other[3]["final"]["check"]["valid"] = False
    rows, _ = PD.pair_rows(base, other)
    bad = [r for r in rows if r["eff_diff"] is None]
    assert len(bad) == 1 and bad[0]["verdict"] == "n/a" and bad[0]["valid_other"] is False and bad[0]["valid_base"] is True
    s = PD.summarize(rows)
    assert s["n_not_comparable"] == 1 and s["n_comparable"] == len(rows) - 1
    text = PD.render(rows, s, "rb", "llm")
    assert "not comparable" in text.lower() and "llm final is not strict-valid" in text and bad[0]["route_other"].split(" > ")[0] in text


# ---------------------------------------------------------------- the report
def test_report_has_every_section(runs):
    rows, un = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["n5_r1_s3"])
    text = PD.render(rows, PD.summarize(rows), "rb", "n5_r1_s3", top=3, unmatched=un)
    for needle in ("Per-part difference, n5_r1_s3 minus rb", "By machine_profile:", "By complexity:", "By material:", "By tolerance:",
                   "By how the routes differ", "By provenance of the n5_r1_s3 final:", "What moves Efficiency", "Physics and feasibility",
                   "over machine power", "Worse pairs (", "Description, not causation"):
        assert needle in text, needle


def test_report_lists_at_most_top_pairs_worst_first(runs):
    rows, _ = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["n5_r1_s3+fz2"])
    worse = sorted((r for r in rows if r["verdict"] == "worse"), key=lambda r: r["eff_diff"])
    assert len(worse) >= 2
    text = PD.render(rows, PD.summarize(rows), "rb", "llm", top=2)
    listed = [r for r in rows if r["part_id"] + " " in text.split("Worse pairs")[1]]
    assert {r["part_id"] for r in listed} == {r["part_id"] for r in worse[:2]}
    assert text.split("Worse pairs")[1].index(worse[0]["part_id"]) < text.split("Worse pairs")[1].index(worse[1]["part_id"])


def test_report_marks_the_operations_that_differ(runs):
    rows, _ = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["n5_r1_s3+fz2"])
    r = next(r for r in rows if r["only_other"] and r["verdict"] == "worse")
    block = PD._pair_block(r, "rb", "llm")
    assert "+" + r["only_other"][0] in block and r["part_id"] in block and "Efficiency" in block and r["cause"] in block


def test_only_better_and_only_all(runs):
    rows, _ = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["n5_r1_s3"])
    s = PD.summarize(rows)
    assert "Better pairs (" in PD.render(rows, s, "rb", "llm", only="better")
    assert "All pairs (" in PD.render(rows, s, "rb", "llm", only="all")


def test_report_without_any_worse_pair_does_not_fail(runs):
    rows, _ = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["rb"])
    text = PD.render(rows, PD.summarize(rows), "rb", "rb2")
    assert "Worse pairs (0), first 0" in text


def test_csv_has_one_row_per_pair_and_joined_lists(runs, tmp_path):
    rows, _ = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["n5_r1_s3"])
    path = str(tmp_path / "pairs.csv")
    PD.write_csv(path, rows)
    back = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(back) == len(rows) and back[0]["part_id"] == rows[0]["part_id"]
    r = next(r for r in rows if r["only_other"])
    assert next(b for b in back if b["part_id"] == r["part_id"])["only_other"] == ";".join(r["only_other"])
    with pytest.raises(ValueError):
        PD.write_csv(path, [])


def test_fz2_run_pairs_with_rb_too(runs):
    rows, _ = PD.pair_rows(runs["recs"]["rb"], runs["recs"]["n5_r1_s3+fz2"])
    assert len(rows) == len(runs["parts"]) and PD.summarize(rows)["n_comparable"] == len(rows)


# ---------------------------------------------------------------- the script
def test_script_prints_the_report_and_writes_csv_and_json(runs, tmp_path, capsys):
    csv_path, json_path = str(tmp_path / "o" / "p.csv"), str(tmp_path / "o" / "s.json")
    code = part_diff.main([runs["dirs"]["rb"], runs["dirs"]["n5_r1_s3"], "--top", "2", "--csv", csv_path, "--json", json_path])
    out = capsys.readouterr().out
    assert code == 0 and "Per-part difference, n5_r1_s3 minus rb" in out and f"wrote {len(runs['parts'])} pairs" in out
    assert len(list(csv.DictReader(open(csv_path, encoding="utf-8")))) == len(runs["parts"])
    assert json.load(open(json_path, encoding="utf-8"))["n_pairs"] == len(runs["parts"])


def test_script_group_and_only_options(runs, capsys):
    assert part_diff.main([runs["dirs"]["rb"], runs["dirs"]["n5_r1_s3"], "--group", "archetype,n_features", "--only", "all"]) == 0
    out = capsys.readouterr().out
    assert "By archetype:" in out and "By n_features:" in out and "By material:" not in out and "All pairs (" in out


def test_script_rejects_unknown_group_missing_folder_and_no_common_pairs(runs, tmp_path, capsys):
    assert part_diff.main([runs["dirs"]["rb"], runs["dirs"]["n5_r1_s3"], "--group", "nonsense"]) == 1
    assert "unknown --group" in capsys.readouterr().err
    assert part_diff.main([runs["dirs"]["rb"], str(tmp_path / "missing")]) == 1
    assert "no records.jsonl" in capsys.readouterr().err
    other = str(tmp_path / "other")
    os.makedirs(other)
    with open(os.path.join(other, "records.jsonl"), "w", encoding="utf-8") as f:
        recs = copy.deepcopy(runs["recs"]["n5_r1_s3"])
        for r in recs:
            r["part_id"] = "ZZ_" + r["part_id"]
            f.write(json.dumps(r) + "\n")
    assert part_diff.main([runs["dirs"]["rb"], other]) == 1
    assert "no pairs" in capsys.readouterr().err


def test_script_labels_two_runs_of_the_same_system_by_folder(runs, tmp_path, capsys):
    twin = str(tmp_path / "twin_rb")
    os.makedirs(twin)
    with open(os.path.join(twin, "records.jsonl"), "w", encoding="utf-8") as f:
        for r in runs["recs"]["rb"]:
            f.write(json.dumps(r) + "\n")
    assert part_diff.main([runs["dirs"]["rb"], twin]) == 0
    assert "twin_rb minus t_rb" in capsys.readouterr().out
