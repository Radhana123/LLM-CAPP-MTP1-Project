"""Plan-confidence calibration (capp/calibration.py, scripts/confidence_tune.py, summary.reflag, run_eval --confidence-flag).
Offline. The statistics are checked against hand-computed answers on records whose confidence and efficiency gap are set by the test."""
import copy
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from capp import calibration as CAL                       # noqa: E402
from capp import dataset, harness                         # noqa: E402
from capp import summary as SM                            # noqa: E402
from capp import systems as SY                            # noqa: E402
import confidence_tune                                    # noqa: E402  (scripts/confidence_tune.py)
import run_eval                                           # noqa: E402  (scripts/run_eval.py)
import test_partdiff as T                                 # noqa: E402  (messy_planner)

CONF = [0.50, 0.55, 0.60, 0.70, 0.80, 0.80, 0.90, 0.90, 0.95, 1.0, 1.0, 1.0]
GAP = [6, 1, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0]                # bad (gap > 3): records 0, 2, 4
BAD = [g > 3 for g in GAP]


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("cal"))
    parts = dataset.load_dataset("dev_v2")[:12]
    res = harness.run_experiment("n5_r1_s3", parts, tag="cal", out_dir=out, progress=None, planner=T.messy_planner)
    return {"records": res["records"], "dir": res["out_dir"], "out": out, "parts": parts}


def inject(records, confs=CONF, gaps=GAP):
    """Deep copies of real records whose confidence and efficiency gap to the best Route Builder variant are the given numbers."""
    recs = copy.deepcopy(records[:len(confs)])
    for r, c, g in zip(recs, confs, gaps):
        assert r["final"]["check"]["valid"]
        best = max(s["efficiency_score"] for s in r["best_scores"])
        r["final"]["scores_eval"]["efficiency_score"] = best - g
        r["info"]["confidence"] = c
        r["info"]["flagged"] = c < 0.8
    return recs


@pytest.fixture()
def known(real):
    return inject(real["records"])


def write_run(path, recs, dataset_name="dev_v2"):
    os.makedirs(path, exist_ok=True)
    harness.write_outputs(str(path), recs, SM.summarize(recs), {"system": recs[0]["system"], "dataset": dataset_name})
    return str(path)


# ------------------------------------------------------------------ statistics, by hand
def test_auroc_by_hand():
    pairs = [(c, b) for c, b in zip(CONF, BAD)]
    assert CAL.auroc(pairs) == pytest.approx(23.5 / 27)
    assert CAL.auroc([(0.1, True), (0.9, False)]) == 1.0 and CAL.auroc([(0.9, True), (0.1, False)]) == 0.0
    assert CAL.auroc([(0.5, True), (0.5, False)]) == 0.5
    assert CAL.auroc([(0.1, True), (0.2, True)]) is None and CAL.auroc([]) is None


def test_at_threshold_flags_below_and_counts():
    pairs = list(zip(CONF, BAD))
    r = CAL.at_threshold(pairs, 0.8)
    assert (r["n"], r["n_flagged"], r["caught"], r["n_bad"]) == (12, 4, 2, 3)
    assert r["flagged_share"] == pytest.approx(4 / 12) and r["precision"] == 0.5 and r["recall"] == pytest.approx(2 / 3)
    assert r["base_rate"] == 0.25 and r["lift"] == pytest.approx(2.0)
    assert CAL.at_threshold(pairs, 0.0)["n_flagged"] == 0 and CAL.at_threshold(pairs, 0.0)["precision"] is None
    assert CAL.at_threshold(pairs, 1.5)["recall"] == 1.0
    assert CAL.at_threshold([(0.5, False)], 1.0)["recall"] is None                   # no bad plan: recall undefined


def test_sweep_has_one_row_per_flagged_set_and_reproduces_it():
    tab = CAL.sweep(list(zip(CONF, BAD)))
    assert [r["threshold"] for r in tab] == [0.5, 0.525, 0.575, 0.65, 0.75, 0.85, 0.925, 0.975, 1.001]
    assert [r["n_flagged"] for r in tab] == [0, 1, 2, 3, 4, 6, 8, 9, 12]
    shares = [r["flagged_share"] for r in tab]
    recalls = [r["recall"] for r in tab]
    assert shares == sorted(shares) and recalls == sorted(recalls) and recalls[0] == 0 and recalls[-1] == 1
    # the printed threshold gives exactly the flagged set of its row, with the pipeline's own rule (score < threshold)
    for r in tab:
        assert sum(1 for c in CONF if c < r["threshold"]) == r["n_flagged"]
    assert CAL.sweep([]) == []
    one = CAL.sweep([(0.9, True), (0.9, False)])
    assert [r["n_flagged"] for r in one] == [0, 2]


def test_midpoint_stays_inside_the_gap():
    assert CAL._midpoint(0.9, 0.95) == 0.925
    lo, hi = 0.1234561, 0.1234569
    t = CAL._midpoint(lo, hi)
    assert lo < t <= hi
    for lo, hi in ((0.0, 1e-9), (0.333333, 0.333334), (0.5, 0.5000001)):
        assert lo < CAL._midpoint(lo, hi) <= hi


def test_recommendations_by_hand():
    tab = CAL.sweep(list(zip(CONF, BAD)))
    b = CAL.recommend_budget(tab, 0.20)                        # at most 2 of 12 plans flagged
    assert (b["threshold"], b["n_flagged"], b["caught"]) == (0.575, 2, 1)
    assert CAL.recommend_budget(tab, 0.0)["n_flagged"] == 0
    assert CAL.recommend_budget(tab, 1.0)["n_flagged"] == 12 or CAL.recommend_budget(tab, 1.0)["recall"] == 1.0
    r = CAL.recommend_recall(tab, 0.80)                        # recall 2/3 is not enough; 3/3 first at 6 flagged
    assert (r["threshold"], r["n_flagged"], r["recall"]) == (0.85, 6, 1.0)
    assert CAL.recommend_recall(tab, 0.5)["n_flagged"] == 3
    assert CAL.recommend_recall(CAL.sweep([(0.5, False), (0.7, False)]), 0.8) is None


def test_part_bootstrap_resamples_whole_parts():
    rows = [{"part_id": "a", "x": 1}, {"part_id": "a", "x": 1}, {"part_id": "b", "x": 0}, {"part_id": "c", "x": 0}]
    groups = CAL._by_part(rows)
    assert sorted(len(g) for g in groups) == [1, 1, 2]
    import random
    for _ in range(20):
        sample = CAL._resample(groups, random.Random(_))
        assert sum(1 for r in sample if r["part_id"] == "a") % 2 == 0           # the two repeats of a part come and go together
    lo, hi = CAL.bootstrap(rows, lambda rs: sum(r["x"] for r in rs) / len(rs), resamples=200)
    assert 0 <= lo <= hi <= 1
    assert CAL.bootstrap(rows[:2], lambda rs: 1.0) == (None, None)             # one part: no interval
    same = [{"part_id": str(i), "x": 5} for i in range(6)]
    assert CAL.bootstrap(same, lambda rs: rs[0]["x"] * 1.0, resamples=50) == (5.0, 5.0)
    assert CAL.bootstrap(rows, lambda rs: len(rs), resamples=50, seed=3) == CAL.bootstrap(rows, lambda rs: len(rs), resamples=50, seed=3)


def test_reliability_bins_never_split_ties():
    rows = [{"scores": {"confidence": c}, "labels": {"gap": b}} for c, b in zip(CONF, BAD)]
    bins = CAL.reliability(rows, "confidence", "gap", bins=4)
    assert sum(b["n"] for b in bins) == 12
    for a, b in zip(bins, bins[1:]):
        assert a["score_max"] < b["score_min"]
    assert bins[0]["score_min"] == 0.5 and bins[-1]["score_max"] == 1.0 and bins[0]["bad_rate"] > bins[-1]["bad_rate"] == 0
    assert CAL.reliability([], "confidence", "gap") == []


# ------------------------------------------------------------------ rows from real records
def test_rows_from_real_records(real):
    rows = CAL.rows_from_records(real["records"])
    assert len(rows) == 12 and [r["part_id"] for r in rows] == [r["part_id"] for r in real["records"]]
    for r, rec in zip(rows, real["records"]):
        assert r["scores"]["confidence"] == rec["info"]["confidence"]
        assert r["flagged_recorded"] == (r["scores"]["confidence"] is not None and r["scores"]["confidence"] < SY.CONFIDENCE_FLAG)
        for s in ("raw_agreement", "raw_valid_share"):
            assert r["scores"][s] is None or 0.0 <= r["scores"][s] <= 1.0
        assert r["labels"]["not_raw"] == (rec["final"]["provenance"] != "raw_llm")
        if r["eff_gap"] is not None:
            assert r["labels"]["gap"] == (r["eff_gap"] > CAL.DEFAULT_BAD_GAP)
        if r["lcs_ratio"] is not None:
            assert r["labels"]["lcs"] == (r["lcs_ratio"] < CAL.DEFAULT_BAD_LCS)
    tight = CAL.rows_from_records(real["records"], bad_gap=0.0, bad_lcs=1.0)
    assert sum(1 for r in tight if r["labels"]["lcs"]) >= sum(1 for r in rows if r["labels"]["lcs"])


def test_raw_signals_follow_the_raw_samples(real):
    rec = copy.deepcopy(real["records"][0])
    samples = [c for c in rec["candidates"] if c["sample_id"] >= 0]
    for c in samples:
        c["llm_ok"], c["llm_steps"] = True, ["Facing", "Drilling", "Inspection"]
    rec["info"]["attempts"], rec["info"]["raw_strict_valid"] = len(samples), 3
    row = CAL.rows_from_records([rec])[0]
    assert row["scores"]["raw_agreement"] == pytest.approx(1.0)                 # identical raw routes
    assert row["scores"]["raw_valid_share"] == pytest.approx(3 / len(samples))
    samples[0]["llm_steps"] = ["Inspection", "Drilling", "Facing"]
    assert CAL.rows_from_records([rec])[0]["scores"]["raw_agreement"] < 1.0
    for c in samples:
        c["llm_ok"] = False
    assert CAL.rows_from_records([rec])[0]["scores"]["raw_agreement"] is None
    rec["info"]["attempts"] = 0
    assert CAL.rows_from_records([rec])[0]["scores"]["raw_valid_share"] is None


# ------------------------------------------------------------------ the analysis on records with known answers
def test_gap_label_and_scores_come_through(known):
    rows = CAL.rows_from_records(known)
    assert [r["labels"]["gap"] for r in rows] == BAD
    assert [r["scores"]["confidence"] for r in rows] == CONF
    assert [round(r["eff_gap"]) for r in rows] == GAP


def test_analyse_run_known_answers(known):
    res = CAL.analyse_run(known, label="gap", budget=0.2, recall=0.8, current=0.8, resamples=200)
    assert res["n_records"] == res["n_used"] == 12 and res["n_parts"] == 12 and res["base_rates"]["gap"] == 0.25
    a = res["auroc"]["confidence|gap"]
    assert a["value"] == pytest.approx(23.5 / 27) and a["n"] == 12 and a["ci95"][0] <= a["value"] + 1e-9 <= a["ci95"][1] + 1e-9 or a["ci95"][0] is not None
    c = res["current"]
    assert (c["n_flagged"], c["caught"]) == (4, 2) and c["precision"] == 0.5 and c["recall"] == pytest.approx(2 / 3)
    lo, hi = c["recall_ci95"]
    assert 0.0 <= lo <= hi <= 1.0
    assert res["recommend_budget"]["threshold"] == 0.575 and res["recommend_budget"]["n_flagged"] == 2
    assert res["recommend_recall"]["threshold"] == 0.85 and res["recommend_recall"]["recall"] == 1.0
    assert len(res["sweep"]) == 9 and res["unscored"] == {"n": 0, "bad": 0}
    assert res["gap_percentiles"]["p50"] == pytest.approx(0.0, abs=1e-9)
    # evaluate() at a sweep threshold agrees with its row, and the recorded flags at 0.8 are the pipeline's
    row = next(r for r in res["sweep"] if r["threshold"] == 0.65)
    ev = CAL.evaluate(CAL.rows_from_records(known), "confidence", "gap", 0.65, resamples=50)
    assert (ev["n_flagged"], ev["caught"], ev["recall"]) == (row["n_flagged"], row["caught"], row["recall"])


def test_current_threshold_defaults_to_the_pipelines(known, monkeypatch):
    assert CAL.analyse_run(known, resamples=20)["current_threshold"] == SY.CONFIDENCE_FLAG
    monkeypatch.setattr(SY, "CONFIDENCE_FLAG", 0.6)
    assert CAL.analyse_run(known, resamples=20)["current"]["n_flagged"] == 2          # confidences 0.5 and 0.55


def test_analysis_is_reproducible(known):
    a = CAL.analyse_run(known, resamples=100)
    b = CAL.analyse_run(known, resamples=100)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def test_unscored_plans_are_counted_and_never_flagged(known):
    recs = copy.deepcopy(known)
    recs[1]["info"]["confidence"], recs[2]["info"]["confidence"] = None, None
    res = CAL.analyse_run(recs, resamples=50)
    assert res["n_used"] == 10 and res["unscored"] == {"n": 2, "bad": 1} and res["scored"]["confidence"] == 10
    assert res["current"]["n"] == 10


def test_a_label_with_one_class_gives_no_recommendation(known):
    recs = inject(known, gaps=[0] * 12)
    res = CAL.analyse_run(recs, label="gap", resamples=30)
    assert res["base_rates"]["gap"] == 0 and res["auroc"]["confidence|gap"]["value"] is None
    text = CAL.format_report(res)
    assert "no plan is bad under this label" in text and "Every threshold" not in text
    allbad = CAL.format_report(CAL.analyse_run(inject(known, gaps=[9] * 12), label="gap", resamples=30))
    assert "every plan is bad" in allbad


def test_errors(real):
    with pytest.raises(CAL.CalibrationError, match="no records"):
        CAL.analyse_run([])
    with pytest.raises(CAL.CalibrationError, match="label must be"):
        CAL.analyse_run(real["records"], label="nope")
    single = copy.deepcopy(real["records"])
    for r in single:
        r["info"]["confidence"] = None
    with pytest.raises(CAL.CalibrationError, match="single sample has no confidence"):
        CAL.analyse_run(single)


def test_a_run_whose_answers_could_not_be_read_says_so_and_not_single_sample(real):
    """Step 36: every answer cut off or invalid (no sample route at all) is not the same as a run with one sample."""
    nothing = copy.deepcopy(real["records"])
    for r in nothing:
        r["info"]["confidence"] = None
        for c in r["candidates"]:
            c["llm_ok"] = False
    with pytest.raises(CAL.CalibrationError, match="no model answer in this run could be read as a route") as e:
        CAL.analyse_run(nothing)
    assert "single sample" not in str(e.value) and "token limit" in str(e.value) and "trunc" in str(e.value)
    assert str(e.value).startswith("no plan has both a confidence score and a gap label (")
    one_readable = copy.deepcopy(nothing)                                    # a single record with a readable sample is enough for the other message
    for c in one_readable[0]["candidates"]:
        c["llm_ok"] = c.get("sample_id", -1) >= 0
    with pytest.raises(CAL.CalibrationError, match="single sample has no confidence"):
        CAL.analyse_run(one_readable)
    with pytest.raises(CAL.CalibrationError, match="no plan has both a raw_agreement score and a gap label$"):
        CAL.analyse_run(nothing, signal="raw_agreement")                     # the hint is about the confidence only


def test_sealed_mode_skips_the_sweep(known):
    res = CAL.analyse_run(known, tune=False, evaluate_at=0.8, resamples=30)
    assert "sweep" not in res and "recommend_budget" not in res and res["evaluate_at"]["n_flagged"] == 4
    text = CAL.format_report(res)
    assert "fixed threshold asked for" in text and "sealed split" in text and "Recommendations" not in text


def test_report_text(known):
    text = CAL.format_report(CAL.analyse_run(known, resamples=50))
    for needle in ("=== n5_r1_s3: 12 plans of 12 parts ===", "base rate 25%", "AUROC", "raw_agreement", "now (CONFIDENCE_FLAG)",
                   "best recall within 20% flagged", "fewest flags for recall 80%", "--confidence-flag T", "Every threshold (9 distinct"):
        assert needle in text, needle
    rows = CAL.sweep_rows(CAL.analyse_run(known, resamples=20))
    assert len(rows) == 9 and rows[0]["label"] == "gap" and {"threshold", "recall", "precision"} <= set(rows[0])


# ------------------------------------------------------------------ flagging at another threshold
def test_reflag_changes_only_the_flags(known):
    before = json.dumps(known, sort_keys=True)
    out = SM.reflag(known, 0.92)
    assert json.dumps(known, sort_keys=True) == before                          # the input is untouched
    assert [r["info"]["flagged"] for r in out] == [c < 0.92 for c in CONF]
    assert all(r["info"]["flag_threshold"] == 0.92 for r in out)
    for a, b in zip(known, out):
        assert a["final"] == b["final"] and a["candidates"] == b["candidates"]
    assert SM.reflag([dict(known[0], info=dict(known[0]["info"], confidence=None))], 0.9)[0]["info"]["flagged"] is False


def test_summary_names_the_threshold_only_when_it_was_changed(known):
    plain = SM.summarize(known)
    assert "flag_threshold" not in plain["confidence"] and "confidence below" not in SM.format_summary("s", plain)
    s = SM.summarize(SM.reflag(known, 0.92))
    assert s["confidence"]["flag_threshold"] == 0.92 and s["confidence"]["n_flagged"] == 8
    assert "confidence below 0.92" in SM.format_summary("s", s)


def test_run_eval_summarize_only_with_a_new_flag(known, tmp_path, monkeypatch, capsys):
    folder = write_run(tmp_path / "dev_n5_r1_s3", known)
    monkeypatch.setattr(SY, "CONFIDENCE_FLAG", SY.CONFIDENCE_FLAG)               # restored afterwards
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--summarize-only", folder, "--confidence-flag", "0.92"])
    run_eval.main()
    out = capsys.readouterr().out
    assert "flagged 8/12" in out and "confidence below 0.92" in out
    assert SY.CONFIDENCE_FLAG == 0.92
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--summarize-only", folder])
    run_eval.main()
    assert "confidence below" not in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["0", "-0.1", "1.5", "nan"])
def test_run_eval_rejects_a_flag_outside_zero_one(bad, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(SY, "CONFIDENCE_FLAG", SY.CONFIDENCE_FLAG)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--system", "rb", "--confidence-flag", bad])
    with pytest.raises(SystemExit) as e:
        run_eval.main()
    assert e.value.code == 2 and "confidence-flag" in capsys.readouterr().err


def test_a_run_uses_the_flag_and_records_it(tmp_path, monkeypatch):
    monkeypatch.setattr(SY, "CONFIDENCE_FLAG", 0.999)
    parts = dataset.load_dataset("dev_v2")[:6]
    res = harness.run_experiment("n5_r1_s3", parts, tag="fl", out_dir=str(tmp_path), progress=None, planner=T.messy_planner)
    assert res["run_info"]["confidence_flag_threshold"] == 0.999
    assert all(r["info"]["flagged"] == (r["info"]["confidence"] is not None and r["info"]["confidence"] < 0.999) for r in res["records"])


# ------------------------------------------------------------------ the script
def test_cli_prints_and_writes(known, tmp_path, capsys):
    folder = write_run(tmp_path / "dev_n5_r1_s3", known)
    csv_path, json_path = tmp_path / "t.csv", tmp_path / "t.json"
    assert confidence_tune.main([folder, "--resamples", "50", "--csv", str(csv_path), "--json", str(json_path)]) == 0
    text = capsys.readouterr().out
    assert "n5_r1_s3: 12 plans" in text and "Recommendations" in text
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("system,label,signal,threshold") and len(rows) == 10
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["label"] == "gap" and data["recommend_budget"]["threshold"] == 0.575 and data["sealed"] is False


def test_cli_options(known, tmp_path, capsys):
    folder = write_run(tmp_path / "dev_n5_r1_s3", known)
    assert confidence_tune.main([folder, "--resamples", "30", "--label", "not_raw", "--signal", "raw_agreement", "--budget", "0.5",
                                 "--recall", "0.5", "--bad-gap", "1", "--current", "0.9", "--evaluate-at", "0.95"]) == 0
    out = capsys.readouterr().out
    assert "signal: raw agreement" in out and "fixed threshold asked for" in out and "threshold 0.9 " in out
    assert confidence_tune.main([folder, "--resamples", "30", "--label", "gap", "--bad-gap", "0.5"]) == 0
    assert "efficiency gap to the best Route Builder variant above 0.5 points" in capsys.readouterr().out


def test_cli_errors(known, tmp_path, capsys):
    folder = write_run(tmp_path / "dev_n5_r1_s3", known)
    assert confidence_tune.main([str(tmp_path / "nowhere")]) == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "records.jsonl").write_text("", encoding="utf-8")
    assert confidence_tune.main([str(empty)]) == 1
    capsys.readouterr()
    for argv in (["--budget", "0"], ["--recall", "1.5"], ["--bad-lcs", "0"], ["--bad-gap", "-1"]):
        with pytest.raises(SystemExit) as e:
            confidence_tune.main([folder] + argv)
        assert e.value.code == 2
    single = copy.deepcopy(known)
    for r in single:
        r["info"]["confidence"] = None
    d = write_run(tmp_path / "single", single)
    assert confidence_tune.main([d]) == 1 and "single sample has no confidence" in capsys.readouterr().err
    nothing = copy.deepcopy(known)
    for r in nothing:
        r["info"]["confidence"] = None
        for c in r["candidates"]:
            c["llm_ok"] = False
    d = write_run(tmp_path / "nothing", nothing)
    assert confidence_tune.main([d]) == 1
    err = capsys.readouterr().err
    assert "no model answer in this run could be read as a route" in err and "single sample" not in err


def test_cli_several_runs_are_not_pooled(known, tmp_path, capsys):
    a = write_run(tmp_path / "a", known)
    other = copy.deepcopy(known)
    for r in other:
        r["system"] = "n3_r1_s3"
    b = write_run(tmp_path / "b", other)
    assert confidence_tune.main([a, b, "--resamples", "20", "--json", str(tmp_path / "x.json")]) == 0
    out = capsys.readouterr().out
    assert out.count("plans of 12 parts ===") == 2 and "=== n3_r1_s3: 12 plans of 12 parts" in out       # one report each, 12 plans each
    assert len(json.loads((tmp_path / "x.json").read_text(encoding="utf-8"))) == 2


def test_cli_refuses_to_tune_on_a_sealed_split(known, tmp_path, capsys):
    folder = write_run(tmp_path / "test_n5_r1_s3", known, dataset_name="test_v2")
    assert confidence_tune.main([folder]) == 1
    err = capsys.readouterr().err
    assert "sealed split" in err and "--evaluate-at" in err
    assert confidence_tune.main([folder, "--evaluate-at", "0.85", "--resamples", "30", "--csv", str(tmp_path / "s.csv"),
                                 "--plot", str(tmp_path / "s.png")]) == 0
    cap = capsys.readouterr()
    assert "fixed threshold asked for" in cap.out and "Every threshold" not in cap.out and "Recommendations" not in cap.out
    assert "nothing to write" in cap.err and not (tmp_path / "s.png").exists()
    bench = write_run(tmp_path / "bench_x", known, dataset_name="bench_v1")
    assert confidence_tune.main([bench]) == 1
    # a folder without run.json (older runs) is treated as a development run
    os.remove(os.path.join(folder, "run.json"))
    assert confidence_tune.main([folder, "--resamples", "20"]) == 0


def test_cli_plot(known, tmp_path):
    pytest.importorskip("matplotlib")
    folder = write_run(tmp_path / "dev_n5_r1_s3", known)
    png = tmp_path / "conf.png"
    assert confidence_tune.main([folder, "--resamples", "20", "--plot", str(png)]) == 0
    assert png.stat().st_size > 5000
    other = write_run(tmp_path / "b", [dict(r, system="n3_r1_s3") for r in known])
    assert confidence_tune.main([folder, other, "--resamples", "20", "--plot", str(png)]) == 0
    assert (tmp_path / "conf_n5_r1_s3.png").exists() and (tmp_path / "conf_n3_r1_s3.png").exists()
    # a label without bad plans still draws
    res = CAL.analyse_run(inject(known, gaps=[0] * 12), resamples=20)
    assert os.path.getsize(CAL.plot(inject(known, gaps=[0] * 12), res, str(tmp_path / "flat.png"))) > 3000


def test_old_records_still_work(real):
    old = copy.deepcopy(real["records"])
    for r in old:
        for c in r["candidates"]:
            c.pop("cascade", None)
            c["meta"].pop("model", None)
    assert CAL.rows_from_records(old)[0]["scores"]["confidence"] == real["records"][0]["info"]["confidence"]
