"""Step 20 (WP7, RQ6): capp/extraction_eval.py (M9, extraction quality on annotated drawings) and scripts/extraction_eval.py.

No network: the extractor is a fake function, and the scoring works on records that are written here by hand, so every number in
the tests was computed on paper first.
"""
import csv
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capp import extraction_eval as EE        # noqa: E402
from capp import extraction_review as ER      # noqa: E402

CLI = os.path.join(ROOT, "scripts", "extraction_eval.py")


# ---------------------------------------------------------------- helpers
def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    return str(path)


def folder_with(tmp_path, *names):
    d = tmp_path / "drawings"
    d.mkdir(exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"fake image " + n.encode())
    return str(d)


def pass_report(features, rejected=()):
    return {"success": True, "features": list(features), "rejected_features": list(rejected), "confidence": "high", "notes": "", "evidence": {}}


def rec(name, truth, g, a, rejected=(), **extra):
    """A record as EE.run writes it, with both passes working."""
    union = list(dict.fromkeys(list(g) + list(a)))
    r = {"drawing": name, "truth": list(truth), "truth_holes": None, "success": True, "error": "", "features": union,
         "rejected_features": list(rejected), "passes": {"geometry": pass_report(g), "annotation": pass_report(a)}, "passes_ok": 2,
         "sample_agreement": {f: (f in g) + (f in a) for f in union}, "confidence": "high", "notes": ""}
    r.update(extra)
    return r


# Three drawings, worked out on paper (G = truth, g / a = the passes):
#   A  G {Hole, Slot, Thread}  g {Hole, Slot}        a {Hole, Thread, Pocket}   u {Hole, Slot, Thread, Pocket}  b {Hole}  highlighted {Slot, Thread, Pocket}
#   B  G {Hole, Boss}          g {Hole}              a {Hole}                   u = b = {Hole}                              highlighted {}
#   C  G {Slot}                g {Slot, Fillet}      a {Slot, Fillet}           u = b = {Slot, Fillet}                      highlighted {}
#                 tp fp fn   precision  recall   F1      corrections / drawing
#   geometry      4  1  2    0.800      0.667    0.727   (1+1+1)/3 = 1.00
#   annotation    4  2  2    0.667      0.667    0.667   (2+1+1)/3 = 1.33
#   union         5  2  1    0.714      0.833    0.769   (1+1+1)/3 = 1.00
#   both          3  1  3    0.750      0.500    0.600   (2+1+1)/3 = 1.33
#   agreement     5  1  1    0.833      0.833    0.833   (0+1+1)/3 = 0.67 + 3/3 = 1.00 confirmations
def three():
    return [rec("A.png", ["Hole", "Slot", "Thread"], ["Hole", "Slot"], ["Hole", "Thread", "Pocket"]),
            rec("B.png", ["Hole", "Boss"], ["Hole"], ["Hole"]),
            rec("C.png", ["Slot"], ["Slot", "Fillet"], ["Slot", "Fillet"])]


def variant(s, name):
    return [v for v in s["variants"] if v["variant"] == name][0]


# ---------------------------------------------------------------- ground truth
def test_ground_truth_in_the_short_and_the_long_form(tmp_path):
    p = write_json(tmp_path / "t.json", {"a.png": ["Hole", "Slot"], "b.png": {"features": ["Hole"], "hole_diameters": [8, 6.5]}})
    t = EE.load_ground_truth(p)
    assert t == {"a.png": {"features": ["Hole", "Slot"], "hole_diameters": None},
                 "b.png": {"features": ["Hole"], "hole_diameters": [8.0, 6.5]}}


def test_ground_truth_inside_a_drawings_key(tmp_path):
    p = write_json(tmp_path / "t.json", {"drawings": {"a.png": ["Hole"]}, "annotator": "R"})
    assert list(EE.load_ground_truth(p)) == ["a.png"]


@pytest.mark.parametrize("body, message", [
    ({"a.png": ["Hole", "Gear"]}, "not in the vocabulary: Gear"),
    ({"a.png": []}, "no features written down yet"),
    ({"a.png": ["Hole", "Hole"]}, "listed twice"),
    ({"a.png": "Hole"}, "must be a list"),
    ({"a.png": [1, 2]}, "must be a list"),
    ({"a.png": {"features": ["Hole"], "hole_diameters": [0]}}, "positive numbers"),
    ({"a.png": {"features": ["Hole"], "hole_diameters": ["8"]}}, "positive numbers"),
    ({"a.png": {"features": ["Hole"], "hole_diameters": [True]}}, "positive numbers"),
    ({"a.png": {"features": ["Hole"], "hole_diameters": 8}}, "positive numbers"),
    ({}, "expected an object"),
    ([], "expected an object"),
])
def test_ground_truth_problems_are_named(tmp_path, body, message):
    with pytest.raises(EE.ExtractionEvalError, match=message):
        EE.load_ground_truth(write_json(tmp_path / "t.json", body))


def test_all_the_problems_of_a_ground_truth_are_reported_together(tmp_path):
    p = write_json(tmp_path / "t.json", {"a.png": [], "b.png": ["Gear"], "c.png": ["Hole"]})
    with pytest.raises(EE.ExtractionEvalError) as e:
        EE.load_ground_truth(p)
    assert "a.png" in str(e.value) and "b.png" in str(e.value) and "c.png" not in str(e.value)


def test_unreadable_ground_truth_files(tmp_path):
    with pytest.raises(EE.ExtractionEvalError, match="cannot read"):
        EE.load_ground_truth(str(tmp_path / "missing.json"))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(EE.ExtractionEvalError, match="cannot read"):
        EE.load_ground_truth(str(bad))


# ---------------------------------------------------------------- drawings
def test_template_lists_every_drawing_with_an_empty_entry(tmp_path):
    d = folder_with(tmp_path, "b.PNG", "a.jpg", "c.jpeg", "notes.txt", "d.gif")
    t = EE.template(d)
    assert list(t) == ["a.jpg", "b.PNG", "c.jpeg"] and t["a.jpg"] == {"features": [], "hole_diameters": None}


def test_a_filled_in_template_loads_and_an_empty_one_does_not(tmp_path):
    d = folder_with(tmp_path, "a.png")
    t = EE.template(d)
    p = write_json(tmp_path / "t.json", t)
    with pytest.raises(EE.ExtractionEvalError, match="no features written down"):
        EE.load_ground_truth(p)
    t["a.png"]["features"] = ["Hole"]
    assert EE.load_ground_truth(write_json(tmp_path / "t2.json", t))["a.png"]["features"] == ["Hole"]


def test_missing_folder():
    with pytest.raises(EE.ExtractionEvalError, match="no such folder"):
        EE.list_images("/no/such/folder")


def test_match_drawings_by_name_and_by_name_without_extension(tmp_path):
    d = folder_with(tmp_path, "a.png", "b.jpg", "extra.png")
    truth = {"a.png": {}, "b": {}, "ghost.png": {}}
    pairs, no_image, no_truth = EE.match_drawings(d, truth)
    assert pairs == [("a.png", os.path.join(d, "a.png")), ("b", os.path.join(d, "b.jpg"))]
    assert no_image == ["ghost.png"] and no_truth == ["extra.png"]


def test_an_extensionless_key_is_ambiguous_when_two_files_share_the_stem(tmp_path):
    d = folder_with(tmp_path, "a.png", "a.jpg")
    pairs, no_image, _ = EE.match_drawings(d, {"a": {}})
    assert pairs == [] and no_image == ["a"]


# ---------------------------------------------------------------- running the extractor
class Fake:
    """An extractor that answers from a table and remembers how it was called."""
    def __init__(self, answers=None, raises=()):
        self.answers, self.raises, self.calls, self.tags = answers or {}, set(raises), [], []

    def __call__(self, data, image_format="png", **kw):
        name = data.decode().replace("fake image ", "")
        self.calls.append({"name": name, "format": image_format, "kw": kw})
        self.tags.append(os.environ.get("CAPP_RUN_TAG"))
        if name in self.raises:
            raise RuntimeError("boom")
        if name in self.answers:
            return self.answers[name]
        return {"success": True, "features": ["Hole"], "rejected_features": [], "confidence": "high", "notes": "n",
                "passes": {"geometry": pass_report(["Hole"]), "annotation": pass_report(["Hole"])}, "passes_ok": 2,
                "sample_agreement": {"Hole": 2}}


TRUTH = {"a.png": {"features": ["Hole"], "hole_diameters": None}, "b.jpg": {"features": ["Slot"], "hole_diameters": [8.0]}}


def test_run_writes_the_records_and_the_run_description(tmp_path):
    d = folder_with(tmp_path, "a.png", "b.jpg", "unused.png")
    out = str(tmp_path / "out")
    fake = Fake()
    seen = []
    records = EE.run(d, TRUTH, out, tag="fe7", extractor=fake, progress=seen.append)
    assert [r["drawing"] for r in records] == ["a.png", "b.jpg"]
    assert records[0]["truth"] == ["Hole"] and records[1]["truth"] == ["Slot"] and records[1]["truth_holes"] == [8.0]
    assert records[0]["success"] and set(records[0]["passes"]) == {"geometry", "annotation"} and records[0]["sample_agreement"] == {"Hole": 2}
    assert [c["format"] for c in fake.calls] == ["png", "jpeg"] and all(c["kw"] == {} for c in fake.calls)
    assert len(seen) == 2 and "[1/2] a.png: ok" in seen[0]
    assert EE.load_records(out) == records
    info = json.load(open(os.path.join(out, "run.json"), encoding="utf-8"))
    assert info["tag"] == "fe7" and info["n_drawings"] == 2 and info["images_without_truth"] == ["unused.png"]
    assert info["with_evidence"] is False and info["with_dimensions"] is False


def test_run_passes_the_options_to_the_extractor_only_when_asked(tmp_path):
    d = folder_with(tmp_path, "a.png")
    t = {"a.png": TRUTH["a.png"]}
    f1, f2, f3 = Fake(), Fake(), Fake()
    EE.run(d, t, str(tmp_path / "o1"), extractor=f1)
    EE.run(d, t, str(tmp_path / "o2"), extractor=f2, with_evidence=True)
    EE.run(d, t, str(tmp_path / "o3"), extractor=f3, with_evidence=True, with_dimensions=True)
    assert f1.calls[0]["kw"] == {} and f2.calls[0]["kw"] == {"with_evidence": True}
    assert f3.calls[0]["kw"] == {"with_evidence": True, "with_dimensions": True}
    assert json.load(open(tmp_path / "o3" / "run.json"))["with_evidence"] is True


def test_run_keeps_the_dimension_pass_in_the_record(tmp_path):
    d = folder_with(tmp_path, "b.jpg")
    ex = {"success": True, "features": ["Hole"], "passes": {}, "hole_diameters": [8.0, 6.0],
          "dimensions": {"success": True, "hole_diameters": [8.0, 6.0]}}
    r = EE.run(d, {"b.jpg": TRUTH["b.jpg"]}, str(tmp_path / "o"), extractor=Fake({"b.jpg": ex}), with_dimensions=True)[0]
    assert r["hole_diameters"] == [8.0, 6.0] and r["dimensions_ok"] is True


def test_run_survives_an_extractor_that_raises_and_one_that_fails(tmp_path):
    d = folder_with(tmp_path, "a.png", "b.jpg", "c.png")
    truth = {"a.png": TRUTH["a.png"], "b.jpg": TRUTH["b.jpg"], "c.png": {"features": ["Boss"], "hole_diameters": None}}
    fake = Fake({"c.png": {"success": False, "error": "Vision API error: 500"}}, raises=["a.png"])
    records = EE.run(d, truth, str(tmp_path / "o"), extractor=fake)
    assert [r["success"] for r in records] == [False, True, False]
    assert records[0]["error"] == "RuntimeError: boom" and records[2]["error"] == "Vision API error: 500"
    assert len(fake.calls) == 3


def test_run_stops_at_once_when_there_is_no_api_key(tmp_path):
    d = folder_with(tmp_path, "a.png", "b.jpg")
    fake = Fake({"a.png": {"success": False, "error": "GROQ_API_KEY not configured."}})
    out = str(tmp_path / "out")
    with pytest.raises(EE.ExtractionEvalError, match="GROQ_API_KEY is not configured"):
        EE.run(d, TRUTH, out, extractor=fake)
    assert [c["name"] for c in fake.calls] == ["a.png"] and not os.path.exists(out)       # b.jpg was not tried, nothing written


def test_a_failure_that_is_not_the_missing_key_does_not_stop_the_run(tmp_path):
    d = folder_with(tmp_path, "a.png", "b.jpg")
    fake = Fake({"a.png": {"success": False, "error": "Vision API error: timeout"}})
    records = EE.run(d, TRUTH, str(tmp_path / "o"), extractor=fake)
    assert len(records) == 2 and len(fake.calls) == 2


def test_run_uses_its_own_run_tag_and_restores_the_old_one(tmp_path, monkeypatch):
    d = folder_with(tmp_path, "a.png")
    t = {"a.png": TRUTH["a.png"]}
    monkeypatch.setenv("CAPP_RUN_TAG", "before")
    f = Fake()
    EE.run(d, t, str(tmp_path / "o1"), tag="fe3", extractor=f)
    assert f.tags == ["fe3_extraction"] and os.environ["CAPP_RUN_TAG"] == "before"
    monkeypatch.delenv("CAPP_RUN_TAG")
    EE.run(d, t, str(tmp_path / "o2"), tag="fe3", extractor=Fake())
    assert "CAPP_RUN_TAG" not in os.environ
    with pytest.raises(EE.ExtractionEvalError):                                 # also restored when the run is cut short
        EE.run(d, t, str(tmp_path / "o3"), extractor=Fake({"a.png": {"success": False, "error": "GROQ_API_KEY not configured."}}))
    assert "CAPP_RUN_TAG" not in os.environ


def test_run_limit(tmp_path):
    d = folder_with(tmp_path, "a.png", "b.jpg")
    fake = Fake()
    assert len(EE.run(d, TRUTH, str(tmp_path / "o"), extractor=fake, limit=1)) == 1 and len(fake.calls) == 1


def test_run_without_a_matching_drawing_is_an_error(tmp_path):
    d = folder_with(tmp_path, "zzz.png")
    with pytest.raises(EE.ExtractionEvalError, match="no drawing of the folder has a ground truth"):
        EE.run(d, TRUTH, str(tmp_path / "o"), extractor=Fake())


def test_load_records_without_a_run_folder(tmp_path):
    with pytest.raises(EE.ExtractionEvalError, match="no records.jsonl"):
        EE.load_records(str(tmp_path))


# ---------------------------------------------------------------- scoring: the three-drawing case, by hand
def test_scoring_matches_the_hand_computed_table():
    s = EE.score(three())
    assert s["n_drawings"] == 3 and s["n_scored"] == 3 and s["excluded"] == []
    expected = {   # variant: (tp, fp, fn, precision, recall, f1, corrections per drawing)
        "geometry": (4, 1, 2, 4 / 5, 4 / 6, 8 / 11, 1.0),
        "annotation": (4, 2, 2, 4 / 6, 4 / 6, 8 / 12, 4 / 3),
        "union": (5, 2, 1, 5 / 7, 5 / 6, 10 / 13, 1.0),
        "both": (3, 1, 3, 3 / 4, 3 / 6, 6 / 10, 4 / 3),
        "agreement": (5, 1, 1, 5 / 6, 5 / 6, 10 / 12, 2 / 3),
    }
    assert [v["variant"] for v in s["variants"]] == list(EE.VARIANTS)
    for name, (tp, fp, fn, p, r, f1, corr) in expected.items():
        v = variant(s, name)
        assert (v["tp"], v["fp"], v["fn"]) == (tp, fp, fn), name
        assert v["precision"] == pytest.approx(p) and v["recall"] == pytest.approx(r) and v["f1"] == pytest.approx(f1), name
        assert v["corrections_mean"] == pytest.approx(corr), name


def test_confirmations_belong_to_the_agreement_variant_only():
    s = EE.score(three())
    assert variant(s, "agreement")["confirmations_mean"] == pytest.approx(1.0)       # 3 highlighted features over 3 drawings
    assert all(variant(s, n)["confirmations_mean"] == 0.0 for n in ("geometry", "annotation", "union", "both"))


def test_exact_shares_and_list_sizes():
    s = EE.score(three())
    assert variant(s, "union")["exact_share"] == 0.0 and variant(s, "both")["exact_share"] == 0.0
    assert variant(s, "agreement")["exact_share"] == pytest.approx(1 / 3)               # only drawing A ends up exactly right
    assert variant(s, "union")["shown_mean"] == pytest.approx((4 + 1 + 2) / 3)
    assert variant(s, "agreement")["shown_mean"] == pytest.approx((4 + 1 + 2) / 3)      # the agreement list shows the whole union
    assert variant(s, "both")["shown_mean"] == pytest.approx((1 + 1 + 2) / 3)


def test_the_drawing_average_f1_differs_from_the_pooled_f1():
    s = EE.score(three())
    # union per drawing: A 2*3/(6+1+0) = 6/7, B 2/(2+0+1) = 2/3, C 2/(2+1+0) = 2/3
    assert variant(s, "union")["drawing_f1_mean"] == pytest.approx((6 / 7 + 2 / 3 + 2 / 3) / 3)
    assert variant(s, "union")["drawing_f1_mean"] != pytest.approx(variant(s, "union")["f1"])


def test_the_overlap_of_the_passes():
    o = EE.score(three())["overlap"]
    assert (o["union_features"], o["seen_by_both"], o["seen_by_one"]) == (7, 4, 3)
    assert o["both_correct_share"] == pytest.approx(3 / 4)       # Hole, Hole, Slot are in the truth; Fillet is not
    assert o["one_correct_share"] == pytest.approx(2 / 3)        # Slot and Thread yes, Pocket no
    assert o["flagged_share"] == pytest.approx(3 / 7) and o["flagged_per_drawing"] == pytest.approx(1.0)
    assert o["drawings_with_nothing_flagged"] == pytest.approx(2 / 3)


def test_per_drawing_rows():
    rows = {r["drawing"]: r for r in EE.score(three())["per_drawing"]}
    a = rows["A.png"]
    assert a["truth"] == ["Hole", "Slot", "Thread"] and a["highlighted"] == ["Pocket", "Slot", "Thread"] and a["missed_by_both"] == []
    assert (a["corrections_union"], a["corrections_agreement"], a["corrections_both"]) == (1, 0, 2)
    assert rows["B.png"]["missed_by_both"] == ["Boss"] and rows["B.png"]["highlighted"] == []
    assert rows["C.png"]["corrections_agreement"] == 1


def test_per_feature_rows():
    pf = {r["feature"]: r for r in EE.score(three())["per_feature"]}
    assert pf["Hole"]["support"] == 2 and pf["Fillet"]["support"] == 0 and "Gear_Teeth" not in pf
    # Slot: in the truth of A and C. geometry saw it in A and C (tp 2), annotation only in C (tp 1, fn 1)
    assert (pf["Slot"]["geometry"]["tp"], pf["Slot"]["annotation"]["tp"], pf["Slot"]["annotation"]["fn"]) == (2, 1, 1)
    # Fillet is never right: geometry and annotation both report it on C
    assert pf["Fillet"]["union"]["fp"] == 1 and pf["Fillet"]["union"]["precision"] == 0.0 and pf["Fillet"]["union"]["recall"] is None
    assert pf["Boss"]["union"]["fn"] == 1 and pf["Boss"]["union"]["recall"] == 0.0


def test_the_intervals_are_reproducible_and_bracket_the_estimate():
    s1, s2 = EE.score(three()), EE.score(three())
    for v in ("union", "agreement"):
        lo, hi = variant(s1, v)["f1_ci95"]
        assert (lo, hi) == variant(s2, v)["f1_ci95"] and 0.0 <= lo <= variant(s1, v)["f1"] <= hi <= 1.0
        clo, chi = variant(s1, v)["corrections_ci95"]
        assert clo <= variant(s1, v)["corrections_mean"] <= chi


def test_one_drawing_has_no_interval():
    s = EE.score(three()[:1])
    assert variant(s, "union")["f1_ci95"] == (None, None) and variant(s, "union")["corrections_ci95"] == (None, None)
    assert "F1 95% CI" in EE.format_report(s)               # and the report still prints


def test_a_perfect_extraction_needs_no_correction():
    s = EE.score([rec("a.png", ["Hole"], ["Hole"], ["Hole"]), rec("b.png", ["Slot", "Boss"], ["Slot", "Boss"], ["Boss", "Slot"])])
    for v in s["variants"]:
        assert v["f1"] == pytest.approx(1.0) and v["corrections_mean"] == 0.0 and v["exact_share"] == 1.0 and v["confirmations_mean"] == 0.0


# ---------------------------------------------------------------- scoring: what is left out
def test_drawings_where_a_pass_failed_are_left_out_of_every_row():
    bad_pass = rec("D.png", ["Hole"], ["Hole"], ["Hole"])
    bad_pass["passes"]["annotation"] = {"success": False, "error": "timeout"}
    failed = rec("E.png", ["Hole"], ["Hole"], ["Hole"], success=False, error="Vision API error: 500")
    s = EE.score(three() + [bad_pass, failed])
    assert s["n_drawings"] == 5 and s["n_scored"] == 3
    assert {e["drawing"]: e["reason"] for e in s["excluded"]} == {"D.png": "a pass failed", "E.png": "Vision API error: 500"}
    assert variant(s, "union")["tp"] == 5                                 # the same numbers as without the two extra records
    text = EE.format_report(s)
    assert "3 of 5 drawings scored" in text and "D.png" in text and "E.png" in text


def test_a_legacy_record_without_passes_is_excluded_not_a_crash():
    old = {"drawing": "old.png", "truth": ["Hole"], "success": True, "features": ["Hole"], "passes": {}}
    s = EE.score(three() + [old])
    assert s["n_scored"] == 3 and s["excluded"][0]["drawing"] == "old.png"


def test_nothing_to_score_is_an_error_that_says_why():
    only_failed = [rec("E.png", ["Hole"], [], [], success=False, error="GROQ_API_KEY not configured.")]
    with pytest.raises(EE.ExtractionEvalError, match="no drawing has both passes.*GROQ_API_KEY"):
        EE.score(only_failed)
    with pytest.raises(EE.ExtractionEvalError, match="no drawing has both passes"):
        EE.score([])


def test_names_outside_the_vocabulary_are_counted_with_a_suggestion():
    rs = three() + [rec("D.png", ["Hole"], ["Hole"], ["Hole"], rejected=["Counter Bore", "Gear"]),
                    rec("E.png", ["Hole"], ["Hole"], ["Hole"], rejected=["Counter Bore"])]
    u = {x["name"]: x for x in EE.score(rs)["unrecognised"]}
    assert u["Counter Bore"]["count"] == 2 and u["Counter Bore"]["suggestion"] == "Counterbore"
    assert u["Gear"]["count"] == 1 and u["Gear"]["suggestion"] is None
    assert "Counter Bore x2 (~ Counterbore)" in EE.format_report(EE.score(rs))


# ---------------------------------------------------------------- scoring: hole diameters
def hole_rec(name, truth_holes, read, ok=True):
    return rec(name, ["Hole"], ["Hole"], ["Hole"], truth_holes=truth_holes, hole_diameters=read, dimensions_ok=ok)


def test_hole_diameters_match_within_the_tolerance_once_each():
    rs = [hole_rec("a.png", [8.0, 6.5], [8.04, 6.0]),         # 8.04 matches 8; 6.0 is 0.5 away from 6.5: tp 1, fp 1, fn 1
          hole_rec("b.png", [10.0], [10.0]),                    # exact
          hole_rec("c.png", [8.0], [8.0, 8.01]),                # the second reading has no truth left to match: tp 1, fp 1
          hole_rec("d.png", [12.0], [], ok=False)]              # a failed call: nothing read, fn 1
    h = EE.score(rs)["holes"]
    assert (h["tp"], h["fp"], h["fn"]) == (3, 2, 2) and h["drawings"] == 4
    assert h["precision"] == pytest.approx(3 / 5) and h["recall"] == pytest.approx(3 / 5)
    assert h["exact_set_share"] == pytest.approx(1 / 4) and h["failed_calls"] == 1
    assert "Hole diameters (dimension pass, 4 drawings" in EE.format_report(EE.score(rs))


def test_hole_tolerance_is_a_twentieth_of_a_millimetre():
    assert EE.HOLE_TOL_MM == 0.05
    assert EE.score([hole_rec("a.png", [8.0], [8.04])])["holes"]["tp"] == 1
    assert EE.score([hole_rec("a.png", [8.0], [8.06])])["holes"]["tp"] == 0


def test_only_drawings_with_a_truth_and_a_dimension_pass_count_for_holes():
    rs = three() + [hole_rec("a.png", [8.0], [8.0]), rec("x.png", ["Hole"], ["Hole"], ["Hole"], truth_holes=[8.0])]   # x: pass not run
    assert EE.score(rs)["holes"]["drawings"] == 1
    assert EE.score(three())["holes"] == {}
    assert "Hole diameters" not in EE.format_report(EE.score(three()))


# ---------------------------------------------------------------- report and CSV
def test_the_report_shows_every_variant_and_the_oracle_warning():
    text = EE.format_report(EE.score(three()), "fe1")
    assert "M9 extraction quality - fe1: 3 of 3 drawings scored" in text
    for label in EE.LABELS.values():
        assert label in text
    assert "ORACLE" in text and "upper bound" in text
    assert "union 1.00" in text and "agreement 0.67" in text and "1.00 confirmations" in text
    assert "seen by both passes" in text and "Read with care" in text


def test_csv_files(tmp_path):
    s = EE.score(three())
    paths = EE.write_csv(str(tmp_path / "sub" / "m9"), s)
    assert [os.path.basename(p) for p in paths] == ["m9_variants.csv", "m9_features.csv", "m9_drawings.csv"]
    with open(paths[0], newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [r["variant"] for r in rows] == list(EE.VARIANTS)
    u = [r for r in rows if r["variant"] == "union"][0]
    assert (u["tp"], u["fp"], u["fn"]) == ("5", "2", "1") and float(u["corrections_mean"]) == pytest.approx(1.0)
    with open(paths[1], newline="", encoding="utf-8-sig") as f:
        feats = {r["feature"]: r for r in csv.DictReader(f)}
    assert feats["Hole"]["support"] == "2" and "union_precision" in feats["Hole"]
    with open(paths[2], newline="", encoding="utf-8-sig") as f:
        drawings = {r["drawing"]: r for r in csv.DictReader(f)}
    assert drawings["A.png"]["truth"] == "Hole;Slot;Thread" and drawings["A.png"]["corrections_agreement"] == "0"


# ---------------------------------------------------------------- the command line
def cli(*args):
    return subprocess.run([sys.executable, CLI, *map(str, args)], capture_output=True, text=True, timeout=120, cwd=ROOT)


def write_run(tmp_path, records, name="run"):
    d = tmp_path / name
    d.mkdir()
    with open(d / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (d / "run.json").write_text(json.dumps({"with_evidence": True}), encoding="utf-8")
    return str(d)


def test_cli_template_then_refuses_to_overwrite(tmp_path):
    d = folder_with(tmp_path, "a.png", "b.jpg")
    out = tmp_path / "truth" / "t.json"
    r = cli("template", d, "--out", out)
    assert r.returncode == 0 and "2 drawings" in r.stdout and "Vocabulary:" in r.stdout
    assert list(json.load(open(out))) == ["a.png", "b.jpg"]
    r = cli("template", d, "--out", out)
    assert r.returncode == 1 and "not overwritten" in r.stderr
    assert cli("template", d, "--out", out, "--force").returncode == 0


def test_cli_template_of_an_empty_folder(tmp_path):
    d = folder_with(tmp_path)
    r = cli("template", d, "--out", tmp_path / "t.json")
    assert r.returncode == 1 and "no drawing" in r.stderr and not (tmp_path / "t.json").exists()


def test_cli_score_prints_the_report_and_writes_csv(tmp_path):
    run = write_run(tmp_path, three())
    r = cli("score", run, "--csv", tmp_path / "out" / "m9")
    assert r.returncode == 0, r.stderr
    assert "3 of 3 drawings scored" in r.stdout and "(evidence prompt)" in r.stdout and "union of both passes (v1)" in r.stdout
    assert (tmp_path / "out" / "m9_variants.csv").exists() and "m9_drawings.csv" in r.stdout


def test_cli_score_of_two_runs_keeps_their_csv_files_apart(tmp_path):
    r1, r2 = write_run(tmp_path, three(), "fe1"), write_run(tmp_path, three()[:2], "fe2")
    r = cli("score", r1, r2, "--csv", tmp_path / "m9")
    assert r.returncode == 0 and "3 of 3" in r.stdout and "2 of 2" in r.stdout
    assert (tmp_path / "m9_fe1_variants.csv").exists() and (tmp_path / "m9_fe2_variants.csv").exists()


def test_cli_score_errors_are_messages_not_tracebacks(tmp_path):
    r = cli("score", tmp_path)
    assert r.returncode == 1 and "no records.jsonl" in r.stderr and "Traceback" not in r.stderr
    run = write_run(tmp_path, [rec("E.png", ["Hole"], [], [], success=False, error="boom")], "failed")
    r = cli("score", run)
    assert r.returncode == 1 and "no drawing has both passes" in r.stderr


def test_cli_run_in_replay_mode_without_cache_says_the_key_is_missing(tmp_path):
    d = folder_with(tmp_path, "a.png")
    truth = write_json(tmp_path / "t.json", {"a.png": ["Hole"]})
    env = dict(os.environ, GROQ_API_KEY="", CAPP_CACHE_DIR=str(tmp_path / "cache"))
    r = subprocess.run([sys.executable, CLI, "run", d, "--truth", truth, "--tag", "t", "--out", str(tmp_path / "o")],
                       capture_output=True, text=True, timeout=120, cwd=ROOT, env=env)
    assert r.returncode == 1 and "GROQ_API_KEY is not configured" in r.stderr and "Traceback" not in r.stderr


def test_cli_run_with_a_bad_ground_truth(tmp_path):
    d = folder_with(tmp_path, "a.png")
    truth = write_json(tmp_path / "t.json", {"a.png": ["Gear"]})
    r = cli("run", d, "--truth", truth, "--out", tmp_path / "o")
    assert r.returncode == 1 and "not in the vocabulary" in r.stderr


def test_cli_log_summarises_the_corrections_log(tmp_path):
    path = str(tmp_path / "log.jsonl")
    ER.append_log(path, ER.build_log_entry({"success": True, "features": ["Hole"], "rejected_features": []}, ["Hole", "Slot"]))
    r = cli("log", path)
    assert r.returncode == 0 and "1 drawing(s)" in r.stdout and "Slot x1" in r.stdout
    r = cli("log", tmp_path / "none.jsonl")
    assert r.returncode == 0 and "empty" in r.stdout
