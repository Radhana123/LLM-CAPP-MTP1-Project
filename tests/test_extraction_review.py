"""Step 20 (WP7, report 6.5): capp/extraction_review.py, the evidence, the agreement rule, the confirmation state and the corrections log.

Pure functions: no network, no Streamlit.
"""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capp import extraction_review as ER      # noqa: E402


def report(features, conf="high", evidence=None, rejected=(), notes=""):
    return {"success": True, "features": list(features), "rejected_features": list(rejected), "confidence": conf, "notes": notes,
            "evidence": dict(evidence or {})}


def extraction(g, a, union=None, rejected=(), **kw):
    """An extraction result as week1/image_feature_extractor.py gives it: the union (geometry first), and what each pass said."""
    g_feats = g.get("features", []) if isinstance(g, dict) else list(g)
    a_feats = a.get("features", []) if isinstance(a, dict) else list(a)
    gr = g if isinstance(g, dict) else report(g_feats)
    ar = a if isinstance(a, dict) else report(a_feats)
    uni = union if union is not None else list(dict.fromkeys(g_feats + a_feats))
    n_ok = sum(bool(r.get("success")) for r in (gr, ar))
    return {"success": True, "features": uni, "rejected_features": list(rejected), "confidence": "high", "notes": "",
            "sample_agreement": {f: (f in g_feats) + (f in a_feats) for f in uni}, "passes": {"geometry": gr, "annotation": ar},
            "passes_ok": n_ok, **kw}


def rows_by_feature(review):
    return {r["feature"]: r for r in review["rows"]}


# ---------------------------------------------------------------- build_review: the agreement rule
def test_a_feature_seen_by_both_passes_is_preaccepted_and_high_confidence():
    rv = ER.build_review(extraction(["Hole", "Slot"], ["Hole", "Thread"]))
    rows = rows_by_feature(rv)
    assert rv["preaccepted"] == ["Hole"] and rv["flagged"] == ["Slot", "Thread"]
    assert rows["Hole"]["agreement"] == ER.BOTH and rows["Hole"]["confidence"] == "high" and not rows["Hole"]["needs_confirmation"]
    assert rows["Hole"]["seen_by"] == ["geometry", "annotation"]
    assert rows["Slot"]["agreement"] == ER.ONE and rows["Slot"]["needs_confirmation"] and rows["Slot"]["seen_by"] == ["geometry"]
    assert rows["Thread"]["seen_by"] == ["annotation"]
    assert rv["has_agreement"] and rv["passes_ok"] == 2 and rv["notes"] == []


def test_the_rows_follow_the_order_of_the_union():
    rv = ER.build_review(extraction(["Hole", "Slot"], ["Thread", "Hole"], union=["Thread", "Hole", "Slot"]))
    assert [r["feature"] for r in rv["rows"]] == ["Thread", "Hole", "Slot"]


def test_the_single_pass_confidence_is_capped_at_medium():
    g = report(["Slot", "Hole"], conf="high")
    a = report(["Hole", "Thread"], conf="low")
    rows = rows_by_feature(ER.build_review(extraction(g, a)))
    assert rows["Slot"]["confidence"] == "medium"          # the pass said high, but the other pass did not see it
    assert rows["Thread"]["confidence"] == "low"           # a low confidence stays low
    assert rows["Hole"]["confidence"] == "high"


def test_a_pass_without_a_valid_confidence_gives_no_confidence():
    g = report(["Slot"], conf="very sure")
    rows = rows_by_feature(ER.build_review(extraction(g, ["Hole"])))
    assert rows["Slot"]["confidence"] is None and rows["Hole"]["confidence"] == "medium"


def test_evidence_is_kept_per_pass_and_joined_for_display():
    g = report(["Hole", "Slot"], evidence={"Hole": "circle in front view", "Slot": "long rectangle"})
    a = report(["Hole"], evidence={"Hole": "note 2x dia 8"})
    rows = rows_by_feature(ER.build_review(extraction(g, a)))
    assert rows["Hole"]["evidence"] == {"geometry": "circle in front view", "annotation": "note 2x dia 8"}
    assert rows["Hole"]["evidence_text"] == "geometry: circle in front view; annotation: note 2x dia 8"
    assert rows["Slot"]["evidence"] == {"geometry": "long rectangle"}


def test_no_evidence_gives_an_empty_text():
    rows = rows_by_feature(ER.build_review(extraction(["Hole"], ["Hole"])))
    assert rows["Hole"]["evidence"] == {} and rows["Hole"]["evidence_text"] == ""


def test_empty_evidence_text_is_ignored():
    g = report(["Hole"], evidence={"Hole": ""})
    rows = rows_by_feature(ER.build_review(extraction(g, ["Hole"])))
    assert rows["Hole"]["evidence"] == {}


# ---------------------------------------------------------------- build_review: results without, or with partial, pass data
def test_a_legacy_result_is_fully_preaccepted():
    legacy = {"success": True, "features": ["Hole", "Slot"], "confidence": "high", "notes": "", "rejected_features": []}
    rv = ER.build_review(legacy)
    assert not rv["has_agreement"] and rv["passes_ok"] is None
    assert rv["flagged"] == [] and rv["preaccepted"] == ["Hole", "Slot"] and rv["notes"] == []
    assert {r["agreement"] for r in rv["rows"]} == {ER.UNKNOWN} and all(r["confidence"] is None for r in rv["rows"])
    assert ER.initial_features(rv) == ["Hole", "Slot"] and ER.can_generate(rv, {}, ["Hole"])


def test_counts_alone_still_give_the_agreement():
    old = {"success": True, "features": ["Hole", "Slot"], "sample_agreement": {"Hole": 2, "Slot": 1}, "passes_ok": 2, "rejected_features": []}
    rv = ER.build_review(old)
    assert rv["has_agreement"] and rv["preaccepted"] == ["Hole"] and rv["flagged"] == ["Slot"]
    assert rows_by_feature(rv)["Slot"]["seen_by"] == []             # which pass saw it is not known


def test_counts_with_one_pass_working_flag_everything():
    one = {"success": True, "features": ["Hole", "Slot"], "sample_agreement": {"Hole": 1, "Slot": 1}, "passes_ok": 1, "rejected_features": []}
    rv = ER.build_review(one)
    assert rv["flagged"] == ["Hole", "Slot"] and rv["preaccepted"] == [] and len(rv["notes"]) == 1


def test_one_pass_failing_flags_every_feature_and_says_so():
    ex = extraction(["Hole", "Slot"], {"success": False, "error": "timeout"}, union=["Hole", "Slot"])
    rv = ER.build_review(ex)
    assert rv["passes_ok"] == 1 and rv["flagged"] == ["Hole", "Slot"] and rv["preaccepted"] == []
    assert "Only 1 of the 2 passes worked" in rv["notes"][0]
    assert all(r["agreement"] == ER.ONE and r["seen_by"] == ["geometry"] for r in rv["rows"])


def test_a_failed_pass_is_not_listed_as_having_seen_anything():
    ex = extraction({"success": False, "error": "x", "features": ["Thread"]}, ["Hole"], union=["Hole"])
    rv = ER.build_review(ex)
    assert rows_by_feature(rv)["Hole"]["seen_by"] == ["annotation"]


def test_an_empty_extraction_has_no_rows():
    rv = ER.build_review({"success": True, "features": [], "passes": {"geometry": report([]), "annotation": report([])}, "passes_ok": 2})
    assert rv["rows"] == [] and rv["flagged"] == [] and rv["preaccepted"] == []
    assert not ER.can_generate(rv, {}, [])                         # nothing to generate from


def test_odd_values_do_not_raise():
    for ex in ({}, {"features": None}, {"features": ["Hole"], "passes": "x"}, {"features": ["Hole"], "sample_agreement": "x"},
               {"features": ["Hole"], "passes": {"geometry": None, "annotation": 5}}):
        rv = ER.build_review(ex)
        assert isinstance(rv["rows"], list)


# ---------------------------------------------------------------- unrecognised names
@pytest.mark.parametrize("name, expected", [
    ("Counter Bore", "Counterbore"), ("counter_bore", "Counterbore"), ("Key Way", "Keyway"), ("Pockets", "Pocket"),
    ("Threading", "Thread"), ("Chamfering", "Chamfer"), ("hole", "Hole"), ("Hole", "Hole"),
])
def test_nearest_vocabulary_finds_the_obvious_feature(name, expected):
    assert ER.nearest_vocabulary(name) == expected


@pytest.mark.parametrize("name", ["Gear", "", "   ", "???", None])
def test_nearest_vocabulary_says_none_when_nothing_is_close(name):
    assert ER.nearest_vocabulary(name) is None


def test_unrecognised_names_come_with_a_suggestion_and_the_pass_that_gave_them():
    a = report(["Hole"], rejected=["Counter Bore", "Gear"])
    g = report(["Hole"], rejected=["Gear"])
    rv = ER.build_review(extraction(g, a, rejected=["Gear", "Counter Bore"]))
    un = {u["name"]: u for u in rv["unrecognised"]}
    assert set(un) == {"Gear", "Counter Bore"} and len(rv["unrecognised"]) == 2
    assert un["Counter Bore"]["suggestion"] == "Counterbore" and un["Counter Bore"]["seen_by"] == ["annotation"]
    assert un["Gear"]["suggestion"] is None and un["Gear"]["seen_by"] == ["geometry", "annotation"]


def test_unrecognised_names_from_the_top_level_alone_are_listed():
    rv = ER.build_review({"success": True, "features": ["Hole"], "rejected_features": ["Counter Bore"]})
    assert [u["name"] for u in rv["unrecognised"]] == ["Counter Bore"] and rv["unrecognised"][0]["seen_by"] == []


def test_pass_notes_only_for_working_passes_with_a_note():
    ex = extraction(report(["Hole"], notes="front view only"), {"success": False, "error": "x"}, union=["Hole"])
    assert ER.pass_notes(ex) == {"geometry": "front view only"}
    assert ER.pass_notes({"features": []}) == {}
    assert ER.pass_notes(extraction(["Hole"], ["Hole"])) == {}


# ---------------------------------------------------------------- confirmation state
@pytest.fixture
def rv():
    return ER.build_review(extraction(["Hole", "Slot"], ["Hole", "Thread"]))


def test_the_list_starts_with_the_preaccepted_features(rv):
    assert ER.initial_features(rv) == ["Hole"]


def test_both_highlighted_features_are_pending_at_first(rv):
    assert ER.pending(rv, {}, ["Hole"]) == ["Slot", "Thread"]
    assert not ER.can_generate(rv, {}, ["Hole"])
    assert ER.pending(rv, None, None) == ["Slot", "Thread"]


def test_accept_puts_the_feature_in_the_list_and_reject_takes_it_out(rv):
    feats, dec = ER.decide(["Hole"], {}, "Slot", ER.ACCEPT)
    assert feats == ["Hole", "Slot"] and dec == {"Slot": "accept"}
    feats, dec = ER.decide(feats, dec, "Thread", ER.REJECT)
    assert feats == ["Hole", "Slot"] and dec == {"Slot": "accept", "Thread": "reject"}
    assert ER.pending(rv, dec, feats) == [] and ER.can_generate(rv, dec, feats)


def test_a_later_decision_replaces_the_earlier_one(rv):
    feats, dec = ER.decide(["Hole"], {}, "Slot", ER.ACCEPT)
    feats, dec = ER.decide(feats, dec, "Slot", ER.REJECT)
    assert feats == ["Hole"] and dec["Slot"] == "reject"
    feats, dec = ER.decide(feats, dec, "Slot", ER.ACCEPT)
    assert feats == ["Hole", "Slot"] and dec["Slot"] == "accept"


def test_accepting_twice_does_not_duplicate_the_feature():
    feats, _ = ER.decide(["Hole", "Slot"], {}, "Slot", ER.ACCEPT)
    assert feats == ["Hole", "Slot"]


def test_decide_does_not_modify_its_inputs():
    feats, dec = ["Hole"], {"Slot": "reject"}
    ER.decide(feats, dec, "Thread", ER.ACCEPT)
    assert feats == ["Hole"] and dec == {"Slot": "reject"}


def test_decide_refuses_a_decision_it_does_not_know():
    with pytest.raises(ER.ExtractionReviewError, match="decision must be"):
        ER.decide([], {}, "Slot", "maybe")


def test_editing_the_list_counts_as_accepting(rv):
    """The user adds a highlighted feature in the multiselect instead of pressing Accept: that is a review of it."""
    assert ER.pending(rv, {}, ["Hole", "Slot"]) == ["Thread"]


def test_removing_a_feature_in_the_list_is_not_a_review_of_a_highlighted_one(rv):
    feats, dec = ER.decide(["Hole"], {}, "Slot", ER.ACCEPT)
    feats = [f for f in feats if f != "Slot"]                       # the user then deletes it in the multiselect
    assert ER.pending(rv, dec, feats) == ["Thread"]                 # Slot still counts as decided (accepted, then removed)


def test_the_status_labels(rv):
    assert ER.status(rv, {}, ["Hole"]) == {"Hole": "pre-accepted", "Slot": "needs confirmation", "Thread": "needs confirmation"}
    feats, dec = ER.decide(["Hole"], {}, "Slot", ER.ACCEPT)
    feats, dec = ER.decide(feats, dec, "Thread", ER.REJECT)
    assert ER.status(rv, dec, feats) == {"Hole": "pre-accepted", "Slot": "accepted", "Thread": "rejected"}
    assert ER.status(rv, dec, ["Slot"])["Hole"] == "removed"        # a pre-accepted feature the user took out


def test_a_highlighted_feature_added_in_the_list_shows_as_accepted(rv):
    assert ER.status(rv, {}, ["Hole", "Slot"])["Slot"] == "accepted"


def test_mapping_an_unrecognised_name_adds_the_feature():
    feats, mp = ER.map_unrecognised(["Hole"], {}, "Counter Bore", "Counterbore")
    assert feats == ["Hole", "Counterbore"] and mp == {"Counter Bore": "Counterbore"}
    feats2, _ = ER.map_unrecognised(feats, mp, "Counter-bore", "Counterbore")
    assert feats2 == ["Hole", "Counterbore"]                          # not twice


def test_mapping_to_something_outside_the_vocabulary_is_refused():
    with pytest.raises(ER.ExtractionReviewError, match="not a feature"):
        ER.map_unrecognised([], {}, "Gear", "Cog")


def test_without_highlighted_features_generate_is_allowed_with_a_feature():
    rv2 = ER.build_review(extraction(["Hole"], ["Hole"]))
    assert ER.can_generate(rv2, {}, ["Hole"]) and not ER.can_generate(rv2, {}, [])


# ---------------------------------------------------------------- the corrections log entry
def test_a_clean_review_has_no_corrections_but_decisions():
    ex = extraction(["Hole", "Slot"], ["Hole", "Thread"])
    rv = ER.build_review(ex)
    feats, dec = ER.decide(["Hole"], {}, "Slot", ER.ACCEPT)
    feats, dec = ER.decide(feats, dec, "Thread", ER.REJECT)
    e = ER.build_log_entry(ex, feats, dec, review=rv, drawing_id="d1", now="2026-10-28T00:00:00Z")
    assert e["n_corrections"] == 0 and e["n_decisions"] == 2 and e["n_highlighted"] == 2
    assert {(x["type"], x["feature"]) for x in e["events"]} == {("accepted", "Slot"), ("rejected", "Thread")}
    assert all(not x["correction"] for x in e["events"])
    assert e["final_features"] == ["Hole", "Slot"] and e["drawing_id"] == "d1" and e["timestamp"] == "2026-10-28T00:00:00Z"


def test_removing_a_preaccepted_feature_and_adding_a_missed_one_are_corrections():
    ex = extraction(["Hole", "Slot"], ["Hole", "Thread"])
    feats, dec = ER.decide(["Hole"], {}, "Slot", ER.REJECT)
    feats, dec = ER.decide(feats, dec, "Thread", ER.REJECT)
    final = ["Pocket"]                                                # Hole removed, Pocket added by hand
    e = ER.build_log_entry(ex, final, dec)
    kinds = {(x["type"], x["feature"]): x["correction"] for x in e["events"]}
    assert kinds == {("removed", "Hole"): True, ("rejected", "Slot"): False, ("rejected", "Thread"): False, ("added", "Pocket"): True}
    assert e["n_corrections"] == 2 and e["n_decisions"] == 2


def test_a_mapping_is_one_correction_not_a_mapping_plus_an_addition():
    ex = extraction(report(["Hole"], rejected=["Counter Bore"]), ["Hole"], rejected=["Counter Bore"])
    feats, mp = ER.map_unrecognised(["Hole"], {}, "Counter Bore", "Counterbore")
    e = ER.build_log_entry(ex, feats, {}, mp)
    assert [x["type"] for x in e["events"]] == ["mapped"] and e["events"][0]["from"] == "Counter Bore"
    assert e["n_corrections"] == 1
    assert e["model_output"]["unrecognised"] == ["Counter Bore"]


def test_a_mapping_that_was_removed_again_leaves_no_event():
    ex = extraction(report(["Hole"], rejected=["Counter Bore"]), ["Hole"], rejected=["Counter Bore"])
    e = ER.build_log_entry(ex, ["Hole"], {}, {"Counter Bore": "Counterbore"})
    assert e["events"] == [] and e["n_corrections"] == 0


def test_a_legacy_extraction_logs_corrections_only_and_says_it_had_no_agreement_info():
    legacy = {"success": True, "features": ["Hole", "Slot"], "rejected_features": []}
    e = ER.build_log_entry(legacy, ["Hole", "Thread"])
    assert {(x["type"], x["feature"]) for x in e["events"]} == {("removed", "Slot"), ("added", "Thread")}
    assert e["n_corrections"] == 2 and e["n_decisions"] == 0 and e["n_highlighted"] == 0
    assert e["model_output"]["has_agreement"] is False


def test_the_entry_keeps_the_model_output_for_later_analysis():
    ex = extraction(["Hole", "Slot"], ["Hole", "Thread"])
    e = ER.build_log_entry(ex, ["Hole"])
    mo = e["model_output"]
    assert mo["features"] == ["Hole", "Slot", "Thread"] and mo["highlighted"] == ["Slot", "Thread"]
    assert mo["agreement"] == {"Hole": "both", "Slot": "one", "Thread": "one"}
    assert mo["seen_by"]["Slot"] == ["geometry"] and mo["passes_ok"] == 2 and mo["has_agreement"] is True


def test_duplicates_in_the_final_list_are_collapsed():
    e = ER.build_log_entry(extraction(["Hole"], ["Hole"]), ["Hole", "Hole"])
    assert e["final_features"] == ["Hole"] and e["n_corrections"] == 0


def test_the_drawing_is_hashed_not_stored():
    ex = extraction(["Hole"], ["Hole"])
    data = b"\x89PNG fake image bytes"
    e = ER.build_log_entry(ex, ["Hole"], image_bytes=data)
    assert len(e["image_sha256"]) == 12 and e["drawing_id"] == e["image_sha256"]
    assert ER.build_log_entry(ex, ["Hole"], image_bytes=data)["image_sha256"] == e["image_sha256"]
    assert ER.build_log_entry(ex, ["Hole"], image_bytes=b"other")["image_sha256"] != e["image_sha256"]
    assert "fake image" not in json.dumps(e)
    assert ER.build_log_entry(ex, ["Hole"])["image_sha256"] == ""


def test_the_timestamp_is_utc_when_not_given():
    e = ER.build_log_entry(extraction(["Hole"], ["Hole"]), ["Hole"])
    assert len(e["timestamp"]) == 20 and e["timestamp"].endswith("Z") and e["version"] == ER.LOG_VERSION


# ---------------------------------------------------------------- the log file
def test_append_and_read_round_trip(tmp_path):
    path = str(tmp_path / "deep" / "dir" / "log.jsonl")
    e1 = ER.build_log_entry(extraction(["Hole"], ["Hole"]), ["Hole"], drawing_id="a")
    e2 = ER.build_log_entry(extraction(["Hole"], ["Slot"]), ["Hole", "Slot"], drawing_id="b")
    ER.append_log(path, e1)
    ER.append_log(path, e2)
    entries, bad = ER.read_log(path)
    assert bad == 0 and [e["drawing_id"] for e in entries] == ["a", "b"] and entries[1]["final_features"] == ["Hole", "Slot"]
    assert open(path, encoding="utf-8").read().count("\n") == 2          # one line per entry


def test_reading_a_missing_log_gives_nothing(tmp_path):
    assert ER.read_log(str(tmp_path / "none.jsonl")) == ([], 0)


def test_unreadable_lines_are_counted_not_fatal(tmp_path):
    path = tmp_path / "log.jsonl"
    good = json.dumps(ER.build_log_entry(extraction(["Hole"], ["Hole"]), ["Hole"]))
    path.write_text(good + "\n\n{broken json\n[1, 2]\n" + json.dumps({"x": 1}) + "\n" + good + "\n", encoding="utf-8")
    entries, bad = ER.read_log(str(path))
    assert len(entries) == 2 and bad == 3


def test_the_log_survives_non_ascii_text(tmp_path):
    path = str(tmp_path / "log.jsonl")
    ex = extraction(report(["Hole"], rejected=["Ø8 Bohrung"]), ["Hole"], rejected=["Ø8 Bohrung"])
    ER.append_log(path, ER.build_log_entry(ex, ["Hole"]))
    entries, bad = ER.read_log(path)
    assert bad == 0 and entries[0]["model_output"]["rejected_features"] == ["Ø8 Bohrung"]


# ---------------------------------------------------------------- precision / recall / F1
def test_prf_values():
    r = ER.prf(3, 1, 2)
    assert r["precision"] == pytest.approx(0.75) and r["recall"] == pytest.approx(0.6) and r["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert (r["tp"], r["fp"], r["fn"]) == (3, 1, 2)


def test_prf_is_undefined_where_nothing_was_predicted_or_expected():
    assert ER.prf(0, 0, 0) == {"tp": 0, "fp": 0, "fn": 0, "precision": None, "recall": None, "f1": None}
    r = ER.prf(0, 0, 2)
    assert r["precision"] is None and r["recall"] == 0.0 and r["f1"] is None
    r = ER.prf(0, 2, 0)
    assert r["precision"] == 0.0 and r["recall"] is None and r["f1"] is None
    r = ER.prf(0, 1, 1)
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0


# ---------------------------------------------------------------- summary of the log
def entries_for_summary():
    e1 = ER.build_log_entry(extraction(["Hole", "Slot"], ["Hole", "Thread"]), ["Hole", "Slot"],
                            {"Slot": "accept", "Thread": "reject"})                                 # 0 corrections, 2 decisions
    e2 = ER.build_log_entry(extraction(["Hole"], ["Hole"]), ["Hole", "Pocket"])                      # 1 correction (Pocket missed)
    ex3 = extraction(report(["Hole", "Boss"], rejected=["Counter Bore"]), ["Hole"], rejected=["Counter Bore"])
    e3 = ER.build_log_entry(ex3, ["Counterbore", "Hole", "Pocket"], {"Boss": "reject"}, {"Counter Bore": "Counterbore"})   # map + add
    e4 = ER.build_log_entry({"success": True, "features": ["Hole"], "rejected_features": []}, ["Hole"])    # legacy, 0
    return [e1, e2, e3, e4]


def test_summary_of_an_empty_log():
    assert ER.summarize_log([]) == {"n_drawings": 0}
    assert "empty" in ER.format_log_report({"n_drawings": 0}, "x.jsonl")


def test_summary_numbers():
    s = ER.summarize_log(entries_for_summary())
    assert s["n_drawings"] == 4
    assert s["corrections"]["mean"] == pytest.approx((0 + 1 + 2 + 0) / 4) and s["corrections"]["max"] == 2
    assert s["corrections"]["median"] == pytest.approx(0.5) and s["corrections"]["share_zero"] == pytest.approx(0.5)
    assert s["decisions_mean"] == pytest.approx((2 + 0 + 1 + 0) / 4)
    assert s["highlighted_mean"] == pytest.approx((2 + 0 + 1 + 0) / 4)
    assert s["without_agreement_info"] == 1


def test_summary_counts_what_the_model_missed_and_invented():
    s = ER.summarize_log(entries_for_summary())
    missed, invented = dict(s["most_missed"]), dict(s["most_invented"])
    assert missed == {"Pocket": 2, "Counterbore": 1} and invented == {"Thread": 1, "Boss": 1}
    # model output vs final: e1 model {Hole,Slot,Thread} final {Hole,Slot}; e2 {Hole}/{Hole,Pocket}; e3 {Hole,Boss}/{Counterbore,Hole,Pocket};
    # e4 {Hole}/{Hole}
    m = s["model_vs_final"]
    assert (m["tp"], m["fp"], m["fn"]) == (2 + 1 + 1 + 1, 1 + 0 + 1 + 0, 0 + 1 + 2 + 0)
    assert s["exact_share"] == pytest.approx(0.25)
    assert s["unrecognised"] == [("Counter Bore", 1)] and s["mapped"] == [("Counter Bore", "Counterbore", 1)]


def test_the_report_is_readable_and_says_what_the_log_is_not():
    s = ER.summarize_log(entries_for_summary())
    text = ER.format_log_report(s, "results/corrections_log.jsonl", bad=2)
    assert "4 drawing(s)" in text and "2 unreadable line(s) skipped" in text
    assert "corrections per drawing: mean 0.75" in text and "Pocket x2" in text
    assert "Counter Bore -> Counterbore x1" in text
    assert "without per-pass data" in text
    assert "not an independent ground truth" in text


def test_the_report_has_no_mapping_lines_when_nothing_was_mapped():
    s = ER.summarize_log([ER.build_log_entry(extraction(["Hole"], ["Hole"]), ["Hole"])])
    text = ER.format_log_report(s)
    assert "mapped by the user" not in text and "unrecognised names" not in text and "without per-pass data" not in text
    assert "drawings needing none: 100%" in text
