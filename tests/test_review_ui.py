"""Step 20 (WP7, report 6.5), the review of an extraction in week6/app.py: highlighted features, Accept / Reject, Generate disabled until
every highlighted feature is decided, name mapping, evidence, and the corrections log.

Headless (Streamlit's AppTest), no network. The extraction result is put in the session state, which is what the app does after the
extractor returns (the extractor itself is tested in tests/test_extraction_passes.py); one test uploads a file to check the option
that is passed to the extractor.
"""
import json
import os
import struct
import sys
import zlib

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
APP = os.path.join(ROOT, "week6", "app.py")
MANUAL = "✍️ Select the features myself"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    monkeypatch.setenv("CAPP_CORRECTIONS_LOG", str(tmp_path / "log" / "corrections.jsonl"))


def log_lines(tmp_path):
    path = tmp_path / "log" / "corrections.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def png(w=4, h=4):
    """A real, tiny PNG (grey), built by hand so that no imaging library is needed."""
    def chunk(kind, data):
        c = struct.pack(">I", len(data)) + kind + data
        return c + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x80" * w for _ in range(h))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def report(features, rejected=(), evidence=None, conf="high", notes=""):
    return {"success": True, "features": list(features), "rejected_features": list(rejected), "confidence": conf, "notes": notes,
            "evidence": dict(evidence or {})}


def extraction(g=("Hole", "Slot"), a=("Hole", "Thread"), rejected=(), g_ev=None, a_ev=None, a_rejected=()):
    union = list(dict.fromkeys(list(g) + list(a)))
    return {"success": True, "confidence": "high", "features": union, "notes": "", "rejected_features": list(rejected),
            "sample_agreement": {f: (f in g) + (f in a) for f in union}, "passes_ok": 2,
            "passes": {"geometry": report(g, evidence=g_ev), "annotation": report(a, rejected=a_rejected or rejected, evidence=a_ev)}}


def legacy(features=("Hole", "Turning")):
    return {"success": True, "confidence": "high", "features": list(features), "notes": "", "rejected_features": []}


def start(ex, confirmed=None, image=None, name="part.png"):
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["extraction_result"] = ex
    at.session_state["confirmed_features"] = list(ex["features"] if confirmed is None else confirmed)
    if image is not None:
        at.session_state["uploaded_image_bytes"] = image
        at.session_state["uploaded_name"] = name
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def started_review(**kw):
    """The extraction as the app starts it after the Extract button: only what both passes saw is in the list."""
    ex = extraction(**kw)
    from capp import extraction_review as ER
    return start(ex, confirmed=ER.initial_features(ER.build_review(ex)))


def click(at, key):
    at.button(key=key).click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def generate_button(at):
    return [b for b in at.button if "Generate" in b.label][0]


def generate(at):
    generate_button(at).click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def keys(at):
    return {b.key for b in at.button if b.key}


def headers(at):
    return [m.value for m in at.markdown if "step-hdr" in m.value]


def table(at):
    """The feature table of the review panel as a list of dicts."""
    frames = [d for d in at.dataframe if "Seen by" in list(d.value.columns)]
    assert len(frames) == 1, "the review table should be on the page once"
    return frames[0].value.to_dict("records")


# ---------------------------------------------------------------- Generate is held back until the highlighted features are decided
def test_generate_is_disabled_while_highlighted_features_are_undecided():
    at = started_review()
    assert at.session_state["confirmed_features"] == ["Hole"]
    assert generate_button(at).disabled
    warning = " ".join(w.value for w in at.warning)
    assert "2 feature(s) seen by one pass only" in warning and "Slot, Thread" in warning
    assert any("Generate stays off" in c.value for c in at.caption)


def test_each_highlighted_feature_has_accept_and_reject_and_a_preaccepted_one_has_none():
    at = started_review()
    assert keys(at) >= {"acc_Slot", "rej_Slot", "acc_Thread", "rej_Thread"}
    assert not keys(at) & {"acc_Hole", "rej_Hole"}


def test_the_table_shows_seen_by_confidence_and_status():
    rows = {r["Feature"]: r for r in table(started_review())}
    assert rows["Hole"]["Seen by"] == "both passes" and rows["Hole"]["Confidence"] == "high" and rows["Hole"]["Status"] == "pre-accepted"
    assert rows["Slot"]["Seen by"] == "geometry only" and rows["Slot"]["Confidence"] == "medium"
    assert rows["Slot"]["Status"] == "needs confirmation"
    assert rows["Thread"]["Seen by"] == "annotation only"


def test_accepting_puts_the_feature_in_the_list_and_deciding_all_enables_generate():
    at = started_review()
    click(at, "acc_Slot")
    assert at.session_state["confirmed_features"] == ["Hole", "Slot"] and at.session_state["feature_decisions"] == {"Slot": "accept"}
    assert generate_button(at).disabled                                            # Thread is still undecided
    assert "1 feature(s) seen by one pass only" in " ".join(w.value for w in at.warning)
    click(at, "rej_Thread")
    assert at.session_state["confirmed_features"] == ["Hole", "Slot"]
    assert not generate_button(at).disabled and not any("seen by one pass only" in w.value for w in at.warning)
    assert {r["Feature"]: r["Status"] for r in table(at)} == {"Hole": "pre-accepted", "Slot": "accepted", "Thread": "rejected"}


def test_a_decision_can_be_changed():
    at = started_review()
    click(at, "acc_Slot")
    click(at, "rej_Slot")
    assert "Slot" not in at.session_state["confirmed_features"] and at.session_state["feature_decisions"]["Slot"] == "reject"
    assert at.button(key="rej_Slot").disabled and not at.button(key="acc_Slot").disabled
    click(at, "acc_Slot")
    assert "Slot" in at.session_state["confirmed_features"]


def test_adding_a_highlighted_feature_in_the_list_counts_as_accepting_it():
    at = started_review()
    [m for m in at.multiselect if m.key == "confirmed_features"][0].set_value(["Hole", "Slot", "Thread"]).run()
    assert not at.exception and not generate_button(at).disabled


def test_generate_runs_the_plan_once_everything_is_decided():
    at = started_review()
    click(at, "acc_Slot")
    click(at, "rej_Thread")
    generate(at)
    assert any("STEP 5" in h for h in headers(at)) and any("Token sequence" in s.value for s in at.success)


def test_a_result_without_per_pass_data_is_not_held_back():
    at = start(legacy())
    assert not generate_button(at).disabled and not at.warning
    assert not [k for k in keys(at) if k.startswith(("acc_", "rej_"))]
    assert any("no per-pass data" in c.value for c in at.caption)
    assert {r["Status"] for r in table(at)} == {"pre-accepted"}
    generate(at)
    assert any("STEP 5" in h for h in headers(at))


def test_when_only_one_pass_worked_every_feature_is_highlighted():
    ex = extraction()
    ex["passes"]["annotation"] = {"success": False, "error": "timeout"}
    ex["features"], ex["sample_agreement"], ex["passes_ok"] = ["Hole", "Slot"], {"Hole": 1, "Slot": 1}, 1
    at = start(ex, confirmed=[])
    assert generate_button(at).disabled and keys(at) >= {"acc_Hole", "acc_Slot"}
    assert any("Only 1 of the 2 passes worked" in w.value for w in at.warning)
    click(at, "acc_Hole")
    click(at, "rej_Slot")
    assert not generate_button(at).disabled and at.session_state["confirmed_features"] == ["Hole"]


def test_an_extraction_without_a_single_feature_cannot_be_generated():
    at = start(extraction(g=(), a=()), confirmed=[])
    generate(at)
    assert any("select at least one feature" in e.value for e in at.error)


# ---------------------------------------------------------------- evidence
def test_the_evidence_is_shown_with_the_feature():
    at = started_review(g_ev={"Hole": "circle in front view", "Slot": "long rectangle"}, a_ev={"Hole": "dia 8 callout", "Thread": "note M10"})
    rows = {r["Feature"]: r for r in table(at)}
    assert rows["Hole"]["Evidence"] == "geometry: circle in front view; annotation: dia 8 callout"
    assert rows["Thread"]["Evidence"] == "annotation: note M10"
    assert any("note M10" in m.value and "Thread" in m.value for m in at.markdown)         # also beside the Accept / Reject buttons


def test_without_evidence_the_notes_of_the_passes_are_shown_instead():
    ex = extraction()
    ex["passes"]["geometry"]["notes"] = "slot in the top view"
    at = start(ex, confirmed=["Hole"])
    assert {r["Evidence"] for r in table(at)} == {"-"}
    assert any("geometry: slot in the top view" in c.value for c in at.caption)


# ---------------------------------------------------------------- unrecognised names
def test_an_unrecognised_name_can_be_mapped_to_its_nearest_feature():
    at = started_review(rejected=["Counter Bore"])
    assert "map_Counter Bore" in keys(at) and at.button(key="map_Counter Bore").label == "Use Counterbore"
    click(at, "map_Counter Bore")
    assert "Counterbore" in at.session_state["confirmed_features"] and at.session_state["unrec_mappings"] == {"Counter Bore": "Counterbore"}
    assert "map_Counter Bore" not in keys(at) and any("was mapped to Counterbore" in c.value for c in at.caption)
    assert generate_button(at).disabled                                  # mapping does not decide the highlighted features


def test_a_name_with_no_close_feature_gets_a_note_and_no_button():
    at = started_review(rejected=["Gear"])
    assert not [k for k in keys(at) if k.startswith("map_")]
    assert any("“Gear”" in c.value and "no close match" in c.value for c in at.caption)


def test_the_old_warning_about_dropped_names_points_to_the_panel():
    at = started_review(rejected=["Gear"])
    assert any("Not in the feature vocabulary" in w.value and "Gear" in w.value for w in at.warning)


# ---------------------------------------------------------------- the drawing beside the table
def test_the_drawing_is_shown_beside_the_table():
    at = start(extraction(), confirmed=["Hole"], image=png())
    assert len(at.get("image")) == 1


def test_no_image_in_the_session_means_no_image_and_no_error():
    at = start(extraction(), confirmed=["Hole"])
    assert len(at.get("image")) == 0 and table(at)


def test_a_drawing_that_cannot_be_shown_does_not_hide_the_table():
    at = start(extraction(), confirmed=["Hole"], image=b"this is not an image")
    assert len(table(at)) == 3


def test_the_review_panel_is_not_shown_in_manual_mode():
    at = started_review()
    at.radio(key="input_method").set_value(MANUAL).run()
    assert not at.exception and not [d for d in at.dataframe if "Seen by" in list(d.value.columns)]
    assert not generate_button(at).disabled                           # a pending upload review never blocks manual input


def test_manual_mode_is_never_blocked_nor_logged(tmp_path):
    at = started_review()
    at.radio(key="input_method").set_value(MANUAL).run()
    [m for m in at.multiselect if m.label == "Features"][0].set_value(["Slot"]).run()
    generate(at)
    assert any("STEP 5" in h for h in headers(at)) and log_lines(tmp_path) == []


def test_a_failed_extraction_shows_the_error_and_no_panel():
    at = start({"success": False, "features": [], "rejected_features": [], "confidence": None, "notes": "", "error": "boom"}, confirmed=[])
    assert any("Extraction failed: boom" in e.value for e in at.error)
    assert not [d for d in at.dataframe if "Seen by" in list(d.value.columns)]


# ---------------------------------------------------------------- the corrections log
def test_generate_writes_one_line_with_the_decisions_and_corrections(tmp_path):
    at = started_review(rejected=["Counter Bore"])
    click(at, "acc_Slot")
    click(at, "rej_Thread")
    click(at, "map_Counter Bore")
    generate(at)
    [e] = log_lines(tmp_path)
    assert e["final_features"] == ["Hole", "Slot", "Counterbore"]
    assert e["n_decisions"] == 2 and e["n_corrections"] == 1 and e["n_highlighted"] == 2
    assert {(x["type"], x["feature"]) for x in e["events"]} == {("accepted", "Slot"), ("rejected", "Thread"), ("mapped", "Counterbore")}
    assert e["model_output"]["features"] == ["Hole", "Slot", "Thread"] and e["model_output"]["highlighted"] == ["Slot", "Thread"]
    assert e["model_output"]["unrecognised"] == ["Counter Bore"]


def test_a_removed_preaccepted_feature_and_an_added_one_are_logged_as_corrections(tmp_path):
    at = started_review()
    click(at, "acc_Slot")
    click(at, "rej_Thread")
    [m for m in at.multiselect if m.key == "confirmed_features"][0].set_value(["Slot", "Pocket"]).run()           # Hole out, Pocket in
    generate(at)
    [e] = log_lines(tmp_path)
    kinds = {(x["type"], x["feature"]): x["correction"] for x in e["events"]}
    assert kinds[("removed", "Hole")] is True and kinds[("added", "Pocket")] is True and kinds[("accepted", "Slot")] is False
    assert e["n_corrections"] == 2


def test_the_drawing_is_logged_by_name_and_hash_not_stored(tmp_path):
    data = png()
    at = start(extraction(g=("Hole",), a=("Hole",)), confirmed=["Hole"], image=data, name="flange.png")
    generate(at)
    [e] = log_lines(tmp_path)
    import hashlib
    assert e["drawing_id"] == "flange.png" and e["image_sha256"] == hashlib.sha256(data).hexdigest()[:12]
    assert e["n_corrections"] == 0 and e["n_highlighted"] == 0


def test_generating_again_in_the_same_state_does_not_log_the_drawing_twice(tmp_path):
    at = started_review()
    click(at, "acc_Slot")
    click(at, "rej_Thread")
    generate(at)
    generate(at)
    assert len(log_lines(tmp_path)) == 1


def test_generating_again_after_a_change_is_a_new_line(tmp_path):
    at = started_review()
    click(at, "acc_Slot")
    click(at, "rej_Thread")
    generate(at)
    at.run()                                                     # any later interaction draws the review panel again (the button is not pressed now)
    click(at, "acc_Thread")                                      # the user changes a decision and generates again
    generate(at)
    lines = log_lines(tmp_path)
    assert len(lines) == 2 and lines[0]["final_features"] == ["Hole", "Slot"] and lines[1]["final_features"] == ["Hole", "Slot", "Thread"]


def test_a_legacy_result_is_logged_as_having_no_agreement_info(tmp_path):
    at = start(legacy(("Hole", "Turning")), confirmed=["Hole", "Turning", "Slot"])
    generate(at)
    [e] = log_lines(tmp_path)
    assert e["model_output"]["has_agreement"] is False and e["n_corrections"] == 1 and e["n_decisions"] == 0
    assert [(x["type"], x["feature"]) for x in e["events"]] == [("added", "Slot")]


def test_a_log_that_cannot_be_written_does_not_stop_the_page(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a folder is needed", encoding="utf-8")
    monkeypatch.setenv("CAPP_CORRECTIONS_LOG", str(blocker / "log.jsonl"))
    at = started_review()
    click(at, "acc_Slot")
    click(at, "rej_Thread")
    generate(at)
    assert any("STEP 5" in h for h in headers(at)) and any("corrections log could not be written" in c.value for c in at.caption)


def test_a_failed_write_is_tried_again_at_the_next_generate(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("CAPP_CORRECTIONS_LOG", str(blocker / "log.jsonl"))
    at = started_review()
    click(at, "acc_Slot")
    click(at, "rej_Thread")
    generate(at)
    assert "_last_logged" not in at.session_state


# ---------------------------------------------------------------- the evidence option reaches the extractor
needs_uploader = pytest.mark.skipif(not hasattr(AppTest, "file_uploader"), reason="this Streamlit's AppTest cannot upload a file")


@pytest.fixture
def fake_extractor(monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "week1"))
    import image_feature_extractor as ife
    calls = []

    def fake(image_bytes, image_format="png", with_dimensions=False, with_evidence=False):
        calls.append({"format": image_format, "with_dimensions": with_dimensions, "with_evidence": with_evidence})
        return extraction(g_ev={"Hole": "circle"} if with_evidence else None)
    monkeypatch.setattr(ife, "extract_features_from_image", fake)
    return calls


def extract_through_the_ui(evidence):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.file_uploader[0].set_value(("part.png", png(), "image/png")).run()
    if evidence:
        [c for c in at.checkbox if "evidence" in c.label][0].set_value(True).run()
    [b for b in at.button if "Extract Features" in b.label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@needs_uploader
def test_the_evidence_option_is_off_by_default_and_is_passed_on_when_ticked(fake_extractor):
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert [c for c in at.checkbox if "evidence" in c.label][0].value is False
    extract_through_the_ui(evidence=False)
    extract_through_the_ui(evidence=True)
    assert [c["with_evidence"] for c in fake_extractor] == [False, True] and all(c["with_dimensions"] for c in fake_extractor)


@needs_uploader
def test_extracting_starts_a_fresh_review(fake_extractor):
    at = extract_through_the_ui(evidence=False)
    assert at.session_state["confirmed_features"] == ["Hole"]                 # only what both passes saw
    assert at.session_state["feature_decisions"] == {} and at.session_state["unrec_mappings"] == {} and at.session_state["uploaded_name"] == "part.png"
    assert generate_button(at).disabled and len(at.get("image")) == 1
    click(at, "acc_Slot")
    assert at.session_state["feature_decisions"] == {"Slot": "accept"}
    [b for b in at.button if "Extract Features" in b.label][0].click().run()          # extracting again forgets the old decisions
    assert at.session_state["feature_decisions"] == {} and at.session_state["confirmed_features"] == ["Hole"]
