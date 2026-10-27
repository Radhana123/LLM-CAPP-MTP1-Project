"""Step 19 (WP9), the document: capp/routesheet.py (Markdown, HTML, JSON, XLSX, PDF from one display model) and scripts/route_sheet.py.

What is checked: every format shows the same content; the DRAFT banner is there and tells the truth about the status of the tables (and
changes only when every range used is confirmed); nothing from a part or a planner is allowed to break a page (HTML escaping, characters
the PDF fonts cannot draw); PDF and XLSX fail with a clear message when their library is missing; the CLI works on typed parts and on run
folders. No API calls, no network.
"""
import copy
import io
import json
import os
import sys

import pytest

from capp import analysis_view as AV
from capp import dataset, harness, systems
from capp import process_plan as PP
from capp import process_tables as PT
from capp import routesheet as RS
from capp import verify as V

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import route_sheet as CLI      # noqa: E402

MARKER = V.MARKER
G = "2026-10-27 12:00 UTC"
DIMS = {"diameter": 40, "length": 120, "depth": 15, "width": 32}
STEPS = ["Facing", "Plain/Cylindrical Turning", MARKER, "Slot Milling", "Center Drilling", "Drilling", "Chamfering", "Inspection"]


def make_plan(material="Steel", steps=STEPS, part_id="UI", hole=8, dims=DIMS, context=None, features=("Turning", "Slot", "Hole", "Chamfer"), ra=1.6, **kw):
    p = AV.make_part(material, list(features), "0.05mm", 100, dims=dims, required_ra_um=ra, hole_diameter=hole if dims else None)
    p["part_id"] = part_id
    return PP.build_plan(steps, p, context=context, **kw)


@pytest.fixture(scope="module")
def plan():
    return make_plan()


# ---------------------------------------------------------------- the display model
def test_display_rows_have_every_column_and_one_changeover_row(plan):
    rows = RS.display_rows(plan)
    ops = [r for r in rows if r["kind"] == "op"]
    assert len(ops) == 7 and sum(r["kind"] == "changeover" for r in rows) == 1
    keys = [k for k, _ in RS.COLUMNS]
    for r in ops:
        assert set(keys) <= set(r) and "description" in r
    co = next(r for r in rows if r["kind"] == "changeover")
    assert "lathe -> milling" in co["text"] and "15 min" in co["text"]
    assert rows[rows.index(co) + 1]["operation"] == "Slot Milling"                          # the row sits right before the first operation on the new machine


def test_display_rows_numbers_and_markers(plan):
    ops = {r["operation"]: r for r in RS.display_rows(plan) if r["kind"] == "op"}
    assert ops["Facing"]["seq"] == "10"
    assert ops["Drilling"]["tool"].startswith("twist drill, Ø8 mm;") and "*" not in ops["Drilling"]["tool"]        # the hole diameter was given
    assert "mm/tooth" in ops["Slot Milling"]["feed"] and "mm/min" in ops["Slot Milling"]["feed"]
    assert ops["Inspection"]["machine"] == "-" and ops["Inspection"]["vc"] == "-" and ops["Inspection"]["time"] == "5.00"
    assert ops["Facing"]["machine"] == "lathe"                                                                    # the lathe operations decide it: no *


def test_assumed_values_are_starred_in_the_table():
    pl = make_plan(hole=None)
    ops = {r["operation"]: r for r in RS.display_rows(pl) if r["kind"] == "op"}
    assert "Ø40 mm*" in ops["Drilling"]["tool"]
    pl = make_plan(steps=["Facing", "Drilling", "Inspection"])
    assert RS.display_rows(pl)[0]["machine"] == "lathe*"


def test_order_edges_are_not_repeated_in_the_rule_column(plan):
    for r in RS.display_rows(plan):
        if r["kind"] == "op":
            assert "ORD-EDGE" not in r["why"]
    first = RS.display_rows(plan)[0]
    assert "[ORD-FIRST]" in first["why"]


def test_high_spindle_load_is_noted(plan):
    notes = " ".join(r["notes"] for r in RS.display_rows(plan) if r["kind"] == "op")
    assert "load " in notes


def test_the_banner_reports_status_counts_and_that_nothing_is_confirmed(plan):
    b = RS.draft_banner(plan)
    assert b.startswith("DRAFT FOR REVIEW") and "none is confirmed against a handbook" in b and "No machining trial" in b
    n = plan["tables"]["ranges_used"]
    assert f"Of the {n} reference ranges used" in b


def test_the_banner_changes_only_when_every_range_used_is_confirmed(plan):
    mixed = copy.deepcopy(plan)
    mixed["tables"]["status"] = dict(mixed["tables"]["status"], confirmed=1)
    assert "none is confirmed" not in RS.draft_banner(mixed) and "DRAFT FOR REVIEW" in RS.draft_banner(mixed)
    allc = copy.deepcopy(plan)
    n = allc["tables"]["ranges_used"]
    allc["tables"]["status"] = {"confirmed": n, "web": 0, "mixed": 0, "indicative": 0}
    b = RS.draft_banner(allc)
    assert "confirmed against a handbook (see Sources)" in b and "none is confirmed" not in b
    assert "DRAFT FOR REVIEW" in b                                       # still a draft: tool rules are indicative, no trial


def test_the_banner_with_no_range_used():
    pl = make_plan(steps=["Inspection"])
    assert "No reference range was used." in RS.draft_banner(pl)


def test_header_items_for_a_typed_part_and_a_run_record(plan):
    h = dict(RS.header_items(plan, G))
    assert h["Part"] == "UI" and h["Material"] == "Steel" and h["Generated"] == G and "hole Ø8" in h["Dimensions"]
    assert h["Required Ra"] == "1.6 µm" and "7.5 kW" in h["Machines"]
    ctx = {"system": "n5_r1_s3", "source": "repaired_llm", "confidence": 0.62, "flagged": True, "confidence_flag": 0.8}
    h = dict(RS.header_items(make_plan(context=ctx), G))
    assert h["Route source"] == "repaired llm (n5_r1_s3)"
    assert h["Plan confidence"] == "0.62 (FLAGGED for human review, threshold 0.8)"
    h = dict(RS.header_items(make_plan(context=dict(ctx, flagged=False)), G))
    assert "not flagged" in h["Plan confidence"]
    ref = dict(RS.header_items(make_plan(dims=None, hole=None), G))
    assert "reference part of v1" in ref["Dimensions"]


def test_sections_content(plan):
    secs = dict(RS.sections(plan, G))
    assert list(secs)[0] == "Totals" and "Checks" in secs and "Assumptions" in secs and "Sources and status of the tables" in secs
    assert any("PASSED" in x for x in secs["Checks"])
    assert any("Estimated time per part" in x for x in secs["Totals"])
    assert any(x.startswith("[S") for x in secs["Sources and status of the tables"])
    assert any(x.startswith("reference range ") for x in secs["Sources and status of the tables"])
    assert "Before and after repair" not in secs and "Stated reasoning of the planner" not in secs


def test_sections_say_when_nothing_was_found_and_when_something_was():
    clean = dict(RS.sections(make_plan(steps=["Facing", "Center Drilling", "Drilling", "Inspection"], features=("Hole",), ra=None), G))["Checks"]
    assert any("no hard finding and no risk" in x for x in clean)
    risky = dict(RS.sections(make_plan(steps=["Facing", "Drilling", "Inspection"], features=("Hole",)), G))["Checks"]
    assert any("risk PHYS:HOLE-PILOT" in x for x in risky) and not any("no hard finding and no risk" in x for x in risky)
    bad = dict(RS.sections(make_plan(steps=["Facing", "Whittling", "Inspection"], features=("Hole",)), G))["Checks"]
    assert any(x.startswith("Rule check of the route") and "FAILED" in x for x in bad)


def test_sections_show_repair_and_reasoning():
    ctx = {"raw_steps": ["Facing", "Drilling", "Whittling", "Inspection"], "reasoning": "Hole needs a pilot.\nSecond line."}
    pl = make_plan(steps=["Facing", "Center Drilling", "Drilling", "Inspection"], context=ctx)
    secs = dict(RS.sections(pl, G))
    rep = secs["Before and after repair"]
    assert any("different" in x for x in rep) and any(x.startswith("raw route: Facing > Drilling > Whittling") for x in rep)
    assert any("Center Drilling+" in x for x in rep) and any(x.startswith("removed: Whittling") for x in rep)
    assert secs["Stated reasoning of the planner"] == ["Hole needs a pilot.\nSecond line."]
    same = make_plan(steps=["Facing", "Drilling", "Inspection"], context={"raw_steps": ["Facing", "Drilling", "Inspection"]})
    assert any("identical" in x for x in dict(RS.sections(same, G))["Before and after repair"])


def test_tables_used_lists_what_the_plan_used_once_each(plan):
    tools, ranges = RS.tables_used(plan)
    keys = [r["key"] for r in tools]
    assert len(keys) == len(set(keys)) and "Facing | Steel" in keys and "Slot Milling | Steel" in keys
    rkeys = [r["key"] for r in ranges]
    assert len(rkeys) == len(set(rkeys)) and set(rkeys) <= {f"{c} | Steel" for c in ("turn", "drill", "mill")}
    assert all(r["status"] in PT.STATUSES for r in ranges) and all(r["sources"] for r in ranges if r["status"] != "indicative")
    assert 1 <= len(ranges) <= plan["tables"]["ranges_used"]                # one row per speed class, however many operations used it


def test_a_machine_specific_tool_row_is_listed_under_its_machine():
    pl = make_plan(steps=["Facing", "Plain/Cylindrical Turning", MARKER, "Slot Milling", "Chamfering", "Inspection"])
    keys = [r["key"] for r in RS.tables_used(pl)[0]]
    assert "Chamfering on milling | Steel" in keys and "Chamfering | Steel" not in keys


# ---------------------------------------------------------------- text formats
def test_markdown_has_banner_table_sections_and_approval(plan):
    md = RS.to_markdown(plan, G)
    assert md.startswith("# ROUTE SHEET") and "> **DRAFT FOR REVIEW" in md
    assert md.count("\n| ") >= 8 + 1                                  # header, separator is on its own line; 7 operations + 1 changeover row
    assert "*MACHINE CHANGEOVER: lathe -> milling" in md
    for title in ("## Totals", "## Checks", "## Assumptions", "## Sources and status of the tables", "## Approval"):
        assert title in md
    assert "Prepared by: ____________" in md and G in md


def test_markdown_table_cells_cannot_break_the_table():
    pl = make_plan(part_id="A|B")
    pl["operations"][0]["flags"].append("pipe | in a note")
    md = RS.to_markdown(pl, G)
    row = next(l for l in md.splitlines() if l.startswith("| 10 "))
    assert row.count("|") == len(RS.COLUMNS) + 1


def test_json_round_trips_and_carries_banner_and_time(plan):
    data = json.loads(RS.to_json(plan, G))
    assert data["generated_at"] == G and data["banner"] == RS.draft_banner(plan)
    assert [e["op"] for e in data["operations"]] == [e["op"] for e in plan["operations"]]
    assert data["totals"]["total_min"] == pytest.approx(plan["totals"]["total_min"])


def test_html_is_a_complete_page_with_the_banner_and_every_operation(plan):
    h = RS.to_html(plan, G)
    assert h.startswith("<!doctype html>") and "<title>ROUTE SHEET UI</title>" in h and "class='banner'" in h
    assert h.count("<tr>") >= 8 and "class='co'" in h and "Approved by" in h
    for e in plan["operations"]:
        assert e["op"].replace("&", "&amp;") in h or e["op"] in h
    assert "@page{size:A4 landscape" in h and "<script" not in h


def test_html_escapes_everything_that_comes_from_a_part_or_a_planner():
    ctx = {"reasoning": "<script>alert(1)</script> & more", "system": "<b>x</b>"}
    pl = make_plan(part_id="<img src=x onerror=alert(1)>", context=ctx)
    h = RS.to_html(pl, G)
    assert "<script>" not in h and "<img src=x" not in h and "<b>x</b>" not in h
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; more" in h


def test_html_parses_as_well_formed_enough():
    from html.parser import HTMLParser

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack, self.bad = [], []

        def handle_starttag(self, tag, attrs):
            if tag not in ("meta", "br", "link"):
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if not self.stack or self.stack.pop() != tag:
                self.bad.append(tag)
    p = P()
    p.feed(RS.to_html(make_plan(), G))
    assert p.bad == [] and p.stack == []


# ---------------------------------------------------------------- XLSX
def test_xlsx_sheets_header_numbers_and_banner(plan):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.load_workbook(io.BytesIO(RS.to_xlsx_bytes(plan, G)))
    assert wb.sheetnames == ["Route sheet", "Checks and notes", "Tables used"]
    ws = wb["Route sheet"]
    assert ws["A1"].value == "ROUTE SHEET" and ws["A2"].value == RS.draft_banner(plan)
    head = next(r for r in range(1, ws.max_row + 1) if ws.cell(r, 1).value == "No.")
    assert [c.value for c in ws[head]] == [h for _, h in RS.COLUMNS]
    first = [c.value for c in ws[head + 1]]
    assert first[0] == 10 and isinstance(first[0], int)                      # numbers are numbers, so they can be summed and sorted
    assert all(isinstance(first[i], (int, float)) for i in (4, 5, 8))        # speed, spindle speed and time are numbers, not text
    assert first[4] == 120 and first[8] == pytest.approx(plan["operations"][0]["time_min"], abs=0.005)
    assert first[1].startswith("Facing\n")
    assert ws.freeze_panes == f"A{head + 1}"
    assert ws.page_setup.orientation == "landscape"
    assert sum(isinstance(ws.cell(r, 9).value, (int, float)) for r in range(head + 1, ws.max_row + 1)) == 7


def test_xlsx_time_column_sums_to_the_plan_total(plan):
    openpyxl = pytest.importorskip("openpyxl")
    ws = openpyxl.load_workbook(io.BytesIO(RS.to_xlsx_bytes(plan, G)))["Route sheet"]
    head = next(r for r in range(1, ws.max_row + 1) if ws.cell(r, 1).value == "No.")
    total = sum(ws.cell(r, 9).value for r in range(head + 1, ws.max_row + 1) if isinstance(ws.cell(r, 9).value, (int, float)))
    assert total == pytest.approx(plan["totals"]["total_min"], abs=0.05)       # each cell is printed to 2 decimals


def test_xlsx_notes_and_tables_sheets(plan):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.load_workbook(io.BytesIO(RS.to_xlsx_bytes(plan, G)))
    notes = [r[0].value for r in wb["Checks and notes"].iter_rows() if r[0].value]
    assert "Totals" in notes and "Assumptions" in notes
    used = list(wb["Tables used"].iter_rows(values_only=True))
    assert used[0][:5] == ("table", "key", "value", "status", "sources")
    assert any(r[0] == "tool" for r in used[1:]) and any(r[0] == "range" for r in used[1:]) and any(r[0] == "source" for r in used[1:])


def test_xlsx_missing_library_gives_a_clear_error(plan, monkeypatch):
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    with pytest.raises(RS.RouteSheetError, match="pip install openpyxl"):
        RS.to_xlsx_bytes(plan)


# ---------------------------------------------------------------- PDF
def test_pdf_is_a_landscape_pdf_with_the_banner_and_every_operation(plan):
    pytest.importorskip("reportlab")
    data = RS.to_pdf_bytes(plan, G)
    assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")
    pypdf = pytest.importorskip("pypdf")
    r = pypdf.PdfReader(io.BytesIO(data))
    assert len(r.pages) >= 1 and float(r.pages[0].mediabox.width) > float(r.pages[0].mediabox.height)
    text = "\n".join(p.extract_text() for p in r.pages)
    assert "ROUTE SHEET" in text and "DRAFT FOR REVIEW" in text and "Approved by" in text
    for op in ("Facing", "Slot Milling", "Center Drilling", "Inspection"):
        assert op in text
    assert "MACHINE CHANGEOVER" in text and "page 1" in text


def test_pdf_repeats_the_draft_footer_on_every_page():
    pytest.importorskip("reportlab")
    pypdf = pytest.importorskip("pypdf")
    steps = ["Facing"] + ["Plain/Cylindrical Turning", "Step Turning", "Taper Turning", "Grooving/Necking"] * 5 + ["Inspection"]
    r = pypdf.PdfReader(io.BytesIO(RS.to_pdf_bytes(make_plan(steps=steps), G)))
    assert len(r.pages) >= 2
    for i, p in enumerate(r.pages, start=1):
        t = p.extract_text()
        assert "DRAFT" in t and f"page {i}" in t


def test_pdf_text_never_breaks_on_characters_the_fonts_cannot_draw():
    assert RS.pdf_text("a → b ≥ c ₹5 – ‘x’") == "a -> b >= c Rs5 - 'x'"
    assert RS.pdf_text("Ø8 µm °") == "Ø8 um °"                               # Ø and ° are drawn by every viewer; µ is not, so it is written u
    out = RS.pdf_text("日本語 ok")
    assert out.endswith(" ok") and "?" in out and out.encode("cp1252")
    assert RS.pdf_text(None) == "" and RS.pdf_text(3.5) == "3.5"


def test_pdf_with_awkward_text_still_builds():
    pytest.importorskip("reportlab")
    ctx = {"reasoning": "日本語 <b>bold</b> & ≥ → ₹ " * 20, "system": "x&y"}
    data = RS.to_pdf_bytes(make_plan(part_id="P<1>&2", context=ctx), G)
    assert data.startswith(b"%PDF-")


def test_pdf_missing_library_gives_a_clear_error(plan, monkeypatch):
    monkeypatch.setitem(sys.modules, "reportlab", None)
    monkeypatch.setitem(sys.modules, "reportlab.lib", None)
    with pytest.raises(RS.RouteSheetError, match="pip install reportlab"):
        RS.to_pdf_bytes(plan)


# ---------------------------------------------------------------- write()
def test_write_chooses_the_format_by_extension(plan, tmp_path):
    for ext in (".md", ".html", ".htm", ".json"):
        path = str(tmp_path / f"s{ext}")
        assert RS.write(plan, path, G) == path
        text = open(path, encoding="utf-8").read()
        assert "DRAFT FOR REVIEW" in text and "Facing" in text
    pytest.importorskip("openpyxl")
    RS.write(plan, str(tmp_path / "s.xlsx"), G)
    assert (tmp_path / "s.xlsx").stat().st_size > 2000
    pytest.importorskip("reportlab")
    RS.write(plan, str(tmp_path / "s.PDF"), G)
    assert (tmp_path / "s.PDF").read_bytes().startswith(b"%PDF-")


def test_write_rejects_an_unknown_extension(plan, tmp_path):
    with pytest.raises(RS.RouteSheetError, match="unknown format"):
        RS.write(plan, str(tmp_path / "s.docx"), G)
    assert not (tmp_path / "s.docx").exists()


def test_the_same_plan_and_time_give_the_same_text(plan):
    assert RS.to_markdown(plan, G) == RS.to_markdown(plan, G)
    assert RS.to_html(plan, G) == RS.to_html(plan, G) and RS.to_json(plan, G) == RS.to_json(plan, G)


def test_a_missing_generated_at_is_filled_in(plan):
    assert "UTC" in RS.to_markdown(plan)


def test_every_material_and_every_single_operation_renders_in_every_text_format():
    for m in PT.MATERIALS:
        for op in ("Facing", "Drilling", "Slot Milling", "Tapping", "Inspection"):
            pl = make_plan(material=m, steps=[op])
            assert RS.to_markdown(pl, G) and RS.to_html(pl, G) and json.loads(RS.to_json(pl, G))


# ---------------------------------------------------------------- the CLI
def run_cli(*args):
    return CLI.main(list(args))


def test_cli_make_prints_markdown_and_writes_files(tmp_path, capsys):
    out = [str(tmp_path / "s.html"), str(tmp_path / "s.md"), str(tmp_path / "s.json")]
    code = run_cli("make", "--material", "Steel", "--features", "Hole,Chamfer,Slot", "--diameter", "40", "--length", "120", "--depth", "15",
                   "--width", "32", "--hole-diameter", "8", "--ra", "1.6", "--out", *out, "--generated-at", G)
    cap = capsys.readouterr()
    assert code == 0 and cap.out.count("wrote ") == 3 and "operations" in cap.err and "reference ranges used" in cap.err
    data = json.loads(open(out[2], encoding="utf-8").read())
    assert data["part"]["material"] == "Steel" and data["part"]["dims"]["hole_diameter"] == 8.0
    assert data["context"]["provenance"] == "route_builder"
    assert data["verifier"]["valid"] is True and data["totals"]["total_min"] > 0
    assert data["generated_at"] == G


def test_cli_make_without_out_prints_markdown(capsys):
    assert run_cli("make", "--material", "Aluminum", "--features", "Hole", "--generated-at", G) == 0
    cap = capsys.readouterr()
    assert cap.out.startswith("# ROUTE SHEET") and "reference part of v1" in cap.out


def test_cli_make_uses_a_given_route(capsys):
    assert run_cli("make", "--material", "Steel", "--features", "Hole", "--route", "Facing > Center Drilling > Drilling > Inspection") == 0
    cap = capsys.readouterr()
    assert "route given on the command line" in cap.out and "| 30 |" in cap.out


def test_cli_make_errors(capsys):
    assert run_cli("make", "--material", "Steel", "--features", "Hole,Foo") == 1
    assert "unknown feature(s): Foo" in capsys.readouterr().err
    assert run_cli("make", "--material", "Steel", "--features", "") == 1
    assert run_cli("make", "--material", "Steel", "--features", "Hole", "--diameter", "20") == 1
    assert "all four dimensions" in capsys.readouterr().err
    assert run_cli("make", "--material", "Steel", "--features", "Hole", "--hole-diameter", "6") == 1
    assert "needs the part dimensions" in capsys.readouterr().err
    assert run_cli("make", "--material", "Gold", "--features", "Hole") == 1
    assert "unknown material" in capsys.readouterr().err


def test_cli_make_unknown_extension_is_an_error_not_a_traceback(tmp_path, capsys):
    assert run_cli("make", "--material", "Steel", "--features", "Hole", "--out", str(tmp_path / "s.docx")) == 1
    assert "unknown format" in capsys.readouterr().err


def make_run(tmp_path, n=3, with_llm=True, threshold=0.8, repeats=1):
    parts = dataset.load_dataset("dev_v2")[:n]
    recs = []
    for p in parts:
        for r in range(repeats):
            rec = systems.run_rb(p).to_dict()
            rec["part"], rec["repeat"] = p, r
            if with_llm:
                final = rec["final"]
                raw = [s for s in final["steps"] if s not in (MARKER,)] + ["Whittling"]
                rec["candidates"] = [{"sample_id": 0, "name": final["name"], "llm_ok": True, "llm_steps": raw, "meta": {"reasoning": f"repeat {r}"}}]
                rec["info"] = dict(rec["info"], confidence=0.7, flagged=True, selection="s3")
                rec["final"] = dict(final, provenance="repaired_llm")
            recs.append(rec)
    d = tmp_path / "run"
    d.mkdir()
    with open(d / "records.jsonl", "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
    if threshold is not None:
        (d / "run.json").write_text(json.dumps({"confidence_flag_threshold": threshold}))
    return str(d), parts


def test_cli_from_run_writes_the_sheet_of_one_part(tmp_path, capsys):
    run, parts = make_run(tmp_path)
    pid = parts[1]["part_id"]
    out = str(tmp_path / "s.json")
    assert run_cli("from-run", run, "--part", pid, "--out", out, "--generated-at", G) == 0
    data = json.loads(open(out, encoding="utf-8").read())
    assert data["part"]["part_id"] == pid and data["context"]["provenance"] == "repaired_llm"
    assert data["context"]["confidence"] == 0.7 and data["context"]["flagged"] is True and data["context"]["confidence_flag"] == 0.8
    assert data["repair"]["removed"] == ["Whittling"] and data["context"]["reasoning"] == "repeat 0"
    assert data["part"]["dims"]["diameter"] == parts[1]["dims"]["diameter"]


def test_cli_from_run_threshold_override_and_default(tmp_path):
    run, parts = make_run(tmp_path, threshold=None)
    pid = parts[0]["part_id"]
    a, b = str(tmp_path / "a.json"), str(tmp_path / "b.json")
    assert run_cli("from-run", run, "--part", pid, "--out", a) == 0
    assert run_cli("from-run", run, "--part", pid, "--confidence-flag", "0.5", "--out", b) == 0
    assert json.load(open(a))["context"]["flagged"] is True                       # the flag the run recorded
    jb = json.load(open(b))["context"]
    assert jb["flagged"] is False and jb["confidence_flag"] == 0.5


def test_cli_from_run_repeats(tmp_path, capsys):
    run, parts = make_run(tmp_path, n=2, repeats=2)
    pid = parts[0]["part_id"]
    out = str(tmp_path / "s.json")
    assert run_cli("from-run", run, "--part", pid, "--out", out) == 0
    assert "2 repeats" in capsys.readouterr().err and json.load(open(out))["context"]["reasoning"] == "repeat 0"
    assert run_cli("from-run", run, "--part", pid, "--repeat", "1", "--out", out) == 0
    assert json.load(open(out))["context"]["reasoning"] == "repeat 1"
    assert run_cli("from-run", run, "--part", pid, "--repeat", "7") == 1
    assert "no repeat 7" in capsys.readouterr().err


def test_cli_from_run_errors(tmp_path, capsys):
    run, parts = make_run(tmp_path)
    assert run_cli("from-run", run, "--part", "NOPE") == 1
    err = capsys.readouterr().err
    assert "'NOPE' is not in this run" in err and parts[0]["part_id"] in err
    assert run_cli("from-run", str(tmp_path / "empty"), "--part", "X") == 1
    assert "no records.jsonl" in capsys.readouterr().err


def test_cli_tables_prints_and_exports_the_tables(tmp_path, capsys):
    assert run_cli("tables") == 0
    out = capsys.readouterr().out
    assert "range drill | Steel" in out and "confirmed 0" in out and "Nothing here is confirmed" in out and "[S1]" in out
    assert run_cli("tables", "--table", "tool") == 0
    assert "tool  Facing | Steel" in capsys.readouterr().out
    csv_path = str(tmp_path / "t.csv")
    assert run_cli("tables", "--csv", csv_path) == 0
    import csv
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    assert len(rows) == len(PT.rows()) and set(rows[0]) == {"table", "key", "value", "status", "sources", "note"}


def test_cli_tables_fails_loudly_when_a_table_is_inconsistent(monkeypatch, capsys):
    monkeypatch.setitem(PT.RANGES, ("drill", "Steel"), dict(PT.RANGES[("drill", "Steel")], status="confirmed", source_note=""))
    assert run_cli("tables") == 1
    assert "TABLE PROBLEMS" in capsys.readouterr().err
