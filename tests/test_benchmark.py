"""Expert benchmark pipeline (capp/benchmark.py, scripts/benchmark.py, the bench split of capp/dataset.py). Offline: the annotators are
simulated by filling the generated workbooks with openpyxl, no API call."""
import csv
import json
import os
import subprocess
import sys

import pytest

openpyxl = pytest.importorskip("openpyxl")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from capp import benchmark as B                                   # noqa: E402
from capp import compare                                          # noqa: E402
from capp import dataset as D                                     # noqa: E402
from capp import harness as H                                     # noqa: E402
from capp import summary as S                                     # noqa: E402
from capp import verify as V                                      # noqa: E402
import benchmark as cli                                           # noqa: E402  (scripts/benchmark.py)
import feature_vocab as fv                                        # noqa: E402
import route_builder as rb                                        # noqa: E402


# ---------------------------------------------------------------- fixtures
def rows():
    return [
        {"part_id": "P1", "name": "Flanged shaft", "source": "Textbook ex 3.1", "drawing": "fig3_1.png", "material": "Steel",
         "features": "Hole; Chamfer", "tolerance": "0.05mm", "batch_size": 100, "diameter": 50, "length": 120, "depth": 20, "width": 50,
         "hole_diameter": 12, "required_ra_um": 1.6, "notes": "from the exercise"},
        {"part_id": "P2", "name": "Threaded pin", "source": "Handbook p.40", "material": "Brass", "features": "Turning;Thread",
         "tolerance": "0.02mm", "batch_size": 500, "diameter": 20, "length": 80, "depth": 20, "width": 20},
        {"part_id": "P3", "name": "Bracket", "source": "Dept drawing 7", "material": "aluminium", "features": "Slot, Pocket, Hole",
         "tolerance": "0.1", "batch_size": 10, "diameter": 100, "length": 60, "depth": 30, "width": 80, "hole_diameter": 8},
        {"part_id": "P4", "name": "Bush", "source": "Textbook ex 4.2", "material": "castiron", "features": "Bore;Groove",
         "tolerance": "0.01mm", "batch_size": 50, "diameter": 60, "length": 40, "depth": 20, "width": 60, "hole_diameter": 30,
         "required_ra_um": 0.8},
        {"part_id": "P5", "name": "Cap", "source": "Textbook ex 4.3", "material": "Plastic", "features": "Face;Hole",
         "tolerance": "0.5mm", "batch_size": 1000, "diameter": 40, "length": 25, "depth": 10, "width": 40},
        {"part_id": "P6", "name": "Spacer", "source": "Handbook p.52", "material": "Titanium", "features": "Face;Chamfer",
         "tolerance": "0.005mm", "batch_size": 5, "diameter": 30, "length": 10, "depth": 10, "width": 30, "required_ra_um": 0.4},
    ]


@pytest.fixture(scope="module")
def spec():
    parts, errs = [], []
    for i, r in enumerate(rows(), 1):
        p, e = B.parse_part_row(r, i)
        parts.append(p)
        errs += e
    assert errs == []
    return parts


def route_for(part):
    return V.clean_steps(rb.generate_valid_routes(part["features"], max_routes=1)[0]["steps"])


def fill(path, plans, comments=None, review=None):
    """Simulate an annotator: plans = {(part_id, alt): [operations]} written into the Routes sheet; review = {part_id: (status, by, why)}."""
    wb = openpyxl.load_workbook(path)
    ws = wb["Routes"]
    head = [c.value for c in ws[1]]
    first = head.index("step_1") + 1
    com = head.index("comment") + 1
    for r in range(2, ws.max_row + 1):
        key = (ws.cell(r, 1).value, ws.cell(r, 2).value)
        if key in plans:
            for j in range(B.MAX_STEPS):                                                    # a plan replaces whatever was in the row
                ws.cell(r, first + j).value = plans[key][j] if j < len(plans[key]) else None
        if comments and key in comments:
            ws.cell(r, com).value = comments[key]
    if review:
        ps = wb["Parts"]
        h = [c.value for c in ps[1]]
        for r in range(2, ps.max_row + 1):
            pid = ps.cell(r, 1).value
            if pid in review:
                for k, v in zip(("review_status", "reviewed_by", "review_comment"), review[pid]):
                    ps.cell(r, h.index(k) + 1).value = v
    wb.save(path)
    return path


@pytest.fixture()
def books(tmp_path, spec):
    """(primary workbook path, second workbook path, second-annotator part list) for the six-part spec, nothing filled yet."""
    prim = B.write_annotation_workbook(spec, str(tmp_path / "primary.xlsx"), "Prof. Deb", "primary")
    sub = B.second_subset(spec, 3)
    sec = B.write_annotation_workbook(sub, str(tmp_path / "second.xlsx"), "Second Person", "second")
    return prim, sec, sub


# ---------------------------------------------------------------- 1. the spec
def test_parse_a_clean_row(spec):
    p = spec[0]
    assert p["part_id"] == "P1" and p["split"] == "bench" and p["in_distribution"] is None
    assert p["material"] == "Steel" and p["features"] == ["Hole", "Chamfer"] and p["tolerance"] == "0.05mm" and p["batch_size"] == 100
    assert p["dims"] == {"diameter": 50, "length": 120, "depth": 20, "width": 50, "hole_diameter": 12}
    assert p["required_ra_um"] == 1.6 and p["machine_preference"] == "auto"
    assert p["name"] == "Flanged shaft" and p["source"] == "Textbook ex 3.1" and p["drawing"] == "fig3_1.png"
    assert p["n_features"] == 2 and p["batch_category"] and p["machine_profile"]       # derived fields from capp.dataset


def test_materials_features_and_tolerances_are_normalised(spec):
    by = {p["part_id"]: p for p in spec}
    assert by["P3"]["material"] == "Aluminum" and by["P4"]["material"] == "Cast Iron"
    assert by["P3"]["features"] == ["Slot", "Pocket", "Hole"]                           # ',' and ';' both separate
    assert by["P3"]["tolerance"] == "0.1mm" and by["P5"]["tolerance"] == "0.5mm"
    assert by["P2"]["required_ra_um"] is None and "hole_diameter" not in by["P2"]["dims"]


@pytest.mark.parametrize("text", ["0.05", "0.05mm", "0.05 mm", "0.050", " 0.05MM"])
def test_tolerance_spellings(text):
    p, errs = B.parse_part_row(dict(rows()[0], tolerance=text), 1)
    assert errs == [] and p["tolerance"] == "0.05mm"


def test_a_duplicate_feature_counts_once():
    p, errs = B.parse_part_row(dict(rows()[0], features="Hole;hole;Chamfer; Hole"), 1)
    assert errs == [] and p["features"] == ["Hole", "Chamfer"]


def test_ids_are_generated_in_row_order():
    p, _ = B.parse_part_row(dict(rows()[0], part_id=""), 7)
    assert p["part_id"] == "BENCH_007"


def test_every_problem_of_a_row_is_reported_at_once():
    bad = {"name": "", "source": "", "material": "Unobtainium", "features": "Hole;Slott", "tolerance": "0.03", "batch_size": "12.5",
           "diameter": 9999, "length": "abc", "depth": 10, "width": 10, "hole_diameter": 500, "required_ra_um": 99}
    p, errs = B.parse_part_row(bad, 1)
    text = " | ".join(errs)
    assert p is None
    assert "'name' is empty" in text and "'source' is empty" in text
    assert "Unobtainium" in text and "Slott" in text and "did you mean Slot" in text
    assert "0.03" in text and "12.5" in text and "diameter" in text and "length 'abc'" in text
    assert "hole_diameter" in text and "required_ra_um" in text


def test_empty_required_cells_are_named():
    p, errs = B.parse_part_row({"part_id": "X"}, 1)
    assert p is None and {f"'{c}' is empty" for c in B.REQUIRED_COLUMNS} <= set(errs)


def test_batch_and_dimension_limits():
    assert B.parse_part_row(dict(rows()[0], batch_size=0), 1)[0] is None
    assert B.parse_part_row(dict(rows()[0], batch_size=10001), 1)[0] is None
    assert B.parse_part_row(dict(rows()[0], diameter=0.5), 1)[0] is None
    assert B.parse_part_row(dict(rows()[0], hole_diameter=0.4), 1)[0] is None
    assert B.parse_part_row(dict(rows()[0], diameter="50,5"), 1)[0]["dims"]["diameter"] == 50.5     # decimal comma


def test_a_parsed_row_is_a_valid_bench_record(spec):
    for p in spec:
        assert D.validate_part(dict(p, reference_routes=[["Facing", "Inspection"]])) == []


def test_template_xlsx_has_the_sheets_and_header(tmp_path):
    path = B.write_spec_template(str(tmp_path / "spec.xlsx"))
    wb = openpyxl.load_workbook(path)
    assert wb.sheetnames == ["Parts", "Lists", "README"]
    assert [c.value for c in wb["Parts"][1]] == list(B.SPEC_COLUMNS)
    lists = wb["Lists"]
    assert {r[0] for r in lists.iter_rows(min_row=2, values_only=True) if r[0]} == set(B.MATERIALS)
    assert {r[2] for r in lists.iter_rows(min_row=2, values_only=True) if r[2]} == set(fv.get_all_features())


def test_template_csv_is_just_the_header(tmp_path):
    path = B.write_spec_template(str(tmp_path / "spec.csv"))
    assert open(path, encoding="utf-8").read().strip().split(",") == list(B.SPEC_COLUMNS)
    parts, errs = B.read_spec(path)
    assert parts == [] and errs == ["the spec has no part rows"]


@pytest.mark.parametrize("ext", [".xlsx", ".csv"])
def test_spec_round_trip(tmp_path, spec, ext):
    path = B.write_spec_rows(rows(), str(tmp_path / ("spec" + ext)))
    parts, errs = B.read_spec(path)
    assert errs == [] and [p["part_id"] for p in parts] == [p["part_id"] for p in spec]
    assert [p["features"] for p in parts] == [p["features"] for p in spec]
    assert [p["dims"] for p in parts] == [p["dims"] for p in spec]


def test_row_numbers_in_errors_count_the_header(tmp_path):
    rs = rows()
    rs[1]["material"] = "Wood"
    path = B.write_spec_rows(rs, str(tmp_path / "spec.csv"))
    parts, errs = B.read_spec(path)
    assert len(parts) == 5 and len(errs) == 1 and errs[0].startswith("row 3:") and "Wood" in errs[0]       # header = row 1, P2 = row 3


def test_duplicate_part_ids_are_an_error(tmp_path):
    rs = rows()
    rs[1]["part_id"] = "P1"
    _, errs = B.read_spec(B.write_spec_rows(rs, str(tmp_path / "spec.csv")))
    assert any("P1 appears 2 times" in e for e in errs)


def test_json_spec_is_read_as_records(tmp_path, spec):
    path = tmp_path / "bench_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    parts, errs = B.read_spec(str(path))
    assert errs == [] and parts == json.loads(json.dumps(spec))
    broken = json.loads(json.dumps(spec))
    broken[0]["dims"]["diameter"] = -3
    path.write_text(json.dumps(broken), encoding="utf-8")
    assert any(e.startswith("P1:") for e in B.read_spec(str(path))[1])


# ---------------------------------------------------------------- placeholder spec from a synthetic set
def test_stratified_pick_is_deterministic_and_keeps_order():
    pool = D.load_dataset("test_v2")
    a, b = B.stratified_pick(pool, 24, seed=7), B.stratified_pick(pool, 24, seed=7)
    assert [p["part_id"] for p in a] == [p["part_id"] for p in b] and len(a) == 24
    ids = [p["part_id"] for p in pool]
    assert [ids.index(p["part_id"]) for p in a] == sorted(ids.index(p["part_id"]) for p in a)
    assert [p["part_id"] for p in B.stratified_pick(pool, 24, seed=8)] != [p["part_id"] for p in a]
    assert len(B.stratified_pick(pool, 10 ** 6)) == len(pool)


def test_stratified_pick_spreads_over_the_classes():
    pool = D.load_dataset("test_v2")
    picked = B.stratified_pick(pool, 24)
    classes = {(p["machine_profile"], S.complexity(len(p["features"]))) for p in pool}
    assert len({(p["machine_profile"], S.complexity(len(p["features"]))) for p in picked}) == min(len(classes), 24)


def test_placeholder_spec_is_labelled_and_readable(tmp_path):
    picked = B.stratified_pick(D.load_dataset("test_v2"), 12)
    rs = B.spec_from_dataset(picked, source="synthetic test_v2 (placeholder)")
    assert [r["part_id"] for r in rs][:2] == ["BENCH_001", "BENCH_002"]
    assert all("placeholder" in r["source"] and picked[i]["part_id"] in r["notes"] for i, r in enumerate(rs))
    parts, errs = B.read_spec(B.write_spec_rows(rs, str(tmp_path / "ph.xlsx")))
    assert errs == [] and len(parts) == 12 and all(p["split"] == "bench" for p in parts)


# ---------------------------------------------------------------- 2. annotation workbooks
def test_primary_workbook_structure(books, spec):
    prim, _, _ = books
    wb = openpyxl.load_workbook(prim)
    assert wb.sheetnames == ["README", "Parts", "Routes", "Operations", "Meta"]
    parts_head = [c.value for c in wb["Parts"][1]]
    assert parts_head[-3:] == ["review_status", "reviewed_by", "review_comment"]
    assert wb["Parts"].max_row == len(spec) + 1
    meta = dict(wb["Meta"].iter_rows(values_only=True))
    assert meta["annotator"] == "Prof. Deb" and meta["role"] == "primary" and meta["n_parts"] == len(spec)


def test_second_workbook_has_no_review_columns_and_only_its_subset(books):
    _, sec, sub = books
    wb = openpyxl.load_workbook(sec)
    assert "review_status" not in [c.value for c in wb["Parts"][1]]
    assert [r[0] for r in wb["Parts"].iter_rows(min_row=2, values_only=True)] == [p["part_id"] for p in sub]
    assert "Supervisor review" not in " ".join(str(r[0]) for r in wb["README"].iter_rows(values_only=True) if r[0])


def test_the_routes_sheet_has_rows_per_alternative_and_no_prefilled_plan(books, spec):
    prim, _, _ = books
    ws = openpyxl.load_workbook(prim)["Routes"]
    head = [c.value for c in ws[1]]
    assert head[:3] == ["part_id", "alt", "part"] and head[3:3 + B.MAX_STEPS] == [f"step_{i}" for i in range(1, B.MAX_STEPS + 1)]
    body = list(ws.iter_rows(min_row=2, values_only=True))
    assert len(body) == len(spec) * 2 and [r[1] for r in body[:4]] == ["A", "B", "A", "B"]
    assert all(c is None for r in body for c in r[3:3 + B.MAX_STEPS])                  # nothing pre-filled: no anchoring, no RB prefill
    assert all((r[2] is not None) == (r[1] == "A") for r in body)                      # the part summary sits on row A only
    assert "Flanged shaft" in body[0][2] and "hole Ø12" in body[0][2]


def test_operations_sheet_and_dropdown(books):
    prim, _, _ = books
    wb = openpyxl.load_workbook(prim)
    ops = [r[0] for r in wb["Operations"].iter_rows(min_row=2, values_only=True)]
    assert ops == sorted(V.CANONICAL_OPS) and len(ops) == 41
    machines = {r[0]: r[1] for r in wb["Operations"].iter_rows(min_row=2, values_only=True)}
    assert machines["Facing"] in ("lathe", "either") and machines["Slot Milling"] == "milling"
    dvs = wb["Routes"].data_validations.dataValidation
    assert len(dvs) == 1 and dvs[0].type == "list" and "Operations!$A$2:$A$42" in dvs[0].formula1
    assert "D2" in str(dvs[0].sqref) and "AA" in str(dvs[0].sqref)


def test_workbook_options_are_checked(tmp_path, spec):
    with pytest.raises(ValueError):
        B.write_annotation_workbook(spec, str(tmp_path / "x.xlsx"), "A", role="third")
    with pytest.raises(ValueError):
        B.write_annotation_workbook(spec, str(tmp_path / "x.xlsx"), "A", alternatives=4)
    wb = openpyxl.load_workbook(B.write_annotation_workbook(spec, str(tmp_path / "one.xlsx"), "A", alternatives=1))
    assert wb["Routes"].max_row == len(spec) + 1


def test_second_subset_is_a_stratified_subset(spec):
    sub = B.second_subset(spec, 3)
    assert len(sub) == 3 and {p["part_id"] for p in sub} <= {p["part_id"] for p in spec}
    assert [p["part_id"] for p in sub] == [p["part_id"] for p in B.second_subset(spec, 3)]


def test_reading_an_unfilled_workbook(books):
    prim, _, _ = books
    ann = B.read_annotations(prim)
    assert ann["annotator"] == "Prof. Deb" and ann["role"] == "primary" and len(ann["parts"]) == 6
    assert all(v["routes"]["A"] == [] and v["unknown"] == [] for v in ann["parts"].values())


def test_reading_a_filled_workbook(books):
    prim, _, _ = books
    fill(prim, {("P1", "A"): ["  facing ", "CENTER DRILLING", "drilling", None, "Chamfering", "inspection"],
                ("P1", "B"): ["Facing", "Drilling", "Chamfering", "Inspection"]},
         comments={("P1", "A"): "usual order", ("P1", "B"): "no centre drill"},
         review={"P1": ("accepted", "Prof. Deb", "fine")})
    ann = B.read_annotations(prim)["parts"]["P1"]
    assert ann["routes"]["A"] == ["Facing", "Center Drilling", "Drilling", "Chamfering", "Inspection"]      # case, spaces, blank cell
    assert ann["routes"]["B"] == ["Facing", "Drilling", "Chamfering", "Inspection"]
    assert ann["comment"] == "A: usual order | B: no centre drill"
    assert (ann["review_status"], ann["reviewed_by"], ann["review_comment"]) == ("accepted", "Prof. Deb", "fine")


def test_an_unknown_operation_name_is_kept_with_suggestions(books):
    prim, _, _ = books
    fill(prim, {("P1", "A"): ["Facing", "Drilng", "Inspection"]})
    ann = B.read_annotations(prim)["parts"]["P1"]
    assert ann["routes"]["A"] == ["Facing", "Inspection"]
    assert [(a, t) for a, t, _ in ann["unknown"]] == [("A", "Drilng")] and "Drilling" in ann["unknown"][0][2]


# ---------------------------------------------------------------- 3. import and checks
def full_plans(spec, only=None):
    return {(p["part_id"], "A"): route_for(p) for p in spec if only is None or p["part_id"] in only}


def test_build_benchmark_happy_path(books, spec):
    prim, sec, sub = books
    plans = full_plans(spec)
    fill(prim, plans, review={"P1": ("accepted", "Prof. Deb", ""), "P2": ("changed", "Prof. Deb", "swapped two steps")})
    fill(sec, {(p["part_id"], "A"): route_for(p) for p in sub})
    bench, rep = B.build_benchmark(spec, B.read_annotations(prim), B.read_annotations(sec))
    assert rep["errors"] == [] and rep["missing"] == [] and rep["conflicts"] == [] and len(bench) == 6
    assert rep["annotator"] == "Prof. Deb" and rep["second_annotator"] == "Second Person"
    assert rep["review"] == {"accepted": 1, "changed": 1, "not reviewed": 4}
    for p in bench:
        assert p["split"] == "bench" and D.validate_part(p) == []
        assert p["reference_routes"] == [plans[(p["part_id"], "A")]]
        assert p["reference_meta"]["annotator"] == "Prof. Deb"
    seconds = {p["part_id"] for p in bench if p.get("second_annotator_routes")}
    assert seconds == {p["part_id"] for p in sub}
    assert all("second_annotator_routes" not in p["reference_meta"] for p in bench)         # the second plan stays out of the reference
    assert next(p for p in bench if p["part_id"] == "P2")["reference_meta"]["review_comment"] == "swapped two steps"


def test_alternatives_are_kept_and_duplicates_dropped(books, spec):
    prim, _, _ = books
    plans = full_plans(spec)
    r1 = plans[("P1", "A")]
    alt = [r1[0], r1[2], r1[1]] + r1[3:]                                                    # same ops, another order
    plans[("P1", "B")] = alt
    plans[("P2", "B")] = list(plans[("P2", "A")])                                           # a copy of A
    fill(prim, plans)
    bench, _ = B.build_benchmark(spec, B.read_annotations(prim))
    by = {p["part_id"]: p for p in bench}
    assert by["P1"]["reference_routes"] == [r1, alt] and len(by["P2"]["reference_routes"]) == 1


def test_a_part_without_a_plan_blocks_unless_allowed(books, spec):
    prim, _, _ = books
    fill(prim, full_plans(spec, only={"P1", "P2", "P3", "P4", "P5"}))
    bench, rep = B.build_benchmark(spec, B.read_annotations(prim))
    assert rep["missing"] == ["P6"] and any("P6" in e and "route A" in e for e in rep["errors"]) and len(bench) == 5
    bench, rep = B.build_benchmark(spec, B.read_annotations(prim), allow_missing=True)
    assert rep["missing"] == ["P6"] and rep["errors"] == [] and [p["part_id"] for p in bench] == ["P1", "P2", "P3", "P4", "P5"]


def test_a_plan_only_in_row_b_does_not_count(books, spec):
    prim, _, _ = books
    plans = full_plans(spec)
    plans[("P3", "B")] = plans.pop(("P3", "A"))
    fill(prim, plans)
    _, rep = B.build_benchmark(spec, B.read_annotations(prim))
    assert rep["missing"] == ["P3"]


def test_unknown_operations_are_blocking_errors_with_the_suggestion(books, spec):
    prim, sec, sub = books
    plans = full_plans(spec)
    plans[("P4", "A")] = ["Facing", "Borring", "Grooving/Necking", "Inspection"]
    fill(prim, plans)
    other = sub[0]["part_id"]
    fill(sec, {(other, "A"): ["Facing", "Warp Drive"]})
    _, rep = B.build_benchmark(spec, B.read_annotations(prim), B.read_annotations(sec))
    assert any("P4 (primary, route A): unknown operation 'Borring'" in e and "Boring" in e for e in rep["errors"])
    assert any(f"{other} (second, route A): unknown operation 'Warp Drive'" in e for e in rep["errors"])


def test_ids_in_the_workbooks_but_not_in_the_spec_are_listed(books, spec):
    prim, _, _ = books
    fill(prim, full_plans(spec))
    _, rep = B.build_benchmark(spec[:4], B.read_annotations(prim))
    assert rep["unknown_parts"] == ["P5", "P6"] and rep["errors"] == []


def test_a_plan_the_rule_base_rejects_is_kept_as_a_conflict(books, spec):
    prim, _, _ = books
    plans = full_plans(spec)
    plans[("P1", "A")] = ["Facing", "Chamfering", "Inspection"]                             # the Hole is not made
    fill(prim, plans)
    bench, rep = B.build_benchmark(spec, B.read_annotations(prim))
    p1 = next(p for p in bench if p["part_id"] == "P1")
    assert p1["reference_routes"] == [plans[("P1", "A")]]                                     # kept: the expert is the ground truth
    assert rep["errors"] == [] and [(c["part_id"], c["route"]) for c in rep["conflicts"]] == [("P1", "A")]
    assert "missing" in rep["conflicts"][0]["kinds"] and "A" in p1["reference_meta"]["rule_conflicts"]


def test_inspection_not_last_is_a_conflict(books, spec):
    prim, _, _ = books
    plans = full_plans(spec)
    r = plans[("P6", "A")]
    plans[("P6", "A")] = ["Inspection"] + [s for s in r if s != "Inspection"]
    fill(prim, plans)
    _, rep = B.build_benchmark(spec, B.read_annotations(prim))
    assert [c["part_id"] for c in rep["conflicts"]] == ["P6"] and "order" in rep["conflicts"][0]["kinds"]


def test_check_route_reports_physical_findings_on_the_real_dimensions(spec):
    p1 = dict(spec[0])
    ok = B.check_route(p1, route_for(p1))
    assert ok["valid"] and ok["infeasible"] == []
    deep = dict(p1, dims=dict(p1["dims"], hole_diameter=3, depth=200))
    assert "HOLE-DEEP" in B.check_route(deep, route_for(deep))["risks"]
    no_hole_made = B.check_route(p1, ["Facing", "Reaming", "Inspection"])
    assert not no_hole_made["valid"] and "HOLE-EXISTS" in no_hole_made["infeasible"]


def test_physical_findings_land_in_the_report_and_the_record(books, spec):
    prim, _, _ = books
    deep = [dict(p, dims=dict(p["dims"], hole_diameter=3, depth=200)) if p["part_id"] == "P1" else p for p in spec]
    fill(prim, full_plans(deep))
    bench, rep = B.build_benchmark(deep, B.read_annotations(prim))
    assert any(c["part_id"] == "P1" and "HOLE-DEEP" in c["risks"] for c in rep["physical"])
    assert "A" in next(p for p in bench if p["part_id"] == "P1")["reference_meta"]["physical"]


# ---------------------------------------------------------------- 4. agreement (M10)
def test_bootstrap_ci_edge_cases():
    assert B.bootstrap_ci([]) == (None, None) and B.bootstrap_ci([0.5]) == (None, None) and B.bootstrap_ci([None, 0.5]) == (None, None)
    assert B.bootstrap_ci([0.7, 0.7, 0.7]) == pytest.approx((0.7, 0.7))


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    vals = [0.2, 0.9, 0.5, 0.7, 1.0, 0.4, 0.8, 0.6]
    lo, hi = B.bootstrap_ci(vals)
    assert (lo, hi) == B.bootstrap_ci(vals) and lo <= sum(vals) / len(vals) <= hi and 0.2 <= lo and hi <= 1.0
    lo99, hi99 = B.bootstrap_ci(vals, level=0.99)
    assert lo99 <= lo and hi <= hi99


def agree_part(pid, ref, second):
    return {"part_id": pid, "reference_routes": ref, "second_annotator_routes": second}


def test_agreement_on_a_handcrafted_pair():
    a, b = ["Facing", "Drilling", "Inspection"], ["Facing", "Inspection", "Drilling"]
    parts = [agree_part("1", [a], [a]), agree_part("2", [a], [b]), agree_part("3", [a], None)]
    ag = B.agreement(parts)
    assert ag["n"] == 2 and [r["part_id"] for r in ag["rows"]] == ["1", "2"]
    assert ag["exact_rate"] == 0.5
    assert ag["lcs_ratio_mean"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert ag["kendall_tau_mean"] == pytest.approx((0.0 + 1 / 3) / 2)
    assert ag["rows"][1]["n_ops_reference"] == 3 and ag["rows"][1]["n_ops_second"] == 3
    assert ag["exact_rate_ci95"][0] is not None


def test_agreement_takes_the_best_matching_reference_alternative():
    a, b = ["Facing", "Drilling", "Inspection"], ["Facing", "Inspection", "Drilling"]
    ag = B.agreement([agree_part("1", [b, a], [a])])
    assert ag["exact_rate"] == 1.0 and ag["kendall_tau_mean"] == 0.0


def test_only_the_first_plan_of_the_second_annotator_is_compared():
    a, b = ["Facing", "Drilling", "Inspection"], ["Facing", "Inspection", "Drilling"]
    assert B.agreement([agree_part("1", [a], [a, b])])["exact_rate"] == 1.0
    assert B.agreement([agree_part("1", [a], [b, a])])["exact_rate"] == 0.0


def test_agreement_without_second_plans():
    ag = B.agreement([agree_part("1", [["Facing", "Inspection"]], None), {"part_id": "2", "reference_routes": [["Facing"]]}])
    assert ag["n"] == 0 and ag["rows"] == [] and ag["exact_rate"] is None and ag["exact_rate_ci95"] == (None, None)


def test_markers_do_not_count_in_the_agreement():
    a = ["Facing", V.MARKER, "Face Milling", "Inspection"]
    ag = B.agreement([agree_part("1", [["Facing", "Face Milling", "Inspection"]], [a])])
    assert ag["exact_rate"] == 1.0 and ag["rows"][0]["n_ops_second"] == 3


# ---------------------------------------------------------------- 5. datasheet
@pytest.fixture()
def bench_and_report(books, spec):
    prim, sec, sub = books
    plans = full_plans(spec)
    plans[("P1", "A")] = ["Facing", "Chamfering", "Inspection"]                           # one conflict
    fill(prim, plans, review={"P1": ("rejected", "Prof. Deb", "no hole"), "P2": ("accepted", "Prof. Deb", "")})
    second = {(p["part_id"], "A"): route_for(p) for p in sub}
    fill(sec, second)
    bench, rep = B.build_benchmark(spec, B.read_annotations(prim), B.read_annotations(sec))
    return bench, rep, B.agreement(bench)


def test_datasheet_sections_and_numbers(bench_and_report):
    bench, rep, ag = bench_and_report
    text = B.datasheet(bench, rep, ag, generated="2026-10-27")
    for head in ("# Expert benchmark: datasheet", "## Purpose", "## Composition", "## Reference plans", "## Human-human agreement (M10)",
                 "## Intended use and limits", "## To fill in by hand"):
        assert head in text
    assert "2026-10-27" in text and "6 parts" in text and "Prof. Deb" in text and "[TO CONFIRM]" in text
    assert "rejected 1" in text and "accepted 1" in text and "not reviewed 4" in text
    assert "rejects: 1" in text and "P1/A" in text                                           # the conflict is listed
    assert f"{ag['n']} parts, plan written independently" in text and "Second Person" in text
    assert "hole diameter given for 3 parts" in text and "Required Ra given for 3 parts" in text
    assert "Steel" in text and "Cast Iron" in text and "features per part" in text
    assert "WARNING" not in text


def test_datasheet_warns_when_the_parts_are_placeholders(bench_and_report):
    bench, rep, ag = bench_and_report
    ph = [dict(p, source="synthetic test_v2 (placeholder)") for p in bench]
    assert "PLACEHOLDER" in B.datasheet(ph, rep, ag)
    mixed = [dict(bench[0], source="synthetic test_v2 (placeholder)")] + bench[1:]
    assert "PLACEHOLDER" in B.datasheet(mixed, rep, ag)


def test_datasheet_without_second_annotator():
    base = {"part_id": "B1", "split": "bench", "name": "x", "source": "s", "material": "Steel", "features": ["Hole"], "tolerance": "0.05mm",
            "batch_size": 10, "dims": {"diameter": 10, "length": 10, "depth": 10, "width": 10}, "required_ra_um": None,
            "machine_preference": "auto", "reference_routes": [["Facing", "Drilling", "Inspection"]], "archetype": "x", "in_distribution": None}
    base.update(D.derived_fields(base))
    text = B.datasheet([base], None, B.agreement([base]))
    assert "M10 is not available" in text and "hole diameter given for 0 parts" in text


def test_datasheet_of_an_empty_benchmark_does_not_crash():
    assert "0 parts" in B.datasheet([])


# ---------------------------------------------------------------- the bench split in capp/dataset.py
def bench_part(**kw):
    p = {"part_id": "B1", "split": "bench", "archetype": "x", "in_distribution": None, "material": "Steel", "features": ["Hole"],
         "tolerance": "0.05mm", "batch_size": 10, "dims": {"diameter": 10, "length": 10, "depth": 10, "width": 10}, "required_ra_um": None,
         "machine_preference": "auto", "reference_routes": [["Facing", "Center Drilling", "Drilling", "Inspection"]]}
    p.update(D.derived_fields(p))
    p.update(kw)
    return p


def test_bench_is_a_split_and_dev_test_are_unchanged():
    assert D.SPLITS == ("dev", "test", "bench")
    assert D.validate_part(bench_part()) == []
    dev = D.load_dataset("dev_v2")
    assert D.validate_part(dev[0]) == []
    assert D.validate_part(dict(dev[0], dims=dict(dev[0]["dims"], hole_diameter=8))) == []            # allowed in the schema for any split


def test_bench_ra_may_be_missing_or_free_but_dev_ra_stays_derived():
    assert D.validate_part(bench_part(required_ra_um=None)) == []
    assert D.validate_part(bench_part(required_ra_um=6.3)) == []                                          # not tied to the tolerance
    assert any("required_ra_um" in e for e in D.validate_part(bench_part(required_ra_um=0.01)))
    assert any("required_ra_um" in e for e in D.validate_part(bench_part(required_ra_um=500)))
    dev = D.load_dataset("dev_v2")[0]
    assert any("required_ra_um" in e for e in D.validate_part(dict(dev, required_ra_um=dev["required_ra_um"] + 1)))
    assert any("required_ra_um" in e for e in D.validate_part(dict(dev, required_ra_um=None)))


def test_bench_dims_and_reference_routes_are_validated():
    ok = {"diameter": 10, "length": 10, "depth": 10, "width": 10}
    assert D.validate_part(bench_part(dims=dict(ok, hole_diameter=5))) == []
    assert any("hole_diameter" in e for e in D.validate_part(bench_part(dims=dict(ok, hole_diameter=0.1))))
    assert any("hole_diameter" in e for e in D.validate_part(bench_part(dims=dict(ok, hole_diameter=True))))
    assert any("dims must have" in e for e in D.validate_part(bench_part(dims=dict(ok, flange=3))))
    assert any("dims must have" in e for e in D.validate_part(bench_part(dims={"diameter": 10})))
    assert any("at least one reference route" in e for e in D.validate_part(bench_part(reference_routes=[])))
    assert any("at least one reference route" in e for e in D.validate_part({k: v for k, v in bench_part().items() if k != "reference_routes"}))
    assert any("reference_routes" in e for e in D.validate_part(bench_part(reference_routes=[[]])))
    assert any("second_annotator_routes" in e for e in D.validate_part(bench_part(second_annotator_routes=["Facing"])))
    assert D.validate_part(bench_part(second_annotator_routes=[["Facing", "Inspection"]])) == []


def test_save_and_load_a_bench_dataset(tmp_path):
    parts = [bench_part(part_id="B1", dims={"diameter": 10, "length": 10, "depth": 10, "width": 10, "hole_diameter": 5}, source="s1"),
             bench_part(part_id="B2", reference_routes=[["Facing", "Inspection"], ["Facing", "Drilling", "Inspection"]], source="s2")]
    path = D.save_dataset(parts, str(tmp_path / "bench_x.json"))
    back = D.load_dataset(path)
    assert back == json.loads(json.dumps(parts))
    with open(path.replace(".json", ".csv"), encoding="utf-8", newline="") as f:
        rs = list(csv.DictReader(f))
    assert rs[0]["hole_diameter"] == "5" and rs[1]["hole_diameter"] == "" and [r["source"] for r in rs] == ["s1", "s2"]
    assert [r["n_reference_routes"] for r in rs] == ["1", "2"]


def test_the_csv_of_a_dev_dataset_keeps_its_columns(tmp_path):
    dev = D.load_dataset("dev_v2")[:3]
    path = D.save_dataset(dev, str(tmp_path / "dev_x.json"))
    head = open(path.replace(".json", ".csv"), encoding="utf-8").readline().strip().split(",")
    assert head[-1] == "machine_profile" and "source" not in head and "hole_diameter" not in head


@pytest.mark.parametrize("name,sealed", [("test_v2", True), ("bench_v1", True), ("data/bench_v1.json", True), ("/tmp/x/test_v3.json", True),
                                         ("dev_v2", False), ("dev_bench", False), ("mytest", False)])
def test_is_sealed(name, sealed):
    assert D.is_sealed(name) is sealed


def test_a_sealed_set_is_refused_as_the_retrieval_pool():
    dev = D.load_dataset("dev_v2")[:1]
    for pool in ("test_v2", "bench_v1"):
        with pytest.raises(H.HarnessError, match="sealed"):
            H.run_experiment("b0+fs2", dev, pool=pool, progress=None, write=False)


def _run_eval(*args, tmp_path):
    env = dict(os.environ, GROQ_API_KEY="", PYTHONIOENCODING="utf-8", CAPP_CACHE_DIR=str(tmp_path / "c"))
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_eval.py"), *args, "--out", str(tmp_path)],
                          capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)


def test_run_eval_refuses_the_benchmark_as_a_few_shot_pool(tmp_path):
    res = _run_eval("--system", "b0+fs2", "--pool", "bench_v1", tmp_path=tmp_path)
    assert res.returncode == 2 and "sealed" in res.stderr and "benchmark" in res.stderr


# ---------------------------------------------------------------- the harness and the comparison table on bench parts
@pytest.fixture()
def bench_parts(books, spec):
    prim, sec, sub = books
    fill(prim, full_plans(spec))
    fill(sec, {(p["part_id"], "A"): route_for(p) for p in sub})
    bench, rep = B.build_benchmark(spec, B.read_annotations(prim), B.read_annotations(sec))
    assert rep["errors"] == []
    return bench


def test_m4_is_measured_against_the_expert_routes(bench_parts, tmp_path):
    out = H.run_experiment("rb", bench_parts, tag="bench", out_dir=str(tmp_path), progress=None)
    s = out["summary"]
    assert "expert" in s["M4"] and "proxy_rb" not in s["M4"]
    assert s["M4"]["expert"]["final"]["lcs_ratio_mean"] is not None


def test_the_comparison_table_names_the_m4_reference(bench_parts, tmp_path):
    out = H.run_experiment("rb", bench_parts, tag="bench", out_dir=str(tmp_path), progress=None)
    text = compare.table([("rb", out["summary"])])
    assert "M4 reference" in text and "expert" in text and "M4 final: LCS ratio" in text and "M4 final: exact-match rate" in text


def test_the_comparison_table_on_a_synthetic_set_uses_the_proxy(tmp_path):
    out = H.run_experiment("rb", D.load_dataset("dev_v2")[:4], tag="d", out_dir=str(tmp_path), progress=None)
    text = compare.table([("rb", out["summary"])])
    assert "proxy_rb" in text


def test_a_summary_without_m4_prints_n_a():
    assert compare._m4_source({}) is None and compare._m4_val({"M4": {}}, "final", "lcs_ratio_mean") == "n/a"


# ---------------------------------------------------------------- the command line (scripts/benchmark.py)
def run_cli(*argv):
    return cli.main(list(argv))


def test_cli_template_and_select(tmp_path, capsys):
    assert run_cli("template", "--out", str(tmp_path / "sub" / "spec.xlsx")) == 0
    assert os.path.exists(tmp_path / "sub" / "spec.xlsx")
    assert run_cli("select", "--from", "test_v2", "--n", "10", "--out", str(tmp_path / "ph.xlsx")) == 0
    out = capsys.readouterr().out
    assert "10 placeholder parts" in out and "synthetic" in out
    parts, errs = B.read_spec(str(tmp_path / "ph.xlsx"))
    assert errs == [] and len(parts) == 10


def test_cli_sheets_writes_both_workbooks_and_the_json_spec(tmp_path, capsys):
    run_cli("select", "--n", "20", "--out", str(tmp_path / "ph.csv"))
    d = tmp_path / "w"
    assert run_cli("sheets", "--spec", str(tmp_path / "ph.csv"), "--out-dir", str(d), "--annotator", "Prof. Deb",
                   "--second-name", "Second Person", "--second-n", "5") == 0
    out = capsys.readouterr().out
    assert sorted(os.listdir(d)) == ["annotate_primary.xlsx", "annotate_second.xlsx", "bench_spec.json"]
    assert "20 parts" in out and "nothing is pre-filled" in out
    assert len(openpyxl.load_workbook(d / "annotate_second.xlsx")["Parts"]["A"]) == 6
    assert len(json.loads((d / "bench_spec.json").read_text(encoding="utf-8"))) == 20
    assert B.read_annotations(str(d / "annotate_second.xlsx"))["annotator"] == "Second Person"


def test_cli_sheets_without_a_second_workbook_and_with_a_bad_spec(tmp_path, capsys):
    run_cli("select", "--n", "6", "--out", str(tmp_path / "ph.csv"))
    assert run_cli("sheets", "--spec", str(tmp_path / "ph.csv"), "--out-dir", str(tmp_path / "w"), "--second-n", "0") == 0
    assert not os.path.exists(tmp_path / "w" / "annotate_second.xlsx")
    rs = rows()
    rs[0]["material"] = "Wood"
    bad = B.write_spec_rows(rs, str(tmp_path / "bad.csv"))
    capsys.readouterr()
    assert run_cli("sheets", "--spec", bad, "--out-dir", str(tmp_path / "w2")) == 1
    assert "row 2" in capsys.readouterr().err and not os.path.exists(tmp_path / "w2")


@pytest.fixture()
def chain(tmp_path):
    """The whole chain on a placeholder spec: select, sheets, then two simulated annotators (the second one changes some plans)."""
    run_cli("select", "--n", "18", "--out", str(tmp_path / "ph.xlsx"))
    run_cli("sheets", "--spec", str(tmp_path / "ph.xlsx"), "--out-dir", str(tmp_path), "--annotator", "Prof. Deb",
            "--second-name", "Second Person", "--second-n", "6")
    spec_path = str(tmp_path / "bench_spec.json")
    parts = json.loads((tmp_path / "bench_spec.json").read_text(encoding="utf-8"))
    fill(str(tmp_path / "annotate_primary.xlsx"), {(p["part_id"], "A"): route_for(p) for p in parts})
    sec = B.read_annotations(str(tmp_path / "annotate_second.xlsx"))["parts"]
    plans = {}
    for i, pid in enumerate(sorted(sec)):
        p = next(q for q in parts if q["part_id"] == pid)
        r = route_for(p)
        if i % 2:
            r = r[:1] + r[2:3] + r[1:2] + r[3:] if len(r) > 3 else r
        plans[(pid, "A")] = r
    fill(str(tmp_path / "annotate_second.xlsx"), plans)
    return tmp_path, spec_path


def test_cli_import_writes_the_dataset_and_the_datasheet(chain, capsys):
    d, spec_path = chain
    capsys.readouterr()
    out = str(d / "bench_v1.json")
    assert run_cli("import", "--spec", spec_path, "--primary", str(d / "annotate_primary.xlsx"),
                   "--second", str(d / "annotate_second.xlsx"), "--out", out) == 0
    text = capsys.readouterr().out
    assert "18 parts in the spec, 18 with a reference plan" in text and "M10 (second plan vs reference, 6 parts)" in text
    assert "sealed" in text
    assert sorted(f for f in os.listdir(d) if f.startswith("bench_v1")) == ["bench_v1.csv", "bench_v1.json", "bench_v1_datasheet.md"]
    bench = D.load_dataset(out)
    assert len(bench) == 18 and all(p["split"] == "bench" and p["reference_routes"] for p in bench)
    assert sum(1 for p in bench if p.get("second_annotator_routes")) == 6
    sheet = (d / "bench_v1_datasheet.md").read_text(encoding="utf-8")
    assert "# Expert benchmark: datasheet" in sheet and "PLACEHOLDER" in sheet and "18 parts" in sheet


def test_cli_import_dry_run_writes_nothing(chain, capsys):
    d, spec_path = chain
    before = set(os.listdir(d))
    assert run_cli("import", "--spec", spec_path, "--primary", str(d / "annotate_primary.xlsx"), "--out", str(d / "bench_v1.json"),
                   "--dry-run") == 0
    out = capsys.readouterr().out
    assert "--dry-run: nothing written" in out and set(os.listdir(d)) == before


def test_cli_import_stops_on_a_missing_plan_and_allow_missing_leaves_it_out(chain, capsys):
    d, spec_path = chain
    parts = json.loads(open(spec_path, encoding="utf-8").read())
    fill(str(d / "annotate_primary.xlsx"), {(parts[3]["part_id"], "A"): []})
    capsys.readouterr()
    out = str(d / "bench_v1.json")
    assert run_cli("import", "--spec", spec_path, "--primary", str(d / "annotate_primary.xlsx"), "--out", out) == 1
    cap = capsys.readouterr()
    assert parts[3]["part_id"] in cap.out and "blocking" in cap.err and "nothing written" in cap.err and not os.path.exists(out)
    assert run_cli("import", "--spec", spec_path, "--primary", str(d / "annotate_primary.xlsx"), "--out", out, "--allow-missing") == 0
    assert len(D.load_dataset(out)) == 17


def test_cli_import_names_the_misspelt_operation(chain, capsys):
    d, spec_path = chain
    parts = json.loads(open(spec_path, encoding="utf-8").read())
    fill(str(d / "annotate_primary.xlsx"), {(parts[2]["part_id"], "A"): ["Facing", "Drillng", "Inspection"]})
    capsys.readouterr()
    assert run_cli("import", "--spec", spec_path, "--primary", str(d / "annotate_primary.xlsx"), "--out", str(d / "bench_v1.json")) == 1
    err = capsys.readouterr().err
    assert "unknown operation 'Drillng'" in err and "Drilling" in err and not os.path.exists(d / "bench_v1.json")


def test_cli_import_lists_the_conflicts_but_still_writes(chain, capsys):
    d, spec_path = chain
    parts = json.loads(open(spec_path, encoding="utf-8").read())
    fill(str(d / "annotate_primary.xlsx"), {(parts[0]["part_id"], "A"): ["Facing", "Inspection"]})      # too short for the part's features
    capsys.readouterr()
    out = str(d / "bench_v1.json")
    assert run_cli("import", "--spec", spec_path, "--primary", str(d / "annotate_primary.xlsx"), "--out", out) == 0
    text = capsys.readouterr().out
    assert "plan(s) the rule base rejects" in text and parts[0]["part_id"] in text
    assert D.load_dataset(out)[0]["reference_routes"] == [["Facing", "Inspection"]]


def test_cli_prints_the_help_without_a_command(capsys):
    with pytest.raises(SystemExit) as e:
        run_cli()
    assert e.value.code == 2
