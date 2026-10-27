"""Step 27 (WP12): the release in one command (capp/release.py, scripts/release.py).

Offline. The runs are made by the real harness with a deterministic fake planner (the helpers of the paper-draft tests), in a fake repository,
so a release is built from real records, the real Table 4 code, the real case-study code and the real paper draft. What the tests pin down:
the recipe is checked strictly; a release holds exactly the files that are listed, with their hashes; building again from the same sources
gives byte-identical files (also on another folder, and with CRLF line endings in the hand-written files); `verify` finds a file that was
changed, removed or added, and a source that changed after the build; the files written by hand are never touched; a failed build leaves a
previous release as it was; and no path of the machine ends up in a file that is compared later.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capp import paperdraft as P, release as R                                                # noqa: E402
from test_paperdraft import ROWS, bench_parts, cfg_dict, full_text, make_runs, sealed, write_extraction_run      # noqa: E402,F401  (sealed is a fixture)


# ---------------------------------------------------------------- a fake repository with finished runs
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    repo = tmp_path_factory.mktemp("world")
    (repo / "results").mkdir()
    make_runs(str(repo / "results"))
    (repo / "configs").mkdir()
    (repo / "configs" / "grid.json").write_text(json.dumps(cfg_dict("results")))
    return repo


@pytest.fixture()
def repo(world, tmp_path):
    dst = tmp_path / "repo"
    shutil.copytree(world, dst)
    return dst


def recipe(repo, **over):
    r = {"name": "t", "grid": "configs/grid.json", "results": "results", "text": "paper/paper_text.json"}
    r.update(over)
    path = repo / "configs" / "release.json"
    path.write_text(json.dumps(r, indent=2))
    return str(path)


def build(repo, out=None, figures=False, **over):
    return R.build(recipe(repo, **over), str(out or repo / "release" / "t"), root=str(repo), figures=figures)


def files_of(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in Path(folder).rglob("*") if p.is_file()}


def run_folder(repo, system, model_tag="p_a"):
    return next(d for d in sorted(os.listdir(repo / "results")) if d == f"{model_tag}_{system}" or d.startswith(model_tag) and d.endswith(system))


def write_prose(repo, text=None, crlf=False):
    p = repo / "paper"
    p.mkdir(exist_ok=True)
    raw = json.dumps(text if text is not None else full_text(), indent=2)
    (p / "paper_text.json").write_bytes((raw.replace("\n", "\r\n") if crlf else raw).encode("utf-8"))
    return p / "paper_text.json"


# ---------------------------------------------------------------- the recipe
MIN = {"name": "x", "grid": "g.json"}


def test_a_minimal_recipe_gets_every_key_and_normalising_twice_changes_nothing():
    r = R.normalise_recipe(MIN)
    assert r == {"name": "x", "grid": "g.json", "results": None, "freeze": None, "benchmark": None, "text": R.DEFAULT_TEXT, "case_studies": None,
                 "extraction": [], "notes": ""}
    full = dict(MIN, case_studies={"runs": ["results/a", "results/b"], "instructive": ["DEV_001:1=the repair kept a wrong order"]},
                extraction=["results/fe1_extraction"], benchmark="data/bench_v1.json", notes="n", _comment="dropped")
    n = R.normalise_recipe(full)
    assert n["case_studies"] == {"runs": ["results/a", "results/b"], "commentary": None, "per_type": 3, "types": None,
                                 "instructive": ["DEV_001:1=the repair kept a wrong order"]}
    assert "_comment" not in n and R.normalise_recipe(n) == n


@pytest.mark.parametrize("change,fragment", [
    ({"nme": "x"}, "unknown key(s) ['nme']"), ({"name": "has space"}, "name:"), ({"name": ""}, "name:"), ({"name": None}, "name:"),
    ({"grid": None}, "grid:"), ({"grid": ""}, "grid:"), ({"results": 3}, "results:"), ({"freeze": ""}, "freeze:"), ({"text": ""}, "text:"),
    ({"extraction": "a"}, "extraction:"), ({"extraction": ["a", "a"]}, "extraction:"), ({"extraction": [1]}, "extraction:"), ({"notes": 3}, "notes:"),
    ({"case_studies": []}, "case_studies:"), ({"case_studies": {}}, "case_studies.runs"), ({"case_studies": {"runs": []}}, "case_studies.runs"),
    ({"case_studies": {"runs": ["a", "a"]}}, "case_studies.runs"), ({"case_studies": {"runs": ["a"], "per_type": 0}}, "per_type"),
    ({"case_studies": {"runs": ["a"], "per_type": 11}}, "per_type"), ({"case_studies": {"runs": ["a"], "per_type": True}}, "per_type"),
    ({"case_studies": {"runs": ["a"], "types": ["nonsense"]}}, "case_studies.types"), ({"case_studies": {"runs": ["a"], "types": []}}, "case_studies.types"),
    ({"case_studies": {"runs": ["a"], "types": "order"}}, "case_studies.types"),
    ({"case_studies": {"runs": ["a"], "instructive": ["no reason given"]}}, "case_studies.instructive"),
    ({"case_studies": {"runs": ["a"], "instructive": [3]}}, "case_studies.instructive"),
    ({"case_studies": {"runs": ["a"], "commentary": 3}}, "case_studies.commentary"), ({"case_studies": {"runs": ["a"], "extra": 1}}, "unknown key(s) ['extra']"),
])
def test_a_bad_recipe_is_refused_with_the_reason(change, fragment):
    with pytest.raises(R.ReleaseError) as exc:
        R.normalise_recipe(dict(MIN, **change))
    assert fragment in str(exc.value)


def test_every_problem_of_a_recipe_is_named_at_once():
    with pytest.raises(R.ReleaseError) as exc:
        R.normalise_recipe({"name": "a b", "extraction": "x", "oops": 1})
    msg = str(exc.value)
    assert "name:" in msg and "grid:" in msg and "extraction:" in msg and "unknown key(s) ['oops']" in msg
    with pytest.raises(R.ReleaseError, match="expected a JSON object"):
        R.normalise_recipe([1])


def test_recipe_files_that_cannot_be_read(tmp_path):
    with pytest.raises(R.ReleaseError, match="cannot be read"):
        R.load_recipe(str(tmp_path / "nope.json"))
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(R.ReleaseError, match="not valid JSON"):
        R.load_recipe(str(tmp_path / "bad.json"))


def test_the_shipped_template_is_a_valid_recipe_for_the_shipped_grid():
    r = R.load_recipe(os.path.join(ROOT, "configs", "release_template.json"))
    assert r["grid"] == "configs/ablation_template.json" and r["case_studies"] is None
    c = R.check(os.path.join(ROOT, "configs", "release_template.json"), ROOT)
    assert c["missing"] == [] and c["grid"]["cells"] == 30 and c["grid"]["sealed"] is False


# ---------------------------------------------------------------- check
def test_check_lists_what_a_build_would_find(repo):
    c = R.check(recipe(repo), str(repo))
    assert c["missing"] == [] and c["grid"]["cells"] == 3 * 8 and c["grid"]["usable"] == 3 * 8 and c["grid"]["built"] == 8 and c["grid"]["rows"] == 9
    assert c["freeze"] is None and "prose" in R.format_check(c) and "not there yet" in R.format_check(c)
    c = R.check(recipe(repo, case_studies={"runs": ["results/nope"]}, benchmark="data/none.json"), str(repo))
    assert [m["path"] for m in c["missing"]] == ["data/none.json", "results/nope"]
    txt = R.format_check(c)
    assert "MISSING" in txt and "build would stop" in txt


def test_a_run_folder_counts_only_with_its_records(repo):
    (repo / "empty_run").mkdir()
    c = R.check(recipe(repo, extraction=["empty_run"]), str(repo))
    assert [m["path"] for m in c["missing"]] == ["empty_run"]


# ---------------------------------------------------------------- build
def test_a_build_writes_the_listed_files_and_nothing_else(repo):
    res = build(repo)
    out = repo / "release" / "t"
    on_disk = set(files_of(out))
    assert on_disk == set(res["files"]) | {R.MANIFEST}
    assert {"RELEASE.md", "SHA256SUMS.txt", "inputs/recipe.json", "inputs/grid.json", "inputs/paper_text.json", "paper/paper_draft.md",
            "paper/readiness.txt", "tables/table4.md", "tables/paired.md", "tables/manifest.json"} <= on_disk
    assert not any(f.startswith(("case_studies/", "benchmark/")) or f in ("FREEZE.json", "freeze_verify.txt") for f in on_disk)
    man = json.load(open(out / R.MANIFEST, encoding="utf-8"))
    assert man["format"] == R.FORMAT and man["name"] == "t" and man["figures"] is False and man["files"] == res["files"] and man["freeze"] is None
    assert man["dataset"] == "dev_v2" and man["sealed"] is False and man["ready"] is False and man["blocking"]
    assert R.MANIFEST not in man["files"] and "SHA256SUMS.txt" in man["files"]
    assert [s[0] for s in res["steps"]] == ["freeze", "tables", "paper draft"]
    assert not [d for d in os.listdir(repo / "release") if d.startswith(".release-")]


def test_the_sums_file_is_sha256sum_compatible_and_complete(repo):
    res = build(repo)
    out = repo / "release" / "t"
    lines = (out / "SHA256SUMS.txt").read_text().splitlines()
    got = dict(reversed(ln.split("  ", 1)) for ln in lines)
    assert set(got) == set(res["files"]) - {"SHA256SUMS.txt"} and all(got[f] == res["files"][f] for f in got)
    assert all(len(h) == 64 for h in got.values())


def test_the_tables_of_the_release_are_the_tables_of_the_draft(repo):
    build(repo)
    out = repo / "release" / "t"
    table = (out / "tables" / "table4.md").read_text()
    rows = [ln for ln in table.splitlines() if ln.startswith("| A")]
    draft = (out / "paper" / "paper_draft.md").read_text()
    assert len(rows) == 3 * len(ROWS) and all(r in draft for r in rows)
    assert "![Ablation path per configuration](../tables/fig_ablation.png)" not in draft          # figures were not asked for


def test_the_release_folder_says_what_it_is(repo):
    build(repo)
    md = (repo / "release" / "t" / "RELEASE.md").read_text()
    assert md.startswith("# Release: t") and "DEVELOPMENT RESULTS, NOT PAPER RESULTS" in md
    assert "| `tables/` |" in md and "| `paper/` |" in md and "| `inputs/` |" in md and "| `RELEASE.md` | this file |" in md
    assert "| `case_studies/` |" not in md and "| `benchmark/` |" not in md
    assert "Not ready: " in md and "- [BLOCKING] " in md and "git tag baseline-v0" in md and "git tag release-t" in md
    assert "What this does not show" in md and "does not show that no test result was seen before it was made" in md
    assert "no manifest (`FREEZE.json`)" in md


def test_the_notes_of_the_recipe_are_in_the_release_file(repo):
    build(repo, notes="Dry run for the supervisor.")
    assert "Dry run for the supervisor." in (repo / "release" / "t" / "RELEASE.md").read_text()


def test_appendix_b_of_the_draft_names_the_release_commands(repo):
    build(repo)
    draft = (repo / "release" / "t" / "paper" / "paper_draft.md").read_text()
    appendix = draft.split("## Appendix B: how to reproduce", 1)[1].split("## Appendix C", 1)[0]
    assert "python scripts/release.py build <recipe>" in appendix and "python scripts/release.py verify <this folder>" in appendix
    assert "make_tables.py" not in appendix


def test_the_prose_is_never_touched_and_is_copied(repo):
    p = write_prose(repo, full_text(title="My title"), crlf=True)
    before = p.read_bytes()
    build(repo)
    assert p.read_bytes() == before                                                       # not even the line endings
    copy = (repo / "release" / "t" / "inputs" / "paper_text.json").read_bytes()
    assert b"\r\n" not in copy and json.loads(copy)["title"] == "My title"
    assert (repo / "release" / "t" / "paper" / "paper_draft.md").read_text().startswith("# My title")


def test_without_prose_the_source_is_not_created_but_the_copy_has_the_empty_sections(repo):
    build(repo)
    assert not (repo / "paper" / "paper_text.json").exists()
    copy = json.load(open(repo / "release" / "t" / "inputs" / "paper_text.json", encoding="utf-8"))
    assert set(copy) == set(P.TEXT_FIELDS) and not any(copy.values())


def test_all_the_prose_but_a_development_grid_leaves_only_the_dataset_blocking(repo):
    write_prose(repo)
    res = build(repo)
    assert res["ready"] is False
    assert [m for m in res["manifest"]["blocking"] if "development set" not in m and "freeze" not in m and "missing, partial" not in m] == []


# ---------------------------------------------------------------- the same sources give the same files
def test_building_again_gives_the_same_bytes_on_another_folder(repo):
    write_prose(repo)
    a = build(repo, out=repo / "r1")
    b = build(repo, out=repo / "elsewhere" / "r2")
    fa, fb = files_of(repo / "r1"), files_of(repo / "elsewhere" / "r2")
    assert a["files"] == b["files"] and set(fa) == set(fb)
    assert all(fa[f] == fb[f] for f in fa if f != R.MANIFEST)


def test_no_path_of_this_machine_is_in_a_release(repo):
    write_prose(repo)
    build(repo, out=repo / "r1")
    for rel, data in files_of(repo / "r1").items():
        if rel == R.MANIFEST or rel.endswith(".png"):
            continue                                                                       # the manifest holds the build record
        text = data.decode("utf-8")
        assert str(repo) not in text and ".release-" not in text and str(repo).replace("\\", "/") not in text, rel


def test_a_problem_message_loses_the_folders_of_the_machine(repo):
    (repo / "data").mkdir()
    (repo / "data" / "bench.json").write_text("{not json")
    (repo / "data" / "bench_datasheet.md").write_text("# a datasheet\n")                  # so the build goes on and the draft names the problem
    res = build(repo, benchmark="data/bench.json")
    readiness = (repo / "release" / "t" / "paper" / "readiness.txt").read_text()
    assert "could not be read" in readiness and "bench.json" in readiness and str(repo) not in readiness
    assert any("could not be read" in m and str(repo) not in m for m in res["manifest"]["blocking"])
    assert R._stable(f"x {repo}/a/b and {repo}", str(repo)) == "x a/b and ."


def test_an_unreadable_benchmark_without_a_datasheet_stops_the_build(repo):
    (repo / "data").mkdir()
    (repo / "data" / "bench.json").write_text("{not json")
    with pytest.raises(R.ReleaseError, match="benchmark data/bench.json cannot be read"):
        build(repo, benchmark="data/bench.json")
    assert not (repo / "release").exists() or not os.listdir(repo / "release")


def test_an_unreadable_extraction_run_is_named_without_the_path_of_the_machine(repo):
    (repo / "fe_bad").mkdir()
    (repo / "fe_bad" / "records.jsonl").write_text("{not json\n")
    res = build(repo, extraction=["fe_bad"])
    assert any("extraction run" in m and "could not be scored" in m and str(repo) not in m for m in res["manifest"]["blocking"])
    assert str(repo) not in (repo / "release" / "t" / "paper" / "readiness.txt").read_text()


def test_line_endings_of_the_hand_written_files_do_not_change_the_release(repo):
    write_prose(repo, crlf=True)
    a = build(repo, out=repo / "r1")["files"]
    write_prose(repo, crlf=False)
    b = build(repo, out=repo / "r2")["files"]
    assert a == b


def test_the_hash_of_a_text_file_ignores_crlf_and_a_png_is_hashed_as_it_is(tmp_path):
    (tmp_path / "a.md").write_bytes(b"x\ny\n")
    (tmp_path / "b.md").write_bytes(b"x\r\ny\r\n")
    (tmp_path / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "b.png").write_bytes(b"\x89PNG\n\x1a\n")
    assert R._sha(str(tmp_path / "a.md")) == R._sha(str(tmp_path / "b.md"))
    assert R._sha(str(tmp_path / "a.png")) != R._sha(str(tmp_path / "b.png"))


# ---------------------------------------------------------------- verify
def test_a_fresh_release_verifies_by_hash_and_by_rebuilding(repo):
    write_prose(repo)
    build(repo)
    v = R.verify(str(repo / "release" / "t"), str(repo))
    assert v["ok"] and v["modified"] == v["missing"] == v["extra"] == [] and v["regen"]["differs"] == [] and v["files"] > 10
    assert "VERIFIED" in R.format_verify(v) and "NOT VERIFIED" not in R.format_verify(v) and "every file is identical" in R.format_verify(v)
    v = R.verify(str(repo / "release" / "t"), str(repo), regenerate=False)
    assert v["ok"] and v["regen"] is None and "skipped" in R.format_verify(v)


def test_a_file_that_was_changed_removed_or_added_is_found(repo):
    build(repo)
    out = repo / "release" / "t"
    (out / "tables" / "table4.md").write_text("tampered\n")
    os.remove(out / "tables" / "paired.csv")
    (out / "tables" / "extra.txt").write_text("not from the build\n")
    v = R.verify(str(out), str(repo), regenerate=False)
    assert v["ok"] is False and v["modified"] == ["tables/table4.md"] and v["missing"] == ["tables/paired.csv"] and v["extra"] == ["tables/extra.txt"]
    txt = R.format_verify(v)
    assert "modified: tables/table4.md" in txt and "missing: tables/paired.csv" in txt and "not in the manifest: tables/extra.txt" in txt and "NOT VERIFIED" in txt
    v = R.verify(str(out), str(repo))                                      # the folder was changed, but a rebuild gives back what was built
    assert v["ok"] is False and v["modified"] == ["tables/table4.md"] and v["regen"]["differs"] == [] and v["regen"]["only_now"] == []


def test_a_source_that_changed_after_the_release_is_found_under_inputs(repo):
    write_prose(repo, full_text(title="First"))
    build(repo)
    write_prose(repo, full_text(title="Second"))
    v = R.verify(str(repo / "release" / "t"), str(repo))
    assert v["ok"] is False and v["modified"] == [] and "inputs/paper_text.json" in v["regen"]["differs"] and "paper/paper_draft.md" in v["regen"]["differs"]
    assert "a difference under inputs/ means a source file changed" in R.format_verify(v)


def test_a_run_added_after_the_release_is_found(repo):
    build(repo)
    for d in os.listdir(repo / "results"):
        if d.endswith("_b1") and d.startswith("p_a"):
            shutil.rmtree(repo / "results" / d)
    v = R.verify(str(repo / "release" / "t"), str(repo))
    assert v["ok"] is False and any(f.startswith("tables/") for f in v["regen"]["differs"])


def test_a_source_that_appears_or_disappears_after_the_release_is_found(repo):
    case = {"runs": [f"results/{run_folder(repo, 'b0')}"], "commentary": "case_commentary.json"}
    out = repo / "release" / "t"
    build(repo, case_studies=case)                                                         # the commentary file is not there yet
    (repo / "case_commentary.json").write_text("{}")
    v = R.verify(str(out), str(repo))
    assert v["ok"] is False and v["regen"]["only_now"] == ["inputs/case_commentary.json"] and v["regen"]["only_release"] == [] and v["regen"]["differs"] == []
    assert "only in the rebuild: inputs/case_commentary.json" in R.format_verify(v)
    build(repo, case_studies=case)                                                         # now it is in the release, and then it goes
    os.remove(repo / "case_commentary.json")
    v = R.verify(str(out), str(repo))
    assert v["ok"] is False and v["regen"]["only_release"] == ["inputs/case_commentary.json"] and v["regen"]["only_now"] == []
    assert "only in the release: inputs/case_commentary.json" in R.format_verify(v)


def test_verify_refuses_what_is_not_a_release(tmp_path, repo):
    with pytest.raises(R.ReleaseError, match="not a release folder"):
        R.verify(str(tmp_path), str(repo))
    (tmp_path / R.MANIFEST).write_text("[]")
    with pytest.raises(R.ReleaseError, match="not a release manifest"):
        R.verify(str(tmp_path), str(repo))
    build(repo)
    os.remove(repo / "release" / "t" / "inputs" / "recipe.json")
    with pytest.raises(R.ReleaseError, match="recipe.json"):
        R.verify(str(repo / "release" / "t"), str(repo))


# ---------------------------------------------------------------- the output folder
def test_building_into_an_old_release_replaces_it_and_leaves_foreign_files(repo):
    out = repo / "release" / "t"
    build(repo, case_studies={"runs": [f"results/{run_folder(repo, 'b0')}"]})
    assert (out / "case_studies" / "cases.json").is_file()
    (out / "my_notes.txt").write_text("mine")
    build(repo)
    assert not (out / "case_studies").exists() and (out / "my_notes.txt").read_text() == "mine"
    assert R.verify(str(out), str(repo), regenerate=False)["extra"] == []                 # my_notes.txt is not in an owned place


def test_a_folder_that_is_not_a_release_is_refused_untouched(repo):
    out = repo / "mine"
    out.mkdir()
    (out / "thesis.docx").write_text("precious")
    with pytest.raises(R.ReleaseError, match="not a release folder"):
        R.build(recipe(repo), str(out), root=str(repo), figures=False)
    assert os.listdir(out) == ["thesis.docx"]
    (repo / "afile").write_text("x")
    with pytest.raises(R.ReleaseError, match="is a file"):
        R.build(recipe(repo), str(repo / "afile"), root=str(repo), figures=False)
    empty = repo / "empty"
    empty.mkdir()
    R.build(recipe(repo), str(empty), root=str(repo), figures=False)
    assert (empty / "RELEASE.md").is_file()


def test_a_missing_required_input_stops_before_anything_is_written(repo):
    with pytest.raises(R.ReleaseError, match="required input\\(s\\) not found: case-study run results/nope"):
        build(repo, case_studies={"runs": ["results/nope"]})
    assert not (repo / "release").exists()


def test_a_failed_build_leaves_the_previous_release_as_it_was(repo):
    write_prose(repo)
    build(repo)
    out = repo / "release" / "t"
    before = files_of(out)
    (repo / "paper" / "paper_text.json").write_text("{broken")
    with pytest.raises(P.PaperError):
        build(repo)
    assert files_of(out) == before
    assert R.verify(str(out), str(repo), regenerate=False)["ok"]
    assert not [d for d in os.listdir(repo / "release") if d.startswith(".release-")]


def test_a_bad_grid_stops_the_build_and_leaves_no_temporary_folder(repo):
    (repo / "configs" / "grid.json").write_text("{}")
    with pytest.raises(Exception):
        build(repo)
    assert not os.path.exists(repo / "release") or not [d for d in os.listdir(repo / "release") if d.startswith(".release-")]


# ---------------------------------------------------------------- the parts of a release
def test_case_studies_are_made_from_the_runs_with_the_commentary_of_the_author(repo):
    folder = f"results/{run_folder(repo, 'b0')}"
    res = build(repo, case_studies={"runs": [folder]})
    out = repo / "release" / "t"
    cases = json.load(open(out / "case_studies" / "cases.json", encoding="utf-8"))["cases"]
    assert cases and {"case_studies.md", "cases.json", "selection.csv"} <= {Path(f).name for f in files_of(out / "case_studies")}
    assert dict(res["steps"])["case studies"] == f"{len(cases)} case(s), {len(cases)} without commentary"
    com = repo / "case_commentary.json"
    com.write_text(json.dumps({cases[0]["id"]: "The model put Boring before Drilling."}))
    before = com.read_bytes()
    res = build(repo, case_studies={"runs": [folder], "commentary": "case_commentary.json"})
    assert com.read_bytes() == before
    assert "The model put Boring before Drilling." in (out / "case_studies" / "case_studies.md").read_text()
    assert dict(res["steps"])["case studies"] == f"{len(cases)} case(s), {len(cases) - 1} without commentary"
    assert json.load(open(out / "inputs" / "case_commentary.json", encoding="utf-8")) == json.loads(before)
    assert "no case studies given" not in (out / "paper" / "readiness.txt").read_text()
    assert "case commentary" in R.format_check(R.check(recipe(repo, case_studies={"runs": [folder], "commentary": "case_commentary.json"}), str(repo)))


def test_two_case_study_runs_with_the_same_folder_name_are_refused(repo):
    a = run_folder(repo, "b0")
    (repo / "other").mkdir()
    shutil.copytree(repo / "results" / a, repo / "other" / a)
    with pytest.raises(R.ReleaseError, match="same folder name"):
        build(repo, case_studies={"runs": [f"results/{a}", f"other/{a}"]})


def test_the_benchmark_is_copied_with_its_datasheet(repo):
    (repo / "data").mkdir()
    json.dump(bench_parts(), open(repo / "data" / "bench.json", "w"))
    res = build(repo, benchmark="data/bench.json")
    out = repo / "release" / "t"
    sheet = (out / "benchmark" / "DATASHEET.md").read_text()
    assert sheet.startswith("# Expert benchmark: datasheet") and "Generated by capp/benchmark.py." in sheet and " on 20" not in sheet
    assert json.load(open(out / "benchmark" / "bench.json", encoding="utf-8")) == json.load(open(repo / "data" / "bench.json", encoding="utf-8"))
    assert "regenerated from the benchmark file" in dict(res["steps"])["benchmark"]
    assert "Expert benchmark `data/bench.json`: 5 parts" in (out / "paper" / "paper_draft.md").read_text()
    assert "| `benchmark/` |" in (out / "RELEASE.md").read_text()
    assert R.verify(str(out), str(repo))["ok"]


def test_the_datasheet_written_at_import_is_used_when_it_is_there(repo):
    (repo / "data").mkdir()
    json.dump(bench_parts(), open(repo / "data" / "bench.json", "w"))
    (repo / "data" / "bench_datasheet.md").write_text("# The datasheet of the import\r\nwith a conflicts section\r\n")
    res = build(repo, benchmark="data/bench.json")
    out = repo / "release" / "t"
    assert (out / "benchmark" / "DATASHEET.md").read_bytes() == b"# The datasheet of the import\nwith a conflicts section\n"
    assert "the datasheet written at import" in dict(res["steps"])["benchmark"] and "the datasheet written at import" in (out / "RELEASE.md").read_text()


def test_a_placeholder_benchmark_is_blocking_in_the_release(repo):
    (repo / "data").mkdir()
    json.dump(bench_parts(placeholder=True), open(repo / "data" / "bench.json", "w"))
    res = build(repo, benchmark="data/bench.json")
    assert any("PLACEHOLDER" in m for m in res["manifest"]["blocking"])
    assert "PLACEHOLDER" in (repo / "release" / "t" / "RELEASE.md").read_text()


def test_an_extraction_run_reaches_the_draft(repo):
    write_extraction_run(str(repo / "fe1_extraction"))
    res = build(repo, extraction=["fe1_extraction"])
    draft = (repo / "release" / "t" / "paper" / "paper_draft.md").read_text()
    assert "M9 extraction quality - fe1_extraction: 3 of 3 drawings scored" in draft
    assert "`fe1_extraction`" in (repo / "release" / "t" / "RELEASE.md").read_text()
    assert "no extraction run given" not in (repo / "release" / "t" / "paper" / "readiness.txt").read_text()
    assert res["steps"][-1][0] == "paper draft"


def test_figures_are_made_on_request_and_verify_uses_the_same_choice(repo):
    pytest.importorskip("matplotlib")
    res = build(repo, figures=True)
    out = repo / "release" / "t"
    assert "tables/fig_ablation.png" in res["files"] and (out / "tables" / "fig_ablation.png").is_file()
    assert "![Ablation path per configuration](../tables/fig_ablation.png)" in (out / "paper" / "paper_draft.md").read_text()
    assert json.load(open(out / R.MANIFEST, encoding="utf-8"))["figures"] is True
    v = R.verify(str(out), str(repo))
    assert v["ok"] and v["regen"]["figures"] == []
    res = build(repo, figures=False)
    assert not any(f.endswith(".png") for f in res["files"]) and R.verify(str(out), str(repo))["ok"]


def test_a_figure_that_differs_only_in_bytes_is_a_warning_not_a_failure(repo, monkeypatch):
    pytest.importorskip("matplotlib")
    build(repo, figures=True)
    out = repo / "release" / "t"
    png = out / "tables" / "fig_ablation.png"
    png.write_bytes(png.read_bytes() + b"x")                                                # the file in the folder changed: a failure
    v = R.verify(str(out), str(repo), regenerate=False)
    assert v["ok"] is False and v["modified"] == ["tables/fig_ablation.png"]
    # the same difference seen only by the rebuild is a warning: simulate an encoder that writes other bytes
    build(repo, figures=True)
    from capp import tablefigs
    real = tablefigs.make_figures

    def other_bytes(res, out_dir):
        made, why = real(res, out_dir)
        for name in made:
            p = os.path.join(out_dir, name)
            with open(p, "ab") as f:
                f.write(b"encoder-version")
        return made, why
    monkeypatch.setattr(tablefigs, "make_figures", other_bytes)
    v = R.verify(str(out), str(repo))
    assert v["ok"] is True and "tables/fig_ablation.png" in v["regen"]["figures"] and "differ in bytes only" in R.format_verify(v)


# ---------------------------------------------------------------- the sealed path
def sealed_recipe(repo, **over):
    r = {"name": "final", "grid": "configs/final.json", "text": "paper/paper_text.json"}
    r.update(over)
    path = Path(repo) / "configs" / "release.json"
    path.write_text(json.dumps(r))
    return str(path)


def test_a_sealed_release_under_a_matching_freeze_is_ready(sealed):
    repo, cfg_path, man, _ = sealed
    write_prose(Path(repo))
    res = R.build(sealed_recipe(repo), str(Path(repo) / "release" / "final"), root=repo, figures=False)
    out = Path(repo) / "release" / "final"
    assert res["ready"] is True and res["manifest"]["blocking"] == [] and res["manifest"]["sealed"] is True
    assert res["freeze"] == {"id": man["id"], "status": "ok"} and "FREEZE.json" in res["files"] and "freeze_verify.txt" in res["files"]
    md = (out / "RELEASE.md").read_text()
    assert "DEVELOPMENT RESULTS" not in md and "NOT READY" not in md and "Ready: every blocking item" in md and f"`{man['id']}`, status **ok**" in md
    assert (out / "freeze_verify.txt").read_text().startswith(f"Freeze {man['id']}: OK")
    assert json.load(open(out / "FREEZE.json", encoding="utf-8"))["id"] == man["id"]
    assert R.verify(str(out), repo)["ok"]


def test_a_drifted_tree_blocks_the_release_and_verify_notices_it_later(sealed):
    repo, cfg_path, man, _ = sealed
    write_prose(Path(repo))
    out = Path(repo) / "release" / "final"
    R.build(sealed_recipe(repo), str(out), root=repo, figures=False)
    (Path(repo) / "capp" / "planners.py").write_text("x = 2\n")                              # a behaviour file, after the build
    v = R.verify(str(out), repo)
    assert v["ok"] is False and "freeze_verify.txt" in v["regen"]["differs"] and "RELEASE.md" in v["regen"]["differs"]
    res = R.build(sealed_recipe(repo), str(out), root=repo, figures=False)
    assert res["ready"] is False and res["freeze"]["status"] == "drift" and any("drifted" in m for m in res["manifest"]["blocking"])
    assert "NOT READY" in (out / "RELEASE.md").read_text() and (out / "freeze_verify.txt").read_text().startswith(f"Freeze {man['id']}: DRIFT")
    assert R.verify(str(out), repo)["ok"]                                                    # it is the release of the drifted tree, and says so


def test_an_unreadable_freeze_manifest_is_named_and_blocks(sealed):
    repo, cfg_path, man, _ = sealed
    write_prose(Path(repo))
    (Path(repo) / "FREEZE.json").write_text("{not json")
    res = R.build(sealed_recipe(repo), str(Path(repo) / "release" / "final"), root=repo, figures=False)
    assert res["ready"] is False and res["freeze"]["status"] == "unreadable" and "FREEZE.json" not in res["files"]
    assert any("freeze manifest" in m and "could not be used" in m for m in res["manifest"]["blocking"])
    assert str(repo) not in (Path(repo) / "release" / "final" / "paper" / "readiness.txt").read_text()


def test_a_named_freeze_manifest_that_is_not_there_is_reported(sealed):
    repo, cfg_path, man, _ = sealed
    res = R.build(sealed_recipe(repo, freeze="nothing/FREEZE.json"), str(Path(repo) / "release" / "final"), root=repo, figures=False)
    assert res["freeze"] is None and any("freeze manifest nothing/FREEZE.json not found" in m for m in res["manifest"]["blocking"])


# ---------------------------------------------------------------- the command line
@pytest.fixture()
def cli():
    spec = importlib.util.spec_from_file_location("release_cli", os.path.join(ROOT, "scripts", "release.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def abs_recipe(repo, tmp_path):
    """A recipe with absolute paths, because the command line works on the real repository root."""
    r = {"name": "cli", "grid": str(repo / "configs" / "grid.json"), "results": str(repo / "results"), "freeze": str(tmp_path / "no_freeze.json"),
         "text": str(repo / "paper" / "paper_text.json")}
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(r))
    return str(path), r


def test_cli_check_build_and_verify(cli, abs_recipe, tmp_path, capsys):
    path, _ = abs_recipe
    assert cli.main(["check", path]) == 0
    assert "All required inputs are there" in capsys.readouterr().out
    out = tmp_path / "rel"
    assert cli.main(["build", path, "--out", str(out), "--no-figures"]) == 0
    text = capsys.readouterr().out
    assert "NOT READY" in text and "[BLOCKING]" in text and "paper draft" in text and "scripts/release.py verify" in text
    assert (out / "RELEASE.md").is_file() and not list(out.glob("tables/*.png"))
    assert cli.main(["verify", str(out)]) == 0
    assert "VERIFIED" in capsys.readouterr().out
    assert cli.main(["verify", str(out), "--no-regenerate"]) == 0
    (out / "paper" / "readiness.txt").write_text("tampered")
    assert cli.main(["verify", str(out), "--no-regenerate"]) == 1
    assert "NOT VERIFIED" in capsys.readouterr().out


def test_cli_strict_exits_1_when_a_blocking_item_is_open_but_still_writes(cli, abs_recipe, tmp_path, capsys):
    path, _ = abs_recipe
    out = tmp_path / "rel"
    assert cli.main(["build", path, "--out", str(out), "--no-figures", "--strict"]) == 1
    assert "--strict: blocking item(s) open." in capsys.readouterr().err and (out / "RELEASE.md").is_file()


def test_cli_problems_exit_2(cli, abs_recipe, tmp_path, capsys):
    path, r = abs_recipe
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dict(r, extraction=["nowhere"])))
    assert cli.main(["check", str(bad)]) == 2
    assert cli.main(["build", str(bad), "--out", str(tmp_path / "x")]) == 2
    assert "STOPPED: required input(s) not found: extraction run nowhere" in capsys.readouterr().err and not (tmp_path / "x").exists()
    (tmp_path / "typo.json").write_text(json.dumps({"name": "a", "grid": "g", "nme": 1}))
    assert cli.main(["build", str(tmp_path / "typo.json")]) == 2
    assert cli.main(["verify", str(tmp_path)]) == 2
    assert "not a release folder" in capsys.readouterr().err
    assert cli.main(["check", str(tmp_path / "missing.json")]) == 2


def test_the_script_runs_as_a_program_on_the_shipped_template():
    res = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "release.py"), "check", os.path.join("configs", "release_template.json")],
                         capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=120)
    assert res.returncode == 0 and "Recipe mtp1_rehearsal" in res.stdout and "All required inputs are there" in res.stdout
