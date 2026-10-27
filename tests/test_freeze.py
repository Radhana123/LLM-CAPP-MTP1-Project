"""Step 25: the configuration freeze (capp/freeze.py, scripts/freeze.py, and its use in run.json, scripts/run_eval.py, capp/tables.py).

Offline. The module works on any folder, so most tests build a small fake repository in a temp folder and a fake config object, and check
that exactly the right edits turn the verdict into drift: a behaviour file, a config constant, a model id, a dataset, the grid, an extra
file; and that the edits that must not (analysis code, the tests, a Windows checkout with CRLF line endings, indentation of the grid
config) do not. A few tests run scripts/run_eval.py as a subprocess to see the gate on the sealed datasets.
"""
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capp import config, dataset, freeze as F, harness, tables as T      # noqa: E402
import route_builder as rb                                              # noqa: E402

PARTS = [{"part_id": "P1", "features": ["Hole"]}, {"part_id": "P2", "features": ["Slot"]}]
GRID = {"name": "g", "dataset": "dev_v2", "rows": [{"id": "A0", "system": "b1"}, {"id": "A1", "system": "b0", "parent": "A0"}]}
FILES = {
    "capp/__init__.py": "", "capp/planners.py": "PROMPT = 'plan'\n", "capp/tables.py": "TABLE = 1\n", "capp/subpkg/deep.py": "deep = 1\n",
    "week1/route_builder.py": "rb = 1\n", "week1/token_map.json": '{"a": 1}\n', "week6/app.py": "ui = 1\n", "week6/full_pipeline.py": "pipe = 1\n",
    "scripts/run_eval.py": "cli = 1\n", "tests/test_x.py": "t = 1\n", "README.md": "hello\n",
}


def fake_cfg(**over):
    c = types.SimpleNamespace(PLANNER_TEMPERATURE=0.3, PLANNER_MAX_TOKENS=200, ROOT=Path("/somewhere"), PRICE_USD_PER_MTOK={"m": (1.0, 2.0)},
                              MODEL_ROLES=("primary", "small"), _private=5, lowercase=1, VALID=("a", "b"), model=lambda role: {"primary": "pm", "small": "sm"}[role],
                              structured_mode=lambda: "json_schema")
    for k, v in over.items():
        setattr(c, k, v)
    return c


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    for rel, text in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode())
    (root / "data").mkdir()
    (root / "data" / "dev_v2.json").write_text(json.dumps(PARTS))
    (root / "data" / "test_v2.json").write_text(json.dumps(PARTS + PARTS))
    (root / "configs").mkdir()
    (root / "configs" / "grid.json").write_text(json.dumps(GRID, indent=2))
    (root / "extra.json").write_text('{"threshold": 0.93}')
    return str(root)


def frozen(repo, cfg=None, **kw):
    kw.setdefault("grids", ["configs/grid.json"])
    return F.create(repo, cfg=cfg or fake_cfg(), now="2026-01-01T00:00:00+00:00", **kw)


def edit(repo, rel, text):
    os.makedirs(os.path.dirname(os.path.join(repo, *rel.split("/"))), exist_ok=True)
    with open(os.path.join(repo, *rel.split("/")), "wb") as f:
        f.write(text.encode())


def verdict(repo, cfg=None):
    return F.verify(repo, cfg=cfg or fake_cfg())


# ---------------------------------------------------------------- hashing and the pieces
def test_hashes_ignore_line_endings_and_json_layout(tmp_path):
    a, b, c = tmp_path / "a.py", tmp_path / "b.py", tmp_path / "c.py"
    a.write_bytes(b"x = 1\ny = 2\n")
    b.write_bytes(b"x = 1\r\ny = 2\r\n")
    c.write_bytes(b"x = 1\ny = 3\n")
    assert F.sha256_file(str(a)) == F.sha256_file(str(b)) != F.sha256_file(str(c))
    j1, j2, j3 = tmp_path / "1.json", tmp_path / "2.json", tmp_path / "3.json"
    j1.write_text('{"a": 1, "b": [1, 2]}')
    j2.write_text('{\n  "b": [1, 2],\n  "a": 1\n}\n')
    j3.write_text('{"a": 1, "b": [1, 3]}')
    assert F.sha256_json_file(str(j1)) == F.sha256_json_file(str(j2)) != F.sha256_json_file(str(j3))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert F.sha256_json_file(str(bad)) == F.sha256_file(str(bad))                  # an invalid JSON file is hashed as bytes
    assert F.canonical({"b": 1, "a": [1]}) == '{"a":[1],"b":1}'


@pytest.mark.parametrize("rel,kind", [("capp/planners.py", "behaviour"), ("capp/constrained.py", "behaviour"), ("capp/brand_new_module.py", "behaviour"),
                                      ("capp/tables.py", "analysis"), ("capp/casestudy.py", "analysis"), ("capp/summary.py", "analysis"),
                                      ("capp/freeze.py", "analysis"), ("week1/route_builder.py", "behaviour"), ("week2/llm_planner.py", "behaviour"),
                                      ("week6/app.py", "analysis"), ("week6/full_pipeline.py", "behaviour"), ("scripts/run_eval.py", "analysis")])
def test_kinds(rel, kind):
    assert F.kind_of(rel) == kind


def test_the_real_repository_classifies_every_analysis_module_that_exists():
    assert F.ANALYSIS_MODULES - set(F.scan_files(ROOT)) == set()                       # no typo in the list: every name is a real file
    for must_behave in ("capp/planners.py", "capp/harness.py", "capp/verify.py", "capp/repair.py", "capp/feasibility.py", "capp/constrained.py",
                        "capp/systems.py", "capp/config.py", "capp/llm_client.py", "week1/precedence_graph.py", "week1/route_builder.py",
                        "week4/nsga2.py"):
        assert F.scan_files(ROOT)[must_behave]["kind"] == "behaviour", must_behave


def test_scan_takes_the_top_level_code_and_json_of_the_scanned_folders_only(repo):
    files = F.scan_files(repo)
    assert set(files) == {"capp/__init__.py", "capp/planners.py", "capp/tables.py", "week1/route_builder.py", "week1/token_map.json",
                          "week6/app.py", "week6/full_pipeline.py", "scripts/run_eval.py"}
    assert "tests/test_x.py" not in files and "capp/subpkg/deep.py" not in files and "README.md" not in files
    assert files["capp/planners.py"]["kind"] == "behaviour" and files["capp/tables.py"]["kind"] == "analysis"


def test_config_snapshot_has_the_upper_case_constants_and_the_resolved_models():
    snap = F.config_snapshot(fake_cfg())
    assert snap["PLANNER_TEMPERATURE"] == 0.3 and snap["PLANNER_MAX_TOKENS"] == 200 and snap["VALID"] == ["a", "b"]
    assert snap["MODELS"] == {"primary": "pm", "small": "sm"} and snap["STRUCTURED_MODE"] == "json_schema"
    assert "ROOT" not in snap and "PRICE_USD_PER_MTOK" not in snap and "_private" not in snap and "lowercase" not in snap
    assert F.price_snapshot(fake_cfg()) == {"m": [1.0, 2.0]}


def test_the_real_config_snapshot_follows_the_environment(monkeypatch):
    base = F.config_snapshot()
    assert base["PLANNER_TEMPERATURE"] == config.PLANNER_TEMPERATURE and base["CG_MAX_TOKENS"] == config.CG_MAX_TOKENS
    assert base["MODELS"]["primary"] == config.model("primary")
    monkeypatch.setenv("CAPP_MODEL_PRIMARY", "someone/else")
    assert F.config_snapshot()["MODELS"]["primary"] == "someone/else"
    monkeypatch.setenv("CAPP_STRUCTURED", "none")
    assert F.config_snapshot()["STRUCTURED_MODE"] == "none"


def test_dataset_snapshot(repo):
    snap = F.dataset_snapshot(repo)
    assert set(snap) == {"dev_v2", "test_v2"}                                        # bench_v1 does not exist: not listed
    assert snap["dev_v2"]["n_parts"] == 2 and snap["test_v2"]["n_parts"] == 4 and snap["dev_v2"]["file"] == "data/dev_v2.json"
    edit(repo, "data/mine.json", json.dumps(PARTS * 3))
    extra = F.dataset_snapshot(repo, ["data/mine.json"])
    assert extra["mine"]["n_parts"] == 6 and extra["mine"]["file"] == "data/mine.json"
    with pytest.raises(F.FreezeError, match="not found"):
        F.dataset_snapshot(repo, ["data/nothing.json"])
    edit(repo, "data/dev_v2.json", "{broken")
    assert F.dataset_snapshot(repo)["dev_v2"]["n_parts"] is None                     # hashed anyway


def test_grid_and_extra_snapshots(repo):
    g = F.grid_snapshot(repo, ["configs/grid.json"])["configs/grid.json"]
    assert g["dataset"] == "dev_v2" and g["rows"] == [["A0", "b1"], ["A1", "b0"]]
    edit(repo, "configs/grid.json", json.dumps(GRID))                                 # same content, no indentation
    assert F.grid_snapshot(repo, ["configs/grid.json"])["configs/grid.json"]["sha256"] == g["sha256"]
    with pytest.raises(F.FreezeError, match="not found"):
        F.grid_snapshot(repo, ["configs/none.json"])
    edit(repo, "configs/grid.json", "{oops")
    with pytest.raises(F.FreezeError, match="not valid JSON"):
        F.grid_snapshot(repo, ["configs/grid.json"])
    assert F.grid_snapshot(repo, ["configs/grid.json"], strict=False)["configs/grid.json"]["rows"] == []
    assert list(F.extra_snapshot(repo, ["extra.json"])) == ["extra.json"]
    with pytest.raises(F.FreezeError, match="not found"):
        F.extra_snapshot(repo, ["nope.json"])


def test_paths_outside_the_repository_are_kept_absolute(repo, tmp_path):
    outside = tmp_path / "elsewhere.json"
    outside.write_text('{"x": 1}')
    snap = F.extra_snapshot(repo, [str(outside)])
    assert list(snap) == [str(outside).replace("\\", "/")]


def test_pytest_summary_parsing():
    assert F.parse_pytest_summary("....\n1464 passed in 128.74s (0:02:08)\n") == {"passed": 1464, "failed": 0, "summary": "1464 passed in 128.74s (0:02:08)"}
    r = F.parse_pytest_summary("== 2 failed, 1462 passed, 1 error in 130s ==")
    assert r["passed"] == 1462 and r["failed"] == 3
    assert F.parse_pytest_summary("") == {"passed": 0, "failed": 0, "summary": ""}


def test_run_tests_uses_pytest_in_the_repository(monkeypatch, repo):
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(cmd=cmd, cwd=kw["cwd"])
        return types.SimpleNamespace(stdout="10 passed in 1s\n", returncode=0)
    monkeypatch.setattr(F.subprocess, "run", fake_run)
    r = F.run_tests(repo)
    assert r["passed"] == 10 and r["failed"] == 0 and r["returncode"] == 0 and seen["cwd"] == repo and seen["cmd"][1:] == ["-m", "pytest", "-q"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_info(repo, tmp_path):
    assert F.git_info(str(tmp_path)) == {"commit": None, "dirty": None, "tags": []}   # not a repository
    sh = lambda *a: subprocess.run(["git", "-c", "user.email=a@b.c", "-c", "user.name=t", "-c", "commit.gpgsign=false", *a], cwd=repo, check=True,
                                   capture_output=True)
    sh("init", "-q")
    sh("add", "-A")
    sh("commit", "-q", "-m", "x")
    g = F.git_info(repo)
    assert len(g["commit"]) == 40 and g["dirty"] is False and g["tags"] == []
    sh("tag", "freeze-v1")
    edit(repo, "README.md", "changed\n")
    g = F.git_info(repo)
    assert g["dirty"] is True and g["tags"] == ["freeze-v1"]


# ---------------------------------------------------------------- creating
def test_create_writes_a_manifest_with_everything_in_it(repo):
    man = frozen(repo, extras=["extra.json"], note="tier 1", name="tier1", tests={"passed": 5, "failed": 0, "summary": "5 passed"})
    path = os.path.join(repo, F.MANIFEST_NAME)
    assert json.load(open(path, encoding="utf-8")) == man
    assert man["schema"] == 1 and man["name"] == "tier1" and man["note"] == "tier 1" and man["replaces"] is None
    assert man["created_utc"] == "2026-01-01T00:00:00+00:00" and len(man["id"]) == 12
    assert set(man["files"]) == set(F.scan_files(repo)) and set(man["datasets"]) == {"dev_v2", "test_v2"}
    assert list(man["grids"]) == ["configs/grid.json"] and list(man["extras"]) == ["extra.json"]
    assert man["config"]["MODELS"]["primary"] == "pm" and man["prices_not_compared"] == {"m": [1.0, 2.0]}
    assert man["tests"]["passed"] == 5
    raw = open(path, "rb").read()
    assert b"\r" not in raw and raw.endswith(b"\n")
    assert list(json.loads(raw)) == sorted(json.loads(raw))                           # sorted keys: a stable diff in git


def test_the_id_depends_on_the_state_not_on_the_time_or_the_note(repo):
    a = frozen(repo, note="one")
    b = F.create(repo, os.path.join(repo, "other.json"), grids=["configs/grid.json"], cfg=fake_cfg(), now="2027-05-05T00:00:00+00:00", note="two")
    assert a["id"] == b["id"]
    edit(repo, "capp/planners.py", "PROMPT = 'changed'\n")
    c = F.create(repo, os.path.join(repo, "third.json"), grids=["configs/grid.json"], cfg=fake_cfg())
    assert c["id"] != a["id"]
    d = F.create(repo, os.path.join(repo, "fourth.json"), grids=["configs/grid.json"], cfg=fake_cfg(PLANNER_TEMPERATURE=0.5))
    assert d["id"] != c["id"]


def test_a_manifest_is_created_once(repo):
    first = frozen(repo)
    with pytest.raises(F.FreezeError, match="exists"):
        frozen(repo)
    edit(repo, "capp/planners.py", "PROMPT = 'v2'\n")
    again = frozen(repo, force=True, note="prompt fixed after review")
    assert again["replaces"] == first["id"] and again["id"] != first["id"]
    open(os.path.join(repo, F.MANIFEST_NAME), "w").write("{broken")
    assert frozen(repo, force=True)["replaces"] == "unreadable"


def test_create_needs_a_grid_and_passing_tests(repo):
    with pytest.raises(F.FreezeError, match="no grid"):
        F.create(repo, cfg=fake_cfg())
    with pytest.raises(F.FreezeError, match="1 failure"):
        frozen(repo, tests={"passed": 9, "failed": 1, "summary": "1 failed, 9 passed"})
    assert not os.path.exists(os.path.join(repo, F.MANIFEST_NAME))                      # nothing half-written
    with pytest.raises(F.FreezeError, match="not found"):
        frozen(repo, grids=["configs/none.json"])


def test_load_rejects_what_is_not_a_manifest(repo, tmp_path):
    with pytest.raises(F.FreezeError, match="not found"):
        F.load(str(tmp_path / "nothing.json"))
    p = tmp_path / "m.json"
    p.write_text("{nope")
    with pytest.raises(F.FreezeError, match="not valid JSON"):
        F.load(str(p))
    p.write_text(json.dumps({"schema": 1, "id": "x"}))
    with pytest.raises(F.FreezeError, match="missing: files"):
        F.load(str(p))
    man = frozen(repo)
    man["schema"] = 99
    p.write_text(json.dumps(man))
    with pytest.raises(F.FreezeError, match="schema 1"):
        F.load(str(p))


# ---------------------------------------------------------------- what is drift and what is not
def test_an_untouched_tree_verifies(repo):
    man = frozen(repo, extras=["extra.json"])
    res = verdict(repo)
    assert res["status"] == "ok" and res["id"] == man["id"]
    assert all(res[k] == [] for k in ("behaviour", "analysis", "config", "datasets", "grids", "extras"))


@pytest.mark.parametrize("rel", ["capp/planners.py", "week1/route_builder.py", "week1/token_map.json", "week6/full_pipeline.py", "capp/__init__.py"])
def test_a_changed_behaviour_file_is_drift(repo, rel):
    frozen(repo)
    edit(repo, rel, "# edited\n")
    res = verdict(repo)
    assert res["status"] == "drift" and res["behaviour"] == [(rel, "changed")] and res["analysis"] == []


def test_a_new_or_a_removed_behaviour_file_is_drift(repo):
    frozen(repo)
    edit(repo, "capp/new_planner.py", "x = 1\n")
    res = verdict(repo)
    assert res["status"] == "drift" and res["behaviour"] == [("capp/new_planner.py", "added")]
    os.remove(os.path.join(repo, "capp", "new_planner.py"))
    os.remove(os.path.join(repo, "capp", "planners.py"))
    res = verdict(repo)
    assert res["status"] == "drift" and res["behaviour"] == [("capp/planners.py", "removed")]


@pytest.mark.parametrize("rel", ["capp/tables.py", "week6/app.py", "scripts/run_eval.py"])
def test_a_changed_analysis_file_is_allowed_but_reported(repo, rel):
    frozen(repo)
    edit(repo, rel, "# a table fix\n")
    res = verdict(repo)
    assert res["status"] == "analysis-changed" and res["analysis"] == [(rel, "changed")] and res["behaviour"] == []
    edit(repo, "scripts/another.py", "x = 1\n")
    assert verdict(repo)["status"] == "analysis-changed"                                 # a new script or analysis file is not drift either


def test_edits_outside_the_scanned_files_do_not_matter(repo):
    frozen(repo)
    edit(repo, "tests/test_x.py", "t = 2\n")
    edit(repo, "README.md", "other\n")
    edit(repo, "capp/subpkg/deep.py", "deep = 2\n")
    edit(repo, "capp/__pycache__/planners.cpython-312.pyc", "junk")
    assert verdict(repo)["status"] == "ok"


def test_a_windows_checkout_gives_the_same_hashes(repo):
    frozen(repo)
    for rel in list(FILES):
        p = os.path.join(repo, *rel.split("/"))
        data = open(p, "rb").read()
        open(p, "wb").write(data.replace(b"\n", b"\r\n"))
    assert verdict(repo)["status"] == "ok"


def test_a_changed_config_constant_or_model_is_drift(repo):
    frozen(repo)
    res = verdict(repo, fake_cfg(PLANNER_TEMPERATURE=0.7))
    assert res["status"] == "drift" and res["config"] == [("PLANNER_TEMPERATURE", 0.3, 0.7)]
    res = verdict(repo, fake_cfg(model=lambda role: {"primary": "other-model", "small": "sm"}[role]))
    assert res["status"] == "drift" and res["config"] == [("MODELS", {"primary": "pm", "small": "sm"}, {"primary": "other-model", "small": "sm"})]
    res = verdict(repo, fake_cfg(structured_mode=lambda: "none"))
    assert res["config"] == [("STRUCTURED_MODE", "json_schema", "none")]
    assert verdict(repo, fake_cfg(PRICE_USD_PER_MTOK={"m": (9.0, 9.0)}))["status"] == "ok"      # prices only turn tokens into a cost afterwards
    assert verdict(repo, fake_cfg(NEW_CONSTANT=1))["config"] == [("NEW_CONSTANT", None, 1)]


def test_each_llm_of_the_grid_may_be_the_primary_model(repo):
    """The final grid runs every LLM with CAPP_MODEL_PRIMARY set to its id: that must not read as drift (and run_eval must not stop on it)."""
    grid = dict(GRID, models=[{"id": "a", "model": "pm"}, {"id": "b", "model": "other/llm-2"}, {"id": "c", "model": "third/llm-3"}])
    edit(repo, "configs/grid.json", json.dumps(grid))
    man = frozen(repo)
    assert man["grids"]["configs/grid.json"]["models"] == ["pm", "other/llm-2", "third/llm-3"]
    other = lambda role: {"primary": "other/llm-2", "small": "sm"}[role]
    assert verdict(repo, fake_cfg(model=other))["status"] == "ok"
    assert F.status(repo, cfg=fake_cfg(model=other))["status"] == "ok"
    res = verdict(repo, fake_cfg(model=lambda role: {"primary": "not/in-the-grid", "small": "sm"}[role]))
    assert res["status"] == "drift" and res["config"][0][0] == "MODELS" and res["config"][0][2]["primary"] == "not/in-the-grid"
    res = verdict(repo, fake_cfg(model=lambda role: {"primary": "other/llm-2", "small": "different-small"}[role]))        # the cascade models are pinned
    assert res["status"] == "drift" and res["config"][0][2] == {"primary": "pm", "small": "different-small"}
    edit(repo, "configs/grid.json", json.dumps(dict(grid, models=grid["models"][:1])))                                  # the list of LLMs is part of the grid
    assert verdict(repo, fake_cfg(model=other))["grids"] == [("configs/grid.json", "changed")]


def test_a_manifest_without_the_model_list_still_works_and_is_strict(repo):
    man = frozen(repo)
    for g in man["grids"].values():
        g.pop("models")
    path = os.path.join(repo, F.MANIFEST_NAME)
    json.dump(man, open(path, "w"))
    assert verdict(repo)["status"] == "ok"
    assert verdict(repo, fake_cfg(model=lambda role: {"primary": "other/llm-2", "small": "sm"}[role]))["status"] == "drift"


def test_a_changed_dataset_is_drift_a_new_one_is_not(repo):
    frozen(repo)
    edit(repo, "data/test_v2.json", json.dumps(PARTS))
    res = verdict(repo)
    assert res["status"] == "drift" and res["datasets"] == [("test_v2", "changed")]
    edit(repo, "data/test_v2.json", json.dumps(PARTS + PARTS))
    edit(repo, "data/bench_v1.json", json.dumps(PARTS))                                    # the benchmark, annotated after the freeze
    res = verdict(repo)
    assert res["status"] == "ok" and res["datasets"] == [("bench_v1", "added")]
    os.remove(os.path.join(repo, "data", "test_v2.json"))
    res = verdict(repo)
    assert res["status"] == "drift" and ("test_v2", "removed") in res["datasets"]


def test_a_dataset_pinned_by_name_is_checked_by_name(repo):
    edit(repo, "data/mine.json", json.dumps(PARTS))
    frozen(repo, datasets=["data/mine.json"])
    assert verdict(repo)["status"] == "ok"
    edit(repo, "data/mine.json", json.dumps(PARTS * 2))
    assert verdict(repo)["datasets"] == [("mine", "changed")]
    os.remove(os.path.join(repo, "data", "mine.json"))
    assert verdict(repo)["datasets"] == [("mine", "removed")]


def test_the_grid_and_the_extras_are_pinned(repo):
    frozen(repo, extras=["extra.json"])
    edit(repo, "configs/grid.json", json.dumps(GRID))                                      # layout only
    edit(repo, "extra.json", '{ "threshold" : 0.93 }')
    assert verdict(repo)["status"] == "ok"
    g2 = copy.deepcopy(GRID)
    g2["rows"][1]["system"] = "b0_r1"
    edit(repo, "configs/grid.json", json.dumps(g2))
    res = verdict(repo)
    assert res["status"] == "drift" and res["grids"] == [("configs/grid.json", "changed")]
    edit(repo, "configs/grid.json", json.dumps(GRID))
    edit(repo, "extra.json", '{"threshold": 0.90}')
    res = verdict(repo)
    assert res["status"] == "drift" and res["extras"] == [("extra.json", "changed")]
    os.remove(os.path.join(repo, "extra.json"))
    assert verdict(repo)["extras"] == [("extra.json", "removed")]
    edit(repo, "configs/grid.json", "{broken")
    assert ("configs/grid.json", "changed") in verdict(repo)["grids"]
    os.remove(os.path.join(repo, "configs", "grid.json"))
    assert ("configs/grid.json", "removed") in verdict(repo)["grids"]


def test_behaviour_drift_wins_over_analysis_changes(repo):
    frozen(repo)
    edit(repo, "capp/tables.py", "# fix\n")
    edit(repo, "capp/planners.py", "# prompt\n")
    res = verdict(repo)
    assert res["status"] == "drift" and len(res["behaviour"]) == 1 and len(res["analysis"]) == 1


def test_verify_without_a_manifest_raises(repo):
    with pytest.raises(F.FreezeError, match="not found"):
        verdict(repo)


# ---------------------------------------------------------------- status (what goes into run.json)
def test_status_none_ok_analysis_drift_unreadable(repo):
    cfg = fake_cfg()
    assert F.status(repo, cfg=cfg) == {"status": "none", "id": None, "manifest": "FREEZE.json"}
    man = frozen(repo)
    st = F.status(repo, cfg=cfg)
    assert st["status"] == "ok" and st["id"] == man["id"] and st["n_differences"] == 0 and st["differences"] == []
    edit(repo, "capp/tables.py", "# fix\n")
    assert F.status(repo, cfg=cfg)["status"] == "analysis-changed"
    edit(repo, "capp/planners.py", "# x\n")
    st = F.status(repo, cfg=cfg)
    assert st["status"] == "drift" and st["n_differences"] == 1 and st["differences"] == ["changed capp/planners.py"]
    open(os.path.join(repo, F.MANIFEST_NAME), "w").write("[1, 2")
    st = F.status(repo, cfg=cfg)
    assert st["status"] == "unreadable" and "not valid JSON" in st["why"]


def test_status_lists_at_most_ten_differences_and_names_the_kind(repo):
    frozen(repo)
    for i in range(14):
        edit(repo, f"capp/new{i}.py", "x = 1\n")
    st = F.status(repo, cfg=fake_cfg(PLANNER_MAX_TOKENS=1))
    assert st["status"] == "drift" and st["n_differences"] == 15 and len(st["differences"]) == 10
    assert st["differences"][0] == "added capp/new0.py" or any(d.startswith("config ") for d in st["differences"] + [])
    full = F.verify(repo, cfg=fake_cfg(PLANNER_MAX_TOKENS=1))
    assert full["config"] == [("PLANNER_MAX_TOKENS", 200, 1)] and len(full["behaviour"]) == 14


def test_status_never_raises(repo, monkeypatch):
    frozen(repo)
    monkeypatch.setattr(F, "fingerprint", lambda *a, **k: 1 / 0)
    st = F.status(repo, cfg=fake_cfg())
    assert st["status"] == "unreadable" and "division" in st["why"]


def test_status_accepts_another_manifest_path(repo):
    other = os.path.join(repo, "FREEZE_tier1.json")
    F.create(repo, other, grids=["configs/grid.json"], cfg=fake_cfg())
    assert F.status(repo, cfg=fake_cfg())["status"] == "none"
    assert F.status(repo, other, cfg=fake_cfg())["status"] == "ok"


# ---------------------------------------------------------------- comparing two manifests, text
def test_compare_two_manifests(repo):
    first = frozen(repo)
    edit(repo, "capp/planners.py", "PROMPT = 'tier 2'\n")
    edit(repo, "capp/constrained.py", "x = 1\n")
    edit(repo, "capp/tables.py", "# fix\n")
    second = F.create(repo, os.path.join(repo, "second.json"), grids=["configs/grid.json"], cfg=fake_cfg(PLANNER_MAX_TOKENS=300))
    res = F.compare(first, second)
    assert res["status"] == "drift"
    assert sorted(res["behaviour"]) == [("capp/constrained.py", "added"), ("capp/planners.py", "changed")]
    assert res["analysis"] == [("capp/tables.py", "changed")] and res["config"] == [("PLANNER_MAX_TOKENS", 200, 300)]
    assert F.compare(first, first)["status"] == "ok"


def test_text_output(repo):
    man = frozen(repo, extras=["extra.json"], note="why", tests={"passed": 7, "failed": 0, "summary": "7 passed"})
    text = F.format_manifest(man)
    assert man["id"] in text and "dev_v2 (2 parts)" in text and "grid configs/grid.json: dataset dev_v2, rows A0=b1, A1=b0" in text
    assert "extra: extra.json" in text and "7 passed, 0 failed" in text and "note: why" in text and "pm" in text
    assert "not run at freeze time" in F.format_manifest(frozen(repo, force=True))
    edit(repo, "capp/planners.py", "# x\n")
    edit(repo, "capp/tables.py", "# y\n")
    out = F.format_comparison(verdict(repo))
    assert "DRIFT" in out and "changed  capp/planners.py" in out and "analysis files (allowed): 1" in out
    for i in range(50):
        edit(repo, f"capp/more{i}.py", "x\n")
    assert "... and 11 more" in F.format_comparison(verdict(repo))


# ---------------------------------------------------------------- the command line
@pytest.fixture()
def cli(repo, monkeypatch):
    spec = importlib.util.spec_from_file_location("freeze_cli", os.path.join(ROOT, "scripts", "freeze.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", repo)
    return mod


def test_cli_create_verify_show(cli, repo, capsys):
    assert cli.main(["create", "--grid", "configs/grid.json", "--note", "n"]) == 0
    out = capsys.readouterr()
    assert "Written:" in out.out and "git tag freeze-v1" in out.out and "tests were not run" in out.err
    assert os.path.isfile(os.path.join(repo, "FREEZE.json"))
    assert cli.main(["create", "--grid", "configs/grid.json"]) == 2                          # once
    assert "exists" in capsys.readouterr().err
    assert cli.main(["verify"]) == 0 and "OK" in capsys.readouterr().out
    assert cli.main(["show"]) == 0 and "Freeze " in capsys.readouterr().out
    edit(repo, "capp/tables.py", "# fix\n")
    assert cli.main(["verify"]) == 0 and "ANALYSIS-CHANGED" in capsys.readouterr().out
    assert cli.main(["verify", "--strict"]) == 1 and "--strict" in capsys.readouterr().err
    edit(repo, "capp/planners.py", "# prompt\n")
    assert cli.main(["verify"]) == 1
    out = capsys.readouterr()
    assert "DRIFT" in out.out and "capp/planners.py" in out.out and "not the frozen configuration" in out.err


def test_cli_without_a_manifest_and_bad_input(cli, repo, capsys):
    assert cli.main(["verify"]) == 2 and "not found" in capsys.readouterr().err
    assert cli.main(["show"]) == 2
    assert cli.main(["create", "--grid", "configs/none.json"]) == 2 and "not found" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["create"])                                                                   # --grid is required


def test_cli_run_tests_is_recorded_and_failures_refuse(cli, repo, capsys, monkeypatch):
    monkeypatch.setattr(F, "run_tests", lambda root: {"passed": 3, "failed": 0, "summary": "3 passed", "command": "x", "returncode": 0})
    assert cli.main(["create", "--grid", "configs/grid.json", "--run-tests"]) == 0
    assert json.load(open(os.path.join(repo, "FREEZE.json")))["tests"]["passed"] == 3
    os.remove(os.path.join(repo, "FREEZE.json"))
    monkeypatch.setattr(F, "run_tests", lambda root: {"passed": 2, "failed": 1, "summary": "1 failed, 2 passed", "command": "x", "returncode": 1})
    assert cli.main(["create", "--grid", "configs/grid.json", "--run-tests"]) == 2
    assert "failure" in capsys.readouterr().err and not os.path.exists(os.path.join(repo, "FREEZE.json"))


def test_cli_diff_and_other_manifest_path(cli, repo, capsys):
    a, b = os.path.join(repo, "a.json"), os.path.join(repo, "b.json")
    assert cli.main(["create", "--grid", "configs/grid.json", "--manifest", a, "--name", "tier1"]) == 0
    edit(repo, "capp/planners.py", "# tier 2\n")
    assert cli.main(["create", "--grid", "configs/grid.json", "--manifest", b]) == 0
    capsys.readouterr()
    assert cli.main(["diff", a, b]) == 0
    out = capsys.readouterr().out
    assert "->" in out and "capp/planners.py" in out
    assert cli.main(["verify", "--manifest", a]) == 1                                         # the tree has moved on since tier 1
    assert cli.main(["verify", "--manifest", b]) == 0
    assert cli.main(["diff", a, os.path.join(repo, "none.json")]) == 2


# ---------------------------------------------------------------- run.json, the gate in run_eval, Table 4
def test_every_run_records_its_freeze_status(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", Path(repo))
    parts = dataset.load_dataset("dev_v2")[:3]

    def run():
        return harness.run_experiment("rb", parts, tag="f", out_dir=str(tmp_path / "out"), dataset_name="dev_v2", progress=None, analysis=False)["run_info"]["freeze"]
    assert run() == {"status": "none", "id": None, "manifest": "FREEZE.json"}
    man = F.create(repo, grids=["configs/grid.json"])
    st = run()
    assert st["status"] == "ok" and st["id"] == man["id"]
    on_disk = json.load(open(tmp_path / "out" / "f_rb" / "run.json", encoding="utf-8"))
    assert on_disk["freeze"]["id"] == man["id"]
    edit(repo, "capp/planners.py", "# changed\n")
    assert run()["status"] == "drift" and run()["differences"][0] == "changed capp/planners.py"


def run_eval(tmp_path, *args, primary=None):
    env = dict(os.environ, GROQ_API_KEY="", PYTHONIOENCODING="utf-8", CAPP_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("CAPP_MODEL_PRIMARY", None)
    if primary:
        env["CAPP_MODEL_PRIMARY"] = primary
    env.pop("CAPP_STRUCTURED", None)
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_eval.py"), *args, "--out", str(tmp_path / "out")],
                          capture_output=True, text=True, cwd=ROOT, env=env, timeout=180)


@pytest.fixture()
def real_manifest(tmp_path):
    saved = {k: os.environ.pop(k, None) for k in ("CAPP_MODEL_PRIMARY", "CAPP_STRUCTURED")}
    try:
        path = str(tmp_path / "FREEZE_test.json")
        F.create(ROOT, path, grids=["configs/ablation_template.json"])
        return path
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_run_eval_stops_on_a_sealed_dataset_without_a_freeze(tmp_path):
    res = run_eval(tmp_path, "--system", "b1", "--dataset", "test_v2", "--limit", "1", "--freeze", str(tmp_path / "missing.json"))
    assert res.returncode == 2
    assert "STOPPED: sealed dataset test_v2 and no freeze manifest" in res.stderr and "--no-freeze-check" in res.stderr
    assert not (tmp_path / "out").exists()                                                        # nothing was run


def test_run_eval_stops_when_the_tree_has_drifted(tmp_path, real_manifest):
    man = json.load(open(real_manifest, encoding="utf-8"))
    man["files"]["capp/planners.py"]["sha256"] = "0" * 64
    json.dump(man, open(real_manifest, "w", encoding="utf-8"))
    res = run_eval(tmp_path, "--system", "b0", "--dataset", "test_v2", "--limit", "1", "--freeze", real_manifest)
    assert res.returncode == 2 and "differs from the freeze" in res.stderr and "changed capp/planners.py" in res.stderr


def test_run_eval_stops_on_an_unreadable_manifest(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    res = run_eval(tmp_path, "--system", "b0", "--dataset", "test_v2", "--limit", "1", "--freeze", str(bad))
    assert res.returncode == 2 and "cannot be used" in res.stderr


def test_run_eval_goes_on_with_a_matching_freeze(tmp_path, real_manifest):
    res = run_eval(tmp_path, "--system", "b1", "--dataset", "test_v2", "--limit", "1", "--freeze", real_manifest)
    man = json.load(open(real_manifest, encoding="utf-8"))
    assert f"Freeze {man['id']}: ok" in res.stderr or f"Freeze {man['id']}: analysis-changed" in res.stderr
    assert "STOPPED: sealed dataset" not in res.stderr                                             # it got as far as the API key


def test_run_eval_accepts_every_llm_of_the_grid_and_stops_on_another(tmp_path, real_manifest):
    man = json.load(open(real_manifest, encoding="utf-8"))
    for model in man["grids"]["configs/ablation_template.json"]["models"]:
        res = run_eval(tmp_path, "--system", "b1", "--dataset", "test_v2", "--limit", "1", "--freeze", real_manifest, primary=model)
        assert f"Freeze {man['id']}: " in res.stderr and "STOPPED: sealed dataset" not in res.stderr, model
    res = run_eval(tmp_path, "--system", "b1", "--dataset", "test_v2", "--limit", "1", "--freeze", real_manifest, primary="not/in-the-grid")
    assert res.returncode == 2 and "differs from the freeze" in res.stderr and "config MODELS" in res.stderr


def test_run_eval_override_is_loud_and_recorded(tmp_path):
    res = run_eval(tmp_path, "--system", "b1", "--dataset", "test_v2", "--limit", "1", "--freeze", str(tmp_path / "missing.json"), "--no-freeze-check")
    assert "continuing because of --no-freeze-check" in res.stderr and "STOPPED: sealed dataset" not in res.stderr


def test_run_json_records_the_manifest_the_gate_used_not_the_one_of_the_repository(tmp_path, real_manifest):
    """The status in run.json is the status against --freeze FILE (default FREEZE.json): a FREEZE.json lying in the repository must not
    change what a run says about the manifest it was checked against."""
    man = json.load(open(real_manifest, encoding="utf-8"))
    run_eval(tmp_path, "--system", "rb", "--dataset", "test_v2", "--limit", "1", "--freeze", real_manifest)
    run = json.load(open(next((tmp_path / "out").glob("*/run.json")), encoding="utf-8"))
    assert run["freeze"]["id"] == man["id"] and run["freeze"]["status"] in ("ok", "analysis-changed")
    other = tmp_path / "other"
    other.mkdir()
    run_eval(other, "--system", "rb", "--dataset", "test_v2", "--limit", "1", "--freeze", str(other / "missing.json"))
    run = json.load(open(next((other / "out").glob("*/run.json")), encoding="utf-8"))
    assert run["freeze"]["status"] == "none" and run["freeze"]["id"] is None


def test_the_route_builder_and_the_development_set_are_not_gated(tmp_path):
    res = run_eval(tmp_path, "--system", "rb", "--dataset", "test_v2", "--limit", "2", "--freeze", str(tmp_path / "missing.json"))
    assert res.returncode == 0 and "STOPPED" not in res.stderr
    run = json.load(open(next((tmp_path / "out").glob("*/run.json")), encoding="utf-8"))
    assert run["freeze"]["status"] == "none"
    res = run_eval(tmp_path, "--system", "rb", "--dataset", "dev_v2", "--limit", "2", "--freeze", str(tmp_path / "missing.json"), "--tag", "dev")
    assert res.returncode == 0 and "sealed" not in res.stderr


# ---- Table 4
SEALED = {"name": "t", "dataset": "test_v2", "limit": 2, "repeats": 1, "resamples": 100, "seed": 1,
          "models": [{"id": "a", "label": "A", "model": "qwen/qwen3.8-27b", "tag": "ta"}, {"id": "b", "label": "B", "model": "openai/gpt-oss-120b", "tag": "tb"},
                     {"id": "c", "label": "C", "model": "openai/gpt-oss-20b", "tag": "tc"}],
          "rows": [{"id": "A0", "label": "B1", "system": "b1"}, {"id": "A1", "label": "B0", "system": "b0", "parent": "A0"}]}


def rb_planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
    route = rb.generate_valid_routes(features, max_routes=15)[0]
    return {"success": True, "steps": list(route["steps"]), "raw_response": "", "model": "fake",
            "meta": {"cached": True, "finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.0}}


@pytest.fixture()
def sealed_runs(tmp_path):
    out = str(tmp_path / "results")
    parts = dataset.load_dataset("test_v2")[:2]
    for m in SEALED["models"]:
        harness.run_experiment("b1", parts, tag=m["tag"], out_dir=out, dataset_name="test_v2", progress=None, planner=rb_planner, analysis=False)
    cfg = copy.deepcopy(SEALED)
    cfg["results_dir"] = out
    return out, T.validate_config(cfg)


def set_freeze(out, tag, status, fid="aaaaaaaaaaaa", drop=False, differences=None):
    path = os.path.join(out, f"{tag}_b1", "run.json")
    info = json.load(open(path, encoding="utf-8"))
    if drop:
        info.pop("freeze", None)
    else:
        info["freeze"] = {"status": status, "id": fid, "manifest": "FREEZE.json", **({"differences": differences} if differences else {})}
    json.dump(info, open(path, "w", encoding="utf-8"))


def freeze_warnings(cfg, out, row="A0", model="a"):
    r = next(x for x in cfg["rows"] if x["id"] == row)
    m = next(x for x in cfg["models"] if x["id"] == model)
    return [w for w in T.load_run(cfg, r, m, out, T.expected_parts(cfg))["warnings"] if "freeze" in w]


def test_table_flags_sealed_runs_that_were_not_frozen(sealed_runs):
    out, cfg = sealed_runs
    set_freeze(out, "ta", "ok")
    assert freeze_warnings(cfg, out) == []
    set_freeze(out, "ta", "analysis-changed")
    assert freeze_warnings(cfg, out) == []
    set_freeze(out, "ta", "none")
    w = freeze_warnings(cfg, out)
    assert len(w) == 1 and "freeze status 'none'" in w[0] and "not a run of a frozen configuration" in w[0]
    set_freeze(out, "ta", "drift", differences=["changed capp/planners.py"])
    w = freeze_warnings(cfg, out)
    assert "'drift'" in w[0] and "changed capp/planners.py" in w[0]
    set_freeze(out, "ta", None, drop=True)
    assert "no freeze status" in freeze_warnings(cfg, out)[0]


def test_table_does_not_ask_for_a_freeze_of_dev_runs(sealed_runs):
    out, cfg = sealed_runs
    dev = copy.deepcopy(cfg)
    dev["dataset"] = "dev_v2"
    set_freeze(out, "ta", "none")
    r = next(x for x in dev["rows"] if x["id"] == "A0")
    run = T.load_run(dev, r, dev["models"][0], out, T.expected_parts(dev))
    assert not [w for w in run["warnings"] if "freeze" in w]


def test_table_lists_runs_made_under_different_freezes(sealed_runs):
    out, cfg = sealed_runs
    for tag, fid in (("ta", "111111111111"), ("tb", "111111111111"), ("tc", "222222222222")):
        set_freeze(out, tag, "ok", fid)
    msgs = [i["message"] for i in T.analyse(cfg)["issues"]]
    assert any("2 different freeze manifests (111111111111, 222222222222)" in m for m in msgs)
    set_freeze(out, "tc", "ok", "111111111111")
    assert not any("different freeze manifests" in i["message"] for i in T.analyse(cfg)["issues"])


def test_the_table_manifest_records_the_freeze_of_each_run(sealed_runs):
    out, cfg = sealed_runs
    for tag in ("ta", "tb", "tc"):
        set_freeze(out, tag, "ok", "abcabcabcabc")
    res = T.analyse(cfg)
    runs = T.manifest(res, {})["runs"]
    a0 = [r for r in runs if r["row"] == "A0"]
    assert {r["freeze"] for r in a0} == {"ok"} and {r["freeze_id"] for r in a0} == {"abcabcabcabc"}


def test_commands_for_a_sealed_dataset_tell_how_to_freeze(sealed_runs, tmp_path):
    out, cfg = sealed_runs
    lines = T.run_commands(cfg, T.load_all(cfg, out))
    text = "\n".join(lines)
    assert "scripts/freeze.py create --grid" in text and "stops on a sealed dataset if FREEZE.json is missing" in text
