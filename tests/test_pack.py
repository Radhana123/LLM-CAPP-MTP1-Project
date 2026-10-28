"""Step 29 (WP12): a release folder in one zip (capp/pack.py, scripts/pack.py).

Offline. The releases are built by the real release code from real records in a fake repository (the fixtures of the release and slides
tests). What the tests pin down: the metadata of the upload is checked strictly and completely; nothing is written while the release is not
verified, the metadata is incomplete or a file holds a secret (and no message ever shows the secret); the archive holds exactly the listed
files with their hashes; the same inputs give the same zip, also when the release was checked out with CRLF line endings; the run folders go in
byte for byte; and `verify` finds every kind of change to an archive without the repository.
"""
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capp import pack as K, release as R      # noqa: E402
from test_release import repo, run_folder, world, write_prose, recipe as write_recipe      # noqa: E402,F401  (world and repo are fixtures)
from test_slides import release      # noqa: E402,F401  (a fixture: (repo, release folder) with benchmark, case studies and an extraction run)

META = {"creators": [{"name": "Author, Test", "affiliation": "A University"}], "license": "cc-by-4.0"}
FAKE_GROQ = "gsk_" + "A1b2C3d4E5f6G7h8I9j0K1l2"            # assembled here so that this file holds no key-looking text
FAKE_SK = "sk-" + "proj" + "Z9y8X7w6V5u4T3s2R1q0P9o8N7m6L5k4"


def meta_file(where, **over):
    p = Path(where) / "zenodo.json"
    d = dict(META)
    d.update(over)
    p.write_text(json.dumps(d))
    return str(p)


def plan(rel, meta=None, **kw):
    repo_, out = rel
    return K.plan(str(out), root=str(repo_), metadata_path=meta or meta_file(repo_), regenerate=kw.pop("regenerate", False), **kw)


def build(rel, out_dir, **kw):
    repo_, out = rel
    return K.build(str(out), str(out_dir), root=str(repo_), metadata_path=kw.pop("meta", None) or meta_file(repo_), regenerate=kw.pop("regenerate", False), **kw)


def members(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        return {i.filename: zf.read(i) for i in zf.infolist()}


def rewrite(zip_path, changes, out=None, keep_sha256=False):
    """The zip with some members replaced (bytes), removed (None) or added. The .sha256 next to it is removed, because it would not fit any more
    (so that a test sees the one change it makes and not that one too)."""
    if not keep_sha256 and os.path.exists(zip_path + ".sha256"):
        os.remove(zip_path + ".sha256")
    data = members(zip_path)
    for k, v in changes.items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v
    out = out or zip_path
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in data.items():
            zf.writestr(k, v)
    return out


def tree(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in Path(folder).rglob("*") if p.is_file()}


# ---------------------------------------------------------------- the metadata
def test_minimal_metadata_gets_the_defaults():
    meta, problems, advice = K.normalise_metadata(META, "v1")
    assert problems == [] and advice == []
    assert meta == {"title": "LLM-CAPP MTP-1: release v1", "upload_type": "dataset", "description": "", "creators": [{"name": "Author, Test", "affiliation": "A University"}],
                    "license": "cc-by-4.0", "access_right": "open", "version": "v1"}


def test_everything_that_is_given_is_kept_and_cleaned():
    meta, problems, _ = K.normalise_metadata(dict(META, _comment="x", title=" My title ", description="D", upload_type="software", access_right="embargoed", version="2",
                                                 keywords=["a", " b ", "a"], related_identifiers=[{"identifier": "https://x.org", "relation": "isSupplementTo", "scheme": "url"}],
                                                 notes="N", creators=[{"name": "A, B", "orcid": "0000-0002-1825-0097"}]), "v1")
    assert problems == []
    assert meta["title"] == "My title" and meta["upload_type"] == "software" and meta["access_right"] == "embargoed" and meta["version"] == "2"
    assert meta["keywords"] == ["a", "b"] and meta["notes"] == "N" and meta["creators"] == [{"name": "A, B", "orcid": "0000-0002-1825-0097"}]
    assert meta["related_identifiers"] == [{"identifier": "https://x.org", "relation": "isSupplementTo", "scheme": "url"}] and "_comment" not in meta


@pytest.mark.parametrize("change,fragment", [
    ({"license": ""}, "license: required"), ({"license": None}, "license: required"), ({"license": 3}, "license: must be text"),
    ({"creators": []}, "creators: required"), ({"creators": None}, "creators: required"), ({"creators": "A"}, "creators: required"),
    ({"creators": [{"name": ""}]}, "creators[1].name: empty"), ({"creators": [{"affiliation": "x"}]}, "creators[1].name: empty"),
    ({"creators": ["A"]}, "creators[1]: must be an object"), ({"creators": [{"name": "A, B", "orcid": "123"}]}, "orcid"),
    ({"creators": [{"name": "A, B", "x": 1}]}, "creators[1]: unknown key(s) ['x']"), ({"creators": [{"name": 3}]}, "creators[1].name: must be text"),
    ({"upload_type": "paper"}, "upload_type"), ({"access_right": "public"}, "access_right"), ({"title": 3}, "title: must be text"),
    ({"keywords": "a"}, "keywords"), ({"keywords": [""]}, "keywords"), ({"keywords": [1]}, "keywords"),
    ({"related_identifiers": "x"}, "related_identifiers: must be a list"), ({"related_identifiers": [{"identifier": "x"}]}, "related_identifiers[1]: identifier and relation"),
    ({"related_identifiers": [{"identifier": "x", "relation": "y", "z": 1}]}, "related_identifiers[1]: unknown key(s) ['z']"), ({"related_identifiers": ["x"]}, "related_identifiers[1]: must be an object"),
    ({"licence": "cc-by-4.0"}, "unknown key(s) ['licence']"),
])
def test_wrong_metadata_is_refused_with_the_reason(change, fragment):
    _, problems, _ = K.normalise_metadata(dict(META, **change), "v1")
    assert any(fragment in p for p in problems), problems


def test_all_the_problems_come_together():
    _, problems, _ = K.normalise_metadata({"creators": [], "upload_type": "x", "colour": 1}, "v1")
    assert len(problems) == 4 and any("colour" in p for p in problems) and any("license" in p for p in problems)


def test_advice_does_not_stop_a_pack():
    _, problems, advice = K.normalise_metadata({"creators": [{"name": "Radhana"}], "license": "Creative Commons"}, "v1")
    assert problems == [] and len(advice) == 2 and any("Family, Given" in a for a in advice) and any("space" in a for a in advice)


def test_a_metadata_that_is_not_an_object_is_a_problem():
    assert K.normalise_metadata([1], "v1")[1] == ["metadata: must be a JSON object"]


def test_the_shipped_template_asks_for_the_creators_and_the_licence(tmp_path):
    meta, problems, _ = K.load_metadata(os.path.join(ROOT, "configs", "zenodo_template.json"), "v1")
    assert any("creators[1].name: empty" in p for p in problems) and any("license: required" in p for p in problems) and len(problems) == 2
    filled = json.load(open(os.path.join(ROOT, "configs", "zenodo_template.json")))
    filled["creators"][0]["name"], filled["license"] = "Author, Test", "cc-by-4.0"
    p = tmp_path / "z.json"
    p.write_text(json.dumps(filled))
    assert K.load_metadata(str(p), "v1")[1] == []


def test_a_missing_or_broken_metadata_file_is_a_problem_not_a_crash(tmp_path):
    assert "not found" in K.load_metadata(str(tmp_path / "none.json"), "v1")[1][0]
    (tmp_path / "bad.json").write_text("{")
    assert "not valid JSON" in K.load_metadata(str(tmp_path / "bad.json"), "v1")[1][0]


# ---------------------------------------------------------------- secrets
@pytest.mark.parametrize("text,what", [
    (FAKE_GROQ, "Groq API key"), (FAKE_SK, "API key (sk-...)"), ("ghp_" + "a" * 36, "GitHub token"), ("github_pat_" + "a" * 40, "GitHub token"),
    ("AKIA" + "A" * 16, "AWS access key id"), ("AIza" + "a" * 35, "Google API key"), ("hf_" + "a" * 34, "Hugging Face token"),
    ("xoxb-" + "1" * 12, "Slack token"), ("-----BEGIN RSA PRIVATE KEY-----", "private key"),
    ('"api_key": "' + "abcdef0123456789abcd" + '"', "key, secret or password assignment"), ("PASSWORD=" + "x" * 24, "key, secret or password assignment"),
])
def test_known_kinds_of_secrets_are_found(text, what):
    found = K.scan_secrets(f"first line\nsecond {text} end\n".encode())
    assert found == [(2, what)]


@pytest.mark.parametrize("text", [
    "task-oriented-process-planning-routes-of-the-parts", "sk-short", "the api_key is read from .env", "GROQ_API_KEY is not set", "hf_ is a prefix",
    '"prompt_tokens": 1500', "tokens: 12345", "secret: set it", "ask-the-expert-about-the-routing-of-the-part-please",
])
def test_ordinary_text_is_not_taken_for_a_secret(text):
    assert K.scan_secrets(text.encode()) == []


def test_the_length_of_an_sk_key_is_a_boundary():
    assert K.scan_secrets((" sk-" + "a" * 31).encode()) == []
    assert K.scan_secrets((" sk-" + "a" * 32).encode()) == [(1, "API key (sk-...)")]


def test_the_secret_itself_is_never_in_what_the_scan_returns():
    found = K.scan_secrets(f"x\n{FAKE_GROQ}\n".encode(), [FAKE_GROQ.encode()])
    assert FAKE_GROQ not in repr(found) and {w for _, w in found} == {"Groq API key", "a value of your .env or environment"}


def test_literal_values_are_found_on_every_line():
    found = K.scan_secrets(b"a\nvalue-of-my-key-1234\nb value-of-my-key-1234 c\n", [b"value-of-my-key-1234"])
    assert [ln for ln, _ in found] == [2, 3]


def test_the_secrets_of_the_machine_come_from_the_environment_and_the_env_file(tmp_path, monkeypatch):
    for k in list(os.environ):
        if K.ENV_NAME_RE.search(k):
            monkeypatch.delenv(k)
    (tmp_path / ".env").write_text('# a comment\nGROQ_API_KEY="' + "k" * 20 + '"\nexport OTHER_TOKEN=' + "t" * 18 + "\nSHORT_KEY=abc\nPATH_X=" + "p" * 30 + "\nnoequals\n")
    monkeypatch.setenv("MY_SECRET", "s" * 17)
    monkeypatch.setenv("SHORT_PASSWORD", "tiny")
    assert K._env_values(str(tmp_path)) == sorted(x.encode() for x in ("k" * 20, "t" * 18, "s" * 17))
    assert K._env_values(str(tmp_path / "none")) == [b"s" * 17]


# ---------------------------------------------------------------- the plan
def test_the_plan_lists_the_release_and_what_stands_in_the_way(release):
    p = plan(release)
    repo_, out = release
    assert p["name"] == "t" and p["dev"] is True and p["blocking"] == []
    arcs = [e["arc"] for e in p["entries"]]
    assert arcs == sorted(arcs) and "release/release_manifest.json" in arcs and "release/RELEASE.md" in arcs and "release/tables/manifest.json" in arcs
    assert all(a.startswith("release/") for a in arcs) and len(arcs) == len(p["manifest"]["files"]) + 1
    assert p["groups"]["release"]["files"] == len(arcs) and p["groups"]["runs"] is None and p["groups"]["cache"] is None
    adv = " | ".join(p["advisory"])
    assert "development release" in adv and "run folders are not in the archive" in adv and "expert benchmark" in adv and "freeze-v1" in adv
    assert "READY TO PACK" in K.format_plan(p)


def test_a_missing_metadata_file_blocks(release):
    repo_, out = release
    p = K.plan(str(out), root=str(repo_), metadata_path=str(repo_ / "none.json"), regenerate=False)
    assert len(p["blocking"]) == 1 and "not found" in p["blocking"][0] and "NOT READY" in K.format_plan(p)
    assert K.plan(str(out), root=str(repo_), metadata_path=None, regenerate=False)["blocking"] == ["no metadata file given"]


def test_the_metadata_problems_block_all_at_once(release):
    repo_, out = release
    p = plan(release, meta_file(repo_, license="", creators=[]))
    assert len(p["blocking"]) == 2


def test_a_folder_that_is_not_a_release_is_refused(tmp_path, release):
    repo_, out = release
    with pytest.raises(R.ReleaseError, match="not a release folder"):
        K.plan(str(tmp_path), root=str(repo_), metadata_path=meta_file(repo_))


def test_a_release_whose_file_changed_blocks_and_writes_nothing(release, tmp_path):
    repo_, out = release
    with open(out / "paper" / "readiness.txt", "a") as f:
        f.write("changed\n")
    res = build(release, tmp_path / "dist")
    assert res["zip"] is None and any("not verified" in b and "modified: paper/readiness.txt" in b for b in res["blocking"])
    assert not (tmp_path / "dist").exists() or os.listdir(tmp_path / "dist") == []


def test_a_file_added_to_the_release_blocks(release):
    repo_, out = release
    (out / "paper" / "extra.txt").write_text("x")
    assert any("not in the manifest: paper/extra.txt" in b for b in plan(release)["blocking"])


def test_the_rebuild_catches_a_source_that_changed_and_no_regenerate_does_not(release):
    repo_, out = release
    write_prose(repo_, {"problem": "A sentence that was not there when the release was built."})
    assert plan(release)["blocking"] == []                                                       # only the hashes of the folder
    p = plan(release, regenerate=True)
    assert any("not verified" in b and "inputs/paper_text.json" in b for b in p["blocking"])
    skip = plan(release)
    assert any("rebuild of the release was skipped" in a for a in skip["advisory"])


def test_a_rebuild_that_cannot_run_is_a_blocking_item_with_the_way_out(release):
    repo_, out = release
    shutil.rmtree(repo_ / "fe1_extraction")
    p = plan(release, regenerate=True)
    assert len(p["blocking"]) == 1 and "cannot be verified here" in p["blocking"][0] and "--no-regenerate" in p["blocking"][0]


# ---------------------------------------------------------------- the archive
def test_the_archive_holds_the_release_and_the_files_made_here(release, tmp_path):
    repo_, out = release
    res = build(release, tmp_path / "dist")
    z = res["zip"]
    assert z == str(tmp_path / "dist" / "t.zip") and os.path.isfile(z) and res["blocking"] == []
    m = members(z)
    assert {n.split("/")[0] for n in m} == {"t"}
    top = sorted(n[2:] for n in m if n.count("/") == 1)
    assert top == ["PACK.md", "PACK_manifest.json", "SHA256SUMS.txt", "zenodo_metadata.json"]
    rel_files = {n[len("t/release/"):] for n in m if n.startswith("t/release/")}
    assert rel_files == set(tree(out)) and res["members"] == len(m)
    assert not any(n.startswith(("t/runs/", "t/cache/")) for n in m)
    side = Path(z + ".sha256").read_text()
    import hashlib
    assert side == f"{hashlib.sha256(Path(z).read_bytes()).hexdigest()}  t.zip\n" == res["zip_sha256"] + "  t.zip\n"


def test_the_release_files_in_the_archive_are_the_files_of_the_release(release, tmp_path):
    repo_, out = release
    m = members(build(release, tmp_path / "d")["zip"])
    for rel, data in tree(out).items():
        want = data.replace(b"\r\n", b"\n") if rel.lower().endswith(R.TEXT_SUFFIXES) else data
        assert m["t/release/" + rel] == want, rel


def test_the_sums_file_and_the_manifest_list_every_member(release, tmp_path):
    import hashlib
    m = members(build(release, tmp_path / "d")["zip"])
    pm = json.loads(m["t/PACK_manifest.json"])
    assert pm["format"] == K.FORMAT and pm["name"] == "t" and pm["sealed"] is False and pm["dataset"] == "dev_v2"
    assert set(pm["files"]) == {n[2:] for n in m} - {"PACK_manifest.json", "SHA256SUMS.txt"}
    assert all(hashlib.sha256(m["t/" + f]).hexdigest() == h for f, h in pm["files"].items())
    assert pm["release_manifest_sha256"] == hashlib.sha256(m["t/release/release_manifest.json"]).hexdigest()
    sums = dict(reversed(line.split("  ", 1)) for line in m["t/SHA256SUMS.txt"].decode().splitlines())
    assert set(sums) == {n[2:] for n in m} - {"SHA256SUMS.txt"} and sums["PACK_manifest.json"] == hashlib.sha256(m["t/PACK_manifest.json"]).hexdigest()
    assert m["t/SHA256SUMS.txt"].decode().splitlines() == sorted(m["t/SHA256SUMS.txt"].decode().splitlines(), key=lambda x: x.split("  ", 1)[1])
    assert pm["metadata"] == json.loads(m["t/zenodo_metadata.json"]) and pm["contents"]["runs"] is None


def test_the_metadata_file_is_the_zenodo_object_with_a_default_description(release, tmp_path):
    meta = json.loads(members(build(release, tmp_path / "d")["zip"])["t/zenodo_metadata.json"])
    assert meta["title"] == "LLM-CAPP MTP-1: release t" and meta["version"] == "t" and meta["license"] == "cc-by-4.0" and meta["upload_type"] == "dataset"
    assert meta["description"].startswith("DEVELOPMENT RESULTS, NOT PAPER RESULTS. Release t of the MTP-1 project") and "_comment" not in meta
    own = json.loads(members(build(release, tmp_path / "e", meta=meta_file(release[0], description="Mine."))["zip"])["t/zenodo_metadata.json"])
    assert own["description"] == "Mine."


def test_pack_md_says_what_it_is_made_from_and_how_to_check_it(release, tmp_path):
    md = members(build(release, tmp_path / "d")["zip"])["t/PACK.md"].decode()
    assert md.startswith("# Archive: t") and "DEVELOPMENT RESULTS, NOT PAPER RESULTS" in md and "Licence: cc-by-4.0." in md
    assert "python scripts/pack.py verify t.zip" in md and "sha256sum -c SHA256SUMS.txt" in md and "t.zip.sha256" in md and "Get-FileHash t.zip" in md
    assert "no run folders" in md and "`runs/`" not in md and "`cache/llm/`" not in md
    assert "The code is not in this archive" in md and "What this does not show" in md


def test_the_same_inputs_give_the_same_zip_file(release, tmp_path):
    a = build(release, tmp_path / "a")
    b = build(release, tmp_path / "b")
    assert Path(a["zip"]).read_bytes() == Path(b["zip"]).read_bytes() and a["zip_sha256"] == b["zip_sha256"]
    assert Path(a["zip"] + ".sha256").read_text() == Path(b["zip"] + ".sha256").read_text()
    with zipfile.ZipFile(a["zip"]) as zf:
        assert {i.date_time for i in zf.infolist()} == {K.ZIP_TIME} and {i.create_system for i in zf.infolist()} == {3}
        names = [i.filename for i in zf.infolist()]
        assert names == sorted(names)


def test_a_release_checked_out_with_crlf_gives_the_same_archive(release, tmp_path):
    repo_, out = release
    before = Path(build(release, tmp_path / "lf")["zip"]).read_bytes()
    for p in Path(out).rglob("*"):
        if p.is_file() and p.suffix in R.TEXT_SUFFIXES:
            p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    assert R.verify(str(out), str(repo_), regenerate=False)["ok"]                                  # the release itself says it is unchanged
    res = build(release, tmp_path / "crlf")
    assert Path(res["zip"]).read_bytes() == before
    assert all(b"\r\n" not in v for k, v in members(res["zip"]).items() if k.lower().endswith(R.TEXT_SUFFIXES))


def test_packing_changes_nothing_in_the_repository(release, tmp_path):
    repo_, out = release
    meta = meta_file(repo_)
    before = tree(repo_)
    build(release, tmp_path / "d", with_runs=True, meta=meta)
    assert tree(repo_) == before


def test_nothing_is_left_behind_when_a_build_is_refused(release, tmp_path):
    repo_, out = release
    res = build(release, tmp_path / "d", meta=meta_file(repo_, license=""))
    assert res["zip"] is None and res["members"] == 0 and not (tmp_path / "d").exists()


@pytest.mark.skipif(os.name == "nt", reason="file modes are not used on Windows")
def test_the_zip_is_readable_like_any_other_file(release, tmp_path):
    z = build(release, tmp_path / "d")["zip"]
    mask = os.umask(0)
    os.umask(mask)
    assert (os.stat(z).st_mode & 0o777) == (0o666 & ~mask)


def test_a_write_that_fails_leaves_no_temporary_file_and_no_zip(release, tmp_path, monkeypatch):
    real, calls = K._read, {"n": 0}
    n_entries = len(plan(release)["entries"])

    def failing(entry):
        calls["n"] += 1
        if calls["n"] > n_entries + 3:                                                           # fails while the zip is being written
            raise OSError("disk full")
        return real(entry)
    monkeypatch.setattr(K, "_read", failing)
    with pytest.raises(OSError, match="disk full"):
        build(release, tmp_path / "d")
    assert os.listdir(tmp_path / "d") == []


def test_a_new_pack_replaces_the_old_zip_and_leaves_no_temp_file(release, tmp_path):
    first = build(release, tmp_path / "d")
    second = build(release, tmp_path / "d", meta=meta_file(release[0], title="Another title"))
    assert first["zip"] == second["zip"] and first["zip_sha256"] != second["zip_sha256"]
    assert sorted(os.listdir(tmp_path / "d")) == ["t.zip", "t.zip.sha256"]
    assert json.loads(members(second["zip"])["t/zenodo_metadata.json"])["title"] == "Another title"


# ---------------------------------------------------------------- runs and cache
def run_dirs(repo_):
    man = json.load(open(repo_ / "release" / "t" / "tables" / "manifest.json"))
    return sorted({r["folder"] for r in man["runs"] if r["folder"]})


def test_the_run_folders_go_in_byte_for_byte_at_their_path_under_the_repository(release, tmp_path):
    repo_, out = release
    rec = repo_ / "results" / run_dirs(repo_)[0] / "records.jsonl"
    assert rec.read_bytes().count(b"\r\n") == 0
    res = build(release, tmp_path / "d", with_runs=True)
    m = members(res["zip"])
    grid = {"results/" + f for f in run_dirs(repo_)}
    got = {"/".join(n.split("/")[2:-1]) for n in m if n.startswith("t/runs/")}
    assert got == grid | {"fe1_extraction"}                                                       # the case-study run is one of the grid runs: once
    for n, data in m.items():
        if n.startswith("t/runs/"):
            assert data == (repo_ / "/".join(n.split("/")[2:])).read_bytes(), n
    assert res["groups"]["runs"]["folders"] == len(got) and res["groups"]["runs"]["files"] == sum(1 for n in m if n.startswith("t/runs/"))
    pm = json.loads(m["t/PACK_manifest.json"])
    assert {x["path"] for x in pm["contents"]["runs"]} == got and set(pm["files"]) >= {n[2:] for n in m if n.startswith("t/runs/")}
    md = m["t/PACK.md"].decode()
    assert "`runs/`" in md and "git checkout" in md and "python scripts/release.py verify release/t" in md


def test_a_half_written_temporary_file_in_a_run_folder_is_left_out(release, tmp_path):
    repo_, out = release
    folder = repo_ / "results" / run_dirs(repo_)[0]
    (folder / "records.jsonl.tmp.77").write_text("half")
    m = members(build(release, tmp_path / "d", with_runs=True)["zip"])
    assert not any(".tmp" in n for n in m) and f"t/runs/results/{run_dirs(repo_)[0]}/records.jsonl" in m


def test_a_run_file_with_crlf_is_not_changed(release, tmp_path):
    repo_, out = release
    f = repo_ / "results" / run_dirs(repo_)[0] / "per_part.csv"
    f.write_bytes(f.read_bytes().replace(b"\n", b"\r\n"))
    m = members(build(release, tmp_path / "d", with_runs=True)["zip"])
    assert m["t/runs/results/" + run_dirs(repo_)[0] + "/per_part.csv"] == f.read_bytes() and b"\r\n" in f.read_bytes()


def test_the_run_files_are_only_listed_when_they_are_asked_for(release, tmp_path):
    a = plan(release)
    b = plan(release, with_runs=True)
    assert len(b["entries"]) > len(a["entries"]) and not any("run folders are not in the archive" in x for x in b["advisory"])


def test_a_missing_run_folder_blocks_when_runs_are_asked_for(release):
    repo_, out = release
    shutil.rmtree(repo_ / "fe1_extraction")
    assert plan(release)["blocking"] == []
    p = plan(release, with_runs=True)
    assert len(p["blocking"]) == 1 and "extraction run fe1_extraction: run folder fe1_extraction not found" in p["blocking"][0]


def test_a_grid_run_that_was_missing_when_the_release_was_built_is_not_a_problem(repo, tmp_path):
    from test_release import build as build_release
    gone = run_folder(repo, "b0")
    shutil.rmtree(repo / "results" / gone)
    out = repo / "release" / "t"
    build_release(repo, out)
    man = json.load(open(out / "tables" / "manifest.json"))
    assert any(r["status"] == "missing" for r in man["runs"])
    p = K.plan(str(out), root=str(repo), metadata_path=meta_file(repo), with_runs=True, regenerate=False)
    assert p["blocking"] == []
    got = {e["arc"].split("/")[2] for e in p["entries"] if e["arc"].startswith("runs/results/")}
    assert gone not in got and len(got) == len({r["folder"] for r in man["runs"] if r["status"] not in ("missing", "not_built") and r["folder"]})
    assert any("did not exist when the release was built" in a for a in p["advisory"])
    assert build_pack_ok(repo, out, tmp_path)


def build_pack_ok(repo_, out, tmp_path):
    res = K.build(str(out), str(tmp_path / "d"), root=str(repo_), metadata_path=meta_file(repo_), with_runs=True, regenerate=True)
    return res["zip"] is not None and K.verify_archive(res["zip"])["ok"]


def test_a_run_folder_outside_the_repository_cannot_be_packed(release, tmp_path_factory):
    repo_, out = release
    elsewhere = tmp_path_factory.mktemp("elsewhere") / "results"
    shutil.copytree(repo_ / "results", elsewhere)
    rec = json.loads((repo_ / "configs" / "release.json").read_text())
    rec["results"], rec["case_studies"] = str(elsewhere), {"runs": [str(elsewhere / run_dirs(repo_)[0])]}
    (repo_ / "configs" / "r2.json").write_text(json.dumps(rec))
    R.build(str(repo_ / "configs" / "r2.json"), str(repo_ / "release" / "far"), root=str(repo_), figures=False)
    p = K.plan(str(repo_ / "release" / "far"), root=str(repo_), metadata_path=meta_file(repo_), with_runs=True, regenerate=False)
    assert any("outside the repository" in b for b in p["blocking"])


def test_the_cache_goes_in_when_asked_and_only_the_json_files(release, tmp_path):
    repo_, out = release
    cache = tmp_path / "cache"
    (cache / "ab").mkdir(parents=True)
    (cache / "ab" / "ab12.json").write_text('{"text": "x"}')
    (cache / "ab" / "ab12.json.tmp.99").write_text("half")
    (cache / "ab" / "README.txt").write_text("not a cache entry")
    (cache / "cd").mkdir()
    (cache / "cd" / "cd34.json").write_bytes(b'{"text": "y"}\r\n')
    res = build(release, tmp_path / "d", with_cache=True, cache_dir=str(cache))
    m = members(res["zip"])
    assert {n for n in m if n.startswith("t/cache/")} == {"t/cache/llm/ab/ab12.json", "t/cache/llm/cd/cd34.json"}
    assert m["t/cache/llm/cd/cd34.json"] == b'{"text": "y"}\r\n'                                   # byte for byte
    assert res["groups"]["cache"]["files"] == 2 and any("every cached LLM call" in a for a in res["advisory"])
    assert "`cache/llm/`" in m["t/PACK.md"].decode()


def test_no_cache_is_a_blocking_item_when_it_is_asked_for(release, tmp_path):
    p = plan(release, with_cache=True, cache_dir=str(tmp_path / "none"))
    assert len(p["blocking"]) == 1 and "no cache files found" in p["blocking"][0]
    (tmp_path / "empty").mkdir()
    assert plan(release, with_cache=True, cache_dir=str(tmp_path / "empty"))["blocking"]


# ---------------------------------------------------------------- secrets in the files
def plant(repo_, text, name="notes.txt"):
    """A file in a run folder (the tables hash only the records file, so the release stays verified)."""
    folder = repo_ / "results" / run_dirs(repo_)[0]
    (folder / name).write_text(text)
    return f"runs/results/{run_dirs(repo_)[0]}/{name}"


def test_a_key_in_a_run_folder_blocks_the_pack_and_is_never_shown(release, tmp_path, monkeypatch):
    repo_, out = release
    arc = plant(repo_, f"line one\nGROQ key was {FAKE_GROQ}\nand {FAKE_SK}\n")
    res = build(release, tmp_path / "d", with_runs=True)
    assert res["zip"] is None and not (tmp_path / "d").exists()
    msg = " ".join(res["blocking"])
    assert "2 possible secret(s)" in msg and f"{arc} line 2 (Groq API key)" in msg and f"{arc} line 3 (API key (sk-...))" in msg
    shown = msg + K.format_plan(res)
    assert FAKE_GROQ not in shown and FAKE_SK not in shown and FAKE_GROQ[4:] not in shown
    assert [s["line"] for s in res["secrets"]] == [2, 3]


def test_the_same_files_pass_when_the_runs_are_not_packed(release, tmp_path):
    repo_, out = release
    plant(repo_, FAKE_GROQ)
    assert build(release, tmp_path / "d")["zip"] is not None


def test_the_value_of_your_own_key_is_found_whatever_it_looks_like(release, tmp_path, monkeypatch):
    repo_, out = release
    (repo_ / ".env").write_text("GROQ_API_KEY=plain-value-without-a-prefix-1234\n")
    arc = plant(repo_, "x\ny plain-value-without-a-prefix-1234 y\n")
    res = build(release, tmp_path / "d", with_runs=True)
    assert res["zip"] is None and f"{arc} line 2 (a value of your .env or environment)" in " ".join(res["blocking"])
    assert "plain-value-without-a-prefix" not in " ".join(res["blocking"])


def test_a_key_in_the_environment_is_found(release, tmp_path, monkeypatch):
    repo_, out = release
    monkeypatch.setenv("GROQ_API_KEY", "an-env-value-no-prefix-98765")
    plant(repo_, "an-env-value-no-prefix-98765")
    assert build(release, tmp_path / "d", with_runs=True)["zip"] is None


def test_a_key_in_the_cache_or_the_release_blocks_too(release, tmp_path):
    repo_, out = release
    cache = tmp_path / "cache" / "ab"
    cache.mkdir(parents=True)
    (cache / "x.json").write_text(json.dumps({"text": "ok", "params": {"api_key": "abcdef0123456789abcdef"}}))
    res = build(release, tmp_path / "d", with_cache=True, cache_dir=str(tmp_path / "cache"))
    assert res["zip"] is None and "cache/llm/ab/x.json line 1 (key, secret or password assignment)" in " ".join(res["blocking"])


def test_many_findings_are_reported_in_short(release):
    repo_, out = release
    plant(repo_, "\n".join(FAKE_GROQ for _ in range(30)))
    p = plan(release, with_runs=True)
    assert len(p["secrets"]) == 30 and "30 possible secret(s)" in p["blocking"][0] and "and 22 more" in p["blocking"][0]


# ---------------------------------------------------------------- verifying an archive
@pytest.fixture()
def packed(release, tmp_path):
    return build(release, tmp_path / "dist", with_runs=True)["zip"]


def test_a_fresh_archive_is_verified(packed):
    v = K.verify_archive(packed)
    assert v["ok"] and v["name"] == "t" and v["top"] == "t" and v["sidecar"] == "ok" and v["sums_ok"] and v["members"] > 20
    assert v["modified"] == v["missing"] == v["extra"] == v["release_modified"] == v["release_missing"] == v["unsafe"] == [] and v["corrupt"] is None
    assert K.format_verify_archive(v).endswith("VERIFIED") and "agrees" in K.format_verify_archive(v)


def test_the_archive_is_checked_without_the_repository_or_the_folders(packed, tmp_path, monkeypatch):
    moved = tmp_path / "elsewhere" / "t.zip"
    moved.parent.mkdir()
    shutil.copy(packed, moved)
    shutil.copy(packed + ".sha256", str(moved) + ".sha256")
    monkeypatch.chdir(tmp_path / "elsewhere")
    assert K.verify_archive("t.zip")["ok"]


@pytest.mark.parametrize("member", ["release/tables/manifest.json", "runs/", "zenodo_metadata.json", "PACK.md"])
def test_a_changed_member_is_found(packed, member):
    m = members(packed)
    name = next(n for n in m if n.startswith("t/" + member) and n.endswith((".json", ".md", ".jsonl", ".csv")))
    rewrite(packed, {name: m[name] + b" "})
    v = K.verify_archive(packed)
    assert not v["ok"] and name[2:] in v["modified"] and "NOT VERIFIED" in K.format_verify_archive(v)


def test_a_removed_member_is_found(packed):
    m = members(packed)
    name = next(n for n in m if n.startswith("t/runs/") and n.endswith("records.jsonl"))
    rewrite(packed, {name: None})
    v = K.verify_archive(packed)
    assert not v["ok"] and v["missing"] == [name[2:]] and not v["sums_ok"]


def test_an_added_member_is_found(packed):
    rewrite(packed, {"t/extra.txt": b"x"})
    v = K.verify_archive(packed)
    assert not v["ok"] and v["extra"] == ["extra.txt"]


def test_a_changed_sums_file_is_found(packed):
    rewrite(packed, {"t/SHA256SUMS.txt": b"0" * 64 + b"  PACK.md\n"})
    v = K.verify_archive(packed)
    assert not v["ok"] and not v["sums_ok"] and "does not agree with the files" in K.format_verify_archive(v)


def test_a_zip_that_does_not_match_its_sha256_file_is_found(packed):
    Path(packed + ".sha256").write_text("0" * 64 + "  t.zip\n")
    v = K.verify_archive(packed)
    assert not v["ok"] and v["sidecar"] == "mismatch" and "does NOT agree" in K.format_verify_archive(v)
    Path(packed + ".sha256").write_text("")
    assert K.verify_archive(packed)["sidecar"] == "mismatch"


def test_a_missing_sha256_file_is_noted_not_failed(packed):
    os.remove(packed + ".sha256")
    v = K.verify_archive(packed)
    assert v["ok"] and v["sidecar"] == "absent" and "not checked" in K.format_verify_archive(v)


def test_a_release_inside_that_disagrees_with_its_own_manifest_is_found_even_if_the_pack_was_remade(packed):
    import hashlib
    m = members(packed)
    name = "t/release/paper/readiness.txt"
    new = m[name] + b"edited\n"
    pm = json.loads(m["t/PACK_manifest.json"])
    pm["files"]["release/paper/readiness.txt"] = hashlib.sha256(new).hexdigest()
    pmb = (json.dumps(pm, indent=2) + "\n").encode()
    sums = dict(reversed(line.split("  ", 1)) for line in m["t/SHA256SUMS.txt"].decode().splitlines())
    sums["release/paper/readiness.txt"], sums["PACK_manifest.json"] = pm["files"]["release/paper/readiness.txt"], hashlib.sha256(pmb).hexdigest()
    rewrite(packed, {name: new, "t/PACK_manifest.json": pmb, "t/SHA256SUMS.txt": "".join(f"{h}  {f}\n" for f, h in sorted(sums.items())).encode()})
    v = K.verify_archive(packed)
    assert not v["ok"] and v["modified"] == [] and v["sums_ok"] and v["release_modified"] == ["paper/readiness.txt"]


def test_a_release_file_missing_inside_is_found(packed):
    m = members(packed)
    pm = json.loads(m["t/PACK_manifest.json"])
    pm["files"].pop("release/paper/readiness.txt")
    rewrite(packed, {"t/release/paper/readiness.txt": None, "t/PACK_manifest.json": (json.dumps(pm) + "\n").encode()})
    v = K.verify_archive(packed)
    assert not v["ok"] and "paper/readiness.txt" in v["release_missing"]


def test_unsafe_member_names_are_reported(packed):
    rewrite(packed, {"t/../evil.txt": b"x"})
    v = K.verify_archive(packed)
    assert not v["ok"] and v["unsafe"] == ["t/../evil.txt"] and "unsafe path" in K.format_verify_archive(v)


def test_a_member_with_damaged_data_is_reported(packed):
    with zipfile.ZipFile(packed) as zf:
        info = zf.getinfo("t/PACK.md")
        start = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)         # the first byte of its compressed data
    data = bytearray(Path(packed).read_bytes())
    for i in range(start, start + 8):
        data[i] ^= 0xFF
    Path(packed).write_bytes(bytes(data))
    v = K.verify_archive(packed)
    assert not v["ok"] and v["corrupt"] == "PACK.md" and "corrupt member: PACK.md" in K.format_verify_archive(v) and v["missing"] == []


def test_something_that_is_not_a_pack_is_refused(tmp_path):
    (tmp_path / "x.zip").write_bytes(b"not a zip")
    with pytest.raises(K.PackError, match="cannot be read as a zip file"):
        K.verify_archive(str(tmp_path / "x.zip"))
    with pytest.raises(K.PackError, match="cannot be read"):
        K.verify_archive(str(tmp_path / "none.zip"))
    for name, files, fragment in (("a", {"a/x.txt": b"1"}, "no PACK_manifest.json"), ("b", {"x.txt": b"1"}, "one top folder"),
                                  ("c", {"a/x": b"1", "b/y": b"1"}, "one top folder"), ("d", {"a/PACK_manifest.json": b"{"}, "not valid JSON"),
                                  ("e", {"a/PACK_manifest.json": b"[]"}, "not a pack manifest"), ("f", {"a/PACK_manifest.json": b'{"format": 2, "files": {}}'}, "not a pack manifest")):
        z = tmp_path / f"{name}.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for k, v in files.items():
                zf.writestr(k, v)
        with pytest.raises(K.PackError, match=fragment):
            K.verify_archive(str(z))


def test_a_folder_renamed_inside_the_zip_is_found(packed, tmp_path):
    m = members(packed)
    out = tmp_path / "renamed.zip"
    with zipfile.ZipFile(out, "w") as zf:
        for k, v in m.items():
            zf.writestr("other/" + k[2:], v)
    v = K.verify_archive(str(out))
    assert not v["ok"] and "is not the name in" in K.format_verify_archive(v)


# ---------------------------------------------------------------- the command line
def cli(*args, cwd=None):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "pack.py"), *args], capture_output=True, text=True, cwd=cwd, timeout=300)


def test_the_command_line_check_build_and_verify(release, tmp_path, capsys, monkeypatch):
    repo_, out = release
    import importlib.util
    spec = importlib.util.spec_from_file_location("pack_cli", os.path.join(ROOT, "scripts", "pack.py"))
    cli_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli_mod)
    monkeypatch.setattr(cli_mod, "ROOT", str(repo_))
    meta = meta_file(repo_)
    assert cli_mod.main(["check", str(out), "--metadata", meta, "--no-regenerate"]) == 0
    assert "READY TO PACK" in capsys.readouterr().out
    assert cli_mod.main(["build", str(out), "--metadata", meta, "--no-regenerate", "--out", str(tmp_path / "dist"), "--with-runs"]) == 0
    text = capsys.readouterr().out
    assert "Written:" in text and "SHA-256 of the zip:" in text and "scripts/pack.py verify" in text
    z = str(tmp_path / "dist" / "t.zip")
    assert cli_mod.main(["verify", z]) == 0 and "VERIFIED" in capsys.readouterr().out
    rewrite(z, {"t/PACK.md": b"changed"})
    assert cli_mod.main(["verify", z]) == 1 and "NOT VERIFIED" in capsys.readouterr().out


def test_the_command_line_does_not_write_when_something_blocks(release, tmp_path, capsys, monkeypatch):
    import importlib.util
    repo_, out = release
    spec = importlib.util.spec_from_file_location("pack_cli2", os.path.join(ROOT, "scripts", "pack.py"))
    cli_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli_mod)
    monkeypatch.setattr(cli_mod, "ROOT", str(repo_))
    assert cli_mod.main(["check", str(out), "--metadata", str(repo_ / "none.json"), "--no-regenerate"]) == 1
    assert "[BLOCKING] metadata file" in capsys.readouterr().out
    assert cli_mod.main(["build", str(out), "--metadata", str(repo_ / "none.json"), "--no-regenerate", "--out", str(tmp_path / "dist")]) == 1
    assert "Nothing was written." in capsys.readouterr().err and not (tmp_path / "dist").exists()


def test_the_command_line_stops_on_a_bad_input(tmp_path):
    (tmp_path / "empty").mkdir()
    r = cli("check", str(tmp_path / "empty"))
    assert r.returncode == 2 and r.stderr.startswith("STOPPED:") and "not a release folder" in r.stderr
    r = cli("verify", str(tmp_path / "none.zip"))
    assert r.returncode == 2 and "STOPPED:" in r.stderr


def test_the_script_runs_as_a_program(release, tmp_path):
    r = cli("--help")
    assert r.returncode == 0 and "check" in r.stdout and "--with-runs" in r.stdout and "verify" in r.stdout


def test_the_template_is_a_valid_json_file_with_the_repository_as_the_related_identifier():
    t = json.load(open(os.path.join(ROOT, "configs", "zenodo_template.json")))
    assert t["upload_type"] == "dataset" and t["access_right"] == "open" and t["related_identifiers"][0]["relation"] == "isSupplementTo"
    assert set(t) - {"_comment"} <= set(K.META_KEYS)
