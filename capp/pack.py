"""Pack (MTP-1 WP12, report 8.4 items iii and iv): a release folder (capp/release.py) in ONE zip file, ready to be put on Zenodo or kept as the
record of the final results.

    release folder (verified)  +  optionally the run folders  +  optionally the LLM cache  +  the metadata of the upload
        -> <name>.zip  and  <name>.zip.sha256

What `build` does, offline: it checks the release (`release.verify`: no file changed, and a rebuild gives the same files), reads the metadata
you wrote (configs/zenodo.json: title, creators, licence), scans every file that would go into the archive for secrets (API keys, tokens, the
value of your GROQ_API_KEY), and only if nothing is open writes the zip. Same inputs => the same members with the same hashes; the zip file
itself uses fixed times and a fixed order, so it is the same file on the same Python/zlib (another zlib version can compress differently: the
hashes of the members are what is compared, see `verify_archive`).

The archive (top folder <name>/):

    PACK.md  PACK_manifest.json  SHA256SUMS.txt  zenodo_metadata.json
    release/   the release folder as it is
    runs/      the run folders, at their path under the repository (only with runs)
    cache/llm/ the whole LLM response cache (only with cache): every call ever cached, not only the ones of this release

Nothing is uploaded and no account is needed: the zip is for you to upload. Packing does not make the results right: it records what they
are made from and that nothing was changed afterwards. See scripts/pack.py.
"""
import hashlib
import json
import os
import re
import tempfile
import zipfile
import zlib

from capp import config, release as R, tables as T

FORMAT = 1
PACK_MANIFEST = "PACK_manifest.json"
SUMS = "SHA256SUMS.txt"
META_FILE = "zenodo_metadata.json"
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
UPLOAD_TYPES = ("dataset", "software", "publication", "poster", "presentation", "image", "video", "other")
ACCESS_RIGHTS = ("open", "embargoed", "restricted", "closed")
META_KEYS = ("title", "description", "creators", "license", "upload_type", "access_right", "keywords", "related_identifiers", "version", "notes")
CREATOR_KEYS = ("name", "affiliation", "orcid")
RELATED_KEYS = ("identifier", "relation", "scheme")
ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
MAX_REPORTED = 20
_B = rb"(?<![A-Za-z0-9])"
SECRET_PATTERNS = (
    ("Groq API key", re.compile(_B + rb"gsk_[A-Za-z0-9]{20,}")),
    ("API key (sk-...)", re.compile(_B + rb"sk-[A-Za-z0-9_-]{32,}")),
    ("GitHub token", re.compile(_B + rb"(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("AWS access key id", re.compile(_B + rb"AKIA[0-9A-Z]{16}")),
    ("Google API key", re.compile(_B + rb"AIza[0-9A-Za-z_-]{35}")),
    ("Hugging Face token", re.compile(_B + rb"hf_[A-Za-z0-9]{34,}")),
    ("Slack token", re.compile(_B + rb"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("private key", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("key, secret or password assignment", re.compile(rb"(?i)[\"']?(?:api[_-]?key|secret|password|passwd|access[_-]?token|auth[_-]?token)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_/+=-]{20,}")),
)
ENV_NAME_RE = re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd)")
MIN_SECRET_LENGTH = 16


class PackError(ValueError):
    """A problem with the release folder, the metadata or the archive, in words for the author."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB"):
        n /= 1024
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"


def _more(items: list, shown: int = MAX_REPORTED) -> list:
    return list(items[:shown]) + ([f"... and {len(items) - shown} more"] if len(items) > shown else [])


# ---------------------------------------------------------------- the metadata of the upload
def normalise_metadata(raw, release_name: str) -> tuple:
    """(metadata, problems, advice): the Zenodo metadata object with the defaults filled in, every problem found (not only the first) and the
    things worth a look that do not stop the pack."""
    problems, advice = [], []
    if not isinstance(raw, dict):
        return None, ["metadata: must be a JSON object"], advice
    raw = {k: v for k, v in raw.items() if k != "_comment"}
    unknown = sorted(set(raw) - set(META_KEYS))
    if unknown:
        problems.append(f"metadata: unknown key(s) {unknown} (allowed: {', '.join(META_KEYS)})")

    def text(key, default=""):
        v = raw.get(key)
        if v is None:
            return default
        if not isinstance(v, str):
            problems.append(f"{key}: must be text")
            return default
        return v.strip() or default

    title = text("title", f"LLM-CAPP MTP-1: release {release_name}")
    description, notes, version = text("description"), text("notes"), text("version", release_name)
    license_ = text("license")
    if not license_:
        problems.append("license: required (a Zenodo licence id such as cc-by-4.0; the choice is yours)")
    elif " " in license_:
        advice.append(f"license: '{license_}' has a space; Zenodo expects an id such as cc-by-4.0")
    upload_type = text("upload_type", "dataset")
    if upload_type not in UPLOAD_TYPES:
        problems.append(f"upload_type: '{upload_type}' is not one of {', '.join(UPLOAD_TYPES)}")
    access = text("access_right", "open")
    if access not in ACCESS_RIGHTS:
        problems.append(f"access_right: '{access}' is not one of {', '.join(ACCESS_RIGHTS)}")

    creators = []
    rc = raw.get("creators")
    if not isinstance(rc, list) or not rc:
        problems.append("creators: required, a list with at least one {name, affiliation, orcid}")
    else:
        for i, c in enumerate(rc, 1):
            if not isinstance(c, dict):
                problems.append(f"creators[{i}]: must be an object")
                continue
            extra = sorted(set(c) - set(CREATOR_KEYS))
            if extra:
                problems.append(f"creators[{i}]: unknown key(s) {extra}")
            vals = {}
            for k in CREATOR_KEYS:
                v = c.get(k)
                if v is not None and not isinstance(v, str):
                    problems.append(f"creators[{i}].{k}: must be text")
                    v = None
                vals[k] = (v or "").strip()
            if not vals["name"]:
                problems.append(f"creators[{i}].name: empty (write it as Family, Given)")
            elif "," not in vals["name"]:
                advice.append(f"creators[{i}].name: Zenodo lists names as 'Family, Given'")
            if vals["orcid"] and not ORCID_RE.match(vals["orcid"]):
                problems.append(f"creators[{i}].orcid: '{vals['orcid']}' is not of the form 0000-0002-1825-0097")
            creators.append({k: v for k, v in vals.items() if v})

    keywords = raw.get("keywords", [])
    if keywords is None:
        keywords = []
    if not isinstance(keywords, list) or not all(isinstance(k, str) and k.strip() for k in keywords):
        problems.append("keywords: must be a list of non-empty text")
        keywords = []
    keywords = list(dict.fromkeys(k.strip() for k in keywords))

    related = raw.get("related_identifiers", [])
    if related is None:
        related = []
    rel_out = []
    if not isinstance(related, list):
        problems.append("related_identifiers: must be a list")
    else:
        for i, r in enumerate(related, 1):
            if not isinstance(r, dict):
                problems.append(f"related_identifiers[{i}]: must be an object")
                continue
            extra = sorted(set(r) - set(RELATED_KEYS))
            if extra:
                problems.append(f"related_identifiers[{i}]: unknown key(s) {extra}")
            vals = {k: (r.get(k) if isinstance(r.get(k), str) else "").strip() for k in RELATED_KEYS}
            if not vals["identifier"] or not vals["relation"]:
                problems.append(f"related_identifiers[{i}]: identifier and relation are required (e.g. relation isSupplementTo)")
            rel_out.append({k: v for k, v in vals.items() if v})

    meta = {"title": title, "upload_type": upload_type, "description": description, "creators": creators, "license": license_,
            "access_right": access, "version": version}
    if keywords:
        meta["keywords"] = keywords
    if rel_out:
        meta["related_identifiers"] = rel_out
    if notes:
        meta["notes"] = notes
    return meta, problems, advice


def load_metadata(path: str, release_name: str) -> tuple:
    """(metadata or None, problems, advice) from the file you wrote; a file that is missing or not JSON is a problem, not a crash."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except OSError:
        return None, [f"metadata file {path} not found: copy configs/zenodo_template.json to configs/zenodo.json and fill it in"], []
    except ValueError as exc:
        return None, [f"metadata file {path} is not valid JSON: {exc}"], []
    return normalise_metadata(raw, release_name)


def default_description(name: str, dev: bool) -> str:
    return (("DEVELOPMENT RESULTS, NOT PAPER RESULTS. " if dev else "") + f"Release {name} of the MTP-1 project on LLM-based process planning (LLM-CAPP): "
            "results tables with paired comparisons, failure case studies, the expert benchmark with its datasheet and the draft of the paper, "
            "built from finished runs. RELEASE.md in the archive says what it was made from.")


# ---------------------------------------------------------------- secrets
def _env_values(root: str) -> list:
    """The values of the secrets of this machine: environment variables and .env entries whose name looks like a key, a token or a password."""
    values = {}
    for k, v in os.environ.items():
        if ENV_NAME_RE.search(k) and len(v) >= MIN_SECRET_LENGTH:
            values[k] = v
    try:
        with open(os.path.join(root, ".env"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip("\"'")                                   # 'export NAME' still has the name in it
                if ENV_NAME_RE.search(k) and len(v) >= MIN_SECRET_LENGTH:
                    values[k] = v
    except OSError:
        pass
    return sorted({v.encode("utf-8") for v in values.values()})


def scan_secrets(data: bytes, literals: list = ()) -> list:
    """[(line, what)] for every secret-looking thing in the bytes. The text of the match is never returned."""
    found = []
    for what, pat in SECRET_PATTERNS:
        for m in pat.finditer(data):
            found.append((data.count(b"\n", 0, m.start()) + 1, what))
    for lit in literals:
        start = 0
        while True:
            i = data.find(lit, start)
            if i < 0:
                break
            found.append((data.count(b"\n", 0, i) + 1, "a value of your .env or environment"))
            start = i + len(lit)
    return sorted(set(found))


# ---------------------------------------------------------------- what goes in
def _read(entry: dict) -> bytes:
    with open(entry["path"], "rb") as f:
        data = f.read()
    if entry["norm"]:
        data = data.replace(b"\r\n", b"\n")
    return data


def _walk(folder: str) -> list:
    out = []
    for dp, dn, fn in os.walk(folder):
        dn[:] = sorted(d for d in dn if d != "__pycache__")
        out += [os.path.join(dp, f) for f in sorted(fn) if ".tmp" not in f]
    return out


def _run_folders(release_dir: str, root: str) -> tuple:
    """([{'what', 'abs', 'rel'}], problems, n_without_run): the grid runs of tables/manifest.json that existed when the release was built (a cell
    whose run was missing or not built has no folder to pack), the case-study runs and the extraction runs of the recipe."""
    problems = []
    rec = R.load_recipe(os.path.join(release_dir, "inputs", "recipe.json"))
    cfg = T.load_config(os.path.join(release_dir, "inputs", "grid.json"))
    results_abs = R._abs(root, rec["results"]) if rec["results"] else None
    rdir = T.resolve_results_dir(cfg, results_abs)
    try:
        with open(os.path.join(release_dir, "tables", "manifest.json"), encoding="utf-8") as f:
            tm = json.load(f)
    except (OSError, ValueError) as exc:
        raise PackError(f"tables/manifest.json of the release cannot be read ({exc}): the run folders cannot be listed")
    cells = [r for r in tm.get("runs", []) if r.get("folder")]
    present = [r for r in cells if r.get("status") not in ("missing", "not_built")]
    want = [(f"grid run {r['row']}/{r['model']}", os.path.join(rdir, r["folder"])) for r in present]
    want += [(i["what"] + " " + i["path"], i["abs"]) for i in R.inputs(rec, root) if i["what"] in ("case-study run", "extraction run")]
    out, seen = [], set()
    for what, path in want:
        a = os.path.normpath(os.path.abspath(path))
        if a in seen:
            continue
        seen.add(a)
        try:
            rel = os.path.relpath(a, root).replace("\\", "/")
        except ValueError:
            rel = ".."
        if rel == ".." or rel.startswith("../"):
            problems.append(f"{what}: the folder is outside the repository ({a}); it cannot be packed at a path under the repository")
        elif not os.path.isdir(a):
            problems.append(f"{what}: run folder {rel} not found")
        else:
            out.append({"what": what, "abs": a, "rel": rel})
    return out, problems, len({r["folder"] for r in cells}) - len({r["folder"] for r in present})


def plan(release_dir: str, root: str = None, metadata_path: str = None, with_runs: bool = False, with_cache: bool = False, cache_dir: str = None,
         regenerate: bool = True) -> dict:
    """What a pack would contain and what stands in its way. {'name', 'manifest', 'meta', 'entries', 'groups', 'blocking', 'advisory', 'verify'}.
    Nothing is written."""
    root = os.path.abspath(root or str(config.ROOT))
    rd = os.path.abspath(release_dir)
    man = R.load_manifest(rd)
    name = man.get("name")
    if not isinstance(name, str) or not R.NAME_RE.match(name):
        raise PackError(f"{R.MANIFEST} has no valid release name")
    dev = not man.get("sealed")
    blocking, advisory = [], []

    # the release must be what it says it is
    v = None
    try:
        v = R.verify(rd, root, regenerate=regenerate)
    except (R.ReleaseError, T.TableError, OSError, ValueError) as exc:
        blocking.append(f"the release cannot be verified here ({exc}); with --no-regenerate only the file hashes are checked")
    if v is not None and not v["ok"]:
        lines = [f"modified: {f}" for f in v["modified"]] + [f"missing: {f}" for f in v["missing"]] + [f"not in the manifest: {f}" for f in v["extra"]]
        if v["regen"]:
            lines += [f"rebuild differs: {f}" for f in v["regen"]["differs"]] + [f"only in the release: {f}" for f in v["regen"]["only_release"]] \
                + [f"only in the rebuild: {f}" for f in v["regen"]["only_now"]]
        blocking.append("the release is not verified (python scripts/release.py verify <folder> says why): " + "; ".join(_more(lines, 5)))
    if not regenerate:
        advisory.append("the rebuild of the release was skipped (--no-regenerate): only the file hashes were checked")

    # the metadata of the upload
    meta, mproblems, advice = (None, ["no metadata file given"], []) if not metadata_path else load_metadata(metadata_path, name)
    blocking += mproblems
    advisory += advice
    if meta is not None and not meta["description"]:
        meta["description"] = default_description(name, dev)

    # the files
    entries = []

    def add(arc, path, group, norm=False):
        entries.append({"arc": arc, "path": path, "norm": norm, "group": group})

    for rel in sorted(set(man["files"]) | {R.MANIFEST}):
        p = os.path.join(rd, rel)
        if os.path.isfile(p):
            add("release/" + rel, p, "release", rel.lower().endswith(R.TEXT_SUFFIXES))
    groups = {"release": {"files": 0, "bytes": 0}, "runs": None, "cache": None}
    if with_runs:
        groups["runs"] = {"folders": 0, "files": 0, "bytes": 0, "list": []}
        try:
            folders, rproblems, absent = _run_folders(rd, root)
        except (R.ReleaseError, T.TableError, OSError, ValueError) as exc:
            folders, rproblems, absent = [], [f"the run folders cannot be listed ({exc})"], 0
        blocking += rproblems
        if absent:
            advisory.append(f"{absent} run folder(s) of the grid did not exist when the release was built (see the issues in tables/): "
                            "the archive holds the others")
        for fo in folders:
            files = _walk(fo["abs"])
            for p in files:
                add("runs/" + fo["rel"] + "/" + os.path.relpath(p, fo["abs"]).replace("\\", "/"), p, "runs")
            groups["runs"]["list"].append({"path": fo["rel"], "files": len(files), "what": fo["what"]})
    else:
        advisory.append("the run folders are not in the archive (--with-runs): the tables cannot be rebuilt from the archive alone; "
                        "tables/manifest.json only holds a hash of each records file")
    if with_cache:
        cdir = os.path.abspath(cache_dir or str(config.get_cache_dir()))
        files = [p for p in _walk(cdir) if p.endswith(".json")] if os.path.isdir(cdir) else []
        if not files:
            blocking.append(f"--with-cache: no cache files found in {cdir}")
        for p in files:
            add("cache/llm/" + os.path.relpath(p, cdir).replace("\\", "/"), p, "cache")
        groups["cache"] = {"files": 0, "bytes": 0}
        if files:
            advisory.append(f"the cache holds every cached LLM call ({len(files)} files), not only the calls of this release, development runs included")

    # hash, size and secrets, one read per file
    literals = _env_values(root)
    secrets = []
    for e in entries:
        data = _read(e)
        e["sha"], e["size"] = _sha256(data), len(data)
        for line, what in scan_secrets(data, literals):
            secrets.append({"file": e["arc"], "line": line, "what": what})
        g = groups[e["group"]]
        g["files"] += 1
        g["bytes"] += e["size"]
    if groups["runs"]:
        groups["runs"]["folders"] = len(groups["runs"]["list"])
    if secrets:
        blocking.append(f"{len(secrets)} possible secret(s) in the files that would go into the archive (take them out, and if a real key was "
                        "in a published place, change the key): " + "; ".join(_more([f"{s['file']} line {s['line']} ({s['what']})" for s in secrets], 8)))

    # things to know
    if dev:
        advisory.append("development release (the grid runs on the development set): a rehearsal, not something to upload as a result")
    if not man.get("ready"):
        advisory.append(f"the paper readiness of the release had {len(man.get('blocking') or [])} blocking item(s) open (paper/readiness.txt)")
    fr = man.get("freeze")
    if not fr or fr.get("status") != "ok":
        advisory.append("freeze: " + ("no manifest when the release was built" if not fr else f"status {fr.get('status')} when the release was built"))
    if (man.get("git") or {}).get("dirty"):
        advisory.append("the working tree had uncommitted changes when the release was built: the commit named in the archive is not the whole state")
    tags = R._git_tags(root)
    lacking = [t for t in (*R.EXPECTED_TAGS, f"release-{name}") if t not in tags]
    if lacking:
        advisory.append(f"git tag(s) not there yet: {', '.join(lacking)} (RELEASE.md has the commands)")
    if any(e["arc"].startswith("release/benchmark/") for e in entries):
        advisory.append("the archive contains the expert benchmark and its annotations: check that the annotators agreed to a publication "
                        "and that the licence fits it")
    return {"name": name, "release_dir": rd, "manifest": man, "meta": meta, "entries": entries, "groups": groups, "blocking": blocking,
            "advisory": advisory, "secrets": secrets, "verify": v, "dev": dev, "with_runs": with_runs, "with_cache": with_cache}


def format_plan(p: dict) -> str:
    man, g = p["manifest"], p["groups"]
    total_files = sum(x["files"] for x in (g["release"], g["runs"], g["cache"]) if x)
    total_bytes = sum(x["bytes"] for x in (g["release"], g["runs"], g["cache"]) if x)
    L = [f"Pack of release {p['name']} (built {man.get('created_utc')}, dataset {man.get('dataset')}, {'development' if p['dev'] else 'sealed'})", ""]
    L.append(f"  release/    {g['release']['files']} files, {_size(g['release']['bytes'])}")
    L.append(f"  runs/       " + (f"{g['runs']['folders']} run folder(s), {g['runs']['files']} files, {_size(g['runs']['bytes'])}" if g["runs"] else "not included"))
    L.append(f"  cache/llm/  " + (f"{g['cache']['files']} files, {_size(g['cache']['bytes'])}" if g["cache"] else "not included"))
    L.append(f"  (and PACK.md, PACK_manifest.json, SHA256SUMS.txt, {META_FILE}; {total_files} files, {_size(total_bytes)} before compression)")
    L.append("")
    if p["blocking"]:
        L += [f"[BLOCKING] {m}" for m in p["blocking"]]
    L += [f"[advisory] {m}" for m in p["advisory"]]
    L += ["", "READY TO PACK" if not p["blocking"] else f"NOT READY: {len(p['blocking'])} blocking item(s)"]
    return "\n".join(L)


# ---------------------------------------------------------------- the files that are made here
def _pack_md(p: dict) -> str:
    man, g, meta = p["manifest"], p["groups"], p["meta"]
    git = man.get("git") or {}
    commit = git.get("commit") or "unknown"
    L = [f"# Archive: {p['name']}", ""]
    if p["dev"]:
        L += ["> **DEVELOPMENT RESULTS, NOT PAPER RESULTS.** The grid of this release runs on the development set: this archive is a rehearsal.", ""]
    L += ["This is the release folder `release/` of the MTP-1 project (LLM-based process planning, LLM-CAPP) with the files that go with it. "
          f"Made by `python scripts/pack.py build` from the release `{p['name']}`, which was verified before it was packed.", "",
          "## Contents", "", "| path | what |", "|---|---|",
          "| `release/` | the release folder as `scripts/release.py` built it: `RELEASE.md` says what it was made from and how ready the numbers are |"]
    if g["runs"]:
        L.append("| `runs/` | the run folders the tables and the case studies are made from, at their path under the repository (byte for byte as they were) |")
    if g["cache"]:
        L.append("| `cache/llm/` | the whole LLM response cache of the project: every cached call, not only the calls of this release |")
    L += [f"| `{META_FILE}` | the metadata for the upload (title, creators, licence), written from `configs/zenodo.json` |",
          f"| `{PACK_MANIFEST}` | what this archive is made from: release, commit, the SHA-256 of every file |",
          f"| `{SUMS}` | the same hashes in the form `sha256sum -c` reads |", "",
          "## Where it comes from", "",
          f"- Release `{p['name']}`, built {man.get('created_utc')}; dataset `{man.get('dataset')}` ({'development' if p['dev'] else 'sealed'}).",
          f"- Git commit `{commit}`" + (" (the working tree had uncommitted changes)" if git.get("dirty") else "") + f"; tags at that time: {', '.join(man.get('tags') or []) or 'none'}.",
          "- The code is not in this archive: it is the git repository at that commit. The results are only meaningful with that code.",
          f"- Licence: {meta['license']}.", "",
          "## Check it", "",
          "In the repository (every file, the sums, the release inside):", "", "```", f"python scripts/pack.py verify {p['name']}.zip", "```", "",
          f"After unzipping, inside the folder `{p['name']}/` (Linux, macOS):", "", "```", f"sha256sum -c {SUMS}", "```", "",
          f"In PowerShell, compare the result with `{p['name']}.zip.sha256`:", "", "```", f"Get-FileHash {p['name']}.zip", "```", ""]
    if g["runs"]:
        L += ["## Rebuild the tables", "",
              f"1. `git checkout {commit}` in a clone of the repository.",
              "2. Copy the contents of `runs/` into the repository folder (the paths below `runs/` are the paths of the run folders).",
              f"3. Copy `release/` to `release/{p['name']}` in the repository and run `python scripts/release.py verify release/{p['name']}`: "
              "it builds the release again from `inputs/recipe.json` and compares every file.", ""]
    else:
        L += ["## Rebuild the tables", "", "This archive has no run folders, so the tables cannot be rebuilt from it alone: `release/tables/manifest.json` "
              "only holds a hash of each records file.", ""]
    L += ["## What this does not show", "",
          "- That the packed files are the ones that were made, and that nothing was changed afterwards. It does not show that the inputs of the release "
          "(the runs, the benchmark annotation, the prose) are right.",
          "- The zip file uses fixed times and a fixed order, but another version of zlib can compress the same files into different bytes. "
          "The SHA-256 of the members does not depend on that: compare those.", ""]
    return "\n".join(L)


def _pack_manifest(p: dict, files: dict) -> dict:
    man, g = p["manifest"], p["groups"]
    return {"format": FORMAT, "tool": "capp.pack (scripts/pack.py)", "name": p["name"], "release_created_utc": man.get("created_utc"),
            "release_manifest_sha256": files["release/" + R.MANIFEST], "dataset": man.get("dataset"), "sealed": bool(man.get("sealed")),
            "git": man.get("git"), "tags": man.get("tags"),
            "contents": {"release": g["release"]["files"], "runs": g["runs"]["list"] if g["runs"] else None,
                         "cache_files": g["cache"]["files"] if g["cache"] else None},
            "metadata": p["meta"], "files": dict(sorted(files.items()))}


def build(release_dir: str, out_dir: str, root: str = None, metadata_path: str = None, with_runs: bool = False, with_cache: bool = False,
          cache_dir: str = None, regenerate: bool = True) -> dict:
    """Writes <out_dir>/<name>.zip and <name>.zip.sha256 when nothing is open. Returns the plan with 'zip', 'sha256_file', 'zip_sha256',
    'members' (None for 'zip' when something blocks: nothing is written then)."""
    p = plan(release_dir, root, metadata_path, with_runs, with_cache, cache_dir, regenerate)
    p.update(zip=None, sha256_file=None, zip_sha256=None, members=0)
    if p["blocking"]:
        return p
    name = p["name"]
    meta_bytes = (json.dumps(p["meta"], indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    files = {e["arc"]: e["sha"] for e in p["entries"]}
    files[META_FILE] = _sha256(meta_bytes)
    pack_md = _pack_md(p).encode("utf-8")
    files["PACK.md"] = _sha256(pack_md)
    pm = (json.dumps(_pack_manifest(p, files), indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    sums_files = dict(files, **{PACK_MANIFEST: _sha256(pm)})
    sums = "".join(f"{h}  {f}\n" for f, h in sorted(sums_files.items())).encode("utf-8")

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, name + ".zip")
    tmp = os.path.join(out_dir, f".pack-{os.getpid()}.zip")                                    # not mkstemp: that would make the zip readable by its owner only
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            def put(arc, data):
                zi = zipfile.ZipInfo(f"{name}/{arc}", date_time=ZIP_TIME)
                zi.compress_type, zi.create_system, zi.external_attr = zipfile.ZIP_DEFLATED, 3, 0o644 << 16
                zf.writestr(zi, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
            fixed = {"PACK.md": pack_md, PACK_MANIFEST: pm, SUMS: sums, META_FILE: meta_bytes}
            by_arc = {e["arc"]: e for e in p["entries"]}
            for arc in sorted(set(by_arc) | set(fixed)):
                put(arc, fixed[arc] if arc in fixed else _read(by_arc[arc]))
        os.replace(tmp, zip_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    with open(zip_path, "rb") as f:
        zsha = _sha256(f.read())
    side = zip_path + ".sha256"
    with open(side, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{zsha}  {name}.zip\n")
    p.update(zip=zip_path, sha256_file=side, zip_sha256=zsha, members=len(sums_files) + 1)
    return p


# ---------------------------------------------------------------- checking an archive
def verify_archive(path: str) -> dict:
    """The archive on its own, without the repository: no member changed, none is missing or added, the sums file and the .sha256 next to the zip
    agree, and the release inside matches its own manifest. {'ok', 'name', 'modified', 'missing', 'extra', 'sums_ok', 'sidecar', 'corrupt',
    'unsafe', 'release_modified', 'release_missing', 'members'}. Raises PackError for something that is not a pack."""
    try:
        zf = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise PackError(f"{path} cannot be read as a zip file ({exc})")
    with zf:
        names = [i.filename for i in zf.infolist() if not i.filename.endswith("/")]
        tops = {n.split("/", 1)[0] for n in names}
        if len(tops) != 1 or any("/" not in n for n in names):
            raise PackError(f"{path}: not a pack (expected one top folder)")
        top = tops.pop()
        if f"{top}/{PACK_MANIFEST}" not in names:
            raise PackError(f"{path}: no {PACK_MANIFEST}: not made by scripts/pack.py")
        try:
            man = json.loads(zf.read(f"{top}/{PACK_MANIFEST}").decode("utf-8"))
        except ValueError as exc:
            raise PackError(f"{PACK_MANIFEST} is not valid JSON: {exc}")
        if not isinstance(man, dict) or man.get("format") != FORMAT or not isinstance(man.get("files"), dict):
            raise PackError(f"{PACK_MANIFEST}: not a pack manifest of format {FORMAT}")
        unsafe = sorted(n for n in names if n.startswith("/") or ".." in n.split("/") or "\\" in n or ":" in n.split("/")[0])
        members = {n[len(top) + 1:]: n for n in names if n not in unsafe}
        hashes, unreadable = {}, []
        for rel, full in members.items():
            try:
                hashes[rel] = _sha256(zf.read(full))
            except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError):
                unreadable.append(rel)
        corrupt = unreadable[0] if unreadable else None                                          # a member whose CRC or compressed data is bad
        recorded = man["files"]
        modified = sorted(r for r, h in recorded.items() if r in hashes and hashes[r] != h)
        missing = sorted(r for r in recorded if r not in hashes and r not in unreadable)
        extra = sorted(r for r in hashes if r not in recorded and r not in (PACK_MANIFEST, SUMS))
        want_sums = "".join(f"{h}  {f}\n" for f, h in sorted((r, h) for r, h in hashes.items() if r != SUMS))
        sums_ok = SUMS in members and zf.read(members[SUMS]).decode("utf-8", "replace") == want_sums
        rel_modified, rel_missing = [], []
        if "release/" + R.MANIFEST in members:
            try:
                rman = json.loads(zf.read(members["release/" + R.MANIFEST]).decode("utf-8"))
                for rel, h in rman["files"].items():
                    full = members.get("release/" + rel)
                    if full is None:
                        rel_missing.append(rel)
                        continue
                    if "release/" + rel in unreadable:
                        continue
                    if _sha256(zf.read(full)) != h:                                              # stored with LF, like the manifest hashes
                        rel_modified.append(rel)
            except (ValueError, KeyError, AttributeError):
                rel_modified.append(R.MANIFEST)
        else:
            rel_missing.append(R.MANIFEST)
    side = path + ".sha256"
    sidecar = "absent"
    if os.path.isfile(side):
        try:
            with open(side, encoding="utf-8") as f:
                want = f.read().split()[0].lower()
            with open(path, "rb") as f:
                sidecar = "ok" if _sha256(f.read()) == want else "mismatch"
        except (OSError, IndexError):
            sidecar = "mismatch"
    ok = not (modified or missing or extra or corrupt or unsafe or rel_modified or rel_missing) and sums_ok and sidecar != "mismatch" and man.get("name") == top
    return {"ok": ok, "name": man.get("name"), "top": top, "modified": modified, "missing": missing, "extra": extra, "sums_ok": sums_ok,
            "sidecar": sidecar, "corrupt": corrupt, "unsafe": unsafe, "release_modified": sorted(rel_modified), "release_missing": sorted(rel_missing),
            "members": len(members)}


def format_verify_archive(v: dict) -> str:
    L = [f"Archive {v['top']}: {v['members']} files"]
    if v["name"] != v["top"]:
        L.append(f"  the folder name '{v['top']}' is not the name in {PACK_MANIFEST} ('{v['name']}')")
    L += [f"  corrupt member: {v['corrupt']}"] if v["corrupt"] else []
    L += [f"  unsafe path: {n}" for n in v["unsafe"]]
    L += [f"  modified: {f}" for f in v["modified"]] + [f"  missing: {f}" for f in v["missing"]] + [f"  not in the manifest: {f}" for f in v["extra"]]
    L += [f"  release inside: {f} does not match release_manifest.json" for f in v["release_modified"]]
    L += [f"  release inside: {f} is missing" for f in v["release_missing"]]
    L.append("  " + (f"{SUMS} agrees with the files." if v["sums_ok"] else f"{SUMS} does not agree with the files."))
    L.append("  " + {"ok": "The .sha256 next to the zip agrees.", "mismatch": "The .sha256 next to the zip does NOT agree with the zip.",
                     "absent": "No .sha256 next to the zip: not checked."}[v["sidecar"]])
    L += ["", "VERIFIED" if v["ok"] else "NOT VERIFIED"]
    return "\n".join(L)
