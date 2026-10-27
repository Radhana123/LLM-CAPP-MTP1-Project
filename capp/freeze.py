"""Configuration freeze (MTP-1 WP0 / WP11, report 7.2 and the gates of Table 6): a manifest that pins what a final run is made of, and a
check that nothing behind it has changed since.

The report says that the test benchmark "is used only for final runs, after all components are frozen, and no prompt or setting is
changed after test results have been seen". Until now that was a sentence; this module makes it checkable.

    FREEZE.json   written by `python scripts/freeze.py create`, committed and tagged (baseline-v0, freeze-v1 ...)
    verify        recomputes the fingerprint and lists what differs
    status        the same, in one word, for run.json: none | ok | analysis-changed | drift | unreadable

What is fingerprinted (SHA-256 of the content, line endings normalised so that a Windows checkout and a Linux one give the same hash):
    files      every .py and .json at the top of capp/, week1 ... week6 and scripts/ (tests are not part of the experiment). Each file has
               a KIND: "analysis" for the code that only reads finished runs (tables, case studies, summaries, route sheet, benchmark
               tools, the UI, the scripts), "behaviour" for everything else, so that a file that is new or unknown counts as behaviour.
    config     every upper-case constant of capp/config.py (temperatures, token limits, ...) and the model id of each role as resolved now
               (so CAPP_MODEL_PRIMARY set differently from the frozen run is drift). Prices are recorded, not compared: they only turn token
               counts into a cost afterwards.
    datasets   the development set, the test set and the benchmark (those that exist), plus any --dataset
    grids      the ablation config(s) of the final runs, as canonical JSON (whitespace does not matter)
    extras     any other file given with --extra (the tuned confidence threshold, a selection of the A5/A6/A7 choices ...)

A dataset that did not exist at freeze time and does now (the benchmark, once annotated) is reported but is not drift.

Verdict of a comparison (the status of a run):
    ok                nothing differs
    analysis-changed  only analysis code differs: allowed (a table bug may be fixed after the freeze), reported
    drift             a behaviour file, a config constant, a model id, a dataset, the grid or an extra differs: the results are not those of
                      the frozen configuration
A manifest is created once. `create` refuses to overwrite one (use --force, and say why in --note); `diff A B` compares two manifests (the
changes between the Tier-1 freeze and the final one).

Honest limits: this proves that the files are the ones that were frozen, not that nobody looked at test results before freezing. The
order of events (freeze, then the first sealed run) is recorded by the file and the run.json of each run, and by the git tag.
"""
import copy
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys

MANIFEST_NAME = "FREEZE.json"
SCHEMA = 1
SCAN_DIRS = ("capp", "week1", "week2", "week3", "week4", "week5", "week6", "scripts")
SCAN_SUFFIXES = (".py", ".json")
DEFAULT_DATASETS = ("dev_v2", "test_v2", "bench_v1")
OK_STATUSES = ("ok", "analysis-changed")

# Code that only reads finished runs or builds a deliverable from them: changing it cannot change a route, a selection or a record.
ANALYSIS_MODULES = {
    "capp/tables.py", "capp/tablefigs.py", "capp/casestudy.py", "capp/summary.py", "capp/compare.py", "capp/partdiff.py",
    "capp/riskview.py", "capp/analysis_view.py", "capp/extraction_eval.py", "capp/extraction_review.py", "capp/routesheet.py",
    "capp/process_plan.py", "capp/process_tables.py", "capp/benchmark.py", "capp/cost.py", "capp/calibration.py", "capp/pareto.py",
    "capp/datagen.py", "capp/archetypes.py", "capp/freeze.py",
    "week6/app.py", "week6/final_report.py", "week6/visualizer.py",
}
ANALYSIS_CONFIG = ("PRICE_USD_PER_MTOK",)           # recorded, not compared
_UPPER = re.compile(r"^[A-Z][A-Z0-9_]*$")


class FreezeError(Exception):
    pass


# ---------------------------------------------------------------- hashing
def _norm(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(_norm(f.read())).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json_file(path: str) -> str:
    """SHA-256 of the canonical form of a JSON file, so that indentation and key order do not matter; any other file as plain bytes."""
    if path.endswith(".json"):
        try:
            with open(path, encoding="utf-8") as f:
                return hashlib.sha256(canonical(json.load(f)).encode("utf-8")).hexdigest()
        except ValueError:
            pass
    return sha256_file(path)


def kind_of(rel: str) -> str:
    return "analysis" if rel in ANALYSIS_MODULES or rel.startswith("scripts/") else "behaviour"


def _rel(root: str, path: str) -> str:
    ap, ar = os.path.abspath(path), os.path.abspath(root)
    if ap == ar or ap.startswith(ar + os.sep):
        return os.path.relpath(ap, ar).replace("\\", "/")
    return ap.replace("\\", "/")


def _abs(root: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(root, *rel.split("/"))


# ---------------------------------------------------------------- the pieces of a fingerprint
def scan_files(root: str) -> dict:
    out = {}
    for d in SCAN_DIRS:
        folder = os.path.join(root, d)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            p = os.path.join(folder, name)
            if os.path.isfile(p) and name.endswith(SCAN_SUFFIXES):
                rel = f"{d}/{name}"
                out[rel] = {"sha256": sha256_file(p), "kind": kind_of(rel)}
    return out


def config_snapshot(cfg=None) -> dict:
    """The upper-case, JSON-able constants of capp/config.py, the resolved model id of each role, and the structured-output mode."""
    if cfg is None:
        from capp import config as cfg
    snap = {}
    for k in sorted(vars(cfg)):
        if _UPPER.match(k) and k not in ANALYSIS_CONFIG:
            try:
                v = json.loads(json.dumps(getattr(cfg, k)))
            except (TypeError, ValueError):
                continue
            snap[k] = v
    snap["MODELS"] = {role: cfg.model(role) for role in cfg.MODEL_ROLES}
    if hasattr(cfg, "structured_mode"):
        snap["STRUCTURED_MODE"] = cfg.structured_mode()
    return snap


def price_snapshot(cfg=None):
    if cfg is None:
        from capp import config as cfg
    try:
        return json.loads(json.dumps(getattr(cfg, "PRICE_USD_PER_MTOK")))
    except (AttributeError, TypeError, ValueError):
        return None


def _dataset_file(root: str, name: str) -> str:
    return _abs(root, name) if name.endswith(".json") else os.path.join(root, "data", f"{name}.json")


def dataset_snapshot(root: str, names=()) -> dict:
    out = {}
    for name in dict.fromkeys(list(DEFAULT_DATASETS) + list(names)):
        p = _dataset_file(root, name)
        if not os.path.isfile(p):
            if name in names:
                raise FreezeError(f"dataset {name!r} not found: {p}")
            continue
        try:
            with open(p, encoding="utf-8") as f:
                n = len(json.load(f))
        except (ValueError, TypeError):
            n = None
        out[os.path.splitext(os.path.basename(p))[0]] = {"file": _rel(root, p), "sha256": sha256_file(p), "n_parts": n}
    return out


def grid_snapshot(root: str, paths=(), strict: bool = True) -> dict:
    """strict=False (verification): a grid that is no longer valid JSON is still hashed (as bytes), so it shows up as changed."""
    out = {}
    for g in paths:
        p = _abs(root, g)
        if not os.path.isfile(p):
            raise FreezeError(f"grid config not found: {g}")
        try:
            with open(p, encoding="utf-8") as f:
                cfg = json.load(f)
        except ValueError as exc:
            if strict:
                raise FreezeError(f"grid config {g} is not valid JSON: {exc}")
            out[_rel(root, p)] = {"sha256": sha256_file(p), "dataset": None, "rows": []}
            continue
        rows = [[r.get("id"), r.get("system")] for r in cfg.get("rows", []) if isinstance(r, dict)] if isinstance(cfg, dict) else []
        out[_rel(root, p)] = {"sha256": sha256_json_file(p), "dataset": cfg.get("dataset") if isinstance(cfg, dict) else None, "rows": rows}
    return out


def extra_snapshot(root: str, paths=()) -> dict:
    out = {}
    for e in paths:
        p = _abs(root, e)
        if not os.path.isfile(p):
            raise FreezeError(f"extra file not found: {e}")
        out[_rel(root, p)] = {"sha256": sha256_json_file(p)}
    return out


def git_info(root: str) -> dict:
    def run(*args):
        try:
            r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    commit = run("rev-parse", "HEAD")
    if not commit:
        return {"commit": None, "dirty": None, "tags": []}
    status = run("status", "--porcelain")
    tags = run("tag", "--points-at", "HEAD")
    return {"commit": commit, "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None, "tags": tags.split() if tags else []}


def parse_pytest_summary(text: str) -> dict:
    """The last non-empty line of `pytest -q`, e.g. '1464 passed in 128.74s (0:02:08)' or '2 failed, 1462 passed in 130s'."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    last = lines[-1] if lines else ""

    def count(word):
        m = re.search(rf"(\d+) {word}", last)
        return int(m.group(1)) if m else 0
    return {"passed": count("passed"), "failed": count("failed") + count("error"), "summary": last.strip("= ").strip()}


def run_tests(root: str) -> dict:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, capture_output=True, text=True)
    return dict(parse_pytest_summary(r.stdout), command="python -m pytest -q", returncode=r.returncode)


# ---------------------------------------------------------------- fingerprint, manifest
def fingerprint(root: str, grids=(), extras=(), datasets=(), cfg=None, strict: bool = True) -> dict:
    """What is compared: files, config, datasets, grids, extras."""
    return {"files": scan_files(root), "config": config_snapshot(cfg), "datasets": dataset_snapshot(root, datasets),
            "grids": grid_snapshot(root, grids, strict), "extras": extra_snapshot(root, extras)}


def manifest_id(fp: dict) -> str:
    return hashlib.sha256(canonical({k: fp[k] for k in ("files", "config", "datasets", "grids", "extras")}).encode("utf-8")).hexdigest()[:12]


def create(root: str, path: str = None, grids=(), extras=(), datasets=(), note: str = "", name: str = "", tests: dict = None,
           force: bool = False, cfg=None, now: str = None) -> dict:
    """Writes the manifest (default <root>/FREEZE.json) and returns it. Refuses to overwrite one without force; refuses if `tests` has failures."""
    path = path or os.path.join(root, MANIFEST_NAME)
    if os.path.exists(path) and not force:
        raise FreezeError(f"{path} exists: a freeze is created once. To replace it on purpose use --force and say why in --note; to compare "
                          "the current state with it use `verify`.")
    if tests and tests.get("failed"):
        raise FreezeError(f"the tests have {tests['failed']} failure(s): freeze a state whose tests pass")
    if not grids:
        raise FreezeError("no grid config: give the ablation config(s) of the final runs with --grid (the list of systems is part of the freeze)")
    fp = fingerprint(root, grids, extras, datasets, cfg)
    replaces = None
    if os.path.exists(path):
        try:
            replaces = load(path)["id"]
        except FreezeError:
            replaces = "unreadable"
    man = {"schema": SCHEMA, "id": manifest_id(fp), "name": name or None, "note": note or None, "replaces": replaces,
           "created_utc": now or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "git": git_info(root), "python": sys.version.split()[0], "platform": platform.platform(),
           "tests": tests, "prices_not_compared": price_snapshot(cfg), **fp}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    return man


def load(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            man = json.load(f)
    except FileNotFoundError:
        raise FreezeError(f"{path} not found: create it with `python scripts/freeze.py create --grid <config>`")
    except ValueError as exc:
        raise FreezeError(f"{path} is not valid JSON: {exc}")
    missing = [k for k in ("schema", "id", "files", "config", "datasets", "grids", "extras") if k not in man]
    if missing or man.get("schema") != SCHEMA:
        raise FreezeError(f"{path} is not a freeze manifest of schema {SCHEMA} (missing: {', '.join(missing) or 'none'})")
    return man


# ---------------------------------------------------------------- comparison
def _diff_map(a: dict, b: dict, key=lambda v: v.get("sha256")):
    out = []
    for k in sorted(set(a) | set(b)):
        if k not in b:
            out.append((k, "removed"))
        elif k not in a:
            out.append((k, "added"))
        elif key(a[k]) != key(b[k]):
            out.append((k, "changed"))
    return out


def compare(frozen: dict, now: dict) -> dict:
    """`frozen` and `now` are manifests or fingerprints. Returns the differences, by part:
    behaviour / analysis: [(file, changed|added|removed)], config: [(key, frozen value, now value)], datasets / grids / extras:
    [(name, changed|added|removed)]; and 'status'."""
    files = _diff_map(frozen["files"], now["files"])
    kind = lambda f: (now["files"].get(f) or frozen["files"].get(f))["kind"]
    out = {"behaviour": [(f, w) for f, w in files if kind(f) == "behaviour"], "analysis": [(f, w) for f, w in files if kind(f) == "analysis"],
           "config": [(k, frozen["config"].get(k), now["config"].get(k)) for k in sorted(set(frozen["config"]) | set(now["config"]))
                      if frozen["config"].get(k) != now["config"].get(k)],
           "datasets": _diff_map(frozen["datasets"], now["datasets"]), "grids": _diff_map(frozen["grids"], now["grids"]),
           "extras": _diff_map(frozen["extras"], now["extras"])}
    if any(out[k] for k in ("behaviour", "config", "grids", "extras")) or any(w != "added" for _, w in out["datasets"]):
        out["status"] = "drift"
    elif out["analysis"]:
        out["status"] = "analysis-changed"
    else:
        out["status"] = "ok"
    return out


def verify(root: str, path: str = None, cfg=None) -> dict:
    """Recomputes the fingerprint of what the manifest lists (same grids, extras and datasets) and compares. A grid, extra or dataset that has
    disappeared is reported as removed. Raises FreezeError if the manifest is missing or unreadable."""
    path = path or os.path.join(root, MANIFEST_NAME)
    man = load(path)
    there = lambda rel: os.path.isfile(_abs(root, rel))
    now = fingerprint(root, [g for g in man["grids"] if there(g)], [e for e in man["extras"] if there(e)],
                      [d["file"] for n, d in man["datasets"].items() if n not in DEFAULT_DATASETS and there(d["file"])], cfg, strict=False)
    res = compare(man, now)
    res.update(id=man["id"], manifest=path, created_utc=man.get("created_utc"))
    return res


def status(root: str, path: str = None, cfg=None) -> dict:
    """The freeze status of this working tree, for run.json: {'status': none|ok|analysis-changed|drift|unreadable, 'id', 'manifest', 'n_behaviour',
    'files' (first ten behaviour/config/dataset/grid/extra differences)}. Never raises."""
    path = path or os.path.join(root, MANIFEST_NAME)
    shown = _rel(root, path)
    if not os.path.isfile(path):
        return {"status": "none", "id": None, "manifest": shown}
    try:
        res = verify(root, path, cfg)
    except Exception as exc:                                   # a run must never fail because of the freeze check
        return {"status": "unreadable", "id": None, "manifest": shown, "why": str(exc)[:200]}
    what = [f"{w} {f}" for f, w in res["behaviour"]] + [f"config {k}" for k, _, _ in res["config"]] + \
           [f"{part} {n} {w}" for part in ("datasets", "grids", "extras") for n, w in res[part]]
    return {"status": res["status"], "id": res["id"], "manifest": shown, "n_differences": len(what), "differences": what[:10]}


# ---------------------------------------------------------------- text
def format_comparison(res: dict, limit: int = 40) -> str:
    L = [f"Freeze {res.get('id', '')}: {res['status'].upper()}"]
    sections = [("behaviour files", [f"{w:8s} {f}" for f, w in res["behaviour"]]),
                ("config", [f"{k}: frozen {a!r}, now {b!r}" for k, a, b in res["config"]]),
                ("datasets", [f"{w:8s} {n}" for n, w in res["datasets"]]), ("grids", [f"{w:8s} {n}" for n, w in res["grids"]]),
                ("extras", [f"{w:8s} {n}" for n, w in res["extras"]]), ("analysis files (allowed)", [f"{w:8s} {f}" for f, w in res["analysis"]])]
    for title, items in sections:
        if items:
            L.append(f"  {title}: {len(items)}")
            L += [f"    {s}" for s in items[:limit]]
            if len(items) > limit:
                L.append(f"    ... and {len(items) - limit} more")
    return "\n".join(L)


def format_manifest(man: dict) -> str:
    kinds = {}
    for v in man["files"].values():
        kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
    g = man.get("git") or {}
    L = [f"Freeze {man['id']}" + (f" ({man['name']})" if man.get("name") else "") + f", created {man.get('created_utc')}",
         f"  git: {(g.get('commit') or 'n/a')[:10]}{' (working tree had uncommitted changes)' if g.get('dirty') else ''}"
         + (f", tags {', '.join(g['tags'])}" if g.get("tags") else ""),
         f"  files: {len(man['files'])} ({kinds.get('behaviour', 0)} behaviour, {kinds.get('analysis', 0)} analysis)",
         f"  models: " + ", ".join(f"{r} {m}" for r, m in sorted(man["config"].get("MODELS", {}).items())),
         "  datasets: " + (", ".join(f"{n} ({d['n_parts']} parts)" for n, d in man["datasets"].items()) or "none")]
    for gname, g2 in man["grids"].items():
        L.append(f"  grid {gname}: dataset {g2.get('dataset')}, rows " + ", ".join(f"{i}={s}" for i, s in g2.get("rows", [])))
    for e in man["extras"]:
        L.append(f"  extra: {e}")
    t = man.get("tests")
    L.append("  tests: " + (f"{t['passed']} passed, {t['failed']} failed" if t else "not run at freeze time (use --run-tests)"))
    if man.get("note"):
        L.append(f"  note: {man['note']}")
    return "\n".join(L)
