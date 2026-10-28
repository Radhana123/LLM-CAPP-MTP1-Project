"""Table 4 of the report (the ablation grid A0-A9, sections 7.3-7.4), regenerated from finished runs by one command (Step 22).

    python scripts/make_tables.py status   configs/ablation_template.json     which runs exist, which are usable
    python scripts/make_tables.py commands configs/ablation_template.json     the run_eval lines (PowerShell) for the missing runs
    python scripts/make_tables.py build    configs/ablation_template.json     Table 4, paired comparisons, per-part CSV, figures, manifest

The CONFIG (JSON) is the experiment definition: dataset, repeats, the LLMs (a model id and a run tag each) and one line per row of
Table 4 (id, label, the system name of scripts/run_eval.py, the row it adds one component to). Nothing is guessed from folder names:
a run counts for a row only if its records and run.json say that it is that system, on that dataset, with that model.

What is reported (the metrics of Table 3, from the same records and the same functions as scripts/run_eval.py):
  M1  first-pass validity: share of the parsed raw LLM routes that pass the strict verifier
  M2  final validity (share of finals that are strict-valid; the Route Builder fallback makes this ~1) and the provenance half of M2:
      "LLM-final" = share of final routes that are LLM routes (raw or repaired), not the Route Builder's. That is what an unreliable
      model really costs, so it is the validity figure that moves between rows
  M4  plan similarity of the final route (LCS ratio) to the nearest reference: expert routes if the parts have them, otherwise the
      Route Builder variants ("proxy_rb": agreement with the rules, NOT with an expert; the table says which)
  M5  repair minimality: share of the LLM's operations that survive whole-route replacement or localized repair (higher = more minimal)
  M6  plan quality: mean Efficiency gap of the final route to the best Route Builder variant, in points (lower is better, negative
      = better than every Route Builder variant)
  M7  inference cost: API calls per part (all calls: samples, second step, re-prompts, cascade) and list-price USD per part
  A cell is "mean ± SD" over the repeats (SD across repeats, M8) and a plain value when there is one repeat. A missing run is "–".

Statistics (section 7.4). Each row is compared with the row it adds a component to (config "parent"), PAIRED over parts:
  * continuous metrics (M1, M4, M6, LLM-final share, M7 calls): per part the mean over its repeats, difference row - parent,
    percentile bootstrap 95% interval over PARTS (a part's repeats stay together; seed in the config)
  * validity, a paired binary outcome: McNemar's exact test, on (a) the first-pass strict validity of the raw samples (the same
    sample slot of the same part and repeat in both runs, parsed in both) and (b) LLM-final per (part, repeat). With several repeats
    these pairs are clustered by part, so the p-values are approximate: the bootstrap over parts is the primary evidence
  * the same comparison for every LLM, and a consistency table: the claim "A helps" needs the same answer for all of them
  Intervals and p-values are not corrected for the number of comparisons made.
"""
import copy
import csv
import hashlib
import json
import math
import os
import random
import re

from capp import apierrors, cost as COST, dataset, metrics as M, plannerspec, summary as S, systems

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 7
RESAMPLES = 1000
DASH = "–"
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")

TOP_KEYS = {"name", "description", "dataset", "limit", "offset", "repeats", "results_dir", "pool", "resamples", "seed", "shared_tag",
            "models", "rows"}
MODEL_KEYS = {"id", "label", "model", "tag"}
ROW_KEYS = {"id", "label", "system", "parent", "repeats", "note"}

METRICS = ("M1", "M2", "M2p", "M4", "M5", "M6", "M7")
DECIMALS = {"M1": 3, "M2": 3, "M2p": 3, "M4": 3, "M5": 3, "M6": 2, "M7": 2, "usd": 5, "tokens": 0, "M5_edit": 2}
# paired continuous metrics: name -> (direction: +1 higher is better, -1 lower is better, words for better / worse)
PAIRED = {"M1": (+1, "better", "worse"), "M2p": (+1, "better", "worse"), "M4": (+1, "better", "worse"),
          "M6": (-1, "better", "worse"), "M7": (-1, "cheaper", "costlier")}
PAIRED_LABELS = {"M1": "M1 first-pass valid", "M2p": "M2 LLM-final share", "M4": "M4 plan similarity (LCS)",
                 "M6": "M6 efficiency gap (pts)", "M7": "M7 API calls / part"}
CONSISTENCY_METRICS = ("M1", "M2p", "M4", "M6")
UNCLEAR = "no clear difference"


class TableError(ValueError):
    """A problem with the config or with what the user asked for (not with a run: those are reported as issues)."""


# ---------------------------------------------------------------- config
def _text(x, what, pattern=None):
    if not isinstance(x, str) or not x.strip():
        raise TableError(f"{what}: expected a non-empty text")
    if pattern is not None and not pattern.match(x):
        raise TableError(f"{what}: {x!r} may contain only letters, digits, '_', '.', '-' (it becomes part of a folder name)")
    return x


def _int(x, what, lo):
    if isinstance(x, bool) or not isinstance(x, int) or x < lo:
        raise TableError(f"{what}: expected a whole number >= {lo}, got {x!r}")
    return x


def validate_config(cfg) -> dict:
    """The normalised copy of a config; raises TableError naming the first problem."""
    if not isinstance(cfg, dict):
        raise TableError("config: expected a JSON object")
    unknown = sorted(set(cfg) - TOP_KEYS)
    if unknown:
        raise TableError(f"config: unknown key(s) {unknown}; allowed: {sorted(TOP_KEYS)}")
    out = {
        "name": _text(cfg.get("name"), "name", ID_RE),
        "description": cfg.get("description", "") if isinstance(cfg.get("description", ""), str) else "",
        "dataset": _text(cfg.get("dataset", "dev_v2"), "dataset"),
        "limit": None if cfg.get("limit") is None else _int(cfg["limit"], "limit", 1),
        "offset": _int(cfg.get("offset", 0), "offset", 0),
        "repeats": _int(cfg.get("repeats", 5), "repeats", 1),
        "results_dir": _text(cfg.get("results_dir", "results"), "results_dir"),
        "pool": _text(cfg.get("pool", "dev_v2"), "pool"),
        "resamples": _int(cfg.get("resamples", RESAMPLES), "resamples", 100),
        "seed": _int(cfg.get("seed", SEED), "seed", 0),
        "shared_tag": None if cfg.get("shared_tag") is None else _text(cfg["shared_tag"], "shared_tag", ID_RE),
    }
    models = cfg.get("models")
    if not isinstance(models, list) or not models:
        raise TableError("models: expected a list with at least one LLM ({id, label, model, tag})")
    seen_id, seen_tag, seen_model = set(), set(), set()
    out["models"] = []
    for i, m in enumerate(models):
        if not isinstance(m, dict):
            raise TableError(f"models[{i}]: expected an object")
        bad = sorted(set(m) - MODEL_KEYS)
        if bad:
            raise TableError(f"models[{i}]: unknown key(s) {bad}; allowed: {sorted(MODEL_KEYS)}")
        mid = _text(m.get("id"), f"models[{i}].id", ID_RE)
        item = {"id": mid, "label": _text(m.get("label", mid), f"models[{i}].label"),
                "model": _text(m.get("model"), f"models[{i}].model (the model id the run was made with)"),
                "tag": _text(m.get("tag"), f"models[{i}].tag", ID_RE)}
        for what, v, seen in (("id", mid, seen_id), ("tag", item["tag"], seen_tag), ("model", item["model"], seen_model)):
            if v in seen:
                raise TableError(f"models[{i}]: {what} {v!r} is used twice")
            seen.add(v)
        out["models"].append(item)

    rows = cfg.get("rows")
    if not isinstance(rows, list) or not rows:
        raise TableError("rows: expected a list of rows (A0, A1, ...)")
    out["rows"] = []
    ids, systems_seen = [], {}
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise TableError(f"rows[{i}]: expected an object")
        bad = sorted(set(r) - ROW_KEYS)
        if bad:
            raise TableError(f"rows[{i}]: unknown key(s) {bad}; allowed: {sorted(ROW_KEYS)}")
        rid = _text(r.get("id"), f"rows[{i}].id", ID_RE)
        if rid in ids:
            raise TableError(f"rows[{i}]: id {rid!r} is used twice")
        system = r.get("system")
        item = {"id": rid, "label": _text(r.get("label"), f"rows[{i}] ({rid}).label"), "system": None, "parent": None,
                "repeats": out["repeats"], "note": r.get("note", "") if isinstance(r.get("note", ""), str) else "",
                "built": system is not None, "independent": False}
        if r.get("repeats") is not None:
            item["repeats"] = _int(r["repeats"], f"rows[{i}] ({rid}).repeats", 1)
        parent = r.get("parent")
        if parent is not None:
            if parent not in ids:
                raise TableError(f"rows[{i}] ({rid}).parent: {parent!r} must be the id of an EARLIER row (rows: {ids or 'none yet'})")
            item["parent"] = parent
        if system is not None:
            _text(system, f"rows[{i}] ({rid}).system")
            try:
                spec = systems.parse_system(system)
                pspec = systems.planner_spec(system)
            except ValueError as exc:
                raise TableError(f"rows[{i}] ({rid}).system: {exc}") from None
            if spec is None:
                raise TableError(f"rows[{i}] ({rid}).system: 'rb' is the reference every metric is measured against, not a row of Table 4")
            if pspec.few_shot and dataset.is_sealed(out["pool"]):
                raise TableError(f"pool {out['pool']!r} is sealed: few-shot examples (row {rid}) must come from the development data only")
            if system in systems_seen:
                raise TableError(f"rows[{i}] ({rid}): system {system!r} is already the system of row {systems_seen[system]} "
                                 "(two rows would read the same run folder)")
            systems_seen[system] = rid
            item["system"] = system
            item["independent"] = bool(pspec.cascade or pspec.model != "primary")
        ids.append(rid)
        out["rows"].append(item)
    if any(r["independent"] for r in out["rows"]) and not out["shared_tag"]:
        names = [r["id"] for r in out["rows"] if r["independent"]]
        raise TableError(f"shared_tag: rows {names} choose their own models (+ms, +ml, +cas1, +cas2), so they do not depend on the LLM "
                         "of a column; give them a run tag with \"shared_tag\"")
    return out


def load_config(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except OSError as exc:
        raise TableError(f"cannot read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise TableError(f"{path} is not valid JSON: {exc}") from None
    return validate_config(raw)


def config_sha(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def resolve_results_dir(cfg: dict, results_dir=None) -> str:
    d = results_dir or cfg["results_dir"]
    return d if os.path.isabs(d) else os.path.join(ROOT, d)


def run_folder(cfg: dict, row: dict, model: dict, results_dir: str) -> str:
    tag = cfg["shared_tag"] if row["independent"] else model["tag"]
    return os.path.join(results_dir, f"{tag}_{row['system']}")


# ---------------------------------------------------------------- the parts every run must cover
def expected_parts(cfg: dict):
    """(part ids in dataset order, {part_id: fingerprint}, source). The dataset is loaded only to read the parts: no run is made.
    If it cannot be loaded, ids is None and the first run read defines the fingerprints."""
    try:
        parts = dataset.load_dataset(cfg["dataset"])[cfg["offset"]:]
    except (OSError, ValueError, KeyError):
        return None, {}, "unavailable"
    if cfg["limit"]:
        parts = parts[:cfg["limit"]]
    return [p["part_id"] for p in parts], {p["part_id"]: _fingerprint(p) for p in parts}, "dataset"


def _fingerprint(part):
    return (tuple(part.get("features", [])), part.get("material"))


# ---------------------------------------------------------------- reading one run
_REQUIRED = ("system", "part_id", "repeat", "part", "candidates", "final", "info")


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except OSError:
        return None, "missing"
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON ({exc.msg})"


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _new_run(row, model, folder, status, **kw):
    run = {"row": row["id"], "model": model["id"], "system": row["system"], "folder": folder, "status": status, "errors": [],
           "warnings": [], "records": [], "run_info": {}, "repeats": [], "parts": [], "records_sha": None, "expected_repeats": row["repeats"]}
    run.update(kw)
    return run


def load_run(cfg: dict, row: dict, model: dict, results_dir: str, expected: dict, cache: dict = None) -> dict:
    """Read and check the run behind one cell. status: not_built | missing | error (unusable, listed in the issues) | partial (usable,
    with a warning) | ok."""
    if not row["built"]:
        return _new_run(row, model, None, "not_built")
    folder = run_folder(cfg, row, model, results_dir)
    if cache is not None and folder in cache and row["independent"]:
        shared = cache[folder]
        return dict(shared, row=row["id"], model=model["id"])
    run = _read_run(cfg, row, model, folder, expected)
    if cache is not None and row["independent"]:
        cache[folder] = run
    return run


def _read_run(cfg, row, model, folder, expected):
    path = os.path.join(folder, "records.jsonl")
    if not os.path.exists(path):
        return _new_run(row, model, folder, "missing")
    run = _new_run(row, model, folder, "ok")
    errors, warnings = run["errors"], run["warnings"]
    try:
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        run["records_sha"] = _sha(path)
    except (OSError, json.JSONDecodeError) as exc:
        run["status"] = "error"
        errors.append(f"records.jsonl cannot be read: {exc}")
        return run
    if not records:
        run["status"] = "error"
        errors.append("records.jsonl is empty")
        return run
    for i, rec in enumerate(records):
        lacking = [k for k in _REQUIRED if k not in rec]
        if lacking:
            run["status"] = "error"
            errors.append(f"record {i + 1} lacks {lacking}: not a run of this harness")
            return run

    names = sorted({r["system"] for r in records})
    if names != [row["system"]]:
        errors.append(f"the records are of system {names}, the row {row['id']} is {row['system']!r}")
    keys = [(r["part_id"], r["repeat"]) for r in records]
    if len(set(keys)) != len(keys):
        errors.append("a (part, repeat) occurs more than once in records.jsonl")

    info, why = _read_json(os.path.join(folder, "run.json"))
    if info is None:
        warnings.append(f"run.json is {why}: the dataset and the model of this run could not be checked")
    else:
        run["run_info"] = info
        if info.get("system") not in (None, row["system"]):
            errors.append(f"run.json says system {info.get('system')!r}, the row {row['id']} is {row['system']!r}")
        ds = info.get("dataset")
        if ds and os.path.basename(str(ds)) != os.path.basename(cfg["dataset"]):
            errors.append(f"run.json says dataset {ds!r}, the config says {cfg['dataset']!r}")
        elif not ds:
            warnings.append("run.json does not name the dataset")
        if dataset.is_sealed(cfg["dataset"]):
            fz = info.get("freeze")
            if not isinstance(fz, dict):
                warnings.append("run.json has no freeze status: was this run made before capp/freeze.py (scripts/freeze.py)?")
            elif fz.get("status") not in ("ok", "analysis-changed"):
                warnings.append(f"this run on a sealed dataset was made with freeze status {fz.get('status')!r}"
                                + (f" ({'; '.join(fz.get('differences') or [])[:200]})" if fz.get("differences") else "")
                                + ": it is not a run of a frozen configuration")
        if not row["independent"]:
            if info.get("model") and info["model"] != model["model"]:
                errors.append(f"run.json says model {info['model']!r}, the column {model['id']} is {model['model']!r}: "
                              "was CAPP_MODEL_PRIMARY set when the run was made?")
            elif not info.get("model"):
                warnings.append("run.json does not name the model")

    ids, prints, source = expected
    first_seen = {}
    for r in records:
        fp = _fingerprint(r["part"])
        ref = prints.get(r["part_id"]) if ids is not None else first_seen.setdefault(r["part_id"], fp)
        if ref is not None and ref != fp:
            errors.append(f"part {r['part_id']}: its features or material differ from the dataset {cfg['dataset']!r} "
                          "(a different dataset version?)")
            break
    if errors:
        run["status"] = "error"
        return run

    if ids is not None:
        in_scope = [r for r in records if r["part_id"] in set(ids)]
        if len(in_scope) < len(records):
            warnings.append(f"{len(records) - len(in_scope)} record(s) of parts outside the configured set are ignored")
        records = in_scope
        have = {r["part_id"] for r in records}
        missing = [p for p in ids if p not in have]
        if missing:
            warnings.append(f"{len(have)} of {len(ids)} parts (missing: {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''})")
    repeats = sorted({r["repeat"] for r in records})
    want = row["repeats"]
    if len(repeats) < want:
        warnings.append(f"{len(repeats)} of {want} repeats")
    elif len(repeats) > want:
        warnings.append(f"{len(repeats)} repeats, the config asks for {want}: all are used")
    n_parts = len({r["part_id"] for r in records})
    if len(records) != n_parts * len(repeats) and not any("parts (missing" in w or "repeats" in w for w in warnings):
        warnings.append(f"{len(records)} records for {n_parts} parts x {len(repeats)} repeats: some (part, repeat) are missing")
    n_api = apierrors.in_records(records)
    if n_api:
        warnings.append(f"{n_api} API error(s) in the records (rate limit, outage): " + apierrors.rerun_advice(n_api))
    run["records"], run["repeats"], run["parts"] = records, repeats, sorted({r["part_id"] for r in records})
    if not records:
        run["status"] = "error"
        errors.append("no record belongs to the configured parts")
    elif warnings:
        run["status"] = "partial"
    return run


def load_all(cfg: dict, results_dir: str = None) -> dict:
    """{(row id, model id): run} for every cell of the grid, plus the part set used for the checks under key 'expected'."""
    rd = resolve_results_dir(cfg, results_dir)
    expected = expected_parts(cfg)
    cache = {}
    runs = {}
    for row in cfg["rows"]:
        for model in cfg["models"]:
            runs[(row["id"], model["id"])] = load_run(cfg, row, model, rd, expected, cache)
    return {"runs": runs, "expected": expected, "results_dir": rd}


# ---------------------------------------------------------------- the metrics of one run
def api_calls(rec) -> int:
    """All API calls behind one record: the samples (and their second steps), the re-prompts, the attempts a cascade set aside.
    The sum over records / n is summary.summarize_set()['M7']['api_calls_per_part']."""
    own = sum(c["meta"].get("n_calls", 1) for c in rec["candidates"] if c["meta"])
    return own + rec["info"].get("repair_calls", 0) + S._other_attempt_calls([rec])


def _fixed(records):
    return [c for r in records for c in r["candidates"] if (c["replaced"] or c.get("repaired")) and c["llm_steps"]]


def _m5(records):
    cs = _fixed(records)
    if not cs:
        return None, None, 0
    return (M.mean_std([M.survived_share(c["llm_steps"], c["steps"]) for c in cs])[0],
            M.mean_std([M.edit_distance(c["llm_steps"], c["steps"]) for c in cs])[0], len(cs))


def repeat_metrics(records, default_model=None) -> dict:
    """The metrics of ONE repeat (all parts), from summarize_set, the function behind run_eval's summary. default_model is the model
    to which calls without a model id (records older than Step 17) are charged: the model of the column, not whatever
    CAPP_MODEL_PRIMARY says on the day the table is built."""
    s = S.summarize_set(records)
    cost = COST.block(records, default_model)
    m4 = s["M4"].get("expert") or s["M4"].get("proxy_rb") or {}
    m4_source = "expert" if "expert" in s["M4"] else "proxy_rb" if "proxy_rb" in s["M4"] else None
    prov = s["M2"]["provenance_share"]
    m5, m5_edit, n_fixed = _m5(records)
    return {"M1": s["M1"]["strict_valid_of_parsed"], "M2": s["M2"]["final_strict_valid"],
            "M2p": 1.0 - prov.get("route_builder", 0.0),
            "M4": (m4.get("final") or {}).get("lcs_ratio_mean"), "M5": m5, "M5_edit": m5_edit,
            "M6": s["M6"]["efficiency_gap_to_rb_best_mean"], "M7": s["M7"]["api_calls_per_part"],
            "usd": cost.get("usd_per_part"), "tokens": cost.get("tokens_per_part"),
            "_m4_source": m4_source, "_n_fixed": n_fixed}


def _agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "sd": None, "n": 0}
    m, sd = M.mean_std(vals)
    return {"mean": m, "sd": sd if len(vals) > 1 else None, "n": len(vals)}


def cell_stats(records, default_model=None) -> dict:
    """The Table 4 cell of a run: for every metric the mean and the SD over repeats (SD None with a single repeat)."""
    by_repeat = {}
    for r in records:
        by_repeat.setdefault(r["repeat"], []).append(r)
    per = [repeat_metrics(by_repeat[k], default_model) for k in sorted(by_repeat)]
    cell = {m: _agg([p[m] for p in per]) for m in (*METRICS, "usd", "tokens", "M5_edit")}
    sources = sorted({p["_m4_source"] for p in per if p["_m4_source"]})
    cell["m4_source"] = sources[0] if len(sources) == 1 else ("mixed" if sources else None)
    cell["n_repeats"] = len(per)
    cell["n_parts"] = len({r["part_id"] for r in records})
    cell["n_fixed"] = sum(p["_n_fixed"] for p in per)
    return cell


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else sum(xs) / len(xs)


def part_values(records) -> dict:
    """Per part the mean over its repeats of: M1, M2p (LLM-final), M4, M6, M7. The unit of the paired bootstrap."""
    by_part = {}
    for r in records:
        by_part.setdefault(r["part_id"], []).append(r)
    prow = {(x["part_id"], x["repeat"]): x for x in S.part_rows(records)}
    out = {}
    for pid, recs in by_part.items():
        m1, m2p, m4, m6, m7 = [], [], [], [], []
        for rec in recs:
            parsed = [c["first_pass"]["valid"] for c in rec["candidates"] if c["llm_ok"] and c["first_pass"] is not None]
            m1.append(None if not parsed else sum(1 for v in parsed if v) / len(parsed))
            m2p.append(0.0 if rec["final"]["provenance"] == "route_builder" else 1.0)
            row = prow[(pid, rec["repeat"])]
            m4.append(row["m4_lcs_ratio"])
            m6.append(row["eff_gap_to_rb_best"])
            m7.append(api_calls(rec))
        out[pid] = {"M1": _mean(m1), "M2p": _mean(m2p), "M4": _mean(m4), "M6": _mean(m6), "M7": _mean(m7), "n_repeats": len(recs)}
    return out


def binary_outcomes(records) -> dict:
    """The two paired binary outcomes: 'sample' = {(part, repeat, sample_id): first-pass strict valid} for the parsed raw routes,
    'final' = {(part, repeat): the final route is an LLM route}."""
    sample, final = {}, {}
    for rec in records:
        final[(rec["part_id"], rec["repeat"])] = rec["final"]["provenance"] != "route_builder"
        for c in rec["candidates"]:
            if c["llm_ok"] and c["first_pass"] is not None:
                sample[(rec["part_id"], rec["repeat"], c["sample_id"])] = bool(c["first_pass"]["valid"])
    return {"sample": sample, "final": final}


# ---------------------------------------------------------------- statistics
def _percentile_ci(values, lo=0.025, hi=0.975):
    v = sorted(values)
    n = len(v)
    return v[int(lo * (n - 1))], v[int(round(hi * (n - 1)))]


def bootstrap_mean(diffs: list, resamples: int, seed_key: str) -> dict:
    """Mean of paired differences and its percentile 95% interval over resampled parts. Deterministic for a given key."""
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    mean = sum(diffs) / n
    if n < 2:
        return {"n": n, "mean": mean, "lo": None, "hi": None}
    rng = random.Random(seed_key)
    means = [sum(rng.choices(diffs, k=n)) / n for _ in range(resamples)]
    lo, hi = _percentile_ci(means)
    return {"n": n, "mean": mean, "lo": lo, "hi": hi}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b (only the row succeeds) and c (only the parent). 1.0 if b + c = 0."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2 ** n)


def paired_binary(row_out: dict, parent_out: dict) -> dict:
    keys = sorted(set(row_out) & set(parent_out))
    both = sum(1 for k in keys if row_out[k] and parent_out[k])
    row_only = sum(1 for k in keys if row_out[k] and not parent_out[k])
    parent_only = sum(1 for k in keys if parent_out[k] and not row_out[k])
    neither = len(keys) - both - row_only - parent_only
    p = mcnemar_exact(row_only, parent_only) if keys else None
    verdict = "n/a" if not keys else (("better" if row_only > parent_only else "worse") if p < 0.05 else UNCLEAR)
    return {"n": len(keys), "both": both, "row_only": row_only, "parent_only": parent_only, "neither": neither, "p": p, "verdict": verdict}


def verdict_of(metric: str, ci: dict) -> str:
    """'better' / 'worse' (or 'cheaper' / 'costlier' for M7) only when the 95% interval excludes 0; otherwise 'no clear difference'."""
    direction, good, bad = PAIRED[metric]
    if ci["mean"] is None:
        return "n/a"
    if ci["lo"] is None:
        return UNCLEAR
    if ci["lo"] > 0 or ci["hi"] < 0:
        return good if (ci["mean"] > 0) == (direction > 0) else bad
    return UNCLEAR


def paired_metric(cfg, model_id, row_id, parent_id, metric, row_vals, parent_vals) -> dict:
    common = sorted(p for p in set(row_vals) & set(parent_vals) if row_vals[p][metric] is not None and parent_vals[p][metric] is not None)
    diffs = [row_vals[p][metric] - parent_vals[p][metric] for p in common]
    ci = bootstrap_mean(diffs, cfg["resamples"], f"{cfg['seed']}:{model_id}:{row_id}:{parent_id}:{metric}")
    return {"model": model_id, "row": row_id, "parent": parent_id, "metric": metric, **ci, "verdict": verdict_of(metric, ci)}


# ---------------------------------------------------------------- the whole grid
def _issue(level, message, model=None, row=None):
    return {"level": level, "model": model, "row": row, "message": message}


def analyse(cfg: dict, results_dir: str = None) -> dict:
    """Reads every run of the grid and computes the cells and the comparisons. No file is written."""
    loaded = load_all(cfg, results_dir)
    runs = loaded["runs"]
    issues = []
    if len(cfg["models"]) < 3:
        issues.append(_issue("warning", f"{len(cfg['models'])} LLM(s) in the config: the report requires the full grid with at least 3, "
                                        "so that no conclusion rests on one model"))
    if dataset.is_sealed(cfg["dataset"]):
        issues.append(_issue("warning", f"dataset {cfg['dataset']!r} is sealed (final runs only): no setting may change once its results are seen"))
    if dataset.is_sealed(cfg["dataset"]):
        ids = sorted({(run.get("run_info") or {}).get("freeze", {}).get("id") for run in runs.values()
                      if isinstance((run.get("run_info") or {}).get("freeze"), dict) and (run["run_info"]["freeze"].get("id"))})
        if len(ids) > 1:
            issues.append(_issue("warning", f"the runs were made under {len(ids)} different freeze manifests ({', '.join(ids)}): a final "
                                            "grid should come from one freeze (python scripts/freeze.py diff A B shows what changed between two)"))
    if loaded["expected"][2] == "unavailable":
        issues.append(_issue("warning", f"dataset {cfg['dataset']!r} could not be loaded: the parts and their features were not checked "
                                        "against it (the first run read defines them)"))
    for (rid, mid), run in runs.items():
        if run["status"] == "missing":
            issues.append(_issue("missing", f"no run: {run['folder']}", mid, rid))
        for e in run["errors"]:
            issues.append(_issue("error", f"{run['folder']}: {e}", mid, rid))
        for w in run["warnings"]:
            issues.append(_issue("warning", w, mid, rid))
    # a shared (model independent) run appears in every column: report it once
    independent = {r["id"] for r in cfg["rows"] if r["independent"]}
    seen, unique = set(), []
    for it in issues:
        if it["row"] in independent:
            if (it["level"], it["message"]) in seen:
                continue
            seen.add((it["level"], it["message"]))
            it = dict(it, model=None)
        unique.append(it)
    issues = unique

    done = {}
    for key, run in runs.items():
        if run["status"] not in ("ok", "partial"):
            continue
        if run["folder"] not in done:                      # a shared run is analysed once for all the columns that show it
            try:
                done[run["folder"]] = {"cell": cell_stats(run["records"], _column_model(cfg, key, run)), "part_values": part_values(run["records"]),
                                       "binary": binary_outcomes(run["records"])}
            except (KeyError, TypeError, ValueError) as exc:   # a record that has the fields but not the content of a harness record
                done[run["folder"]] = f"{type(exc).__name__}: {exc}"
                issues.append(_issue("error", f"{run['folder']}: the records could not be analysed ({done[run['folder']]})",
                                     None if run["row"] in independent else key[1], key[0]))
        if isinstance(done[run["folder"]], dict):
            run.update(done[run["folder"]])
        else:
            run["status"] = "error"
            run["errors"] = run["errors"] + [f"the records could not be analysed ({done[run['folder']]})"]
    usable = {k: r for k, r in runs.items() if r["status"] in ("ok", "partial")}

    paired, mcnemar = [], []
    row_by_id = {r["id"]: r for r in cfg["rows"]}
    for row in cfg["rows"]:
        if not row["parent"]:
            continue
        for model in cfg["models"]:
            a, b = usable.get((row["id"], model["id"])), usable.get((row["parent"], model["id"]))
            if a is None or b is None:
                continue
            for metric in PAIRED:
                paired.append(paired_metric(cfg, model["id"], row["id"], row["parent"], metric, a["part_values"], b["part_values"]))
            for kind in ("sample", "final"):
                mcnemar.append({"model": model["id"], "row": row["id"], "parent": row["parent"], "outcome": kind,
                                **paired_binary(a["binary"][kind], b["binary"][kind])})
    return {"cfg": cfg, "runs": runs, "usable": usable, "expected": loaded["expected"], "results_dir": loaded["results_dir"],
            "issues": issues, "paired": paired, "mcnemar": mcnemar, "consistency": consistency(cfg, paired),
            "row_by_id": row_by_id}


def _column_model(cfg, key, run):
    """The model id of the calls in a run: the column's model, or what run.json says for a row that chooses its own models."""
    if run["row"] in {r["id"] for r in cfg["rows"] if r["independent"]}:
        return (run.get("run_info") or {}).get("model")
    return next(m["model"] for m in cfg["models"] if m["id"] == key[1])


def consistency(cfg: dict, paired: list) -> list:
    """For every comparison and quality metric: what each LLM says, and whether they agree (section 7.4: at least three LLMs)."""
    out = []
    for row in cfg["rows"]:
        if not row["parent"]:
            continue
        for metric in CONSISTENCY_METRICS:
            per = {p["model"]: p for p in paired if p["row"] == row["id"] and p["metric"] == metric and p["mean"] is not None}
            if not per:
                continue
            verdicts = {m["id"]: per[m["id"]]["verdict"] if m["id"] in per else "no data" for m in cfg["models"]}
            _, good, bad = PAIRED[metric]
            n = len(per)
            k_good, k_bad = list(verdicts.values()).count(good), list(verdicts.values()).count(bad)
            if n < 2:
                label = "one LLM only"
            elif k_good == n:
                label = f"{good} in all {n} LLMs"
            elif k_bad == n:
                label = f"{bad} in all {n} LLMs"
            elif k_good and k_bad:
                label = "mixed: the LLMs disagree"
            elif k_good:
                label = f"{good} in {k_good} of {n} LLMs, no clear difference in the rest"
            elif k_bad:
                label = f"{bad} in {k_bad} of {n} LLMs, no clear difference in the rest"
            else:
                label = f"no clear difference in any of the {n} LLMs"
            if n < len(cfg["models"]):
                label += f" ({n} of {len(cfg['models'])} LLMs have both runs)"
            out.append({"row": row["id"], "parent": row["parent"], "metric": metric, "verdicts": verdicts, "n_models": n,
                        "n_good": k_good, "n_bad": k_bad, "label": label})
    return out


# ---------------------------------------------------------------- text output
def _fmt(x, nd):
    return "n/a" if x is None else f"{x:.{nd}f}"


def cell_text(stat, metric, latex=False) -> str:
    if stat is None or stat["mean"] is None:
        return DASH
    nd = DECIMALS[metric]
    if stat["sd"] is None:
        return f"{stat['mean']:.{nd}f}"
    pm = r" \pm " if latex else " ± "
    return f"{stat['mean']:.{nd}f}{pm}{stat['sd']:.{nd}f}"


def row_marks(run) -> str:
    return "†" if run and run["status"] == "partial" else ""


def table_rows(res: dict, model_id: str) -> list:
    """One dict per row of Table 4 for a model: the texts of the cells, ready for Markdown, CSV and LaTeX."""
    cfg = res["cfg"]
    out = []
    for row in cfg["rows"]:
        run = res["runs"][(row["id"], model_id)]
        cell = run.get("cell") if run["status"] in ("ok", "partial") else None
        entry = {"id": row["id"], "label": row["label"], "system": row["system"], "status": run["status"], "mark": row_marks(run),
                 "note": row["note"], "cell": cell, "run": run}
        out.append(entry)
    return out


COLUMNS = (("M1", "M1 first-pass valid"), ("M2", "M2 final valid"), ("M2p", "M2 LLM-final"), ("M4", "M4 similarity"),
           ("M5", "M5 ops kept"), ("M6", "M6 gap (pts)"), ("M7", "M7 calls/part"), ("usd", "M7 USD/part"))


def _usd_text(cell, latex=False):
    stat = cell["usd"]
    return "n/a" if stat["mean"] is None else cell_text(stat, "usd", latex)


def cell_texts(entry, latex=False) -> list:
    cell = entry["cell"]
    if cell is None:
        if entry["status"] == "not_built":
            return ["not built"] + [DASH] * (len(COLUMNS) - 1)
        return [DASH] * len(COLUMNS)
    out = []
    for key, _ in COLUMNS:
        out.append(_usd_text(cell, latex) if key == "usd" else cell_text(cell[key], key, latex))
    return out


def _m4_note(res):
    sources = {r["cell"]["m4_source"] for r in res["usable"].values() if r.get("cell")}
    if not sources or sources == {None}:
        return "M4: no reference route available."
    if sources == {"proxy_rb"}:
        return ("M4 is measured against the Route Builder variants (\"proxy_rb\"): it says how close the plan is to the rule-based "
                "plans, NOT to an expert's plan. Expert references exist only for the benchmark (bench_v1).")
    if sources == {"expert"}:
        return "M4 is measured against the expert reference routes."
    return f"M4 references differ between runs ({', '.join(sorted(s for s in sources if s))}): do not compare M4 across them."


def footnotes(res: dict, model_id: str) -> list:
    """What the reader of a model's table needs besides the numbers: why a cell is missing or marked, and the notes of the config."""
    notes, extra = [], []
    for e in table_rows(res, model_id):
        run = e["run"]
        if run["status"] == "partial":
            notes.append(f"† {e['id']}: " + "; ".join(run["warnings"]))
        elif run["status"] == "error":
            notes.append(f"{e['id']}: excluded, " + "; ".join(run["errors"]))
        elif run["status"] == "missing":
            notes.append(f"{e['id']}: no run yet")
        elif run["status"] == "not_built":
            notes.append(f"{e['id']}: not built" + (f" ({e['note']})" if e["note"] and "not built" not in e["note"].lower() else ""))
        if e["note"] and run["status"] != "not_built":
            extra.append(f"{e['id']}: {e['note']}")
        if res["row_by_id"][e["id"]]["independent"] and e["cell"] is not None:
            extra.append(f"{e['id']}: one run for every LLM (the row chooses its own models), so this cell is the same in every table.")
    return notes + extra


METRIC_LEGEND = [
    "M1 first-pass valid: share of the parsed raw LLM routes that pass the strict verifier.",
    "M2 final valid: share of final routes that are strict-valid (the Route Builder fallback keeps this near 1). "
    "M2 LLM-final: share of final routes that are LLM routes, raw or repaired, not the Route Builder's.",
    "M5 ops kept: share of the LLM's operations that survive replacement or repair (higher = more minimal); "
    "– where nothing had to be fixed.",
    "M6 gap: mean Efficiency of the final route below the best Route Builder variant, in points (lower is better, negative = better).",
    "M7: API calls per part (all calls) and list-price USD per part (n/a: no price for a model, see capp/config.py).",
]


def render_markdown(res: dict) -> str:
    cfg = res["cfg"]
    out = [f"# Table 4: ablation grid, {cfg['name']}", ""]
    if cfg["description"]:
        out += [cfg["description"], ""]
    out += [f"Dataset {cfg['dataset']}" + (f" (parts {cfg['offset'] + 1}-{cfg['offset'] + cfg['limit']})" if cfg["limit"] else "")
            + f", {cfg['repeats']} repeats by default. A cell is mean ± SD over the repeats; a plain value is one repeat; – = no data.", ""]
    out += [f"- {line}" for line in METRIC_LEGEND] + [f"- {_m4_note(res)}", ""]
    for model in cfg["models"]:
        out += [f"## {model['label']} (`{model['model']}`)", ""]
        out.append("| ID | Configuration | System | " + " | ".join(h for _, h in COLUMNS) + " |")
        out.append("|---|---|---|" + "---|" * len(COLUMNS))
        for e in table_rows(res, model["id"]):
            out.append(f"| {e['id']}{e['mark']} | {e['label']} | {('`' + e['system'] + '`') if e['system'] else DASH} | "
                       + " | ".join(cell_texts(e)) + " |")
        notes = footnotes(res, model["id"])
        if notes:
            out += [""] + [f"- {n}" for n in notes]
        out.append("")
    return "\n".join(out)


_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "$": r"\$",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _tex(s: str) -> str:
    return "".join(_TEX.get(ch, ch) for ch in s)


def render_latex(res: dict) -> str:
    """A tabular per LLM (booktabs: \\usepackage{booktabs}). A fragment, to be \\input in the report."""
    cfg = res["cfg"]
    out = [f"% Table 4, {cfg['name']}: generated by scripts/make_tables.py, do not edit by hand"]
    for model in cfg["models"]:
        out += ["", f"% {model['label']} ({model['model']})", r"\begin{tabular}{ll" + "r" * len(COLUMNS) + "}", r"\toprule",
                "ID & Configuration & " + " & ".join(_tex(h) for _, h in COLUMNS) + r" \\", r"\midrule"]
        for e in table_rows(res, model["id"]):
            cells = [f"${c}$" if r"\pm" in c else ("--" if c == DASH else _tex(c)) for c in cell_texts(e, latex=True)]
            ident = _tex(e["id"]) + (r"$^\dagger$" if e["mark"] else "")
            out.append(f"{ident} & {_tex(e['label'])} & " + " & ".join(cells) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def table_csv_rows(res: dict) -> list:
    cfg = res["cfg"]
    rows = []
    for model in cfg["models"]:
        for e in table_rows(res, model["id"]):
            cell, run = e["cell"], e["run"]
            r = {"model": model["id"], "model_id": model["model"], "row": e["id"], "label": e["label"], "system": e["system"] or "",
                 "status": e["status"], "parent": res["row_by_id"][e["id"]]["parent"] or "",
                 "n_parts": "" if cell is None else cell["n_parts"], "n_repeats": "" if cell is None else cell["n_repeats"],
                 "expected_repeats": res["row_by_id"][e["id"]]["repeats"], "m4_source": "" if cell is None else (cell["m4_source"] or ""),
                 "note": "; ".join(run["warnings"] + run["errors"])}
            for key in (*METRICS, "M5_edit", "usd", "tokens"):
                stat = None if cell is None else cell[key]
                r[key + "_mean"] = "" if stat is None or stat["mean"] is None else repr(stat["mean"])
                r[key + "_sd"] = "" if stat is None or stat["sd"] is None else repr(stat["sd"])
            rows.append(r)
    return rows


def paired_csv_rows(res: dict) -> list:
    rows = []
    for p in res["paired"]:
        rows.append({"model": p["model"], "row": p["row"], "parent": p["parent"], "test": "paired bootstrap", "metric": p["metric"],
                     "n": p["n"], "mean_diff": "" if p["mean"] is None else repr(p["mean"]), "ci_lo": "" if p["lo"] is None else repr(p["lo"]),
                     "ci_hi": "" if p["hi"] is None else repr(p["hi"]), "row_only": "", "parent_only": "", "p": "", "verdict": p["verdict"]})
    for m in res["mcnemar"]:
        rows.append({"model": m["model"], "row": m["row"], "parent": m["parent"], "test": "McNemar exact",
                     "metric": "M1 sample" if m["outcome"] == "sample" else "M2 LLM-final", "n": m["n"], "mean_diff": "", "ci_lo": "",
                     "ci_hi": "", "row_only": m["row_only"], "parent_only": m["parent_only"],
                     "p": "" if m["p"] is None else repr(m["p"]), "verdict": m["verdict"]})
    return rows


PAIRED_FIELDS = ["model", "row", "parent", "test", "metric", "n", "mean_diff", "ci_lo", "ci_hi", "row_only", "parent_only", "p", "verdict"]


def per_part_rows(res: dict) -> list:
    rows = []
    for model in res["cfg"]["models"]:
        for row in res["cfg"]["rows"]:
            run = res["usable"].get((row["id"], model["id"]))
            if run is None:
                continue
            for pid in sorted(run["part_values"]):
                v = run["part_values"][pid]
                rows.append({"model": model["id"], "row": row["id"], "system": row["system"], "part_id": pid, "n_repeats": v["n_repeats"],
                             **{k: "" if v[k] is None else repr(v[k]) for k in ("M1", "M2p", "M4", "M6", "M7")}})
    return rows


PER_PART_FIELDS = ["model", "row", "system", "part_id", "n_repeats", "M1", "M2p", "M4", "M6", "M7"]


def _ci_text(p, nd):
    if p["mean"] is None:
        return DASH
    if p["lo"] is None:
        return f"{p['mean']:+.{nd}f} (n={p['n']}, no interval)"
    return f"{p['mean']:+.{nd}f} [{p['lo']:+.{nd}f}, {p['hi']:+.{nd}f}]"


def _p_text(p):
    return DASH if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def render_paired(res: dict) -> str:
    cfg = res["cfg"]
    out = ["# Paired comparisons: each row against the row it adds a component to", "",
           "Differences are row minus parent, per part (mean over its repeats), 95% bootstrap interval over parts "
           f"({cfg['resamples']} resamples, seed {cfg['seed']}). A verdict other than \"{UNCLEAR}\" means the interval excludes 0. "
           "No correction for the number of comparisons. McNemar: exact, on pairs; with several repeats these pairs are clustered "
           "by part, so read the p-values as approximate.", ""]
    for model in cfg["models"]:
        out += [f"## {model['label']}", ""]
        comps = [(r["id"], r["parent"]) for r in cfg["rows"] if r["parent"]]
        any_row = False
        for rid, pid in comps:
            ps = [p for p in res["paired"] if p["model"] == model["id"] and p["row"] == rid]
            ms = [m for m in res["mcnemar"] if m["model"] == model["id"] and m["row"] == rid]
            if not ps:
                out += [f"### {rid} vs {pid}", "", "not compared: one of the two runs is missing or unusable", ""]
                continue
            any_row = True
            out += [f"### {rid} vs {pid}", "", "| Metric | Parts | Difference [95% CI] | Verdict |", "|---|---|---|---|"]
            for p in ps:
                out.append(f"| {PAIRED_LABELS[p['metric']]} | {p['n']} | {_ci_text(p, DECIMALS[p['metric']])} | {p['verdict']} |")
            out += ["", "| Validity (McNemar) | Pairs | Only " + rid + " | Only " + pid + " | p | Verdict |", "|---|---|---|---|---|---|"]
            for m in ms:
                name = "M1 raw sample strict-valid" if m["outcome"] == "sample" else "M2 final route is an LLM route"
                out.append(f"| {name} | {m['n']} | {m['row_only']} | {m['parent_only']} | {_p_text(m['p'])} | {m['verdict']} |")
            out.append("")
        if not comps:
            out += ["no row has a parent in the config", ""]
        elif not any_row:
            out += ["nothing to compare yet", ""]
    return "\n".join(out)


def render_consistency(res: dict) -> str:
    cfg = res["cfg"]
    out = ["# Across LLMs: does a component help for every model?", "",
           "Verdicts as in the paired comparisons (interval over parts excludes 0). Section 7.4 asks for at least three LLMs so that "
           "no conclusion rests on one model; \"mixed\" means the models disagree.", ""]
    if not res["consistency"]:
        return "\n".join(out + ["nothing to compare yet", ""])
    heads = [m["label"] for m in cfg["models"]]
    out += ["| Comparison | Metric | " + " | ".join(heads) + " | Across LLMs |", "|---|---|" + "---|" * len(heads) + "---|"]
    for c in res["consistency"]:
        out.append(f"| {c['row']} vs {c['parent']} | {PAIRED_LABELS[c['metric']]} | "
                   + " | ".join(c["verdicts"][m["id"]] for m in cfg["models"]) + f" | {c['label']} |")
    out.append("")
    return "\n".join(out)


def render_issues(res: dict) -> str:
    if not res["issues"]:
        return "No issues: every run of the grid is present, checked and complete.\n"
    order = {"error": 0, "missing": 1, "warning": 2}
    out = []
    for it in sorted(res["issues"], key=lambda i: (order.get(i["level"], 9), i["row"] or "", i["model"] or "", i["message"])):
        where = " ".join(x for x in (it["row"], it["model"]) if x)
        out.append(f"{it['level'].upper():8s}{(' ' + where) if where else ''}: {it['message']}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- the status grid and the run commands
def status_grid(cfg: dict, loaded_or_res: dict) -> str:
    runs = loaded_or_res["runs"]
    sym = {"ok": "ok", "partial": "partial", "missing": "missing", "error": "ERROR", "not_built": "not built"}
    w0 = max(len(r["id"] + "  " + r["label"][:40]) for r in cfg["rows"])
    heads = [m["label"][:18] for m in cfg["models"]]
    ws = [max(len(h), 9) for h in heads]
    out = [" " * w0 + "  " + "  ".join(h.ljust(w) for h, w in zip(heads, ws))]
    for row in cfg["rows"]:
        cells = []
        for m, w in zip(cfg["models"], ws):
            run = runs[(row["id"], m["id"])]
            cells.append((sym[run["status"]] + (" (shared)" if row["independent"] and run["status"] in ("ok", "partial") else "")).ljust(w))
        out.append((row["id"] + "  " + row["label"][:40]).ljust(w0) + "  " + "  ".join(cells))
    n = len({r["folder"] for r in runs.values() if r["status"] in ("missing", "error", "partial")})
    out.append("")
    out.append(f"{n} run(s) to make or redo." if n else "Every run is present and complete.")
    return "\n".join(out)


def _cache_key(system):
    """Systems with the same key send the same prompts, so their API answers are shared through the cache of one tag. Run the system
    with the most samples first and the others cost nothing (n=1 is the first sample of n=5)."""
    spec = systems.parse_system(system)
    p = systems.planner_spec(system)
    return (p.few_shot, p.loo, p.hint, p.two_step, p.model, p.cascade, p.constrained), -spec[0]


def run_commands(cfg: dict, loaded: dict) -> list:
    """The run_eval lines for every cell whose run is missing, unusable or incomplete, grouped by LLM, cheapest order first.
    Returns a list of text lines (comments start with #)."""
    runs = loaded["runs"]
    out, warn = [], []
    if dataset.is_sealed(cfg["dataset"]):
        warn = [f"# WARNING: {cfg['dataset']} is sealed. Run it only for the final results (WP11): once its results are seen no prompt,",
                "# setting or choice of row (A5, A7, A8 ...) may change. Develop and decide on the development set first, then freeze:",
                "#   python scripts/freeze.py create --grid <this config> --run-tests ; git add FREEZE.json ; git commit -m Freeze ; git tag freeze-v1",
                "# run_eval stops on a sealed dataset if FREEZE.json is missing or the working tree has drifted from it.", ""]
    todo = 0

    def line(row, tag, why):
        parts = [f"python scripts/run_eval.py --system {row['system']} --dataset {cfg['dataset']} --tag {tag} --repeats {row['repeats']}"]
        if cfg["limit"]:
            parts.append(f"--limit {cfg['limit']}")
        if cfg["offset"]:
            parts.append(f"--offset {cfg['offset']}")
        p = systems.planner_spec(row["system"])
        if p.few_shot and cfg["pool"] != "dev_v2":
            parts.append(f"--pool {cfg['pool']}")
        text = " ".join(parts)
        return f"{text}    # {why}" if why else text

    def reason(run):
        if run["status"] == "error":
            return "existing folder is unusable and will be overwritten: " + "; ".join(run["errors"])[:150]
        if run["status"] == "partial":
            return "incomplete (cached answers are reused): " + "; ".join(run["warnings"])[:150]
        return ""

    for model in cfg["models"]:
        block = []
        pending = [r for r in cfg["rows"] if r["built"] and not r["independent"]
                   and runs[(r["id"], model["id"])]["status"] in ("missing", "error", "partial")]
        for r in sorted(pending, key=lambda r: _cache_key(r["system"])):
            block.append(line(r, model["tag"], reason(runs[(r["id"], model["id"])])))
        if block:
            todo += len(block)
            out += [f"# --- {model['label']}: {len(block)} run(s). Same tag for all of them, so repeated prompts are answered from the cache.",
                    f'$env:CAPP_MODEL_PRIMARY = "{model["model"]}"'] + block + [""]
    shared = [r for r in cfg["rows"] if r["built"] and r["independent"]
              and runs[(r["id"], cfg["models"][0]["id"])]["status"] in ("missing", "error", "partial")]
    if shared:
        todo += len(shared)
        out += [f"# --- rows that choose their own models (+ms, +ml, +cas1, +cas2): run once, they appear in every column. "
                "The models come from CAPP_MODEL_SMALL / CAPP_MODEL_LARGE (capp/config.py).",
                "Remove-Item Env:CAPP_MODEL_PRIMARY -ErrorAction SilentlyContinue"]
        out += [line(r, cfg["shared_tag"], reason(runs[(r["id"], cfg["models"][0]["id"])])) for r in sorted(shared, key=lambda r: _cache_key(r["system"]))]
        out.append("")
    if not todo:
        out = out + ["# Nothing to run: every run of the grid is present and complete."]
    else:
        out = [f"# {todo} run(s) for {cfg['name']} (PowerShell, from the repo root). Then: python scripts/make_tables.py build <config>", ""] + out
    unbuilt = [r["id"] for r in cfg["rows"] if not r["built"]]
    if unbuilt:
        out.append(f"# Not built, nothing to run: {', '.join(unbuilt)}")
    return warn + out


# ---------------------------------------------------------------- writing everything
def _write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def manifest(res: dict, files: dict) -> dict:
    cfg = res["cfg"]
    runs = []
    for row in cfg["rows"]:
        for model in cfg["models"]:
            run = res["runs"][(row["id"], model["id"])]
            info = run.get("run_info") or {}
            runs.append({"row": row["id"], "model": model["id"], "system": row["system"], "status": run["status"],
                         "folder": None if not run["folder"] else os.path.relpath(run["folder"], res["results_dir"]).replace("\\", "/"),
                         "records_sha256_16": run["records_sha"], "n_records": len(run["records"]), "n_parts": len(run["parts"]),
                         "repeats": run["repeats"], "run_json": {k: info.get(k) for k in ("system", "tag", "dataset", "model", "git_commit",
                                                                                         "cascade_level") if k in info},
                         "freeze": (info.get("freeze") or {}).get("status") if isinstance(info.get("freeze"), dict) else None,
                         "freeze_id": (info.get("freeze") or {}).get("id") if isinstance(info.get("freeze"), dict) else None})
    return {"tool": "capp.tables (scripts/make_tables.py)", "config_sha256_16": config_sha(cfg), "config": cfg,
            "dataset_checked_against": res["expected"][2], "runs": runs, "issues": res["issues"], "files": files}


def build(cfg: dict, out_dir: str, results_dir: str = None, figures: bool = True) -> dict:
    """Analyse the grid and write Table 4 (Markdown, CSV, LaTeX), the paired comparisons, the per-part results, the issues, the figures
    and the manifest into out_dir. Returns the analysis with 'files' (name -> path). Same runs and config => the same text files."""
    res = analyse(cfg, results_dir)
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    def put(name, writer):
        p = os.path.join(out_dir, name)
        writer(p)
        paths[name] = p

    put("table4.md", lambda p: _write_text(p, render_markdown(res)))
    put("table4.tex", lambda p: _write_text(p, render_latex(res)))
    rows = table_csv_rows(res)
    put("table4.csv", lambda p: _write_csv(p, list(rows[0].keys()), rows))
    put("paired.md", lambda p: _write_text(p, render_paired(res)))
    put("paired.csv", lambda p: _write_csv(p, PAIRED_FIELDS, paired_csv_rows(res)))
    put("consistency.md", lambda p: _write_text(p, render_consistency(res)))
    put("per_part.csv", lambda p: _write_csv(p, PER_PART_FIELDS, per_part_rows(res)))
    put("issues.txt", lambda p: _write_text(p, render_issues(res)))
    if figures:
        from capp import tablefigs
        made, why = tablefigs.make_figures(res, out_dir)
        for name in made:
            paths[name] = os.path.join(out_dir, name)
        if why:
            res["issues"].append(_issue("info", why))
            _write_text(paths["issues.txt"], render_issues(res))
    hashes = {name: _sha(p) for name, p in sorted(paths.items()) if name.endswith((".md", ".csv", ".tex", ".txt"))}
    put("manifest.json", lambda p: _write_text(p, json.dumps(manifest(res, hashes), indent=2, ensure_ascii=False, default=str) + "\n"))
    res["files"] = paths
    return res
