"""One text file that says how the project stands (MTP-1 Step 31).

The author used to paste ten different outputs into the chat. `scripts/diagnostics.py` collects them in one go, offline:

  summary        the problems on top, then what ran and what was skipped
  environment    python, git (commit, uncommitted files), whether an API key is set (never its value), cache folder
  freeze         FREEZE.json: none / ok / analysis-changed / drift
  datasets       dev_v2, test_v2, bench_v1: there or not, number of parts, validation problems
  runs           one row per results/<tag>_<system>: system, model, mode, strict-valid share, parse rate, calls, truncated answers, freeze
  G1 baselines   which of the three baseline runs are still to make, with the exact command from RUNBOOK.md
  checks         the offline scripts, each with its exit code and output (physics, feasibility, objective, feasrepair, compare, part_diff,
                 confidence_tune, risk_compare, cascade_pareto, the grid status, freeze verify); one whose input is not there yet is skipped, with the reason

Nothing here makes an API call or starts a run: the checks are a fixed list of read-only scripts (CHECK_SCRIPTS). The only thing written is the one
output file. A line of the output that looks like a secret (capp/pack.py patterns, plus the values of .env and of the environment) is removed.
"""
import datetime
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time

from capp import dataset as D, freeze as F, pack as PK

DEFAULT_TAG = "dev"
DEFAULT_DATASET = "dev_v2"
DEFAULT_GRID = "configs/ablation_template.json"
DEFAULT_OUT = "results/diagnostics.txt"
TIMEOUT = 900                                    # seconds, per check
MAX_LINES = 400                                  # per check: the first 300 and the last 100 lines
MAX_RUN_ROWS = 200
DATASETS = ("dev_v2", "test_v2", "bench_v1")
BASELINES = (                                    # the G1 runs of RUNBOOK.md: (system, run_eval options after --tag)
    ("rb", ""),
    ("b1", " --limit 20"),
    ("b0", " --limit 20 --throttle 1"),
)
RUN_SYSTEMS = ("rb", "b1", "b0", "b0_r1")
CHECK_SCRIPTS = frozenset({"physics_check", "feasibility_check", "objective_check", "feasrepair_check", "compare_runs", "part_diff",
                           "confidence_tune", "risk_compare", "cascade_pareto", "make_tables", "freeze"})      # all of them offline; no run_eval, no smoke_llm
KEY_NAME = "GROQ_API_KEY"
STATUS_ORDER = ("ok", "failed", "timeout", "skipped")


class DiagnosticsError(ValueError):
    """A wrong argument, in words for the author."""


# ---------------------------------------------------------------- small helpers
def _decode(raw) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return raw.replace("\r\n", "\n").replace("\r", "\n")


def _show(argv: list) -> str:
    """The command as the author would type it in PowerShell: python scripts\\x.py results\\dev_rb ..."""
    parts = ["python"] + [a.replace("/", "\\") if a.startswith(("results/", "configs/", "scripts/")) else a for a in argv[1:]]
    return " ".join(parts)


def _clip(text: str, max_lines: int) -> str:
    lines = text.rstrip("\n").split("\n")
    if len(lines) <= max_lines:
        return "\n".join(lines)
    head, tail = max_lines * 3 // 4, max_lines // 4
    return "\n".join(lines[:head] + [f"[... {len(lines) - head - tail} line(s) left out: this file keeps the first {head} and the last {tail} ...]"] + lines[-tail:])


def _cell(v, width: int = 0) -> str:
    s = "-" if v in (None, "") else str(v)
    return s.ljust(width)


def _pct(v) -> str:
    return "-" if not isinstance(v, (int, float)) else f"{100 * v:.1f}%"


def _table(rows: list, header: list) -> list:
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)] if rows else [len(h) for h in header]
    fmt = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip()
    return [fmt(header), fmt(["-" * w for w in widths])] + [fmt(r) for r in rows]


def _read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_dir(root: str, tag: str, system: str) -> str:
    return os.path.join(root, "results", f"{tag}_{system}")


def _has_records(folder: str) -> bool:
    return os.path.isfile(os.path.join(folder, "records.jsonl"))


# ---------------------------------------------------------------- running a command
def run_command(argv: list, cwd: str, timeout: int = TIMEOUT):
    """(exit code or None after the time limit, text). Standard input is closed, so nothing can wait for a keystroke."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        p = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        return None, _decode(exc.stdout or b"") + f"\n[stopped after {timeout} s]"
    except OSError as exc:
        return 127, f"[could not start: {exc}]"
    text = _decode(p.stdout)
    err = _decode(p.stderr).strip("\n")
    return p.returncode, text + (("\n" if text and not text.endswith("\n") else "") + "[stderr]\n" + err if err else "")


# ---------------------------------------------------------------- the pieces of the file
def _git(root: str, *args):
    try:
        p = subprocess.run(["git", "-C", root] + list(args), capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def _key_state(root: str) -> str:
    if os.environ.get(KEY_NAME, "").strip():
        return "yes (environment variable)"
    try:
        with open(os.path.join(root, ".env"), encoding="utf-8") as f:
            for line in f:
                k, _, v = line.strip().partition("=")
                if k.strip() in (KEY_NAME, "export " + KEY_NAME) and v.strip().strip("\"'"):
                    return "yes (.env)"
    except OSError:
        pass
    return "NO"


def environment(root: str, cache_dir: str = None) -> dict:
    """{'lines', 'problems', 'notes'}: a problem stops or damages something, a note is only good to know."""
    lines, problems, notes = [f"python      {sys.version.split()[0]}  ({platform.platform()})"], [], []
    if _git(root, "rev-parse", "--is-inside-work-tree") != "true":
        lines.append("git         not a git repository, or git is not installed")
    else:
        head = _git(root, "rev-parse", "--short", "HEAD") or "(none yet)"
        dirty = [l for l in (_git(root, "status", "--porcelain") or "").split("\n") if l.strip()]
        tags = _git(root, "tag", "--points-at", "HEAD") or ""
        lines.append(f"git         commit {head}" + (f", tag {', '.join(tags.split())}" if tags else "") + f", {len(dirty)} uncommitted file(s)")
        lines += [f"              {l.strip()}" for l in dirty[:10]] + ([f"              ... and {len(dirty) - 10} more"] if len(dirty) > 10 else [])
        if dirty:
            notes.append(f"git: {len(dirty)} uncommitted file(s): fine while working, commit before a freeze or a release")
    key = _key_state(root)
    lines.append(f"{KEY_NAME}  {key}")
    if key == "NO":
        problems.append(f"{KEY_NAME} is not set (environment or .env): a live LLM run will stop")
    cache = cache_dir or os.environ.get("CAPP_CACHE_DIR") or os.path.join(root, "cache", "llm")
    n = size = 0
    for r, _, fs in os.walk(cache):
        for f in fs:
            if f.endswith(".json"):
                n += 1
                size += os.path.getsize(os.path.join(r, f))
    lines.append(f"LLM cache   {n} answer(s), {size / 1e6:.1f} MB" if os.path.isdir(cache) else "LLM cache   no cache folder yet (no live call made so far)")
    lines.append(f"            ({os.path.relpath(cache, root) if cache.startswith(root) else cache})")
    return {"lines": lines, "problems": problems, "notes": notes}


def freeze_state(root: str) -> dict:
    st = F.status(root)
    lines = [f"status      {st['status']}" + (f"  (freeze {st['id']})" if st.get("id") else "")]
    problems = []
    if st["status"] == "none":
        lines.append("            no FREEZE.json: right until gate G2 (RUNBOOK.md: the freeze comes before the first run on a sealed set)")
    else:
        lines.append(f"manifest    {st['manifest']}")
        lines += [f"  difference  {d}" for d in st.get("differences", [])]
        if st.get("why"):
            lines.append(f"  why         {st['why']}")
        if st["status"] in ("drift", "unreadable"):
            problems.append(f"freeze: FREEZE.json is {st['status']}: run_eval will refuse sealed sets. If you did not create it on purpose, it is a leftover of a rehearsal")
    return {"lines": lines, "problems": problems}


def datasets_state(root: str, names=DATASETS) -> dict:
    rows, problems = [], []
    for name in names:
        path = os.path.join(root, "data", f"{name}.json")
        if not os.path.isfile(path):
            hint = "python scripts\\make_dataset_v2.py" if name != "bench_v1" else "scripts\\benchmark.py import (when the annotators have answered)"
            rows.append([name, "sealed" if D.is_sealed(name) else "open", "missing", "-", f"made by {hint}"])
            if name == "dev_v2":
                problems.append(f"data\\{name}.json is missing: {hint}")
            continue
        try:
            parts = _read_json(path)
            errs = D.validate_dataset(parts)
            rows.append([name, "sealed" if D.is_sealed(name) else "open", str(len(parts)), "ok" if not errs else f"{len(errs)} problem(s)", ""])
            if errs:
                problems.append(f"data\\{name}.json: {len(errs)} validation problem(s), first: {errs[0]}")
        except Exception as exc:                      # a damaged file must show up in the file, not stop it
            rows.append([name, "sealed" if D.is_sealed(name) else "open", "unreadable", "-", str(exc)[:80]])
            problems.append(f"data\\{name}.json cannot be read: {str(exc)[:120]}")
    return {"lines": _table(rows, ["dataset", "use", "parts", "validation", "note"]), "problems": problems}


def _freeze_label(run: dict) -> str:
    fr = run.get("freeze")
    return fr.get("status", "-") if isinstance(fr, dict) else "-"


def runs_state(root: str) -> dict:
    results = os.path.join(root, "results")
    folders = sorted(d for d in (os.listdir(results) if os.path.isdir(results) else []) if os.path.isfile(os.path.join(results, d, "run.json")))
    rows, problems = [], []
    for d in folders:
        base = os.path.join(results, d)
        try:
            run = _read_json(os.path.join(base, "run.json"))
            summ = _read_json(os.path.join(base, "summary.json")) if os.path.isfile(os.path.join(base, "summary.json")) else None
        except Exception as exc:
            rows.append([d, "unreadable"] + ["-"] * 13)
            problems.append(f"results\\{d}: run.json or summary.json cannot be read ({str(exc)[:80]})")
            continue
        if summ is None:
            problems.append(f"results\\{d}: there is no summary.json (an interrupted run?)")
        lines = 0
        try:
            with open(os.path.join(base, "records.jsonl"), "rb") as f:
                lines = sum(1 for l in f if l.strip())
        except OSError:
            problems.append(f"results\\{d}: there is no records.jsonl")
        n_rec = (summ or {}).get("n_records")
        if n_rec is not None and n_rec != lines:
            problems.append(f"results\\{d}: records.jsonl has {lines} record(s), summary.json says {n_rec}")
        m1, m2 = (summ or {}).get("M1") or {}, (summ or {}).get("M2") or {}
        cl = run.get("llm_client_summary") or {}
        toks = (cl.get("prompt_tokens") or 0) + (cl.get("completion_tokens") or 0)
        trunc = cl.get("truncated")
        api = m1.get("api_errors")
        if api:
            problems.append(f"results\\{d}: {api} API error(s) (rate limit or outage, not the model's fault): run the same command again with the "
                            "same --tag; the answers already received come from the cache and only these calls are made again")
        if trunc:
            problems.append(f"results\\{d}: {trunc} answer(s) hit the token limit (finish_reason length): they are cut off; tell Claude")
        fz = _freeze_label(run)
        rows.append([d, run.get("system"), run.get("dataset"), run.get("n_parts"), run.get("repeats"), run.get("model"), run.get("llm_mode"),
                     _pct(m2.get("final_strict_valid")), _pct(m1.get("parse_rate")), cl.get("calls") if cl else None, trunc if cl else None,
                     api, cl.get("retries") if cl else None, f"{toks / 1000:.0f}k" if cl else None, fz])
    shown = [[_cell(c) for c in r] for r in rows[:MAX_RUN_ROWS]]
    header = ["folder", "system", "dataset", "parts", "rep", "model", "mode", "strict-valid", "parse", "calls", "trunc", "api-err", "retries", "tokens", "freeze"]
    lines = [f"{len(folders)} run folder(s) in results\\ with a run.json"] if folders else ["no run folder yet in results\\"]
    if folders:
        lines += _table(shown, header)
        if len(rows) > MAX_RUN_ROWS:
            lines.append(f"... and {len(rows) - MAX_RUN_ROWS} more")
    return {"lines": lines, "problems": problems, "n": len(folders)}


def baselines_state(root: str, tag: str, dataset: str) -> dict:
    lines, missing = [], 0
    for system, extra in BASELINES:
        have = _has_records(_run_dir(root, tag, system))
        cmd = f"python scripts\\run_eval.py --system {system} --dataset {dataset} --tag {tag}{extra}"
        lines.append(f"{'done   ' if have else 'to make'}  results\\{tag}_{system}" + ("" if have else f"    {cmd}"))
        missing += not have
    if missing == 0:
        lines.append("all three are there: next is `compare_runs` (below) and the benchmark in RUNBOOK.md, G1")
    return {"lines": lines, "problems": []}


# ---------------------------------------------------------------- the offline checks
def check_list(root: str, tag: str, dataset: str, grid: str, tmp_dir: str) -> list:
    """[{'title', 'argv' (without python), 'needs': [(path, why)]}] in the order they appear in the file. `argv` always starts with the script."""
    rb, b1, b0, b0r1 = (f"results/{tag}_{s}" for s in RUN_SYSTEMS)
    ds = f"data/{dataset}.json"
    dsneed = [(ds, f"{ds.replace('/', chr(92))} is not there (python scripts\\make_dataset_v2.py)")]
    runs = lambda *fs: [(f + "/records.jsonl", f"{f.replace('/', chr(92))} is not there yet") for f in fs]
    cs = lambda name, *args: ["scripts/%s.py" % name, *args]
    present = [f for f in (rb, b1, b0, b0r1) if _has_records(os.path.join(root, f))]
    out = [
        {"title": "physics_check", "argv": cs("physics_check", "--dataset", dataset), "needs": dsneed},
        {"title": "feasibility_check", "argv": cs("feasibility_check", "--dataset", dataset), "needs": dsneed},
        {"title": "objective_check", "argv": cs("objective_check", "--dataset", dataset, "--out", tmp_dir), "needs": dsneed},      # it writes a CSV and a JSON: into a temporary folder
        {"title": "feasrepair_check", "argv": cs("feasrepair_check", "--dataset", dataset), "needs": dsneed},
        {"title": "grid status", "argv": cs("make_tables", "status", grid), "needs": [(grid, f"{grid.replace('/', chr(92))} is not there")]},
        {"title": "compare_runs", "argv": cs("compare_runs", *present), "needs": [] if len(present) >= 2 else
         [(next(f for f in (rb, b1, b0, b0r1) if f not in present) + "/records.jsonl",          # a folder that is not there, so the check is skipped
           f"fewer than two of results\\{tag}_rb, _b1, _b0, _b0_r1 are there yet")]},
        {"title": "part_diff rb vs b0", "argv": cs("part_diff", rb, b0), "needs": runs(rb, b0)},
        {"title": "confidence_tune b0_r1", "argv": cs("confidence_tune", b0r1), "needs": runs(b0r1)},
        {"title": "risk_compare rb vs b0", "argv": cs("risk_compare", rb, b0, "--physics"), "needs": runs(rb, b0)},
        {"title": "risk_compare rb vs b0_r1", "argv": cs("risk_compare", rb, b0r1, "--physics"), "needs": runs(rb, b0r1)},
        {"title": "cascade_pareto b0_r1 b0", "argv": cs("cascade_pareto", b0r1, b0), "needs": runs(b0r1, b0)},
        {"title": "freeze verify", "argv": cs("freeze", "verify"), "needs": [("FREEZE.json", "there is no FREEZE.json (right until gate G2)")]},
    ]
    return out


def run_checks(root: str, tag: str, dataset: str, grid: str, timeout: int, max_lines: int, runner, progress=None) -> list:
    """[{'title', 'command', 'status' ok|exit|timeout|skipped, 'code', 'seconds', 'text', 'why'}]"""
    results = []
    with tempfile.TemporaryDirectory(prefix="diagnostics-") as tmp:
        for c in check_list(root, tag, dataset, grid, tmp):
            script = os.path.basename(c["argv"][0])[:-3]
            if script not in CHECK_SCRIPTS:                                # nothing outside the offline list is ever started
                raise DiagnosticsError(f"{script} is not one of the offline scripts")
            shown = list(c["argv"])
            if "--out" in shown:                                           # the temporary folder is not something to show
                i = shown.index("--out")
                del shown[i:i + 2]
            display = _show([sys.executable] + shown)
            rec = {"title": c["title"], "command": display, "status": "skipped", "code": None, "seconds": 0.0, "text": "", "why": ""}
            missing = [why for path, why in c["needs"] if not os.path.isfile(os.path.join(root, path))]
            if missing:
                rec["why"] = missing[0]
            else:
                if progress:
                    progress(f"running {c['title']} ...")
                t0 = time.time()
                code, text = runner([sys.executable] + c["argv"], root, timeout)
                rec.update(code=code, seconds=round(time.time() - t0, 1), text=_clip(_decode(text), max_lines),
                           status="ok" if code == 0 else ("timeout" if code is None else "failed"))
            results.append(rec)
    return results


# ---------------------------------------------------------------- the whole file
def redact(text: str, root: str) -> tuple:
    """(text without the lines that look like a secret, [(line, what)]). The match itself is never kept."""
    found = PK.scan_secrets(text.encode("utf-8"), PK._env_values(root))
    if not found:
        return text, []
    lines = text.split("\n")
    for line, what in found:
        lines[line - 1] = f"[line removed: it looked like a secret ({what})]"
    return "\n".join(lines), found


def build_report(root: str, tag: str = DEFAULT_TAG, dataset: str = DEFAULT_DATASET, grid: str = DEFAULT_GRID, checks: bool = True,
                 timeout: int = TIMEOUT, max_lines: int = MAX_LINES, runner=None, now: datetime.datetime = None, progress=None) -> dict:
    """{'text', 'problems', 'notes', 'counts', 'removed'}. Nothing is written."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag or ""):
        raise DiagnosticsError(f"--tag {tag!r}: a run tag is letters, digits, _ . - only (the folder is results\\<tag>_<system>)")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset or ""):
        raise DiagnosticsError(f"--dataset {dataset!r}: a dataset name such as dev_v2")
    runner = runner or run_command
    sections = [("environment", environment(root)), ("freeze", freeze_state(root)), ("datasets", datasets_state(root)),
                ("runs", runs_state(root)), ("G1 baselines", baselines_state(root, tag, dataset))]
    problems = [p for _, s in sections for p in s["problems"]]
    notes = [n for _, s in sections for n in s.get("notes", [])]
    done = run_checks(root, tag, dataset, grid, timeout, max_lines, runner, progress) if checks else []
    counts = {k: sum(1 for c in done if c["status"] == k) for k in STATUS_ORDER}
    for c in done:
        if c["status"] == "failed" and c["title"] != "freeze verify":          # its exit 1 is the freeze section's drift
            problems.append(f"check '{c['title']}' ended with exit code {c['code']}")
        if c["status"] == "timeout":
            problems.append(f"check '{c['title']}' did not finish in {timeout} s")
    stamp = (now or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"DIAGNOSTICS  ({stamp})", f"repo {os.path.basename(os.path.abspath(root))}   tag {tag}   dataset {dataset}   grid {grid}", ""]
    L.append("PROBLEMS" + (f" ({len(problems)})" if problems else ""))
    L += [f"  - {p}" for p in problems] or ["  none"]
    if notes:
        L += ["", "NOTES"] + [f"  - {n}" for n in notes]
    if checks:
        L.append("")
        L.append("CHECKS  " + ", ".join(f"{counts[k]} {k}" for k in STATUS_ORDER if counts[k]) if done else "CHECKS  none")
        for c in done:
            L.append(f"  {c['status']:8s}{c['title']}" + (f"   ({c['why']})" if c["status"] == "skipped" else f"   exit {c['code']}, {c['seconds']} s"
                                                      if c["code"] is not None else f"   {c['seconds']} s"))
    for title, s in sections:
        L += ["", "=" * 8 + f" {title} " + "=" * max(0, 60 - len(title))] + s["lines"]
    for c in done:
        L += ["", "=" * 8 + f" check: {c['title']} " + "=" * max(0, 52 - len(c["title"])), f"$ {c['command']}"]
        L += [f"(skipped: {c['why']})"] if c["status"] == "skipped" else [f"(exit {c['code']}, {c['seconds']} s)"] + (c["text"].split("\n") if c["text"] else ["(no output)"])
    text = "\n".join(L) + "\n"
    text, found = redact(text, root)
    if found:
        text = text.replace("\n", f"\nNOTE: {len(found)} line(s) of this file looked like a secret and were removed.\n", 1)
    return {"text": text, "problems": problems, "notes": notes, "counts": counts, "removed": found}


def write_text(text: str, out: str) -> str:
    """Atomically: a temporary file next to the target, then os.replace. LF line ends, UTF-8. Returns the path."""
    folder = os.path.dirname(os.path.abspath(out))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".diag-", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8"))
        os.chmod(tmp, 0o644)
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return out
