"""Where the project stands and the command to run next (MTP-1 Step 41).

    python scripts/where.py

About fifteen lines, offline, a second: it reads the folders (results/, configs/, FREEZE.json, .env) and never starts a run, never calls the API,
writes nothing. It follows the order of RUNBOOK.md, gate by gate, and the first line that says TODO is where you are; the commands under NEXT
are the ones to type. A screenshot of its output is what the author can send instead of asking "what next?".

  key        GROQ_API_KEY found (the value is never shown)
  LLM runs   the runs of results/ that called a model: made with today's settings, cut off at the token limit, made with old settings (Step 39)
  smoke      no LLM run with today's settings and no cut-off answer yet: run scripts/smoke_llm.py (3 calls) before the long runs
  G1         rb, b1, b0 on dev_v2 (the check of scripts/g1.py: records, API errors, cut-off answers, model settings), then results/diagnostics.txt
  G2         the ablation grid on dev_v2 (scripts/g2.py), its table, the final grid config, the freeze
  G3/G4      the final grid on the sealed set, once the freeze is there
  release    release/v1

What it cannot see is told as such: the choices of the 'decide' rows, the case-study commentary, the prose of the paper and the Zenodo upload are
the author's (RUNBOOK.md). Any line it cannot work out says so (WARN) instead of stopping.
"""
import glob
import json
import os
from collections import namedtuple

from capp import config, dataset as dataset_mod, diagnostics, freeze, g1, runcheck, tables

GRID = "configs/ablation_template.json"
FINAL = "configs/final.json"
RELEASE_CONFIG = "configs/release_final.json"
RELEASE_DIR = "release/v1"
WIDTH = 100                                          # a line is cut here, so a screenshot reads

Row = namedtuple("Row", "label text state commands")      # state: ok | todo | warn | info; commands: what to type if this is the first todo

_ERRORS = (OSError, ValueError, TypeError, KeyError, AttributeError, tables.TableError, g1.G1Error)


def _win(path: str) -> str:
    return path.replace("/", "\\")


def _py(script: str, *args: str) -> str:
    return " ".join(["python", "scripts\\" + script] + [a for a in args if a])


# ---------------------------------------------------------------- the pieces
def key_row(root: str) -> Row:
    state = diagnostics._key_state(root)
    if state == "NO":
        return Row("key", f"{diagnostics.KEY_NAME} NOT found (needed for every run that calls the API)", "todo",
                   (f"put  {diagnostics.KEY_NAME}=...  in the file .env in the repository folder (or set the environment variable)",))
    return Row("key", f"{diagnostics.KEY_NAME} found {state.partition(' ')[2]}", "ok", ())


def scan_runs(root: str) -> dict:
    """Counts over results/*/run.json of the runs that called a model: {'llm', 'usable', 'cut', 'old'} (a run can be both cut and old)."""
    out = {"llm": 0, "usable": 0, "cut": 0, "old": 0}
    for path in sorted(glob.glob(os.path.join(root, "results", "*", "run.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, ValueError):
            continue
        if not runcheck.models_of(info):
            continue
        cut, old = runcheck.truncated(info) > 0, bool(runcheck.stale_settings(info))
        calls = (info.get("llm_client_summary") or {}).get("calls") if isinstance(info.get("llm_client_summary"), dict) else 0
        out["llm"] += 1
        out["cut"] += cut
        out["old"] += old
        out["usable"] += int(not cut and not old and isinstance(calls, int) and calls > 0)
    return out


def runs_rows(root: str, need_llm: bool) -> list:
    """The 'LLM runs' line and, when a long LLM run is still to come and nothing shows that today's settings work, the 'smoke' line."""
    try:
        c = scan_runs(root)
    except ValueError as exc:                                       # a wrong CAPP_QWEN_REASONING
        return [Row("setting", str(exc)[:300], "todo", ("in PowerShell: Remove-Item Env:CAPP_QWEN_REASONING   (or set it to none, low, medium, high, default, omit)",))]
    if not c["llm"]:
        rows = [Row("LLM runs", "none yet in results\\ (no run has called a model)", "info", ())]
    else:
        text = f"{c['llm']} in results\\: {c['usable']} made with today's settings and not cut off"
        text += f", {c['cut']} with answers cut off at the token limit" if c["cut"] else ""
        text += f", {c['old']} with old model settings (Steps 35/37)" if c["old"] else ""
        rows = [Row("LLM runs", text, "warn" if c["cut"] or c["old"] else "info", ())]
    if need_llm and not c["usable"]:
        rows.append(Row("smoke", "no run yet shows that today's settings work on your account (3 calls)", "todo",
                        (_py("smoke_llm.py") + "      # the last line must say RESULT: OK; if it says PROBLEM, send me its last 5 lines",)))
    return rows


def g1_rows(root: str, tag: str, dataset: str) -> tuple:
    """(rows, to_run): the three baselines by the check of scripts/g1.py, then the diagnostics file."""
    all_parts = dataset_mod.load_dataset(dataset)
    rows, left = [], []
    for system, extra in diagnostics.BASELINES:
        state, why = g1.step_state(root, tag, system, g1.wanted_parts(extra, len(all_parts)))
        text = {"done": "done", "make": "not made yet", "redo": "to run again: " + why}[state]
        rows.append(Row(f"G1 {system}", text, "ok" if state == "done" else "todo", ()))
        if state != "done":
            left.append(system)
    cmd = _py("g1.py")
    rows = [r._replace(commands=(cmd + "      # the same command later continues; --dry-run shows the plan first",) if r.state == "todo" else ()) for r in rows]
    diag = os.path.join(root, "results", "diagnostics.txt")
    if os.path.isfile(diag):
        rows.append(Row("G1 diag", "results\\diagnostics.txt is there", "ok", ()))
    else:
        rows.append(Row("G1 diag", "results\\diagnostics.txt not written yet", "todo" if not left else "info",
                        (_py("diagnostics.py"),) if not left else ()))
    return rows, left


def _load_grid(root: str, path: str) -> dict:
    cfg = tables.load_config(path if os.path.isabs(path) else os.path.join(root, path))
    if not os.path.isabs(cfg["results_dir"]):
        cfg["results_dir"] = os.path.join(root, cfg["results_dir"])
    return cfg


def grid_counts(cfg: dict) -> dict:
    """{'cells', 'ok', 'pending', 'columns': first column that has runs to make or None} for a grid config."""
    loaded = tables.load_all(cfg)
    built = [r["id"] for r in cfg["rows"] if r["built"]]
    cells = [loaded["runs"][(rid, m["id"])] for rid in built for m in cfg["models"]]
    pending = tables.pending_runs(cfg, loaded)
    return {"cells": len(cells), "ok": sum(1 for c in cells if c["status"] == "ok"), "pending": len(pending),
            "first_column": pending[0]["column"] if pending else None, "columns": sorted({j["column"] for j in pending})}


def g2_rows(root: str, grid: str, g1_done: bool) -> tuple:
    """(rows, grid_complete)."""
    try:
        cfg = _load_grid(root, grid)
        n = grid_counts(cfg)
    except _ERRORS as exc:
        return [Row("G2 grid", f"cannot be read ({_win(grid)}): {exc}"[:300], "warn", ())], False
    if not n["pending"]:
        table = os.path.join(cfg["results_dir"], "tables", cfg["name"], "table4.md")
        rows = [Row("G2 grid", f"all {n['ok']} cells ok ({_win(grid)})", "ok", ())]
        if os.path.isfile(table):
            rows.append(Row("G2 table", f"results\\tables\\{cfg['name']}\\table4.md is there", "ok", ()))
        else:
            rows.append(Row("G2 table", "Table 4 not built yet", "todo", (_py("make_tables.py", "build", _win(grid)),
                                                                          "# then choose the 'decide' rows (A5, A6, A7) from the numbers: RUNBOOK.md, G2")))
        return rows, True
    first = n["first_column"]
    cmds = (_py("g2.py", _win(grid), "--dry-run") + "      # what is to run, and the calls it can make",
            _py("g2.py", _win(grid), "--only", str(first)) + "      # one column; if it stops, the same command continues")
    return [Row("G2 grid", f"{n['ok']} of {n['cells']} cells ok, {n['pending']} run(s) to make or run again ({_win(grid)})",
                "todo" if g1_done else "info", cmds if g1_done else ())], False


def freeze_row(root: str, g2_done: bool, final_exists: bool) -> Row:
    st = freeze.status(root)
    if st["status"] == "none":
        if not g2_done:
            return Row("freeze", "FREEZE.json: none yet (right: the freeze comes after G2)", "ok", ())
        if not final_exists:
            return Row("freeze", "G2 is complete: the final grid config is next, then the freeze", "todo",
                       ("copy configs\\ablation_template.json configs\\final.json      # edit it as in RUNBOOK.md, 'Freeze (G2 ...)'",))
        return Row("freeze", "not created yet", "todo",
                   (_py("freeze.py", "create", "--grid", "configs\\final.json", "--run-tests", '--note "Tier 1"'),
                    _py("freeze.py", "verify") + "      # then: git add FREEZE.json; git commit -m Freeze; git tag freeze-v1"))
    text = f"FREEZE.json is {st['status']}" + (f" (freeze {st['id']})" if st.get("id") else "")
    if not g2_done:
        return Row("freeze", text + ": it is there BEFORE G2 is complete. If you did not make it on purpose, remove it (git rm FREEZE.json, then commit)",
                   "warn", ())
    if st["status"] in ("drift", "unreadable"):
        return Row("freeze", text + f": run_eval will refuse the sealed set. {_py('freeze.py', 'verify')} says what differs", "warn", ())
    return Row("freeze", text, "ok", ())


def final_row(root: str, final: str, frozen: bool) -> Row:
    path = final if os.path.isabs(final) else os.path.join(root, final)
    if not os.path.isfile(path):
        return Row("G3/G4", f"{_win(final)} not made yet", "info", ())
    try:
        n = grid_counts(_load_grid(root, final))
    except _ERRORS as exc:
        return Row("G3/G4", f"cannot be read ({_win(final)}): {exc}"[:300], "warn", ())
    if not n["pending"]:
        return Row("G3/G4", f"all {n['ok']} cells ok ({_win(final)})", "ok", ())
    text = f"{n['ok']} of {n['cells']} cells ok, {n['pending']} run(s) to make ({_win(final)}); runs on the sealed set are made once, by hand"
    return Row("G3/G4", text, "todo" if frozen else "info",
               (_py("make_tables.py", "commands", _win(final)), _py("make_tables.py", "status", _win(final))) if frozen else ())


def release_row(root: str, final_done: bool) -> Row:
    folder = os.path.join(root, *RELEASE_DIR.split("/"))
    if os.path.isdir(folder):
        return Row("release", f"{_win(RELEASE_DIR)} is there (check it: python scripts\\release.py verify {_win(RELEASE_DIR)})", "ok", ())
    if not final_done:
        return Row("release", "not started (after G3/G4)", "info", ())
    cmds = (("copy configs\\release_template.json configs\\release_final.json",) if not os.path.isfile(os.path.join(root, *RELEASE_CONFIG.split("/"))) else ()) \
        + (_py("release.py", "check", _win(RELEASE_CONFIG)),)
    return Row("release", "not built yet", "todo", cmds)


# ---------------------------------------------------------------- everything
def build_rows(root: str, tag: str = diagnostics.DEFAULT_TAG, dataset: str = diagnostics.DEFAULT_DATASET, grid: str = GRID, final: str = FINAL) -> list:
    rows = []
    try:
        g1_rows_, left = g1_rows(root, tag, dataset)
    except _ERRORS as exc:
        g1_rows_, left = [Row("G1", f"cannot be worked out: {exc}"[:300], "warn", ())], ["?"]
    g1_done = not left
    g2_rows_, g2_done = g2_rows(root, grid, g1_done)
    final_exists = os.path.isfile(os.path.join(root, *final.replace("\\", "/").split("/")))
    frozen = freeze.status(root)["status"] in freeze.OK_STATUSES
    f_row = freeze_row(root, g2_done, final_exists)
    fin = final_row(root, final, frozen)
    long_runs_left = bool(left) or not g2_done                       # an LLM run is still to come: the key and the smoke test matter
    if long_runs_left:
        rows.append(key_row(root))
    rows += runs_rows(root, need_llm=long_runs_left)
    rows += g1_rows_ + g2_rows_ + [f_row, fin, release_row(root, fin.state == "ok")]
    return rows


def next_commands(rows: list) -> list:
    for r in rows:
        if r.state == "todo" and r.commands:
            return list(r.commands)
    return []


def format_rows(rows: list, nxt: list, root: str) -> str:
    tag = {"ok": "ok  ", "todo": "TODO", "warn": "WARN", "info": "    "}
    L = [f"MTP-1: where you are   ({os.path.basename(os.path.abspath(root))}; offline, nothing is run)"]
    for r in rows:
        line = f"  {tag[r.state]} {r.label:<9s} {r.text}"
        L.append(line if len(line) <= WIDTH else line[:WIDTH - 3] + "...")
    L.append("")
    if nxt:
        L.append("NEXT")
        L += [f"  {c}" for c in nxt]
        L.append("  (then  python scripts\\where.py  again)")
    elif any(r.state == "warn" for r in rows):
        L.append("NEXT  nothing to start, but read the WARN line(s) above.")
    else:
        L.append("NEXT  nothing left that this tool can see. What is left is yours: the 'decide' rows, the case-study commentary, the prose of the")
        L.append("      paper and slides, the Zenodo fields (RUNBOOK.md).")
    return "\n".join(L)


def report(root: str, **kw) -> str:
    rows = build_rows(root, **kw)
    return format_rows(rows, next_commands(rows), root)
