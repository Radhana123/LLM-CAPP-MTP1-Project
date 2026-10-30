"""The G2 ablation grid on the development set in one command (MTP-1 Step 40).

    python scripts/g2.py configs/ablation_template.json --dry-run     # the runs still to make, with the plan of each column; nothing is run
    python scripts/g2.py configs/ablation_template.json               # run them, one column at a time, asking before each column
    python scripts/g2.py configs/ablation_template.json --only qwen   # one column only ("shared" = the rows that choose their own models)
    python scripts/g2.py configs/ablation_template.json --only qwen --rows A0 A1 A2 A3 A4 A5 A6    # only these rows (the order stays the grid's)

It runs the lines that `python scripts/make_tables.py commands <config>` prints (capp/tables.py pending_runs is the one list behind both, so
they cannot differ), in that order, with CAPP_MODEL_PRIMARY set for each column the way the PowerShell lines do. What it adds is only care,
the same care as scripts/g1.py:

  * a run that is present and usable is not run again; one that is missing, incomplete, cut off at the token limit, made with other model
    settings than today's, or has API errors is run (again) with the same --tag, and the cache makes that cost only the calls that are missing;
  * before the runs of a column that call the API it prints the plan (capp/runplan.py: the most calls they can make, and tokens, cost and
    time from your earlier runs) and asks once for the column; --yes skips the question;
  * it does not start an API run without GROQ_API_KEY, and says where to put it;
  * the first run that stops (an error, --max-calls reached, an answer of no, Ctrl-C) stops the sequence; what is cached is not lost: the same
    command later continues. After each run it says whether the grid now counts that run, and at the end it prints the status grid.

It refuses a sealed dataset (the final grid is not run from here: the freeze comes first, RUNBOOK.md). It starts only scripts/run_eval.py and
changes nothing else.
"""
import contextlib
import os
import subprocess
import sys

from capp import config, dataset as dataset_mod, diagnostics, laststop, runplan, systems, tables


class G2Error(ValueError):
    """A wrong argument or config, in words for the author."""


def default_runner(argv: list, cwd: str, model) -> int:
    """Runs `python <argv>` in cwd with CAPP_MODEL_PRIMARY = model (None: the variable is removed, as the shared rows need)."""
    env = dict(os.environ)
    if model is None:
        env.pop("CAPP_MODEL_PRIMARY", None)
    else:
        env["CAPP_MODEL_PRIMARY"] = model
    return subprocess.run([sys.executable, *argv], cwd=cwd, env=env).returncode


def default_ask(question: str) -> bool:
    try:
        return input(question + " [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


@contextlib.contextmanager
def _primary(model):
    """CAPP_MODEL_PRIMARY = model (None: removed) for the planning of one column, then as it was."""
    before = os.environ.get("CAPP_MODEL_PRIMARY")
    try:
        if model is None:
            os.environ.pop("CAPP_MODEL_PRIMARY", None)
        else:
            os.environ["CAPP_MODEL_PRIMARY"] = model
        yield
    finally:
        if before is None:
            os.environ.pop("CAPP_MODEL_PRIMARY", None)
        else:
            os.environ["CAPP_MODEL_PRIMARY"] = before


def uses_api(job: dict) -> bool:
    return systems.parse_system(job["row"]["system"]) is not None


def plan_of(job: dict, parts: list, hist: dict) -> dict:
    """runplan's plan (with the estimate from earlier runs) of one job, for the model of its column."""
    with _primary(job["model"]):
        return runplan.estimate(runplan.plan(job["row"]["system"], parts, repeats=job["row"]["repeats"]), hist)


def totals(plans: list) -> dict:
    """{"calls": bound, "usd": sum | None, "seconds": sum | None, "tokens": sum | None}: each is None as soon as one plan lacks its estimate (no
    earlier run with its model in results/; for usd also no price); nothing is guessed. tokens = in + out, as the quota counts them."""
    usd, sec, tok, known, tok_known = 0.0, 0.0, 0.0, True, True
    for p in plans:
        if not p["uses_llm"]:
            continue
        rows = p.get("estimate") or []
        if not rows or any("note" in e for e in rows) or any("tokens_in" not in e or "tokens_out" not in e for e in rows):
            tok_known = False
        else:
            tok += sum(e["tokens_in"] + e["tokens_out"] for e in rows)
        if not rows or any("note" in e or e.get("usd") is None for e in rows):
            known = False
            continue
        usd += sum(e["usd"] for e in rows)
        sec += p["estimate_seconds"]
    return {"calls": sum(p["max_calls"] for p in plans), "usd": usd if known else None, "seconds": sec if known else None,
            "tokens": tok if tok_known else None}


def totals_text(t: dict, per_day=None) -> str:
    """per_day: the tokens per day of the quota of the model of these runs (None: no days are shown)."""
    if not t["calls"]:
        return "no API call"
    text = f"API calls: at most {t['calls']}"
    if t["usd"] is None:
        text += " (no estimate of cost and time: an earlier run with that model and a price in capp/config.py are missing)"
    else:
        text += f", about ${t['usd']:.2f} at list price [TO CONFIRM] and {t['seconds'] / 60:.0f} min if every call is live"
    if t.get("tokens") is not None:
        text += f"; about {t['tokens'] / 1000:.0f}K tokens"
        if per_day:
            text += f" = {t['tokens'] / per_day:.1f} day(s) of the free quota ({per_day / 1000:.0f}K tokens/day) [TO CONFIRM]"
    return text


def _select(cfg: dict, jobs: list, only=None, rows=None) -> list:
    """The jobs of the columns in `only` (empty: all) and of the rows in `rows` (empty: all), in the order of `jobs`."""
    if only:
        known = [m["id"] for m in cfg["models"]] + [tables.SHARED]
        bad = [o for o in only if o not in known]
        if bad:
            raise G2Error(f"--only {', '.join(bad)}: no such column. Columns of this grid: {', '.join(known)}")
        jobs = [j for j in jobs if j["column"] in only]
    if rows:
        ids = [r["id"] for r in cfg["rows"]]
        bad = [x for x in rows if x not in ids]
        if bad:
            raise G2Error(f"--rows {', '.join(bad)}: no such row. Rows of this grid: {', '.join(ids)}")
        unbuilt = [x for x in rows if not next(r for r in cfg["rows"] if r["id"] == x)["built"]]
        if unbuilt:
            raise G2Error(f"--rows {', '.join(unbuilt)}: not built (the row has no system yet), so there is nothing to run")
        jobs = [j for j in jobs if j["row"]["id"] in rows]
    return jobs


def _per_day(column: str):
    """Tokens per day of the quota of a column's model; None for `shared` (its rows choose their own models, so there is no one quota)."""
    return None if column == tables.SHARED else config.QUOTA_TOKENS_PER_DAY


def _name(job: dict) -> str:
    return f"{job['row']['id']} {job['row']['system']}"


def _cell_text(cfg: dict, loaded: dict, job: dict) -> str:
    """What the grid says about the run of this job now ('ok', or its status and why). `loaded` only gives the folder and the parts every run
    must cover; the run itself is read from disk again."""
    column = cfg["models"][0] if job["column"] == tables.SHARED else next(m for m in cfg["models"] if m["id"] == job["column"])
    run = tables.load_run(cfg, job["row"], column, loaded["results_dir"], loaded["expected"], {})
    detail = "; ".join(run["errors"] + (run["warnings"] if run["status"] == "partial" else []))[:200]
    return run["status"] + (f": {detail}" if detail else "")


def run_g2(cfg: dict, root: str, only=None, yes: bool = False, dry_run: bool = False, max_calls: int = None, runner=default_runner,
           ask=default_ask, out=print, rows=None) -> int:
    """Returns 0 when no run of the selection is left to make at the end (or after --dry-run), 1 when something is left."""
    if max_calls is not None and (isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1):
        raise G2Error(f"--max-calls must be a whole number of 1 or more, not {max_calls!r}")
    if dataset_mod.is_sealed(cfg["dataset"]):
        raise G2Error(f"{cfg['dataset']} is sealed (final runs only): the final grid is run after the freeze, not from here "
                      "(RUNBOOK.md, Freeze). The development grid is run on dev_v2.")
    try:
        all_parts = dataset_mod.load_dataset(cfg["dataset"])
    except (OSError, ValueError) as exc:
        raise G2Error(f"dataset {cfg['dataset']}: {exc}") from None
    parts = all_parts[cfg["offset"]:]
    if cfg["limit"]:
        parts = parts[:cfg["limit"]]
    try:
        wait = config.quota_wait()
    except ValueError as exc:
        raise G2Error(str(exc)) from None
    loaded = tables.load_all(cfg)
    selection = _select(cfg, tables.pending_runs(cfg, loaded), only, rows)
    key_ok = diagnostics._key_state(root) != "NO"
    out(f"G2 grid {cfg['name']} on {cfg['dataset']}: {len(cfg['rows'])} rows x {len(cfg['models'])} LLMs, {cfg['repeats']} repeat(s) by default")
    out(f"{diagnostics.KEY_NAME}: " + ("set" if key_ok else "NOT set (needed for every run except the Route Builder)"))
    if wait:
        out(f"Quota slow mode is on (CAPP_QUOTA_WAIT={wait:g}): a call waits up to {wait:g} s for the quota instead of stopping. "
            "Keep the laptop on and awake.")
    if not selection:
        chosen = list(only or []) + list(rows or [])
        out("\nNothing to run: " + (f"{', '.join(chosen)} has nothing to make." if chosen else "every run of the grid is present and complete."))
        out("\n" + tables.status_grid(cfg, loaded))
        return 0
    hist = runplan.history(loaded["results_dir"])
    plans = [plan_of(j, parts, hist) for j in selection]
    out(f"\nRuns to make or run again: {len(selection)}")
    for i, (job, p) in enumerate(zip(selection, plans), 1):
        who = job["column"] if job["column"] != tables.SHARED else "shared"
        out(f"  [{i:>2d}] {who:<8s} {_name(job):<36s} " + ("make" if not job["why"] else "again: " + job["why"][:110])
            + (f"   (<= {p['max_calls']} calls)" if p["uses_llm"] else "   (no API call)"))
    groups = []
    for job, p in zip(selection, plans):
        if not groups or groups[-1][0] != job["column"]:
            groups.append((job["column"], []))
        groups[-1][1].append((job, p))
    out("")
    for column, items in groups:
        out(f"  {column}: {len(items)} run(s), " + totals_text(totals([p for _, p in items]), _per_day(column)))
    out("  (Each bound counts its run alone with an empty cache. Runs that send the same prompts share the cache of one tag, so fewer calls "
        "are really made. Tokens and days use the answers of your earlier calls; two-step (+ts) answers are longer, so they can be higher.)")
    if dry_run:
        out("\nDry run: nothing was run.")
        return 0
    n_done, stopped, total = 0, "", len(selection)
    for column, items in groups:
        calls = totals([p for _, p in items])
        if calls["calls"]:
            if not key_ok:
                stopped = (f"{diagnostics.KEY_NAME} is not set. Put {diagnostics.KEY_NAME}=... in the .env file in the repository folder "
                           "(or set the environment variable), then run this command again; the runs that are done are not repeated.")
                break
            if not yes and not ask(f"\nRun the {len(items)} run(s) of {column} now? This calls the API ({totals_text(calls, _per_day(column))})."):
                stopped = f"stopped before {column}: not confirmed. Run this command again when you want to go on."
                break
        for job, p in items:
            n_done += 1
            argv = list(job["argv"]) + (["--max-calls", str(max_calls)] if max_calls is not None and uses_api(job) else [])
            out(f"\n=== [{n_done}/{total}] {column} {_name(job)}: python {' '.join(argv)}")
            laststop.clear(loaded["results_dir"])
            try:
                code = runner(argv, root, job["model"])
            except KeyboardInterrupt:
                out(f"\nInterrupted during {_name(job)}. What was answered is in the cache: run this command again to go on.")
                return 1
            if code != 0:
                why = laststop.read(loaded["results_dir"])
                stopped = (f"{_name(job)} ({column}) stopped (exit code {code}). "
                           + (f"Why: {why}" if why else "The reason is printed above the status grid; to see only that run, start it alone: "
                              f"python {' '.join(argv)}")
                           + "\nThe answers made so far are in the cache: fix that, then run this command again.")
                break
            out(f"--- {_name(job)}: the grid now says " + _cell_text(cfg, loaded, job))
        if stopped:
            break
    loaded = tables.load_all(cfg)
    left = _select(cfg, tables.pending_runs(cfg, loaded), only, rows)
    out("\n" + tables.status_grid(cfg, loaded))
    if stopped:                                               # under the grid: the end of the output is what stays on the screen
        out("\n" + stopped)
    if left:
        out("Still to do: " + ", ".join(f"{j['column']}/{_name(j)}" for j in left[:12]) + (" ..." if len(left) > 12 else "")
            + ". Run this command again; python scripts/make_tables.py status <config> says why.")
        return 1
    out("Nothing is left to run in this selection. Next: python scripts/make_tables.py build <config>")
    return 0
