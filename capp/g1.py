"""The G1 baselines in one command (MTP-1 Step 34).

    python scripts/g1.py                  # rb, b1, b0 on dev_v2 with tag dev, then results/diagnostics.txt
    python scripts/g1.py --dry-run        # what it would do, nothing is run
    python scripts/g1.py --yes            # do not ask before the runs that call the API

It runs the same three commands as RUNBOOK.md, G1 (the list is capp/diagnostics.BASELINES, so the two cannot differ), one after the other,
with the same options, and then writes the diagnostics file. What it adds is only care:

  * a run that is there and complete is not run again (records are there, the number of parts is the one wanted, no API error in them);
    a run with API errors or the wrong number of parts is run again with the same command, and the cache makes that cost only the calls that
    are missing (capp/apierrors.py);
  * before a run that calls the API it prints the plan (capp/runplan.py: the most calls it can make, and tokens, cost and time from your
    earlier runs) and asks; --yes skips the question;
  * it does not start an API run without GROQ_API_KEY, and says where to put it;
  * the first run that stops (an error, an answer of no, Ctrl-C) stops the sequence, the diagnostics file is still written, and what is
    cached is not lost: the same command later continues.

It starts only run_eval.py (the three G1 systems) and diagnostics.py; it changes nothing else, and no FREEZE.json is touched.
"""
import os
import shlex
import subprocess
import sys

from capp import apierrors, dataset as dataset_mod, diagnostics, harness, runplan


class G1Error(ValueError):
    """A wrong argument, in words for the author."""


def _limit_of(extra: str):
    toks = shlex.split(extra)
    return int(toks[toks.index("--limit") + 1]) if "--limit" in toks else None


def wanted_parts(extra: str, dataset_size: int) -> int:
    limit = _limit_of(extra)
    return min(limit, dataset_size) if limit else dataset_size


def step_state(root: str, tag: str, system: str, wanted: int) -> tuple:
    """("done", "") | ("make", "") | ("redo", why) for one baseline run folder."""
    folder = diagnostics._run_dir(root, tag, system)
    if not diagnostics._has_records(folder):
        return "make", ""
    try:
        records = harness.load_records(folder)
        n_api = apierrors.in_records(records)
    except (OSError, ValueError, TypeError, AttributeError):
        return "redo", "its records cannot be read"
    if len(records) != wanted:
        return "redo", f"it has {len(records)} record(s), {wanted} wanted"
    if n_api:
        return "redo", f"{n_api} API error(s) in it (rate limit or outage, not the model)"
    return "done", ""


def command(system: str, extra: str, tag: str, dataset: str, max_calls: int = None) -> list:
    argv = ["scripts/run_eval.py", "--system", system, "--dataset", dataset, "--tag", tag, *shlex.split(extra)]
    if max_calls is not None and system != "rb":
        argv += ["--max-calls", str(max_calls)]
    return argv


def show(argv: list) -> str:
    return "python " + " ".join(shlex.quote(a) if " " in a else a for a in argv)


def default_runner(argv: list, cwd: str) -> int:
    return subprocess.run([sys.executable, *argv], cwd=cwd).returncode


def default_ask(question: str) -> bool:
    try:
        return input(question + " [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def run_g1(root: str, tag: str = diagnostics.DEFAULT_TAG, dataset: str = diagnostics.DEFAULT_DATASET, yes: bool = False,
           dry_run: bool = False, max_calls: int = None, diag: bool = True, runner=default_runner, ask=default_ask,
           load_parts=dataset_mod.load_dataset, out=print) -> int:
    """Returns 0 when all three baselines are there at the end, 1 when the sequence stopped before that (the reason is printed)."""
    if max_calls is not None and (isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1):
        raise G1Error(f"--max-calls must be a whole number of 1 or more, not {max_calls!r}")
    try:
        all_parts = load_parts(dataset)
    except (OSError, ValueError) as exc:
        raise G1Error(f"dataset {dataset}: {exc}") from None
    out(f"G1 baselines on {dataset} (tag {tag}): " + ", ".join(s for s, _ in diagnostics.BASELINES))
    out(f"{diagnostics.KEY_NAME}: " + ("set" if diagnostics._key_state(root) != "NO" else "NOT set (rb needs none; b1 and b0 do)"))
    todo = []
    for system, extra in diagnostics.BASELINES:
        wanted = wanted_parts(extra, len(all_parts))
        state, why = step_state(root, tag, system, wanted)
        out(f"  {system:<3s}: " + {"done": "done, not run again", "make": "to make", "redo": "to run again: " + why}[state])
        if state != "done":
            todo.append((system, extra, wanted))
    bound = sum(runplan.plan(system, all_parts[:wanted], throttle=_throttle_of(extra))["max_calls"] for system, extra, wanted in todo)
    if bound:
        out(f"API calls in this command: at most {bound} (answers already in the cache are free)")
    stopped = ""
    for system, extra, wanted in todo:
        argv = command(system, extra, tag, dataset, max_calls)
        out(f"\n=== {system}: {show(argv)}")
        if dry_run:
            if system != "rb":
                parts = all_parts[:wanted]
                p = runplan.estimate(runplan.plan(system, parts, throttle=_throttle_of(extra)), runplan.history(os.path.join(root, "results")))
                out(runplan.format_plan(p))
            continue
        if system != "rb":
            if diagnostics._key_state(root) == "NO":
                stopped = (f"{diagnostics.KEY_NAME} is not set. Put {diagnostics.KEY_NAME}=... in the .env file in the repository folder "
                           "(or set the environment variable), then run this command again; the runs that are done are not repeated.")
                break
            parts = all_parts[:wanted]
            p = runplan.estimate(runplan.plan(system, parts, throttle=_throttle_of(extra)), runplan.history(os.path.join(root, "results")))
            out(runplan.format_plan(p))
            if not yes and not ask(f"Run {system} now? This calls the API."):
                stopped = f"stopped before {system}: not confirmed. Run this command again when you want to go on."
                break
        try:
            code = runner(argv, root)
        except KeyboardInterrupt:
            out(f"\nInterrupted during {system}. What was answered is in the cache: run this command again to go on.")
            return 1
        if code != 0:
            stopped = (f"{system} stopped (exit code {code}); the reason is printed just above. "
                       "The answers made so far are in the cache: fix that, then run this command again.")
            break
    if stopped:
        out("\n" + stopped)
    if dry_run:
        out("\nDry run: nothing was run.")
        return 0
    if diag:
        out("\n=== diagnostics: python scripts/diagnostics.py")
        code = runner(["scripts/diagnostics.py", "--tag", tag, "--dataset", dataset], root)          # it prints where the file is
        if code != 0:
            out(f"diagnostics.py failed (exit code {code}); its message is just above.")
    left = [s for s, extra in diagnostics.BASELINES if step_state(root, tag, s, wanted_parts(extra, len(all_parts)))[0] != "done"]
    out("All three baselines are there." if not left else "Still to do: " + ", ".join(left))
    return 0 if not left else 1


def _throttle_of(extra: str) -> float:
    toks = shlex.split(extra)
    return float(toks[toks.index("--throttle") + 1]) if "--throttle" in toks else 0.0
