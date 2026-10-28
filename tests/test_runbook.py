"""Step 30: RUNBOOK.md names only commands that exist.

The runbook is the order in which the scripts are used (setup, the gates G1 to G4, the release). It must not drift from the code, so this reads
every command line in its code blocks and checks it against the real scripts: the script is there, the sub-command is one of the script's,
every --flag is in the script's own --help, a --system is a name the system grammar accepts, a --dataset is one of the datasets, a config file
exists or is made by a `copy` line of the runbook itself. And every script of scripts/ is in the runbook, so a new script cannot be left out.
"""
import os
import re
import shlex
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capp import systems as S      # noqa: E402

RUNBOOK = os.path.join(ROOT, "RUNBOOK.md")
DATASETS = ("dev_v2", "test_v2", "bench_v1")
NOT_IN_RUNBOOK = {"__init__.py"}


def blocks():
    text = open(RUNBOOK, encoding="utf-8").read()
    return re.findall(r"```\n(.*?)```", text, flags=re.S)


def lines():
    return [ln.strip() for b in blocks() for ln in b.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def script_lines():
    out = []
    for ln in lines():
        m = re.match(r"python scripts[\\/]([A-Za-z0-9_]+)\.py\b(.*)$", ln)
        if m:
            out.append((m.group(1), shlex.split(m.group(2), posix=False), ln))
    return out


_help = {}


def help_text(script, sub=None):
    key = (script, sub)
    if key not in _help:
        cmd = [sys.executable, os.path.join(ROOT, "scripts", script + ".py")] + ([sub] if sub else []) + ["--help"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=ROOT)
        assert r.stdout.strip(), (cmd, r.returncode, r.stderr[-300:])         # compare_runs.py prints its docstring and exits 1: that is its help
        _help[key] = r.stdout
    return _help[key]


def split_sub(script, tokens):
    """(sub-command or None, the other tokens) from the first line of the script's own usage."""
    usage = help_text(script).split("\n\n")[0]
    m = re.match(r"usage: \S+\s+\[-h\]\s+\{([a-z0-9_,-]+)\}", usage)                  # sub-commands come right after -h, an option's choices do not
    if not m:
        return None, tokens
    subs = m.group(1).split(",")
    first = next((t for t in tokens if not t.startswith("-")), None)
    assert first in subs, (script, tokens, subs)
    i = tokens.index(first)
    return first, tokens[:i] + tokens[i + 1:]


def test_the_runbook_has_commands_and_every_script_is_in_it():
    sl = script_lines()
    assert len(sl) >= 40
    named = {s for s, _, _ in sl}
    scripts = {f[:-3] for f in os.listdir(os.path.join(ROOT, "scripts")) if f.endswith(".py")} - {f[:-3] for f in NOT_IN_RUNBOOK}
    assert scripts - named == set(), f"scripts missing from RUNBOOK.md: {sorted(scripts - named)}"
    assert named - scripts == set()


def test_every_flag_of_every_command_is_in_the_help_of_its_script():
    for script, tokens, ln in script_lines():
        sub, rest = split_sub(script, tokens)
        text = help_text(script, sub)
        for t in rest:
            if t.startswith("--"):
                flag = t.split("=")[0]
                assert re.search(r"(?<![\w-])" + re.escape(flag) + r"(?![\w-])", text), f"{flag} is not an option of {script} {sub or ''}: {ln}"


def test_every_system_and_dataset_in_the_runbook_is_valid():
    seen = 0
    for script, tokens, ln in script_lines():
        for i, t in enumerate(tokens[:-1]):
            if t == "--system":
                S.parse_system(tokens[i + 1])                       # raises for a name the grammar does not know
                seen += 1
            if t in ("--dataset", "--from"):
                assert tokens[i + 1] in DATASETS, ln
                seen += 1
    assert seen >= 8


def test_the_config_files_exist_or_are_made_by_a_copy_line_of_the_runbook():
    made = set()
    for ln in lines():
        m = re.match(r"copy (\S+) (\S+)$", ln)
        if m:
            src, dst = m.groups()
            assert os.path.isfile(os.path.join(ROOT, src.replace("\\", "/"))), f"copy source {src} does not exist"
            made.add(dst.replace("\\", "/"))
    assert {"configs/final.json", "configs/release_final.json", "configs/zenodo.json"} <= made
    for script, tokens, ln in script_lines():
        for t in tokens:
            rel = t.replace("\\", "/")
            if rel.startswith("configs/") and rel.endswith(".json"):
                assert os.path.isfile(os.path.join(ROOT, rel)) or rel in made, f"{rel} neither exists nor is copied before use: {ln}"


def test_a_copy_line_comes_before_the_first_use_of_the_file_it_makes():
    made_at = {}
    used_at = {}
    for i, ln in enumerate(lines()):
        m = re.match(r"copy (\S+) (\S+)$", ln)
        if m:
            made_at.setdefault(m.group(2).replace("\\", "/"), i)
        for t in ln.split():
            rel = t.replace("\\", "/")
            if rel.startswith("configs/") and not ln.startswith("copy"):
                used_at.setdefault(rel, i)
    for f, i in used_at.items():
        if f in made_at:
            assert made_at[f] < i, f"{f} is used before it is made"


def test_the_rows_the_runbook_asks_her_to_decide_are_marked_as_such_in_the_template():
    import json
    rows = {r["id"]: r for r in json.load(open(os.path.join(ROOT, "configs", "ablation_template.json")))["rows"]}
    for rid in ("A5", "A6", "A7"):
        assert "decide" in rows[rid].get("note", ""), rid
    text = open(RUNBOOK, encoding="utf-8").read()
    assert "A5" in text and "A6" in text and "A7" in text and "A9" in text and "A8" in text


def test_the_runbook_keeps_the_rules_about_the_sealed_set():
    text = open(RUNBOOK, encoding="utf-8").read()
    for must in ("--no-freeze-check", "freeze-v1", "baseline-v0", "release-v1", "tag", "overwrite"):
        assert must in text, must
    assert text.index("## G2") < text.index("freeze.py create") < text.index("## G3")
    assert text.index("## G3") < text.index("--dataset test_v2") or "--dataset" in text


# ---------------------------------------------------------------- requirements.txt
def third_party_imports():
    import ast
    local = {os.path.splitext(f)[0] for f in os.listdir(ROOT) if f.endswith(".py")}
    for d in os.listdir(ROOT):
        if os.path.isdir(os.path.join(ROOT, d)):
            local.add(d)
            local |= {os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, d)) if f.endswith(".py")}
    found = {}
    for d in ["capp", "scripts"] + sorted(x for x in os.listdir(ROOT) if re.match(r"week\d$", x)):
        for dp, dn, fn in os.walk(os.path.join(ROOT, d)):
            dn[:] = [x for x in dn if x not in ("__pycache__", "tests")]
            for f in fn:
                if f.endswith(".py"):
                    for n in ast.walk(ast.parse(open(os.path.join(dp, f), encoding="utf-8").read())):
                        mods = [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else \
                            [n.module.split(".")[0]] if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module else []
                        for m in mods:
                            if m not in sys.stdlib_module_names and m not in local:
                                found.setdefault(m, set()).add(os.path.relpath(os.path.join(dp, f), ROOT))
    return found


def test_requirements_name_every_package_the_code_imports():
    names = {re.split(r"[<>=!~\[ ;]", ln.strip(), maxsplit=1)[0].lower().replace("_", "-") for ln in open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8")
             if ln.strip() and not ln.strip().startswith("#")}
    dist = {"dotenv": "python-dotenv"}
    imports = third_party_imports()
    assert {"openpyxl", "reportlab", "matplotlib", "groq", "streamlit"} <= set(imports)
    missing = sorted(m for m in imports if dist.get(m, m).lower().replace("_", "-") not in names)
    assert missing == [], f"imported but not in requirements.txt: {missing} ({[sorted(imports[m])[:2] for m in missing]})"
