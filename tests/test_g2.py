"""Step 40: the G2 ablation grid in one command (capp/g2.py, scripts/g2.py, capp/tables.py pending_runs / run_argv).

Offline. The runner that stands in for `python scripts/run_eval.py ...` really makes the run folder with the real harness and a fake planner
(three parts, no API call), so what is checked is the whole loop: which runs are listed, in which order and with which model, what is asked,
where the sequence stops, what the grid says after each run, and that a second call continues where the first stopped.
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from capp import config, dataset, g2 as G, harness, runplan, systems, tables as T, verify as V      # noqa: E402
import g2 as g2_script                                                                           # noqa: E402
import route_builder as rb                                                                       # noqa: E402

N = 3
QWEN, GPT = "qwen/qwen3.8-27b", "openai/gpt-oss-120b"
PARTS = dataset.load_dataset("dev_v2")[:N]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("GROQ_API_KEY", "CAPP_QWEN_REASONING", "CAPP_MODEL_PRIMARY"):
        monkeypatch.delenv(name, raising=False)


def make_cfg(results_dir, **over):
    cfg = {"name": "g2t", "dataset": "dev_v2", "limit": N, "repeats": 1, "results_dir": str(results_dir), "resamples": 100, "seed": 7,
           "shared_tag": "g_shared",
           "models": [{"id": "a", "label": "Model A", "model": QWEN, "tag": "g_a"},
                      {"id": "b", "label": "Model B", "model": GPT, "tag": "g_b"}],
           "rows": [{"id": "A0", "label": "B1", "system": "b1"},
                    {"id": "A1", "label": "B0", "system": "b0", "parent": "A0"},
                    {"id": "A2", "label": "A1 + R1", "system": "b0_r1", "parent": "A1"},
                    {"id": "A3", "label": "A2 + selection", "system": "n5_r1_s3", "parent": "A2"},
                    {"id": "A4", "label": "A3 small model", "system": "n5_r1_s3+ms", "parent": "A3"},
                    {"id": "A5", "label": "not built", "system": None, "parent": "A3"}]}
    cfg.update(over)
    return T.validate_config(cfg)


def planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
    routes = rb.generate_valid_routes(features, max_routes=15)
    if not routes:
        return {"success": False, "steps": [], "raw_response": ""}
    return {"success": True, "steps": V.clean_steps(routes[sample_id % len(routes)]["steps"]), "raw_response": "", "model": "fake",
            "meta": {"cached": True, "finish_reason": "stop", "prompt_tokens": 100, "completion_tokens": 20, "latency_s": 0.1}}


def arg(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


class Runner:
    """Stands in for `python <argv>`: records (argv, cwd, model) and makes the run folder with the real harness and a fake planner."""

    def __init__(self, results, codes=None, interrupt=None, make=True, after=None):
        self.results, self.codes, self.interrupt, self.make, self.after, self.calls = str(results), codes or {}, interrupt, make, after, []

    def __call__(self, argv, cwd, model):
        self.calls.append((list(argv), cwd, model))
        system = arg(argv, "--system")
        if self.interrupt == system:
            raise KeyboardInterrupt
        code = self.codes.get(system, 0)
        if code == 0 and self.make:
            mp = pytest.MonkeyPatch()
            try:
                if model is None:
                    mp.delenv("CAPP_MODEL_PRIMARY", raising=False)
                else:
                    mp.setenv("CAPP_MODEL_PRIMARY", model)
                kw = {"planner_factory": lambda part: planner} if "+ms" in system else {"planner": planner}
                harness.run_experiment(system, PARTS, repeats=int(arg(argv, "--repeats")), tag=arg(argv, "--tag"), out_dir=self.results,
                                       dataset_name="dev_v2", progress=None, analysis=False, **kw)
            finally:
                mp.undo()
        if self.after:
            self.after(system, code)
        return code

    @property
    def systems(self):
        return [arg(c[0], "--system") for c in self.calls]

    @property
    def tags(self):
        return [(arg(c[0], "--tag"), arg(c[0], "--system")) for c in self.calls]


class Say:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        self.lines.append(str(text))

    @property
    def text(self):
        return "\n".join(self.lines)


def go(tmp_path, cfg=None, runner=None, key=True, answers=None, **kw):
    results = tmp_path / "results"
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = cfg or make_cfg(results)
    if key:
        (tmp_path / ".env").write_text("GROQ_API_KEY=abc\n", encoding="utf-8")
    runner = runner or Runner(results)
    say, asked = Say(), []
    queue = list(answers) if answers is not None else None

    def ask(question):
        asked.append(question)
        return True if queue is None else (queue.pop(0) if queue else False)
    code = G.run_g2(cfg, str(tmp_path), runner=runner, ask=ask, out=say, **kw)
    return code, runner, say, asked, cfg


# Within a column the runs that send the same prompts go together, the one with most samples first (capp/tables.py _cache_key): the later ones
# find their answers in the cache. b1 has one sample, so it comes last.
FULL_ORDER = [("g_a", "b0"), ("g_a", "b0_r1"), ("g_a", "n5_r1_s3"), ("g_a", "b1"),
              ("g_b", "b0"), ("g_b", "b0_r1"), ("g_b", "n5_r1_s3"), ("g_b", "b1"), ("g_shared", "n5_r1_s3+ms")]


# ================================================================ tables.pending_runs / run_argv
def test_the_pending_runs_are_the_missing_cells_per_column_then_the_shared_rows(tmp_path):
    cfg = make_cfg(tmp_path / "results")
    jobs = T.pending_runs(cfg, T.load_all(cfg))
    assert [j["column"] for j in jobs] == ["a"] * 4 + ["b"] * 4 + ["shared"]
    assert [(j["tag"], j["row"]["system"]) for j in jobs] == FULL_ORDER
    assert {j["row"]["id"] for j in jobs} == {"A0", "A1", "A2", "A3", "A4"}                 # A5 is not built
    assert [j["model"] for j in jobs] == [QWEN] * 4 + [GPT] * 4 + [None]
    assert [j["tag"] for j in jobs] == ["g_a"] * 4 + ["g_b"] * 4 + ["g_shared"]
    assert all(j["why"] == "" for j in jobs)


def test_the_commands_text_is_the_same_runs_as_the_pending_list(tmp_path):
    cfg = make_cfg(tmp_path / "results")
    jobs = T.pending_runs(cfg, T.load_all(cfg))
    lines = [x for x in T.run_commands(cfg, T.load_all(cfg)) if x.startswith("python ")]
    assert lines == ["python " + " ".join(j["argv"]) for j in jobs]
    text = "\n".join(T.run_commands(cfg, T.load_all(cfg)))
    assert f'$env:CAPP_MODEL_PRIMARY = "{QWEN}"' in text and f'$env:CAPP_MODEL_PRIMARY = "{GPT}"' in text
    assert "Remove-Item Env:CAPP_MODEL_PRIMARY" in text and "# 9 run(s) for g2t" in text


def test_run_argv_names_the_run_the_way_the_grid_will_read_it(tmp_path):
    cfg = make_cfg(tmp_path / "results", offset=2, repeats=4)
    row = next(r for r in cfg["rows"] if r["id"] == "A1")
    assert T.run_argv(cfg, row, "g_a") == ["scripts/run_eval.py", "--system", "b0", "--dataset", "dev_v2", "--tag", "g_a", "--repeats", "4",
                                           "--limit", str(N), "--offset", "2", "--out", str(tmp_path / "results")]


def test_run_argv_leaves_out_what_is_default(tmp_path):
    cfg = make_cfg("results", limit=None)
    row = next(r for r in cfg["rows"] if r["id"] == "A1")
    assert T.run_argv(cfg, row, "g_a") == ["scripts/run_eval.py", "--system", "b0", "--dataset", "dev_v2", "--tag", "g_a", "--repeats", "1"]


def test_a_few_shot_row_gets_its_pool_only_when_the_pool_is_not_the_default(tmp_path):
    rows = [{"id": "A0", "label": "fs", "system": "b0_r1+fs2"}]
    row = lambda c: c["rows"][0]                                                              # noqa: E731
    assert "--pool" not in T.run_argv(make_cfg("results", rows=rows), row(make_cfg("results", rows=rows)), "t")
    c = make_cfg("results", rows=rows, pool="other_dev")
    assert T.run_argv(c, row(c), "t")[-2:] == ["--pool", "other_dev"]


def test_a_missing_partial_or_unusable_run_is_pending_with_its_reason(tmp_path):
    results = tmp_path / "results"
    cfg = make_cfg(results)
    go(tmp_path, cfg=cfg)                                                                     # everything is made
    assert T.pending_runs(cfg, T.load_all(cfg)) == []
    folder = results / "g_a_b0"
    info = json.load(open(folder / "run.json", encoding="utf-8"))
    info["llm_client_summary"]["truncated"] = 5
    json.dump(info, open(folder / "run.json", "w", encoding="utf-8"))
    lines = open(results / "g_b_b1" / "records.jsonl", encoding="utf-8").read().splitlines()
    open(results / "g_b_b1" / "records.jsonl", "w", encoding="utf-8").write("\n".join(lines[:-1]) + "\n")
    jobs = T.pending_runs(cfg, T.load_all(cfg))
    why = {(j["column"], j["row"]["system"]): j["why"] for j in jobs}
    assert set(why) == {("a", "b0"), ("b", "b1")}
    assert "cut off at the token limit" in why[("a", "b0")] and why[("a", "b0")].startswith("existing folder is unusable")
    assert why[("b", "b1")].startswith("incomplete")


# ================================================================ the dry run
def test_a_dry_run_lists_the_runs_and_the_plan_of_each_column_and_runs_nothing(tmp_path):
    code, runner, say, asked, cfg = go(tmp_path, dry_run=True)
    assert code == 0 and runner.calls == [] and asked == []
    assert not (tmp_path / "results").exists()
    assert "Runs to make or run again: 9" in say.text and "Dry run: nothing was run." in say.text
    for column in ("a", "b", "shared"):
        assert any(line.strip().startswith(column + ":") and "API calls: at most" in line for line in say.lines)
    assert say.text.count("make   (<=") == 9
    assert "share the cache of one tag" in say.text


def test_a_dry_run_names_the_key_state_and_does_not_need_one(tmp_path):
    code, runner, say, _, _ = go(tmp_path, key=False, dry_run=True)
    assert code == 0 and "GROQ_API_KEY: NOT set" in say.text
    code, runner, say, _, _ = go(tmp_path, key=True, dry_run=True)
    assert "GROQ_API_KEY: set" in say.text


def test_the_bound_of_a_column_is_the_sum_of_its_runs_for_the_model_of_that_column(tmp_path):
    cfg = make_cfg(tmp_path / "results")
    jobs = T.pending_runs(cfg, T.load_all(cfg))
    plans = [G.plan_of(j, PARTS, {}) for j in jobs]
    assert [p["tiers"][0]["model"] for p in plans[:4]] == [QWEN] * 4 and [p["tiers"][0]["model"] for p in plans[4:8]] == [GPT] * 4
    assert [t["model"] for t in plans[8]["tiers"]] == [config.model("small")]                  # +ms: its own model, not the column's
    assert G.totals(plans[:4])["calls"] == sum(p["max_calls"] for p in plans[:4]) > 0
    assert plans[2]["max_calls"] == runplan.plan("n5_r1_s3", PARTS, repeats=1)["max_calls"] > 0


def test_planning_leaves_the_environment_as_it_was(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path / "results")
    jobs = T.pending_runs(cfg, T.load_all(cfg))
    G.plan_of(jobs[0], PARTS, {})
    assert "CAPP_MODEL_PRIMARY" not in os.environ
    monkeypatch.setenv("CAPP_MODEL_PRIMARY", "mine/model")
    G.plan_of(jobs[0], PARTS, {})
    G.plan_of(jobs[-1], PARTS, {})
    assert os.environ["CAPP_MODEL_PRIMARY"] == "mine/model"
    go(tmp_path, dry_run=True)
    assert os.environ["CAPP_MODEL_PRIMARY"] == "mine/model"


def test_the_shared_rows_are_planned_without_the_variable_of_a_column(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path / "results")
    shared = T.pending_runs(cfg, T.load_all(cfg))[-1]
    monkeypatch.setenv("CAPP_MODEL_PRIMARY", "mine/model")
    assert [t["model"] for t in G.plan_of(shared, PARTS, {})["tiers"]] == [config.model("small")]


# ================================================================ totals
def fake_plan(calls, estimate=None, seconds=0.0, llm=True):
    p = {"uses_llm": llm, "max_calls": calls, "estimate_seconds": seconds}
    if estimate is not None:
        p["estimate"] = estimate
    return p


def test_totals_add_up_and_say_nothing_where_an_estimate_is_missing():
    a = fake_plan(100, [{"usd": 0.5}], 60.0)
    b = fake_plan(50, [{"usd": 0.25}], 30.0)
    assert G.totals([a, b]) == {"calls": 150, "usd": 0.75, "seconds": 90.0, "tokens": None}
    assert G.totals([a, fake_plan(10, [{"model": "m", "note": "no earlier run"}])])["usd"] is None
    assert G.totals([a, fake_plan(10, [{"usd": None}])])["seconds"] is None                  # a model without a price
    assert G.totals([a, fake_plan(10)])["usd"] is None                                       # no estimate at all
    assert G.totals([fake_plan(0, llm=False)]) == {"calls": 0, "usd": 0.0, "seconds": 0.0, "tokens": 0.0}


def test_totals_text_says_what_is_known_and_never_a_made_up_number():
    assert G.totals_text({"calls": 0, "usd": 0.0, "seconds": 0.0, "tokens": 0.0}) == "no API call"
    known = G.totals_text({"calls": 120, "usd": 1.234, "seconds": 600.0})
    assert "at most 120" in known and "$1.23" in known and "10 min" in known and "[TO CONFIRM]" in known
    unknown = G.totals_text({"calls": 120, "usd": None, "seconds": None})
    assert "at most 120" in unknown and "$" not in unknown and "no estimate" in unknown


def test_the_estimate_of_an_earlier_run_reaches_the_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runplan, "history", lambda d=None: {QWEN: {"calls": 10, "prompt": 10000, "completion": 1000, "latency_s": 20.0, "runs": 1}})
    monkeypatch.setattr(config, "price", lambda m: (1.0, 2.0) if m == QWEN else None)
    code, _, say, _, _ = go(tmp_path, dry_run=True)
    a_line = next(line for line in say.lines if line.strip().startswith("a:"))
    b_line = next(line for line in say.lines if line.strip().startswith("b:"))
    assert "$" in a_line and "no estimate" not in a_line
    assert "no estimate" in b_line


# ================================================================ the run
def test_all_runs_are_made_in_order_with_the_model_of_their_column_and_the_grid_ends_complete(tmp_path):
    code, runner, say, asked, cfg = go(tmp_path, yes=True)
    assert code == 0 and asked == []
    assert runner.tags == FULL_ORDER
    assert [c[2] for c in runner.calls] == [QWEN] * 4 + [GPT] * 4 + [None]
    assert all(c[1] == str(tmp_path) for c in runner.calls)
    assert say.text.count("--- ") >= 9 and say.text.count("the grid now says ok") == 9
    assert "Every run is present and complete." in say.text and "Nothing is left to run in this selection" in say.text
    assert T.pending_runs(cfg, T.load_all(cfg)) == []


def test_a_second_call_has_nothing_to_do(tmp_path):
    go(tmp_path, yes=True)
    code, runner, say, asked, _ = go(tmp_path, yes=True)
    assert code == 0 and runner.calls == [] and asked == []
    assert "Nothing to run" in say.text and "Every run is present and complete." in say.text


def test_the_command_each_run_gets_is_the_one_make_tables_prints(tmp_path):
    cfg = make_cfg(tmp_path / "results")
    expected = [j["argv"] for j in T.pending_runs(cfg, T.load_all(cfg))]
    _, runner, _, _, _ = go(tmp_path, cfg=cfg, yes=True)
    assert [c[0] for c in runner.calls] == expected
    assert all(a[0] == "scripts/run_eval.py" for a in expected)


def test_it_asks_once_per_column_with_the_bound_and_goes_on_when_the_answer_is_yes(tmp_path):
    code, runner, say, asked, _ = go(tmp_path)
    assert code == 0 and len(asked) == 3 and len(runner.calls) == 9
    assert "4 run(s) of a" in asked[0] and "4 run(s) of b" in asked[1] and "1 run(s) of shared" in asked[2]
    assert all("calls the API" in q and "at most" in q for q in asked)


def test_no_to_the_first_column_runs_nothing_and_says_how_to_go_on(tmp_path):
    code, runner, say, asked, _ = go(tmp_path, answers=[False])
    assert code == 1 and runner.calls == [] and len(asked) == 1
    assert "stopped before a: not confirmed" in say.text and "Still to do:" in say.text


def test_no_to_the_second_column_keeps_the_first_and_a_later_call_continues_there(tmp_path):
    code, runner, say, asked, cfg = go(tmp_path, answers=[True, False])
    assert code == 1 and len(runner.calls) == 4 and {c[2] for c in runner.calls} == {QWEN}
    assert "stopped before b: not confirmed" in say.text
    code, runner2, say2, _, _ = go(tmp_path, yes=True)
    assert code == 0 and len(runner2.calls) == 5 and QWEN not in [c[2] for c in runner2.calls]


def test_yes_never_asks_and_a_dry_run_never_asks(tmp_path):
    assert go(tmp_path / "one", yes=True)[3] == []
    assert go(tmp_path / "two", dry_run=True)[3] == []


def test_without_a_key_nothing_is_started_and_the_message_says_where_to_put_it(tmp_path):
    code, runner, say, asked, _ = go(tmp_path, key=False, yes=True)
    assert code == 1 and runner.calls == [] and asked == []
    assert "GROQ_API_KEY is not set" in say.text and ".env" in say.text


def test_the_key_in_the_environment_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "abc")
    code, runner, _, _, _ = go(tmp_path, key=False, yes=True)
    assert code == 0 and len(runner.calls) == 9


def test_a_run_that_stops_stops_the_sequence_and_the_rest_waits_for_the_next_call(tmp_path):
    results = tmp_path / "results"
    code, runner, say, _, cfg = go(tmp_path, runner=Runner(results, codes={"n5_r1_s3": 2}), yes=True)
    assert code == 1 and runner.systems[-1] == "n5_r1_s3" and len(runner.calls) <= 4
    assert "(a) stopped (exit code 2)" in say.text and "run this command again" in say.text
    code, runner2, _, _, _ = go(tmp_path, yes=True)
    assert code == 0 and len(runner2.calls) == 9 - (len(runner.calls) - 1)                     # the failed one is made again, the rest follows


def test_ctrl_c_returns_1_and_says_the_cache_keeps_what_was_answered(tmp_path):
    results = tmp_path / "results"
    code, runner, say, _, _ = go(tmp_path, runner=Runner(results, interrupt="b0"), yes=True)
    assert code == 1 and "Interrupted during" in say.text and "cache" in say.text


def test_a_run_that_exits_0_but_is_not_usable_is_called_by_its_name_and_left_on_the_list(tmp_path):
    results = tmp_path / "results"
    code, runner, say, _, cfg = go(tmp_path, runner=Runner(results, make=False), yes=True)
    assert code == 1 and len(runner.calls) == 9
    assert say.text.count("the grid now says missing") == 9
    assert "Still to do: a/A1 b0, a/A2 b0_r1, a/A3 n5_r1_s3, a/A0 b1, b/A1 b0" in say.text


def test_a_cut_off_run_is_listed_as_again_and_made_again_with_the_same_tag(tmp_path):
    results = tmp_path / "results"
    cfg = make_cfg(results)
    go(tmp_path, cfg=cfg, yes=True)
    folder = results / "g_a_b0_r1"
    info = json.load(open(folder / "run.json", encoding="utf-8"))
    info["llm_client_summary"]["truncated"] = 7
    json.dump(info, open(folder / "run.json", "w", encoding="utf-8"))
    code, runner, say, _, _ = go(tmp_path, cfg=cfg, yes=True)
    assert code == 0 and runner.tags == [("g_a", "b0_r1")]
    assert "again: existing folder is unusable" in say.text and "7 answer(s) were cut off" in say.text
    assert json.load(open(folder / "run.json", encoding="utf-8"))["llm_client_summary"]["truncated"] == 0


def test_a_run_with_old_model_settings_is_made_again(tmp_path):
    results = tmp_path / "results"
    cfg = make_cfg(results)
    go(tmp_path, cfg=cfg, yes=True)
    folder = results / "g_a_b1"
    info = json.load(open(folder / "run.json", encoding="utf-8"))
    info["model_params"] = {}
    json.dump(info, open(folder / "run.json", "w", encoding="utf-8"))
    code, runner, say, _, _ = go(tmp_path, cfg=cfg, yes=True)
    assert code == 0 and runner.tags == [("g_a", "b1")] and "other model settings" in say.text


# ================================================================ the selection and the options
def test_only_one_column_runs_that_column_and_leaves_the_rest(tmp_path):
    code, runner, say, asked, cfg = go(tmp_path, only=["b"], yes=True)
    assert code == 0 and [c[2] for c in runner.calls] == [GPT] * 4
    assert len(T.pending_runs(cfg, T.load_all(cfg))) == 5
    assert "Nothing is left to run in this selection" in say.text


def test_only_shared_runs_the_rows_that_choose_their_own_models_without_the_variable(tmp_path):
    code, runner, _, _, _ = go(tmp_path, only=["shared"], yes=True)
    assert code == 0 and runner.tags == [("g_shared", "n5_r1_s3+ms")] and runner.calls[0][2] is None


def test_only_two_columns_keep_the_order_of_the_grid(tmp_path):
    _, runner, _, _, _ = go(tmp_path, only=["shared", "a"], yes=True)
    assert [c[2] for c in runner.calls] == [QWEN] * 4 + [None]


def test_a_column_that_does_not_exist_is_said_in_words(tmp_path):
    with pytest.raises(G.G2Error, match=r"--only qwen: no such column\. Columns of this grid: a, b, shared"):
        go(tmp_path, only=["qwen"], yes=True)


def test_only_with_nothing_to_make_says_so_and_returns_0(tmp_path):
    go(tmp_path, only=["a"], yes=True)
    code, runner, say, _, _ = go(tmp_path, only=["a"], yes=True)
    assert code == 0 and runner.calls == [] and "a has nothing to make" in say.text


def test_max_calls_is_added_to_every_run_and_nothing_else_changes(tmp_path):
    cfg = make_cfg(tmp_path / "results")
    base = [j["argv"] for j in T.pending_runs(cfg, T.load_all(cfg))]
    _, runner, _, _, _ = go(tmp_path, cfg=cfg, yes=True, max_calls=40)
    assert [c[0] for c in runner.calls] == [a + ["--max-calls", "40"] for a in base]


@pytest.mark.parametrize("bad", [0, -3, True, "5", 2.5])
def test_a_wrong_max_calls_is_said_in_words(tmp_path, bad):
    with pytest.raises(G.G2Error, match="--max-calls must be a whole number of 1 or more"):
        go(tmp_path, max_calls=bad, dry_run=True)


def test_a_sealed_dataset_is_refused_before_anything_else(tmp_path):
    cfg = make_cfg(tmp_path / "results", dataset="test_v2", limit=None)
    with pytest.raises(G.G2Error, match=r"test_v2 is sealed .*after the freeze"):
        go(tmp_path, cfg=cfg, yes=True)
    assert not (tmp_path / "results").exists()


def test_a_dataset_that_cannot_be_loaded_is_said_in_words(tmp_path):
    cfg = make_cfg(tmp_path / "results", dataset=str(tmp_path / "missing.json"))
    with pytest.raises(G.G2Error, match="dataset .*missing.json"):
        go(tmp_path, cfg=cfg, dry_run=True)


def test_the_status_grid_is_printed_at_the_end(tmp_path):
    code, runner, say, _, _ = go(tmp_path, answers=[True, False])
    assert "Model A" in say.text and "Model B" in say.text and "run(s) to make or redo." in say.text


# ================================================================ the real runner and the question
def test_the_default_runner_sets_or_removes_the_variable_for_the_child_only(monkeypatch):
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, cwd=None, env=None: seen.append((cmd, cwd, env)) or subprocess.CompletedProcess(cmd, 3))
    monkeypatch.setenv("CAPP_MODEL_PRIMARY", "mine/model")
    assert G.default_runner(["scripts/run_eval.py", "--system", "b0"], "/r", QWEN) == 3
    assert G.default_runner(["scripts/run_eval.py"], "/r", None) == 3
    (cmd1, cwd1, env1), (cmd2, cwd2, env2) = seen
    assert cmd1 == [sys.executable, "scripts/run_eval.py", "--system", "b0"] and cwd1 == "/r"
    assert env1["CAPP_MODEL_PRIMARY"] == QWEN and "CAPP_MODEL_PRIMARY" not in env2
    assert os.environ["CAPP_MODEL_PRIMARY"] == "mine/model"                                    # this process is untouched


def test_the_default_question_reads_yes_and_treats_the_end_of_input_as_no(monkeypatch):
    import builtins
    monkeypatch.setattr(builtins, "input", lambda q: "Y")
    assert G.default_ask("?") is True
    monkeypatch.setattr(builtins, "input", lambda q: "")
    assert G.default_ask("?") is False

    def eof(q):
        raise EOFError
    monkeypatch.setattr(builtins, "input", eof)
    assert G.default_ask("?") is False


# ================================================================ scripts/g2.py
def write_cfg(tmp_path, **over):
    cfg = {"name": "g2t", "dataset": "dev_v2", "limit": N, "repeats": 1, "results_dir": str(tmp_path / "results"), "shared_tag": "g_shared",
           "models": [{"id": "a", "label": "Model A", "model": QWEN, "tag": "g_a"}],
           "rows": [{"id": "A0", "label": "B1", "system": "b1"}]}
    cfg.update(over)
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return str(path)


def test_the_script_dry_run_exits_0_and_prints_the_plan(tmp_path, capsys):
    assert g2_script.main([write_cfg(tmp_path), "--dry-run", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Runs to make or run again: 1" in out and "Dry run: nothing was run." in out


def test_the_script_says_a_bad_config_in_words_and_exits_2(tmp_path, capsys):
    path = write_cfg(tmp_path, models=[])
    assert g2_script.main([path, "--dry-run"]) == 2
    assert "STOPPED: config error:" in capsys.readouterr().err


def test_the_script_refuses_a_sealed_dataset_and_a_bad_column_with_exit_2(tmp_path, capsys):
    assert g2_script.main([write_cfg(tmp_path, dataset="test_v2", limit=None), "--dry-run", "--root", str(tmp_path)]) == 2
    assert "is sealed" in capsys.readouterr().err
    assert g2_script.main([write_cfg(tmp_path), "--only", "zzz", "--dry-run", "--root", str(tmp_path)]) == 2
    assert "no such column" in capsys.readouterr().err
    assert g2_script.main([write_cfg(tmp_path), "--max-calls", "0", "--dry-run", "--root", str(tmp_path)]) == 2
    assert "--max-calls" in capsys.readouterr().err


def test_the_template_of_the_repository_plans_without_a_call(tmp_path, capsys):
    cfg_path = os.path.join(ROOT, "configs", "ablation_template.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    cfg["results_dir"] = str(tmp_path / "empty")
    path = tmp_path / "template.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    assert g2_script.main([str(path), "--dry-run", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Runs to make or run again: 28" in out
    assert "qwen:" in out and "gpt120:" in out and "gpt20:" in out and "shared:" in out
    assert not (tmp_path / "empty").exists()


# ================================================================ details found by mutating the code (Step 40)
def test_primary_sets_removes_and_restores_the_variable(monkeypatch):
    monkeypatch.setenv("CAPP_MODEL_PRIMARY", "mine/model")
    with G._primary(QWEN):
        assert os.environ["CAPP_MODEL_PRIMARY"] == QWEN
    assert os.environ["CAPP_MODEL_PRIMARY"] == "mine/model"
    with G._primary(None):
        assert "CAPP_MODEL_PRIMARY" not in os.environ
    assert os.environ["CAPP_MODEL_PRIMARY"] == "mine/model"
    monkeypatch.delenv("CAPP_MODEL_PRIMARY")
    with pytest.raises(RuntimeError):
        with G._primary(GPT):
            raise RuntimeError("boom")
    assert "CAPP_MODEL_PRIMARY" not in os.environ                                             # also after an error


def test_the_bound_counts_the_repeats_of_the_row(tmp_path):
    rows = [{"id": "A0", "label": "B0 twice", "system": "b0", "repeats": 2}, {"id": "A1", "label": "B0 once", "system": "b0_r1"}]
    cfg = make_cfg(tmp_path / "results", rows=rows)
    jobs = T.pending_runs(cfg, T.load_all(cfg))
    by = {j["row"]["id"]: j for j in jobs if j["column"] == "a"}
    assert by["A0"]["argv"][by["A0"]["argv"].index("--repeats") + 1] == "2" and by["A1"]["argv"][by["A1"]["argv"].index("--repeats") + 1] == "1"
    p0, p1 = G.plan_of(by["A0"], PARTS, {}), G.plan_of(by["A1"], PARTS, {})
    assert p0["repeats"] == 2 and p0["max_calls"] == 2 * 5 * N                                  # 5 samples x 3 parts x 2 repeats
    assert p1["repeats"] == 1 and p1["max_calls"] == 5 * N                                      # one repeat (R1 repairs without a new call)


def test_only_with_an_empty_list_means_every_column(tmp_path):
    _, runner, _, _, _ = go(tmp_path, only=[], yes=True)
    assert len(runner.calls) == 9


def test_the_bound_is_for_the_parts_the_run_will_read_limit_and_offset(tmp_path):
    rows = [{"id": "A0", "label": "constrained", "system": "n5_r0_s1+cg1"}]
    full = dataset.load_dataset("dev_v2")
    offset = next(o for o in range(1, 30) if runplan.plan("n5_r0_s1+cg1", full[o:o + N])["max_calls"] != runplan.plan("n5_r0_s1+cg1", full[:N])["max_calls"])
    cfg = make_cfg(tmp_path / "results", rows=rows, offset=offset)
    _, _, say, _, _ = go(tmp_path, cfg=cfg, dry_run=True)
    want = runplan.plan("n5_r0_s1+cg1", full[offset:offset + N])["max_calls"]
    assert f"(<= {want} calls)" in say.text
    cfg = make_cfg(tmp_path / "results", rows=rows, limit=2)
    _, _, say, _, _ = go(tmp_path, cfg=cfg, dry_run=True)
    assert f"(<= {runplan.plan('n5_r0_s1+cg1', full[:2])['max_calls']} calls)" in say.text
    cfg = make_cfg(tmp_path / "results", rows=[{"id": "A0", "label": "B0", "system": "b0"}], limit=None)
    _, _, say, _, _ = go(tmp_path, cfg=cfg, dry_run=True)
    assert f"(<= {5 * len(full)} calls)" in say.text                                           # no limit: the whole dataset


def test_a_group_that_makes_no_call_needs_neither_the_key_nor_the_question(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "plan_of", lambda job, parts, hist: {"uses_llm": False, "max_calls": 0, "estimate_seconds": 0.0})
    code, runner, say, asked, _ = go(tmp_path, key=False)
    assert code == 0 and asked == [] and len(runner.calls) == 9
    assert "NOT set" in say.text and "is not set. Put" not in say.text and "no API call" in say.text


def test_every_run_header_counts_from_one_to_the_total(tmp_path):
    _, _, say, _, _ = go(tmp_path, yes=True)
    heads = [line.lstrip() for line in say.lines if line.lstrip().startswith("=== [")]
    assert [h.split("]")[0] for h in heads] == [f"=== [{i}/9" for i in range(1, 10)]


def test_the_line_after_a_run_gives_the_reason_when_the_grid_does_not_count_it(tmp_path):
    results = tmp_path / "results"
    cfg = make_cfg(results)
    go(tmp_path, cfg=cfg, yes=True)
    folder = results / "g_a_b1"
    info = json.load(open(folder / "run.json", encoding="utf-8"))
    info["llm_client_summary"]["truncated"] = 7
    json.dump(info, open(folder / "run.json", "w", encoding="utf-8"))
    code, runner, say, _, _ = go(tmp_path, cfg=cfg, runner=Runner(results, make=False), yes=True)
    assert code == 1 and runner.tags == [("g_a", "b1")]
    assert "A0 b1: the grid now says error: 7 answer(s) were cut off at the token limit" in say.text
    assert "Still to do: a/A0 b1" in say.text


def test_the_shared_rows_are_run_in_the_same_cheapest_first_order(tmp_path):
    rows = [{"id": "A0", "label": "one sample", "system": "b1+ms"}, {"id": "A1", "label": "five samples", "system": "n5_r1_s3+ms"}]
    cfg = make_cfg(tmp_path / "results", rows=rows)
    shared = [j["row"]["system"] for j in T.pending_runs(cfg, T.load_all(cfg)) if j["column"] == "shared"]
    assert shared == ["n5_r1_s3+ms", "b1+ms"]


def test_the_incomplete_run_says_the_cache_will_be_reused(tmp_path):
    results = tmp_path / "results"
    cfg = make_cfg(results)
    go(tmp_path, cfg=cfg, yes=True)
    lines = open(results / "g_b_b1" / "records.jsonl", encoding="utf-8").read().splitlines()
    open(results / "g_b_b1" / "records.jsonl", "w", encoding="utf-8").write("\n".join(lines[:-1]) + "\n")
    job = T.pending_runs(cfg, T.load_all(cfg))[0]
    assert job["why"].startswith("incomplete (cached answers are reused): ")
    _, _, say, _, _ = go(tmp_path, cfg=cfg, dry_run=True)
    assert "again: incomplete (cached answers are reused)" in say.text


def test_the_script_hands_every_option_to_the_run(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(g2_script.g2, "run_g2", lambda cfg, root, **kw: seen.update(root=root, **kw) or 0)
    path = write_cfg(tmp_path)
    assert g2_script.main([path, "--only", "a", "shared", "--yes", "--dry-run", "--max-calls", "7", "--root", str(tmp_path)]) == 0
    assert seen == {"root": str(tmp_path), "only": ["a", "shared"], "yes": True, "dry_run": True, "max_calls": 7, "rows": None}
    seen.clear()
    assert g2_script.main([path]) == 0
    assert seen == {"root": ROOT, "only": None, "yes": False, "dry_run": False, "max_calls": None, "rows": None}


def test_a_cut_off_shared_run_is_pending_again_like_any_other(tmp_path):
    results = tmp_path / "results"
    cfg = make_cfg(results)
    go(tmp_path, cfg=cfg, yes=True)
    folder = results / "g_shared_n5_r1_s3+ms"
    info = json.load(open(folder / "run.json", encoding="utf-8"))
    info["llm_client_summary"]["truncated"] = 2
    json.dump(info, open(folder / "run.json", "w", encoding="utf-8"))
    jobs = T.pending_runs(cfg, T.load_all(cfg))
    assert [(j["column"], j["row"]["system"]) for j in jobs] == [("shared", "n5_r1_s3+ms")] and "cut off" in jobs[0]["why"]
    code, runner, _, _, _ = go(tmp_path, cfg=cfg, yes=True)
    assert code == 0 and runner.tags == [("g_shared", "n5_r1_s3+ms")] and runner.calls[0][2] is None


def test_the_reason_of_an_error_run_is_its_errors_not_its_warnings(tmp_path):
    results = tmp_path / "results"
    cfg = make_cfg(results)
    jobs = T.pending_runs(cfg, T.load_all(cfg))
    go(tmp_path, cfg=cfg, yes=True)
    job = next(j for j in jobs if j["column"] == "a" and j["row"]["id"] == "A1")
    assert G._cell_text(cfg, T.load_all(cfg), job) == "ok"
    folder = results / "g_a_b0"
    info = json.load(open(folder / "run.json", encoding="utf-8"))
    del info["dataset"]                                                                       # a warning
    info["system"] = "b1"                                                                     # a short error
    json.dump(info, open(folder / "run.json", "w", encoding="utf-8"))
    loaded = T.load_all(cfg)
    run = loaded["runs"][("A1", "a")]
    assert run["status"] == "error" and run["warnings"] == ["run.json does not name the dataset"]
    text = G._cell_text(cfg, loaded, job)
    assert text == "error: run.json says system 'b1', the row A1 is 'b0'"


# ================================================================ --rows (Step 42)
def test_rows_keeps_only_those_rows_in_the_order_of_the_grid(tmp_path):
    _, runner, say, _, cfg = go(tmp_path, rows=["A3", "A1"], yes=True)
    assert runner.tags == [("g_a", "b0"), ("g_a", "n5_r1_s3"), ("g_b", "b0"), ("g_b", "n5_r1_s3")]       # not the order the author typed
    assert len(T.pending_runs(cfg, T.load_all(cfg))) == 9 - 4


def test_rows_and_only_together_are_both_applied(tmp_path):
    _, runner, _, _, _ = go(tmp_path, only=["a", "shared"], rows=["A0", "A4"], yes=True)
    assert runner.tags == [("g_a", "b1"), ("g_shared", "n5_r1_s3+ms")]


def test_a_row_that_does_not_exist_is_said_with_the_rows_there_are(tmp_path):
    with pytest.raises(G.G2Error, match=r"--rows A9, X1: no such row\. Rows of this grid: A0, A1, A2, A3, A4, A5"):
        go(tmp_path, rows=["A9", "X1", "A0"], yes=True)


def test_a_row_without_a_system_cannot_be_run(tmp_path):
    with pytest.raises(G.G2Error, match=r"--rows A5: not built"):
        go(tmp_path, rows=["A5"], yes=True)


def test_an_empty_list_of_rows_means_every_row(tmp_path):
    _, runner, _, _, _ = go(tmp_path, rows=[], yes=True)
    assert len(runner.calls) == 9


def test_rows_leaves_the_plan_and_the_end_message_to_the_chosen_rows(tmp_path):
    code, runner, say, _, _ = go(tmp_path, rows=["A0"], dry_run=True)
    assert code == 0 and "Runs to make or run again: 2" in say.text                       # A0 in the two columns, nothing else
    code, runner, say, _, _ = go(tmp_path, rows=["A0"], yes=True)
    assert code == 0 and runner.tags == [("g_a", "b1"), ("g_b", "b1")] and "Nothing is left to run in this selection" in say.text
    code, runner, say, _, _ = go(tmp_path, rows=["A0"], yes=True)
    assert code == 0 and runner.calls == [] and "Nothing to run: A0 has nothing to make." in say.text


def test_the_end_message_counts_only_the_chosen_rows_as_left(tmp_path):
    code, runner, say, _, _ = go(tmp_path, rows=["A1"], runner=Runner(tmp_path / "results", make=False), yes=True)
    assert code == 1 and "Still to do: a/A1 b0, b/A1 b0." in say.text and "A2" not in say.text.split("Still to do:")[1]


def test_the_script_hands_rows_to_the_run(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(g2_script.g2, "run_g2", lambda cfg, root, **kw: seen.update(root=root, **kw) or 0)
    assert g2_script.main([write_cfg(tmp_path), "--rows", "A0", "A2", "--root", str(tmp_path)]) == 0
    assert seen["rows"] == ["A0", "A2"]
    seen.clear()
    g2_script.main([write_cfg(tmp_path)])
    assert seen["rows"] is None
