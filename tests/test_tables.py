"""Step 22: the Table 4 regenerator (capp/tables.py, capp/tablefigs.py, scripts/make_tables.py, configs/ablation_template.json).

Offline. The runs are made by the real harness with a deterministic fake planner that makes the kind of mistakes an LLM makes (a
dropped operation, two swapped), three "LLMs" that differ in how often they do, no API call. The statistics are also checked on
numbers worked out by hand.
"""
import copy
import csv
import json
import os
import random
import shutil
import statistics
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from capp import config, dataset, harness, summary as S, systems, tablefigs, tables as T, verify as V      # noqa: E402
import make_tables                                                                             # noqa: E402
import route_builder as rb                                                                     # noqa: E402

N_PARTS = 8
REPEATS = 2
MODELS = (("a", "Model A", "qwen/qwen3.8-27b", "g_a", "A", 0.5, 0.5),         # priced model: has USD figures
          ("b", "Model B", "openai/gpt-oss-120b", "g_b", "B", 0.4, 0.3),
          ("c", "Model C", "vendor/unpriced-model", "g_c", "C", 0.6, 0.6))      # no price: USD n/a
SYSTEMS = ("b1", "b0", "b0_r1", "n5_r1_s3", "n5_r1_s3+fz2")


def make_planner(salt, p_drop, p_swap):
    """A planner that answers with a Route Builder route and damages it now and then; a different draw for every repeat."""
    calls = {}

    def planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        routes = rb.generate_valid_routes(features, max_routes=15)
        if not routes:
            return {"success": False, "steps": [], "raw_response": ""}
        steps = V.clean_steps(routes[(sample_id * 3 + len(features)) % len(routes)]["steps"])
        key = (tuple(features), sample_id)
        calls[key] = calls.get(key, 0) + 1
        r = random.Random(f"{salt}-{','.join(features)}-{sample_id}-{calls[key]}")
        if r.random() < p_drop and len(steps) > 3:
            steps = steps[:]
            steps.pop(r.randrange(len(steps)))
        if r.random() < p_swap:
            steps = steps[:]
            i = r.randrange(len(steps) - 1)
            steps[i], steps[i + 1] = steps[i + 1], steps[i]
        return {"success": True, "steps": steps, "raw_response": "", "model": "fake",
                "meta": {"cached": True, "finish_reason": "stop", "prompt_tokens": 100, "completion_tokens": 20, "latency_s": 0.1}}
    return planner


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("grid"))
    parts = dataset.load_dataset("dev_v2")[:N_PARTS]
    mp = pytest.MonkeyPatch()
    try:
        for mid, label, model, tag, salt, pd, ps in MODELS:
            mp.setenv("CAPP_MODEL_PRIMARY", model)
            for system in SYSTEMS:
                harness.run_experiment(system, parts, repeats=REPEATS, tag=tag, out_dir=out, dataset_name="dev_v2", progress=None,
                                       planner=make_planner(salt, pd, ps), analysis=False)
        mp.delenv("CAPP_MODEL_PRIMARY")
        shared = make_planner("S", 0.5, 0.5)                                         # +ms: a planner option, so the planner comes by factory
        harness.run_experiment("n5_r1_s3+ms", parts, repeats=REPEATS, tag="g_shared", out_dir=out, dataset_name="dev_v2", progress=None,
                               planner_factory=lambda part: shared, analysis=False)
    finally:
        mp.undo()
    return out


def make_cfg(results_dir, **over):
    cfg = {"name": "t", "dataset": "dev_v2", "limit": N_PARTS, "repeats": REPEATS, "results_dir": results_dir, "resamples": 200, "seed": 7,
           "shared_tag": "g_shared",
           "models": [{"id": mid, "label": label, "model": model, "tag": tag} for mid, label, model, tag, *_ in MODELS],
           "rows": [{"id": "A0", "label": "B1: one call", "system": "b1"},
                    {"id": "A1", "label": "B0: five samples", "system": "b0", "parent": "A0"},
                    {"id": "A2", "label": "A1 + R1 repair", "system": "b0_r1", "parent": "A1", "note": "decide: this one"},
                    {"id": "A3", "label": "A2 + selection", "system": "n5_r1_s3", "parent": "A2"},
                    {"id": "A4", "label": "A3 + feasibility", "system": "n5_r1_s3+fz2", "parent": "A3"},
                    {"id": "A5", "label": "A3 on the small model", "system": "n5_r1_s3+ms", "parent": "A3"},
                    {"id": "A6", "label": "Constrained planner", "system": None, "parent": "A4", "note": "optional, not built"}]}
    cfg.update(over)
    return cfg


@pytest.fixture(scope="module")
def cfg(grid):
    return T.validate_config(make_cfg(grid))


@pytest.fixture(scope="module")
def res(cfg):
    return T.analyse(cfg)


def copy_grid(grid, tmp_path, only=None):
    dst = str(tmp_path / "results")
    os.makedirs(dst)
    for name in os.listdir(grid):
        if only is None or name in only:
            shutil.copytree(os.path.join(grid, name), os.path.join(dst, name))
    return dst


def rewrite_records(folder, fn):
    path = os.path.join(folder, "records.jsonl")
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    recs = fn(recs)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def edit_run_json(folder, **kw):
    path = os.path.join(folder, "run.json")
    info = json.load(open(path, encoding="utf-8"))
    info.update(kw)
    json.dump(info, open(path, "w", encoding="utf-8"))


def one_run(cfg, results_dir, row_id="A2", model_id="a"):
    row = next(r for r in cfg["rows"] if r["id"] == row_id)
    model = next(m for m in cfg["models"] if m["id"] == model_id)
    return T.load_run(cfg, row, model, results_dir, T.expected_parts(cfg))


# ================================================================ the config
def test_the_template_loads_and_every_system_of_it_parses():
    cfg = T.load_config(os.path.join(ROOT, "configs", "ablation_template.json"))
    assert [r["id"] for r in cfg["rows"]] == [f"A{i}" for i in range(10)]
    built = [r for r in cfg["rows"] if r["built"]]
    assert [r["id"] for r in cfg["rows"] if not r["built"]] == ["A9"]
    for r in built:
        assert systems.parse_system(r["system"]) is not None and systems.is_llm(r["system"])
    assert len(cfg["models"]) == 3
    assert [r["id"] for r in cfg["rows"] if r["independent"]] == ["A8"]
    parents = {r["id"]: r["parent"] for r in cfg["rows"]}
    assert parents["A0"] is None and parents["A1"] == "A0" and parents["A2"] == "A1" and parents["A5"] == "A3" and parents["A7"] == "A6"


def test_the_template_rows_follow_the_report():
    cfg = T.load_config(os.path.join(ROOT, "configs", "ablation_template.json"))
    sysm = {r["id"]: r["system"] for r in cfg["rows"]}
    assert sysm["A0"] == "b1" and sysm["A1"] == "b0"                           # B1, B0 (5 samples, whole-route replacement)
    spec = {k: systems.parse_system(v) for k, v in sysm.items() if v}
    assert spec["A1"] == (5, "r0", "s1") and spec["A2"] == (5, "r1", "s1")      # + localized repair
    assert systems.planner_spec(sysm["A3"]).two_step and systems.planner_spec(sysm["A3"]).hint
    assert systems.planner_spec(sysm["A4"]).two_step and not systems.planner_spec(sysm["A4"]).hint
    assert spec["A5"][2] == "s3"                                                # multi-candidate selection
    assert systems.planner_spec(sysm["A6"]).few_shot == 2
    assert systems.planner_spec(sysm["A7"]).feas == 2
    assert systems.planner_spec(sysm["A8"]).cascade == 1


def test_a_valid_config_is_normalised(cfg):
    assert cfg["rows"][0]["parent"] is None and cfg["rows"][1]["parent"] == "A0"
    assert [r["independent"] for r in cfg["rows"]] == [False] * 5 + [True, False]
    assert cfg["rows"][6]["built"] is False and cfg["rows"][6]["system"] is None
    assert cfg["rows"][0]["repeats"] == REPEATS


def test_a_row_can_override_the_repeats():
    c = T.validate_config(make_cfg("x", rows=[{"id": "A0", "label": "l", "system": "b1", "repeats": 1}]))
    assert c["rows"][0]["repeats"] == 1


BAD = [
    ("name", lambda c: c.pop("name"), "name"),
    ("name chars", lambda c: c.update(name="a b/c"), "may contain only"),
    ("unknown key", lambda c: c.update(repeat=3), "unknown key"),
    ("repeats zero", lambda c: c.update(repeats=0), "repeats"),
    ("repeats bool", lambda c: c.update(repeats=True), "repeats"),
    ("limit text", lambda c: c.update(limit="20"), "limit"),
    ("no models", lambda c: c.update(models=[]), "models"),
    ("model unknown key", lambda c: c["models"][0].update(name="x"), "unknown key"),
    ("model no tag", lambda c: c["models"][0].pop("tag"), "tag"),
    ("same tag twice", lambda c: c["models"][1].update(tag="g_a"), "tag 'g_a' is used twice"),
    ("same model twice", lambda c: c["models"][1].update(model="qwen/qwen3.8-27b"), "model 'qwen/qwen3.8-27b' is used twice"),
    ("same model id twice", lambda c: c["models"][1].update(id="a"), "id 'a' is used twice"),
    ("no rows", lambda c: c.update(rows=[]), "rows"),
    ("row id twice", lambda c: c["rows"][1].update(id="A0"), "id 'A0' is used twice"),
    ("parent later", lambda c: c["rows"][0].update(parent="A3"), "EARLIER"),
    ("parent unknown", lambda c: c["rows"][1].update(parent="Z9"), "EARLIER"),
    ("bad system", lambda c: c["rows"][1].update(system="b7"), "unknown system"),
    ("bad option", lambda c: c["rows"][1].update(system="b0+zz"), "unknown planner option"),
    ("rb row", lambda c: c["rows"][1].update(system="rb"), "reference"),
    ("system twice", lambda c: c["rows"][2].update(system="b0"), "already the system of row A1"),
    ("fz needs repair", lambda c: c["rows"][1].update(system="b0+fz1"), "needs repair"),
    ("row unknown key", lambda c: c["rows"][1].update(sistem="b0"), "unknown key"),
    ("no shared tag", lambda c: c.update(shared_tag=None), "shared_tag"),
    ("sealed pool", lambda c: (c.update(pool="test_v2"), c["rows"][1].update(system="b0+fs2")), "sealed"),
    ("not a dict", None, "JSON object"),
]


@pytest.mark.parametrize("label,mutate,message", BAD, ids=[b[0] for b in BAD])
def test_a_bad_config_is_refused_with_the_reason(label, mutate, message):
    raw = make_cfg("x")
    if mutate is None:
        raw = ["not", "a", "dict"]
    else:
        mutate(raw)
    with pytest.raises(T.TableError) as e:
        T.validate_config(raw)
    assert message in str(e.value)


def test_load_config_reports_unreadable_and_invalid_json(tmp_path):
    with pytest.raises(T.TableError, match="cannot read"):
        T.load_config(str(tmp_path / "nope.json"))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(T.TableError, match="not valid JSON"):
        T.load_config(str(bad))


def test_the_config_hash_changes_with_the_content_only(cfg):
    assert T.config_sha(cfg) == T.config_sha(copy.deepcopy(cfg))
    other = copy.deepcopy(cfg)
    other["rows"][0]["label"] = "changed"
    assert T.config_sha(other) != T.config_sha(cfg)


# ================================================================ statistics, worked by hand
@pytest.mark.parametrize("b,c,p", [(0, 0, 1.0), (1, 0, 1.0), (5, 0, 0.0625), (10, 0, 2 / 1024), (8, 2, 2 * 56 / 1024), (3, 3, 1.0),
                                   (2, 6, 2 * (1 + 8 + 28) / 256)])
def test_mcnemar_exact_values(b, c, p):
    assert T.mcnemar_exact(b, c) == pytest.approx(p)
    assert T.mcnemar_exact(c, b) == pytest.approx(p)                            # symmetric


def test_mcnemar_of_a_large_sample_does_not_overflow():
    p = T.mcnemar_exact(900, 700)
    assert 0 <= p < 1e-5


def test_paired_binary_counts_only_the_keys_both_runs_have():
    row = {1: True, 2: True, 3: False, 4: False, 5: True, 9: True}
    parent = {1: True, 2: False, 3: True, 4: False, 5: False, 8: False}
    r = T.paired_binary(row, parent)
    assert (r["n"], r["both"], r["row_only"], r["parent_only"], r["neither"]) == (5, 1, 2, 1, 1)
    assert r["p"] == pytest.approx(1.0) and r["verdict"] == T.UNCLEAR


def test_paired_binary_verdict_follows_the_exact_test():
    row = {i: True for i in range(12)}
    parent = {i: False for i in range(12)}
    r = T.paired_binary(row, parent)
    assert r["row_only"] == 12 and r["p"] == pytest.approx(2 / 4096) and r["verdict"] == "better"
    r2 = T.paired_binary(parent, row)
    assert r2["verdict"] == "worse"
    assert T.paired_binary({}, {})["verdict"] == "n/a" and T.paired_binary({}, {})["p"] is None


def test_bootstrap_of_constant_differences_is_that_constant():
    r = T.bootstrap_mean([0.5] * 6, 200, "k")
    assert r == {"n": 6, "mean": 0.5, "lo": 0.5, "hi": 0.5}


def test_bootstrap_is_deterministic_and_depends_on_the_key():
    d = [0.3, -0.1, 0.8, 0.2, 0.5, -0.4, 0.9, 0.1]
    a, b, c = T.bootstrap_mean(d, 300, "k"), T.bootstrap_mean(d, 300, "k"), T.bootstrap_mean(d, 300, "other")
    assert a == b and a["mean"] == pytest.approx(sum(d) / len(d)) and a["lo"] <= a["mean"] <= a["hi"]
    assert (a["lo"], a["hi"]) != (c["lo"], c["hi"])


def test_bootstrap_edge_cases():
    assert T.bootstrap_mean([], 100, "k") == {"n": 0, "mean": None, "lo": None, "hi": None}
    one = T.bootstrap_mean([2.0], 100, "k")
    assert one["mean"] == 2.0 and one["lo"] is None and one["hi"] is None


def test_the_interval_excludes_zero_for_a_clear_effect_and_not_for_noise():
    clear = T.bootstrap_mean([1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.0, 1.1], 400, "k")
    assert clear["lo"] > 0
    noise = T.bootstrap_mean([1.0, -1.0, 0.8, -0.9, 1.1, -1.2, 0.2, -0.3], 400, "k")
    assert noise["lo"] < 0 < noise["hi"]


@pytest.mark.parametrize("metric,ci,verdict", [
    ("M6", {"mean": -1.0, "lo": -2.0, "hi": -0.5}, "better"),          # a smaller gap is better
    ("M6", {"mean": 1.0, "lo": 0.5, "hi": 2.0}, "worse"),
    ("M6", {"mean": -1.0, "lo": -2.0, "hi": 0.5}, T.UNCLEAR),
    ("M1", {"mean": 0.2, "lo": 0.1, "hi": 0.3}, "better"),             # more validity is better
    ("M1", {"mean": -0.2, "lo": -0.3, "hi": -0.1}, "worse"),
    ("M7", {"mean": 3.0, "lo": 2.0, "hi": 4.0}, "costlier"),
    ("M7", {"mean": -3.0, "lo": -4.0, "hi": -2.0}, "cheaper"),
    ("M4", {"mean": None, "lo": None, "hi": None}, "n/a"),
    ("M4", {"mean": 0.1, "lo": None, "hi": None}, T.UNCLEAR),
])
def test_verdict_names_the_direction_only_when_the_interval_excludes_zero(metric, ci, verdict):
    assert T.verdict_of(metric, ci) == verdict


def test_paired_metric_uses_the_common_parts_with_values_on_both_sides():
    row = {"P1": {"M6": 1.0}, "P2": {"M6": 2.0}, "P3": {"M6": None}, "P4": {"M6": 5.0}}
    parent = {"P1": {"M6": 3.0}, "P2": {"M6": 5.0}, "P3": {"M6": 1.0}, "P5": {"M6": 0.0}}
    cfg = {"resamples": 200, "seed": 7}
    p = T.paired_metric(cfg, "a", "A2", "A1", "M6", row, parent)
    assert p["n"] == 2 and p["mean"] == pytest.approx(-2.5)                     # (1-3, 2-5) -> -2, -3
    assert (p["model"], p["row"], p["parent"], p["metric"]) == ("a", "A2", "A1", "M6")
    assert p["lo"] <= -2.5 <= p["hi"] and -3 <= p["lo"] and p["hi"] <= -2


# ================================================================ the cells, against the summary of the harness
def test_the_cell_equals_the_mean_and_sd_over_the_repeats_of_the_summary(grid, cfg):
    recs = harness.load_records(os.path.join(grid, "g_a_b0_r1"))
    cell = T.cell_stats(recs, "qwen/qwen3.8-27b")
    per = [S.summarize_set([r for r in recs if r["repeat"] == k]) for k in (1, 2)]
    for key, get in (("M1", lambda s: s["M1"]["strict_valid_of_parsed"]), ("M2", lambda s: s["M2"]["final_strict_valid"]),
                     ("M6", lambda s: s["M6"]["efficiency_gap_to_rb_best_mean"]), ("M7", lambda s: s["M7"]["api_calls_per_part"]),
                     ("M2p", lambda s: 1 - s["M2"]["provenance_share"].get("route_builder", 0))):
        vals = [get(s) for s in per]
        assert cell[key]["mean"] == pytest.approx(statistics.fmean(vals)), key
        assert cell[key]["sd"] == pytest.approx(statistics.stdev(vals)), key
        assert cell[key]["n"] == 2
    m4 = [s["M4"]["proxy_rb"]["final"]["lcs_ratio_mean"] for s in per]
    assert cell["M4"]["mean"] == pytest.approx(statistics.fmean(m4)) and cell["m4_source"] == "proxy_rb"
    assert cell["n_repeats"] == 2 and cell["n_parts"] == N_PARTS


def test_m5_is_the_share_of_llm_operations_kept_over_every_replaced_or_repaired_route(grid):
    recs = harness.load_records(os.path.join(grid, "g_a_b0_r1"))
    cell = T.cell_stats(recs)
    expected = []
    for k in (1, 2):
        rows = S.candidate_rows([r for r in recs if r["repeat"] == k])
        vals = [x["survived_share"] for x in rows if x["survived_share"] is not None]
        assert vals
        expected.append(statistics.fmean(vals))
    assert cell["M5"]["mean"] == pytest.approx(statistics.fmean(expected))
    assert cell["n_fixed"] == sum(1 for x in S.candidate_rows(recs) if x["survived_share"] is not None)


def test_m5_pools_the_two_kinds_of_fix_of_the_summary(grid):
    recs = [r for r in harness.load_records(os.path.join(grid, "g_a_b0_r1")) if r["repeat"] == 1]
    s = S.summarize_set(recs)
    n_r, n_l = s["M5_whole_route_replacement"]["n_replaced"], s["M5_localized_repair"]["n_repaired"]
    sv_r, sv_l = s["M5_whole_route_replacement"]["survived_share_mean"], s["M5_localized_repair"]["survived_share_mean"]
    pooled = sum(n * v for n, v in ((n_r, sv_r), (n_l, sv_l)) if n) / (n_r + n_l)
    assert T.repeat_metrics(recs)["M5"] == pytest.approx(pooled)


def test_a_single_repeat_has_a_value_and_no_sd(grid):
    recs = [r for r in harness.load_records(os.path.join(grid, "g_a_b0")) if r["repeat"] == 1]
    cell = T.cell_stats(recs)
    assert cell["n_repeats"] == 1 and cell["M1"]["sd"] is None and cell["M1"]["mean"] is not None
    assert T.cell_text(cell["M1"], "M1") == f"{cell['M1']['mean']:.3f}"


def test_the_api_calls_of_a_record_add_up_to_the_summary(grid):
    recs = copy.deepcopy(harness.load_records(os.path.join(grid, "g_a_b0_r1")))
    recs[0]["candidates"][0]["meta"]["n_calls"] = 2
    recs[0]["info"]["repair_calls"] = 3
    recs[1]["candidates"][1]["cascade"] = {"small": {"meta": {"n_calls": 2}}}
    assert sum(T.api_calls(r) for r in recs) / len(recs) == pytest.approx(S.summarize_set(recs)["M7"]["api_calls_per_part"])
    assert T.api_calls(recs[0]) == 5 + 1 + 3 and T.api_calls(recs[1]) == 5 + 2


def test_the_cost_of_old_records_is_charged_to_the_model_of_the_column(grid):
    recs = harness.load_records(os.path.join(grid, "g_a_b0"))
    priced = T.cell_stats(recs, "qwen/qwen3.8-27b")
    other = T.cell_stats(recs, "vendor/unpriced-model")
    assert priced["usd"]["mean"] == pytest.approx(5 * (100 * 0.80 + 20 * 4.00) / 1e6)       # 5 calls of 100 + 20 tokens
    assert other["usd"]["mean"] is None                                                     # no price: n/a, never 0
    assert priced["tokens"]["mean"] == pytest.approx(600)


def test_part_values_are_the_means_over_the_repeats_of_a_part(grid):
    recs = harness.load_records(os.path.join(grid, "g_b_b0_r1"))
    pv = T.part_values(recs)
    rows = S.part_rows(recs)
    assert sorted(pv) == sorted({r["part_id"] for r in recs}) and all(v["n_repeats"] == REPEATS for v in pv.values())
    for pid, v in pv.items():
        gaps = [r["eff_gap_to_rb_best"] for r in rows if r["part_id"] == pid and r["eff_gap_to_rb_best"] is not None]
        assert v["M6"] == pytest.approx(statistics.fmean(gaps))
        prov = [r["final_provenance"] != "route_builder" for r in rows if r["part_id"] == pid]
        assert v["M2p"] == pytest.approx(sum(prov) / len(prov))
        assert v["M7"] == 5.0


def test_binary_outcomes_have_one_entry_per_final_and_per_parsed_sample(grid):
    recs = harness.load_records(os.path.join(grid, "g_b_b0"))
    out = T.binary_outcomes(recs)
    assert len(out["final"]) == N_PARTS * REPEATS
    parsed = [c for r in recs for c in r["candidates"] if c["llm_ok"] and c["first_pass"] is not None]
    assert len(out["sample"]) == len(parsed) and sum(out["sample"].values()) == sum(1 for c in parsed if c["first_pass"]["valid"])
    assert sum(out["final"].values()) == sum(1 for r in recs if r["final"]["provenance"] != "route_builder")


# ================================================================ reading a run: what makes it count for a cell
def test_a_good_run_is_ok_and_carries_its_records(cfg, grid):
    run = one_run(cfg, grid)
    assert run["status"] == "ok" and not run["errors"] and not run["warnings"]
    assert len(run["records"]) == N_PARTS * REPEATS and run["repeats"] == [1, 2] and len(run["parts"]) == N_PARTS
    assert run["records_sha"] == T._sha(os.path.join(grid, "g_a_b0_r1", "records.jsonl"))


def test_a_missing_folder_is_missing_and_a_row_without_a_system_is_not_built(cfg, tmp_path):
    assert one_run(cfg, str(tmp_path))["status"] == "missing"
    assert one_run(cfg, str(tmp_path), "A6")["status"] == "not_built"


def test_a_run_of_another_system_does_not_count(cfg, grid, tmp_path):
    rd = copy_grid(grid, tmp_path, only=["g_a_b0"])
    shutil.copytree(os.path.join(rd, "g_a_b0"), os.path.join(rd, "g_a_b0_r1"))             # b0 records standing in for b0_r1
    run = one_run(cfg, rd)
    assert run["status"] == "error" and any("system" in e for e in run["errors"])


def test_run_json_of_another_system_dataset_or_model_does_not_count(cfg, grid, tmp_path):
    rd = copy_grid(grid, tmp_path, only=["g_a_b0_r1"])
    folder = os.path.join(rd, "g_a_b0_r1")
    edit_run_json(folder, system="b0")
    assert any("run.json says system" in e for e in one_run(cfg, rd)["errors"])
    edit_run_json(folder, system="b0_r1", dataset="test_v2")
    assert any("run.json says dataset" in e for e in one_run(cfg, rd)["errors"])
    edit_run_json(folder, dataset="dev_v2", model="openai/gpt-oss-20b")
    run = one_run(cfg, rd)
    assert run["status"] == "error" and any("CAPP_MODEL_PRIMARY" in e for e in run["errors"])


def test_a_run_made_with_the_wrong_model_is_caught_per_column(cfg, grid, tmp_path):
    rd = copy_grid(grid, tmp_path, only=["g_b_b0_r1"])
    shutil.copytree(os.path.join(rd, "g_b_b0_r1"), os.path.join(rd, "g_a_b0_r1"))           # model B's run under model A's tag
    assert one_run(cfg, rd, "A2", "a")["status"] == "error"
    assert one_run(cfg, rd, "A2", "b")["status"] == "ok"


def test_without_run_json_the_run_is_used_with_a_warning(cfg, grid, tmp_path):
    rd = copy_grid(grid, tmp_path, only=["g_a_b0_r1"])
    os.remove(os.path.join(rd, "g_a_b0_r1", "run.json"))
    run = one_run(cfg, rd)
    assert run["status"] == "partial" and any("run.json" in w for w in run["warnings"]) and run["records"]


def test_fewer_repeats_than_the_config_asks_is_partial(cfg, grid, tmp_path):
    rd = copy_grid(grid, tmp_path, only=["g_a_b0_r1"])
    rewrite_records(os.path.join(rd, "g_a_b0_r1"), lambda recs: [r for r in recs if r["repeat"] == 1])
    run = one_run(cfg, rd)
    assert run["status"] == "partial" and run["warnings"] == ["1 of 2 repeats"] and len(run["records"]) == N_PARTS


def test_more_repeats_than_the_config_asks_is_a_warning(cfg, grid, tmp_path):
    rd = copy_grid(grid, tmp_path, only=["g_a_b0_r1"])
    run = one_run(T.validate_config(make_cfg(grid, repeats=1)), rd)
    assert run["status"] == "partial" and "2 repeats, the config asks for 1" in run["warnings"][0]


def test_missing_parts_are_named(cfg, grid, tmp_path):
    rd = copy_grid(grid, tmp_path, only=["g_a_b0_r1"])
    drop = {"DEV_002", "DEV_005"}
    rewrite_records(os.path.join(rd, "g_a_b0_r1"), lambda recs: [r for r in recs if r["part_id"] not in drop])
    run = one_run(cfg, rd)
    assert run["status"] == "partial" and len(run["parts"]) == N_PARTS - 2
    assert f"{N_PARTS - 2} of {N_PARTS} parts" in run["warnings"][0] and "DEV_002" in run["warnings"][0] and "DEV_005" in run["warnings"][0]


def test_records_of_parts_outside_the_configured_set_are_ignored_with_a_warning(cfg, grid, tmp_path):
    rd = copy_grid(grid, tmp_path, only=["g_a_b0_r1"])
    small = T.validate_config(make_cfg(grid, limit=N_PARTS - 3))
    row = next(r for r in small["rows"] if r["id"] == "A2")
    run = T.load_run(small, row, small["models"][0], rd, T.expected_parts(small))
    assert run["status"] == "partial" and len(run["parts"]) == N_PARTS - 3
    assert any("outside the configured set" in w for w in run["warnings"])


def test_ragged_records_are_reported(cfg, grid, tmp_path):
    rd = copy_grid(grid, tmp_path, only=["g_a_b0_r1"])
    rewrite_records(os.path.join(rd, "g_a_b0_r1"), lambda recs: [r for r in recs if not (r["part_id"] == "DEV_003" and r["repeat"] == 2)])
    run = one_run(cfg, rd)
    assert run["status"] == "partial" and any("some (part, repeat) are missing" in w for w in run["warnings"])


def test_a_duplicate_part_and_repeat_is_an_error(cfg, grid, tmp_path):
    rd = copy_grid(grid, tmp_path, only=["g_a_b0_r1"])
    rewrite_records(os.path.join(rd, "g_a_b0_r1"), lambda recs: recs + [recs[0]])
    run = one_run(cfg, rd)
    assert run["status"] == "error" and "more than once" in run["errors"][0]


def test_a_part_that_differs_from_the_dataset_is_an_error(cfg, grid, tmp_path):
    rd = copy_grid(grid, tmp_path, only=["g_a_b0_r1"])

    def change(recs):
        recs[3]["part"]["features"] = recs[3]["part"]["features"] + ["Chamfer"] if "Chamfer" not in recs[3]["part"]["features"] \
            else recs[3]["part"]["features"][:-1]
        return recs
    rewrite_records(os.path.join(rd, "g_a_b0_r1"), change)
    run = one_run(cfg, rd)
    assert run["status"] == "error" and "differ from the dataset" in run["errors"][0]


@pytest.mark.parametrize("content,message", [("", "empty"), ("{broken\n", "cannot be read"), ('{"a": 1}\n', "lacks")])
def test_an_empty_corrupt_or_foreign_records_file_is_an_error(cfg, grid, tmp_path, content, message):
    rd = copy_grid(grid, tmp_path, only=["g_a_b0_r1"])
    with open(os.path.join(rd, "g_a_b0_r1", "records.jsonl"), "w", encoding="utf-8") as f:
        f.write(content)
    run = one_run(cfg, rd)
    assert run["status"] == "error" and message in run["errors"][0]


def test_a_run_that_cannot_be_analysed_is_an_error_not_a_crash(cfg, grid, tmp_path):
    rd = copy_grid(grid, tmp_path, only=["g_a_b1"])

    def damage(recs):
        for r in recs:
            r["candidates"] = [{"sample_id": 0}]
        return recs
    rewrite_records(os.path.join(rd, "g_a_b1"), damage)
    c = T.validate_config(make_cfg(rd, models=make_cfg(rd)["models"][:1], rows=[{"id": "A0", "label": "B1", "system": "b1"}]))
    r = T.analyse(c)
    assert r["runs"][("A0", "a")]["status"] == "error"
    assert any(i["level"] == "error" and "could not be analysed" in i["message"] for i in r["issues"])
    assert "–" in T.render_markdown(r)


def test_without_the_dataset_the_first_run_defines_the_parts(grid, tmp_path):
    c = T.validate_config(make_cfg(grid, dataset="no_such_dataset_v9"))
    assert T.expected_parts(c) == (None, {}, "unavailable")
    rd = copy_grid(grid, tmp_path, only=["g_a_b0_r1"])
    edit_run_json(os.path.join(rd, "g_a_b0_r1"), dataset="no_such_dataset_v9")
    run = one_run(c, rd)
    assert run["status"] == "ok" and len(run["records"]) == N_PARTS * REPEATS


# ================================================================ the whole grid
def test_every_cell_of_the_grid_is_read(res, cfg):
    assert len(res["runs"]) == len(cfg["rows"]) * len(cfg["models"])
    status = {k: r["status"] for k, r in res["runs"].items()}
    assert {s for (rid, _), s in status.items() if rid in ("A0", "A1", "A2", "A3", "A4", "A5")} == {"ok"}
    assert {s for (rid, _), s in status.items() if rid == "A6"} == {"not_built"}
    assert not [i for i in res["issues"] if i["level"] in ("error", "missing")]


def test_a_row_that_chooses_its_own_models_is_one_run_in_every_column(res, cfg):
    a = [res["runs"][("A5", m)] for m in ("a", "b", "c")]
    assert a[0]["folder"] == a[1]["folder"] == a[2]["folder"] and a[0]["folder"].endswith("g_shared_n5_r1_s3+ms")
    assert a[0]["cell"] == a[1]["cell"] == a[2]["cell"]
    assert a[0]["run_info"]["model"] == config.model("small") and a[0]["status"] == "ok"        # the small model, whatever the column says


def test_the_paired_results_cover_every_row_with_a_parent_for_every_model(res, cfg):
    comps = 5                                                                      # A1/A0, A2/A1, A3/A2, A4/A3, A5/A3 (A6 is not built)
    assert len(res["paired"]) == comps * 3 * len(T.PAIRED)
    assert len(res["mcnemar"]) == comps * 3 * 2
    assert not [p for p in res["paired"] if p["row"] == "A6"]                       # the parent of nothing built is not compared


def test_a_paired_difference_is_the_difference_of_the_part_means(res):
    a, b = res["runs"][("A2", "a")], res["runs"][("A1", "a")]
    p = next(x for x in res["paired"] if (x["model"], x["row"], x["metric"]) == ("a", "A2", "M6"))
    diffs = [a["part_values"][k]["M6"] - b["part_values"][k]["M6"] for k in a["part_values"]
             if a["part_values"][k]["M6"] is not None and b["part_values"][k]["M6"] is not None]
    assert p["n"] == len(diffs) and p["mean"] == pytest.approx(sum(diffs) / len(diffs)) and p["parent"] == "A1"


def test_repair_cannot_change_the_raw_validity_and_the_paired_numbers_say_so(res):
    for m in ("a", "b", "c"):
        p = next(x for x in res["paired"] if (x["model"], x["row"], x["metric"]) == (m, "A2", "M1"))     # A2 = A1 + R1: same raw samples
        assert p["mean"] == 0 and p["verdict"] == T.UNCLEAR
        q = next(x for x in res["mcnemar"] if (x["model"], x["row"], x["outcome"]) == (m, "A2", "sample"))
        assert q["row_only"] == 0 and q["parent_only"] == 0 and q["p"] == 1.0
        f = next(x for x in res["mcnemar"] if (x["model"], x["row"], x["outcome"]) == (m, "A2", "final"))
        assert f["row_only"] >= f["parent_only"]                                       # repair only ever adds LLM finals here


def test_the_mcnemar_table_adds_up(res):
    for q in res["mcnemar"]:
        assert q["both"] + q["row_only"] + q["parent_only"] + q["neither"] == q["n"]
        assert q["n"] > 0 and (q["n"] == N_PARTS * REPEATS if q["outcome"] == "final" else q["n"] >= N_PARTS * REPEATS)


def test_consistency_is_one_line_per_comparison_and_metric(res, cfg):
    cons = res["consistency"]
    assert {c["metric"] for c in cons} == set(T.CONSISTENCY_METRICS)
    assert len(cons) == 5 * len(T.CONSISTENCY_METRICS)
    for c in cons:
        assert set(c["verdicts"]) == {"a", "b", "c"} and c["n_models"] == 3 and c["label"]


def test_consistency_labels(cfg):
    def paired(row, metric, verdicts, means):
        return [{"model": m, "row": row, "parent": "A0", "metric": metric, "mean": mean, "lo": 0, "hi": 0, "n": 5, "verdict": v}
                for m, v, mean in zip("abc", verdicts, means)]
    one = {"id": "A1", "parent": "A0"}
    c = {"models": cfg["models"], "rows": [one]}
    lab = lambda verdicts, means=(1, 1, 1), metric="M6": T.consistency(c, paired("A1", metric, verdicts, means))[0]["label"]    # noqa: E731
    assert lab(["better"] * 3) == "better in all 3 LLMs"
    assert lab(["worse"] * 3) == "worse in all 3 LLMs"
    assert lab(["better", "worse", T.UNCLEAR]) == "mixed: the LLMs disagree"
    assert lab(["better", T.UNCLEAR, T.UNCLEAR]) == "better in 1 of 3 LLMs, no clear difference in the rest"
    assert lab([T.UNCLEAR, "worse", "worse"]) == "worse in 2 of 3 LLMs, no clear difference in the rest"
    assert lab([T.UNCLEAR] * 3) == "no clear difference in any of the 3 LLMs"
    partial = T.consistency(c, paired("A1", "M1", ["better", "better", "better"], (1, 1, 1))[:2])[0]
    assert partial["label"] == "better in all 2 LLMs (2 of 3 LLMs have both runs)" and partial["verdicts"]["c"] == "no data"
    solo = T.consistency(c, paired("A1", "M1", ["better"] * 3, (1, 1, 1))[:1])[0]
    assert solo["label"].startswith("one LLM only")


def test_fewer_than_three_llms_and_a_sealed_dataset_are_listed_as_issues(grid):
    c = T.validate_config(make_cfg(grid, models=make_cfg(grid)["models"][:2]))
    msgs = [i["message"] for i in T.analyse(c)["issues"]]
    assert any("at least 3" in m for m in msgs)
    sealed = T.validate_config(make_cfg(grid, dataset="test_v2"))
    assert any("sealed" in i["message"] for i in T.analyse(sealed)["issues"])


def test_missing_runs_make_dashes_and_missing_issues_but_do_not_stop_the_table(grid, tmp_path):
    rd = copy_grid(grid, tmp_path)
    shutil.rmtree(os.path.join(rd, "g_b_b0_r1"))
    c = T.validate_config(make_cfg(rd))
    r = T.analyse(c)
    assert r["runs"][("A2", "b")]["status"] == "missing"
    assert any(i["level"] == "missing" and i["row"] == "A2" and i["model"] == "b" for i in r["issues"])
    assert not [p for p in r["paired"] if p["model"] == "b" and "A2" in (p["row"], p["parent"])]       # nothing to pair with
    assert any(p["model"] == "b" and p["row"] == "A1" for p in r["paired"])                              # the rest is still compared
    md = T.render_markdown(r)
    block = md.split("## Model B")[1].split("## Model C")[0]
    line = next(x for x in block.splitlines() if x.startswith("| A2 "))
    assert line.count("–") == len(T.COLUMNS) and "no run yet" in block


def test_a_partial_run_is_marked_and_explained(grid, tmp_path):
    rd = copy_grid(grid, tmp_path)
    rewrite_records(os.path.join(rd, "g_a_b0_r1"), lambda recs: [r for r in recs if r["repeat"] == 1])
    r = T.analyse(T.validate_config(make_cfg(rd)))
    md = T.render_markdown(r)
    assert "| A2† |" in md and "† A2: 1 of 2 repeats" in md
    assert "| A2† |" not in md.split("## Model B")[1]
    assert "$^\\dagger$" in T.render_latex(r)


# ================================================================ the files
def test_markdown_has_every_row_model_and_the_reference_note(res):
    md = T.render_markdown(res)
    for label in ("Model A", "Model B", "Model C"):
        assert f"## {label}" in md
    assert md.count("| A0 ") == 3 and md.count("| A6 ") == 3 and "not built" in md
    assert "proxy_rb" in md and "NOT to an expert" in md
    assert "±" in md and "decide: this one" in md and "one run for every LLM" in md


def test_the_not_built_row_shows_its_note_once(res):
    notes = [line for line in T.render_markdown(res).splitlines() if line.startswith("- A6:")]
    assert notes == ["- A6: not built"] * 3                                         # the note says "not built" already: not repeated


def test_the_unpriced_model_has_na_dollars_not_zero(res):
    md = T.render_markdown(res)
    c_block = md.split("## Model C")[1]
    row = next(x for x in c_block.splitlines() if x.startswith("| A1 "))
    assert row.rstrip(" |").endswith("n/a")
    a_row = next(x for x in md.split("## Model B")[0].splitlines() if x.startswith("| A1 "))
    assert not a_row.rstrip(" |").endswith("n/a") and "0.00" in a_row.split("|")[-2]


def test_csv_has_one_row_per_model_and_row_and_the_numbers_of_the_cells(res, cfg):
    rows = T.table_csv_rows(res)
    assert len(rows) == len(cfg["models"]) * len(cfg["rows"])
    r = next(x for x in rows if x["model"] == "a" and x["row"] == "A2")
    cell = res["runs"][("A2", "a")]["cell"]
    assert float(r["M1_mean"]) == cell["M1"]["mean"] and float(r["M6_sd"]) == cell["M6"]["sd"] and r["system"] == "b0_r1"
    assert (r["parent"], r["status"], r["n_repeats"], r["m4_source"]) == ("A1", "ok", 2, "proxy_rb")
    nb = next(x for x in rows if x["row"] == "A6")
    assert nb["status"] == "not_built" and nb["M1_mean"] == "" and nb["system"] == ""
    assert next(x for x in rows if x["model"] == "c" and x["row"] == "A1")["usd_mean"] == ""


def test_latex_is_a_booktabs_tabular_per_model_with_escaped_text(res):
    tex = T.render_latex(res)
    assert tex.count(r"\begin{tabular}") == 3 and tex.count(r"\toprule") == 3 and tex.count(r"\bottomrule") == 3
    assert r"\pm" in tex and "--" in tex and "not built" in tex
    assert "_" not in tex.replace(r"\_", "").split("\\begin{tabular}")[1].split("\\end{tabular}")[0]


def test_tex_escaping():
    assert T._tex("a_b & 5% #1 {x} $y") == r"a\_b \& 5\% \#1 \{x\} \$y"
    assert T._tex("a\\b") == r"a\textbackslash{}b"
    assert T._tex("x^2~y") == r"x\textasciicircum{}2\textasciitilde{}y"


def test_cell_text():
    assert T.cell_text({"mean": 0.12345, "sd": 0.01, "n": 2}, "M1") == "0.123 ± 0.010"
    assert T.cell_text({"mean": 0.12345, "sd": 0.01, "n": 2}, "M1", latex=True) == r"0.123 \pm 0.010"
    assert T.cell_text({"mean": 1.5, "sd": None, "n": 1}, "M6") == "1.50"
    assert T.cell_text({"mean": None, "sd": None, "n": 0}, "M6") == "–" and T.cell_text(None, "M6") == "–"


def test_the_paired_report_and_csv(res):
    md = T.render_paired(res)
    assert "### A2 vs A1" in md and "### A4 vs A3" in md and "McNemar" in md and "no correction" in md.lower()
    assert "### A6 vs A4" in md and "not compared" in md
    rows = T.paired_csv_rows(res)
    assert len(rows) == len(res["paired"]) + len(res["mcnemar"])
    assert {r["test"] for r in rows} == {"paired bootstrap", "McNemar exact"}
    b = next(r for r in rows if r["test"] == "paired bootstrap" and r["model"] == "a" and r["row"] == "A1" and r["metric"] == "M6")
    p = next(x for x in res["paired"] if (x["model"], x["row"], x["metric"]) == ("a", "A1", "M6"))
    assert float(b["mean_diff"]) == p["mean"] and float(b["ci_lo"]) == p["lo"]


def test_the_consistency_report(res):
    md = T.render_consistency(res)
    assert "| Model A | Model B | Model C | Across LLMs |" in md and "A2 vs A1" in md


def test_per_part_csv_has_one_row_per_usable_run_and_part(res):
    rows = T.per_part_rows(res)
    assert len(rows) == sum(len(r["part_values"]) for r in res["usable"].values()) == (len(res["usable"]) * N_PARTS)
    r = next(x for x in rows if x["model"] == "a" and x["row"] == "A1" and x["part_id"] == "DEV_001")
    assert r["system"] == "b0" and float(r["M7"]) == 5.0 and r["n_repeats"] == REPEATS


def test_issues_text_orders_errors_first(grid, tmp_path):
    rd = copy_grid(grid, tmp_path)
    shutil.rmtree(os.path.join(rd, "g_b_b0_r1"))
    edit_run_json(os.path.join(rd, "g_a_b0"), dataset="test_v2")
    text = T.render_issues(T.analyse(T.validate_config(make_cfg(rd))))
    lines = text.strip().splitlines()
    assert lines[0].startswith("ERROR") and "run.json says dataset" in lines[0]
    assert any(line.startswith("MISSING") for line in lines) and lines.index(next(x for x in lines if x.startswith("MISSING"))) > 0
    assert "No issues" in T.render_issues({"issues": []})


def test_build_writes_every_file_and_a_manifest(grid, cfg, tmp_path):
    out = str(tmp_path / "out")
    r = T.build(cfg, out)
    names = {"table4.md", "table4.csv", "table4.tex", "paired.md", "paired.csv", "consistency.md", "per_part.csv", "issues.txt",
             "fig_ablation.png", "fig_cost.png", "manifest.json"}
    assert set(os.listdir(out)) == names and set(r["files"]) == names
    m = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
    assert m["config_sha256_16"] == T.config_sha(cfg) and m["config"] == cfg and m["dataset_checked_against"] == "dataset"
    run = next(x for x in m["runs"] if x["row"] == "A2" and x["model"] == "a")
    assert run["records_sha256_16"] == T._sha(os.path.join(grid, "g_a_b0_r1", "records.jsonl")) and run["folder"] == "g_a_b0_r1"
    assert run["run_json"]["model"] == "qwen/qwen3.8-27b" and run["n_records"] == N_PARTS * REPEATS and run["status"] == "ok"
    assert next(x for x in m["runs"] if x["row"] == "A6")["folder"] is None
    for name, digest in m["files"].items():
        assert digest == T._sha(os.path.join(out, name))
    assert "fig_ablation.png" not in m["files"]


def test_the_same_runs_and_config_give_byte_identical_text_files(cfg, tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    T.build(cfg, a, figures=False)
    T.build(cfg, b, figures=False)
    for name in sorted(os.listdir(a)):
        assert open(os.path.join(a, name), "rb").read() == open(os.path.join(b, name), "rb").read(), name


def test_build_does_not_modify_the_runs(grid, cfg, tmp_path):
    def hashes():
        return {f: [T._sha(os.path.join(grid, f, n)) for n in ("records.jsonl", "run.json")] for f in sorted(os.listdir(grid))}
    before = hashes()
    T.build(cfg, str(tmp_path / "o"), figures=False)
    assert before == hashes() and len(before) == 16


# ================================================================ figures
def test_the_figures_are_png_files(cfg, tmp_path):
    r = T.build(cfg, str(tmp_path / "f"))
    for name in ("fig_ablation.png", "fig_cost.png"):
        data = open(r["files"][name], "rb").read()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 10000


def test_the_palette_is_the_validated_one_with_a_marker_shape_per_series():
    assert tablefigs.PALETTE == ("#2a78d6", "#eb6834", "#1baf7a")
    assert len(set(tablefigs.MARKERS)) == len(tablefigs.PALETTE) == tablefigs.MAX_LLMS == 3


def test_a_fourth_llm_is_left_out_of_the_figures_but_not_of_the_tables(grid, tmp_path):
    raw = make_cfg(grid)
    raw["models"].append({"id": "d", "label": "Model D", "model": "vendor/fourth", "tag": "g_d"})
    c = T.validate_config(raw)
    r = T.analyse(c)
    made, note = tablefigs.make_figures(r, str(tmp_path))
    assert made == ["fig_ablation.png", "fig_cost.png"] and "first 3 LLMs" in note
    assert "## Model D" in T.render_markdown(r)


def test_no_figures_without_a_usable_run(grid, tmp_path):
    r = T.analyse(T.validate_config(make_cfg(str(tmp_path / "empty"))))
    made, note = tablefigs.make_figures(r, str(tmp_path))
    assert made == [] and "no usable run" in note


def test_one_llm_makes_figures_without_a_legend(grid, tmp_path):
    c = T.validate_config(make_cfg(grid, models=make_cfg(grid)["models"][:1]))
    made, _ = tablefigs.make_figures(T.analyse(c), str(tmp_path))
    assert made == ["fig_ablation.png", "fig_cost.png"]


def test_build_without_figures_makes_none(cfg, tmp_path):
    T.build(cfg, str(tmp_path / "n"), figures=False)
    assert not [f for f in os.listdir(tmp_path / "n") if f.endswith(".png")]


# ================================================================ status and the run commands
def test_status_grid_shows_each_cell(res, cfg):
    text = T.status_grid(cfg, res)
    assert "Every run is present and complete." in text and "not built" in text and "(shared)" in text
    assert text.splitlines()[0].split()[:2] == ["Model", "A"]


def test_commands_for_the_template_on_an_empty_results_folder(tmp_path):
    cfg = T.load_config(os.path.join(ROOT, "configs", "ablation_template.json"))
    lines = T.run_commands(cfg, T.load_all(cfg, str(tmp_path)))
    cmds = [x for x in lines if x.startswith("python scripts/run_eval.py")]
    assert len(cmds) == 3 * 8 + 1                                                          # 8 rows x 3 LLMs, and the cascade once
    assert sum(1 for x in lines if x.startswith("$env:CAPP_MODEL_PRIMARY")) == 3
    assert '$env:CAPP_MODEL_PRIMARY = "qwen/qwen3.8-27b"' in lines and 'Remove-Item Env:CAPP_MODEL_PRIMARY -ErrorAction SilentlyContinue' in lines
    assert sum("--tag abl_shared" in x for x in cmds) == 1 and all("--limit 20" in x and "--repeats 3" in x and "--dataset dev_v2" in x for x in cmds)
    assert "# Not built, nothing to run: A9" in lines
    assert lines[0].startswith("# 25 run(s)")
    for x in cmds:                                                                          # every command is something run_eval accepts
        system = x.split("--system ")[1].split()[0]
        assert systems.parse_system(system) is not None


def test_commands_put_the_five_sample_system_before_the_one_sample_system(tmp_path):
    cfg = T.load_config(os.path.join(ROOT, "configs", "ablation_template.json"))
    cmds = [x for x in T.run_commands(cfg, T.load_all(cfg, str(tmp_path))) if "--tag abl_qwen " in x]
    systems_in_order = [x.split("--system ")[1].split()[0] for x in cmds]
    assert systems_in_order.index("b0") < systems_in_order.index("b1")                    # B1's one sample is B0's first: from the cache
    assert systems_in_order.index("b0_r1") < systems_in_order.index("b1")
    assert systems_in_order.index("b0_r1+ts+h1") < systems_in_order.index("n5_r1_s3+ts+h1+fs2")


def test_commands_name_the_few_shot_pool_only_when_it_is_not_the_default(tmp_path):
    raw = make_cfg(str(tmp_path), rows=[{"id": "A0", "label": "l", "system": "n5_r1_s3+fs2"}], shared_tag=None)
    assert "--pool" not in "\n".join(T.run_commands(T.validate_config(raw), T.load_all(T.validate_config(raw), str(tmp_path))))
    raw["pool"] = "other_dev"
    c = T.validate_config(raw)
    assert "--pool other_dev" in "\n".join(T.run_commands(c, T.load_all(c, str(tmp_path))))


def test_commands_after_the_runs_exist(grid, cfg):
    lines = T.run_commands(cfg, T.load_all(cfg))
    assert [x for x in lines if x.startswith("python")] == [] and "# Nothing to run" in lines[0]


def test_commands_for_a_partial_and_an_unusable_run(grid, tmp_path):
    rd = copy_grid(grid, tmp_path)
    rewrite_records(os.path.join(rd, "g_a_b0_r1"), lambda recs: [r for r in recs if r["repeat"] == 1])
    edit_run_json(os.path.join(rd, "g_b_b0"), model="somebody/else")
    c = T.validate_config(make_cfg(rd))
    cmds = [x for x in T.run_commands(c, T.load_all(c)) if x.startswith("python")]
    assert len(cmds) == 2
    assert any("--tag g_a" in x and "--system b0_r1" in x and "incomplete" in x and "1 of 2 repeats" in x for x in cmds)
    assert any("--tag g_b" in x and "--system b0 " in x and "will be overwritten" in x for x in cmds)


def test_commands_warn_about_a_sealed_dataset(tmp_path):
    c = T.validate_config(make_cfg(str(tmp_path), dataset="test_v2", limit=None))
    lines = T.run_commands(c, T.load_all(c))
    assert lines[0].startswith("# WARNING: test_v2 is sealed") and not any("--limit" in x for x in lines)


# ================================================================ the command line
def write_cfg(path, **over):
    path.write_text(json.dumps(make_cfg(**over)), encoding="utf-8")
    return str(path)


def test_cli_status_and_commands(grid, tmp_path, capsys):
    p = write_cfg(tmp_path / "c.json", results_dir=grid)
    assert make_tables.main(["status", p]) == 0
    assert "Every run is present and complete." in capsys.readouterr().out
    assert make_tables.main(["commands", p]) == 0
    assert "Nothing to run" in capsys.readouterr().out
    p2 = write_cfg(tmp_path / "c2.json", results_dir=str(tmp_path / "none"))
    assert make_tables.main(["commands", p2]) == 0
    assert "python scripts/run_eval.py --system b1" in capsys.readouterr().out
    assert make_tables.main(["status", p2, "--results", grid]) == 0                       # --results overrides the config
    assert "Every run is present" in capsys.readouterr().out


def test_cli_build_writes_into_results_tables_name_by_default(grid, tmp_path, capsys):
    rd = copy_grid(grid, tmp_path)
    p = write_cfg(tmp_path / "c.json", results_dir=rd, name="mygrid")
    assert make_tables.main(["build", p, "--no-figures"]) == 0
    out = capsys.readouterr().out
    assert "No issues" in out and os.path.exists(os.path.join(rd, "tables", "mygrid", "table4.md"))


def test_cli_build_with_out_and_strict(grid, tmp_path, capsys):
    p = write_cfg(tmp_path / "c.json", results_dir=grid)
    assert make_tables.main(["build", p, "--out", str(tmp_path / "o"), "--no-figures", "--strict"]) == 0
    capsys.readouterr()
    p2 = write_cfg(tmp_path / "c2.json", results_dir=str(tmp_path / "none"))
    assert make_tables.main(["build", p2, "--out", str(tmp_path / "o2"), "--no-figures"]) == 0       # missing runs: listed, not fatal
    assert "MISSING" in capsys.readouterr().out
    assert make_tables.main(["build", p2, "--out", str(tmp_path / "o3"), "--no-figures", "--strict"]) == 1
    assert "--strict" in capsys.readouterr().err
    assert "–" in open(tmp_path / "o3" / "table4.md", encoding="utf-8").read()


def test_cli_with_a_bad_config_exits_1_and_names_the_problem(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(make_cfg("x", rows=[{"id": "A0", "label": "l", "system": "b9"}])), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        make_tables.main(["status", str(bad)])
    assert e.value.code == 1 and "unknown system" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        make_tables.main(["build", str(tmp_path / "missing.json")])
    assert e.value.code == 1 and "cannot read" in capsys.readouterr().err


def test_cli_rejects_an_unknown_command():
    with pytest.raises(SystemExit) as e:
        make_tables.main(["nonsense", "x.json"])
    assert e.value.code == 2


def test_the_template_builds_on_an_empty_results_folder_with_nothing_but_dashes(tmp_path, capsys):
    out = str(tmp_path / "t")
    rc = make_tables.main(["build", os.path.join(ROOT, "configs", "ablation_template.json"), "--results", str(tmp_path / "empty"), "--out", out])
    assert rc == 0
    md = open(os.path.join(out, "table4.md"), encoding="utf-8").read()
    assert md.count("| A0 ") == 3 and "not built" in md
    assert "no usable run" in open(os.path.join(out, "issues.txt"), encoding="utf-8").read()
    assert not [f for f in os.listdir(out) if f.endswith(".png")]
