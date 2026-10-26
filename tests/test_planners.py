import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import compare                # noqa: E402
from capp import dataset as D           # noqa: E402
from capp import harness as H           # noqa: E402
from capp import llm_client             # noqa: E402
from capp import planners as P          # noqa: E402
from capp import plannerspec as PS      # noqa: E402
from capp import summary as SM          # noqa: E402
from capp import systems as SY          # noqa: E402
from capp import verify as V            # noqa: E402
import llm_planner                      # noqa: E402
import route_builder as rb              # noqa: E402

FS = PS.PlannerSpec


@pytest.fixture(scope="module")
def dev():
    return D.load_dataset("dev_v2")


@pytest.fixture(scope="module")
def pool(dev):
    return P.Pool.from_dataset(dev, name="dev_v2")


@pytest.fixture(autouse=True)
def clean_client(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GROQ_API_KEY", "")                      # a test can never reach the real API
    monkeypatch.delenv("CAPP_LLM_MODE", raising=False)
    monkeypatch.delenv("CAPP_RUN_TAG", raising=False)
    llm_client.reset_transport()
    llm_client.reset_call_log()
    yield
    llm_client.reset_transport()


def _features_of(messages):
    m = re.search(r"Features: (.*)\n", messages[-1]["content"])
    return m.group(1).split(", ")


def capture():
    """Transport that answers with the Route Builder's route for the features in the last user message."""
    seen = []

    def transport(model, messages, params):
        seen.append({"model": model, "messages": messages, "params": params})
        route = V.clean_steps(rb.generate_valid_routes(_features_of(messages), max_routes=1)[0]["steps"])
        return {"text": json.dumps(route), "prompt_tokens": 100 * len(messages), "completion_tokens": 30, "finish_reason": "stop"}
    llm_client.set_transport(transport)
    return seen


def ex(pid, material, feats, plan=None):
    return P.Example(pid, material, tuple(feats), "0.05mm", 100, tuple(plan or ("Facing", pid, "Inspection")))


# ---- names ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,base,spec", [
    ("b0", "b0", FS()),
    ("b0+fs0+h0", "b0", FS()),
    ("b0+h0", "b0", FS()),
    ("b0+fs2", "b0", FS(few_shot=2)),
    ("n5_r1_s3+fs4x", "n5_r1_s3", FS(few_shot=4, loo="features")),
    ("n5_r1_s3+h1", "n5_r1_s3", FS(hint=True)),
    ("n5_r1_s3+h1+fs2", "n5_r1_s3", FS(few_shot=2, hint=True)),
    ("b1_r2+fs2x+h1", "b1_r2", FS(few_shot=2, loo="features", hint=True)),
])
def test_split_name(name, base, spec):
    assert PS.split_name(name) == (base, spec)


def test_default_spec_and_canonical_names():
    assert FS().is_default and FS().name == ""
    assert not FS(few_shot=2).is_default and not FS(hint=True).is_default
    assert FS(few_shot=4, loo="features", hint=True).name == "fs4x+h1"
    assert PS.split_name("b0+h1+fs2")[1].name == "fs2+h1"


@pytest.mark.parametrize("bad", ["b0+fs3", "b0+fs1", "b0+fs2+fs4", "b0+h1+h0", "b0+", "b0++fs2", "b0+fs2xx", "b0+hint", "b0+FS2"])
def test_bad_options_are_rejected(bad):
    with pytest.raises(ValueError):
        PS.split_name(bad)
    with pytest.raises(ValueError):
        SY.parse_system(bad)


def test_parse_system_ignores_options_but_rb_takes_none():
    assert SY.parse_system("b0+fs2+h1") == SY.parse_system("b0") == (5, "r0", "s1")
    assert SY.parse_system("n8_r1_s3+fs4x") == (8, "r1", "s3")
    assert SY.planner_spec("n8_r1_s3+fs4x") == FS(few_shot=4, loo="features")
    assert SY.is_llm("b0+h1") and SY.reprompts("b0_r2+fs2") and not SY.reprompts("b0_r1+fs2")
    for bad in ("rb+fs2", "rb+h1", "rb+h0"):
        with pytest.raises(ValueError):
            SY.parse_system(bad)
    assert callable(SY.resolve("n5_r1_s3+fs2"))


# ---- the default planner is v1, byte for byte ---------------------------------------------------------------
def test_default_options_send_exactly_the_v1_request_and_share_its_cache(dev):
    seen = capture()
    part = dev[3]
    kw = dict(tolerance=part["tolerance"], batch_size=part["batch_size"], sample_id=2)
    v1 = llm_planner.generate_process_plan(part["material"], part["features"], **kw)
    ours = P.make_factory(FS())(part)(part["material"], part["features"], **kw)
    assert len(seen) == 1                                     # the second call was answered from v1's cache entry
    assert ours["meta"]["cached"] is True and ours["steps"] == v1["steps"]
    assert ours["model"] == v1["model"]
    assert ours["meta"]["prompt_context"] == {"examples": [], "hint": False}
    msgs = P.build_messages(part["material"], part["features"], part["tolerance"], part["batch_size"])
    assert msgs == seen[0]["messages"]


def test_default_harness_run_makes_no_new_calls_after_plain_b0(dev):
    seen = capture()
    parts = dev[:3]
    H.run_experiment("b0", parts, tag="x", progress=None, write=False)
    n = len(seen)
    assert n == 15
    H.run_experiment("b0+fs0+h0", parts, tag="x", progress=None, write=False)
    assert len(seen) == n


# ---- retrieval ----------------------------------------------------------------------------------------------
def small_pool():
    return P.Pool([
        ex("P1", "Steel", ["Hole", "Face"], ("Facing", "Drilling", "Inspection")),
        ex("P2", "Aluminum", ["Hole", "Face", "Chamfer"]),
        ex("P3", "Steel", ["Face", "Chamfer", "Hole"]),               # same feature set as P2, Steel
        ex("P4", "Steel", ["Hole", "Face"], ("Facing", "Drilling", "Inspection")),      # same set and plan as P1
        ex("P5", "Steel", ["Slot"]),
        ex("P6", "Steel", ["Hole", "Face", "Keyway"], ("Facing", "Drilling", "Inspection")),   # plan equal to P1's
    ], name="small")


QUERY = {"part_id": "Q", "material": "Steel", "features": ["Hole", "Face", "Chamfer"]}


def ids(exs):
    return [e.part_id for e in exs]


def test_most_similar_first_with_material_tie_break_and_one_example_per_feature_set():
    # P2 and P3 are identical problems; the Steel one is shown. P1/P4 likewise. P6 is dropped: its plan equals P1's.
    assert ids(small_pool().retrieve(QUERY, 3)) == ["P3", "P1", "P5"]
    assert ids(small_pool().retrieve(QUERY, 2)) == ["P3", "P1"]
    q2 = dict(QUERY, material="Aluminum")
    assert ids(small_pool().retrieve(q2, 1)) == ["P2"]            # now Aluminum is the closer material


def test_material_class_is_the_second_tie_break():
    pl = P.Pool([ex("A", "Aluminum", ["Hole"]), ex("B", "Titanium", ["Hole"], ("Facing", "B", "Inspection")),
                 ex("C", "Brass", ["Hole"], ("Facing", "C", "Inspection"))])
    # three identical feature sets form ONE group; its representative is the closest material
    assert ids(pl.retrieve({"part_id": "Q", "material": "Copper", "features": ["Hole"]}, 3)) == ["A"]
    assert P.material_rank("Steel", "Steel") == 0 and P.material_rank("Steel", "Cast Iron") == 1
    assert P.material_rank("Steel", "Aluminum") == 2 and P.material_rank("Brass", "Copper") == 1
    assert P.material_rank("Steel", "Unobtainium") == 2 and P.material_rank("Unobtainium", "Unobtainium") == 0
    # B and C are equally similar (2/3); Cast Iron is the same class as Steel, Titanium is not
    pl2 = P.Pool([ex("A", "Aluminum", ["Hole"]), ex("B", "Titanium", ["Hole", "Face"], ("Facing", "B", "Inspection")),
                  ex("C", "Cast Iron", ["Hole", "Slot"], ("Facing", "C", "Inspection"))])
    assert ids(pl2.retrieve({"part_id": "Q", "material": "Steel", "features": ["Hole", "Face", "Slot"]}, 3)) == ["C", "B", "A"]


def test_equal_similarity_and_material_is_decided_by_part_id():
    pl = P.Pool([ex("Z", "Steel", ["Hole", "Face"]), ex("M", "Steel", ["Hole", "Slot"])])
    assert ids(pl.retrieve({"part_id": "Q", "material": "Steel", "features": ["Hole"]}, 2)) == ["M", "Z"]


def test_leave_one_out_by_part_and_by_feature_set():
    q_in_pool = dict(QUERY, part_id="P3")
    assert "P3" not in ids(small_pool().retrieve(q_in_pool, 5))
    assert ids(small_pool().retrieve(q_in_pool, 1)) == ["P2"]           # the identical problem of another part is still there
    assert ids(small_pool().retrieve(q_in_pool, 5, loo="features")) == ["P1", "P5"]
    assert ids(small_pool().retrieve(QUERY, 5, loo="features")) == ["P1", "P5"]


def test_k_larger_than_pool_empty_pool_and_k_zero():
    assert len(small_pool().retrieve(QUERY, 50)) == 3                   # only three distinct problems/plans exist
    assert P.Pool([]).retrieve(QUERY, 4) == []
    assert small_pool().retrieve(QUERY, 0) == [] and small_pool().retrieve(QUERY, -1) == []


def test_retrieval_does_not_mutate_the_pool_and_is_deterministic():
    pl = small_pool()
    first = ids(pl.retrieve(QUERY, 3))
    assert [ids(pl.retrieve(QUERY, 3)) for _ in range(5)] == [first] * 5 and len(pl) == 6


def test_dev_pool_has_a_strict_valid_plan_for_every_part(dev, pool):
    assert len(pool) == len(dev) == 200
    by_id = {p["part_id"]: p for p in dev}
    for e in pool.examples:
        assert V.check_route(list(e.plan), by_id[e.part_id]["features"]).valid


def test_expert_reference_route_is_preferred_and_invalid_ones_are_skipped(dev):
    p = dict(dev[0], part_id="E1", features=["Hole"], reference_routes=[["Facing", "Center Drilling", "Drilling", "Inspection"]])
    bad = dict(dev[0], part_id="E2", features=["Hole"], reference_routes=[["Drilling", "Facing"]])
    pl = P.Pool.from_dataset([p, bad])
    assert ids(pl.examples) == ["E1"] and pl.examples[0].plan == ("Facing", "Center Drilling", "Drilling", "Inspection")


def test_on_the_real_dev_pool_examples_are_distinct_sorted_and_never_the_part_itself(dev, pool):
    for part in dev[::7]:
        for loo in ("part", "features"):
            got = pool.retrieve(part, 4, loo)
            assert len(got) == 4 and part["part_id"] not in ids(got)
            assert len({e.plan for e in got}) == 4 and len({frozenset(e.features) for e in got}) == 4
            js = [P.jaccard(part["features"], e.features) for e in got]
            assert js == sorted(js, reverse=True)
            if loo == "features":
                assert all(set(e.features) != set(part["features"]) for e in got)
            allowed = [e for e in pool.examples if e.part_id != part["part_id"]
                       and (loo == "part" or set(e.features) != set(part["features"]))]
            assert js[0] == max(P.jaccard(part["features"], e.features) for e in allowed)      # nothing closer was skipped


# ---- the prompt -----------------------------------------------------------------------------------------------
def test_few_shot_messages_are_chat_turns_with_the_most_similar_example_last():
    exs = small_pool().retrieve(QUERY, 2)                                # P3, P1
    msgs = P.build_messages("Steel", QUERY["features"], "0.02mm", 250, exs)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[0] == P.build_messages("Steel", ["Hole"], "0.05mm", 1)[0]
    # least similar (P1) first, most similar (P3) right before the question
    assert "Features: Hole, Face\n" in msgs[1]["content"] and json.loads(msgs[2]["content"]) == list(exs[1].plan)
    assert "Features: Face, Chamfer, Hole\n" in msgs[3]["content"] and json.loads(msgs[4]["content"]) == list(exs[0].plan)
    assert msgs[5] == {"role": "user", "content": llm_planner_prompt("Steel", QUERY["features"], "0.02mm", 250)}
    # example questions are written exactly like the real one
    assert msgs[1]["content"] == llm_planner_prompt("Steel", ["Hole", "Face"], "0.05mm", 100)


def llm_planner_prompt(material, features, tolerance, batch_size):
    seen = capture()
    llm_planner.generate_process_plan(material, features, tolerance, batch_size, sample_id=99)
    llm_client.reset_transport()
    return seen[0]["messages"][-1]["content"]


def test_hint_is_the_route_builder_route_inside_the_user_message(dev):
    part = dev[5]
    hint = P.rb_plan(part["features"], part.get("machine_preference", "auto"))
    msgs = P.build_messages(part["material"], part["features"], part["tolerance"], part["batch_size"], (), hint)
    assert [m["role"] for m in msgs] == ["system", "user"]
    plain = P.build_messages(part["material"], part["features"], part["tolerance"], part["batch_size"])[1]["content"]
    assert msgs[1]["content"].startswith(plain[:-len("Output ONLY a JSON array of operation names.")])
    assert msgs[1]["content"].endswith("Output ONLY a JSON array of operation names.")
    assert json.dumps(hint) in msgs[1]["content"] and "rule-based planner proposes" in msgs[1]["content"]
    assert not any(s.startswith("---") for s in hint)               # no changeover markers in the hint


def test_hint_equals_the_route_of_the_rb_baseline(dev):
    for part in dev[:25]:
        final = SY.run_rb(part).to_dict()["final"]["steps"]
        assert P.rb_plan(part["features"], part.get("machine_preference", "auto")) == V.clean_steps(final)


def test_factory_planner_has_the_v1_result_shape_and_reports_its_context(dev, pool):
    seen = capture()
    part = dev[10]
    planner = P.make_factory(FS(few_shot=2, hint=True), pool)(part)
    res = planner(part["material"], part["features"], part["tolerance"], part["batch_size"], sample_id=1)
    assert set(res) >= {"success", "steps", "raw_response", "model", "meta"} and res["success"]
    ctx = res["meta"]["prompt_context"]
    assert len(ctx["examples"]) == 2 and part["part_id"] not in ctx["examples"] and ctx["hint"] is True
    assert len(seen[0]["messages"]) == 6 and seen[0]["params"]["temperature"] == 0.3
    assert res["meta"]["prompt_tokens"] == 600


def test_few_shot_without_a_pool_is_an_error():
    with pytest.raises(ValueError):
        P.make_factory(FS(few_shot=2))
    P.make_factory(FS(hint=True))                       # the hint alone needs no pool


def test_missing_key_gives_the_message_the_harness_guard_stops_on(dev, pool):
    part = dev[0]
    res = P.make_factory(FS(few_shot=2), pool)(part)(part["material"], part["features"])
    assert res["success"] is False and res["raw_response"] == H.NO_KEY_MSG


# ---- harness ------------------------------------------------------------------------------------------------------
def test_fs2_run_sends_six_message_conversations_and_records_the_examples(dev, tmp_path):
    seen = capture()
    parts = dev[:4]
    out = H.run_experiment("b0+fs2", parts, tag="fs", out_dir=str(tmp_path), progress=None)
    assert len(seen) == 20 and all(len(s["messages"]) == 6 for s in seen)
    for rec in out["records"]:
        for c in rec["candidates"]:
            exs = c["meta"]["prompt_context"]["examples"]
            assert len(exs) == 2 and rec["part_id"] not in exs and c["meta"]["prompt_context"]["hint"] is False
    info = json.loads((tmp_path / "fs_b0+fs2" / "run.json").read_text())
    assert info["planner_options"] == "fs2" and info["few_shot_k"] == 2 and info["few_shot_leave_out"] == "part"
    assert info["pool"] == ["dev_v2", 200] and info["hint"] is False and info["system"] == "b0+fs2"
    csv_text = (tmp_path / "fs_b0+fs2" / "per_candidate.csv").read_text()
    assert "examples" in csv_text.splitlines()[0] and "DEV_" in csv_text
    s = out["summary"]["planner_prompt"]
    assert s["examples_per_call"] == 2.0 and s["hint_share"] == 0.0
    assert "planner prompt" in SM.format_summary("x", out["summary"])


def test_new_prompts_mean_new_cache_keys_and_a_second_run_is_free(dev):
    seen = capture()
    parts = dev[:3]
    H.run_experiment("b0", parts, tag="k", progress=None, write=False)
    n0 = len(seen)
    H.run_experiment("b0+fs2", parts, tag="k", progress=None, write=False)
    n1 = len(seen)
    assert n1 == n0 + 15
    H.run_experiment("b0+fs2", parts, tag="k", progress=None, write=False)
    H.run_experiment("b0+h1", parts, tag="k", progress=None, write=False)
    assert len(seen) == n1 + 15
    H.run_experiment("b0+h1", parts, tag="k", progress=None, write=False)
    assert len(seen) == n1 + 15


def test_hint_run_puts_the_rb_route_in_every_prompt(dev):
    seen = capture()
    out = H.run_experiment("b1+h1", dev[:3], tag="h", progress=None, write=False)
    assert len(seen) == 3 and all(len(s["messages"]) == 2 for s in seen)
    for s, part in zip(seen, dev[:3]):
        assert json.dumps(P.rb_plan(part["features"], part.get("machine_preference", "auto"))) in s["messages"][1]["content"]
    assert out["summary"]["planner_prompt"]["hint_share"] == 1.0
    assert out["run_info"]["hint"] is True and out["run_info"]["pool"] is None


def test_fs4x_run_never_shows_a_part_with_the_same_feature_set(dev):
    capture()
    out = H.run_experiment("b1+fs4x", dev[:12], tag="x", progress=None, write=False)
    by_id = {p["part_id"]: p for p in dev}
    for rec in out["records"]:
        own = set(rec["part"]["features"])
        for pid in rec["candidates"][0]["meta"]["prompt_context"]["examples"]:
            assert set(by_id[pid]["features"]) != own
    assert out["run_info"]["few_shot_leave_out"] == "features"


def test_repair_stages_work_with_planner_options(dev):
    capture()
    out = H.run_experiment("n3_r1_s3+fs2", dev[:3], tag="r", progress=None, write=False)
    assert out["summary"]["M2"]["final_strict_valid"] == 1.0 and out["run_info"]["selection"] == "s3"
    assert out["records"][0]["system"] == "n3_r1_s3+fs2"


def test_r2_reprompts_do_not_carry_examples_or_hint(dev):
    seen = []

    def transport(model, messages, params):
        seen.append(messages)
        feats = _features_of([m for m in messages if m["role"] == "user"][-2:-1] or messages)
        bad = ["Drilling", "Facing", "Inspection"]
        good = V.clean_steps(rb.generate_valid_routes(feats, max_routes=1)[0]["steps"])
        return {"text": json.dumps(bad if params.get("temperature") != 0.0 else good), "prompt_tokens": 1,
                "completion_tokens": 1, "finish_reason": "stop"}
    llm_client.set_transport(transport)
    H.run_experiment("b1_r2+fs2+h1", dev[:2], tag="rr", progress=None, write=False)
    sample_calls = [m for m in seen if m[-1]["content"].startswith("Generate") or "rule-based" in m[-1]["content"]]
    repair_calls = [m for m in seen if m[-1]["content"].startswith("A verifier rejected")]
    assert sample_calls and repair_calls
    assert all(len(m) == 6 for m in sample_calls) and all(len(m) == 4 for m in repair_calls)
    assert all("rule-based planner" not in m[1]["content"] for m in repair_calls)


def test_planner_replacement_cannot_be_combined_with_options(dev):
    def fake(*a, **k):
        raise AssertionError("must not be called")
    with pytest.raises(H.HarnessError, match="planner_factory"):
        H.run_experiment("b0+fs2", dev[:1], planner=fake, progress=None, write=False)
    with pytest.raises(H.HarnessError, match="planner_factory"):
        H.run_experiment("b0", dev[:1], planner=fake, planner_factory=lambda part: fake, progress=None, write=False)


def test_custom_factory_and_pool_object(dev, pool):
    calls = []

    def factory(part):
        def planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
            calls.append((part["part_id"], sample_id))
            steps = V.clean_steps(rb.generate_valid_routes(features, max_routes=1)[0]["steps"])
            return {"success": True, "steps": steps, "raw_response": json.dumps(steps), "model": "fake",
                    "meta": {"cached": True, "finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.0}}
        return planner
    H.run_experiment("b1", dev[:3], planner_factory=factory, progress=None, write=False)
    assert [c[0] for c in calls] == [p["part_id"] for p in dev[:3]]
    capture()
    out = H.run_experiment("b1+fs2", dev[:2], pool=pool, progress=None, write=False)
    assert out["run_info"]["pool"] == ["dev_v2", 200]
    small = P.Pool(small_pool().examples, name="tiny")
    out = H.run_experiment("b1+fs4", dev[:2], pool=small, progress=None, write=False)
    assert out["summary"]["planner_prompt"]["examples_per_call"] <= 3 and out["run_info"]["pool"] == ["tiny", 6]


def test_sealed_test_set_is_refused_as_pool(dev):
    capture()
    with pytest.raises(H.HarnessError, match="sealed"):
        H.run_experiment("b0+fs2", dev[:1], pool="test_v2", progress=None, write=False)
    H.run_experiment("b0+h1", dev[:1], pool="test_v2", progress=None, write=False)      # no examples used, so no pool is read


def test_guards_still_stop_a_few_shot_run(dev):
    with pytest.raises(H.HarnessError, match="GROQ_API_KEY"):
        H.run_experiment("b0+fs2", dev[:2], tag="nokey", progress=None, write=False)
    llm_client.reset_transport()


def test_replay_mode_cache_miss_stops_a_few_shot_run(dev, monkeypatch):
    capture()
    H.run_experiment("b0", dev[:2], tag="rp", progress=None, write=False)
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    H.run_experiment("b0", dev[:2], tag="rp", progress=None, write=False)              # filled: works offline
    with pytest.raises(H.HarnessError, match="not in cache"):
        H.run_experiment("b0+fs2", dev[:2], tag="rp", progress=None, write=False)      # new prompts: not in the cache


def test_summary_of_a_plain_run_says_v1_and_prints_no_prompt_line(dev):
    capture()
    out = H.run_experiment("b0", dev[:2], tag="p", progress=None, write=False)
    assert out["summary"]["planner_prompt"]["options"] == "v1"
    assert "planner prompt" not in SM.format_summary("b0", out["summary"])


def test_compare_table_shows_the_new_rows_and_strategy_names_with_options(dev, tmp_path):
    capture()
    for name in ("b0_r1", "b0_r1+fs2"):
        H.run_experiment(name, dev[:3], tag="c", out_dir=str(tmp_path), progress=None)
    text = compare.compare([str(tmp_path / "c_b0_r1"), str(tmp_path / "c_b0_r1+fs2")])
    assert "retrieved examples / call" in text and "prompt tokens / part" in text and "b0_r1+fs2" in text
    recs = H.load_records(str(tmp_path / "c_b0_r1+fs2"))
    assert compare._strategy(recs) == "R1"


# ---- command line ----------------------------------------------------------------------------------------------------
def _cli(*args, tmp_path):
    env = dict(os.environ, GROQ_API_KEY="", PYTHONIOENCODING="utf-8", CAPP_CACHE_DIR=str(tmp_path / "c"))
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_eval.py"), *args, "--out", str(tmp_path)],
                          capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)


def test_cli_rejects_bad_options_and_a_sealed_pool(tmp_path):
    res = _cli("--system", "b0+fs3", tmp_path=tmp_path)
    assert res.returncode == 2 and "fs3" in res.stderr
    res = _cli("--system", "rb+fs2", tmp_path=tmp_path)
    assert res.returncode == 2 and "rb" in res.stderr
    res = _cli("--system", "b0+fs2", "--pool", "test_v2", tmp_path=tmp_path)
    assert res.returncode == 2 and "sealed" in res.stderr


def test_cli_few_shot_without_key_stops_with_a_clear_message(tmp_path):
    res = _cli("--system", "b0+fs2", "--limit", "2", "--tag", "nokey", tmp_path=tmp_path)
    assert res.returncode == 2 and "STOPPED" in res.stderr and "GROQ_API_KEY" in res.stderr
