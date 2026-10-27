import json
import os
import subprocess
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import dataset as D          # noqa: E402
from capp import harness as H          # noqa: E402
from capp import llm_client            # noqa: E402
from capp import summary as SM         # noqa: E402
import route_builder as rb             # noqa: E402
from capp import verify as V           # noqa: E402


def _meta(cached=False):
    return {"cached": cached, "finish_reason": "stop", "prompt_tokens": 300, "completion_tokens": 40, "latency_s": 0.4}


def fake_planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
    routes = rb.generate_valid_routes(features, max_routes=15)
    base = V.clean_steps(routes[0]["steps"])
    k = sample_id % 4
    steps = base if k in (0, 3) else (list(reversed(base)) if k == 1 else base[:-1] + ["Deburring", "Inspection"])
    return {"success": True, "steps": steps, "raw_response": json.dumps(steps), "model": "fake", "meta": _meta()}


@pytest.fixture(scope="module")
def parts():
    return D.load_dataset("dev_v2")[:10]


@pytest.fixture(autouse=True)
def clean_client(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CAPP_LLM_MODE", raising=False)
    monkeypatch.delenv("CAPP_RUN_TAG", raising=False)
    llm_client.reset_transport()
    llm_client.reset_call_log()
    yield
    llm_client.reset_transport()


# ---- running and writing ----------------------------------------------------------------------------------
def test_rb_run_writes_all_files(parts, tmp_path):
    out = H.run_experiment("rb", parts, tag="t", out_dir=str(tmp_path), dataset_name="dev_v2", progress=None)
    d = tmp_path / "t_rb"
    for f in ("records.jsonl", "per_part.csv", "per_candidate.csv", "summary.json", "run.json"):
        assert (d / f).exists() and (d / f).stat().st_size > 0, f
    assert out["summary"]["M2"]["final_strict_valid"] == 1.0
    run = json.loads((d / "run.json").read_text())
    assert run["system"] == "rb" and run["dataset"] == "dev_v2" and run["model"] is None and run["n_parts"] == 10


def test_b0_run_tables_and_roundtrip(parts, tmp_path):
    out = H.run_experiment("b0", parts, tag="t", out_dir=str(tmp_path), planner=fake_planner, progress=None)
    d = tmp_path / "t_b0"
    rows = (d / "per_part.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 10
    assert {"final_provenance", "final_strict_valid", "m4_source", "efficiency", "efficiency_v1"} <= set(rows[0].split(","))
    cand_rows = (d / "per_candidate.csv").read_text(encoding="utf-8").splitlines()
    assert len(cand_rows) == 1 + 10 * 5         # header + 5 samples per part (no safety net with this planner)
    recs = H.load_records(str(d))
    assert len(recs) == 10 and recs[0]["part"]["part_id"] == parts[0]["part_id"] and recs[0]["repeat"] == 1
    assert SM.summarize(recs)["M1"] == out["summary"]["M1"]
    run = json.loads((d / "run.json").read_text())
    assert run["model"] and run["temperature"] == 0.3 and run["max_tokens"] == 200


def test_b0_summary_numbers_for_a_known_planner(parts):
    out = H.run_experiment("b0", parts, tag="t", planner=fake_planner, progress=None, write=False)
    s = out["summary"]
    # per part: samples 0..4 -> k = 0,1,2,3,0 : valid, reversed, invented, valid, valid
    assert s["M1"]["llm_calls"] == 50 and s["M1"]["parsed_routes"] == 50
    assert s["M1"]["strict_valid_of_parsed"] == pytest.approx(3 / 5)
    assert s["M1"]["v1_valid_of_parsed"] == pytest.approx(4 / 5)       # the invented-operation route passes v1
    assert s["M1"]["v1_valid_but_strict_invalid"] == 10
    assert set(s["M3"]["kinds_among_invalid"]) >= {"order", "invented"}
    assert s["M2"]["whole_route_replacements_per_part"] == pytest.approx(1.0)
    assert s["M7"]["calls_per_part"] == 5 and s["M7"]["truncated_calls"] == 0
    assert sum(s["M2"]["provenance_share"].values()) == pytest.approx(1.0)


def test_repeats_use_distinct_run_tags_and_restore_the_environment(parts):
    tags = []

    def spy(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        tags.append(os.environ["CAPP_RUN_TAG"])
        return fake_planner(material, features, tolerance, batch_size, sample_id)

    out = H.run_experiment("b1", parts[:3], repeats=3, tag="dev", planner=spy, progress=None, write=False)
    assert sorted(set(tags)) == ["dev-r1", "dev-r2", "dev-r3"] and len(tags) == 9
    assert "CAPP_RUN_TAG" not in os.environ
    assert {r["repeat"] for r in out["records"]} == {1, 2, 3}
    assert out["summary"]["M8"]["repeats"] == 3


def test_rb_ignores_repeats(parts):
    out = H.run_experiment("rb", parts[:3], repeats=5, tag="t", progress=None, write=False)
    assert len(out["records"]) == 3 and out["run_info"]["repeats"] == 1


def count_sleeps(monkeypatch, slept):
    """Record the sleeps of the calling thread only: time.sleep is patched for the whole process, and a background thread left by an
    earlier test (a Streamlit AppTest) may sleep at the same moment (test_twostep's throttle test failed once in a full run for that)."""
    caller = threading.current_thread()
    monkeypatch.setattr(H.time, "sleep", lambda s: slept.append(s) if threading.current_thread() is caller else None)


def test_unknown_system_and_throttle(parts, monkeypatch):
    with pytest.raises(H.HarnessError):
        H.run_experiment("b7", parts, progress=None, write=False)
    slept = []
    count_sleeps(monkeypatch, slept)
    H.run_experiment("b1", parts[:3], planner=fake_planner, throttle=0.5, progress=None, write=False)
    assert slept == [0.5, 0.5, 0.5]

    def cached_planner(*a, **k):
        r = fake_planner(*a, **k)
        r["meta"] = _meta(cached=True)
        return r
    slept.clear()
    H.run_experiment("b1", parts[:3], planner=cached_planner, throttle=0.5, progress=None, write=False)
    assert slept == []                                     # nothing live, nothing to wait for


# ---- guards -----------------------------------------------------------------------------------------------
def test_missing_key_stops_the_run_instead_of_faking_results(parts, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(H.HarnessError, match="GROQ_API_KEY"):
        H.run_experiment("b0", parts[:2], tag="nokey", progress=None, write=False)


def test_replay_mode_cache_miss_stops_the_run(parts, monkeypatch):
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    with pytest.raises(H.HarnessError, match="not in cache"):
        H.run_experiment("b0", parts[:2], tag="empty", progress=None, write=False)


def test_long_api_outage_stops_the_run(parts):
    def down(*a, **k):
        return {"success": False, "steps": [], "raw_response": "API Error: 503"}
    with pytest.raises(H.HarnessError, match="in a row"):
        H.run_experiment("b0", parts, planner=down, max_consecutive_errors=7, progress=None, write=False)


def test_a_few_api_errors_are_recorded_not_fatal(parts):
    state = {"n": 0}

    def flaky(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        state["n"] += 1
        if state["n"] % 4 == 0:
            return {"success": False, "steps": [], "raw_response": "API Error: 429"}
        return fake_planner(material, features, tolerance, batch_size, sample_id)

    out = H.run_experiment("b0", parts[:5], planner=flaky, progress=None, write=False)
    assert out["summary"]["M1"]["parse_rate"] < 1.0 and out["summary"]["M2"]["final_strict_valid"] == 1.0


# ---- through the real planner, llm_client and cache -------------------------------------------------------
def _transport_factory(log):
    seq = [["Facing", "Center Drilling", "Drilling", "Chamfering", "Inspection"], ["Inspection", "Facing"]]

    def transport(model, messages, params):
        log.append(1)
        steps = seq[len(log) % 2]
        return {"text": json.dumps(steps), "prompt_tokens": 310, "completion_tokens": 25, "finish_reason": "stop"}
    return transport


def _hole_chamfer_parts(n):
    base = D.load_dataset("dev_v2")[0]
    out = []
    for i in range(n):
        p = dict(base, part_id=f"X{i}", features=["Hole", "Chamfer"], batch_size=100 + i)   # distinct prompts, distinct cache keys
        out.append(p)
    return out


def test_real_planner_path_caches_and_b1_shares_b0_cache():
    log = []
    llm_client.set_transport(_transport_factory(log))
    ps = _hole_chamfer_parts(3)

    r1 = H.run_experiment("b0", ps, tag="cmp", progress=None, write=False)
    assert len(log) == 15 and r1["summary"]["M7"]["cache_hit_rate"] == 0.0
    assert r1["summary"]["M7"]["prompt_tokens_per_part"] == 5 * 310

    r2 = H.run_experiment("b0", ps, tag="cmp", progress=None, write=False)      # same tag: everything from cache
    assert len(log) == 15 and r2["summary"]["M7"]["cache_hit_rate"] == 1.0
    assert [r["final"]["steps"] for r in r2["records"]] == [r["final"]["steps"] for r in r1["records"]]

    r3 = H.run_experiment("b1", ps, tag="cmp", progress=None, write=False)      # B1 = B0's first sample, no new call
    assert len(log) == 15 and r3["summary"]["M7"]["cache_hit_rate"] == 1.0

    H.run_experiment("b1", ps, tag="other", progress=None, write=False)         # new tag = new draw
    assert len(log) == 18


def test_replay_mode_reproduces_a_finished_run_offline(monkeypatch):
    log = []
    llm_client.set_transport(_transport_factory(log))
    ps = _hole_chamfer_parts(2)
    first = H.run_experiment("b0", ps, tag="rp", progress=None, write=False)
    llm_client.reset_transport()                                               # no transport at all now
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    again = H.run_experiment("b0", ps, tag="rp", progress=None, write=False)
    assert again["summary"]["M1"] == first["summary"]["M1"]
    assert [r["final"]["name"] for r in again["records"]] == [r["final"]["name"] for r in first["records"]]


# ---- summary ----------------------------------------------------------------------------------------------
def test_classes_and_text_report(parts):
    out = H.run_experiment("b0", parts, tag="t", planner=fake_planner, progress=None, write=False)
    s = out["summary"]
    assert set(s["by_class"]) == {"machine_profile", "complexity", "in_distribution"}
    assert sum(v["n_records"] for v in s["by_class"]["complexity"].values()) == 10
    text = SM.format_summary("b0", s)
    for token in ("M1 first pass", "M2 final", "M3 failure kinds", "M4 similarity", "M5 replacement", "M6 quality", "M7 cost"):
        assert token in text


def test_m4_uses_expert_references_when_present(parts):
    p = dict(parts[0])
    routes = rb.generate_valid_routes(p["features"], max_routes=15)
    p["reference_routes"] = [V.clean_steps(routes[0]["steps"])]
    out = H.run_experiment("rb", [p], tag="t", progress=None, write=False)
    m4 = out["summary"]["M4"]
    assert "expert" in m4 and "proxy_rb" not in m4 and m4["expert"]["final"]["exact_rate"] == 1.0


def test_summary_of_nothing_is_safe():
    assert SM.summarize([]) == {"n_records": 0}
    assert "no records" in SM.format_summary("x", {"n_records": 0})


# ---- command line -----------------------------------------------------------------------------------------
def test_cli_rb_run_and_summarize_only(tmp_path):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_eval.py"), "--system", "rb", "--dataset", "dev_v2",
           "--limit", "6", "--tag", "cli", "--out", str(tmp_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
    assert res.returncode == 0, res.stderr
    assert "=== rb: 6 parts" in res.stdout and (tmp_path / "cli_rb" / "summary.json").exists()
    res2 = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_eval.py"), "--summarize-only",
                           str(tmp_path / "cli_rb")], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
    assert res2.returncode == 0 and "=== rb: 6 parts" in res2.stdout


def test_cli_without_key_stops_with_a_clear_message(tmp_path):
    # An empty GROQ_API_KEY wins over a key in the repo's .env (python-dotenv never overrides an existing variable),
    # so this test can never make a real API call, even on a machine that has a .env.
    env = dict(os.environ, GROQ_API_KEY="", PYTHONIOENCODING="utf-8", CAPP_CACHE_DIR=str(tmp_path / "c"))
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_eval.py"), "--system", "b0", "--limit", "2",
           "--out", str(tmp_path), "--tag", "nokey"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
    assert res.returncode == 2 and "STOPPED" in res.stderr and "GROQ_API_KEY" in res.stderr


def test_b0_r1_reuses_b0_cache_and_reports_localized_repair(tmp_path):
    log = []
    llm_client.set_transport(_transport_factory(log))
    ps = _hole_chamfer_parts(3)
    b0 = H.run_experiment("b0", ps, tag="rr", progress=None, write=False)
    n_calls = len(log)
    out = H.run_experiment("b0_r1", ps, tag="rr", out_dir=str(tmp_path), progress=None)       # same tag: all from cache
    assert len(log) == n_calls and out["summary"]["M7"]["cache_hit_rate"] == 1.0
    s0, s1 = b0["summary"], out["summary"]
    assert s1["M1"]["strict_valid_of_parsed"] == s0["M1"]["strict_valid_of_parsed"]            # same raw routes
    assert s0["M5_localized_repair"]["n_repaired"] == 0 and s1["M5_localized_repair"]["n_repaired"] > 0
    assert s1["M2"]["final_strict_valid"] == 1.0
    assert s1["M2"]["localized_repairs_total"] == s1["M5_localized_repair"]["n_repaired"] > 0
    assert s1["M2"]["whole_route_replacements_total"] == 0 and s0["M2"]["whole_route_replacements_total"] > 0
    text = SM.format_summary("b0_r1", s1)
    assert "M5 localized" in text and "R1: n=" in text
    rows = list(__import__("csv").DictReader(open(tmp_path / "rr_b0_r1" / "per_candidate.csv", encoding="utf-8")))
    repaired = [r for r in rows if r["repaired"] == "True"]
    assert repaired and all(r["repair_strategy"] == "r1" and r["edit_distance"] != "" for r in repaired)
    run = json.loads((tmp_path / "rr_b0_r1" / "run.json").read_text())
    assert run["system"] == "b0_r1" and run["model"]


# ---- R2 / R3 through the real planner, reprompt, llm_client and cache -------------------------------------
GOOD = ["Facing", "Center Drilling", "Drilling", "Chamfering", "Inspection"]


def _r2_transport(log):
    """Samples alternate between a valid and an invalid route; every 4-message conversation is a re-prompt."""
    def transport(model, messages, params):
        log.append("repair" if len(messages) == 4 else "sample")
        if len(messages) == 4:
            steps = GOOD
        else:
            steps = [GOOD, ["Inspection", "Facing"]][len([x for x in log if x == "sample"]) % 2]
        return {"text": json.dumps(steps), "prompt_tokens": 500 if len(messages) == 4 else 310,
                "completion_tokens": 25, "finish_reason": "stop"}
    return transport


def test_b0_r2_real_path_counts_repair_calls_and_caches_them(tmp_path):
    log = []
    llm_client.set_transport(_r2_transport(log))
    ps = _hole_chamfer_parts(3)
    out = H.run_experiment("b0_r2", ps, tag="r2", out_dir=str(tmp_path), progress=None)
    s = out["summary"]
    n_repair = log.count("repair")
    assert log.count("sample") == 15 and n_repair > 0
    assert s["M7"]["repair_calls_per_part"] == pytest.approx(n_repair / 3)
    assert s["M7"]["total_calls_per_part"] == pytest.approx((15 + n_repair) / 3)
    assert s["M7"]["repair_prompt_tokens_per_part"] == pytest.approx(500 * n_repair / 3)
    assert s["M7"]["repair_cache_hit_rate"] == 0.0 and s["M2"]["repaired_by_r2_total"] == n_repair
    assert s["M5_localized_repair"]["by_stage"]["r2"]["n"] == n_repair and s["M2"]["final_strict_valid"] == 1.0
    assert "repair calls" in SM.format_summary("b0_r2", s) and "R2: n=" in SM.format_summary("b0_r2", s)

    again = H.run_experiment("b0_r2", ps, tag="r2", progress=None, write=False)        # all from the cache, repairs included
    assert len(log) == 15 + n_repair and again["summary"]["M7"]["repair_cache_hit_rate"] == 1.0
    assert [r["final"]["steps"] for r in again["records"]] == [r["final"]["steps"] for r in out["records"]]

    H.run_experiment("b1_r2", ps, tag="r2", progress=None, write=False)                # B1's sample 0 and its repair are B0's
    assert len(log) == 15 + n_repair

    rows = list(__import__("csv").DictReader(open(tmp_path / "r2_b0_r2" / "per_candidate.csv", encoding="utf-8")))
    assert {r["repair_strategy"] for r in rows if r["repaired"] == "True"} == {"r2"}
    run = json.loads((tmp_path / "r2_b0_r2" / "run.json").read_text())
    assert run["repair_temperature"] == 0.0


def test_b0_r3_real_path_makes_the_same_calls_as_r2_when_r2_succeeds():
    log = []
    llm_client.set_transport(_r2_transport(log))
    ps = _hole_chamfer_parts(3)
    H.run_experiment("b0_r2", ps, tag="x", progress=None, write=False)
    n = len(log)
    out = H.run_experiment("b0_r3", ps, tag="x", progress=None, write=False)
    assert len(log) == n and out["summary"]["M2"]["repaired_by_r1_total"] == 0


def test_repair_planner_guards(parts):
    def good(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        return fake_planner(material, features, tolerance, batch_size, sample_id)

    def down(*a, **k):
        return {"success": False, "steps": [], "raw_response": "API Error: 503"}

    def nokey(*a, **k):
        return {"success": False, "steps": [], "raw_response": "GROQ_API_KEY not configured"}

    def miss(*a, **k):
        return {"success": False, "steps": [], "raw_response": "API Error: not in cache (replay mode): key=abc"}

    with pytest.raises(H.HarnessError, match="in a row"):
        H.run_experiment("b0_r2", parts, planner=good, repair_planner=down, max_consecutive_errors=3, progress=None, write=False)
    with pytest.raises(H.HarnessError, match="GROQ_API_KEY"):
        H.run_experiment("b0_r2", parts, planner=good, repair_planner=nokey, progress=None, write=False)
    with pytest.raises(H.HarnessError, match="replay mode"):
        H.run_experiment("b0_r3", parts, planner=good, repair_planner=miss, progress=None, write=False)


def test_replay_of_a_run_without_cached_repairs_stops(monkeypatch):
    log = []
    llm_client.set_transport(_r2_transport(log))
    ps = _hole_chamfer_parts(2)
    H.run_experiment("b0", ps, tag="rp2", progress=None, write=False)                  # only the samples are cached
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    with pytest.raises(H.HarnessError, match="not in cache"):
        H.run_experiment("b0_r2", ps, tag="rp2", progress=None, write=False)


def test_throttle_also_waits_after_a_live_repair_call(parts, monkeypatch):
    slept = []
    count_sleeps(monkeypatch, slept)

    def cached_planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        r = fake_planner(material, features, tolerance, batch_size, sample_id)
        r["meta"] = _meta(cached=True)
        return r

    def live_repair(material, features, tolerance="0.05mm", batch_size=100, sample_id=0, previous_steps=(), violations=()):
        steps = V.clean_steps(rb.generate_valid_routes(features, max_routes=1)[0]["steps"])
        return {"success": True, "steps": steps, "raw_response": "x", "model": "f", "meta": _meta(cached=False)}

    H.run_experiment("b0_r2", parts[:3], planner=cached_planner, repair_planner=live_repair, throttle=0.5, progress=None, write=False)
    assert slept and set(slept) == {0.5}
