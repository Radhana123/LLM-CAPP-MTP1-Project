import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import archetypes as A       # noqa: E402
from capp import compare as C          # noqa: E402
from capp import dataset as D          # noqa: E402
from capp import harness as H          # noqa: E402
from capp import llm_client            # noqa: E402
from capp import summary as SM         # noqa: E402
from capp import systems as S          # noqa: E402
from capp import verify as V           # noqa: E402
import nsga2 as v1_nsga2               # noqa: E402
import route_builder as rb             # noqa: E402

# a part whose first front has several members with different Efficiency (for most parts the front members are exact ties)
GEAR = next(a for a in A.ARCHETYPES if a["name"] == "Sealed Housing Cover")


def _meta():
    return {"cached": False, "finish_reason": "stop", "prompt_tokens": 300, "completion_tokens": 40, "latency_s": 0.4}


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CAPP_LLM_MODE", raising=False)
    monkeypatch.delenv("CAPP_RUN_TAG", raising=False)
    llm_client.reset_transport()


def gear_part(pid="G"):
    return {"part_id": pid, "material": "Steel", "features": GEAR["features"], "tolerance": "0.02mm", "batch_size": 100,
            "dims": dict(S.DEFAULT_DIMS), "machine_preference": "auto"}


@pytest.fixture(scope="module")
def pool():
    """Distinct valid routes of one part (no markers, like LLM routes), scored; the first front has several members."""
    p = S.part_inputs(gear_part())
    cands, seen = [], set()
    for i, r in enumerate(rb.generate_valid_routes(GEAR["features"], max_routes=15)):
        steps = V.clean_steps(r["steps"])
        if tuple(steps) not in seen:
            seen.add(tuple(steps))
            cands.append(S.Candidate(i, f"R{i}", "raw_llm", steps, llm_ok=True, llm_steps=steps))
    S._finish_candidates(cands, p)
    front = [cands[i] for i in v1_nsga2.non_dominated_sort([S._Ind(c, p, S.DEFAULT_DIMS) for c in cands])[0]]
    assert len(front) >= 3 and len({c.scores["efficiency_score"] for c in front}) >= 2
    return p, cands, front


def sel(pool, selection, **kw):
    p, cands, _ = pool
    return S._select(cands, p, S.DEFAULT_DIMS, selection, **kw)


# ---- S1 = v1, S0 -------------------------------------------------------------------------------------------
def test_s1_is_the_v1_rule_best_efficiency_of_the_first_front(pool):
    _, _, front = pool
    best, names, _ = sel(pool, "s1")
    assert best is max(front, key=lambda c: (c.scores["efficiency_score"], -front.index(c)))
    assert set(names) == {c.name for c in front}


def test_s0_takes_the_first_valid_candidate_in_order(pool):
    p, cands, _ = pool
    bad = S.Candidate(99, "bad", "raw_llm", list(reversed(cands[0].steps)), llm_ok=True)
    S._finish_candidates([bad], p)
    best, _, detail = S._select([bad] + cands, p, S.DEFAULT_DIMS, "s0")
    assert best is cands[0] and "no_valid_candidate" not in detail
    best, _, detail = S._select([bad], p, S.DEFAULT_DIMS, "s0")
    assert best is bad and detail["no_valid_candidate"] is True


# ---- S2 ----------------------------------------------------------------------------------------------------
def test_s2_single_priority_picks_the_best_of_that_objective(pool):
    _, _, front = pool
    for w, key in (((1, 0, 0), "time_min"), ((0, 1, 0), "cost_inr"), ((0, 0, 1), "energy_kwh")):
        best, _, detail = sel(pool, "s2", weights=w)
        assert best.scores[key] == min(c.scores[key] for c in front), key
        assert detail["weights"] == [float(x) for x in w]


def test_s2_equal_weights_match_an_independent_computation(pool):
    _, _, front = pool
    keys = ("time_min", "cost_inr", "energy_kwh")
    lo = {k: min(c.scores[k] for c in front) for k in keys}
    hi = {k: max(c.scores[k] for c in front) for k in keys}
    sc = {c.name: sum((c.scores[k] - lo[k]) / (hi[k] - lo[k]) if hi[k] > lo[k] else 0.0 for k in keys) / 3 for c in front}
    best, _, detail = sel(pool, "s2")
    assert best.name == min(sc, key=lambda n: (sc[n], -next(c.scores["efficiency_score"] for c in front if c.name == n)))
    assert detail["scalarised_score"] == pytest.approx(sc[best.name])


def test_weights_are_normalised_and_validated():
    assert S.normalize_weights((2, 1, 1)) == pytest.approx((0.5, 0.25, 0.25))
    assert S.normalize_weights(None) == pytest.approx((1 / 3,) * 3)
    for bad in ((1, 2), (-1, 1, 1), (0, 0, 0), (1, 1, 1, 1)):
        with pytest.raises(ValueError):
            S.normalize_weights(bad)


# ---- S3 and confidence -------------------------------------------------------------------------------------
def _best_and_a_different_route(pool):
    """The efficiency-best front member, and a front member with lower Efficiency and a different operation order."""
    _, _, front = pool
    best = sel(pool, "s1")[0]
    other = next(c for c in front if c.scores["efficiency_score"] < best.scores["efficiency_score"]
                 and S.route_distance(c.steps, best.steps) > 0)
    return best, other


def test_s3_picks_the_route_the_other_samples_agree_with_not_the_most_efficient(pool):
    best, target = _best_and_a_different_route(pool)
    original = target.sample_id
    target.sample_id = 1000
    try:                       # three samples produced `target`'s route (three distinct sample ids), one produced the best one
        routes = [(1000, target.steps), (1002, target.steps), (1003, target.steps), (1001, best.steps)]
        chosen, _, detail = sel(pool, "s3", sample_routes=routes)
    finally:
        target.sample_id = original
    assert chosen is target and chosen is not best
    assert detail["selected_mean_distance"] == pytest.approx(S.route_distance(target.steps, best.steps) / 3)


def test_s3_without_other_samples_falls_back_to_the_most_efficient(pool):
    assert sel(pool, "s3", sample_routes=[])[0] is sel(pool, "s1")[0]


def test_route_distance_and_agreement():
    a = ["Facing", "Center Drilling", "Drilling", "Reaming", "Inspection"]
    rev = list(reversed(a))
    assert S.route_distance(a, a) == 0.0 and S.route_distance(a, rev) == 1.0
    one_swap = ["Facing", "Drilling", "Center Drilling", "Reaming", "Inspection"]
    assert S.route_distance(a, one_swap) == pytest.approx(1 / 10)                 # 1 discordant pair of C(5,2)
    # one extra operation: the 5 pairs that involve it cannot be compared and count 0.5 each, of C(6,2) = 15 pairs
    assert S.route_distance(a, a[:4] + ["Chamfering"] + a[4:]) == pytest.approx(5 * 0.5 / 15)
    # a different choice of operations is disagreement even when the common operations are in the same order
    alt = ["Facing", "Boring", "Inspection"]
    assert S.route_distance(a, alt) > 0 and S.route_distance(a, alt) == S.route_distance(alt, a)
    assert S.route_distance(["Facing"], ["Inspection"]) == 0.5 and S.route_distance(["Facing"], ["Facing"]) == 0.0
    assert S.route_distance([], []) == 0.0 and S.route_distance([], ["Facing"]) == 1.0
    assert S.agreement([a]) is None and S.agreement([]) is None
    assert S.agreement([a, a, a]) == 1.0 and S.agreement([a, rev]) == 0.0
    assert S.agreement([a, a, rev]) == pytest.approx(1 / 3)
    assert S.agreement([a + [V.MARKER], a]) == 1.0                              # markers never matter
    assert S.agreement([a, alt]) < 1.0                                          # the tau on common operations alone says 1.0


# ---- through run_llm_first ---------------------------------------------------------------------------------
def planner_from(routes):
    def planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        steps = list(routes[sample_id % len(routes)])
        return {"success": True, "steps": steps, "raw_response": str(steps), "model": "fake", "meta": _meta()}
    return planner


def test_identical_samples_have_full_confidence_and_are_not_flagged(pool):
    _, cands, _ = pool
    rec = S.run_b0(gear_part(), planner=planner_from([cands[0].steps]))
    assert rec.info["confidence"] == 1.0 and rec.info["flagged"] is False and rec.info["confidence_routes"] == 5
    assert rec.info["selection"] == "s1"


def test_confidence_counts_duplicates_and_matches_agreement_of_the_final_sample_routes(pool):
    _, cands, front = pool
    A_, B_ = front[0].steps, front[1].steps
    rec = S.run_llm_first(gear_part(), n_samples=5, planner=planner_from([A_, A_, A_, B_, B_]), system="x")
    routes = [c.steps for c in rec.candidates if c.llm_ok and c.steps]
    assert len(routes) == 5 and rec.info["deduped"] == 3                      # dedupe shrinks the pool, not the confidence
    assert rec.info["confidence"] == pytest.approx(S.agreement(routes))
    assert rec.info["population"] == 2


def test_s3_end_to_end_prefers_the_majority_route(pool):
    best, other = _best_and_a_different_route(pool)
    planner = planner_from([other.steps, other.steps, other.steps, best.steps, best.steps])
    r1 = S.run_llm_first(gear_part(), planner=planner, system="x", selection="s1")
    r3 = S.run_llm_first(gear_part(), planner=planner, system="x", selection="s3")
    assert V.clean_steps(r1.final["steps"]) == best.steps and V.clean_steps(r3.final["steps"]) == other.steps
    assert r3.info["selection"] == "s3" and r3.info["selected_mean_distance"] < S.route_distance(best.steps, other.steps)


def test_low_confidence_is_flagged(monkeypatch, pool):
    _, _, front = pool
    monkeypatch.setattr(S, "CONFIDENCE_FLAG", 1.01)
    rec = S.run_b0(gear_part(), planner=planner_from([front[0].steps]))
    assert rec.info["flagged"] is True


def test_single_sample_has_no_confidence(pool):
    rec = S.run_b1(gear_part(), planner=planner_from([pool[1][0].steps]))
    assert rec.info["confidence"] is None and rec.info["flagged"] is False


def test_failed_samples_are_left_out_of_the_confidence(pool):
    ok = pool[1][0].steps

    def planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
        if sample_id % 2:
            return {"success": False, "steps": [], "raw_response": "API Error: x"}
        return {"success": True, "steps": list(ok), "raw_response": "x", "model": "f", "meta": _meta()}
    rec = S.run_b0(gear_part(), planner=planner)
    assert rec.info["confidence_routes"] == 3 and rec.info["confidence"] == 1.0


def test_unknown_selection_and_bad_weights_are_rejected_early(pool):
    with pytest.raises(ValueError):
        S.run_llm_first(gear_part(), planner=planner_from([pool[1][0].steps]), selection="s9")
    with pytest.raises(ValueError):
        S.run_llm_first(gear_part(), planner=planner_from([pool[1][0].steps]), selection="s2", weights=(0, 0, 0))


def test_s1_default_does_not_change_b0(pool):
    _, cands, front = pool
    planner = planner_from([front[0].steps, front[1].steps, front[2].steps, cands[-1].steps, cands[0].steps])
    a = S.run_b0(gear_part(), planner=planner)
    b = S.run_llm_first(gear_part(), n_samples=5, planner=planner, system="b0", selection="s1")
    assert a.final == b.final and a.front == b.front


# ---- system names ------------------------------------------------------------------------------------------
def test_system_name_parsing():
    assert S.parse_system("rb") is None
    assert S.parse_system("b0") == (5, "r0", "s1") and S.parse_system("b1") == (1, "r0", "s1")
    assert S.parse_system("b0_r2") == (5, "r2", "s1") and S.parse_system("b1_r3") == (1, "r3", "s1")
    assert S.parse_system("n8_r1_s3") == (8, "r1", "s3") and S.parse_system("n5_r0_s1") == S.parse_system("b0")
    for bad in ("n0_r0_s1", "n21_r0_s1", "n5_r4_s1", "n5_r0_s4", "x", "b2", "b0_r4", "n5_r1", ""):
        with pytest.raises(ValueError):
            S.parse_system(bad)
    assert S.is_llm("n3_r1_s0") and not S.is_llm("rb")
    assert S.reprompts("n5_r2_s1") and S.reprompts("b0_r3") and not S.reprompts("n5_r1_s3") and not S.reprompts("b0")
    assert S.resolve("b0") is S.SYSTEMS["b0"] and S.resolve("rb") is S.run_rb


def test_generic_names_use_the_requested_sample_count_and_equal_the_legacy_names(pool):
    calls = []
    base = planner_from([pool[1][0].steps, pool[1][1].steps])

    def counting(*a, **k):
        calls.append(k.get("sample_id"))
        return base(*a, **k)
    rec = S.resolve("n8_r1_s3")(gear_part(), planner=counting)
    assert calls == list(range(8)) and rec.system == "n8_r1_s3" and rec.info["selection"] == "s3"
    legacy_rec, generic_rec = S.resolve("b0_r1")(gear_part(), planner=base), S.resolve("n5_r1_s1")(gear_part(), planner=base)
    assert legacy_rec.final == generic_rec.final and legacy_rec.info["attempts"] == generic_rec.info["attempts"] == 5
    assert S.resolve("n3_r0_s1")(gear_part(), planner=base).info["attempts"] == 3


# ---- harness, summary, compare, CLI ------------------------------------------------------------------------
def test_harness_passes_weights_and_records_the_configuration(tmp_path, pool):
    ps = [gear_part(f"G{i}") for i in range(2)]
    planner = planner_from([c.steps for c in pool[2]] + [pool[1][-1].steps])
    out = H.run_experiment("n5_r1_s2", ps, tag="w", out_dir=str(tmp_path), planner=planner, weights=(1, 0, 0), progress=None)
    run = out["run_info"]
    assert run["selection"] == "s2" and run["n_samples"] == 5 and run["repair"] == "r1" and run["weights"] == [1.0, 0.0, 0.0]
    assert all(r["info"]["weights"] == [1.0, 0.0, 0.0] for r in out["records"])
    assert (tmp_path / "w_n5_r1_s2" / "run.json").exists()
    default = H.run_experiment("n5_r1_s2", ps, tag="w", planner=planner, progress=None, write=False)
    assert default["run_info"]["weights"] == pytest.approx([1 / 3] * 3)


def test_harness_rejects_bad_names_and_weights(pool):
    with pytest.raises(H.HarnessError):
        H.run_experiment("n9_r9_s9", [gear_part()], progress=None, write=False)
    with pytest.raises(H.HarnessError, match="weights"):
        H.run_experiment("n5_r0_s2", [gear_part()], planner=planner_from([pool[1][0].steps]), weights=(0, 0, 0),
                         progress=None, write=False)


def test_confidence_in_summary_csv_and_compare(tmp_path, pool):
    ps = [gear_part(f"G{i}") for i in range(4)]
    _, _, front = pool
    mixed = planner_from([front[0].steps, front[1].steps, front[2].steps, front[0].steps, front[1].steps])
    same = planner_from([front[0].steps])
    out = str(tmp_path / "r")
    H.run_experiment("rb", ps, tag="c", out_dir=out, progress=None)
    H.run_experiment("n5_r0_s3", ps, tag="c", out_dir=out, planner=mixed, progress=None)
    s_same = H.run_experiment("n5_r0_s1", ps, tag="c", out_dir=out, planner=same, progress=None)["summary"]
    s_mixed = SM.summarize(H.load_records(os.path.join(out, "c_n5_r0_s3")))
    cf = s_mixed["confidence"]
    assert cf["n"] == 4 and 0 < cf["mean"] < 1 and cf["selected_mean_distance_mean"] is not None
    assert s_same["confidence"]["mean"] == 1.0 and s_same["confidence"]["flagged_share"] == 0.0
    assert s_mixed["selection"] == ["s3"]
    text = SM.format_summary("n5_r0_s3", s_mixed)
    assert "confidence      : selection s3" in text
    assert "confidence      :" not in SM.format_summary("rb", SM.summarize(H.load_records(os.path.join(out, "c_rb"))))
    import csv
    rows = list(csv.DictReader(open(os.path.join(out, "c_n5_r0_s3", "per_part.csv"), encoding="utf-8")))
    assert {"selection", "confidence", "flagged"} <= set(rows[0]) and rows[0]["selection"] == "s3"
    table = C.compare([os.path.join(out, "c_rb"), os.path.join(out, "c_n5_r0_s3")])
    assert "plan confidence mean" in table and "plans flagged for review" in table


def test_cli_validates_system_name_and_weights(tmp_path):
    env = dict(os.environ, GROQ_API_KEY="", PYTHONIOENCODING="utf-8", CAPP_CACHE_DIR=str(tmp_path / "c"))
    run = lambda *args: subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_eval.py"), *args], capture_output=True,
                                       text=True, cwd=ROOT, env=env, timeout=120)
    bad = run("--system", "n99_r1_s1")
    assert bad.returncode == 2 and "unknown system" in bad.stderr
    bad = run("--system", "n5_r0_s2", "--weights", "1,2")
    assert bad.returncode == 2 and "--weights" in bad.stderr
    ok = run("--system", "rb", "--limit", "3", "--out", str(tmp_path), "--tag", "t")
    assert ok.returncode == 0
    nokey = run("--system", "n5_r1_s3", "--limit", "2", "--out", str(tmp_path), "--tag", "nk")
    assert nokey.returncode == 2 and "GROQ_API_KEY" in nokey.stderr                 # generic names reach the real guarded planner
