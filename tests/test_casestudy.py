"""Step 23: the failure case-study generator (capp/casestudy.py, scripts/case_studies.py).

Offline. The runs are made by the real harness with a deterministic fake planner that makes each kind of mistake (a dropped operation, two
swapped, an invented operation, a duplicate), a different one for each sample of a part, so every failure type has cases. What the module
says is checked against the records read independently in the test, and on hand-built records for the corners.
"""
import copy
import csv
import json
import os
import random
import re
import shutil
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from capp import casestudy as C, dataset, harness, metrics as M, summary as S, verify as V      # noqa: E402
import case_studies                                                                            # noqa: E402
import route_builder as rb                                                                     # noqa: E402

N_PARTS = 12
REPEATS = 2
MODES = ("ok", "drop", "swap", "invent", "duplicate")


def planner_factory_for(salt):
    """A planner per part. Which mistake sample s makes depends on the part and the repeat, so the first failing sample differs between parts."""
    seen = {}

    def factory(part, role=None):
        pid = part["part_id"]
        seen[pid] = seen.get(pid, 0) + 1
        repeat = seen[pid]
        shift = sum(ord(ch) for ch in pid + salt) + repeat

        def planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0, **kw):
            routes = rb.generate_valid_routes(features, max_routes=15)
            if not routes:
                return {"success": False, "steps": [], "raw_response": ""}
            steps = list(V.clean_steps(routes[(sample_id + len(features)) % len(routes)]["steps"]))
            mode = MODES[(sample_id + shift) % len(MODES)]
            r = random.Random(f"{salt}-{pid}-{repeat}-{sample_id}")
            if mode == "drop" and len(steps) > 3:
                steps.pop(r.randrange(1, len(steps) - 1))
            elif mode == "swap" and len(steps) > 3:
                i = r.randrange(len(steps) - 1)
                steps[i], steps[i + 1] = steps[i + 1], steps[i]
            elif mode == "invent":
                steps.insert(len(steps) // 2, "Warp Drilling")
            elif mode == "duplicate":
                i = r.randrange(1, len(steps) - 1)
                steps.insert(i, steps[i])
            return {"success": True, "steps": steps, "raw_response": "", "model": "fake",
                    "meta": {"cached": True, "finish_reason": "stop", "prompt_tokens": 100, "completion_tokens": 20, "latency_s": 0.1}}
        return planner
    return factory


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("runs"))
    parts = dataset.load_dataset("dev_v2")[:N_PARTS]
    harness.run_experiment("n5_r1_s3+fz2", parts, repeats=REPEATS, tag="cs", out_dir=out, dataset_name="dev_v2", progress=None,
                           planner_factory=planner_factory_for("A"), analysis=True)
    harness.run_experiment("b0_r1", parts, repeats=REPEATS, tag="cs_noan", out_dir=out, dataset_name="dev_v2", progress=None,
                           planner_factory=planner_factory_for("B"), analysis=False)
    harness.run_experiment("rb", parts, repeats=1, tag="cs", out_dir=out, dataset_name="dev_v2", progress=None)
    return out


def folder(runs_dir, system, tag):
    return os.path.join(runs_dir, f"{tag}_{system}")


@pytest.fixture(scope="module")
def run_a(runs_dir):
    return C.load_run(folder(runs_dir, "n5_r1_s3+fz2", "cs"))


@pytest.fixture(scope="module")
def run_b(runs_dir):
    return C.load_run(folder(runs_dir, "b0_r1", "cs_noan"))


@pytest.fixture(scope="module")
def run_rb(runs_dir):
    return C.load_run(folder(runs_dir, "rb", "cs"))


@pytest.fixture(scope="module")
def res_a(run_a):
    return C.analyse([run_a], per_type=4)


def pnum(part_id):
    return int(part_id.split("_")[1])


def raw_routes(records):
    for rec in records:
        for c in rec["candidates"]:
            if c["sample_id"] >= 0 and c["llm_ok"] and c["llm_steps"]:
                yield rec, c


def expected_cases(records, t, k):
    """[(part id, repeat, sample id)]: the rule, written again here from the records."""
    first = {}
    for rec, c in raw_routes(records):
        hit = t in c["first_pass"]["kinds"] or (
            t == "infeasible" and any(v["kind"] == "infeasible" for v in (c.get("raw_feasibility") or {}).get("violations", [])))
        if hit:
            key = (rec["repeat"], c["sample_id"])
            if rec["part_id"] not in first or key < first[rec["part_id"]]:
                first[rec["part_id"]] = key
    return [(p, *first[p]) for p in sorted(first, key=pnum)][:k]


def got(res, run_label, t):
    return [(c["part"]["part_id"], c["repeat"], c["sample_id"]) for c in res["cases"] if c["run"] == run_label and c["type"] == t]


# ---------------------------------------------------------------- small pieces
def test_natural_order_of_part_ids():
    ids = ["DEV_010", "DEV_002", "DEV_001", "DEV_100", "DEV_009"]
    assert sorted(ids, key=C.natural_key) == ["DEV_001", "DEV_002", "DEV_009", "DEV_010", "DEV_100"]
    assert sorted(["B_2", "A_10", "A_9"], key=C.natural_key) == ["A_9", "A_10", "B_2"]


@pytest.mark.parametrize("text,expect", [
    ("DEV_016=why", {"part_id": "DEV_016", "repeat": None, "sample_id": None, "note": "why"}),
    ("DEV_016:2=why", {"part_id": "DEV_016", "repeat": 2, "sample_id": None, "note": "why"}),
    ("DEV_016:2:3=why = not = split", {"part_id": "DEV_016", "repeat": 2, "sample_id": 3, "note": "why = not = split"}),
    ("DEV_016::3=why", {"part_id": "DEV_016", "repeat": None, "sample_id": 3, "note": "why"}),
    (" DEV_016 : 1 =  spaced out  ", {"part_id": "DEV_016", "repeat": 1, "sample_id": None, "note": "spaced out"}),
])
def test_parse_instructive(text, expect):
    assert C.parse_instructive(text) == expect


@pytest.mark.parametrize("text", ["DEV_016", "DEV_016=", "DEV_016=   ", "=why", "DEV_016:1:2:3=why", "DEV_016:x=why", "DEV_016:1:y=why"])
def test_parse_instructive_rejects(text):
    with pytest.raises(C.CaseError):
        C.parse_instructive(text)


def test_route_diff_is_computed_from_the_two_routes():
    llm = ["Facing", "Drilling", "Boring", "Inspection"]
    d = C.route_diff(llm, ["Facing", "Center Drilling", "Drilling", "Inspection"])
    assert d["removed"] == ["Boring"] and d["inserted"] == ["Center Drilling"] and d["kept_in_order"] == 3
    assert d["n_llm_ops"] == 4 and d["n_fixed_ops"] == 4 and d["survived_share"] == 0.75 and d["edit_distance"] == 2 and not d["identical"]
    assert not d["order_changed"]
    swapped = C.route_diff(["Facing", "Boring", "Drilling", "Inspection"], ["Facing", "Drilling", "Boring", "Inspection"])
    assert swapped["order_changed"] and swapped["removed"] == [] and swapped["inserted"] == []
    same = C.route_diff(["Facing", V.MARKER, "Inspection"], ["Facing", "Inspection"])
    assert same["identical"] and same["survived_share"] == 1.0                  # the changeover marker is not an operation


def cand(**kw):
    c = {"sample_id": 0, "name": "Route_LLM_1", "provenance": "raw_llm", "steps": ["Facing", "Inspection"], "llm_ok": True,
         "llm_steps": ["Facing", "Inspection"], "replaced": False, "repaired": False, "repair": None, "in_pool": True,
         "first_pass": {"valid": True, "kinds": [], "violations": []}}
    c.update(kw)
    return c


@pytest.mark.parametrize("c,expect", [
    (cand(replaced=True, provenance="route_builder", repair={"tried": ["r2", "r1"], "stage": None}), "replaced by the Route Builder's route (R0) after R2, R1 did not give"),
    (cand(replaced=True, provenance="route_builder", repair={"tried": []}), "replaced by the Route Builder's route (R0)"),
    (cand(repaired=True, provenance="repaired_llm", repair={"stage": "r1"}), "repaired by R1"),
    (cand(repaired=True, provenance="repaired_llm", repair={"stage": "r2"}), "repaired by R2"),
    (cand(steps=[]), "no route"),
    (cand(repair={"stage": None}), "kept as sampled"),
    (cand(), "valid as sampled"),
    (cand(first_pass={"valid": False, "kinds": ["order"], "violations": []}), "not repaired"),
])
def test_repair_outcome_words(c, expect):
    assert expect in C.repair_outcome(c)


def test_repair_facts_state_what_each_stage_did():
    rp = {"outcome": "x", "stage": "r1", "tried": ["r2", "r1"], "fallback": None,
          "r1": {"ok": True, "removed_unknown": ["Warp Drilling"], "removed_duplicates": ["Facing"], "inserted": ["Drilling", "Reaming"],
                 "fz_inserted": ["Reaming"], "fz_edges": [["Drilling", "Boring"]], "discordant_pairs": 2, "unrepairable": ["CAP-X"], "flagged": ["TOL-SPEC"]},
          "r2": {"ok": False, "reason": "answer still violates: HOLE-EXISTS", "steps": [], "kinds": []}}
    d = C.route_diff(["Facing", "Boring", "Inspection"], ["Facing", "Drilling", "Boring", "Inspection"])
    text = "\n".join(C._repair_facts(rp, d, ["Facing", "Boring", "Inspection"], ["Facing", "Drilling", "Boring", "Inspection"]))
    for piece in ("removed invented operation(s) Warp Drilling", "removed duplicate(s) Facing", "inserted Drilling, Reaming (for feasibility: Reaming)",
                  "added the ordering Drilling before Boring", "resolved 2 operation pair(s)", "cannot repair: CAP-X", "Flagged for the engineer, not changed: TOL-SPEC",
                  "R2 re-prompt: the answer was not accepted (answer still violates: HOLE-EXISTS)", "3 of the LLM's 3 operations"):
        assert piece in text, piece
    failed = "\n".join(C._repair_facts({"outcome": "x", "stage": None, "r1": {"ok": False, "reason": "no order found"}, "r2": None}, None, [], []))
    assert "R1 could not produce a valid route: no order found" in failed
    assert "changed nothing" in "\n".join(C._repair_facts({"outcome": "x", "stage": "r1", "r1": {"ok": True}, "r2": None}, None, [], []))
    accepted = "\n".join(C._repair_facts({"outcome": "x", "stage": "r2", "r1": None, "r2": {"ok": True, "reason": ""}}, None, [], []))
    assert "the answer was accepted" in accepted


# ---------------------------------------------------------------- reading the runs
def test_load_run_checks_what_it_reads(tmp_path):
    with pytest.raises(C.CaseError, match="no records.jsonl"):
        C.load_run(str(tmp_path))
    (tmp_path / "records.jsonl").write_text("")
    with pytest.raises(C.CaseError, match="empty"):
        C.load_run(str(tmp_path))
    (tmp_path / "records.jsonl").write_text('{"part_id": "X"}\n')
    with pytest.raises(C.CaseError, match="not a run of this harness"):
        C.load_run(str(tmp_path))
    (tmp_path / "records.jsonl").write_text("not json\n")
    with pytest.raises(C.CaseError, match="cannot be read"):
        C.load_run(str(tmp_path))


def test_load_run_label_and_run_info(runs_dir):
    run = C.load_run(folder(runs_dir, "n5_r1_s3+fz2", "cs") + os.sep)
    assert run["label"] == "cs_n5_r1_s3+fz2" and run["run_info"]["system"] == "n5_r1_s3+fz2" and len(run["records"]) == N_PARTS * REPEATS


# ---------------------------------------------------------------- the rule
@pytest.mark.parametrize("t", C.TYPES)
def test_cases_are_the_first_parts_in_id_order_with_their_first_failing_route(run_a, res_a, t):
    want = expected_cases(run_a["records"], t, 4)
    assert got(res_a, run_a["label"], t) == want
    assert [c["rank"] for c in res_a["cases"] if c["type"] == t] == list(range(1, len(want) + 1))


def test_every_type_the_planner_makes_has_cases(run_a, res_a):
    for t in ("order", "missing", "invented", "duplicate"):
        assert len(got(res_a, run_a["label"], t)) == 4, t                       # the fake planner makes each of them in many parts
    assert got(res_a, run_a["label"], "machine_grouping") == []                 # it never does: no case, no invented one


def test_one_case_per_part_and_type(res_a):
    seen = set()
    for c in res_a["cases"]:
        key = (c["run"], c["type"], c["part"]["part_id"])
        assert key not in seen
        seen.add(key)


def test_a_case_really_has_the_failure_it_is_listed_under(res_a):
    for c in res_a["cases"]:
        if c["type"] in C.TYPES:
            assert c["type"] in c["types"]
            if c["type"] == "infeasible":
                assert any(v["kind"] == "infeasible" for v in c["feasibility"])
            else:
                assert c["type"] in {v["kind"] for v in c["violations"]}


def test_per_type_limits_the_number_and_keeps_the_order(run_a):
    r1 = C.analyse([run_a], types=["order"], per_type=1)
    r4 = C.analyse([run_a], types=["order"], per_type=4)
    assert [c["id"] for c in r1["cases"]] == [r4["cases"][0]["id"]]


@pytest.mark.parametrize("k", [0, -1, 11, 2.5, "3", True, None])
def test_per_type_must_be_a_whole_number_from_1_to_10(run_a, k):
    with pytest.raises(C.CaseError, match="--per-type"):
        C.analyse([run_a], per_type=k)


@pytest.mark.parametrize("types", [["nonsense"], ["order", "order"], [], ["order", "Order"]])
def test_types_are_validated(run_a, types):
    with pytest.raises(C.CaseError):
        C.analyse([run_a], types=types)


def test_no_run_is_an_error():
    with pytest.raises(C.CaseError, match="no run"):
        C.analyse([])


def test_the_order_of_the_records_does_not_matter(run_a):
    shuffled = dict(run_a, records=list(reversed(run_a["records"])))
    a = C.analyse([run_a], per_type=3)
    b = C.analyse([shuffled], per_type=3)
    assert [c["id"] for c in a["cases"]] == [c["id"] for c in b["cases"]]


def test_a_route_that_fails_two_ways_is_a_candidate_for_both(run_a):
    recs = copy.deepcopy(run_a["records"])
    rec = next(r for r in recs if r["part_id"] == "DEV_001" and r["repeat"] == 1)
    c = next(c for c in rec["candidates"] if c["sample_id"] == 0)
    c["first_pass"] = {"valid": False, "kinds": ["order", "missing"],
                       "violations": [{"kind": "order", "message": "m1", "ops": [], "feature": "", "rule_id": "ORD-X", "positions": []},
                                      {"kind": "missing", "message": "m2", "ops": [], "feature": "", "rule_id": "COV:X", "positions": []}]}
    for other in recs:                                         # nothing else fails, so these are the only routes of the two types
        for o in other["candidates"]:
            if o is not c and o["sample_id"] >= 0:
                o["first_pass"] = {"valid": True, "kinds": [], "violations": []}
                o["raw_feasibility"] = None
    res = C.analyse([dict(run_a, records=recs)], types=["order", "missing"], per_type=3)
    assert got(res, run_a["label"], "order") == got(res, run_a["label"], "missing") == [("DEV_001", 1, 0)]
    assert C.failure_types(c) == ["order", "missing"]
    assert any("fails in several ways (order, missing)" in f for f in res["cases"][0]["facts"])


def test_soft_kinds_machine_grouping_and_risk(run_a):
    recs = copy.deepcopy(run_a["records"])
    rec = next(r for r in recs if r["part_id"] == "DEV_003" and r["repeat"] == 2)
    c = next(c for c in rec["candidates"] if c["sample_id"] == 1)
    c["first_pass"] = {"valid": True, "kinds": ["machine_grouping"],
                       "violations": [{"kind": "machine_grouping", "message": "4 changeovers, 2 are enough", "ops": [], "feature": "", "rule_id": "GRP", "positions": []}]}
    res = C.analyse([dict(run_a, records=recs)], types=["machine_grouping"], per_type=3)
    assert got(res, run_a["label"], "machine_grouping") == [("DEV_003", 2, 1)]
    v = res["cases"][0]["violations"][0]
    assert v["kind"] == "machine_grouping" and v["hard"] is False
    assert "(soft)" in C.render_markdown(res)


# ---------------------------------------------------------------- the facts
def test_diff_and_survival_match_the_metrics(res_a):
    for c in res_a["cases"]:
        if c["type"] in C.TYPES and c["diff"]:
            assert c["diff"]["survived_share"] == M.survived_share(c["llm_route"], c["candidate_route"])
            assert c["diff"]["edit_distance"] == M.edit_distance(c["llm_route"], c["candidate_route"])
            assert sorted(c["diff"]["inserted"]) == sorted(
                x for x in set(c["candidate_route"]) if c["candidate_route"].count(x) > c["llm_route"].count(x))


def test_the_route_of_a_case_is_the_raw_llm_route_with_no_marker(run_a, res_a):
    for c in res_a["cases"]:
        if c["type"] in C.TYPES:
            rec = next(r for r in run_a["records"] if r["part_id"] == c["part"]["part_id"] and r["repeat"] == c["repeat"])
            cand_ = next(x for x in rec["candidates"] if x["sample_id"] == c["sample_id"])
            assert c["llm_route"] == V.clean_steps(cand_["llm_steps"]) and V.MARKER not in c["llm_route"]
            assert c["candidate"] == cand_["name"]
            assert c["selected_as_final"] == (rec["final"]["name"] == cand_["name"] and cand_["in_pool"])


def test_violations_are_the_verifiers_words(run_a, res_a):
    for c in res_a["cases"]:
        if c["type"] in C.TYPES:
            rec = next(r for r in run_a["records"] if r["part_id"] == c["part"]["part_id"] and r["repeat"] == c["repeat"])
            cand_ = next(x for x in rec["candidates"] if x["sample_id"] == c["sample_id"])
            assert [v["message"] for v in c["violations"]] == [v["message"] for v in cand_["first_pass"]["violations"]]
            assert [v["rule_id"] for v in c["violations"]] == [v["rule_id"] for v in cand_["first_pass"]["violations"]]


def test_final_route_block_is_the_records_final(run_a, res_a):
    for c in res_a["cases"]:
        rec = next(r for r in run_a["records"] if r["part_id"] == c["part"]["part_id"] and r["repeat"] == c["repeat"])
        fin = c["final"]
        assert fin["route"] == V.clean_steps(rec["final"]["steps"]) and fin["valid"] == rec["final"]["check"]["valid"]
        assert fin["provenance"] == rec["final"]["provenance"]
        if fin["valid"]:
            assert fin["efficiency_gap_to_rb_best"] == pytest.approx(
                max(s["efficiency_score"] for s in rec["best_scores"]) - rec["final"]["scores_eval"]["efficiency_score"])
            assert fin["efficiency_gap_to_rb_best"] >= -1e-9                    # the best of the RB variants is at least this good


def test_facts_mention_the_numbers_of_the_records(res_a):
    for c in res_a["cases"]:
        if c["type"] in C.TYPES:
            text = " ".join(c["facts"])
            assert f"({len(c['llm_route'])} operations)" in text
            assert c["repair"]["outcome"] in text
            assert c["final"]["provenance"] in text


def test_feasibility_findings_are_listed_and_hard_ones_marked(res_a):
    inf = [c for c in res_a["cases"] if c["type"] == "infeasible"]
    assert inf
    for c in inf:
        hard = [v for v in c["feasibility"] if v["hard"]]
        assert hard and all(v["kind"] == "infeasible" for v in hard)
        assert f"{len(hard)} hard and {len(c['feasibility']) - len(hard)} soft finding(s)" in " ".join(c["facts"])


def test_references_are_labelled_proxy_without_expert_routes(res_a):
    for c in res_a["cases"]:
        assert c["final"]["reference_kind"] in ("proxy_rb", None)
        if c["final"]["reference_kind"]:
            assert "a proxy, not an expert" in " ".join(c["facts"])


def test_a_part_with_expert_routes_is_compared_with_them(run_a):
    recs = copy.deepcopy(run_a["records"])
    for r in recs:
        if r["part_id"] == "DEV_001":
            r["part"]["reference_routes"] = [["Facing", "Inspection"]]
    res = C.analyse([dict(run_a, records=recs)], types=["order", "missing"], per_type=10)
    one = next(c for c in res["cases"] if c["part"]["part_id"] == "DEV_001")
    assert one["final"]["reference_kind"] == "expert" and one["part"]["has_expert_reference"]
    assert "an expert reference route" in " ".join(one["facts"])
    other = next(c for c in res["cases"] if c["part"]["part_id"] != "DEV_001")
    assert other["final"]["reference_kind"] == "proxy_rb" and not other["part"]["has_expert_reference"]


# ---------------------------------------------------------------- repair stages, with R2 and R0 made by the harness
@pytest.fixture(scope="module")
def r2_runs(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("r2"))
    parts = dataset.load_dataset("dev_v2")[:8]

    def good(material, features, tolerance, batch_size, sample_id=0, previous_steps=(), violations=()):
        steps = V.clean_steps(rb.generate_valid_routes(features, max_routes=3)[0]["steps"])
        return {"success": True, "steps": steps, "raw_response": str(steps), "model": "fake",
                "meta": {"cached": True, "finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.1}}

    def same_mistake(material, features, tolerance, batch_size, sample_id=0, previous_steps=(), violations=()):
        return {"success": True, "steps": list(previous_steps), "raw_response": "", "model": "fake",
                "meta": {"cached": True, "finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.1}}
    harness.run_experiment("n3_r2_s1", parts, repeats=1, tag="r2ok", out_dir=out, dataset_name="dev_v2", progress=None,
                           planner_factory=planner_factory_for("C"), repair_planner=good, analysis=True)
    harness.run_experiment("n3_r2_s1", parts, repeats=1, tag="r2bad", out_dir=out, dataset_name="dev_v2", progress=None,
                           planner_factory=planner_factory_for("C"), repair_planner=same_mistake, analysis=True)
    return out


def test_an_accepted_reprompt_is_reported(r2_runs):
    run = C.load_run(os.path.join(r2_runs, "r2ok_n3_r2_s1"))
    res = C.analyse([run], types=["missing", "order"], per_type=10)
    assert res["cases"]
    for c in res["cases"]:
        assert c["repair"]["stage"] == "r2" and c["repair"]["r2"]["ok"] is True
        assert "repaired by R2" in c["repair"]["outcome"] and "R2 re-prompt: the answer was accepted" in " ".join(c["facts"])
        assert c["repair"]["r2"]["steps"]
    assert "The model's second answer (R2)" in C.render_markdown(res)


def test_a_refused_reprompt_ends_in_r0_and_says_so(r2_runs):
    run = C.load_run(os.path.join(r2_runs, "r2bad_n3_r2_s1"))
    res = C.analyse([run], types=["missing", "order"], per_type=10)
    assert res["cases"]
    for c in res["cases"]:
        assert c["repair"]["r2"]["ok"] is False and c["repair"]["tried"] == ["r2"]
        assert c["repair"]["outcome"].startswith("replaced by the Route Builder's route (R0) after R2")
        assert "the answer was not accepted" in " ".join(c["facts"])
        assert c["repair"]["fallback"] == "r0" and c["candidate_route"] and c["candidate_route"] != c["llm_route"]


# ---------------------------------------------------------------- the aggregate table
def test_the_table_agrees_with_the_summary_of_the_run(run_a):
    s = S.summarize_set(run_a["records"])
    table = {r["type"]: r for r in C.type_table(run_a["records"], C.DEFAULT_TYPES)}
    n_parsed = sum(1 for _ in raw_routes(run_a["records"]))
    rows = [r for r in S.candidate_rows(run_a["records"]) if r["llm_ok"]]
    for t in ("order", "missing", "invented", "duplicate"):
        assert table[t]["routes"] == s["M3"]["kinds_among_invalid"].get(t, 0)
        assert table[t]["routes"] == sum(1 for r in rows if t in r["kinds"].split(";"))                  # the per_candidate.csv column
        assert table[t]["share"] == pytest.approx(table[t]["routes"] / n_parsed)
        assert table[t]["parts"] == len({rec["part_id"] for rec, c in raw_routes(run_a["records"]) if t in c["first_pass"]["kinds"]})
    assert table["infeasible"]["routes"] == sum(1 for r in rows if r["raw_feas_infeasible"])
    assert table["machine_grouping"]["routes"] == s["M3"]["soft_machine_grouping_in_parsed"]


def test_the_table_counts_parts_that_can_give_a_case(run_a, res_a):
    for row in C.type_table(run_a["records"], C.DEFAULT_TYPES):
        n = len(got(res_a, run_a["label"], row["type"]))
        assert n == min(4, row["eligible"])


def test_runs_made_without_the_analysis_have_no_infeasible_cases(run_b):
    res = C.analyse([run_b], per_type=3)
    inf = next(r for r in res["runs"][0]["table"] if r["type"] == "infeasible")
    assert inf["routes"] == 0 and inf["denominator"] == 0 and "no feasibility analysis" in inf["note"]
    assert got(res, run_b["label"], "infeasible") == []
    md = C.render_markdown(res)
    assert "no feasibility analysis in these records" in md
    for c in res["cases"]:
        if c["type"] in C.TYPES:
            assert c["feasibility"] is None and "was not run on this route" in " ".join(c["facts"])


def test_a_route_builder_run_has_nothing_to_study(run_rb):
    res = C.analyse([run_rb], per_type=3)
    assert res["cases"] == [] and res["runs"][0]["n_parsed_routes"] == 0
    assert "no raw LLM routes" in C.render_markdown(res)


# ---------------------------------------------------------------- final valid, unlike the reference
def test_differs_lists_valid_finals_that_are_not_an_exact_match(run_a, res_a):
    want = []
    seen = set()
    for rec in sorted(run_a["records"], key=lambda r: (pnum(r["part_id"]), r["repeat"])):
        refs = rec["references"]
        if rec["part_id"] in seen or not rec["final"]["check"]["valid"]:
            continue
        if not any(V.clean_steps(rec["final"]["steps"]) == V.clean_steps(r) for r in refs):
            want.append((rec["part_id"], rec["repeat"], None))
            seen.add(rec["part_id"])
    assert got(res_a, run_a["label"], "differs") == want[:4]
    for c in res_a["cases"]:
        if c["type"] == "differs":
            assert c["final"]["valid"] and not c["final"]["similarity"]["exact"]
            assert c["reference_delta"]["only_in_reference"] is not None


def test_differs_uses_expert_routes_when_the_part_has_them(run_a):
    recs = copy.deepcopy(run_a["records"])
    for r in recs:
        if r["part_id"] == "DEV_001":
            r["part"]["reference_routes"] = [list(r["final"]["steps"])]            # the expert route is exactly the final route: not a case
        if r["part_id"] == "DEV_002":
            r["part"]["reference_routes"] = [["Facing", "Inspection"]]             # differs from the final route: a case
    res = C.analyse([dict(run_a, records=recs)], types=["differs"], per_type=10)
    ids = {c["part"]["part_id"]: c for c in res["cases"]}
    assert "DEV_001" not in ids or ids["DEV_001"]["final"]["reference_kind"] != "expert"
    assert ids["DEV_002"]["final"]["reference_kind"] == "expert" and ids["DEV_002"]["final"]["nearest_reference"] == ["Facing", "Inspection"]
    assert "the references are expert routes" in " ".join(ids["DEV_002"]["facts"])


def test_an_invalid_final_route_is_never_a_differs_case(run_a):
    recs = copy.deepcopy(run_a["records"])
    for r in recs:
        r["final"]["check"]["valid"] = False
    assert C.analyse([dict(run_a, records=recs)], types=["differs"], per_type=5)["cases"] == []


def test_differs_case_carries_the_facts_of_the_sample_it_came_from(run_a, res_a):
    made = [c for c in res_a["cases"] if c["type"] == "differs" and c["from_candidate"]]
    assert made
    for c in made:
        src = c["from_candidate"]
        assert src["candidate"] == c["final"]["name"] and src["sample_id"] is not None
        assert f"It was built from sample {src['sample_id']}" in " ".join(c["facts"])


# ---------------------------------------------------------------- instructive cases
def test_an_instructive_case_is_labelled_and_outside_the_rule(run_a):
    spec = C.parse_instructive("DEV_005:2=the repair kept a wrong decision")
    res = C.analyse([run_a], types=["order"], per_type=2, instructive=[spec])
    ins = [c for c in res["cases"] if c["type"] == "instructive"]
    assert len(ins) == 1 and ins[0]["source"] == "instructive" and ins[0]["note"] == "the repair kept a wrong decision"
    assert ins[0]["part"]["part_id"] == "DEV_005" and ins[0]["repeat"] == 2
    assert [c["id"] for c in res["cases"] if c["type"] == "order"] == [c["id"] for c in C.analyse([run_a], types=["order"], per_type=2)["cases"]]
    md = C.render_markdown(res)
    assert "instructive, chosen by the author: the repair kept a wrong decision" in md and "Instructive cases (chosen by the author)" in md


def test_an_instructive_case_without_a_sample_takes_the_first_invalid_route(run_a):
    res = C.analyse([run_a], types=["order"], per_type=1, instructive=[C.parse_instructive("DEV_004:1=x")])
    ins = next(c for c in res["cases"] if c["type"] == "instructive")
    rec = next(r for r in run_a["records"] if r["part_id"] == "DEV_004" and r["repeat"] == 1)
    first_bad = next(c for c in sorted(rec["candidates"], key=lambda c: c["sample_id"]) if c["sample_id"] >= 0 and c["llm_steps"] and not c["first_pass"]["valid"])
    assert ins["sample_id"] == first_bad["sample_id"]


def test_instructive_entries_must_exist_in_some_run(run_a, run_rb):
    with pytest.raises(C.CaseError, match="DEV_999"):
        C.analyse([run_a], instructive=[C.parse_instructive("DEV_999=x")])
    with pytest.raises(C.CaseError, match="sample 77"):
        C.analyse([run_a], instructive=[C.parse_instructive("DEV_001:1:77=x")])
    with pytest.raises(C.CaseError, match="Route Builder run has no LLM routes"):
        C.analyse([run_rb], instructive=[C.parse_instructive("DEV_001=x")])
    res = C.analyse([run_rb, run_a], types=["order"], per_type=1, instructive=[C.parse_instructive("DEV_001=x")])        # found in one run is enough
    assert [c["run"] for c in res["cases"] if c["type"] == "instructive"] == [run_a["label"]]


# ---------------------------------------------------------------- commentary
def test_commentary_is_kept_and_new_cases_get_an_empty_entry(run_a, tmp_path):
    path = str(tmp_path / "commentary.json")
    assert C.read_commentary(path) == {} and C.read_commentary(None) == {}
    first = C.analyse([run_a], types=["order"], per_type=2)
    assert C.sync_commentary(path, first["cases"]) == 2
    data = C.read_commentary(path)
    key = first["cases"][0]["id"]
    data[key] = "The model forgot the precedence of the drill."
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    bigger = C.analyse([run_a], types=["order"], per_type=3, commentary=C.read_commentary(path))
    assert C.sync_commentary(path, bigger["cases"]) == 1                          # only the new case is added
    assert C.read_commentary(path)[key] == "The model forgot the precedence of the drill."
    assert bigger["n_without_commentary"] == 2 and bigger["cases"][0]["commentary"].startswith("The model forgot")
    md = C.render_markdown(bigger)
    assert "**Commentary.** The model forgot the precedence of the drill." in md and md.count(C.GAP) == 2
    assert "2 of 3 case(s) still have no commentary" in md


def test_commentary_of_cases_no_longer_selected_is_reported_not_deleted(run_a, tmp_path):
    path = str(tmp_path / "c.json")
    big = C.analyse([run_a], types=["order"], per_type=3)
    data = {c["id"]: f"text {i}" for i, c in enumerate(big["cases"])}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    small = C.analyse([run_a], types=["order"], per_type=1, commentary=C.read_commentary(path))
    assert len(small["orphan_commentary"]) == 2 and "2 written commentary entries match no case" in C.render_markdown(small)
    C.sync_commentary(path, small["cases"])
    assert C.read_commentary(path) == data


def test_a_bad_commentary_file_is_an_error(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{not json")
    with pytest.raises(C.CaseError, match="not a valid commentary file"):
        C.read_commentary(str(p))
    p.write_text('["a"]')
    with pytest.raises(C.CaseError, match="one JSON object"):
        C.read_commentary(str(p))
    p.write_text('{"a": 3}')
    with pytest.raises(C.CaseError, match="one JSON object"):
        C.read_commentary(str(p))


def test_whitespace_only_commentary_counts_as_missing(run_a):
    first = C.analyse([run_a], types=["order"], per_type=1)
    again = C.analyse([run_a], types=["order"], per_type=1, commentary={first["cases"][0]["id"]: "   \n "})
    assert again["n_without_commentary"] == 1


# ---------------------------------------------------------------- the output
def test_markdown_is_deterministic_and_complete(run_a):
    a = C.render_markdown(C.analyse([run_a], per_type=3))
    b = C.render_markdown(C.analyse([run_a], per_type=3))
    assert a == b
    for piece in ("# Failure case studies", "## How the cases were selected", "| Type | Raw routes with it | Share | Parts | Cases here |",
                  "**LLM route** (as sampled)", "**Violation report** (strict check)", "**Repair applied:**", "**Final route of the part**",
                  "**Facts from the records:**", "**Commentary.**", C.GAP):
        assert piece in a, piece
    assert "~~" in a                                                                 # an operation the repair removed is struck through
    assert re.search(r"\*\*Route after the repair:\*\*[^\n]*\*\*[A-Z][^*\n]+\*\*", a)       # one the repair inserted is in bold


def test_every_selected_type_without_a_case_says_why(run_a):
    md = C.render_markdown(C.analyse([run_a], types=["machine_grouping", "order"], per_type=1))
    assert "### Wrong machine grouping: more machine changeovers than needed (soft)\n\nNo case: no failure of this type in this run." in md


def test_sealed_datasets_get_a_note(run_a):
    sealed = dict(run_a, run_info=dict(run_a["run_info"], dataset="test_v2"))
    assert "is sealed: cases from it belong in the report only if they come from the final run" in C.render_markdown(C.analyse([sealed], per_type=1))
    assert "is sealed" not in C.render_markdown(C.analyse([run_a], per_type=1))


def test_write_outputs_makes_the_three_files(res_a, tmp_path):
    paths = C.write_outputs(res_a, str(tmp_path / "o"))
    assert set(paths) == {"case_studies.md", "cases.json", "selection.csv"}
    with open(paths["cases.json"], encoding="utf-8") as f:
        assert json.load(f)["cases"][0]["id"] == res_a["cases"][0]["id"]
    with open(paths["selection.csv"], encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(res_a["cases"]) and list(rows[0].keys()) == C.SELECTION_FIELDS
    assert rows[0]["case_id"] == res_a["cases"][0]["id"] and rows[0]["has_commentary"] == "False"
    assert not any("\r\n" in open(p, encoding="utf-8", newline="").read() for p in paths.values())          # same bytes on every platform


# ---------------------------------------------------------------- the command line
def cli(*args):
    return case_studies.main([str(a) for a in args])


def test_cli_writes_the_files_and_keeps_the_commentary(runs_dir, tmp_path, capsys):
    out = tmp_path / "out"
    run = folder(runs_dir, "n5_r1_s3+fz2", "cs")
    assert cli(run, "--per-type", 2, "--types", "order,missing", "--out", out) == 0
    text = capsys.readouterr().out
    assert "order" in text and "4 case(s); 4 without commentary" in text
    assert sorted(os.listdir(out)) == ["case_studies.md", "cases.json", "commentary.json", "selection.csv"]
    com = json.loads((out / "commentary.json").read_text(encoding="utf-8"))
    key = sorted(com)[0]
    com[key] = "written by hand"
    (out / "commentary.json").write_text(json.dumps(com), encoding="utf-8")
    assert cli(run, "--per-type", 2, "--types", "order,missing", "--out", out) == 0
    assert "3 without commentary" in capsys.readouterr().out
    assert json.loads((out / "commentary.json").read_text(encoding="utf-8"))[key] == "written by hand"
    assert "written by hand" in (out / "case_studies.md").read_text(encoding="utf-8")


def test_cli_default_out_is_beside_the_run(runs_dir, tmp_path, capsys):
    results = tmp_path / "results"
    shutil.copytree(folder(runs_dir, "n5_r1_s3+fz2", "cs"), results / "cs_n5_r1_s3+fz2")
    assert cli(results / "cs_n5_r1_s3+fz2", "--types", "order", "--per-type", 1) == 0
    assert (results / "case_studies" / "cs_n5_r1_s3+fz2" / "case_studies.md").exists()
    assert case_studies.default_out(["a/x", "a/y"]).endswith(os.path.join("case_studies", "x+1more"))


def test_cli_strict_fails_until_every_case_has_commentary(runs_dir, tmp_path, capsys):
    out = tmp_path / "o"
    run = folder(runs_dir, "n5_r1_s3+fz2", "cs")
    assert cli(run, "--types", "order", "--per-type", 1, "--out", out, "--strict") == 1
    capsys.readouterr()
    com = json.loads((out / "commentary.json").read_text(encoding="utf-8"))
    (out / "commentary.json").write_text(json.dumps({k: "done" for k in com}), encoding="utf-8")
    assert cli(run, "--types", "order", "--per-type", 1, "--out", out, "--strict") == 0


def test_cli_instructive_option(runs_dir, tmp_path, capsys):
    out = tmp_path / "o"
    assert cli(folder(runs_dir, "n5_r1_s3+fz2", "cs"), "--types", "order", "--per-type", 1, "--instructive", "DEV_004:1=because", "--out", out) == 0
    assert "instructive, chosen by the author: because" in (out / "case_studies.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("extra,msg", [
    (["--types", "nonsense"], "unknown type"),
    (["--per-type", "99"], "--per-type"),
    (["--instructive", "DEV_001"], "reason is required"),
    (["--instructive", "DEV_999=x"], "DEV_999"),
])
def test_cli_errors_exit_1_with_a_message(runs_dir, tmp_path, capsys, extra, msg):
    with pytest.raises(SystemExit) as e:
        cli(folder(runs_dir, "n5_r1_s3+fz2", "cs"), "--out", tmp_path / "o", *extra)
    assert e.value.code == 1 and msg in capsys.readouterr().err
    assert not (tmp_path / "o").exists()                                          # nothing half-written


def test_cli_rejects_a_folder_that_is_not_a_run_and_duplicate_names(runs_dir, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        cli(tmp_path, "--out", tmp_path / "o")
    assert e.value.code == 1 and "no records.jsonl" in capsys.readouterr().err
    run = folder(runs_dir, "n5_r1_s3+fz2", "cs")
    other = tmp_path / "x" / os.path.basename(run)
    shutil.copytree(run, other)
    with pytest.raises(SystemExit) as e:
        cli(run, other, "--out", tmp_path / "o2")
    assert e.value.code == 1 and "same folder name" in capsys.readouterr().err


def test_cli_with_a_route_builder_run_says_nothing_to_study(runs_dir, tmp_path, capsys):
    assert cli(folder(runs_dir, "rb", "cs"), "--out", tmp_path / "o") == 0
    assert "no raw LLM routes: nothing to study" in capsys.readouterr().out
