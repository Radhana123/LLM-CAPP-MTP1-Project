"""Step 24: the constrained (graph-guided) planner (capp/constrained.py, option +cg1; row A9 of the ablation grid, RQ5).

Offline. The LLM is a fake transport that answers the "which option comes next" question by a policy (first, last, random, garbage ...),
so every route the planner can build is exercised, over all 200 development parts. What the tests pin down: the route is valid by
construction whatever the model answers; the model is only asked when there is a real choice; an unreadable answer is counted and takes
the Route Builder's order; the calls, tokens and latency are all in the record; the option plugs into the grammar, the harness, the
summary, the table regenerator and the case-study generator.
"""
import itertools
import json
import os
import random
import re
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from capp import (casestudy, compare, config, constrained as CG, dataset, feasibility as F, harness, llm_client, planners, plannerspec as PS,
                  summary as S, systems, tables, verify as V)                                      # noqa: E402
import feature_vocab as fv                                                                          # noqa: E402
import precedence_graph as pg                                                                       # noqa: E402

OPT = re.compile(r"^(\d+)\. (.+?) \((.+?)\)$", re.M)


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.delenv("CAPP_LLM_MODE", raising=False)
    monkeypatch.delenv("CAPP_RUN_TAG", raising=False)
    monkeypatch.setattr(llm_client, "_sleep", lambda s: None)
    llm_client.reset_transport()
    llm_client.reset_call_log()
    yield
    llm_client.reset_transport()


@pytest.fixture(scope="module")
def dev():
    return dataset.load_dataset("dev_v2")


def options_of(messages):
    """[(number, operation, machine)] of the question in the last message."""
    return [(int(n), op, m) for n, op, m in OPT.findall(messages[-1]["content"])]


def transport_with(policy, seen=None):
    """A fake LLM: policy(options, messages) -> the text of the answer (options: the operation names in the order shown)."""
    def transport(model, messages, params):
        opts = [op for _, op, _ in options_of(messages)]
        if seen is not None:
            seen.append({"model": model, "messages": messages, "params": params, "options": opts})
        return {"text": policy(opts, messages), "prompt_tokens": 100, "completion_tokens": 3, "finish_reason": "stop"}
    return transport


def pick_first(opts, messages):
    return "1"


def pick_last(opts, messages):
    return str(len(opts))


def pick_random(opts, messages):
    return str(random.Random(messages[-1]["content"]).randrange(1, len(opts) + 1))


def run_planner(part, policy=pick_first, sample_id=0, seen=None, model=None):
    """One sample. The cache key does not know the policy, so every policy gets its own run tag (the same policy object shares one)."""
    llm_client.set_transport(transport_with(policy, seen))
    old = os.environ.get("CAPP_RUN_TAG")
    os.environ["CAPP_RUN_TAG"] = f"policy-{id(policy)}"
    try:
        planner = CG.make_planner(part, model=model)
        return planner(part["material"], part["features"], part.get("tolerance", "0.05mm"), part.get("batch_size", 100), sample_id=sample_id)
    finally:
        if old is None:
            os.environ.pop("CAPP_RUN_TAG", None)
        else:
            os.environ["CAPP_RUN_TAG"] = old


def alphabetical_topological_order(ops, edges):
    """The order the fake 'always the first option' model gets, worked out again here: Facing, then at each step the alphabetically first
    operation without an unplaced prerequisite, Inspection last."""
    rem, out = set(ops), []
    while rem:
        if "Facing" in rem:
            nxt = "Facing"
        else:
            free = sorted(o for o in rem if not any(a in rem and b == o for a, b in edges))
            if "Inspection" in free and len(rem) > 1:
                free.remove("Inspection")
            nxt = free[0]
        out.append(nxt)
        rem.remove(nxt)
    return out


# ---------------------------------------------------------------- the grammar
def test_option_in_the_grammar():
    base, spec = PS.split_name("n1_r0_s1+cg1")
    assert base == "n1_r0_s1" and spec.constrained == 1 and spec.name == "cg1" and not spec.is_default
    assert PS.split_name("n1_r0_s1+cg0")[1].constrained == 0 and PS.split_name("n1_r0_s1+cg0")[1].is_default
    assert systems.parse_system("n5_r0_s3+cg1") == (5, "r0", "s3")
    assert systems.parse_system("n3_r1_s3+cg1+ms+fz2") == (3, "r1", "s3")
    assert systems.planner_spec("n3_r1_s3+cg1+ms+fz2").model == "small"
    assert PS.PlannerSpec(few_shot=2).constrained == 0 and PS.PlannerSpec().name == ""


@pytest.mark.parametrize("name", ["n1_r0_s1+cg2", "n1_r0_s1+cg3", "n1_r0_s1+cg", "n1_r0_s1+cg1+cg0", "n1_r0_s1+cg1+ts", "n1_r0_s1+cg1+fs2",
                                  "n1_r0_s1+cg1+h1", "b0+cg1+fs2x+h1", "rb+cg1"])
def test_grammar_rejects(name):
    with pytest.raises(ValueError):
        systems.parse_system(name)


def test_cascade_and_model_options_are_exclusive_as_before():
    with pytest.raises(ValueError, match="cascade"):
        systems.parse_system("n5_r1_s3+cg1+ms+cas1")
    assert systems.planner_spec("n5_r1_s3+cg1+cas1").cascade == 1


# ---------------------------------------------------------------- the constraints
def test_required_operations_are_the_route_builders_own(dev):
    for part in dev:
        ops = CG.required_ops(part["features"], part.get("machine_preference", "auto"))
        assert ops == planners.rb_plan(part["features"], part.get("machine_preference", "auto"))
        assert V.check_route(ops, part["features"]).valid


def test_without_a_route_builder_route_the_graph_gives_the_set():
    """Turning + Slot + Hole + Chamfer: the Route Builder has no route (Chamfering lands in the lathe phase but must follow Slot Milling)."""
    feats = ["Turning", "Slot", "Hole", "Chamfer"]
    assert CG.required_ops(feats) is None
    ops = CG.graph_ops(feats)
    assert ops[0] == "Facing" and ops[-1] == "Inspection" and len(set(ops)) == len(ops)
    assert V.check_route(ops, feats).valid                                          # the graph alone has a valid order, with the coverage
    r = run_planner({"material": "Steel", "features": feats}, pick_random)
    assert r["success"] and r["meta"]["cg"]["set_source"] == "graph" and sorted(r["steps"]) == sorted(ops)
    chk = V.check_route(r["steps"], feats)
    assert chk.valid and chk.v1_valid
    fb = run_planner({"material": "Steel", "features": feats}, lambda o, m: "no idea")                  # unreadable answers: the graph's own order
    assert fb["steps"] == ops and fb["meta"]["cg"]["fallback_steps"] == fb["meta"]["cg"]["choice_steps"] > 0
    assert run_planner({"material": "Steel", "features": ["Turning", "Slot", "Hole"]}, pick_first)["meta"]["cg"]["set_source"] == "route_builder"


def test_a_feature_outside_the_vocabulary_is_a_failed_sample():
    llm_client.set_transport(transport_with(pick_first))
    r = CG.make_planner({})("Steel", ["Teleportation"], "0.05mm", 100)
    assert not r["success"] and r["steps"] == [] and r["raw_response"].startswith("no operation set") and not llm_client.get_call_log()
    with pytest.raises(ValueError):
        CG.graph_ops(["Teleportation"])


def test_a_cycle_among_the_graph_operations_is_reported(monkeypatch):
    monkeypatch.setattr(CG, "edges_for", lambda ops: {("Drilling", "Center Drilling"), ("Center Drilling", "Drilling")})
    with pytest.raises(ValueError, match="cycle"):
        CG.graph_ops(["Hole"])


def test_edges_are_the_precedence_edges_among_the_operations():
    ops = ["Facing", "Center Drilling", "Drilling", "Chamfering", "Inspection"]
    assert CG.edges_for(ops) == {(a, b) for a, b in pg.PRECEDENCE_EDGES if a in ops and b in ops}
    assert ("Center Drilling", "Drilling") in CG.edges_for(ops) and CG.edges_for(["Facing"]) == set()


def test_frontier_rules():
    ops = ["Facing", "Plain/Cylindrical Turning", "Boring", "Drilling", "Center Drilling", "Chamfering", "Inspection"]
    E = CG.edges_for(ops)
    assert CG.frontier(ops, E) == ["Facing"]                                              # Facing is the only option until it is placed
    rest = [o for o in ops if o != "Facing"]
    assert CG.frontier(rest, E) == ["Center Drilling", "Plain/Cylindrical Turning"]       # alphabetical; Inspection and the dependents are held back
    assert CG.frontier(["Inspection"], E) == ["Inspection"]
    assert CG.frontier(["Boring", "Chamfering", "Inspection"], E) == ["Boring"]           # Chamfering waits for Boring
    assert CG.frontier(["Chamfering", "Inspection"], E) == ["Chamfering"]
    assert CG.frontier(["Drilling", "Boring"], {("Drilling", "Boring"), ("Boring", "Drilling")}) == []      # a cycle leaves nothing


def test_a_cycle_in_the_constraints_stops_the_sample(monkeypatch):
    monkeypatch.setattr(CG, "edges_for", lambda ops: {("Drilling", "Center Drilling"), ("Center Drilling", "Drilling")})
    r = run_planner({"material": "Steel", "features": ["Hole"]})
    assert not r["success"] and r["raw_response"].startswith("constraint cycle")


def test_machine_classes():
    assert CG.machine_of("Boring") == "either machine" and CG.machine_of("Plain/Cylindrical Turning") == "lathe"
    assert CG.machine_of("Slot Milling") == "milling"


def test_physical_order_rules_are_implied_by_the_graph():
    """HOLE-EXISTS and SEQ-PARTOFF as ordering edges add nothing to the precedence graph for any operation set the Route Builder makes
    from one or two features (the same was found for all 7806 sets of up to three features when the planner was designed)."""
    feats = sorted(fv.ALL_FEATURES)
    checked = 0
    for k in (1, 2):
        for combo in itertools.combinations(feats, k):
            ops = CG.required_ops(list(combo))
            if not ops:
                continue
            checked += 1
            edges = CG.edges_for(ops)
            reach = {o: set() for o in ops}
            for a, b in edges:
                reach[a].add(b)
            changed = True
            while changed:                                  # transitive closure
                changed = False
                for a in ops:
                    new = set().union(*[reach[b] for b in reach[a]]) - reach[a] if reach[a] else set()
                    if new:
                        reach[a] |= new
                        changed = True
            for a, b in CG.physical_order_edges(ops):
                assert b in reach[a], (combo, a, b)
            assert CG.order_gaps(ops) == [], combo
    assert checked > 600


def test_the_one_gap_ordering_cannot_close():
    """The Route Builder's own route for Groove + Pocket + Chamfer has an Internal Grooving and no hole-making operation."""
    ops = CG.required_ops(["Chamfer", "Groove", "Pocket"])
    assert "Internal Grooving" in ops and not {"Drilling", "Boring"} & set(ops)
    assert CG.order_gaps(ops) == ["HOLE-EXISTS:Internal Grooving"]
    assert CG.order_gaps(["Facing", "Drilling", "Tapping", "Inspection"]) == []
    r = run_planner({"material": "Steel", "features": ["Chamfer", "Groove", "Pocket"]})
    assert r["success"] and r["meta"]["cg"]["unsatisfiable"] == ["HOLE-EXISTS:Internal Grooving"]


# ---------------------------------------------------------------- the answer
@pytest.mark.parametrize("text,expect", [
    ("2", "B"), (" 1 ", "A"), ("3.", "C"), ("Option 2", "B"), ("<think>maybe 3</think>2", "B"), ("I choose 3 because", "C"),
    ("0", None), ("4", None), ("7 is wrong, 2", None), ("", None), ("no idea", None),
    ("B", "B"), ("I would do b next", "B"), ("A or B", None), ("Drilling", None),
])
def test_parse_choice(text, expect):
    opts = ["A", "B", "C"]
    assert CG.parse_choice(text, opts) == expect


def test_a_name_inside_a_longer_name_is_not_a_second_match():
    opts = ["Center Drilling", "Drilling"]
    assert CG.parse_choice("Center Drilling", opts) == "Center Drilling" and CG.parse_choice("Drilling", opts) == "Drilling"
    assert CG.parse_choice("Drilling, then Center Drilling", opts) is None or CG.parse_choice("Drilling, then Center Drilling", opts) == "Center Drilling"
    assert CG.parse_choice("Redrilling", ["Drilling", "Boring"]) is None                                # inside a word is not the word


def test_parse_choice_by_operation_name():
    opts = ["Boring", "Drilling", "Reaming"]
    assert CG.parse_choice("Drilling", opts) == "Drilling" and CG.parse_choice("drilling first", opts) == "Drilling"
    assert CG.parse_choice("Boring or Reaming", opts) is None                       # two names: not guessed


# ---------------------------------------------------------------- the question
def test_the_question_shows_only_the_frontier_and_says_what_is_left():
    part = {"material": "Steel", "features": ["Hole", "Slot", "Chamfer"], "tolerance": "0.02mm", "batch_size": 50}
    seen = []
    run_planner(part, pick_first, seen=seen)
    assert seen
    ops = CG.required_ops(part["features"])
    E = CG.edges_for(ops)
    for call in seen:
        msgs = call["messages"]
        assert [m["role"] for m in msgs] == ["system", "user"] and msgs[0]["content"] == CG.SYSTEM_PROMPT
        user = msgs[1]["content"]
        placed = re.search(r"Route so far: (.*)", user).group(1)
        placed = [] if placed == "(empty)" else placed.split(" > ")
        left = [o for o in ops if o not in placed]
        assert call["options"] == CG.frontier(left, E) == sorted(call["options"]) and len(call["options"]) > 1
        after = re.search(r"Operations still to place after this choice: (.*)", user).group(1)
        assert set(after.split(", ")) | set(call["options"]) == set(left)                 # nothing is hidden, nothing is offered too early
        assert "Steel" in user and "Hole, Slot, Chamfer" in user and "tolerance 0.02mm" in user and "batch size 50" in user
        for _, op, machine in options_of(msgs):
            assert machine == CG.machine_of(op)
    assert "Machine of the last operation" in seen[0]["messages"][1]["content"]
    assert seen[0]["params"] == {"temperature": config.PLANNER_TEMPERATURE, "max_tokens": config.CG_MAX_TOKENS}


def test_the_question_names_the_machine_of_the_last_operation():
    seen = []
    run_planner({"material": "Steel", "features": ["Turning", "Slot", "Hole"]}, pick_first, seen=seen)
    lasts = [re.search(r"Machine of the last operation: (.*)", c["messages"][1]["content"]).group(1) for c in seen]
    assert set(lasts) <= {"lathe", "milling", "either machine", "none yet"}


# ---------------------------------------------------------------- valid by construction
@pytest.mark.parametrize("policy", [pick_first, pick_last, pick_random])
def test_every_route_is_valid_for_every_development_part(dev, policy):
    n = 0
    for part in dev:
        r = run_planner(part, policy, sample_id=n % 3)
        assert r["success"], r
        chk = V.check_route(r["steps"], part["features"])
        assert chk.valid and chk.v1_valid, (part["part_id"], [v.message for v in chk.violations])
        assert set(chk.kinds) <= {"machine_grouping"}
        assert r["steps"][0] == "Facing" and r["steps"][-1] == "Inspection"
        assert sorted(r["steps"]) == sorted(CG.required_ops(part["features"], part.get("machine_preference", "auto")))      # the SET is untouched
        n += 1


def test_always_first_is_the_alphabetical_topological_order(dev):
    for part in dev[:60]:
        ops = CG.required_ops(part["features"], part.get("machine_preference", "auto"))
        r = run_planner(part, pick_first)
        assert r["steps"] == alphabetical_topological_order(ops, CG.edges_for(ops)), part["part_id"]


def test_the_model_decides_the_order():
    part = {"material": "Steel", "features": ["Turning", "Slot", "Hole"]}
    a = run_planner(part, pick_first)["steps"]
    b = run_planner(part, pick_last)["steps"]
    assert a != b and sorted(a) == sorted(b)


# ---------------------------------------------------------------- calls, tokens, latency
def test_calls_are_the_real_choices_only(dev):
    for part in dev[:40]:
        seen = []
        r = run_planner(part, pick_random, seen=seen)
        ops = CG.required_ops(part["features"], part.get("machine_preference", "auto"))
        E = CG.edges_for(ops)
        rem, expected = list(ops), 0
        for op in r["steps"]:
            expected += len(CG.frontier(rem, E)) > 1
            rem.remove(op)
        cg, meta = r["meta"]["cg"], r["meta"]
        assert expected == meta["n_calls"] == cg["choice_steps"] == len(meta["calls"])
        assert len(seen) == sum(1 for c in meta["calls"] if not c["cached"])               # a question asked before comes from the cache
        assert cg["forced_steps"] + cg["choice_steps"] == cg["n_ops"] == len(r["steps"])
        assert [c["pos"] for c in cg["choices"]] == list(range(len(r["steps"]))) and [c["choice"] for c in cg["choices"]] == r["steps"]
        assert [c["how"] for c in cg["choices"]].count("forced") == cg["forced_steps"]
        for c in cg["choices"]:
            assert c["choice"] in c["options"] and (c["how"] != "forced" or c["options"] == [c["choice"]])


def test_meta_sums_the_calls():
    part = {"material": "Steel", "features": ["Turning", "Slot", "Hole", "Groove"]}
    r = run_planner(part, pick_first)
    m = r["meta"]
    assert m["n_calls"] >= 2 and m["prompt_tokens"] == 100 * m["n_calls"] and m["completion_tokens"] == 3 * m["n_calls"]
    assert m["latency_s"] == pytest.approx(sum(c["latency_s"] for c in m["calls"])) and m["cached"] is False and m["finish_reason"] == "stop"
    assert m["prompt_context"] == {"examples": [], "hint": False, "constrained": True} and m["model"] == r["model"]
    assert json.loads(r["raw_response"]) == r["steps"]
    assert all(c["role"] == "choose" and c["n_options"] >= 2 for c in m["calls"])
    again = run_planner(part, pick_first)
    assert again["meta"]["cached"] is True and again["steps"] == r["steps"]                  # the same questions come out of the cache
    other = run_planner(part, pick_first, sample_id=1)
    assert other["meta"]["cached"] is False                                                   # another sample is another draw


def test_a_part_with_no_choice_needs_no_call(dev):
    forced = []
    for p in dev:
        seen = []
        r = run_planner(p, pick_first, seen=seen)
        if r["meta"]["cg"]["choice_steps"] == 0:
            forced.append((p, r, seen))
    assert forced, "the development set has parts whose order the graph leaves open nowhere"
    for p, r, seen in forced:
        ops = CG.required_ops(p["features"], p.get("machine_preference", "auto"))
        rem, E = list(ops), CG.edges_for(ops)
        for op in r["steps"]:
            assert len(CG.frontier(rem, E)) == 1                                       # one order only, checked independently
            rem.remove(op)
        assert not seen and r["success"] and r["meta"]["calls"] == [] and r["meta"]["cached"] is True
        assert r["meta"]["prompt_tokens"] == 0 and r["meta"]["latency_s"] == 0 and r["meta"]["finish_reason"] == "stop"
        assert r["steps"] == ops                                                       # and it is the Route Builder's route


def test_the_model_id_goes_to_the_transport():
    seen = []
    run_planner({"material": "Steel", "features": ["Turning", "Slot", "Hole"]}, pick_first, seen=seen, model="openai/gpt-oss-20b")
    assert seen and {c["model"] for c in seen} == {"openai/gpt-oss-20b"}


# ---------------------------------------------------------------- an answer that cannot be used
def test_unreadable_answers_take_the_route_builders_order(dev):
    for part in dev[:50]:
        r = run_planner(part, lambda o, m: "I cannot decide")
        assert r["success"]
        cg = r["meta"]["cg"]
        assert cg["fallback_steps"] == cg["choice_steps"] and all(c["how"] in ("forced", "fallback") for c in cg["choices"])
        assert r["steps"] == CG.required_ops(part["features"], part.get("machine_preference", "auto"))         # the Route Builder's own route
        assert V.check_route(r["steps"], part["features"]).valid


def test_out_of_range_numbers_and_names_are_told_apart():
    part = {"material": "Steel", "features": ["Turning", "Slot", "Hole", "Groove"]}
    out_of_range = run_planner(part, lambda o, m: str(len(o) + 5))
    assert out_of_range["meta"]["cg"]["fallback_steps"] == out_of_range["meta"]["cg"]["choice_steps"] > 0
    by_name = run_planner(part, lambda o, m: o[-1])                                       # the name of the last option, no number
    assert by_name["meta"]["cg"]["fallback_steps"] == 0 and by_name["steps"] == run_planner(part, pick_last)["steps"]
    for c in by_name["meta"]["cg"]["choices"]:
        assert c["how"] in ("forced", "llm") and len(c["answer"]) <= CG.ANSWER_CHARS


def test_think_blocks_are_not_the_answer():
    part = {"material": "Steel", "features": ["Turning", "Slot", "Hole"]}
    r = run_planner(part, lambda o, m: f"<think>option 1 looks fine, 2 too</think>{len(o)}")
    assert r["steps"] == run_planner(part, pick_last)["steps"] and r["meta"]["cg"]["fallback_steps"] == 0


# ---------------------------------------------------------------- errors are not hidden
def test_no_api_key_is_a_failed_sample():
    part = {"material": "Steel", "features": ["Turning", "Slot", "Hole"]}

    def no_key(model, messages, params):
        raise llm_client.NoApiKey("no key")
    llm_client.set_transport(no_key)
    r = CG.make_planner(part)("Steel", part["features"], "0.05mm", 100)
    assert r == {"success": False, "steps": [], "raw_response": "GROQ_API_KEY not configured", "model": r["model"]}


def test_an_api_error_in_the_middle_keeps_the_calls_made_so_far():
    part = {"material": "Steel", "features": ["Turning", "Slot", "Hole", "Groove"]}
    n = {"k": 0}

    def flaky(model, messages, params):
        n["k"] += 1
        if n["k"] >= 2:                                 # llm_client retries, so the failure has to last
            raise RuntimeError("boom")
        return {"text": "1", "prompt_tokens": 10, "completion_tokens": 1, "finish_reason": "stop"}
    llm_client.set_transport(flaky)
    r = CG.make_planner(part)("Steel", part["features"], "0.05mm", 100)
    assert not r["success"] and r["steps"] == [] and r["raw_response"].startswith("API Error")
    assert r["meta"]["n_calls"] == 1 and r["meta"]["prompt_tokens"] == 10 and r["meta"]["finish_reason"] == "error"       # the paid call is counted


def test_replay_mode_without_a_cache_entry_fails_the_sample(monkeypatch):
    monkeypatch.setenv("CAPP_LLM_MODE", "replay")
    r = run_planner({"material": "Steel", "features": ["Turning", "Slot", "Hole"]}, pick_first)
    assert not r["success"] and "API Error" in r["raw_response"]


# ---------------------------------------------------------------- the factory
def test_the_factory_builds_the_constrained_planner_for_the_part(dev):
    spec = PS.split_name("n1_r0_s1+cg1+ml")[1]
    part = dict(dev[0])
    seen = []
    llm_client.set_transport(transport_with(pick_first, seen))
    planner = planners.make_factory(spec)(part)
    r = planner(part["material"], part["features"], part["tolerance"], part["batch_size"], sample_id=0)
    assert r["success"] and r["meta"]["cg"]["n_ops"] == len(r["steps"])
    assert all(c["model"] == config.model("large") for c in seen)
    small = planners.make_factory(PS.split_name("n1_r0_s1+cg1")[1])(part, role="small")
    seen.clear()
    small(part["material"], part["features"], part["tolerance"], part["batch_size"], sample_id=0)
    assert all(c["model"] == config.model("small") for c in seen)


def test_the_machine_preference_comes_from_the_part():
    part = {"material": "Steel", "features": ["Turning", "Slot", "Hole", "Groove"], "machine_preference": "prefer_lathe"}
    assert CG.required_ops(part["features"], "prefer_lathe")
    seen = []
    run_planner(part, pick_first, seen=seen)
    assert seen
    for c in seen:
        assert "machine preference: prefer_lathe" in c["messages"][1]["content"]
    seen.clear()
    run_planner(dict(part, machine_preference="auto"), pick_first, seen=seen)
    assert all("machine preference: auto" in c["messages"][1]["content"] for c in seen)


# ---------------------------------------------------------------- harness, summary, tables, case studies
N_RUN = 14


@pytest.fixture(scope="module")
def cg_runs(tmp_path_factory, dev):
    out = str(tmp_path_factory.mktemp("cg"))
    mp = pytest.MonkeyPatch()
    mp.setenv("CAPP_CACHE_DIR", str(tmp_path_factory.mktemp("cgcache")))
    mp.setenv("GROQ_API_KEY", "")
    def v1_or_choice(model, messages, params):
        user = messages[-1]["content"]
        if "Options for the next operation" in user:
            return transport_with(pick_random)(model, messages, params)
        feats = re.search(r"Features: (.*)\n", user).group(1).split(", ")           # the v1 prompt of system b1: answer with the Route Builder's route
        import route_builder as rb
        steps = V.clean_steps(rb.generate_valid_routes(feats, max_routes=1)[0]["steps"])
        return {"text": json.dumps(steps), "prompt_tokens": 100, "completion_tokens": 20, "finish_reason": "stop"}
    llm_client.set_transport(v1_or_choice)
    try:
        parts = dev[:N_RUN]
        res = {}
        for system in ("n1_r0_s1+cg1", "n3_r0_s3+cg1", "n3_r1_s3+cg1+fz2", "b1"):
            res[system] = harness.run_experiment(system, parts, repeats=2, tag="cg", out_dir=out, dataset_name="dev_v2", progress=None, analysis=True)
    finally:
        llm_client.reset_transport()
        mp.undo()
    return out, res


def records_of(cg_runs, system):
    return cg_runs[1][system]["records"]


def test_a_run_is_valid_by_construction_and_says_so(cg_runs):
    for system in ("n1_r0_s1+cg1", "n3_r0_s3+cg1"):
        s = cg_runs[1][system]["summary"]
        assert s["M1"]["strict_valid_of_parsed"] == 1.0 and s["M1"]["v1_valid_but_strict_invalid"] == 0
        assert s["M3"]["invalid_raw_routes"] == 0 and s["M3"]["kinds_among_invalid"] == {}
        for rec in records_of(cg_runs, system):
            assert rec["info"].get("repaired", 0) == 0 and rec["info"].get("replaced", 0) == 0
            assert all(c["provenance"] == "raw_llm" for c in rec["candidates"] if c["sample_id"] >= 0)


def test_run_json_records_the_option(cg_runs):
    info = json.load(open(os.path.join(cg_runs[0], "cg_n1_r0_s1+cg1", "run.json"), encoding="utf-8"))
    assert info["constrained"] == 1 and info["planner_options"] == "cg1" and info["max_tokens"] == config.CG_MAX_TOKENS
    assert info["two_step"] is False and info["few_shot_k"] == 0 and info["hint"] is False
    plain = json.load(open(os.path.join(cg_runs[0], "cg_b1", "run.json"), encoding="utf-8"))
    assert plain["constrained"] == 0 and plain["max_tokens"] == config.PLANNER_MAX_TOKENS


def test_the_summary_block_agrees_with_the_records(cg_runs):
    for system in ("n1_r0_s1+cg1", "n3_r0_s3+cg1"):
        recs = records_of(cg_runs, system)
        blocks = [c["meta"]["cg"] for r in recs for c in r["candidates"] if c["meta"].get("cg")]
        b = cg_runs[1][system]["summary"]["constrained"]
        assert b["n"] == len(blocks) and b["n"] == sum(1 for r in recs for c in r["candidates"] if c["sample_id"] >= 0)
        assert b["choice_calls_per_sample_mean"] == pytest.approx(sum(x["choice_steps"] for x in blocks) / len(blocks))
        assert b["forced_steps_per_sample_mean"] == pytest.approx(sum(x["forced_steps"] for x in blocks) / len(blocks))
        assert b["ops_per_route_mean"] == pytest.approx(sum(x["n_ops"] for x in blocks) / len(blocks))
        assert b["fallback_steps"] == 0 and b["fallback_share_of_choices"] == 0.0 and b["samples_with_unsatisfiable_rule"] == sum(1 for x in blocks if x["unsatisfiable"])
        assert b["samples_with_graph_set"] == 0                                    # every dev part has a Route Builder route
        assert b["forced_share_of_steps"] == pytest.approx(sum(x["forced_steps"] for x in blocks) / sum(x["n_ops"] for x in blocks))
    assert cg_runs[1]["b1"]["summary"]["constrained"] == {"n": 0}


def test_m7_counts_the_choice_calls(cg_runs):
    s = cg_runs[1]["n1_r0_s1+cg1"]["summary"]
    assert s["M7"]["api_calls_per_part"] == pytest.approx(s["constrained"]["choice_calls_per_sample_mean"])      # one sample per part
    s3 = cg_runs[1]["n3_r0_s3+cg1"]["summary"]
    assert s3["M7"]["api_calls_per_part"] == pytest.approx(3 * s3["constrained"]["choice_calls_per_sample_mean"])
    assert tables.api_calls(records_of(cg_runs, "n3_r0_s3+cg1")[0]) == sum(
        c["meta"]["n_calls"] for c in records_of(cg_runs, "n3_r0_s3+cg1")[0]["candidates"])
    cost = s["cost"]
    assert cost["calls_per_part"] == pytest.approx(s["M7"]["api_calls_per_part"])                      # the cost block counts the same calls
    assert cost["tokens_per_part"] == pytest.approx(103 * s["M7"]["api_calls_per_part"])               # the fake model: 100 tokens in, 3 out


def test_two_calls_are_not_mistaken_for_a_two_step_sample(cg_runs):
    recs = records_of(cg_runs, "n3_r0_s3+cg1")
    two = [c for r in recs for c in r["candidates"] if c["meta"].get("n_calls") == 2]
    assert two, "the fake run should have routes with exactly two choice calls"
    assert cg_runs[1]["n3_r0_s3+cg1"]["summary"]["two_step"] == {"n": 0}


def test_the_text_summary_and_the_comparison_show_it(cg_runs):
    text = S.format_summary("n1_r0_s1+cg1", cg_runs[1]["n1_r0_s1+cg1"]["summary"])
    line = next(x for x in text.splitlines() if x.startswith("constrained"))
    assert "choice calls" in line and "forced steps per route" in line and "unreadable answers" in line
    assert not any(x.startswith("constrained") for x in S.format_summary("b1", cg_runs[1]["b1"]["summary"]).splitlines())
    table = compare.table([("cg", cg_runs[1]["n1_r0_s1+cg1"]["summary"]), ("b1", cg_runs[1]["b1"]["summary"])])
    row = next(x for x in table.splitlines() if x.startswith("constrained: choice calls / route"))
    assert row.split()[-1] == "-" and "-" in row and row.split()[-2] != "-"
    assert "constrained: unreadable answers (share of choices)" in table


def test_selection_works_on_constrained_samples(cg_runs):
    for rec in records_of(cg_runs, "n3_r0_s3+cg1"):
        assert rec["final"]["check"]["valid"] and rec["info"].get("selection") == "s3"
        assert rec["final"]["provenance"] == "raw_llm"


def test_repair_after_a_constrained_sample_can_add_what_the_planner_could_not(cg_runs):
    """The report's limitation: the planner cannot invent an operation. Feasibility-aware repair (+fz2) can, afterwards, and the record
    shows it: a part whose hole tolerance needs a Reaming has none in the constrained route and has one in the repaired final."""
    plain = {(r["part_id"], r["repeat"]): r for r in records_of(cg_runs, "n3_r0_s3+cg1")}
    fz = records_of(cg_runs, "n3_r1_s3+cg1+fz2")
    added = [r for r in fz if "Reaming" in V.clean_steps(r["final"]["steps"]) and "Reaming" not in V.clean_steps(plain[(r["part_id"], r["repeat"])]["final"]["steps"])]
    assert added, "no part in the fake run needed a Reaming"
    for r in added:
        assert r["final"]["provenance"] == "repaired_llm" and r["final"]["check"]["valid"]


def test_the_table_regenerator_knows_the_option(cg_runs):
    assert tables._cache_key("n5_r0_s3+cg1")[0] != tables._cache_key("n5_r0_s3")[0]
    assert tables._cache_key("n5_r0_s3+cg1") == tables._cache_key("n5_r1_s3+cg1+fz2")
    tables.validate_config({"name": "t", "dataset": "dev_v2", "limit": 5, "repeats": 1, "results_dir": cg_runs[0], "shared_tag": "x",
                            "models": [{"id": "a", "label": "A", "model": "qwen/qwen3.8-27b", "tag": "cg"}],
                            "rows": [{"id": "A0", "label": "b1", "system": "b1"},
                                     {"id": "A9", "label": "constrained", "system": "n1_r0_s1+cg1", "parent": "A0"}]})


def test_case_studies_run_on_a_constrained_run(cg_runs):
    run = casestudy.load_run(os.path.join(cg_runs[0], "cg_n1_r0_s1+cg1"))
    res = casestudy.analyse([run], per_type=3)
    assert res["runs"][0]["n_parsed_routes"] > 0
    rows = {r["type"]: r for r in res["runs"][0]["table"]}
    for t in ("order", "missing", "invented", "duplicate"):
        assert rows[t]["routes"] == 0                                              # valid by construction: none of the four types can occur
    assert "No case" in casestudy.render_markdown(res)
