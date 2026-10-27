"""Constrained (graph-guided) planner (report 6.8, row A9, answer to RQ5). Option `+cg1` of a system name.

    python scripts/run_eval.py --system n1_r0_s1+cg1 ...       one constrained route per part, no repair needed
    python scripts/run_eval.py --system n5_r0_s3+cg1 ...       five constrained routes, consistency-aware selection

RQ5: is it better to repair an LLM plan after generation, or to constrain generation step by step with the precedence graph?
Here the route is built one operation at a time:

  1. The operation SET is fixed before the LLM is asked anything: the operations of the Route Builder's first route for these
     features (the same operations as the RB baseline, so a difference in the plan is a difference in ORDER; the rare feature sets
     for which the Route Builder has no route get the first alternative chain of every feature, meta.cg.set_source == "graph"). The LLM can neither add nor drop one,
     so a route can never be "missing" or "invented" or "duplicate"; if the rule base lacks an operation that the part needs (a
     Reaming for a fine hole tolerance, say) this planner cannot supply it. That is the limitation the report names, and it is
     measured, not hidden: compare the flags and M4 of this row with the generate-then-repair rows.
  2. At every step the LLM is shown ONLY the operations whose prerequisites are already placed (the frontier of the precedence
     graph) and answers with the number of the one that comes next. Facing is placed first and Inspection last without a question.
     A step with a single option is placed without a question as well (a "forced" step, no API call).
  3. So every route is valid by construction for the rules of the graph: Facing first, Inspection last, every precedence edge, the
     feature coverage, no duplicate. What remains open is quality: the LLM can still pick a bad order among valid ones (more
     machine changeovers, a poor sequence), and the physical-feasibility rules the graph does not know.

The physical-feasibility rules that are about ORDER (HOLE-EXISTS: a hole-making operation before every operation that works in a
hole; SEQ-PARTOFF: every lathe operation before Parting-off) need no constraint of their own: the precedence graph implies them for
every operation set the Route Builder produces (tests/test_constrained.py checks this over all single and pair feature combinations
of the vocabulary). The one thing this planner cannot do about them is an operation the SET lacks: the Route Builder's own route for
Groove + Pocket + Chamfer has an Internal Grooving and no Drilling or Boring, which HOLE-EXISTS calls infeasible, and no ordering
helps. Such a gap is recorded per route (meta.cg.unsatisfiable) and counted in the summary.

An answer that cannot be read (no number in range, no unique option name) is not retried: the step takes the option that comes
first in the Route Builder's route, and the step is counted as a fallback step (meta.cg.fallback_steps). A run with many fallback
steps is a run of the Route Builder with a little noise, and the summary says so. An API error stops the sample (success False),
exactly as in the other planners; nothing is quietly replaced.

Cost: one call per step that has more than one option (meta.n_calls), not one per step. On the 200 development parts a route has 6.2
operations on average and only about 2.1 real choices (0 to 6; 12 parts have none: the graph leaves one order), so most steps are forced and
cost nothing. M7 counts the calls like the second call of the two-step planner: prompt and completion tokens and latency are summed over
the calls of a sample. Temperature is the planner temperature, so the samples of a part differ. The options are listed in alphabetical
order, so that position carries no information.

    planner = constrained.make_planner(part)             # signature of llm_planner.generate_process_plan
    result = planner(material, features, tolerance, batch_size, sample_id=0)
    result["steps"], result["meta"]["cg"]                # the route; the choices with the options at each step
"""
import json
import re

from capp import legacy  # noqa: F401
from capp import config, feasibility as F, llm_client, systems
from capp import verify as V
from capp.twostep import strip_think

import llm_planner
import precedence_graph as pg
import route_builder as rb

ANSWER_CHARS = 60                      # how much of an answer is kept in the record

SYSTEM_PROMPT = (
    "You are a manufacturing process planning expert. A machining route is built one operation at a time. The set of operations is "
    "already fixed and the precedence rules are enforced for you: every option you are offered is allowed by them. Your only task "
    "is to choose which option comes NEXT.\n\n"
    "A good route follows sound machining practice (for example roughing before finishing) and keeps the operations of one machine "
    "together, because every switch between the lathe and the milling machine costs a changeover.\n\n"
    "Reply with ONLY the number of your choice.")


def machine_of(op: str) -> str:
    return "lathe" if op in rb.LATHE_ONLY_OPS else "milling" if op in rb.MILLING_ONLY_OPS else "either machine"


# ---------------------------------------------------------------- the constraints
def required_ops(features, machine_preference="auto"):
    """The operations of the Route Builder's route for these features, in the order of that route (the default order of a step that
    cannot be decided), or None if the Route Builder has no route."""
    routes, _ = systems._rb_routes(list(features), machine_preference, 15)
    return list(V.clean_steps(routes[0]["steps"])) if routes else None


def graph_ops(features) -> list:
    """The operation set when the Route Builder has no route (its machine-phase construction fails for about 0.4% of the feature sets
    of 4-5 features, e.g. Turning + Slot + Hole + Chamfer, where a shared operation lands in the lathe phase but must follow a milling
    operation): Facing, the first alternative chain of every feature, Inspection; in the alphabetical topological order of the graph,
    which is also the default order of a step that cannot be decided. ValueError for a feature outside the vocabulary."""
    combos = rb.get_operation_combinations(list(features))
    ops = set(combos[0]) | {pg.ALWAYS_FIRST, pg.ALWAYS_LAST}
    edges, rem, order = edges_for(ops), set(ops), []
    while rem:
        free = frontier(rem, edges)
        if not free:
            raise ValueError("the precedence graph has a cycle among " + ", ".join(sorted(rem)))
        order.append(free[0])
        rem.remove(free[0])
    return order


def edges_for(ops):
    """The precedence edges (a, b) = a before b that apply to the operations `ops`."""
    ops = set(ops)
    return {(a, b) for a, b in pg.PRECEDENCE_EDGES if a in ops and b in ops}


def order_gaps(ops) -> list:
    """Physical order rules that no ordering of `ops` can satisfy because the SET lacks what they need: ['HOLE-EXISTS:<op>'] for an
    operation that works in a hole when the set has no Drilling or Boring."""
    ops = set(ops)
    if any(m in ops for m in F.HOLE_MAKERS):
        return []
    return [f"HOLE-EXISTS:{op}" for op in F.HOLE_DEPENDENT if op in ops]


def physical_order_edges(ops):
    """The edges the two physical ORDER rules would add for `ops` (HOLE-EXISTS: Drilling, else Boring, before each hole-dependent
    operation; SEQ-PARTOFF: each lathe operation before Parting-off). Not used to build routes: the precedence graph implies them, and
    the test suite checks that, so a change to the graph that breaks it is noticed."""
    ops = set(ops)
    edges = set()
    maker = next((m for m in F.HOLE_MAKERS if m in ops), None)
    if maker:
        edges.update((maker, dep) for dep in F.HOLE_DEPENDENT if dep in ops)
    if "Parting-off" in ops:
        edges.update((op, "Parting-off") for op in ops if op in rb.LATHE_ONLY_OPS and op != "Parting-off")
    return edges


def frontier(remaining, edges) -> list:
    """The operations of `remaining` whose prerequisites are all placed, alphabetically. Facing is the only option while it is not
    placed (the rule: Facing is the first step), and Inspection is offered only when it is the last one left."""
    rem = set(remaining)
    if pg.ALWAYS_FIRST in rem:
        return [pg.ALWAYS_FIRST]
    free = sorted(op for op in rem if not any(a in rem and a != op for a, b in edges if b == op))
    if pg.ALWAYS_LAST in free and len(rem) > 1:
        free = [op for op in free if op != pg.ALWAYS_LAST]
    return free


# ---------------------------------------------------------------- the question and the answer
def step_messages(material, features, tolerance, batch_size, machine_preference, placed, remaining, options) -> list:
    last = machine_of(placed[-1]) if placed else "none yet"
    others = [op for op in remaining if op not in options]
    user = (f"Part: {material}; features: {', '.join(features)}; tolerance {tolerance}; batch size {batch_size}; "
            f"machine preference: {machine_preference}.\n"
            f"Route so far: {' > '.join(placed) if placed else '(empty)'}\n"
            f"Machine of the last operation: {last}\n"
            f"Operations still to place after this choice: {', '.join(others) if others else 'none'}\n\n"
            f"Options for the next operation (all are allowed by the precedence rules):\n"
            + "\n".join(f"{i}. {op} ({machine_of(op)})" for i, op in enumerate(options, 1))
            + "\n\nReply with ONLY the number of your choice.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_choice(text: str, options: list):
    """The chosen option, or None. A whole number in range wins (the first one in the answer); otherwise the one option whose name
    appears in the answer, if exactly one does."""
    t = strip_think(text or "")
    for m in re.finditer(r"\d+", t):
        n = int(m.group())
        if 1 <= n <= len(options):
            return options[n - 1]
        break                                           # the first number is the answer; an out-of-range one is not guessed around
    named = [op for op in options if re.search(r"(?<![A-Za-z])" + re.escape(op) + r"(?![A-Za-z])", t, re.IGNORECASE)]
    named = [op for op in named if not any(op != other and op.lower() in other.lower() for other in named)]      # "Center Drilling" is not "Drilling"
    return named[0] if len(named) == 1 else None


# ---------------------------------------------------------------- the planner
def make_planner(part: dict = None, model: str = None):
    """planner(material, features, tolerance, batch_size, sample_id) -> the v1 result dict, with meta.cg. `part` supplies the machine
    preference (the planner signature does not carry it); `model` is the model id of every call (default: the primary model)."""
    pref = (part or {}).get("machine_preference", "auto")
    context = {"examples": [], "hint": False, "constrained": True}

    def planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0, _model=model):
        mid = _model or llm_planner.MODEL
        source = "route_builder"
        try:
            ops = default_order = required_ops(features, pref)
            if not ops:                                   # the Route Builder has no route: the graph alone gives the set
                source = "graph"
                ops = default_order = graph_ops(features)
        except ValueError as exc:                         # a feature outside the vocabulary
            return {"success": False, "steps": [], "raw_response": f"no operation set for these features: {exc}", "model": mid}
        edges, unsat = edges_for(ops), order_gaps(ops)
        remaining, placed, choices, calls = list(ops), [], [], []
        fallback = forced = 0

        def meta(last_finish="stop"):
            return {"cached": all(c["cached"] for c in calls), "finish_reason": last_finish,
                    "prompt_tokens": sum(c["prompt_tokens"] for c in calls), "completion_tokens": sum(c["completion_tokens"] for c in calls),
                    "latency_s": sum(c["latency_s"] for c in calls), "n_calls": len(calls), "calls": calls,
                    "prompt_context": context, "model": mid,
                    "cg": {"n_ops": len(ops), "set_source": source, "choice_steps": len(calls), "forced_steps": forced, "fallback_steps": fallback,
                           "unsatisfiable": unsat, "choices": choices}}

        while remaining:
            options = frontier(remaining, edges)
            if not options:
                return {"success": False, "steps": [], "raw_response": "constraint cycle: " + ", ".join(remaining), "model": mid,
                        "meta": meta()}
            if len(options) == 1:
                op, how, answer = options[0], "forced", ""
                forced += 1
            else:
                try:
                    r = llm_client.chat(step_messages(material, features, tolerance, batch_size, pref, placed, remaining, options),
                                        model=mid, sample_id=sample_id, temperature=config.PLANNER_TEMPERATURE, max_tokens=config.CG_MAX_TOKENS)
                except llm_client.NoApiKey:
                    return {"success": False, "steps": [], "raw_response": "GROQ_API_KEY not configured", "model": mid}
                except Exception as exc:
                    return {"success": False, "steps": [], "raw_response": f"API Error: {exc}", "model": mid,
                            "meta": meta("error") if calls else None}
                calls.append({"role": "choose", "step": len(placed), "n_options": len(options), "cached": r.cached,
                              "finish_reason": r.finish_reason, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                              "latency_s": r.latency_s})
                answer = strip_think(r.text)[:ANSWER_CHARS]
                op = parse_choice(r.text, options)
                how = "llm"
                if op is None:
                    op = min(options, key=lambda o: default_order.index(o))
                    how = "fallback"
                    fallback += 1
            choices.append({"pos": len(placed), "options": options, "choice": op, "how": how, "answer": answer})
            placed.append(op)
            remaining.remove(op)
        finish = calls[-1]["finish_reason"] if calls else "stop"
        return {"success": True, "steps": list(placed), "raw_response": json.dumps(placed), "model": mid, "meta": meta(finish)}
    return planner
