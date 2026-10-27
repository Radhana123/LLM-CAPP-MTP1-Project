"""Planner variants of report 6.2 that need no extra LLM call: retrieval-based few-shot prompting and the Route Builder hint.

    pool = planners.Pool.from_dataset(dataset.load_dataset("dev_v2"), name="dev_v2")
    factory = planners.make_factory(plannerspec.PlannerSpec(few_shot=2, hint=True), pool)
    planner = factory(part)                      # same call signature as llm_planner.generate_process_plan
    result = planner(material, features, tolerance, batch_size, sample_id=0)

Retrieval (variant planning, [25]): the k parts of the pool most similar to the query part, by Jaccard overlap of the feature
sets, ties broken by material (same material, then same class, then any). Every distinct feature set is one candidate, so k
examples are k different problems, not k copies of the same one; a candidate whose plan equals that of a more similar one is
dropped. The query part itself is never an example (leave-one-out by part id); with loo="features" no part with the same feature
set is either, which simulates a combination that the pool has not seen. The "validated plan" of a pool part is its expert
reference route if the part has one (WP2), otherwise the Route Builder's first route (valid by construction, strict-checked).
Examples are drawn from the pool only, and the pool is the development set, never the test benchmark.

(With the option +ts the same examples and hint go into the reasoning prompt of the two-step planner, capp/twostep.py, as text.)

Prompt: the v1 system prompt, then one user/assistant pair per example (least similar first, so the most similar example is
the closest to the question), then the question. With no example and no hint the messages are exactly v1's, so the cache holds
their answers already. The hint (H1) is one more paragraph in the user message; the repair prompts of R2 do not carry examples
or hint.
"""
import json
from dataclasses import dataclass

from capp import legacy  # noqa: F401
from capp import config, llm_client, systems
from capp import verify as V
from capp.reprompt import _original_user_prompt
from capp.plannerspec import PlannerSpec

import llm_planner

MATERIAL_CLASS = {"Steel": "ferrous", "Cast Iron": "ferrous", "Aluminum": "non-ferrous", "Brass": "non-ferrous",
                  "Copper": "non-ferrous", "Titanium": "titanium", "Plastic": "polymer"}

HINT_TEXT = ("A rule-based planner proposes this route:\n{route}\n"
             "Keep it where it is sound and change it where you see a better plan.\n\n")


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def material_rank(query: str, example: str) -> int:
    """0: same material, 1: same class, 2: anything else (smaller is closer)."""
    if query == example:
        return 0
    return 1 if MATERIAL_CLASS.get(query) is not None and MATERIAL_CLASS.get(query) == MATERIAL_CLASS.get(example) else 2


def rb_plan(features, machine_preference="auto"):
    """The Route Builder's first route for these features, without changeover markers: the route of the RB baseline."""
    routes, _ = systems._rb_routes(list(features), machine_preference, 15)
    return V.clean_steps(routes[0]["steps"]) if routes else None


@dataclass(frozen=True)
class Example:
    part_id: str
    material: str
    features: tuple
    tolerance: str
    batch_size: int
    plan: tuple


class Pool:
    def __init__(self, examples, name=""):
        self.examples = list(examples)
        self.name = name

    def __len__(self):
        return len(self.examples)

    @classmethod
    def from_dataset(cls, parts, name=""):
        out = []
        for part in parts:
            p = systems.part_inputs(part)
            refs = part.get("reference_routes")
            plan = V.clean_steps(refs[0]) if refs else rb_plan(p["features"], p["pref"])
            if plan and V.check_route(plan, p["features"]).valid:
                out.append(Example(part["part_id"], p["material"], tuple(p["features"]), p["tolerance"], p["batch_size"],
                                   tuple(plan)))
        return cls(out, name)

    def retrieve(self, part: dict, k: int, loo: str = "part") -> list:
        """The k most similar examples, most similar first."""
        if k <= 0:
            return []
        own = frozenset(part["features"])
        groups = {}
        for e in self.examples:
            if e.part_id == part["part_id"] or (loo == "features" and frozenset(e.features) == own):
                continue
            groups.setdefault(frozenset(e.features), []).append(e)
        scored = []
        for feats, members in groups.items():
            rep = min(members, key=lambda e: (material_rank(part["material"], e.material), e.part_id))
            scored.append((-jaccard(own, feats), material_rank(part["material"], rep.material), rep.part_id, rep))
        scored.sort(key=lambda t: t[:3])
        out, plans = [], set()
        for _, _, _, rep in scored:
            if rep.plan in plans:
                continue
            plans.add(rep.plan)
            out.append(rep)
            if len(out) == k:
                break
        return out


def build_messages(material, features, tolerance, batch_size, examples=(), hint=None) -> list:
    msgs = [{"role": "system", "content": llm_planner.SYSTEM_PROMPT.format(valid_ops=llm_planner._get_valid_operations_list())}]
    for ex in reversed(list(examples)):                       # least similar first
        msgs.append({"role": "user", "content": _original_user_prompt(ex.material, list(ex.features), ex.tolerance, ex.batch_size)})
        msgs.append({"role": "assistant", "content": json.dumps(list(ex.plan))})
    user = _original_user_prompt(material, features, tolerance, batch_size)
    if hint:
        tail = "Output ONLY a JSON array of operation names."
        assert user.endswith(tail)
        user = user[:-len(tail)] + HINT_TEXT.format(route=json.dumps(list(hint))) + tail
    msgs.append({"role": "user", "content": user})
    return msgs


def model_id(role: str) -> str:
    """The model id of a role (primary, small, large). The primary model is llm_planner.MODEL, as in v1, so that its cache keys are v1's."""
    return llm_planner.MODEL if role == "primary" else config.model(role)


def make_factory(spec: PlannerSpec, pool: Pool = None):
    """factory(part, role=None) -> planner. The planner has the call signature of llm_planner.generate_process_plan and returns the
    same dict; its meta also carries prompt_context (ids of the examples and whether a hint was given) and the model id. With
    spec.two_step the planner is capp.twostep's (two calls per sample); examples and hint then go into its reasoning prompt.
    `role` (primary, small or large) picks the model; None means the model of the spec (+ms, +ml, default primary). The model
    cascade asks for the small and the large planner of the same part: same prompts, other model, other cache keys."""
    if spec.few_shot and pool is None:
        raise ValueError("a few-shot planner needs a pool (planners.Pool.from_dataset(...))")

    def factory(part: dict, role: str = None):
        mid = model_id(role or spec.model)
        examples = pool.retrieve(part, spec.few_shot, spec.loo) if spec.few_shot else []
        hint = rb_plan(part["features"], part.get("machine_preference", "auto")) if spec.hint else None
        context = {"examples": [e.part_id for e in examples], "hint": hint is not None and spec.hint}
        if spec.two_step:                               # reason, then formalise: its own prompts (capp/twostep.py)
            from capp import twostep
            return twostep.make_planner(examples, hint, context, model=mid)

        def planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
            model = mid                                        # the primary model is v1's, so that fs0/h0 shares its cache keys
            try:
                resp = llm_client.chat(build_messages(material, features, tolerance, batch_size, examples, hint),
                                       model=model, sample_id=sample_id, temperature=config.PLANNER_TEMPERATURE,
                                       max_tokens=config.PLANNER_MAX_TOKENS)
                raw = resp.text.strip()
                steps = llm_planner._parse_steps(raw)
                meta = {"cached": resp.cached, "finish_reason": resp.finish_reason, "prompt_tokens": resp.prompt_tokens,
                        "completion_tokens": resp.completion_tokens, "latency_s": resp.latency_s, "prompt_context": context, "model": model}
                return {"success": len(steps) > 0, "steps": steps, "raw_response": raw, "model": model, "meta": meta}
            except llm_client.NoApiKey:
                return {"success": False, "steps": [], "raw_response": "GROQ_API_KEY not configured", "model": model}
            except Exception as e:
                return {"success": False, "steps": [], "raw_response": f"API Error: {str(e)}", "model": model}
        return planner
    return factory
