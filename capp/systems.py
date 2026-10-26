"""The three reference systems of Table 2 (RB, B1, B0), run headless and recorded candidate by candidate.

The logic of B0/B1 follows v1 exactly (week4/nsga2.py "llm_first" + the selection step of week6/app.py):
  1. sample the LLM n times (sample_id = 0..n-1), T = 0.3
  2. a sample that is v1-valid (precedence rules + feature coverage) is kept as is           -> provenance raw_llm
  3. an invalid sample is replaced in its entirety by the Route Builder's first route         -> provenance route_builder
  4. duplicates removed; if nothing is left, the Route Builder's own routes are used (safety net)
  5. non-dominated sort on (time, cost, energy) with v1's default dimensions; the member of the first front with the
     highest Efficiency (scored with the part's real dimensions) is the final route
With repair="r1" (systems b0_r1 and b1_r1) step 3 changes: a sample with a HARD violation under the strict verifier (this
includes invented and duplicate operations, which v1 lets through) is repaired by R1 (capp/repair.py, minimal edit, keeps
the LLM's work) instead of being replaced. Provenance repaired_llm. If R1 finds no valid route, R0 (the replacement above)
is the fallback, so the validity guarantee is the same as in v1. Everything else (dedupe, safety net, selection) is unchanged,
so the difference between b0 and b0_r1 is the repair strategy alone.
What is new is only the bookkeeping: provenance, the first-pass check of every raw LLM route with the strict verifier
(capp/verify.py), and the scores with the part's dimensions. tests/test_systems.py checks that B0 returns the same route
as run_nsga2 + the UI selection on the same LLM outputs.

Note on provenance: v1 names a replaced sample "Route_LLM_n_corrected", but the route is the Route Builder's work.
Here it is labelled route_builder (replaced=True).
"""
from dataclasses import asdict, dataclass, field

from capp import legacy  # noqa: F401
from capp import repair as R
from capp import verify as V

import agents
import fsm_validator
import nsga2 as v1_nsga2
import route_builder as rb

DEFAULT_DIMS = {"diameter": 25.0, "length": 50.0, "depth": 20.0, "width": 20.0}   # what v1's NSGA-II scores with
INCOMPLETE_FALLBACK = ["Facing", "Inspection"]                                    # v1's last resort, covers no feature

class FatalRunError(RuntimeError):
    """Raised by a planner wrapper to stop the whole run (missing key, cache miss in replay mode, long outage).
    run_llm_first must not swallow it the way v1 swallows ordinary planner exceptions."""


PROV_RAW_LLM = "raw_llm"
PROV_REPAIRED = "repaired_llm"      # produced from Step 6 on (localized repair)
PROV_RB = "route_builder"


@dataclass
class Candidate:
    sample_id: int                    # LLM sample index; -1 for Route Builder routes
    name: str                         # v1 style route name
    provenance: str
    steps: list                       # the route as it enters selection (markers possible)
    llm_ok: bool = False              # the LLM call succeeded and a route could be parsed
    llm_steps: list = None            # raw LLM route
    llm_raw: str = ""
    replaced: bool = False            # whole-route replacement by the Route Builder happened (R0)
    repaired: bool = False            # localized repair of the LLM route happened (R1)
    repair: dict = None               # details of the repair attempt: strategy, ok, reason, inserted, removed_*, discordant_pairs
    in_pool: bool = True              # entered selection (False: LLM failed, or an exact duplicate of an earlier route)
    deduped: bool = False             # dropped as a duplicate (its first-pass result still counts for M1)
    first_pass: dict = None           # strict check of the raw LLM route
    v1_valid: bool = None             # v1's decision on the raw LLM route
    scores: dict = None               # v1 scoring of `steps` as they are (used for selection): time_min, cost_inr, energy_kwh, efficiency_score
    scores_eval: dict = None          # same, after verify.normalize_markers (fair: every machine switch is charged); used for M6
    meta: dict = field(default_factory=dict)   # finish_reason, tokens, latency, cached


@dataclass
class RunRecord:
    system: str
    part_id: str
    candidates: list
    final: dict                       # name, steps, provenance, scores, strict check
    front: list                       # names of the first front
    info: dict                        # counts and flags
    references: list                  # Route Builder variants (list of step lists), proxy reference for M4
    best_scores: list                 # fair (marker-normalised) scores of all Route Builder variants, the reference for M6

    def to_dict(self):
        return asdict(self)


def part_inputs(part: dict) -> dict:
    return {
        "material": part["material"], "features": list(part["features"]),
        "tolerance": part.get("tolerance", "0.05mm"), "batch_size": part.get("batch_size", 100),
        "dims": part.get("dims") or DEFAULT_DIMS, "pref": part.get("machine_preference", "auto"),
    }


def score_route(steps, material, batch_size, dims) -> dict:
    r = agents.evaluate_route("r", steps, material, batch_size, dims)
    return {k: r[k] for k in ("time_min", "cost_inr", "energy_kwh", "efficiency_score")}


def check_to_dict(c: V.Check) -> dict:
    return {"valid": c.valid, "v1_valid": c.v1_valid, "kinds": c.kinds, "changeovers": c.changeovers,
            "violations": [{"kind": v.kind, "message": v.message, "ops": list(v.ops), "feature": v.feature,
                            "rule_id": v.rule_id, "positions": list(v.positions)} for v in c.violations]}


def _rb_routes(features, pref, max_routes):
    """Route Builder routes, falling back to 'auto' if the preference leaves none (as v1 does)."""
    routes = rb.generate_valid_routes(features, max_routes=max_routes, machine_preference=pref)
    overridden = False
    if not routes and pref != "auto":
        routes = rb.generate_valid_routes(features, max_routes=max_routes, machine_preference="auto")
        overridden = bool(routes)
    return routes, overridden


def eval_score(steps, p) -> dict:
    return score_route(V.normalize_markers(steps), p["material"], p["batch_size"], p["dims"])


def _finish_candidates(cands, p):
    for c in cands:
        c.scores = score_route(c.steps, p["material"], p["batch_size"], p["dims"])
        c.scores_eval = eval_score(c.steps, p)
    return cands


class _Ind:
    """Same interface as week4.nsga2.Individual (objectives, rank), but the dimensions are a parameter."""
    def __init__(self, cand, p, dims):
        self.cand = cand
        self.route_name = cand.name
        self.steps = cand.steps
        self.rank = 0
        self.crowding = 0.0
        self.objectives = (agents.time_agent(cand.steps, p["material"], dims),
                           agents.cost_agent(cand.steps, p["material"], p["batch_size"], dims),
                           agents.energy_agent(cand.steps, p["material"], dims))


def _select(cands, p, front_dims):
    """v1: first non-dominated front, then highest Efficiency (strict >, so the first of equals wins)."""
    pop = [_Ind(c, p, front_dims) for c in cands]
    fronts = v1_nsga2.non_dominated_sort(pop)
    front = [pop[i] for i in fronts[0]]
    best, best_eff = None, -1
    for ind in front:
        if ind.cand.scores["efficiency_score"] > best_eff:
            best, best_eff = ind, ind.cand.scores["efficiency_score"]
    return best.cand, [i.route_name for i in front]


def _final(cand, features):
    chk = V.check_route(cand.steps, features)
    return {"name": cand.name, "steps": cand.steps, "provenance": cand.provenance, "scores": cand.scores,
            "scores_eval": cand.scores_eval, "check": check_to_dict(chk), "replaced": cand.replaced,
            "repaired": cand.repaired}


def _references(features, p, max_routes=15):
    routes = rb.generate_valid_routes(features, max_routes=max_routes, machine_preference="auto")
    steps = [r["steps"] for r in routes]
    scores = [eval_score(s, p) for s in steps]
    return steps, scores


# ---------------------------------------------------------------------------------------------------------
def run_rb(part: dict, max_routes: int = 15, **_) -> RunRecord:
    """Route Builder alone. Final route = its first route (what v1 uses whenever it needs one route)."""
    p = part_inputs(part)
    routes, overridden = _rb_routes(p["features"], p["pref"], max_routes)
    incomplete = not routes
    if incomplete:
        routes = [{"steps": list(INCOMPLETE_FALLBACK)}]
    cands = [Candidate(-1, f"Route_{i + 1}", PROV_RB, r["steps"]) for i, r in enumerate(routes)]
    _finish_candidates(cands, p)
    refs, ref_scores = _references(p["features"], p)
    return RunRecord("rb", part["part_id"], cands, _final(cands[0], p["features"]), [], {
        "llm_calls": 0, "attempts": 0, "llm_failed": 0, "raw_v1_valid": 0, "replaced": 0, "replacement_failed": 0,
        "repaired": 0, "r1_failed": 0, "safety_net": False, "fallback_incomplete": incomplete, "machine_pref_overridden": overridden,
        "population": len(cands), "deduped": 0}, refs, ref_scores)


def run_llm_first(part: dict, n_samples: int = 5, planner=None, max_routes: int = 30,
                  front_uses_dims: bool = False, system: str = None, repair: str = "r0") -> RunRecord:
    """B0 (n_samples=5) and B1 (n_samples=1). `planner(material, features, tolerance, batch_size, sample_id=)`
    defaults to week2's generate_process_plan, looked up at call time so tests can substitute it.
    repair: "r0" (v1: replace an invalid sample by the Route Builder's route) or "r1" (minimal-edit repair, R0 as fallback)."""
    if repair not in ("r0", "r1"):
        raise ValueError(f"unknown repair strategy {repair!r}")
    if planner is None:
        import llm_planner
        planner = llm_planner.generate_process_plan
    p = part_inputs(part)
    features = p["features"]
    info = {"llm_calls": 0, "attempts": 0, "llm_failed": 0, "raw_v1_valid": 0, "raw_strict_valid": 0,
            "replaced": 0, "replacement_failed": 0, "repaired": 0, "r1_failed": 0, "safety_net": False,
            "fallback_incomplete": False, "machine_pref_overridden": False, "population": 0, "deduped": 0}
    pop = []

    for attempt in range(n_samples):
        info["attempts"] += 1
        info["llm_calls"] += 1
        try:
            res = planner(p["material"], features, p["tolerance"], p["batch_size"], sample_id=attempt)
        except FatalRunError:
            raise
        except Exception as exc:                                   # v1 swallows these as "failed"
            res = {"success": False, "steps": [], "raw_response": f"Exception: {exc}"}
        if not res["success"]:
            info["llm_failed"] += 1
            pop.append(Candidate(attempt, f"Route_LLM_{attempt + 1}_failed", PROV_RB, [], llm_ok=False,
                                 llm_raw=res.get("raw_response", ""), meta=res.get("meta", {})))
            continue

        steps = res["steps"]
        chk = V.check_route(steps, features)
        v1_ok = fsm_validator.validate_sequence(steps)["valid"] and rb.is_complete(steps, features)
        base = dict(sample_id=attempt, llm_ok=True, llm_steps=list(steps), llm_raw=res.get("raw_response", ""),
                    first_pass=check_to_dict(chk), v1_valid=v1_ok, meta=res.get("meta", {}))
        info["raw_v1_valid"] += int(v1_ok)
        info["raw_strict_valid"] += int(chk.valid)
        needs_repair = (not chk.valid) if repair == "r1" else (not v1_ok)    # r1 repairs what the strict verifier rejects
        if not needs_repair:
            pop.append(Candidate(name=f"Route_LLM_{attempt + 1}", provenance=PROV_RAW_LLM, steps=steps, **base))
        else:
            attempt_info = None
            if repair == "r1":
                rep1 = R.r1(steps, features)
                attempt_info = {k: v for k, v in rep1.to_dict().items() if k not in ("steps", "route")}
                if rep1.ok:
                    info["repaired"] += 1
                    pop.append(Candidate(name=f"Route_LLM_{attempt + 1}_repaired", provenance=PROV_REPAIRED,
                                         steps=rep1.route, repaired=True, repair=attempt_info, **base))
                    continue
                info["r1_failed"] += 1
                attempt_info["fallback"] = "r0"
            corrected = fsm_validator.fix_sequence_with_builder(features)
            if (corrected and fsm_validator.validate_sequence(corrected)["valid"]
                    and rb.is_complete(corrected, features)):
                info["replaced"] += 1
                pop.append(Candidate(name=f"Route_LLM_{attempt + 1}_corrected", provenance=PROV_RB,
                                     steps=corrected, replaced=True, repair=attempt_info, **base))
            else:                                      # v1 replacement uses max_routes=1 and can find nothing (e.g. Valve Body)
                info["llm_failed"] += 1
                info["replacement_failed"] += 1
                pop.append(Candidate(attempt, f"Route_LLM_{attempt + 1}_failed", PROV_RB, [], llm_ok=True,
                                     llm_steps=list(steps), llm_raw=res.get("raw_response", ""),
                                     first_pass=base["first_pass"], v1_valid=False, meta=res.get("meta", {}),
                                     repair=attempt_info))

    seen, usable = set(), []
    for c in pop:                                      # v1 removes duplicate routes, keeping the first
        if not c.steps:
            c.in_pool = False
            continue
        key = tuple(c.steps)
        if key in seen:
            c.in_pool, c.deduped = False, True
            info["deduped"] += 1
            continue
        seen.add(key)
        usable.append(c)

    if not usable:                                     # safety net
        info["safety_net"] = True
        routes, overridden = _rb_routes(features, p["pref"], max(3, max_routes))
        info["machine_pref_overridden"] = overridden
        if not routes:
            routes = [{"steps": list(INCOMPLETE_FALLBACK)}]
            info["fallback_incomplete"] = True
        usable = [Candidate(-1, f"Route_Builder_Fallback_{i + 1}", PROV_RB, r["steps"]) for i, r in enumerate(routes)]
        pop.extend(usable)

    info["population"] = len(usable)
    all_cands = pop
    _finish_candidates([c for c in all_cands if c.steps], p)
    best, front_names = _select(usable, p, p["dims"] if front_uses_dims else DEFAULT_DIMS)
    refs, ref_scores = _references(features, p)
    return RunRecord(system or f"b{n_samples}", part["part_id"], all_cands, _final(best, features), front_names,
                     info, refs, ref_scores)


def run_b0(part, **kw):
    return run_llm_first(part, n_samples=5, system="b0", **kw)


def run_b1(part, **kw):
    return run_llm_first(part, n_samples=1, system="b1", **kw)


def run_b0_r1(part, **kw):
    return run_llm_first(part, n_samples=5, system="b0_r1", repair="r1", **kw)


def run_b1_r1(part, **kw):
    return run_llm_first(part, n_samples=1, system="b1_r1", repair="r1", **kw)


SYSTEMS = {"rb": run_rb, "b1": run_b1, "b0": run_b0, "b1_r1": run_b1_r1, "b0_r1": run_b0_r1}
LLM_SYSTEMS = ("b1", "b0", "b1_r1", "b0_r1")
