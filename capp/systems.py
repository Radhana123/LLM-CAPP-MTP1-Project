"""The three reference systems of Table 2 (RB, B1, B0), run headless and recorded candidate by candidate.

The logic of B0/B1 follows v1 exactly (week4/nsga2.py "llm_first" + the selection step of week6/app.py):
  1. sample the LLM n times (sample_id = 0..n-1), T = 0.3
  2. a sample that is v1-valid (precedence rules + feature coverage) is kept as is           -> provenance raw_llm
  3. an invalid sample is replaced in its entirety by the Route Builder's first route         -> provenance route_builder
  4. duplicates removed; if nothing is left, the Route Builder's own routes are used (safety net)
  5. non-dominated sort on (time, cost, energy) with v1's default dimensions; the member of the first front with the
     highest Efficiency (scored with the part's real dimensions) is the final route
With repair="r1", "r2" or "r3" (systems b0_r1 ... b1_r3) step 3 changes: a sample with a HARD violation under the strict
verifier (this includes invented and duplicate operations, which v1 lets through) is repaired instead of replaced:
  r1  minimal-edit repair by rules (capp/repair.py)
  r2  one LLM re-prompt that carries the violation report (capp/reprompt.py); the answer must pass the strict verifier
  r3  hybrid: r2, and if its answer is still invalid, r1 on the original LLM route
A repaired route has provenance repaired_llm. If every stage fails, R0 (the replacement above) is the last resort, so the
validity guarantee is the same as in v1. Everything else (dedupe, safety net, selection) is unchanged, so the difference
between b0 and b0_r1 / b0_r2 / b0_r3 is the repair strategy alone. Re-prompt calls are extra LLM calls: info["repair_calls"].
What is new is only the bookkeeping: provenance, the first-pass check of every raw LLM route with the strict verifier
(capp/verify.py), and the scores with the part's dimensions. tests/test_systems.py checks that B0 returns the same route
as run_nsga2 + the UI selection on the same LLM outputs.

Selection (report 6.4), the last step, is a parameter too (default s1 = v1):
  s0  first valid candidate (sample order)           s1  v1: first non-dominated front, then highest Efficiency
  s2  first front, then weighted scalarisation of time, cost, energy (user priorities, min-max scaled inside the front)
  s3  first front, then the medoid: the route with the smallest mean Kendall-tau distance to the other samples' routes
      (over the union of the operations, so a different choice of operations counts as disagreement, see route_distance)
The plan confidence is the mean pairwise agreement (1 - route_distance) of the routes of all samples after repair,
duplicates included (a route that three of five samples produced counts three times). Below CONFIDENCE_FLAG the plan is
flagged for human review. The confidence is recorded for every run whatever the selection.

Note on provenance: v1 names a replaced sample "Route_LLM_n_corrected", but the route is the Route Builder's work.
Here it is labelled route_builder (replaced=True).
"""
import re
from dataclasses import asdict, dataclass, field

from capp import legacy  # noqa: F401
from capp import metrics as M
from capp import plannerspec as PS
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
    repair: dict = None               # the repair attempt: strategy, stage (r1|r2 that produced the route), tried, r1 {...}, r2 {...}, fallback
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


SELECTIONS = ("s0", "s1", "s2", "s3")
DEFAULT_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)       # time, cost, energy
CONFIDENCE_FLAG = 0.8                         # plans with lower confidence are flagged for human review (tune on the dev set)


def normalize_weights(weights):
    w = tuple(float(x) for x in (weights if weights is not None else DEFAULT_WEIGHTS))
    if len(w) != 3 or any(x < 0 for x in w) or sum(w) <= 0:
        raise ValueError(f"weights must be three non-negative numbers (time, cost, energy) with a positive sum, got {weights!r}")
    t = sum(w)
    return tuple(x / t for x in w)


def route_distance(a, b, p=0.5):
    """Disagreement of two routes in [0, 1] (markers ignored): Kendall-tau distance over the union of their operations.
    A pair of operations that both routes contain counts 1 if they are in opposite order. A pair in which at least one operation
    is missing from one of the routes cannot be compared and counts p (0.5, neutral; Fagin et al.'s K^(p) for partial rankings).
    For routes with the same operations this is the plain normalised Kendall-tau distance (0 = same order, 1 = reversed); it
    also notices that two routes chose different alternative chains, which the tau on the common operations alone would not."""
    a, b = V.clean_steps(a), V.clean_steps(b)
    pos_a = {}
    pos_b = {}
    for i, op in enumerate(a):
        pos_a.setdefault(op, i)
    for i, op in enumerate(b):
        pos_b.setdefault(op, i)
    union = list(dict.fromkeys(a + b))
    n = len(union)
    if n < 2:
        return 0.0 if set(a) == set(b) else 1.0
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = union[i], union[j]
            if x in pos_a and y in pos_a and x in pos_b and y in pos_b:
                if (pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y]) < 0:
                    pen += 1.0
            else:
                pen += p
    return pen / (n * (n - 1) / 2)


def agreement(routes):
    """Mean pairwise agreement (1 - route_distance) of a list of routes; None for fewer than two."""
    if len(routes) < 2:
        return None
    ds = [route_distance(routes[i], routes[j]) for i in range(len(routes)) for j in range(i + 1, len(routes))]
    return 1.0 - sum(ds) / len(ds)


def _select(cands, p, front_dims, selection="s1", weights=None, sample_routes=()):
    """Returns (selected candidate, names of the first front, detail). The front is always computed (front size is reported);
    s1 is v1's rule: highest Efficiency in the first front (strict >, so the first of equals wins)."""
    pop = [_Ind(c, p, front_dims) for c in cands]
    fronts = v1_nsga2.non_dominated_sort(pop)
    front = [pop[i] for i in fronts[0]]
    names = [i.route_name for i in front]
    detail = {"selection": selection}

    def best_efficiency(members):
        best, best_eff = None, -1
        for ind in members:
            if ind.cand.scores["efficiency_score"] > best_eff:
                best, best_eff = ind, ind.cand.scores["efficiency_score"]
        return best

    if selection == "s1":
        return best_efficiency(front).cand, names, detail
    if selection == "s0":
        for c in cands:
            if V.check_route(c.steps, p["features"]).valid:
                return c, names, detail
        return cands[0], names, dict(detail, no_valid_candidate=True)
    if selection == "s2":
        w = normalize_weights(weights)
        keys = ("time_min", "cost_inr", "energy_kwh")
        lo = [min(i.cand.scores[k] for i in front) for k in keys]
        hi = [max(i.cand.scores[k] for i in front) for k in keys]

        def score(ind):
            return sum(wk * ((ind.cand.scores[k] - l) / (h - l) if h > l else 0.0)
                       for wk, k, l, h in zip(w, keys, lo, hi))
        best, best_s = None, None
        for ind in front:                                   # ties: higher Efficiency, then the first
            sc = score(ind)
            if (best is None or sc < best_s - 1e-12
                    or (abs(sc - best_s) <= 1e-12 and ind.cand.scores["efficiency_score"] > best.cand.scores["efficiency_score"])):
                best, best_s = ind, sc
        return best.cand, names, dict(detail, weights=list(w), scalarised_score=best_s)
    if selection == "s3":
        def mean_distance(ind):
            others = [r for sid, r in sample_routes if sid != ind.cand.sample_id]
            return sum(route_distance(ind.cand.steps, r) for r in others) / len(others) if others else 0.0
        best, best_d = None, None
        for ind in front:                                   # ties: higher Efficiency, then the first
            d = mean_distance(ind)
            if (best is None or d < best_d - 1e-12
                    or (abs(d - best_d) <= 1e-12 and ind.cand.scores["efficiency_score"] > best.cand.scores["efficiency_score"])):
                best, best_d = ind, d
        return best.cand, names, dict(detail, selected_mean_distance=best_d)
    raise ValueError(f"unknown selection {selection!r}")


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


REPAIR_STAGES = {"r0": (), "r1": ("r1",), "r2": ("r2",), "r3": ("r2", "r1")}      # tried in this order, R0 is always last


def _r2_stage(repair_planner, p, steps, chk, attempt, info, features):
    """One re-prompt. Returns (route or None, detail dict)."""
    info["repair_calls"] += 1
    try:
        rr = repair_planner(p["material"], features, p["tolerance"], p["batch_size"], sample_id=attempt,
                            previous_steps=list(steps), violations=chk.hard_violations)
    except FatalRunError:
        raise
    except Exception as exc:
        rr = {"success": False, "steps": [], "raw_response": f"Exception: {exc}"}
    chk2 = V.check_route(rr["steps"], features) if rr.get("success") else None
    ok = bool(chk2 and chk2.valid)
    detail = {"ok": ok, "steps": list(rr.get("steps", [])), "raw_response": rr.get("raw_response", ""),
              "meta": rr.get("meta", {}), "kinds": chk2.kinds if chk2 else [],
              "reason": "" if ok else ("no route in the answer" if not rr.get("success") else
                                       "answer is still invalid: " + ", ".join(chk2.kinds))}
    if ok:
        info["repaired_r2"] += 1
        return V.normalize_markers(rr["steps"]), detail
    info["r2_failed"] += 1
    return None, detail


def run_llm_first(part: dict, n_samples: int = 5, planner=None, max_routes: int = 30,
                  front_uses_dims: bool = False, system: str = None, repair: str = "r0",
                  repair_planner=None, selection: str = "s1", weights=None) -> RunRecord:
    """B0 (n_samples=5) and B1 (n_samples=1). `planner(material, features, tolerance, batch_size, sample_id=)`
    defaults to week2's generate_process_plan, looked up at call time so tests can substitute it.
    repair: "r0" (v1: replace an invalid sample by the Route Builder's route), "r1" (minimal-edit repair), "r2" (LLM re-prompt
    with the violation report) or "r3" (r2, then r1); R0 is the last resort of all of them.
    `repair_planner(material, features, tolerance, batch_size, sample_id=, previous_steps=, violations=)` defaults to
    capp.reprompt.reprompt and is only called by r2 and r3."""
    if repair not in REPAIR_STAGES:
        raise ValueError(f"unknown repair strategy {repair!r}")
    if selection not in SELECTIONS:
        raise ValueError(f"unknown selection {selection!r}")
    if selection == "s2":
        normalize_weights(weights)
    if "r2" in REPAIR_STAGES[repair] and repair_planner is None:
        from capp import reprompt
        repair_planner = reprompt.reprompt
    if planner is None:
        import llm_planner
        planner = llm_planner.generate_process_plan
    p = part_inputs(part)
    features = p["features"]
    info = {"llm_calls": 0, "attempts": 0, "llm_failed": 0, "raw_v1_valid": 0, "raw_strict_valid": 0,
            "replaced": 0, "replacement_failed": 0, "repaired": 0, "repaired_r1": 0, "repaired_r2": 0, "r1_failed": 0,
            "r2_failed": 0, "repair_calls": 0, "safety_net": False,
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
        needs_repair = (not v1_ok) if repair == "r0" else (not chk.valid)    # r1..r3 repair what the strict verifier rejects
        if not needs_repair:
            pop.append(Candidate(name=f"Route_LLM_{attempt + 1}", provenance=PROV_RAW_LLM, steps=steps, **base))
        else:
            attempt_info = None
            if repair != "r0":
                attempt_info = {"strategy": repair, "stage": None, "tried": []}
                routed = None
                for stage in REPAIR_STAGES[repair]:
                    attempt_info["tried"].append(stage)
                    if stage == "r1":
                        rep1 = R.r1(steps, features)
                        attempt_info["r1"] = {k: v for k, v in rep1.to_dict().items() if k not in ("steps", "route")}
                        if rep1.ok:
                            info["repaired_r1"] += 1
                            routed = rep1.route
                        else:
                            info["r1_failed"] += 1
                    else:
                        routed, detail = _r2_stage(repair_planner, p, steps, chk, attempt, info, features)
                        attempt_info["r2"] = detail
                    if routed:
                        attempt_info["stage"] = stage
                        break
                if routed:
                    info["repaired"] += 1
                    pop.append(Candidate(name=f"Route_LLM_{attempt + 1}_repaired", provenance=PROV_REPAIRED,
                                         steps=routed, repaired=True, repair=attempt_info, **base))
                    continue
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
    sample_routes = [(c.sample_id, c.steps) for c in all_cands if c.llm_ok and c.steps and c.sample_id >= 0]
    best, front_names, detail = _select(usable, p, p["dims"] if front_uses_dims else DEFAULT_DIMS, selection, weights,
                                        sample_routes)
    conf = agreement([r for _, r in sample_routes])
    info.update({"selection": selection, "confidence": conf, "flagged": conf is not None and conf < CONFIDENCE_FLAG,
                 "confidence_routes": len(sample_routes)})
    info.update({k: v for k, v in detail.items() if k not in ("selection",)})
    refs, ref_scores = _references(features, p)
    return RunRecord(system or f"b{n_samples}", part["part_id"], all_cands, _final(best, features), front_names,
                     info, refs, ref_scores)


def run_b0(part, **kw):
    return run_llm_first(part, n_samples=5, system="b0", **kw)


def run_b1(part, **kw):
    return run_llm_first(part, n_samples=1, system="b1", **kw)


_LEGACY = {"b0": (5, "r0", "s1"), "b1": (1, "r0", "s1")}
for _n, _label in ((5, "b0"), (1, "b1")):
    for _r in ("r1", "r2", "r3"):
        _LEGACY[f"{_label}_{_r}"] = (_n, _r, "s1")
_GENERIC = re.compile(r"^n(\d{1,2})_r([0-3])_s([0-3])$")
MAX_SAMPLES = 20


def planner_spec(name: str) -> PS.PlannerSpec:
    """The planner options of a system name: the part after '+' (fs2, fs4x, h1 ...). Default = the v1 planner."""
    return PS.split_name(name)[1]


def parse_system(name: str):
    """'rb' -> None. Otherwise (n_samples, repair, selection). Legacy names: b0 (5 samples, v1), b1 (1 sample), and
    b0_r1 ... b1_r3 (same with a repair strategy). Generic names: n<N>_r<0-3>_s<0-3>, e.g. n8_r1_s3 = 8 samples,
    R1 repair, selection S3 (n5_r0_s1 is b0, n1_r0_s1 is b1). Any LLM system may end with planner options, e.g.
    n5_r1_s3+fs2+h1 (see capp/plannerspec.py); they do not change the tuple, see planner_spec()."""
    base, _ = PS.split_name(name)                      # raises ValueError for an unknown option
    if base == "rb":
        if "+" in name:
            raise ValueError("rb has no planner, so it takes no +fs/+h options")
        return None
    name = base
    if name in _LEGACY:
        return _LEGACY[name]
    m = _GENERIC.match(name)
    if not m or not 1 <= int(m.group(1)) <= MAX_SAMPLES:
        raise ValueError(f"unknown system {name!r}: use rb, b0, b1, b0_r1..b1_r3 or n<1-{MAX_SAMPLES}>_r<0-3>_s<0-3> (e.g. n5_r1_s3), "
                         f"optionally followed by +fs2 / +fs4 / +h1")
    return int(m.group(1)), "r" + m.group(2), "s" + m.group(3)


def make_system(name: str):
    spec = parse_system(name)
    if spec is None:
        return run_rb
    n_samples, repair, selection = spec

    def run(part, **kw):
        return run_llm_first(part, n_samples=n_samples, system=name, repair=repair, selection=selection, **kw)
    run.__name__ = "run_" + name
    return run


def resolve(name: str):
    """The run function of a system name (legacy or generic, with or without planner options)."""
    return SYSTEMS[name] if name in SYSTEMS else make_system(name)


def is_llm(name: str) -> bool:
    return parse_system(name) is not None


def reprompts(name: str) -> bool:
    """True if the system makes extra LLM calls for repair (R2, R3)."""
    spec = parse_system(name)
    return spec is not None and "r2" in REPAIR_STAGES[spec[1]]


run_b0_r1, run_b1_r1 = make_system("b0_r1"), make_system("b1_r1")
run_b0_r2, run_b1_r2 = make_system("b0_r2"), make_system("b1_r2")
run_b0_r3, run_b1_r3 = make_system("b0_r3"), make_system("b1_r3")

SYSTEMS = {"rb": run_rb, "b1": run_b1, "b0": run_b0, "b1_r1": run_b1_r1, "b0_r1": run_b0_r1,
           "b1_r2": run_b1_r2, "b0_r2": run_b0_r2, "b1_r3": run_b1_r3, "b0_r3": run_b0_r3}
LLM_SYSTEMS = tuple(k for k in SYSTEMS if k != "rb")
REPROMPT_SYSTEMS = tuple(k for k in SYSTEMS if reprompts(k))      # legacy systems that make extra LLM calls for repair
