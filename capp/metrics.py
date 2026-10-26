"""Route-level metrics for the evaluation harness (pure functions, no I/O).

M4 plan similarity     exact match, normalized LCS ratio, Kendall-tau distance on the common operations
M5 repair minimality   edit distance LLM route -> final route, share of LLM operations that survive
M6 plan quality        gap of a route's Time / Cost / Energy / Efficiency to the best known route
Routes are lists of operation names; Machine-Changeover markers are removed before any comparison.
"""
import math
import statistics

from capp.verify import clean_steps


def _c(route):
    return clean_steps(route)


def exact_match(a: list, b: list) -> bool:
    return _c(a) == _c(b)


def lcs_length(a: list, b: list) -> int:
    a, b = _c(a), _c(b)
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def lcs_ratio(a: list, b: list) -> float:
    """LCS length divided by the length of the longer route (1.0 = identical order and content)."""
    la, lb = len(_c(a)), len(_c(b))
    if max(la, lb) == 0:
        return 1.0
    return lcs_length(a, b) / max(la, lb)


def kendall_tau_distance(a: list, b: list):
    """Share of operation pairs, among the operations both routes contain, that are in opposite order.
    0.0 = same relative order, 1.0 = fully reversed. None when fewer than 2 operations are common."""
    a, b = _c(a), _c(b)
    common = [op for op in dict.fromkeys(a) if op in set(b)]
    if len(common) < 2:
        return None
    pos_a = {op: a.index(op) for op in common}
    pos_b = {op: b.index(op) for op in common}
    discordant = total = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            x, y = common[i], common[j]
            total += 1
            if (pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y]) < 0:
                discordant += 1
    return discordant / total


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two operation sequences (insert, delete, substitute = 1 each)."""
    a, b = _c(a), _c(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def survived_share(llm_route: list, final_route: list) -> float:
    """Share of the LLM's operations that are still in the final route in the same relative order (LCS / len(llm))."""
    n = len(_c(llm_route))
    return 1.0 if n == 0 else lcs_length(llm_route, final_route) / n


def similarity(route: list, reference: list) -> dict:
    return {"exact": exact_match(route, reference), "lcs_ratio": lcs_ratio(route, reference),
            "kendall_tau": kendall_tau_distance(route, reference)}


def best_similarity(route: list, references: list):
    """Similarity to the closest of several acceptable references (alternative orderings are all correct).
    exact: any reference matches; lcs_ratio: max; kendall_tau: min (None if undefined for every reference)."""
    if not references:
        return None
    sims = [similarity(route, r) for r in references]
    taus = [s["kendall_tau"] for s in sims if s["kendall_tau"] is not None]
    return {"exact": any(s["exact"] for s in sims), "lcs_ratio": max(s["lcs_ratio"] for s in sims),
            "kendall_tau": min(taus) if taus else None}


def best_known(score_list: list) -> dict:
    """score_list: dicts with time_min, cost_inr, energy_kwh, efficiency_score. Best value of each, over valid routes."""
    if not score_list:
        return {}
    return {"time_min": min(s["time_min"] for s in score_list), "cost_inr": min(s["cost_inr"] for s in score_list),
            "energy_kwh": min(s["energy_kwh"] for s in score_list),
            "efficiency_score": max(s["efficiency_score"] for s in score_list)}


def quality_gap(scores: dict, best: dict) -> dict:
    """Relative excess of time, cost, energy over the best known (0 = as good as the best), and the efficiency shortfall
    in points. Never negative when `best` was computed over a set that contains this route."""
    out = {}
    for k in ("time_min", "cost_inr", "energy_kwh"):
        out[k + "_gap"] = (scores[k] - best[k]) / best[k] if best[k] else 0.0
    out["efficiency_gap"] = best["efficiency_score"] - scores["efficiency_score"]
    return out


def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(x: list, y: list):
    """Spearman rank correlation with average ranks for ties. None if either list is constant or shorter than 2."""
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (dx * dy)


def mean_std(values: list):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def percentile(values: list, q: float):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))]


def rate(flags: list):
    """Share of True in a list of booleans; None when the list is empty."""
    return None if not flags else sum(1 for f in flags if f) / len(flags)


def is_finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)
