"""Validity against cost: the Pareto comparison of the model cascade (report 6.4, "reported as a Pareto comparison of validity against
expected cost"). Pure functions over run records; scripts/cascade_pareto.py is the command line.

One point per run (a system on a set of parts). Two readings of "validity", both from the final routes:
  llm_route_share   the share of parts whose final route is an LLM route (raw_llm or repaired_llm), i.e. the verifier accepted or
                    repaired what the model said. The final route is always strict-valid whatever the model (the Route Builder, R0, is the
                    last resort), so the share of Route Builder routes is what an unreliable model really costs: the LLM contributed nothing.
  efficiency_gap    mean Efficiency shortfall of the final route to the best Route Builder variant, in points (M6; 0 = as good as the best).
Cost: list-price dollars per part (capp/cost.py), with the tokens and the sum of call latencies next to it. A point is on the
frontier if no other point is at most as expensive and at least as good, and better in one of the two. A point without a price is shown
and kept out of the frontier.
"""
from capp import summary as S


def point(label: str, records: list) -> dict:
    """The row of one run."""
    s = S.summarize_set(records)
    cost, casc = s["cost"], s["cascade"]
    prov = s["M2"]["provenance_share"]
    return {
        "label": label, "n_records": s["n_records"], "n_parts": s["n_parts"],
        "models": [] if not cost["n"] else list(cost["by_model"]),
        "usd_per_part": cost.get("usd_per_part"), "tokens_per_part": cost.get("tokens_per_part"),
        "latency_s_per_part": cost.get("latency_s_per_part"), "api_calls_per_part": s["M7"]["api_calls_per_part"],
        "llm_route_share": 1.0 - prov.get("route_builder", 0.0),
        "raw_strict_valid": s["M1"]["strict_valid_of_parsed"],
        "efficiency_gap": s["M6"]["efficiency_gap_to_rb_best_mean"],
        "m4_lcs": ((s["M4"].get("expert") or s["M4"].get("proxy_rb") or {}).get("final") or {}).get("lcs_ratio_mean"),
        "escalation_rate": casc.get("escalation_rate") if casc["n"] else None,
        "final_from_large": casc.get("parts_with_final_from_large") if casc["n"] else None,
        "unpriced": cost.get("unpriced_models", []),
    }


def same_parts(runs: list) -> bool:
    """True if every run covers the same (part, repeat) pairs: only then are the points comparable."""
    keys = [{(r["part_id"], r["repeat"]) for r in recs} for _, recs in runs]
    return all(k == keys[0] for k in keys)


def frontier(points: list, cost_key: str, value_key: str, higher_is_better: bool) -> set:
    """Labels of the points that no other point dominates (cheaper-or-equal and better-or-equal, strictly better in one)."""
    usable = [p for p in points if p.get(cost_key) is not None and p.get(value_key) is not None]

    def dominates(a, b):
        ca, cb = a[cost_key], b[cost_key]
        va, vb = (a[value_key], b[value_key]) if higher_is_better else (-a[value_key], -b[value_key])
        return ca <= cb + 1e-15 and va >= vb - 1e-12 and (ca < cb - 1e-15 or va > vb + 1e-12)
    return {p["label"] for p in usable if not any(dominates(q, p) for q in usable if q is not p)}


def analyse(runs: list) -> dict:
    """runs: [(label, records)]. Returns {"points", "comparable", "frontier_validity", "frontier_quality", "price_of_validity"}."""
    points = [point(lab, recs) for lab, recs in runs]
    fv = frontier(points, "usd_per_part", "llm_route_share", True)
    fq = frontier(points, "usd_per_part", "efficiency_gap", False)
    for p in points:
        p["on_validity_frontier"] = p["label"] in fv
        p["on_quality_frontier"] = p["label"] in fq
    steps, prev = [], None            # along the validity frontier, cheapest first: what one more percentage point of LLM routes costs
    for p in sorted((q for q in points if q["on_validity_frontier"]), key=lambda q: (q["usd_per_part"], -q["llm_route_share"])):
        if prev is not None and p["llm_route_share"] > prev["llm_route_share"] + 1e-12:
            steps.append({"from": prev["label"], "to": p["label"],
                          "extra_usd_per_part": p["usd_per_part"] - prev["usd_per_part"],
                          "extra_llm_route_points": 100 * (p["llm_route_share"] - prev["llm_route_share"]),
                          "usd_per_part_per_point": (p["usd_per_part"] - prev["usd_per_part"]) / (100 * (p["llm_route_share"] - prev["llm_route_share"]))})
        prev = p
    return {"points": points, "comparable": same_parts(runs), "frontier_validity": sorted(fv), "frontier_quality": sorted(fq),
            "price_of_validity": steps}


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def render(res: dict) -> str:
    pts = res["points"]
    head = ["system", "models", "USD/part", "tokens/part", "latency s/part", "LLM routes", "raw valid", "eff gap", "M4 LCS", "escalated",
            "final from large", "frontier"]
    rows = []
    for p in sorted(pts, key=lambda q: (q["usd_per_part"] is None, q["usd_per_part"] or 0, q["label"])):
        mark = ("V" if p["on_validity_frontier"] else "-") + ("Q" if p["on_quality_frontier"] else "-")
        rows.append([p["label"], "+".join(m.split("/")[-1] for m in p["models"]) or "n/a",
                     "n/a" if p["usd_per_part"] is None else f"{p['usd_per_part']:.6f}", _f(p["tokens_per_part"], 0), _f(p["latency_s_per_part"], 2),
                     _f(p["llm_route_share"], 2), _f(p["raw_strict_valid"], 3), _f(p["efficiency_gap"], 2), _f(p["m4_lcs"], 3),
                     "-" if p["escalation_rate"] is None else _f(p["escalation_rate"], 2),
                     "-" if p["final_from_large"] is None else _f(p["final_from_large"], 2), mark])
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    out = ["  ".join(h.ljust(wi) if i < 2 else h.rjust(wi) for i, (h, wi) in enumerate(zip(head, w)))]
    out += ["  ".join(c.ljust(wi) if i < 2 else c.rjust(wi) for i, (c, wi) in enumerate(zip(r, w))) for r in rows]
    out.append("")
    out.append("LLM routes = share of parts whose final route is an LLM route (raw or repaired) rather than the Route Builder's; raw valid = strict-valid share of the")
    out.append("raw answers served (M1); eff gap = mean Efficiency shortfall to the best Route Builder variant in points (smaller is better); frontier: V = on the")
    out.append("(cost, LLM routes) Pareto frontier, Q = on the (cost, eff gap) frontier. USD = list price x tokens, cache hits included.")
    if not res["comparable"]:
        out.append("WARNING: the runs do not cover the same (part, repeat) pairs, so the points are not comparable.")
    unpriced = sorted({m for p in pts for m in p["unpriced"]})
    if unpriced:
        out.append("No list price for " + ", ".join(unpriced) + " (capp/config.py PRICE_USD_PER_MTOK): those runs are shown but kept off the frontier.")
    if res["price_of_validity"]:
        out.append("")
        out.append("Price of validity along the frontier (cheapest first):")
        for st in res["price_of_validity"]:
            out.append(f"  {st['from']} -> {st['to']}: +{st['extra_usd_per_part']:.6f} USD/part for +{st['extra_llm_route_points']:.1f} points of LLM routes "
                       f"({st['usd_per_part_per_point']:.6f} USD/part per point)")
    return "\n".join(out)


def plot(res: dict, path: str, title: str = ""):
    """Scatter of cost against the share of LLM routes, frontier joined by a line. Needs matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = [p for p in res["points"] if p["usd_per_part"] is not None]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    front = sorted((p for p in pts if p["on_validity_frontier"]), key=lambda p: p["usd_per_part"])
    ax.plot([p["usd_per_part"] for p in front], [p["llm_route_share"] for p in front], color="#888888", lw=1, zorder=1)
    for p in pts:
        ax.scatter(p["usd_per_part"], p["llm_route_share"], s=55, zorder=2, color="#1f6feb" if p["on_validity_frontier"] else "#bbbbbb")
        ax.annotate(p["label"], (p["usd_per_part"], p["llm_route_share"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("list-price cost, USD per part")
    ax.set_ylabel("share of parts with an LLM route (raw or repaired)")
    ax.set_title(title or "Validity against cost")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
