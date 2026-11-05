"""The numbers and texts of the home page (Step 52). Every number is read from the code or the data, so the page cannot go stale again
(it showed '1 USD = Rs.95.595' and 'Dataset 200' after the model had moved on).

    from capp import home_stats as HS
    HS.cards()    # [(number text, label, tooltip)]
    HS.blurbs()   # [(title, text)]
"""
import json
import os

from capp import legacy  # noqa: F401
import agents
import precedence_graph as pg
from feature_vocab import GEOMETRY_FEATURES

from capp import calculator as CALC
from capp import op_dims as OPD

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATASETS = (("dev_v2", "dev"), ("test_v2", "test"))


def dataset_sizes(data_dir: str = None) -> dict:
    """{'dev': n, 'test': n}: the number of parts in each data file (only counted, never used). A file that is missing or unreadable is left out."""
    out = {}
    for name, label in DATASETS:
        try:
            with open(os.path.join(data_dir or _DATA, name + ".json"), encoding="utf-8") as f:
                out[label] = len(json.load(f))
        except (OSError, ValueError, TypeError):
            continue
    return out


def usd_text() -> str:
    return f"₹{agents.USD_TO_INR:g}"


def cards(data_dir: str = None) -> list:
    """The five cards: (number, label, tooltip)."""
    sizes = dataset_sizes(data_dir)
    n_parts = sum(sizes.values())
    detail = " + ".join(f"{n} {k}" for k, n in sizes.items())
    return [
        (str(len(GEOMETRY_FEATURES)), "Features", "geometry features the planner knows"),
        (str(len(OPD.ALL_OPS)), "Operations", "operations of the vocabulary, each with its own dimensions"),
        (str(n_parts) if n_parts else "–", "Dataset Parts", detail or "data files not found"),
        (str(len(pg.PRECEDENCE_EDGES)), "Precedence Rules", "ordering rules the verifier checks"),
        (usd_text(), "Per USD", "exchange rate of the v1 cost model (week3/agents.py)"),
    ]


def blurbs() -> list:
    """(title, text) of the blocks under the cards."""
    return [
        ("\U0001F3ED Machine-Aware Routing",
         f"The AI planner proposes the route, the verifier checks it against {len(pg.PRECEDENCE_EDGES)} precedence rules, and a route that fails is repaired "
         "or replaced by the Route Builder. Lathe ops \U0001F7E1 and milling ops \U0001F535 are grouped; every switch between the machines is charged as a changeover."),
        ("\U0001F9EE Calculator Time & Cost Model",
         f"Full Analysis calculates time and cost the way your Time-and-Cost-calculator does: its formulas for {len(CALC.PROCESSES)} machining processes and the "
         f"{len(CALC.EXTRA)} other processes, plus {len(CALC.MILLING)} milling processes built the same way from the milling formulas (Vc = \u03c0Dn/1000, "
         "Vf = n\u00b7fz\u00b7Z, face and slab milling time). Labour cost per hour and material cost are typed in the sidebar. Total = material + labour + tool + "
         "extra process cost. The operations with no formula (forming, counterboring, ...) use the v1 cutting time, and energy stays the v1 figure: "
         "there is no energy formula."),
        ("⏱️ v1 Time Model (report numbers)",
         f"The model of the experiments: Machining + Tool Change (+{agents.TOOL_CHANGE_TIME_MIN} min) + Position Change (+{agents.POSITION_CHANGE_TIME_MIN} min) "
         f"+ Changeover (+{agents.CHANGEOVER_TIME_MIN} min), multi-pass depth included. Choose it in the sidebar to get the numbers of the report."),
        ("\U0001F4D0 Dimensions of every operation",
         f"All {len(OPD.ALL_OPS)} operations are in a dropdown (the calculator model adds the {len(CALC.CALCULATOR_ONLY)} items of your calculator that no operation maps to); "
         "the ones your features need are selected. Numbers written on the drawing for the feature an operation "
         "makes are filled in, the rest comes from the Part Dimensions, and every box can be changed. Add an operation that is missing or take one out: "
         "the calculator's time and cost follow."),
        ("\U0001F4B0 INR Costing, Physics and Route Sheet",
         f"All costs in ₹ INR (1 USD = {usd_text()}). The v1 cost is machining time × rate per minute; batch discount >100 → 8% off, >500 → 15% off "
         "(the calculator prices one job, so no discount in calculator mode). Step 5 checks cutting physics and feasibility, and the route sheet can be downloaded."),
    ]
