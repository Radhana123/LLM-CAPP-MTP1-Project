"""Model cascade (report 6.4, row A8): a small and fast model is asked first, the large model only where the verifier cannot help.

A verifier call is cheap next to an LLM call, so the verdict on the small model's route can decide whether the large model is worth
paying for. Two levels, both per sample (one sample slot = one call to the small model, plus one to the large model if it escalates):

  cas1  the small model's sample can be neither validated nor repaired: the whole repair chain (R1, R2 if the system has it) failed and
        the sample would be replaced by the Route Builder (R0). The slot is asked again with the LARGE model, and that answer goes through
        the same chain (verify, repair, R0 as the last resort). "Cannot validate or repair", read strictly.
  cas2  eager: the small model's raw route is not accepted as it is (strict-invalid, or a hard physical-feasibility finding with +fz, or
        no usable answer at all). The slot is asked again with the LARGE model, and the better of the two results is kept: raw_llm before
        repaired_llm before route_builder, the large model's answer on a tie. "Cannot validate", read broadly: more large calls, fewer
        repaired routes.

Both reduce to the small model alone if it is always right, and to the large model alone if the small one never is, so the systems
+ms (small only) and +ml (large only) are the two ends of the Pareto line and +cas1 / +cas2 lie between them. They share the cache
with the cascade: a cascade run after +ms and +ml with the same --tag needs no sample call that those two have not made (R2
re-prompts excepted, when their violation reports differ).

Cost: capp/cost.py turns the token counts of a run into list-price dollars. This module only decides; the loop is in
capp/systems.py (run_llm_first), the model roles are in capp/config.py, the options in capp/plannerspec.py.
"""
from dataclasses import dataclass
from typing import Callable

LEVELS = (1, 2)
RANK = {"raw_llm": 0, "repaired_llm": 1, "route_builder": 2}      # smaller is better (the order of trust in an LLM route)
TRIGGER = {1: "repair failed", 2: "raw route not accepted as is"}


@dataclass
class Cascade:
    """The second tier of a cascade. The first tier is the `planner` / `repair_planner` of run_llm_first, as in any other system."""
    level: int                               # 1 or 2, see the module docstring
    planner: Callable                        # large-model planner, the call signature of llm_planner.generate_process_plan
    repair_planner: Callable = None          # large-model R2 re-prompt (only used by systems with repair r2 / r3)

    def __post_init__(self):
        if self.level not in LEVELS:
            raise ValueError(f"unknown cascade level {self.level!r}: 1 or 2")
        if not callable(self.planner):
            raise ValueError("a cascade needs a callable large-model planner")


def escalate(level: int, provenance: str, raw_ok: bool) -> bool:
    """Does the small model's sample go to the large model? `provenance` is what the sample became after repair (raw_llm,
    repaired_llm, route_builder), `raw_ok` whether the verifier accepted the raw route as it was."""
    if level == 1:
        return provenance == "route_builder"
    return not raw_ok


def keep_large(small_provenance: str, large_provenance: str) -> bool:
    """Which of the two results of an escalated slot is kept: the better provenance, the large model's on a tie."""
    return RANK[large_provenance] <= RANK[small_provenance]
