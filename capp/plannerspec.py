"""Planner options that can be appended to a system name with '+' (report 6.2).

    n5_r1_s3+fs2        retrieval few-shot, k = 2 most similar development parts as examples
    n5_r1_s3+fs4x       k = 4, and parts with the same feature set as the query are left out (stricter: simulates a new combination)
    n5_r1_s3+h1         the Route Builder's route is given to the LLM as a hint (H1); h0 = no hint (the default)
    n5_r1_s3+ts         two-step planner: a reasoning call, then a formalisation call under structured output (capp/twostep.py)
    b0+fs2+h1           both few-shot and hint; all three options combine: n5_r1_s3+fs2x+h1+ts
    n5_r1_s3+fz1        feasibility-aware repair (row A7): R1 also fixes the HARD physical-feasibility violations (capp/feasrepair.py)
    n5_r1_s3+fz2        fz1, and R1 also inserts what the RISK rules ask for (centre drill, reaming); selection prefers feasible routes

fs0 + h0 (no suffix at all) is exactly the v1 planner, byte for byte, so its LLM answers are the ones already in the cache.
fz is not a planner option: it changes what happens to the samples (repair, selection), not the prompt, so +fz1/+fz2 give the same LLM
answers as the system without it (same cache) and need no extra API call. It lives in this grammar because the suffix does.
This module only parses; it imports nothing heavy so that capp/systems.py can use it.
"""
import re
from dataclasses import dataclass

FEW_SHOT_K = (0, 2, 4)
_FS = re.compile(r"^fs(0|2|4)(x?)$")
FEAS_LEVELS = (0, 1, 2)


@dataclass(frozen=True)
class PlannerSpec:
    few_shot: int = 0              # number of retrieved examples in the prompt (0, 2, 4)
    loo: str = "part"              # "part": leave the query part out; "features": also every part with the same feature set
    hint: bool = False             # H1: the Route Builder's route is in the prompt
    two_step: bool = False         # reason first, then formalise (two LLM calls per sample)
    feas: int = 0                  # feasibility-aware repair level (0 off, 1 hard rules, 2 hard + risk insertions); not a prompt option

    @property
    def is_default(self) -> bool:                   # "the v1 prompt": fz does not touch the prompt, so it is not part of this
        return self.few_shot == 0 and not self.hint and not self.two_step

    @property
    def name(self) -> str:
        tokens = []
        if self.few_shot:
            tokens.append(f"fs{self.few_shot}" + ("x" if self.loo == "features" else ""))
        if self.hint:
            tokens.append("h1")
        if self.two_step:
            tokens.append("ts")
        return "+".join(tokens)


def split_name(name: str):
    """'n5_r1_s3+fs2+h1' -> ('n5_r1_s3', PlannerSpec(few_shot=2, hint=True)). Raises ValueError for an unknown option."""
    base, *tokens = name.split("+")
    few_shot = hint = two_step = feas = None
    loo = "part"
    for t in tokens:
        m = _FS.match(t)
        if m:
            if few_shot is not None:
                raise ValueError(f"two few-shot options in {name!r}")
            few_shot, loo = int(m.group(1)), ("features" if m.group(2) else "part")
        elif t in ("h0", "h1"):
            if hint is not None:
                raise ValueError(f"two hint options in {name!r}")
            hint = t == "h1"
        elif t == "ts":
            if two_step is not None:
                raise ValueError(f"two ts options in {name!r}")
            two_step = True
        elif t in ("fz0", "fz1", "fz2"):
            if feas is not None:
                raise ValueError(f"two feasibility options in {name!r}")
            feas = int(t[2])
        else:
            raise ValueError(f"unknown planner option {t!r} in {name!r}: use fs0, fs2 or fs4 (add x to also leave out parts with "
                             f"the same feature set), h0 or h1, ts (two-step planner), and fz0, fz1 or fz2 (feasibility-aware repair)")
    return base, PlannerSpec(few_shot or 0, loo, bool(hint), bool(two_step), feas or 0)
