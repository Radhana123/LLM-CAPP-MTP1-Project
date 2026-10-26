"""Planner options that can be appended to a system name with '+' (report 6.2).

    n5_r1_s3+fs2        retrieval few-shot, k = 2 most similar development parts as examples
    n5_r1_s3+fs4x       k = 4, and parts with the same feature set as the query are left out (stricter: simulates a new combination)
    n5_r1_s3+h1         the Route Builder's route is given to the LLM as a hint (H1); h0 = no hint (the default)
    n5_r1_s3+ts         two-step planner: a reasoning call, then a formalisation call under structured output (capp/twostep.py)
    b0+fs2+h1           both few-shot and hint; all three options combine: n5_r1_s3+fs2x+h1+ts

fs0 + h0 (no suffix at all) is exactly the v1 planner, byte for byte, so its LLM answers are the ones already in the cache.
This module only parses; it imports nothing heavy so that capp/systems.py can use it.
"""
import re
from dataclasses import dataclass

FEW_SHOT_K = (0, 2, 4)
_FS = re.compile(r"^fs(0|2|4)(x?)$")


@dataclass(frozen=True)
class PlannerSpec:
    few_shot: int = 0              # number of retrieved examples in the prompt (0, 2, 4)
    loo: str = "part"              # "part": leave the query part out; "features": also every part with the same feature set
    hint: bool = False             # H1: the Route Builder's route is in the prompt
    two_step: bool = False         # reason first, then formalise (two LLM calls per sample)

    @property
    def is_default(self) -> bool:
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
    few_shot = hint = two_step = None
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
        else:
            raise ValueError(f"unknown planner option {t!r} in {name!r}: use fs0, fs2 or fs4 (add x to also leave out parts with "
                             f"the same feature set), h0 or h1, and ts (two-step planner)")
    return base, PlannerSpec(few_shot or 0, loo, bool(hint), bool(two_step))
