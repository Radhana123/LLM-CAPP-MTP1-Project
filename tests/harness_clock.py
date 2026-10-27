"""Test helper: count the pauses the harness asks for (throttle) without waiting."""
import time

from capp import harness as H


class HarnessClock:
    """capp.harness.time with only sleep() replaced (it records the pause instead of waiting). Patching time.sleep itself is
    process-wide: subprocess.run(timeout=...) polls with time.sleep, and the harness runs `git rev-parse` that way, so on a loaded machine
    the polling loop spun and was counted as thousands of sleeps (the flaky throttle tests). Replacing the harness's own reference
    leaves every other module's time untouched."""
    def __init__(self, slept):
        self._slept = slept

    def sleep(self, seconds):
        self._slept.append(seconds)

    def __getattr__(self, name):
        return getattr(time, name)


def count_sleeps(monkeypatch, slept):
    monkeypatch.setattr(H, "time", HarnessClock(slept))
