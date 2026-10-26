import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import compare as C          # noqa: E402
from capp import dataset as D          # noqa: E402
from capp import harness as H          # noqa: E402
from capp import llm_client            # noqa: E402
from capp import verify as V           # noqa: E402
import route_builder as rb             # noqa: E402


def planner(material, features, tolerance="0.05mm", batch_size=100, sample_id=0):
    routes = rb.generate_valid_routes(features, max_routes=15)
    base = V.clean_steps(routes[0]["steps"])
    k = sample_id % 4
    if k == 0:
        steps = base
    elif k == 1:
        steps = list(base)
        if len(steps) > 3:
            steps[1], steps[2] = steps[2], steps[1]            # one swapped pair: a typical small error
    elif k == 2:
        steps = base[:-1] + ["Deburring", "Inspection"]
    else:
        steps = list(reversed(base))
    meta = {"cached": False, "finish_reason": "stop", "prompt_tokens": 300, "completion_tokens": 40, "latency_s": 0.4}
    return {"success": True, "steps": steps, "raw_response": json.dumps(steps), "model": "fake", "meta": meta}


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CAPP_LLM_MODE", raising=False)
    monkeypatch.delenv("CAPP_RUN_TAG", raising=False)
    llm_client.reset_transport()


@pytest.fixture
def runs(tmp_path):
    ps = D.load_dataset("dev_v2")[:12]
    out = str(tmp_path / "res")
    for s in ("rb", "b0", "b0_r1"):
        H.run_experiment(s, ps, tag="c", out_dir=out, planner=planner, progress=None)
    return out


def test_compare_prints_table_and_paired_sections(runs):
    text = C.compare([os.path.join(runs, f"c_{s}") for s in ("rb", "b0", "b0_r1")])
    assert "M2 final strict-valid" in text and "b0_r1" in text
    assert "Paired final Efficiency, b0_r1 vs rb" in text
    assert "Same raw LLM routes, R0 (b0) vs R1 (b0_r1)" in text


def test_paired_repairs_r1_keeps_more_of_the_llm_route(runs):
    r0 = H.load_records(os.path.join(runs, "c_b0"))
    r1 = H.load_records(os.path.join(runs, "c_b0_r1"))
    p = C.paired_repairs(r0, r1)
    assert p["n"] > 0
    assert p["survived_r1"] >= p["survived_r0"]
    assert p["edit_distance_r1"] <= p["edit_distance_r0"]


def test_paired_quality_counts_add_up(runs):
    base = H.load_records(os.path.join(runs, "c_rb"))
    other = H.load_records(os.path.join(runs, "c_b0_r1"))
    q = C.paired_quality(base, other)
    assert q["n"] == q["better"] + q["equal"] + q["worse"] == 12


def test_cli(runs):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "compare_runs.py")] + [os.path.join(runs, f"c_{s}") for s in ("rb", "b0_r1")]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0 and "M6 efficiency" in res.stdout
    bad = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "compare_runs.py"), "nope1", "nope2"],
                         capture_output=True, text=True, timeout=60)
    assert bad.returncode == 1
