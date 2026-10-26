"""Everything the evaluation reports must be the same in every Python process (PYTHONHASHSEED changes the iteration order
of string sets). The Route Builder once chose the machine of each shared operation in set-iteration order, so one 7-feature
archetype got 15 routes in one process and 12 in another, and every Route Builder number moved with it."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE = r"""
import json, sys
sys.path.insert(0, %r)
from capp import archetypes, legacy, systems, verify
import route_builder as rb
out = {}
for a in archetypes.ARCHETYPES:
    key = a["name"]
    routes = rb.generate_valid_routes(a["features"], max_routes=15, machine_preference="auto")
    out[key] = [r["steps"] for r in routes]
    out[key + "/prefer_lathe"] = [r["steps"] for r in rb.generate_valid_routes(a["features"], max_routes=5, machine_preference="prefer_lathe")]
    out[key + "/warnings"] = [r["warnings"] for r in rb.build_routes(a["features"], max_routes=5, machine_preference="prefer_milling")]
    out[key + "/fix"] = systems.fsm_validator.fix_sequence_with_builder(a["features"])
    out[key + "/R1"] = None
    from capp import repair
    out[key + "/R1"] = repair.r1(list(reversed(out[key][0])) if out[key] else [], a["features"]).steps if out[key] else None
print(json.dumps(out, sort_keys=True))
""" % ROOT


def _run(seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONIOENCODING="utf-8")
    res = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_route_builder_repair_and_replacement_do_not_depend_on_the_hash_seed():
    runs = [_run(s) for s in (0, 1, 2, 3)]
    for other in runs[1:]:
        diff = [k for k in runs[0] if runs[0][k] != other[k]]
        assert not diff, f"differs between Python processes: {diff[:5]}"
    assert len(runs[0]) >= 5 * 40
