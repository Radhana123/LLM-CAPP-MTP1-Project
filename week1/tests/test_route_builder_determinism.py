# test_route_builder_determinism.py
# MTP-1 WP0: Route Builder ab reproducible hai (same input -> same routes).
# Pehle random.shuffle + set-iteration ki wajah se output har run mein badal sakta tha.

import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from route_builder import generate_valid_routes_simple

FEATURES = ["Thread", "Fillet"]


def test_same_input_same_routes():
    assert generate_valid_routes_simple(FEATURES, 5) == generate_valid_routes_simple(FEATURES, 5)


def test_independent_of_global_random_state():
    random.seed(1)
    a = generate_valid_routes_simple(FEATURES, 5)
    random.seed(2)
    b = generate_valid_routes_simple(FEATURES, 5)
    assert a == b


def test_independent_of_python_hash_seed():
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from route_builder import generate_valid_routes_simple as g; "
        "print(g(['Thread','Fillet','Slot'], 5))" % os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )
    outs = []
    for hash_seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout)
    assert outs[0] == outs[1] == outs[2]
