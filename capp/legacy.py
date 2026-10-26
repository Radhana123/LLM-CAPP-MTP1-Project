"""Makes the v1 modules (week1..week5, which use flat imports like `import feature_vocab`) importable from capp.

Import this module first:  `from capp import legacy  # noqa: F401`
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for _w in ("week1", "week2", "week3", "week4", "week5"):
    _p = os.path.join(ROOT, _w)
    if _p not in sys.path:
        sys.path.insert(0, _p)
