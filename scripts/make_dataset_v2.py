"""Regenerate dataset v2 (data/dev_v2.json/.csv and data/test_v2.json/.csv) and print a summary.

Run from the repo root:   python scripts/make_dataset_v2.py
The output is deterministic (fixed seeds), so re-running must not change the files (git status stays clean).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import dataset as D          # noqa: E402
from capp import datagen as G          # noqa: E402


def main():
    dev = G.generate_dev()
    test = G.generate_test(dev_parts=dev)
    for name, parts in (("dev_v2", dev), ("test_v2", test)):
        errs = D.validate_dataset(parts)
        if errs:
            print(f"{name}: {len(errs)} validation errors", errs[:5])
            sys.exit(1)
        path = D.save_dataset(parts, name)
        print(f"\n== {name}  ->  {path}")
        print(json.dumps(D.summarize(parts), indent=2))
    print("\nDone. Dev is for design/tuning. Test is sealed until the final runs.")


if __name__ == "__main__":
    main()
