"""Configuration freeze (MTP-1 WP0 / WP11): pin what a final run is made of, and check later that nothing has changed.

    python scripts/freeze.py create --grid configs/ablation_template.json [--extra results/threshold.json] [--run-tests] [--note "Tier 1"]
    python scripts/freeze.py verify                  # exit 0: ok / only analysis code changed; exit 1: drift; exit 2: no usable FREEZE.json
    python scripts/freeze.py verify --strict         # also exit 1 when analysis code changed
    python scripts/freeze.py show
    python scripts/freeze.py diff FREEZE_tier1.json FREEZE.json

`create` writes FREEZE.json (code files, config constants, model ids, datasets, grid, extras: SHA-256 of each). Run it when the configuration is
final (gate G2 for Tier 1, G4 for the rest), then commit it and tag: git add FREEZE.json; git commit -m "Freeze"; git tag freeze-v1.
scripts/run_eval.py refuses to run a sealed dataset (test_v2, bench_v1) with LLM calls unless FREEZE.json exists and the working tree still
matches it; every run.json records the freeze status.  See capp/freeze.py for exactly what is compared.
No API call is made. `--run-tests` runs the test suite once (about two minutes) and records the result; a freeze with failing tests is refused.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import freeze as F          # noqa: E402


def _path(a):
    return a.manifest or os.path.join(ROOT, F.MANIFEST_NAME)


def cmd_create(a):
    tests = None
    if a.run_tests:
        print("running the test suite ...", file=sys.stderr)
        tests = F.run_tests(ROOT)
        print(f"tests: {tests['summary']}", file=sys.stderr)
    try:
        man = F.create(ROOT, _path(a), grids=a.grid, extras=a.extra, datasets=a.dataset, note=a.note or "", name=a.name or "", tests=tests,
                       force=a.force)
    except F.FreezeError as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    print(F.format_manifest(man))
    print(f"\nWritten: {_path(a)}")
    git = man.get("git") or {}
    if git.get("dirty"):
        print("WARNING: the working tree has uncommitted changes. The manifest covers the files as they are now; commit them together with "
              "FREEZE.json so that the tag points at exactly this state.", file=sys.stderr)
    if not tests:
        print("Note: the tests were not run at freeze time (--run-tests).", file=sys.stderr)
    print("Next: git add FREEZE.json; git commit -m \"Freeze\"; git tag freeze-v1")
    return 0


def cmd_verify(a):
    try:
        res = F.verify(ROOT, _path(a))
    except F.FreezeError as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    print(F.format_comparison(res))
    if res["status"] == "drift":
        print("\nThe working tree is not the frozen configuration. Results made now are not those of the freeze: undo the change "
              "(git diff shows it), or, if it is deliberate, create a new freeze and say why (create --force --note ...).", file=sys.stderr)
        return 1
    if res["status"] == "analysis-changed" and a.strict:
        print("\n--strict: analysis code changed after the freeze.", file=sys.stderr)
        return 1
    return 0


def cmd_show(a):
    try:
        print(F.format_manifest(F.load(_path(a))))
    except F.FreezeError as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_diff(a):
    try:
        first, second = F.load(a.first), F.load(a.second)
    except F.FreezeError as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    res = F.compare(first, second)
    res["id"] = f"{first['id']} -> {second['id']}"
    print(F.format_comparison(res))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="write FREEZE.json")
    c.add_argument("--grid", action="append", default=[], required=True, help="ablation config of the final runs (repeatable)")
    c.add_argument("--extra", action="append", default=[], help="another file to pin, e.g. the tuned confidence threshold (repeatable)")
    c.add_argument("--dataset", action="append", default=[], help="another dataset to pin (dev_v2, test_v2 and bench_v1 are pinned if they exist)")
    c.add_argument("--run-tests", action="store_true", help="run the test suite and record the result")
    c.add_argument("--note", help="why / what this freeze is (Tier 1, final ...)")
    c.add_argument("--name", help="a short name, e.g. tier1")
    c.add_argument("--force", action="store_true", help="replace an existing manifest (the old id is recorded as 'replaces')")
    c.add_argument("--manifest", help="where to write (default FREEZE.json in the repository)")
    c.set_defaults(fn=cmd_create)
    v = sub.add_parser("verify", help="compare the working tree with the manifest")
    v.add_argument("--manifest")
    v.add_argument("--strict", action="store_true")
    v.set_defaults(fn=cmd_verify)
    s = sub.add_parser("show", help="print the manifest")
    s.add_argument("--manifest")
    s.set_defaults(fn=cmd_show)
    d = sub.add_parser("diff", help="what differs between two manifests")
    d.add_argument("first")
    d.add_argument("second")
    d.set_defaults(fn=cmd_diff)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
