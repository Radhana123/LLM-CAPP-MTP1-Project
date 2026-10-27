"""Release (MTP-1 WP12, report 8.4 items iii and iv): the tables, figures, case studies, benchmark datasheet and paper draft of the final
runs in ONE folder, made by ONE command, and a check that it can be made again.

    python scripts/release.py check  configs/release_template.json                  what a build would find (offline, writes nothing)
    python scripts/release.py build  configs/release_final.json                     into release/<name>/
    python scripts/release.py build  configs/release_final.json --out release/v1 --strict
    python scripts/release.py verify release/v1                                     files unchanged? and: build again, is it identical?
    python scripts/release.py verify release/v1 --no-regenerate                     only the file hashes

The recipe (configs/release_template.json) names the grid, the freeze manifest, the benchmark, the run folders of the case studies and the
files you write by hand (the prose of the paper, the commentary of the case studies). Build runs: freeze check, Table 4 + paired tests +
figures, case studies, benchmark datasheet, paper draft, then writes RELEASE.md, SHA256SUMS.txt and release_manifest.json. The files you
write by hand are never changed: the release works on copies (inputs/).

--strict: exit 1 if a blocking item of the paper readiness is open (the folder is still written). Exit 2: a problem with the recipe or an
input. verify: exit 0 verified, 1 not verified, 2 not a release folder. No API call is made and no run is started; see capp/release.py.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import casestudy as C, freeze, paperdraft as P, release as R, tables as T      # noqa: E402

ERRORS = (R.ReleaseError, T.TableError, P.PaperError, C.CaseError, freeze.FreezeError, OSError, UnicodeError)


def cmd_check(a):
    c = R.check(a.recipe)
    print(R.format_check(c))
    return 2 if c["missing"] or c["grid_error"] else 0


def cmd_build(a):
    recipe = R.load_recipe(a.recipe)
    out = a.out or os.path.join(ROOT, "release", recipe["name"])
    res = R.build(a.recipe, out, ROOT, figures=not a.no_figures)
    print(f"Release {recipe['name']} -> {res['out']}\n")
    for name, what in res["steps"]:
        print(f"  {name:<13} {what}")
    blocking = [c for c in res["checks"] if c["level"] == "blocking" and not c["ok"]]
    advisory = [c for c in res["checks"] if c["level"] == "advisory" and not c["ok"]]
    print(f"\n{len(res['files'])} files. " + ("READY: no blocking item open." if res["ready"] else f"NOT READY: {len(blocking)} blocking, {len(advisory)} advisory item(s) open."))
    for c in blocking:
        print(f"  [BLOCKING] {c['message']}")
    for c in advisory:
        print(f"  [advisory] {c['message']}")
    g = res["git"]
    if g.get("dirty"):
        print("\nNote: the working tree has uncommitted changes; commit them before you tag the release.")
    lacking = [t for t in R.EXPECTED_TAGS if t not in res["tags"]]
    if lacking:
        print(f"Tags not there yet: {', '.join(lacking)} (RELEASE.md has the commands).")
    print(f"\nCheck it later:  python scripts/release.py verify {os.path.relpath(res['out'], os.getcwd())}")
    if a.strict and not res["ready"]:
        print("--strict: blocking item(s) open.", file=sys.stderr)
        return 1
    return 0


def cmd_verify(a):
    v = R.verify(a.folder, ROOT, regenerate=not a.no_regenerate)
    print(R.format_verify(v))
    return 0 if v["ok"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="what a build would find, without writing anything")
    c.add_argument("recipe")
    c.set_defaults(fn=cmd_check)
    b = sub.add_parser("build", help="make the release folder")
    b.add_argument("recipe")
    b.add_argument("--out", help="output folder (default release/<name>)")
    b.add_argument("--no-figures", action="store_true", help="skip the PNG figures")
    b.add_argument("--strict", action="store_true", help="exit 1 if a blocking item is open")
    b.set_defaults(fn=cmd_build)
    v = sub.add_parser("verify", help="are the files unchanged, and does a rebuild give the same files")
    v.add_argument("folder")
    v.add_argument("--no-regenerate", action="store_true", help="only check the file hashes")
    v.set_defaults(fn=cmd_verify)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except ERRORS as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
