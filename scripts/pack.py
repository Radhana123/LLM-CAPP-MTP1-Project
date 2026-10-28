"""Pack (MTP-1 WP12, report 8.4 items iii and iv): a release folder in ONE zip file for Zenodo or for your records.

    python scripts/pack.py check  release/v1                                  what would go in, and what stands in the way (writes nothing)
    python scripts/pack.py build  release/v1 --with-runs                      -> release/v1.zip and release/v1.zip.sha256
    python scripts/pack.py build  release/v1 --with-runs --with-cache --out dist
    python scripts/pack.py verify release/v1.zip                              the archive on its own: every file, the sums, the release inside

Before the zip is written: the release is verified (python scripts/release.py verify), the metadata you wrote (configs/zenodo.json, copy of
configs/zenodo_template.json: title, creators, licence) must be complete, and no file may contain a secret (API keys, tokens, the value of your
GROQ_API_KEY). If anything blocks, nothing is written (exit 1). --with-runs puts the run folders in, so the tables can be rebuilt from the
archive plus the code; --with-cache adds the whole LLM cache (every cached call, not only this release's). --no-regenerate checks only the file
hashes of the release instead of building it again. Exit 2: a problem with the input (not a release, not a pack). verify: 0 verified, 1 not.

Nothing is uploaded and no API call is made; the zip is for you to upload. See capp/pack.py.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import pack as K, release as R, tables as T      # noqa: E402

ERRORS = (K.PackError, R.ReleaseError, T.TableError, OSError, UnicodeError)
DEFAULT_METADATA = os.path.join("configs", "zenodo.json")


def _args(a):
    meta = a.metadata or os.path.join(ROOT, DEFAULT_METADATA)
    return dict(root=ROOT, metadata_path=meta, with_runs=a.with_runs, with_cache=a.with_cache, cache_dir=a.cache_dir, regenerate=not a.no_regenerate)


def cmd_check(a):
    p = K.plan(a.folder, **_args(a))
    print(K.format_plan(p))
    return 1 if p["blocking"] else 0


def cmd_build(a):
    out = a.out or os.path.join(ROOT, "release")
    p = K.build(a.folder, out, **_args(a))
    print(K.format_plan(p))
    if p["zip"] is None:
        print("\nNothing was written.", file=sys.stderr)
        return 1
    print(f"\nWritten: {p['zip']}\n         {p['sha256_file']}\nSHA-256 of the zip: {p['zip_sha256']}\n{p['members']} files in the archive.")
    print(f"Check it later:  python scripts/pack.py verify {os.path.relpath(p['zip'], os.getcwd())}")
    return 0


def cmd_verify(a):
    v = K.verify_archive(a.archive)
    print(K.format_verify_archive(v))
    return 0 if v["ok"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in (("check", cmd_check, "what a pack would contain and what blocks it (writes nothing)"), ("build", cmd_build, "write the zip")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("folder", help="the release folder (made by scripts/release.py)")
        s.add_argument("--metadata", help=f"the upload metadata (default {DEFAULT_METADATA})")
        s.add_argument("--with-runs", action="store_true", help="put the run folders into the archive")
        s.add_argument("--with-cache", action="store_true", help="put the whole LLM cache into the archive")
        s.add_argument("--cache-dir", help="the cache folder (default: the one of the configuration, cache/llm)")
        s.add_argument("--no-regenerate", action="store_true", help="check only the file hashes of the release, do not build it again")
        if name == "build":
            s.add_argument("--out", help="folder for the zip (default release/)")
        s.set_defaults(fn=fn)
    v = sub.add_parser("verify", help="check an archive on its own")
    v.add_argument("archive")
    v.set_defaults(fn=cmd_verify)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except ERRORS as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
