"""Show how one part was planned in a finished run: every sample's reasoning (two-step runs), the route the LLM gave, what the
verifier said, what was repaired, and the final route. No API calls.

    python scripts/show_plan.py results/dev_b1+ts DEV_007
    python scripts/show_plan.py results/dev_b0 DEV_007 --repeat 2
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capp import harness   # noqa: E402


def render(rec: dict) -> str:
    part, info, fin = rec["part"], rec["info"], rec["final"]
    L = [f"=== {rec['part_id']} ({rec['system']}, repeat {rec['repeat']}) ===",
         f"{part['material']}, features: {', '.join(part['features'])}; tolerance {part.get('tolerance', '')}, batch {part.get('batch_size', '')}"]
    for c in rec["candidates"]:
        if c["sample_id"] < 0:
            continue
        meta = c["meta"] or {}
        L.append(f"\n--- sample {c['sample_id']} ({c['name']}, {c['provenance']}) ---")
        ctx = meta.get("prompt_context") or {}
        if ctx.get("examples"):
            L.append(f"examples in prompt: {', '.join(ctx['examples'])}" + ("; Route Builder hint given" if ctx.get("hint") else ""))
        elif ctx.get("hint"):
            L.append("Route Builder hint given")
        if meta.get("reasoning"):
            L.append("reasoning:\n  " + meta["reasoning"].replace("\n", "\n  "))
        if meta.get("n_calls") == 2:
            faithful = {True: "yes", False: "NO", None: "unknown"}[meta.get("faithful")]
            L.append(f"formalisation: structured output {meta.get('structured_used')}, parser {meta.get('parser')}; "
                     f"step 2 kept the stated order: {faithful}")
        if not c["llm_ok"]:
            L.append("LLM route: (none: the call failed or nothing could be parsed)")
        else:
            L.append("LLM route: " + " > ".join(c["llm_steps"]))
            fp = c["first_pass"]
            L.append("verifier : " + ("valid" if fp["valid"] else "INVALID: " + "; ".join(v["message"] for v in fp["violations"] if v["kind"] != "machine_grouping")))
        if c.get("repaired") or c.get("replaced"):
            how = (c.get("repair") or {}).get("stage") or ("R0 replacement" if c.get("replaced") else "repair")
            L.append(f"repaired by {how}: " + " > ".join(s for s in c["steps"] if not s.startswith("---")))
    L.append(f"\nFINAL ({fin['provenance']}, selection {info.get('selection', 's1')}): "
             + " > ".join(s for s in fin["steps"] if not s.startswith("---")))
    if info.get("confidence") is not None:
        L.append(f"plan confidence {info['confidence']:.2f}" + (" (flagged for review)" if info.get("flagged") else ""))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("part_id")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    if not os.path.exists(os.path.join(args.run_dir, "records.jsonl")):
        ap.error(f"no records.jsonl in {args.run_dir}")
    recs = [r for r in harness.load_records(args.run_dir) if r["part_id"] == args.part_id and r["repeat"] == args.repeat]
    if not recs:
        ap.error(f"{args.part_id} (repeat {args.repeat}) is not in {args.run_dir}")
    print(render(recs[0]))


if __name__ == "__main__":
    main()
