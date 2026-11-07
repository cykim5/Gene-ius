"""Command line entry point.

    python -m geneius "4 year old with seizures, low muscle tone and speech delay"
    python -m geneius --agent "..."     # use the LLM agent
    python -m geneius --json "..."      # raw JSON output
"""
from __future__ import annotations

import argparse
import json


def _print_pipeline(result: dict) -> None:
    print("HPO terms:")
    for t in result["hpo_terms"]:
        print(f"  {t['hpo_id']}  {t['label']}  (from \"{t['matched_text']}\")")
    if not result["candidates"]:
        print("\nNo candidates. Try describing the symptoms differently.")
        return
    print("\nCandidates:")
    for i, c in enumerate(result["candidates"], 1):
        print(f"  {i:>2}. {c['gene']:<10} {c['score']:.3f}  {c['disease']}")
        for p in c.get("evidence", [])[:3]:
            print("        " + " > ".join(n["name"] for n in p["path"]))
    f = result.get("followup")
    if f and f.get("question"):
        print(f"\nFollow-up: {f['question']}")
        print(f"  Would support: {', '.join(f['distinguishes'])}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="geneius", description="Rare disease diagnosis copilot")
    ap.add_argument("text", help="symptoms in plain English")
    ap.add_argument("--agent", action="store_true", help="use the LLM agent instead of the fixed pipeline")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    args = ap.parse_args(argv)

    if args.agent:
        from .agent import run_agent

        result = run_agent(args.text, verbose=not args.json)
        print(json.dumps(result, indent=2) if args.json else result["answer"])
        return

    from .pipeline import run_pipeline

    result = run_pipeline(args.text, top_k=args.top_k)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_pipeline(result)


if __name__ == "__main__":
    main()
