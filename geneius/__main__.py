"""Command line entry point.

    python -m geneius variant RHO p.Pro23His       # predictions + assay (stub until the GNN exists)
    python -m geneius ask "What does RHO P23H do?" # LLM agent
    python -m geneius symptoms "night blindness and retinitis pigmentosa"
"""
from __future__ import annotations

import argparse
import json


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="geneius", description="Mechanism-aware VUS interpretation agent")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("variant", help="predict A to D for one variant")
    v.add_argument("gene")
    v.add_argument("change", help="protein change, e.g. p.Pro23His")

    a = sub.add_parser("ask", help="ask the LLM agent a question")
    a.add_argument("question")

    s = sub.add_parser("symptoms", help="map free text to HPO terms")
    s.add_argument("text")

    args = ap.parse_args(argv)

    if args.cmd == "variant":
        from .agent.tools import gnn_predict, suggest_assay, suggest_treatment

        name = f"{args.gene.upper()} {args.change}"
        pred = gnn_predict(name)
        mech = max(pred["mechanism"], key=pred["mechanism"].get)
        out = {
            "prediction": pred,
            "treatment": suggest_treatment(mech, gene=args.gene),
            "assay": suggest_assay(args.gene),
        }
        print(json.dumps(out, indent=2))
    elif args.cmd == "ask":
        from .agent.agent import run_agent

        print(run_agent(args.question, verbose=True)["answer"])
    else:
        from .agent.tools import map_symptoms

        for t in map_symptoms(args.text):
            print(f"{t['hpo_id']}  {t['label']}  (from \"{t['matched_text']}\")")


if __name__ == "__main__":
    main()
