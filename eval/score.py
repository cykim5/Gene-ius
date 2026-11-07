"""Score a ranker with hits@1, hits@5, hits@10 for the correct gene.

    python eval/score.py --patients eval/data/simulated.jsonl --method overlap
    python eval/score.py --patients eval/data/real_cases.jsonl --method overlap --mode transductive

Modes:
  heldout       (default for simulated patients) the held-out diseases' phenotype
                annotations are hidden from the ranker, same as for the GNN. Fair comparison.
  transductive  nothing hidden. Use for real case reports.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geneius.baseline_overlap import OverlapRanker  # noqa: E402
from geneius.data import get_hpo, get_kg  # noqa: E402

HERE = Path(__file__).resolve().parent
KS = (1, 5, 10)


def load_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def make_ranker(method: str, hide: set[str]):
    if method == "overlap":
        r = OverlapRanker(get_kg(), get_hpo(), hide_diseases=hide)
        return r.rank_genes
    if method == "gnn":
        from geneius.gnn.predict import rank_genes

        return rank_genes
    # TODO: "llm" (LLM alone) and "agent" (full agent) baselines
    raise ValueError(f"Unknown method: {method}")


def score(patients: list[dict], rank) -> dict:
    hits = {k: 0 for k in KS}
    for p in patients:
        genes = [c["gene"] for c in rank(p["hpo_ids"], max(KS))]
        truth = set(p["true_genes"])
        for k in KS:
            hits[k] += any(g in truth for g in genes[:k])
    n = len(patients)
    return {"n": n, **{f"hits@{k}": hits[k] / n if n else 0.0 for k in KS}}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", default=str(HERE / "data" / "simulated.jsonl"))
    ap.add_argument("--method", default="overlap", choices=["overlap", "gnn"])
    ap.add_argument("--mode", default="heldout", choices=["heldout", "transductive"])
    ap.add_argument("--heldout", default=str(HERE / "data" / "heldout_diseases.txt"))
    args = ap.parse_args(argv)

    hide = set()
    if args.mode == "heldout":
        hide = set(Path(args.heldout).read_text().split("\n")) - {""}
    result = score(load_jsonl(args.patients), make_ranker(args.method, hide))
    print(f"{args.method} ({args.mode}), n={result['n']}")
    for k in KS:
        print(f"  hits@{k:<2} {result[f'hits@{k}']:.3f}")
    return result


if __name__ == "__main__":
    main()
