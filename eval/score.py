"""Score rankers with hits@1, hits@5, hits@10 for the correct gene.

    python eval/score.py --method all
    python eval/score.py --patients eval/data/real_cases.jsonl --method all --mode transductive

Methods: overlap (baseline), gnn, hybrid (reciprocal rank fusion of the two), or all.

Modes:
  heldout       (default for simulated patients) the held-out diseases' phenotype
                annotations are hidden from every ranker. The GNN scores against its
                training graph and must have been trained with these diseases held out.
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
METHODS = ("overlap", "gnn", "hybrid")


def load_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def make_rankers(methods: list[str], hide: set[str], mode: str) -> dict:
    overlap = OverlapRanker(get_kg(), get_hpo(), hide_diseases=hide)
    rankers = {"overlap": overlap.rank_genes}
    if any(m != "overlap" for m in methods):
        from geneius.gnn.predict import check_heldout, rank_genes, rank_hybrid

        graph = "heldout" if mode == "heldout" else "full"
        if mode == "heldout":
            check_heldout(hide)
        rankers["gnn"] = lambda hpo_ids, k: rank_genes(hpo_ids, k, graph=graph)
        rankers["hybrid"] = lambda hpo_ids, k: rank_hybrid(hpo_ids, k, overlap=overlap, graph=graph)
    # TODO: "llm" (LLM alone) and "agent" (full agent) baselines
    return {m: rankers[m] for m in methods}


def score(patients: list[dict], rank) -> dict:
    hits = {k: 0 for k in KS}
    for p in patients:
        genes = [c["gene"] for c in rank(p["hpo_ids"], max(KS))]
        truth = set(p["true_genes"])
        for k in KS:
            hits[k] += any(g in truth for g in genes[:k])
    n = len(patients)
    return {"n": n, **{f"hits@{k}": hits[k] / n if n else 0.0 for k in KS}}


def print_table(results: dict[str, dict], mode: str, patients: str) -> None:
    print(f"{mode} · {patients}")
    print(f"{'method':<8} {'n':>5} " + " ".join(f"{f'hits@{k}':>7}" for k in KS))
    for method, r in results.items():
        print(f"{method:<8} {r['n']:>5} " + " ".join(f"{r[f'hits@{k}']:>7.3f}" for k in KS))


def main(argv=None) -> dict[str, dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", default=str(HERE / "data" / "simulated.jsonl"))
    ap.add_argument("--method", nargs="+", default=["overlap"], choices=[*METHODS, "all"])
    ap.add_argument("--mode", default="heldout", choices=["heldout", "transductive"])
    ap.add_argument("--heldout", default=str(HERE / "data" / "heldout_diseases.txt"))
    args = ap.parse_args(argv)

    methods = list(METHODS) if "all" in args.method else list(dict.fromkeys(args.method))
    hide = set()
    if args.mode == "heldout":
        hide = set(Path(args.heldout).read_text().split("\n")) - {""}
    patients = load_jsonl(args.patients)
    rankers = make_rankers(methods, hide, args.mode)
    results = {m: score(patients, rankers[m]) for m in methods}
    print_table(results, args.mode, args.patients)
    return results


if __name__ == "__main__":
    main()
