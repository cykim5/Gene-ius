"""GNN inference. Must return the same shape as rank_candidates: [{gene, disease, score}]."""
from __future__ import annotations


def rank_genes(hpo_ids: list[str], top_k: int = 10) -> list[dict]:
    raise NotImplementedError("Train the GNN first (geneius/gnn/train.py), or set GENEIUS_RANKER=overlap.")
