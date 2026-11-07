"""rank_candidates: HPO terms -> ranked candidate genes (with their best-matching disease).

GENEIUS_RANKER=overlap (default) uses the phenotype overlap baseline.
GENEIUS_RANKER=gnn uses the trained GNN (train with `python -m geneius.gnn.train`).
GENEIUS_RANKER=hybrid fuses the overlap and GNN rankings (reciprocal rank fusion, k=60).
"""
from __future__ import annotations

import os

from .. import config
from ..baseline_overlap import get_overlap_ranker
from . import stubs


def rank_candidates(hpo_ids: list[str], top_k: int = 10) -> list[dict]:
    """Returns [{gene, disease, score}], best first."""
    if config.use_stubs():
        return [dict(x) for x in stubs.RANK_CANDIDATES[:top_k]]
    ranker = os.environ.get("GENEIUS_RANKER", "overlap")
    if ranker == "gnn":
        from ..gnn.predict import rank_genes

        return rank_genes(hpo_ids, top_k)
    if ranker == "hybrid":
        from ..gnn.predict import rank_hybrid

        return rank_hybrid(hpo_ids, top_k)
    return get_overlap_ranker().rank_genes(hpo_ids, top_k)
