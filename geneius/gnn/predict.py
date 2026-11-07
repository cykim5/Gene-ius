"""GNN inference. Must return the same shape as rank_candidates: [{gene, disease, score}].

Disease score = mean over the patient's HPO terms of sigmoid(decoder logit), skipping terms
that aren't in the graph. The logit is scale * (h_disease . h_term) + bias on the saved
decoder-space embeddings (unit vectors for the cosine decoder). A gene gets the score of its
best disease, as in baseline_overlap.py.

graph="full" scores against every edge (real patients, what rank_candidates uses).
graph="heldout" scores against the training graph, where the held-out diseases' phenotype
edges are gone (simulated patients in eval/score.py).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch

from .. import config
from ..baseline_overlap import OverlapRanker, get_overlap_ranker
from ..data import get_hpo, get_kg

RRF_K = 60
_ALL = 10**9


class GNNRanker:
    def __init__(self, gnn_dir: Path, graph: str):
        index = json.loads((gnn_dir / "index.json").read_text())
        saved = torch.load(gnn_dir / "embeddings.pt")
        self.diseases: list[str] = index["disease"]
        self.heldout = frozenset(index["heldout"])
        self.pheno_row = {t: i for i, t in enumerate(index["phenotype"])}
        self.disease_emb: torch.Tensor = saved[graph]["disease"]
        self.pheno_emb: torch.Tensor = saved[graph]["phenotype"]
        self.scale = saved.get("logit_scale", 1.0)
        self.bias = saved.get("logit_bias", 0.0)

    def score_diseases(self, hpo_ids: list[str]) -> dict[str, float]:
        hpo = get_hpo()
        rows = [self.pheno_row[t] for t in dict.fromkeys(hpo.primary(x) for x in hpo_ids) if t in self.pheno_row]
        if not rows:
            return {}
        logits = self.scale * (self.disease_emb @ self.pheno_emb[rows].T) + self.bias
        scores = torch.sigmoid(logits).mean(1)
        return dict(zip(self.diseases, scores.tolist()))

    def rank_genes(self, hpo_ids: list[str], top_k: int = 10) -> list[dict]:
        disease_genes = get_kg().disease_genes
        best: dict[str, tuple[float, str]] = {}
        for d, s in self.score_diseases(hpo_ids).items():
            for g in disease_genes.get(d, ()):
                if g not in best or s > best[g][0] or (s == best[g][0] and d < best[g][1]):
                    best[g] = (s, d)
        ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))[:top_k]
        return [{"gene": g, "disease": d, "score": round(s, 4)} for g, (s, d) in ranked]


@lru_cache(maxsize=4)
def _cached_ranker(gnn_dir: str, graph: str) -> GNNRanker:
    return GNNRanker(Path(gnn_dir), graph)


def get_gnn_ranker(graph: str = "full") -> GNNRanker:
    if graph not in ("full", "heldout"):
        raise ValueError(f"graph must be 'full' or 'heldout', not {graph!r}")
    d = config.gnn_dir()
    if not (d / "embeddings.pt").exists():
        raise FileNotFoundError(
            f"No trained GNN in {d}. Run `python -m geneius.gnn.train`, or set GENEIUS_RANKER=overlap."
        )
    return _cached_ranker(str(d.resolve()), graph)


def clear_cache() -> None:
    _cached_ranker.cache_clear()


def load_model():
    """The trained network itself. Ranking only needs the saved embeddings."""
    from .model import GeneiusGNN

    ckpt = torch.load(config.gnn_dir() / "model.pt")
    node_types, edge_types = ckpt["metadata"]
    model = GeneiusGNN(
        ckpt["num_nodes"], (node_types, [tuple(e) for e in edge_types]), ckpt["hidden"],
        ckpt.get("decoder", "dot"), ckpt.get("pair_norm", False),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def check_heldout(hide: set[str]) -> None:
    """Raise if any disease hidden in eval was visible to the GNN during training."""
    leaked = set(hide) - get_gnn_ranker("heldout").heldout
    if leaked:
        raise ValueError(
            f"{len(leaked)} eval diseases were not held out when the GNN was trained "
            f"(e.g. {sorted(leaked)[0]!r}). Retrain with the same list: "
            f"python -m geneius.gnn.train --heldout <file>"
        )


def rank_genes(hpo_ids: list[str], top_k: int = 10, graph: str = "full") -> list[dict]:
    return get_gnn_ranker(graph).rank_genes(hpo_ids, top_k)


def fuse(rankings: list[list[dict]], top_k: int, k: int = RRF_K) -> list[dict]:
    """Reciprocal rank fusion. A gene's disease comes from the list that ranked it highest."""
    score: dict[str, float] = {}
    best: dict[str, tuple[int, str]] = {}
    for ranking in rankings:
        for rank, c in enumerate(ranking, 1):
            g = c["gene"]
            score[g] = score.get(g, 0.0) + 1.0 / (k + rank)
            if g not in best or rank < best[g][0]:
                best[g] = (rank, c["disease"])
    ranked = sorted(score, key=lambda g: (-score[g], g))[:top_k]
    return [{"gene": g, "disease": best[g][1], "score": round(score[g], 6)} for g in ranked]


def rank_hybrid(
    hpo_ids: list[str], top_k: int = 10, overlap: OverlapRanker | None = None, graph: str = "full"
) -> list[dict]:
    """RRF (k=60) over the full overlap and GNN rankings."""
    overlap = overlap or get_overlap_ranker()
    return fuse([overlap.rank_genes(hpo_ids, _ALL), rank_genes(hpo_ids, _ALL, graph)], top_k)
