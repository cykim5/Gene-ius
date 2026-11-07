"""Graph construction and the GNN itself, shared by train.py and predict.py.

Node types: gene, disease, phenotype. Genes and diseases are keyed by their PrimeKG name and
phenotypes by HPO id, the same keys data.py uses, so results line up with get_kg().
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import PairNorm, SAGEConv, to_hetero

from ..data import _edges, get_hpo, norm_hpo

# PrimeKG relation -> (x_type, y_type, edge type in the graph)
RELATIONS = {
    "disease_phenotype_positive": ("disease", "effect/phenotype", ("disease", "has_phenotype", "phenotype")),
    "disease_protein": ("disease", "gene/protein", ("disease", "associated_with", "gene")),
    "phenotype_protein": ("effect/phenotype", "gene/protein", ("phenotype", "associated_with", "gene")),
    "phenotype_phenotype": ("effect/phenotype", "effect/phenotype", ("phenotype", "is_a", "phenotype")),
    "disease_disease": ("disease", "disease", ("disease", "related_to", "disease")),
    "protein_protein": ("gene/protein", "gene/protein", ("gene", "interacts_with", "gene")),
}
NODE_TYPES = ("gene", "disease", "phenotype")
TARGET = ("disease", "has_phenotype", "phenotype")
REV_TARGET = ("phenotype", "rev_has_phenotype", "disease")


def relations(include_ppi: bool) -> list[str]:
    return [r for r in RELATIONS if include_ppi or r != "protein_protein"]


def _keys(node_type: str, ids: pd.Series, names: pd.Series) -> np.ndarray:
    if node_type != "phenotype":
        return names.to_numpy(dtype=object)
    hpo = get_hpo()
    lookup = {i: hpo.primary(norm_hpo(i)) for i in pd.unique(ids)}
    return ids.map(lookup).to_numpy(dtype=object)


def typed_edges(df: pd.DataFrame, rels: list[str]) -> dict[tuple, pd.DataFrame]:
    """PrimeKG rows -> {edge_type: DataFrame(src, dst)} keyed by node name / HPO id."""
    out = {}
    for rel in rels:
        x_type, y_type, etype = RELATIONS[rel]
        e = _edges(df, rel, x_type, y_type)
        out[etype] = pd.DataFrame({
            "src": _keys(etype[0], e.a_id, e.a_name),
            "dst": _keys(etype[2], e.b_id, e.b_name),
        }).drop_duplicates()
    return out


def node_index(edges: dict[tuple, pd.DataFrame]) -> dict[str, list[str]]:
    keys: dict[str, set] = {t: set() for t in NODE_TYPES}
    for (src, _, dst), e in edges.items():
        keys[src].update(e.src)
        keys[dst].update(e.dst)
    return {t: sorted(k) for t, k in keys.items()}


def build_graph(edges: dict[tuple, pd.DataFrame], index: dict[str, list[str]], rels: list[str]) -> HeteroData:
    """Every edge type is present even when empty, so all graphs share one model signature."""
    row = {t: {k: i for i, k in enumerate(keys)} for t, keys in index.items()}
    data = HeteroData()
    for t in NODE_TYPES:
        data[t].num_nodes = len(index[t])
    for rel in rels:
        etype = RELATIONS[rel][2]
        e = edges.get(etype)
        if e is None or e.empty:
            data[etype].edge_index = torch.empty((2, 0), dtype=torch.long)
            continue
        src = e.src.map(row[etype[0]])
        dst = e.dst.map(row[etype[2]])
        ok = src.notna() & dst.notna()
        pairs = np.stack([src[ok].to_numpy(dtype=np.int64), dst[ok].to_numpy(dtype=np.int64)])
        data[etype].edge_index = torch.from_numpy(pairs).unique(dim=1)
    return T.ToUndirected()(data)


class _SAGE(nn.Module):
    def __init__(self, hidden: int, pair_norm: bool):
        super().__init__()
        self.conv1 = SAGEConv(hidden, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.norm1 = PairNorm() if pair_norm else nn.Identity()
        self.norm2 = PairNorm() if pair_norm else nn.Identity()

    def forward(self, x, edge_index):
        x = self.norm1(self.conv1(x, edge_index)).relu()
        return self.norm2(self.conv2(x, edge_index))


class GeneiusGNN(nn.Module):
    """2-layer heterogeneous GraphSAGE.

    Genes and phenotypes get learned ID embeddings. Every disease starts from the same learned
    vector, so a disease is described only by its neighbors. Held-out diseases have no phenotype
    edges in training, and a per-disease embedding for them would never move from its random init.

    Because every disease starts from that one vector, disease embeddings come out nearly parallel
    (mean pairwise cosine ~0.99). PairNorm after each layer centers each node type, so diseases
    differ in direction. Without it, a dot-product model separates them by norm alone, and a few
    large-norm diseases outrank everything for every phenotype.

    decoder="cosine" scores pairs by scale * cos(h_disease, h_term) + bias instead of a dot product.
    """

    def __init__(self, num_nodes: dict[str, int], metadata, hidden: int = 128, decoder: str = "dot",
                 pair_norm: bool = True):
        super().__init__()
        if decoder not in ("cosine", "dot"):
            raise ValueError(f"decoder must be 'cosine' or 'dot', not {decoder!r}")
        self.decoder = decoder
        self.gene = nn.Embedding(num_nodes["gene"], hidden)
        self.phenotype = nn.Embedding(num_nodes["phenotype"], hidden)
        self.disease = nn.Parameter(torch.empty(1, hidden))
        for w in (self.gene.weight, self.phenotype.weight, self.disease):
            nn.init.normal_(w, std=0.1)
        self.encoder = to_hetero(_SAGE(hidden, pair_norm), metadata, aggr="sum")
        if decoder == "cosine":
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
            self.logit_bias = nn.Parameter(torch.zeros(()))

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x = {
            "gene": self.gene.weight,
            "phenotype": self.phenotype.weight,
            "disease": self.disease.expand(data["disease"].num_nodes, -1),
        }
        return self.encoder(x, data.edge_index_dict)

    def project(self, emb: torch.Tensor) -> torch.Tensor:
        """Embeddings in the space the decoder takes dot products in."""
        return F.normalize(emb, dim=-1) if self.decoder == "cosine" else emb

    def scale_bias(self) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        if self.decoder == "cosine":
            return self.logit_scale.exp().clamp(max=100.0), self.logit_bias
        return 1.0, 0.0

    def logits(self, h: dict[str, torch.Tensor], pairs: torch.Tensor) -> torch.Tensor:
        """Logits for (disease, phenotype) index pairs."""
        d = self.project(h["disease"])[pairs[0]]
        p = self.project(h["phenotype"])[pairs[1]]
        scale, bias = self.scale_bias()
        return scale * (d * p).sum(-1) + bias
