"""Heterogeneous GNN with four linked outputs.

  Encoder: heterogeneous GraphSAGE, 2 layers
  A: probability harmful  (done)
  B: probabilities over MECHANISMS  (to do)
  C: score for each of the gene's diseases, plus predicted HPO terms (to do, takes B as input)
  D: treatment strategy from B and C (to do)

Output A: node classification on variant nodes, loss only on train labels (class_2023).
Validation is ~10% of train variants, held out by whole gene, never the test set. Validation
variants get no disease edges, like test variants. Each epoch also drops half of the remaining
variant-disease edges, so the model can't lean on edges that test variants never have.
Early stopping on validation AUPRC. Full-batch training (the graph fits on a laptop CPU).

    python -m geneius.models.gnn     # train, then write predictions for every variant
"""
from __future__ import annotations

import copy
import json
import random
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, to_hetero

from .. import config
from ..graph.build import VARIANT_DISEASE, VariantGraph

REV_VARIANT_DISEASE = ("disease", "rev_linked_to", "variant")


def default_dir() -> Path:
    return config.gnn_a_dir()


class _SAGE(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.conv1 = SAGEConv(hidden, hidden)
        self.conv2 = SAGEConv(hidden, hidden)

    def forward(self, x, edge_index):
        return self.conv2(self.conv1(x, edge_index).relu(), edge_index)


class VariantGNN(nn.Module):
    def __init__(self, num_nodes: dict[str, int], n_features: int, metadata, hidden: int = 128):
        super().__init__()
        self.emb = nn.ModuleDict({t: nn.Embedding(num_nodes[t], hidden) for t in ("gene", "disease", "phenotype")})
        self.variant_in = nn.Linear(n_features, hidden)
        self.encoder = to_hetero(_SAGE(hidden), metadata, aggr="sum")
        self.head = nn.Linear(hidden, 1)

    def forward(self, data) -> torch.Tensor:
        x = {t: e.weight for t, e in self.emb.items()}
        x["variant"] = self.variant_in(data["variant"].x)
        return self.head(self.encoder(x, data.edge_index_dict)["variant"]).squeeze(-1)


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def gene_split(genes: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Train-variant indices for validation: whole genes, about `frac` of train, both classes present."""
    rng = random.Random(seed)
    for _ in range(100):
        order = sorted(set(genes[train_idx]))
        rng.shuffle(order)
        val, chosen = [], set()
        for g in order:
            if len(val) >= frac * len(train_idx):
                break
            chosen.add(g)
            val = [i for i in train_idx if genes[i] in chosen]
        val = np.array(val, dtype=int)
        if len(val) and 0 < labels[val].sum() < len(val):
            return val
    return np.array(rng.sample(list(train_idx), max(1, int(frac * len(train_idx)))), dtype=int)


def _without_edges(data, keep: torch.Tensor):
    out = copy.copy(data)
    fwd = data[VARIANT_DISEASE].edge_index[:, keep]
    out[VARIANT_DISEASE].edge_index = fwd
    out[REV_VARIANT_DISEASE].edge_index = fwd.flip(0)
    return out


def train(graph: VariantGraph, genes: list[str], out_dir: Path | None = None, hidden: int = 128,
          lr: float = 0.005, epochs: int = 200, patience: int = 20, seed: int = 0, drop_edge: float = 0.5,
          log_every: int = 10) -> dict:
    """Trains Output A, writes model.pt, predictions.csv, and metrics.json, and returns the metrics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = out_dir or default_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = graph.data
    y = data["variant"].y
    train_mask = data["variant"].train_mask
    train_idx = np.flatnonzero(train_mask.numpy())
    gene_arr = np.asarray(genes)
    val_idx = gene_split(gene_arr, np.nan_to_num(y.numpy()), train_idx, 0.1, seed)
    fit_idx = np.setdiff1d(train_idx, val_idx)
    fit_t, val_t = torch.from_numpy(fit_idx), torch.from_numpy(val_idx)

    vd = data[VARIANT_DISEASE].edge_index
    not_val = ~torch.isin(vd[0], val_t)
    eval_data = _without_edges(data, not_val)
    n_pos = float(y[fit_t].sum())
    pos_weight = torch.tensor((len(fit_idx) - n_pos) / max(n_pos, 1.0))

    num_nodes = {t: data[t].num_nodes for t in data.node_types}
    model = VariantGNN(num_nodes, data["variant"].x.size(1), data.metadata(), hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed + 1)
    history, best, best_state, bad = [], -1.0, None, 0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        keep = not_val & (torch.rand(vd.size(1), generator=gen) >= drop_edge)
        logits = model(_without_edges(data, keep))
        loss = F.binary_cross_entropy_with_logits(logits[fit_t], y[fit_t], pos_weight=pos_weight)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            p_val = torch.sigmoid(model(eval_data))[val_t].numpy()
        val_auprc = auprc(y[val_t].numpy(), p_val)
        history.append({"epoch": epoch, "loss": round(loss.item(), 5), "val_auprc": round(val_auprc, 5),
                        "seconds": round(time.perf_counter() - t0, 1)})
        if log_every and (epoch == 1 or epoch % log_every == 0):
            print(f"[gnn] epoch {epoch:3d}  loss {loss.item():.4f}  val_auprc {val_auprc:.4f}  "
                  f"{time.perf_counter() - t0:5.1f}s", flush=True)
        if val_auprc > best:
            best, best_state, bad = val_auprc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_all = torch.sigmoid(model(data)).numpy()

    pd.DataFrame({"variant_id": graph.variant_ids, "gene": genes, "harmful": np.round(p_all, 5)}).to_csv(
        out_dir / "predictions.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "hidden": hidden, "num_nodes": num_nodes,
                "n_features": data["variant"].x.size(1), "feature_columns": graph.feature_columns},
               out_dir / "model.pt")
    metrics = {
        "train_seconds": round(time.perf_counter() - t0, 1),
        "epochs_run": len(history),
        "best_val_auprc": round(best, 5),
        "val_genes": sorted(str(g) for g in set(gene_arr[val_idx])),
        "n_fit": int(len(fit_idx)),
        "n_val": int(len(val_idx)),
        "val_positive_rate": round(float(y[val_t].mean()), 4),
        "feature_columns": graph.feature_columns,
        "disease_match": graph.disease_match,
        "seed": seed,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    clear_cache()
    print(f"[gnn] done: {len(history)} epochs, best val AUPRC {best:.4f} on genes {metrics['val_genes']} "
          f"(n={len(val_idx)}). Saved to {out_dir}", flush=True)
    return metrics


@lru_cache(maxsize=2)
def _predictions(path: str) -> dict[str, float]:
    df = pd.read_csv(path, dtype={"variant_id": str})
    return dict(zip(df["variant_id"], df["harmful"]))


def clear_cache() -> None:
    _predictions.cache_clear()


def _confidence(p: float) -> str:
    margin = abs(p - 0.5)
    return "high" if margin >= 0.35 else "medium" if margin >= 0.2 else "low"


def predict(variant_ids: list[str]) -> list[dict]:
    """Returns [{variant, harmful, mechanism, disease, treatment, confidence}]. Only A exists so far."""
    path = default_dir() / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"No trained GNN at {path.parent}. Run `python -m geneius.models.gnn`.")
    preds = _predictions(str(path))
    out = []
    for vid in variant_ids:
        p = preds.get(str(vid))
        out.append({
            "variant": str(vid),
            "harmful": None if p is None else round(float(p), 4),
            "mechanism": None,
            "disease": None,
            "treatment": None,
            "confidence": None if p is None else _confidence(float(p)),
        })
    return out


def no_alphamissense_dir() -> Path:
    return config.gnn_a_dir().with_name("gnn_a_no_alphamissense")


def main(argv=None) -> None:
    import argparse

    from ..data import load_variants
    from ..graph.build import build_graph

    ap = argparse.ArgumentParser(description="Train GNN output A")
    ap.add_argument("--no-alphamissense", action="store_true", help="drop AlphaMissense from the variant features")
    ap.add_argument("--am-impute", action="store_true",
                    help="no am_missing indicator: missing AlphaMissense is just the train median")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: config.gnn_a_dir())")
    args = ap.parse_args(argv)
    variants = load_variants()
    drop = ("alphamissense",) if args.no_alphamissense else ()
    graph = build_graph(variants, drop_features=drop, am_indicator=not args.am_impute)
    out = args.out or (no_alphamissense_dir() if args.no_alphamissense else None)
    train(graph, variants["gene"].tolist(), out_dir=out)


if __name__ == "__main__":
    main()
