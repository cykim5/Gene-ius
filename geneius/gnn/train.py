"""Train the GNN ranker.

    python eval/simulate_patients.py --n 200   # writes eval/data/heldout_diseases.txt
    python -m geneius.gnn.train               # --ppi adds protein_protein edges

Task: link prediction on disease -> phenotype edges of the non-held-out diseases.
graph_build.training_edges removes the held-out diseases' phenotype edges and the
phenotype-gene edges copied from them. RandomLinkSplit keeps 10% of the remaining edges for
validation and uses 30% of the training edges only as labels, never for message passing.

Negatives are resampled every epoch: each positive edge gets --negatives copies with the disease
or the phenotype swapped for one that has phenotype edges in training (uniform), or one drawn in
proportion to its degree (--neg-sampling degree). Held-out diseases have no training edges and are
never drawn, so they aren't pushed away from their own hidden phenotypes.

Early stopping watches filtered validation hits@10: for each validation edge, the rank of its
disease among all diseases for that phenotype, ignoring other diseases known to have it. That is
the ranking the eval asks for. AUROC against random pairs is logged too.

Defaults (PairNorm, dot product, 1 uniform negative, no protein_protein edges) were chosen on
validation hits@10. The cosine decoder, 10 negatives, degree-proportional negatives, and
protein_protein edges each scored lower. Negatives drawn by
degree, or 10 per edge from the same pool, almost never hit the diseases that have no phenotype
edges, and those then outrank everything.

Writes to $GENEIUS_GNN_DIR (default $GENEIUS_DATA_DIR/gnn/):
  model.pt       weights and model config
  index.json     node key -> row maps, held-out diseases, relations used
  embeddings.pt  decoder-space disease and phenotype embeddings for two inference graphs,
                 plus the decoder's logit scale and bias:
                   heldout  the training graph, for scoring simulated patients
                   full     every edge, for real patients (what rank_candidates uses)
  metrics.json   training log
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from .. import config
from ..data import read_primekg
from .graph_build import training_edges
from .model import REV_TARGET, TARGET, GeneiusGNN, build_graph, node_index, relations, typed_edges

DEFAULT_HELDOUT = config.REPO_ROOT / "eval" / "data" / "heldout_diseases.txt"
VAL_CHUNK = 2048


def _read_heldout(path: Path) -> set[str]:
    if not path.exists():
        print(f"[train] No held-out list at {path}. Training on every disease.", flush=True)
        return set()
    return set(path.read_text().split("\n")) - {""}


def _corrupt(pos: torch.Tensor, k: int, disease_w: torch.Tensor, pheno_w: torch.Tensor,
             gen: torch.Generator) -> torch.Tensor:
    """k negatives per positive: swap the disease or the phenotype, drawn by weight."""
    neg = pos.repeat(1, k)
    n = neg.size(1)
    head = torch.rand(n, generator=gen) < 0.5
    n_head = int(head.sum())
    if n_head:
        neg[0, head] = torch.multinomial(disease_w, n_head, replacement=True, generator=gen)
    if n - n_head:
        neg[1, ~head] = torch.multinomial(pheno_w, n - n_head, replacement=True, generator=gen)
    return neg


def auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Mann-Whitney AUROC. Ties are ordered arbitrarily, which is fine for continuous scores."""
    scores = torch.cat([pos, neg]).double()
    ranks = torch.empty_like(scores)
    ranks[scores.argsort()] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _known_diseases(val_pos: torch.Tensor, all_pos: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per chunk of validation edges: (disease rows, chunk columns) of the other diseases known to
    have that edge's phenotype. They get masked so a correct answer isn't counted as a miss."""
    order = all_pos[1].argsort()
    phenos, counts = all_pos[1, order].unique_consecutive(return_counts=True)
    by_pheno = dict(zip(phenos.tolist(), all_pos[0, order].split(counts.tolist())))
    chunks = []
    for start in range(0, val_pos.size(1), VAL_CHUNK):
        rows, cols = [], []
        for j, (d, t) in enumerate(val_pos[:, start:start + VAL_CHUNK].T.tolist()):
            others = by_pheno.get(t)
            if others is None:
                continue
            others = others[others != d]
            rows.append(others)
            cols.append(torch.full_like(others, j))
        empty = torch.empty(0, dtype=torch.long)
        chunks.append((torch.cat(rows) if rows else empty, torch.cat(cols) if cols else empty))
    return chunks


@torch.no_grad()
def _val_ranking(model: GeneiusGNN, h: dict, val_pos: torch.Tensor, known: list) -> tuple[float, float]:
    """Filtered hits@10 and MRR of the true disease among all diseases, per validation edge."""
    diseases = model.project(h["disease"])
    phenos = model.project(h["phenotype"])
    hits, rr = 0.0, 0.0
    for c, (rows, cols) in enumerate(known):
        pairs = val_pos[:, c * VAL_CHUNK:(c + 1) * VAL_CHUNK]
        scores = diseases @ phenos[pairs[1]].T
        true = scores[pairs[0], torch.arange(pairs.size(1))]
        scores[rows, cols] = float("-inf")
        rank = (scores > true).sum(0) + 1
        hits += float((rank <= 10).sum())
        rr += float((1.0 / rank).sum())
    n = val_pos.size(1)
    return hits / n, rr / n


@torch.no_grad()
def _embed(model: GeneiusGNN, graph) -> dict[str, torch.Tensor]:
    model.eval()
    h = model(graph)
    return {"disease": model.project(h["disease"]).cpu(), "phenotype": model.project(h["phenotype"]).cpu()}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Train the GENEIUS GNN ranker")
    ap.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    ap.add_argument("--out", type=Path, default=None, help="default: $GENEIUS_GNN_DIR or $GENEIUS_DATA_DIR/gnn")
    ap.add_argument("--ppi", action="store_true", help="add protein_protein edges")
    ap.add_argument("--no-pair-norm", action="store_true", help="drop PairNorm (the original design)")
    ap.add_argument("--decoder", default="dot", choices=["dot", "cosine"])
    ap.add_argument("--negatives", type=int, default=1, help="negatives per positive edge")
    ap.add_argument("--neg-sampling", default="uniform", choices=["uniform", "degree"])
    ap.add_argument("--stop-on", default="hits10", choices=["hits10", "auroc"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=20, help="epochs without --stop-on improving")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu or mps")
    ap.add_argument("--log-every", type=int, default=5, help="0 to log only the summary")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = args.out or config.gnn_dir()
    out.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    heldout = _read_heldout(args.heldout)
    rels = relations(include_ppi=args.ppi)
    full_edges = typed_edges(read_primekg(rels), rels)
    index = node_index(full_edges)
    graph = build_graph(typed_edges(training_edges(heldout, rels), rels), index, rels)
    load_seconds = time.perf_counter() - t_load
    num_nodes = {t: len(index[t]) for t in index}
    num_edges = {"__".join(e): int(graph[e].edge_index.size(1)) for e in graph.edge_types}
    print(f"[train] graph built in {load_seconds:.0f}s: nodes {num_nodes}", flush=True)
    print(f"[train] edges {num_edges}", flush=True)
    print(f"[train] held-out diseases: {len(heldout)}", flush=True)

    n_pos = graph[TARGET].edge_index.size(1)
    split = T.RandomLinkSplit(
        num_val=int(0.1 * n_pos) if n_pos >= 10 else 0,
        num_test=0,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=0.0,
        add_negative_train_samples=False,
        edge_types=TARGET,
        rev_edge_types=REV_TARGET,
    )
    train_data, val_data, _ = split(graph)
    supervision = train_data[TARGET].edge_label_index
    val_pos = val_data[TARGET].edge_label_index
    train_pos = val_data[TARGET].edge_index  # every training edge: message passing + supervision

    disease_deg = torch.bincount(train_pos[0], minlength=num_nodes["disease"]).float()
    pheno_deg = torch.bincount(train_pos[1], minlength=num_nodes["phenotype"]).float()
    in_train = (disease_deg > 0).float(), (pheno_deg > 0).float()
    weights = (disease_deg, pheno_deg) if args.neg_sampling == "degree" else in_train
    # Fixed 1:1 uniform negatives so val AUROC means the same thing for every config.
    val_neg = _corrupt(val_pos, 1, *in_train, torch.Generator().manual_seed(args.seed + 1))
    known = _known_diseases(val_pos, graph[TARGET].edge_index)
    train_gen = torch.Generator().manual_seed(args.seed + 2)
    print(
        f"[train] supervision edges {supervision.size(1)}, "
        f"message-passing {train_data[TARGET].edge_index.size(1)}, val {val_pos.size(1)}; "
        f"pair_norm {not args.no_pair_norm}, decoder {args.decoder}, "
        f"{args.negatives} {args.neg_sampling} negatives per edge, early stop on {args.stop_on}",
        flush=True,
    )

    model = GeneiusGNN(num_nodes, graph.metadata(), args.hidden, args.decoder, not args.no_pair_norm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_data, val_data = train_data.to(device), val_data.to(device)
    pos = supervision.to(device)
    val_pos, val_neg = val_pos.to(device), val_neg.to(device)
    labels = torch.cat([torch.ones(pos.size(1)), torch.zeros(pos.size(1) * args.negatives)]).to(device)

    history, best_key, best, best_state, bad = [], None, {}, None, 0
    t_train = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        h = model(train_data)
        neg = _corrupt(supervision, args.negatives, *weights, train_gen).to(device)
        logits = torch.cat([model.logits(h, pos), model.logits(h, neg)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()
        loss_value = loss.item()

        val = {"auroc": float("nan"), "hits@10": float("nan"), "mrr": float("nan")}
        if val_pos.numel():
            model.eval()
            with torch.no_grad():
                hv = model(val_data)
                val["auroc"] = auroc(model.logits(hv, val_pos).cpu(), model.logits(hv, val_neg).cpu())
                val["hits@10"], val["mrr"] = _val_ranking(model, hv, val_pos, known)
        elapsed = time.perf_counter() - t_train
        history.append({"epoch": epoch, "loss": round(loss_value, 5),
                        **{f"val_{k}": round(v, 5) for k, v in val.items()}, "seconds": round(elapsed, 1)})
        if args.log_every and (epoch == 1 or epoch % args.log_every == 0):
            print(
                f"[train] epoch {epoch:3d}  loss {loss_value:.4f}  val_auroc {val['auroc']:.4f}  "
                f"val_hits@10 {val['hits@10']:.4f}  val_mrr {val['mrr']:.4f}  {elapsed:6.1f}s",
                flush=True,
            )

        if not val_pos.numel():
            best, best_state = {"epoch": epoch}, None
            continue
        key = (val["hits@10"], val["mrr"]) if args.stop_on == "hits10" else (val["auroc"],)
        if best_key is None or key > best_key:
            best_key, best, bad = key, {"epoch": epoch, **val}, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= args.patience:
                break
    train_seconds = time.perf_counter() - t_train
    if best_state is not None:
        model.load_state_dict(best_state)

    full_graph = build_graph(full_edges, index, rels).to(device)
    scale, bias = (float(torch.as_tensor(v).detach()) for v in model.scale_bias())
    embeddings = {
        "heldout": _embed(model, graph.to(device)),
        "full": _embed(model, full_graph),
        "logit_scale": scale,
        "logit_bias": bias,
    }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": args.hidden,
            "decoder": args.decoder,
            "pair_norm": not args.no_pair_norm,
            "num_nodes": num_nodes,
            "metadata": [list(graph.metadata()[0]), [list(e) for e in graph.metadata()[1]]],
        },
        out / "model.pt",
    )
    torch.save(embeddings, out / "embeddings.pt")
    (out / "index.json").write_text(json.dumps({**index, "heldout": sorted(heldout), "relations": rels}))
    best_val = {k: round(v, 5) for k, v in best.items() if k != "epoch"}
    metrics = {
        "load_seconds": round(load_seconds, 1),
        "train_seconds": round(train_seconds, 1),
        "epochs_run": len(history),
        "best_epoch": best.get("epoch"),
        "best_val": best_val or None,
        "include_ppi": args.ppi,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "heldout_diseases": len(heldout),
        "seed": args.seed,
        "args": {k: str(v) for k, v in vars(args).items()},
        "history": history,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    summary = (
        f"val hits@10 {best_val['hits@10']:.4f}, MRR {best_val['mrr']:.4f}, AUROC {best_val['auroc']:.4f}"
        if best_val else "no val edges"
    )
    print(
        f"[train] done: {len(history)} epochs in {train_seconds:.0f}s. Best epoch {best.get('epoch')}: "
        f"{summary}. Saved to {out}",
        flush=True,
    )
    return metrics


if __name__ == "__main__":
    main()
