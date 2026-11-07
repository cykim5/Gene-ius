"""Heterogeneous GNN with four linked outputs.

  Encoder: heterogeneous GraphSAGE, 2-3 layers (try HGT if time allows)
  A: probability harmful
  B: probabilities over MECHANISMS
  C: score for each of the gene's diseases, plus predicted HPO terms (takes B as input)
  D: treatment strategy from B and C (rules plus a small learned layer where labels exist)

Training: loss = A + B + C (+ D), each only on variants with that label.
Mini-batches with NeighborLoader. Validation is 10% of labeled 2023 variants, never the test set.
VUS stay in the graph without labels. Fix seeds and log settings and data versions.
"""
from __future__ import annotations


def train(graph, out_dir="checkpoints"):
    raise NotImplementedError


def predict(variant_ids: list[str]) -> list[dict]:
    """Returns [{variant, harmful, mechanism, disease, treatment, confidence}]."""
    raise NotImplementedError
