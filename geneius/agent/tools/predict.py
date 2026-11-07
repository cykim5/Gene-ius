"""gnn_predict and gnn_score. Stubs until models/gnn.py is trained."""
from __future__ import annotations

from . import stubs


def gnn_predict(variant: str) -> dict:
    """Outputs A to D with confidence.
    Returns {variant, harmful, mechanism: {class: p}, disease: {name: p}, treatment, confidence}."""
    return stubs.gnn_predict(variant)


def gnn_score(source: str, target: str) -> dict:
    """GNN score for any proposed link. Used to guide the agent's graph search.
    Returns {source, target, score}."""
    return stubs.gnn_score(source, target)
