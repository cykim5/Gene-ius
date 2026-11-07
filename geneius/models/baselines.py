"""Baselines.

| Baseline                         | Why                                         |
| AlphaMissense alone              | number to beat for A                        |
| LoGoFunc alone                   | number to beat for B                        |
| XGBoost on variant features      | do we even need a graph?                    |
| XGBoost + graph features         | strong, fair opponent                       |
| GNN without AlphaMissense        | results aren't borrowed from AlphaMissense  |
| GNN minus one layer at a time    | what each data source adds                  |
| Phenotype overlap                | non-learning baseline for symptom questions (phenotype_overlap.py) |
"""
from __future__ import annotations


def xgboost_baseline(features, labels, use_graph_features: bool = False):
    """TODO: GPU XGBoost (tree_method='hist', device='cuda')."""
    raise NotImplementedError
