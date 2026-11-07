"""Training graph with no test leakage.

For each held-out disease we drop:
  1. its disease-phenotype edges
  2. phenotype-gene edges between its genes and its phenotypes. HPO derives most
     gene-phenotype links from disease annotations, so keeping these would leak the answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import get_hpo, get_kg, norm_hpo, read_primekg


def training_edges(heldout: set[str], relations: list[str] | None = None) -> pd.DataFrame:
    """PrimeKG edges minus the leaks above. Pass relations to load only those edge types."""
    df = read_primekg(relations).reset_index(drop=True)
    kg, hpo = get_kg(), get_hpo()

    is_dp = df.relation.isin(["disease_phenotype_positive", "disease_phenotype_negative"])
    touches = ((df.x_type == "disease") & df.x_name.isin(heldout)) | (
        (df.y_type == "disease") & df.y_name.isin(heldout)
    )
    drop = (is_dp & touches).to_numpy().copy()

    leak = {(g, t) for d in heldout for g in kg.disease_genes.get(d, ()) for t in kg.disease_hpo.get(d, ())}
    pp = np.flatnonzero((df.relation == "phenotype_protein").to_numpy())
    if leak and len(pp):
        sub = df.iloc[pp]
        genes = sub.x_name.where(sub.x_type == "gene/protein", sub.y_name)
        phenos = sub.x_id.where(sub.x_type == "effect/phenotype", sub.y_id)
        hit = [(g, hpo.primary(norm_hpo(p))) in leak for g, p in zip(genes, phenos)]
        drop[pp[np.array(hit, dtype=bool)]] = True

    return df[~drop]
