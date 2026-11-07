"""ESM-2 variant effect scores and protein embeddings (BioNeMo or fair-esm, needs a GPU)."""
from __future__ import annotations


def effect_scores(variants) -> "pd.Series":  # noqa: F821
    """TODO: log-likelihood of mutant vs original amino acid at that position."""
    raise NotImplementedError


def protein_embedding(sequence: str):
    """TODO: mean of ESM-2 per-residue outputs."""
    raise NotImplementedError
