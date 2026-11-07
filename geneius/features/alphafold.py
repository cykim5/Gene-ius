"""AlphaFold structure features: pLDDT, buried vs surface, distance to functional sites."""
from __future__ import annotations


def structure_features(variants) -> "pd.DataFrame":  # noqa: F821
    """TODO: pLDDT (B-factor column), relative solvent accessibility (FreeSASA or DSSP),
    3D distance to the nearest annotated site."""
    raise NotImplementedError
