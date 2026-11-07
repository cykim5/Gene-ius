"""UniProt region flags and simple sequence features."""
from __future__ import annotations


def region_flags(variants) -> "pd.DataFrame":  # noqa: F821
    """TODO: yes/no for domain, membrane segment, active site, binding site, disulfide, modified residue."""
    raise NotImplementedError


def relative_position(pos: int, length: int) -> float:
    return pos / length
