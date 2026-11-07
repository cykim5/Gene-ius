"""3D-neighbor and related-protein edges between variants."""
from __future__ import annotations

NEIGHBOR_ANGSTROMS = 10.0
MIN_PLDDT = 70


def neighbor_edges(uniprot_id: str, positions: list[int]) -> list[tuple[int, int, float]]:
    """TODO: pairs of variant residues whose alpha carbons are within NEIGHBOR_ANGSTROMS.
    Returns (pos_a, pos_b, distance). Skip residues below MIN_PLDDT."""
    raise NotImplementedError


def related_protein_edges(gene: str, family: list[str]):
    """TODO: align sequences in the family and connect variants at matching positions."""
    raise NotImplementedError
