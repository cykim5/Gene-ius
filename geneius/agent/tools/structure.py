"""structural_neighbors and show_structure. Stubs until graph/structural_edges.py exists."""
from __future__ import annotations

from . import stubs


def structural_neighbors(variant: str) -> dict:
    """Known variants nearby in 3D and at matching residues in related proteins, with their labels.
    Returns {variant, neighbors_3d: [{variant, distance, label, mechanism}], related_protein: [...]}."""
    return stubs.structural_neighbors(variant)


def show_structure(variant: str) -> dict:
    """Payload for the 3D viewer. Returns {variant, uniprot_id, highlight: [pos], neighbors: [pos]}."""
    return stubs.show_structure(variant)
