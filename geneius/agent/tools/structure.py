"""structural_neighbors (stub) and show_structure.

show_structure returns the teammates' 3D viewer page for the 7 demo variants. Viewer files are
named by the ClinVar protein change (viewer_{GENE}_p{Ref}{Pos}{Alt}.html), not canonical numbering.
"""
from __future__ import annotations

from ... import config
from ...data import demo
from . import stubs


def structural_neighbors(variant: str) -> dict:
    """Known variants nearby in 3D and at matching residues in related proteins, with their labels.
    Returns {variant, neighbors_3d: [{variant, distance, label, mechanism}], related_protein: [...]}."""
    return stubs.structural_neighbors(variant)


def show_structure(variant: str) -> dict:
    """3D viewer page for a demo variant. Returns {variant, viewer_file, clinvar_change} or a message."""
    if not (config.processed_dir() / "predictions.json").exists():
        return stubs.show_structure(variant)
    ids = demo.resolve(variant)
    if not ids:
        return {"variant": variant, "status": "not found"}
    ft = demo.feature_table()
    for vid in ids:
        change = demo.clinvar_changes().get(vid)
        path = demo.viewer_file(ft.at[vid, "gene"], change) if change else None
        if path is not None:
            return {"variant": ft.at[vid, "variant_id"], "variation_id": vid, "clinvar_change": change,
                    "viewer_file": str(path)}
    return {"variant": variant, "status": "3D view available for demo variants only."}
