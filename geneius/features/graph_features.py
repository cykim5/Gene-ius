"""Variant links for the graph and the matching hand-built features for XGBoost.

No torch here, so eval/reclassification.py can use it next to xgboost.

Residue positions: canonical UniProt numbering. uniprot_pos is used when the sequence has
uniprot_ref there; otherwise position when the sequence has ref_aa there; otherwise the variant
has no residue and gets no paralog or 3D links.

Paralog links: a panel variant at residue r links to 2023 pathogenic paralog variants at the
aligned paralog residue ("same") or one residue either side of it ("near").

3D links: variants in the same protein whose C-alpha atoms are within 10 A, both residues with
pLDDT >= 70. Distance is kept as an edge attribute.

The 3D features count 2023-pathogenic and benign *training* neighbors, so they use labels. A
variant never counts itself, and training-label sets are passed in, so held-out genes contribute
no labels.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from .. import config

NEIGHBOR_ANGSTROMS = 10.0
MIN_PLDDT = 70.0
PARALOG_FEATURES = ["paralog_p_same", "paralog_p_near"]
STRUCTURE_FEATURES = ["struct_p_10a", "struct_b_10a", "struct_p_frac_10a"]


def residue_positions(variants: pd.DataFrame, sequences: dict[str, str]) -> np.ndarray:
    """Canonical UniProt residue per variant, or 0 when it can't be placed."""
    out = np.zeros(len(variants), dtype=int)
    for i, (g, up, ur, pos, ref) in enumerate(zip(variants["gene"], variants["uniprot_pos"], variants["uniprot_ref"],
                                                  variants["position"], variants["ref_aa"])):
        seq = sequences.get(g, "")
        if pd.notna(up) and 0 < int(up) <= len(seq) and seq[int(up) - 1] == ur:
            out[i] = int(up)
        elif pd.notna(pos) and 0 < int(pos) <= len(seq) and seq[int(pos) - 1] == ref:
            out[i] = int(pos)
    return out


def paralog_links(variants: pd.DataFrame, residues: np.ndarray, paralog_variants: pd.DataFrame,
                  maps: dict[tuple[str, str], dict[int, int]]) -> pd.DataFrame:
    """Rows (variant_row, paralog_row, kind) with kind "same" or "near"."""
    by_site: dict[tuple[str, int], list[int]] = {}
    for j, (g, p) in enumerate(zip(paralog_variants["gene"], paralog_variants["pos"])):
        by_site.setdefault((g, int(p)), []).append(j)
    rows = []
    for i, (gene, r) in enumerate(zip(variants["gene"], residues)):
        if not r:
            continue
        for (g, para), m in maps.items():
            if g != gene or r not in m:
                continue
            q = m[r]
            rows += [(i, j, "same") for j in by_site.get((para, q), [])]
            rows += [(i, j, "near") for d in (-1, 1) for j in by_site.get((para, q + d), [])]
    return pd.DataFrame(rows, columns=["variant_row", "paralog_row", "kind"])


def paralog_features(n_variants: int, links: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=range(n_variants), columns=PARALOG_FEATURES)
    for kind, col in (("same", "paralog_p_same"), ("near", "paralog_p_near")):
        counts = links[links["kind"] == kind].groupby("variant_row").size()
        out.loc[counts.index, col] = counts.to_numpy(dtype=float)
    return out


def load_ca_coords() -> dict[str, dict[int, tuple]]:
    """gene -> {residue: (x, y, z, plddt)}. Unpickles only plain Python and numpy types."""

    class _Safe(pickle.Unpickler):
        def find_class(self, module, name):
            if module in {"builtins", "collections", "_codecs"} or module.startswith("numpy"):
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

    with open(config.ca_coords_pkl(), "rb") as f:
        return _Safe(f).load()


def structure_pairs(variants: pd.DataFrame, residues: np.ndarray, coords: dict) -> pd.DataFrame:
    """Rows (a, b, distance) for variant pairs within NEIGHBOR_ANGSTROMS, both directions, no self pairs."""
    rows = []
    for gene, idx in pd.Series(range(len(variants))).groupby(variants["gene"].to_numpy()):
        res = coords.get(gene, {})
        keep = [i for i in idx if residues[i] and residues[i] in res and res[residues[i]][3] >= MIN_PLDDT]
        if len(keep) < 2:
            continue
        xyz = np.array([res[residues[i]][:3] for i in keep], dtype=float)
        dist = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
        a, b = np.nonzero((dist <= NEIGHBOR_ANGSTROMS) & ~np.eye(len(keep), dtype=bool))
        keep = np.asarray(keep)
        rows.append(pd.DataFrame({"a": keep[a], "b": keep[b], "distance": dist[a, b]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["a", "b", "distance"])


def extra_links(variants: pd.DataFrame, is_train: np.ndarray, paralog: bool, structure: bool) -> dict:
    """Paralog and 3D links plus their count features. Label-based counts use only is_train rows.

    variants needs gene, position, ref_aa, uniprot_pos, uniprot_ref, and class_2023.
    """
    if not (paralog or structure):
        return {}
    from ..data.paralogs import family_maps, load_paralog_variants, load_sequences

    residues = residue_positions(variants, load_sequences())
    out: dict = {"features": pd.DataFrame(index=range(len(variants)))}
    if paralog:
        pv = load_paralog_variants()
        links = paralog_links(variants, residues, pv, family_maps())
        out.update(paralog_variants=pv, paralog_links=links)
        out["features"] = pd.concat([out["features"], paralog_features(len(variants), links)], axis=1)
    if structure:
        pairs = structure_pairs(variants, residues, load_ca_coords())
        label = variants["class_2023"].map({"P": 1.0, "B": 0.0}).to_numpy(dtype=float, na_value=np.nan)
        out["structure_pairs"] = pairs
        out["features"] = pd.concat(
            [out["features"], structure_features(len(variants), pairs, np.where(is_train, label, np.nan))], axis=1)
    return out


def structure_features(n_variants: int, pairs: pd.DataFrame, train_label: np.ndarray) -> pd.DataFrame:
    """Counts of pathogenic and benign training neighbors within 10 A. train_label: 1, 0, or NaN (not training)."""
    out = pd.DataFrame(0.0, index=range(n_variants), columns=STRUCTURE_FEATURES)
    if len(pairs):
        lab = train_label[pairs["b"].to_numpy(dtype=int)]
        a = pairs["a"].to_numpy(dtype=int)
        out["struct_p_10a"] = np.bincount(a[lab == 1], minlength=n_variants).astype(float)
        out["struct_b_10a"] = np.bincount(a[lab == 0], minlength=n_variants).astype(float)
    total = out["struct_p_10a"] + out["struct_b_10a"]
    out["struct_p_frac_10a"] = np.where(total > 0, out["struct_p_10a"] / total.where(total > 0, 1.0), np.nan)
    return out
