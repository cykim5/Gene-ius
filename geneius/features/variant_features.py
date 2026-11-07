"""Per-variant features from the amino-acid change, plus empty slots for teammates' features.

variant_features(df) takes variant rows with ref_aa and alt_aa and returns one row per variant.
Placeholder columns stay NaN until their data lands (AlphaMissense, AlphaFold pLDDT, UniProt
regions). model_columns() picks only columns that have values, so filling a placeholder makes the
models use it with no other change.

Relative position needs protein length, which isn't available yet (data/uniprot.py), so it's
not computed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AA = "ARNDCQEGHILKMFPSTWYV"
# BLOSUM62, rows and columns in AA order
_BLOSUM62 = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
BLOSUM62 = {
    (a, b): int(v)
    for a, row in zip(AA, _BLOSUM62.strip().splitlines())
    for b, v in zip(AA, row.split())
}
CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0}
# Kyte-Doolittle hydropathy
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

CHANGE_COLUMNS = [
    "blosum62", "charge_change", "hydropathy_change", "to_proline", "from_proline", "to_glycine", "from_glycine",
]
PLACEHOLDER_COLUMNS = [
    "alphamissense", "plddt", "in_domain", "in_transmembrane", "in_active_site", "in_binding_site",
    "in_disulfide", "in_modified_residue",
]
FEATURE_COLUMNS = CHANGE_COLUMNS + PLACEHOLDER_COLUMNS


def variant_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per input row, indexed like df. Unknown amino acids give NaN change features."""
    ref = df["ref_aa"].astype(str).str.upper()
    alt = df["alt_aa"].astype(str).str.upper()
    known = ref.isin(list(AA)) & alt.isin(list(AA))
    out = pd.DataFrame(index=df.index)
    out["blosum62"] = [BLOSUM62.get((r, a), np.nan) for r, a in zip(ref, alt)]
    out["charge_change"] = alt.map(CHARGE).fillna(0.0) - ref.map(CHARGE).fillna(0.0)
    out["hydropathy_change"] = alt.map(HYDROPATHY) - ref.map(HYDROPATHY)
    out["to_proline"] = (alt == "P").astype(float)
    out["from_proline"] = (ref == "P").astype(float)
    out["to_glycine"] = (alt == "G").astype(float)
    out["from_glycine"] = (ref == "G").astype(float)
    out.loc[~known, CHANGE_COLUMNS] = np.nan
    for col in PLACEHOLDER_COLUMNS:
        out[col] = np.nan
    return out


def model_columns(features: pd.DataFrame) -> list[str]:
    """Feature columns that have at least one value."""
    return [c for c in FEATURE_COLUMNS if c in features and features[c].notna().any()]
