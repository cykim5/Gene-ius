"""ClinVar helpers: protein change parsing and review stars.

The full 2023 vs current comparison that built data/processed/variants.csv is in
notebooks/omar_clinvar_vus.ipynb.
"""
from __future__ import annotations

import re

AA3 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
    "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
    "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
_PCHANGE = re.compile(r"\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)")

# ClinVar review status -> stars
STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
}


def parse_protein_change(name: str) -> tuple[str, int, str] | None:
    """'NM_000539.3(RHO):c.68C>A (p.Pro23His)' -> ('P', 23, 'H').

    Returns None for anything that isn't a missense change (nonsense, synonymous, frameshift).
    """
    m = _PCHANGE.search(name or "")
    if not m:
        return None
    ref, pos, alt = m.groups()
    if ref not in AA3 or alt not in AA3 or ref == alt:
        return None
    return AA3[ref], int(pos), AA3[alt]


def review_stars(status: str) -> int:
    return STARS.get((status or "").strip().lower(), 0)
