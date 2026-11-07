"""ClinVar loading and the 2023 vs current comparison (Stage 2).

Done: parsing protein changes and review stars.
TODO: load_missense(), compare_releases(), and the go/no-go count in eval/go_no_go.py.
Check the column headers in each release first. ClinVar has renamed some
classification columns over time.
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


def load_missense(path) -> "pd.DataFrame":  # noqa: F821
    """TODO: GRCh38 only, missense only, 1+ star. Columns: variant_id, gene, ref, pos, alt,
    classification, stars, diseases (MONDO), hpo_ids."""
    raise NotImplementedError


def compare_releases(old, new) -> "pd.DataFrame":  # noqa: F821
    """TODO: per variant, 2023 class vs current class and stars. Flag VUS reclassified since."""
    raise NotImplementedError
