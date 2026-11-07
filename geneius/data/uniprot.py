"""UniProt ID mapping and the residue check (Stage 2).

The residue check matters: ClinVar numbers residues on a RefSeq/MANE transcript,
which usually but not always matches UniProt's canonical isoform. A mismatch puts
features on the wrong residue without any error.
"""
from __future__ import annotations


def gene_to_uniprot(genes: list[str]) -> dict[str, str]:
    """TODO: HGNC symbol -> reviewed UniProt accession (e.g. RHO -> P08100)."""
    raise NotImplementedError


def residue_matches(uniprot_id: str, pos: int, ref: str) -> bool:
    """TODO: True if UniProt's sequence has amino acid `ref` at position `pos` (1-based)."""
    raise NotImplementedError
