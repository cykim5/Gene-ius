"""Paralogs of the panel genes, their 2023 pathogenic variants, and residue alignment.

paralog_pathogenic_2023.csv lists pathogenic variants in related genes but not which panel gene
each belongs to, so the families are written out here. Sequences are canonical reviewed human
UniProt entries, cached in paralog_sequences.json:

    python -m geneius.data.paralogs     # fetch sequences from UniProt (needs network)

Each panel protein is aligned to each of its paralogs (global, BLOSUM62), giving a map from panel
residue to paralog residue.
"""
from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from functools import lru_cache

import pandas as pd

from .. import config

PARALOG_FAMILIES = {
    "ABCA4": ["ABCA1", "ABCA3", "ABCA12"],
    "CACNA1C": ["CACNA1A", "CACNA1D", "CACNA1E", "CACNA1F", "CACNA1S"],
    "GLA": ["NAGA"],
    "GUCY2D": ["NPR2"],
    "KCNH2": ["KCNH1", "KCNH5"],
    "KCNJ2": ["KCNJ1", "KCNJ5", "KCNJ6", "KCNJ10", "KCNJ11", "KCNJ13"],
    "KCNQ1": ["KCNQ2", "KCNQ3", "KCNQ4", "KCNQ5"],
    "LMNA": ["LMNB1", "LMNB2"],
    "MYBPC3": ["MYBPC1"],
    "MYH7": ["MYH2", "MYH3", "MYH6", "MYH8"],
    "PAX6": ["PAX2", "PAX3", "PAX7"],
    "RHO": ["OPN1LW", "OPN1MW"],
    "SCN5A": ["SCN1A", "SCN2A", "SCN3A", "SCN4A", "SCN8A", "SCN9A", "SCN11A"],
}
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"


def load_paralog_variants() -> pd.DataFrame:
    """2023 pathogenic paralog variants: variant_id, gene, pos, ref, alt, stars."""
    df = pd.read_csv(config.paralog_variants_csv())
    df = df[df["cls"] == "P"]
    return pd.DataFrame({
        "variant_id": "paralog:" + df["VariationID"].astype(str),
        "gene": df["gene"].astype(str),
        "pos": df["pos"].astype(int),
        "ref": df["ref"].astype(str),
        "alt": df["alt"].astype(str),
        "stars": df["stars"].astype(int),
    }).drop_duplicates("variant_id").reset_index(drop=True)


def _uniprot(query: str) -> tuple[str, str] | None:
    import ssl

    import certifi

    params = urllib.parse.urlencode({"query": query, "fields": "accession,sequence", "format": "tsv", "size": 1})
    # python.org builds of Python ship without CA certificates.
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(f"{UNIPROT_SEARCH}?{params}", timeout=60, context=ctx) as resp:
        lines = resp.read().decode().strip().splitlines()
    if len(lines) < 2:
        return None
    acc, seq = lines[1].split("\t")[:2]
    return acc, seq


def fetch_sequences(panel_accessions: dict[str, str]) -> dict:
    """Canonical sequences for panel genes (by accession) and their paralogs (by gene name)."""
    out = {}
    for gene, acc in sorted(panel_accessions.items()):
        hit = _uniprot(f"accession:{acc}")
        if hit:
            out[gene] = {"uniprot": hit[0], "sequence": hit[1]}
    for gene in sorted({p for ps in PARALOG_FAMILIES.values() for p in ps}):
        hit = _uniprot(f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true")
        if hit:
            out[gene] = {"uniprot": hit[0], "sequence": hit[1]}
    return out


@lru_cache(maxsize=1)
def load_sequences() -> dict[str, str]:
    path = config.paralog_sequences_json()
    if not path.exists():
        raise FileNotFoundError(f"No sequences at {path}. Run `python -m geneius.data.paralogs`.")
    # Teammates' file maps gene -> sequence; the fetched one maps gene -> {uniprot, sequence}.
    return {g: v["sequence"] if isinstance(v, dict) else v for g, v in json.loads(path.read_text()).items()}


def align_map(seq_a: str, seq_b: str) -> dict[int, int]:
    """1-based residue in seq_a -> aligned 1-based residue in seq_b (global alignment, BLOSUM62)."""
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner(mode="global", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                              open_gap_score=-10.0, extend_gap_score=-0.5)
    aln = aligner.align(seq_a, seq_b)[0]
    mapping = {}
    for (a0, a1), (b0, b1) in zip(*aln.aligned):
        for k in range(int(a1 - a0)):
            mapping[int(a0) + k + 1] = int(b0) + k + 1
    return mapping


def family_maps() -> dict[tuple[str, str], dict[int, int]]:
    """(panel gene, paralog) -> residue map, cached on disk keyed by the sequences."""
    seqs = load_sequences()
    pairs = [(g, p) for g, ps in PARALOG_FAMILIES.items() for p in ps if g in seqs and p in seqs]
    digest = hashlib.sha1("".join(f"{g}{p}{seqs[g]}{seqs[p]}" for g, p in pairs).encode()).hexdigest()[:12]
    path = config.data_dir() / "cache" / f"paralog_maps_{digest}.json"
    if path.exists():
        raw = json.loads(path.read_text())
        return {tuple(k.split("|")): {int(a): b for a, b in v.items()} for k, v in raw.items()}
    maps = {(g, p): align_map(seqs[g], seqs[p]) for g, p in pairs}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({f"{g}|{p}": m for (g, p), m in maps.items()}))
    return maps


def main() -> None:
    from . import load_variants

    v = load_variants()
    accessions = v.dropna(subset=["uniprot_id"]).groupby("gene")["uniprot_id"].first().to_dict()
    path = config.paralog_sequences_json()
    seqs = json.loads(path.read_text()) if path.exists() else {}
    todo = {g: a for g, a in accessions.items() if g not in seqs}
    if todo or not seqs:
        seqs.update(fetch_sequences(todo) if seqs else fetch_sequences(accessions))
    # Panel genes with no UniProt id in the table (CACNA1C, PRPH2): look up by gene name.
    for gene in sorted(set(v["gene"]) - set(seqs)):
        hit = _uniprot(f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true")
        if hit:
            seqs[gene] = {"uniprot": hit[0], "sequence": hit[1]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seqs, indent=1))
    wanted = set(v["gene"]) | {p for ps in PARALOG_FAMILIES.values() for p in ps}
    missing = sorted(wanted - set(seqs))
    print(f"wrote {len(seqs)} sequences to {path}; missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
