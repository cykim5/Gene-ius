"""Live-demo data: predictions.json, demo cards, 1000 Genomes carriers, MGI, FoldX, viewers.

Everything is keyed on ClinVar variation_id (never feature_table's variant_id, which repeats when
two DNA changes give the same protein change).

LEAKAGE: predictions.json harmful, clinvar_today, evidence, and computational_evidence, and the
cls_now / diseases_now / disease_groups_now columns, describe ClinVar after the 2023 cutoff. They
are for the live demo only. 2023-split training and evaluation must not read this module except
through the time_machine fields (harmful_2023_model, evidence_2023).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import pandas as pd

from .. import config

AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
       "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
       "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}
_CHANGE = re.compile(r"^\s*([A-Za-z0-9-]+)\s*[:\s]\s*(?:p\.)?([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})\s*$")
TIME_MACHINE_FIELDS = ("harmful_2023_model", "evidence_2023")


def _path(name: str):
    return config.processed_dir() / name


@lru_cache(maxsize=1)
def feature_table() -> pd.DataFrame:
    """feature_table.csv indexed by variation_id (str). pos is the canonical UniProt residue."""
    df = pd.read_csv(_path("feature_table.csv"), low_memory=False)
    df["variation_id"] = df["variation_id"].astype(str)
    return df.set_index("variation_id", drop=False)


@lru_cache(maxsize=1)
def predictions() -> dict[str, dict]:
    return {str(r["variation_id"]): r for r in json.loads(_path("predictions.json").read_text())}


@lru_cache(maxsize=1)
def clinvar_changes() -> dict[str, str]:
    """variation_id -> ClinVar protein change in ClinVar's own numbering, e.g. 'p.R2030Q'."""
    from . import load_variants

    v = load_variants()
    return {str(i): f"p.{r}{p}{a}" for i, r, p, a in zip(v["variant_id"], v["ref_aa"], v["position"], v["alt_aa"])}


def parse_change(text: str) -> tuple[str, str, int, str] | None:
    """'RPE65:H68P', 'RPE65 p.His68Pro' -> ('RPE65', 'H', 68, 'P')."""
    m = _CHANGE.match(text or "")
    if not m:
        return None
    gene, ref, pos, alt = m.groups()
    ref = ref.upper() if len(ref) == 1 else AA3.get(ref.capitalize())
    alt = alt.upper() if len(alt) == 1 else AA3.get(alt.capitalize())
    if not ref or not alt:
        return None
    return gene.upper(), ref, int(pos), alt


def resolve(text: str) -> list[str]:
    """variation_ids whose canonical protein change matches 'GENE:RefPosAlt'. Several ids means
    the same protein change from different DNA changes. Unmappable rows have no canonical residue."""
    parsed = parse_change(text)
    if parsed is None:
        return []
    gene, ref, pos, alt = parsed
    ft = feature_table()
    hit = ft[(ft["gene"] == gene) & (ft["ref"] == ref) & (ft["pos"] == pos) & (ft["alt"] == alt)
             & (ft["numbering"] != "unmappable")]
    return hit["variation_id"].tolist()


@lru_cache(maxsize=1)
def kg1000_carriers() -> pd.DataFrame:
    return pd.read_csv(_path("kg1000_carriers.csv"))


@lru_cache(maxsize=1)
def kg1000_healthy() -> pd.DataFrame:
    return pd.read_csv(_path("kg1000_healthy_pathogenic_carriers.csv"))


@lru_cache(maxsize=1)
def kg1000_two_hit() -> pd.DataFrame:
    return pd.read_csv(_path("kg1000_two_hit_recessive.csv"))


@lru_cache(maxsize=1)
def mouse_mgi() -> dict[str, str]:
    df = pd.read_csv(_path("mouse_phenotypes_mgi.csv"))
    return dict(zip(df["gene"], df["mouse_evidence"]))


@lru_cache(maxsize=1)
def demo_cards() -> list[dict]:
    return json.loads(_path("demo_cards.json").read_text())


def viewer_file(gene: str, clinvar_change: str):
    """viewer_{GENE}_p{Ref}{Pos}{Alt}.html if it exists, else None."""
    path = _path(f"viewer_{gene}_{clinvar_change.replace('.', '')}.html")
    return path if path.exists() else None


def clear_caches() -> None:
    for fn in (feature_table, predictions, clinvar_changes, kg1000_carriers, kg1000_healthy, kg1000_two_hit,
               mouse_mgi, demo_cards):
        fn.cache_clear()
