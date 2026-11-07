"""Paths and settings. Everything is read from environment variables (.env is loaded automatically)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent

# The 2023 freeze (Stage 0). Everything the model learns from must be available
# at this date. Everything we test on was added after it. Change it only here.
CUTOFF = "2023-01"


def data_dir() -> Path:
    raw = os.environ.get("GENEIUS_DATA_DIR") or str(REPO_ROOT / "data")
    return Path(raw).expanduser()


def primekg_csv() -> Path:
    return data_dir() / "kg.csv"


def hpo_obo() -> Path:
    """HPO from the cutoff date. Use hpo_obo_current() for the time-based test."""
    return data_dir() / f"hp_{CUTOFF}.obo"


def hpo_obo_current() -> Path:
    return data_dir() / "hp_current.obo"


def clinvar_cutoff() -> Path:
    return data_dir() / f"variant_summary_{CUTOFF}.txt.gz"


def clinvar_current() -> Path:
    return data_dir() / "variant_summary_current.txt.gz"


def variants_csv() -> Path:
    """The ClinVar 2023 vs current variant table. Committed in the repo, so it's not under data_dir()."""
    raw = os.environ.get("GENEIUS_VARIANTS_CSV") or str(REPO_ROOT / "data" / "processed" / "variants.csv")
    return Path(raw).expanduser()


def feature_table_csv() -> Path:
    """Teammates' variant feature table (AlphaMissense, pLDDT, UniProt regions). Local only, not tracked."""
    raw = os.environ.get("GENEIUS_FEATURE_TABLE") or str(REPO_ROOT / "data" / "processed" / "feature_table.csv")
    return Path(raw).expanduser()


def baseline_predictions_csv() -> Path:
    """Teammates' predictions.csv. Its harmful_2023_model column is a 2023-trained baseline."""
    raw = os.environ.get("GENEIUS_BASELINE_PREDICTIONS") or str(REPO_ROOT / "data" / "processed" / "predictions.csv")
    return Path(raw).expanduser()


def _processed(name: str, env: str) -> Path:
    return Path(os.environ.get(env) or str(REPO_ROOT / "data" / "processed" / name)).expanduser()


def paralog_variants_csv() -> Path:
    return _processed("paralog_pathogenic_2023.csv", "GENEIUS_PARALOG_VARIANTS")


def paralog_sequences_json() -> Path:
    return _processed("paralog_sequences.json", "GENEIUS_PARALOG_SEQUENCES")


def processed_dir() -> Path:
    """Teammates' processed files (predictions.json, demo cards, 1000 Genomes, viewers). Local only."""
    return Path(os.environ.get("GENEIUS_PROCESSED_DIR") or str(REPO_ROOT / "data" / "processed")).expanduser()


def ca_coords_pkl() -> Path:
    """AlphaFold C-alpha coordinates and pLDDT per residue, keyed by gene."""
    return _processed("alphafold_ca_coords.pkl", "GENEIUS_CA_COORDS")


def gnn_a_dir() -> Path:
    """Output A checkpoint and predictions.csv. Reading predictions needs no torch import."""
    raw = os.environ.get("GENEIUS_GNN_A_DIR") or str(data_dir() / "gnn_a")
    return Path(raw).expanduser()


def alphamissense() -> Path:
    return data_dir() / "AlphaMissense_aa_substitutions.tsv.gz"


_warned = False


def use_stubs() -> bool:
    """True when tools should return fake data (forced, or data not downloaded yet)."""
    global _warned
    forced = os.environ.get("GENEIUS_USE_STUBS") == "1"
    missing = not (primekg_csv().exists() and hpo_obo().exists())
    if (forced or missing) and not _warned:
        why = "GENEIUS_USE_STUBS=1" if forced else f"data not found in {data_dir()}"
        print(f"[geneius] Using STUB tool outputs ({why}).", file=sys.stderr)
        _warned = True
    return forced or missing
