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


def data_dir() -> Path:
    raw = os.environ.get("GENEIUS_DATA_DIR") or str(REPO_ROOT / "data")
    return Path(raw).expanduser()


def primekg_csv() -> Path:
    return data_dir() / "kg.csv"


def hpo_obo() -> Path:
    return data_dir() / "hp.obo"


def hpoa() -> Path:
    return data_dir() / "phenotype.hpoa"


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
