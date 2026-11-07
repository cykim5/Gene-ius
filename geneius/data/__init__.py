"""Stages 1 and 2: load the data and line up IDs across databases."""
from .kg import (
    HPO,
    HPO_ROOT,
    KG,
    PHENO_ROOT,
    get_hpo,
    get_kg,
    norm_hpo,
    parse_obo,
    read_primekg,
)

__all__ = ["HPO", "HPO_ROOT", "KG", "PHENO_ROOT", "get_hpo", "get_kg", "norm_hpo", "parse_obo", "read_primekg"]
