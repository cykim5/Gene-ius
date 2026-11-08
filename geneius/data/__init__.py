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
from .variants import TEST_ONLY_COLUMNS, load_variants

__all__ = ["HPO", "HPO_ROOT", "KG", "PHENO_ROOT", "TEST_ONLY_COLUMNS", "get_hpo", "get_kg", "load_variants",
           "norm_hpo", "parse_obo", "read_primekg"]
