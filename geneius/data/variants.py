"""The variant table: ClinVar missense variants with their 2023 and current classification.

One row per variant. Columns ending in _now describe ClinVar after the 2023 cutoff. They are
test labels only and must never become features or training labels, so load_variants() drops
them unless the caller asks for them (eval code only).
"""
from __future__ import annotations

import pandas as pd

from .. import config

TEST_ONLY_COLUMNS = [
    "class_now", "stars_now", "disease_now", "pheno_ids_now",
    "diseases_now_specific", "group_now", "syndromic_now",
]


def load_variants(with_test_labels: bool = False) -> pd.DataFrame:
    path = config.variants_csv()
    if not path.exists():
        raise FileNotFoundError(f"No variant table at {path}. It's data/processed/variants.csv in the repo.")
    df = pd.read_csv(path, low_memory=False)
    if not with_test_labels:
        df = df.drop(columns=[c for c in TEST_ONLY_COLUMNS if c in df.columns])
    return df
