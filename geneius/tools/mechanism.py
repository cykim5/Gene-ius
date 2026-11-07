"""get_mechanism: disease -> mechanism summary.

STUB for now. TODO: back this with DisMech. Keep the return shape:
{disease, mechanism_summary, key_phenotypes: [hpo_id], source}
"""
from __future__ import annotations

from . import stubs


def get_mechanism(disease: str) -> dict:
    return stubs.mechanism(disease)
