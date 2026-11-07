"""Time-saved curve (Stage 8, test 3).

Rank resolved 2023 VUS by predicted harmfulness and count real pathogenic variants
in the top k. Compare against random order, AlphaMissense, and our model without
AlphaMissense.
"""
from __future__ import annotations

KS = (10, 25, 50, 100)


def found_in_top_k(is_pathogenic: list[int], scores: list[float], ks=KS) -> dict[int, int]:
    """How many truly pathogenic variants land in the top k when ranked by score."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranked = [is_pathogenic[i] for i in order]
    return {k: sum(ranked[:k]) for k in ks}


def random_expected(is_pathogenic: list[int], ks=KS) -> dict[int, float]:
    """Expected count for random order: pathogenic rate times k."""
    rate = sum(is_pathogenic) / len(is_pathogenic) if is_pathogenic else 0.0
    return {k: rate * min(k, len(is_pathogenic)) for k in ks}


# TODO: plot all four curves (x = variants tested, y = disease variants found)
# with bootstrap confidence intervals.
