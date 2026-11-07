"""Confidence and evidence calibration.

1. Reliability plot on the validation split. Temperature scaling if needed.
2. Pick the threshold below which the agent says it's unsure.
3. Map scores to PP3/BP4 strength (supporting, moderate, strong) with the
   ClinGen method (Pejaver et al., 2022) applied to our own scores.
"""
from __future__ import annotations


def temperature_scale(logits, labels) -> float:
    raise NotImplementedError


def evidence_strength(score: float) -> str:
    """Returns 'PP3_strong', 'PP3_moderate', 'PP3_supporting', 'none', 'BP4_supporting', ..."""
    raise NotImplementedError
