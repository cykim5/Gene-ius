"""evidence_by_source and evidence_report. Stubs for now.

The report should follow ACMG/AMP: predictions A to D with confidence, PP3/BP4 with
calibrated strength, evidence paths and 3D neighbors, the suggested functional test
(would count as PS3/BS3), the treatment hypothesis, and this disclaimer:
"Research use only. Supports, but does not replace, expert variant classification."
"""
from __future__ import annotations

from . import stubs

DISCLAIMER = "Research use only. Supports, but does not replace, expert variant classification."


def evidence_by_source(variant: str) -> dict:
    """How much each data source contributed. Returns {variant, contributions: {source: share}}."""
    return stubs.evidence_by_source(variant)


def evidence_report(variant: str) -> dict:
    """One-page clinician-style report. Returns {variant, report_markdown}."""
    return stubs.evidence_report(variant)
