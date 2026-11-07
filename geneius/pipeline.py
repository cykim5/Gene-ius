"""Fixed pipeline: runs every tool in order, no LLM.

Works offline and gives the same answer every time. Used for the demo fallback
and as the "no agent" baseline.
"""
from __future__ import annotations

from .tools import get_evidence_paths, map_to_hpo, rank_candidates, suggest_followup


def run_pipeline(text: str, top_k: int = 10, explain_top: int = 3) -> dict:
    terms = map_to_hpo(text)
    hpo_ids = [t["hpo_id"] for t in terms]
    candidates = rank_candidates(hpo_ids, top_k) if hpo_ids else []
    for c in candidates[:explain_top]:
        c["evidence"] = get_evidence_paths(hpo_ids, c["gene"])
    followup = suggest_followup(candidates, hpo_ids) if candidates else None
    return {"hpo_terms": terms, "candidates": candidates, "followup": followup}
