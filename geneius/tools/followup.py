"""suggest_followup: which symptom question best separates the top candidates?

Picks an HPO term the patient hasn't reported that some (not all) candidate diseases
have. Prefers informative terms that split the field evenly, with a bonus for terms
that separate candidate #1 from candidate #2.

TODO (PhD #2): refine what makes a clinically good question (e.g. easy to check at
the bedside, use phenotype frequency from phenotype.hpoa).
"""
from __future__ import annotations

from .. import config
from ..baseline_overlap import get_overlap_ranker
from ..data import get_hpo, get_kg
from . import stubs

MAX_DISEASES = 5


def suggest_followup(candidates: list[dict], known_hpo_ids: list[str]) -> dict:
    """Returns {question, hpo_id, distinguishes: [disease]}. question is None if nothing separates them."""
    if config.use_stubs():
        return dict(stubs.FOLLOWUP)
    kg, hpo, r = get_kg(), get_hpo(), get_overlap_ranker()
    diseases = list(dict.fromkeys(
        c["disease"] for c in candidates if c.get("disease") in r.disease_exp
    ))[:MAX_DISEASES]
    empty = {"question": None, "hpo_id": None, "distinguishes": []}
    if len(diseases) < 2:
        return empty

    known = {hpo.primary(t) for t in known_hpo_ids}
    known_anc = set().union(*(hpo.ancestors(t) for t in known)) if known else set()
    n = len(diseases)
    best, best_score = None, 0.0
    for t in sorted(set().union(*(kg.disease_hpo[d] for d in diseases))):
        if t in known_anc:
            continue
        has = [d for d in diseases if t in r.disease_exp[d]]
        if not 0 < len(has) < n:
            continue
        ic = r.ic.get(t, 0.0)
        score = ic * min(len(has), n - len(has))
        if (diseases[0] in has) != (diseases[1] in has):
            score += ic
        if score > best_score:
            best, best_score = (t, has), score
    if not best:
        return empty
    t, has = best
    return {
        "question": f"Does the patient have {hpo.label(t)} ({t})?",
        "hpo_id": t,
        "distinguishes": has,
    }
