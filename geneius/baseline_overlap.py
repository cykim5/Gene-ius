"""Phenotype overlap baseline (Phenomizer-style).

For each patient term, find the most informative HPO term it shares with the disease
(walking up the is_a tree), and average those information content (IC) scores.
Rare, specific matches count a lot; generic matches like "Abnormality of the nervous
system" count almost nothing. Genes get the score of their best-matching disease.

No training needed, so this is the first working version of rank_candidates() and
the first baseline in the eval.
"""
from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache

from .data import HPO, KG, get_hpo, get_kg


class OverlapRanker:
    def __init__(self, kg: KG, hpo: HPO, hide_diseases: set[str] | frozenset = frozenset()):
        self.kg, self.hpo = kg, hpo
        # Each disease's annotations expanded with all their ancestors.
        self.disease_exp: dict[str, frozenset] = {
            d: frozenset().union(*(hpo.ancestors(t) for t in ts))
            for d, ts in kg.disease_hpo.items()
            if ts and d not in hide_diseases
        }
        n = max(len(self.disease_exp), 1)
        counts = Counter(t for s in self.disease_exp.values() for t in s)
        self.ic: dict[str, float] = {t: -math.log(c / n) for t, c in counts.items()}

    def score_diseases(self, hpo_ids: list[str]) -> dict[str, float]:
        terms = [t for t in dict.fromkeys(self.hpo.primary(x) for x in hpo_ids) if t in self.hpo.names]
        if not terms:
            return {}
        # Ancestors sorted most-informative first, so the first hit is the best match.
        anc = [sorted(self.hpo.ancestors(t), key=lambda a: -self.ic.get(a, 0.0)) for t in terms]
        scores = {}
        for d, dexp in self.disease_exp.items():
            s = 0.0
            for alist in anc:
                for a in alist:
                    if a in dexp:
                        s += self.ic.get(a, 0.0)
                        break
            scores[d] = s / len(anc)
        return scores

    def rank_genes(self, hpo_ids: list[str], top_k: int = 10) -> list[dict]:
        best: dict[str, tuple[float, str]] = {}
        for d, s in self.score_diseases(hpo_ids).items():
            if s <= 0:
                continue
            for g in self.kg.disease_genes.get(d, ()):
                if g not in best or s > best[g][0] or (s == best[g][0] and d < best[g][1]):
                    best[g] = (s, d)
        ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))[:top_k]
        return [{"gene": g, "disease": d, "score": round(s, 4)} for g, (s, d) in ranked]


@lru_cache(maxsize=1)
def get_overlap_ranker() -> OverlapRanker:
    return OverlapRanker(get_kg(), get_hpo())
