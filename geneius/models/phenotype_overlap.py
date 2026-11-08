"""Phenotype overlap baseline (Phenomizer-style).

For each patient term, find the most informative HPO term it shares with the disease
(walking up the is_a tree), and average those information content (IC) scores.
Rare, specific matches count a lot; generic matches like "Abnormality of the nervous
system" count almost nothing. Genes get the score of their best-matching disease.

No training needed. Used as the simple baseline for symptom-based questions
in the agent benchmark.
"""
from __future__ import annotations

import math
import os
import pickle
from collections import Counter
from functools import lru_cache

from .. import config
from ..data import HPO, KG, get_hpo, get_kg


class OverlapRanker:
    def __init__(self, kg: KG, hpo: HPO, hide_diseases: set[str] | frozenset = frozenset(),
                 _precomputed: tuple[dict, dict] | None = None):
        self.kg, self.hpo = kg, hpo
        if _precomputed is not None:
            self.disease_exp, self.ic = _precomputed
            return
        # Each disease's annotations expanded with all their ancestors.
        self.disease_exp: dict[str, frozenset] = {
            d: frozenset().union(*(hpo.ancestors(t) for t in ts))
            for d, ts in kg.disease_hpo.items()
            if ts and d not in hide_diseases
        }
        n = max(len(self.disease_exp), 1)
        counts = Counter(t for s in self.disease_exp.values() for t in s)
        self.ic: dict[str, float] = {t: -math.log(c / n) for t, c in counts.items()}

    def score_diseases(self, hpo_ids: list[str], diseases: set[str] | None = None) -> dict[str, float]:
        terms = [t for t in dict.fromkeys(self.hpo.primary(x) for x in hpo_ids) if t in self.hpo.names]
        if not terms:
            return {}
        # Ancestors sorted most-informative first, so the first hit is the best match.
        anc = [sorted(self.hpo.ancestors(t), key=lambda a: -self.ic.get(a, 0.0)) for t in terms]
        items = ((d, self.disease_exp[d]) for d in diseases if d in self.disease_exp) if diseases is not None \
            else self.disease_exp.items()
        scores = {}
        for d, dexp in items:
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


def _cache_path():
    kg_m = config.primekg_csv().stat().st_mtime_ns
    hpo_m = config.hpo_obo().stat().st_mtime_ns
    return config.data_dir() / "cache" / f"overlap_ranker_{kg_m}_{hpo_m}.pkl"


@lru_cache(maxsize=1)
def get_overlap_ranker() -> OverlapRanker:
    kg, hpo = get_kg(), get_hpo()
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return OverlapRanker(kg, hpo)
    path = _cache_path()
    if path.exists():
        with path.open("rb") as f:
            disease_exp, ic = pickle.load(f)
        return OverlapRanker(kg, hpo, _precomputed=(disease_exp, ic))
    ranker = OverlapRanker(kg, hpo)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump((ranker.disease_exp, ranker.ic), f, protocol=pickle.HIGHEST_PROTOCOL)
    return ranker
