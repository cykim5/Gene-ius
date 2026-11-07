"""Loaders for the HPO ontology and PrimeKG.

Both are cached: the first call is slow (PrimeKG is large), later calls are instant.
All HPO IDs are normalized to "HP:0001250". Genes are PrimeKG gene names (HGNC symbols).
Diseases are keyed by their PrimeKG name.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from . import config

HPO_ROOT = "HP:0000001"  # All
PHENO_ROOT = "HP:0000118"  # Phenotypic abnormality


def norm_hpo(raw) -> str:
    """'1250', '1250.0', 'HP:1250', 'HP:0001250' -> 'HP:0001250'."""
    s = str(raw).strip()
    if s.upper().startswith("HP:"):
        return "HP:" + s[3:].zfill(7)
    return f"HP:{int(float(s)):07d}"



@dataclass
class HPO:
    names: dict[str, str]
    synonyms: dict[str, list[str]]
    parents: dict[str, set[str]]
    alt: dict[str, str]  # obsolete/alternate id -> primary id
    _anc: dict[str, frozenset] = field(default_factory=dict, repr=False)

    def primary(self, t: str) -> str:
        return self.alt.get(t, t)

    def label(self, t: str) -> str:
        return self.names.get(t, t)

    def ancestors(self, t: str) -> frozenset:
        """All ancestors of t, including t itself."""
        if t in self._anc:
            return self._anc[t]
        out, stack = {t}, [t]
        while stack:
            for p in self.parents.get(stack.pop(), ()):
                if p not in out:
                    out.add(p)
                    stack.append(p)
        self._anc[t] = frozenset(out)
        return self._anc[t]

    def is_phenotypic(self, t: str) -> bool:
        return t != PHENO_ROOT and PHENO_ROOT in self.ancestors(t)


def parse_obo(path) -> HPO:
    names: dict[str, str] = {}
    syns: dict[str, list[str]] = defaultdict(list)
    parents: dict[str, set[str]] = defaultdict(set)
    alt: dict[str, str] = {}

    def commit(term):
        if not term or "id" not in term:
            return
        if term.get("is_obsolete", ["false"])[0] == "true":
            return
        tid = term["id"][0]
        names[tid] = term.get("name", [tid])[0]
        for s in term.get("synonym", []):
            # format: "Seizures" EXACT [] or "Fits" RELATED layperson []
            text = s.split('"')[1] if s.count('"') >= 2 else ""
            scope = s.rsplit('"', 1)[-1]
            if text and ("EXACT" in scope or "layperson" in scope):
                syns[tid].append(text)
        for p in term.get("is_a", []):
            parents[tid].add(p.split()[0])
        for a in term.get("alt_id", []):
            alt[a] = tid

    term = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("["):
                commit(term)
                term = {} if line == "[Term]" else None
                continue
            if term is None or ": " not in line:
                continue
            key, _, val = line.partition(": ")
            term.setdefault(key, []).append(val)
    commit(term)
    return HPO(names, dict(syns), dict(parents), alt)


@lru_cache(maxsize=1)
def get_hpo() -> HPO:
    return parse_obo(config.hpo_obo())



@dataclass
class KG:
    disease_hpo: dict[str, set[str]]  # disease -> direct phenotype annotations
    disease_genes: dict[str, set[str]]  # disease -> genes
    gene_diseases: dict[str, set[str]]  # gene -> diseases
    hpo_genes: dict[str, set[str]]  # phenotype -> genes (phenotype_protein edges)


PRIMEKG_COLS = ["relation", "x_id", "x_type", "x_name", "y_id", "y_type", "y_name"]


def _edges(df: pd.DataFrame, relation: str, a_type: str, b_type: str) -> pd.DataFrame:
    """Edges of one relation as (a_id, a_name, b_id, b_name), whichever way PrimeKG stored them."""
    sub = df[df.relation == relation]
    fwd = sub[(sub.x_type == a_type) & (sub.y_type == b_type)]
    rev = sub[(sub.x_type == b_type) & (sub.y_type == a_type)]
    out = pd.concat(
        [
            pd.DataFrame({"a_id": fwd.x_id.values, "a_name": fwd.x_name.values,
                          "b_id": fwd.y_id.values, "b_name": fwd.y_name.values}),
            pd.DataFrame({"a_id": rev.y_id.values, "a_name": rev.y_name.values,
                          "b_id": rev.x_id.values, "b_name": rev.x_name.values}),
        ],
        ignore_index=True,
    )
    return out.drop_duplicates()


def read_primekg(relations: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(config.primekg_csv(), usecols=PRIMEKG_COLS, dtype=str, low_memory=False)
    if relations:
        df = df[df.relation.isin(relations)]
    return df


@lru_cache(maxsize=1)
def get_kg() -> KG:
    hpo = get_hpo()
    df = read_primekg(["disease_phenotype_positive", "disease_protein", "phenotype_protein"])

    disease_hpo: dict[str, set[str]] = defaultdict(set)
    for d, p in zip(*_cols(_edges(df, "disease_phenotype_positive", "disease", "effect/phenotype"), "a_name", "b_id")):
        disease_hpo[d].add(hpo.primary(norm_hpo(p)))

    disease_genes: dict[str, set[str]] = defaultdict(set)
    gene_diseases: dict[str, set[str]] = defaultdict(set)
    for d, g in zip(*_cols(_edges(df, "disease_protein", "disease", "gene/protein"), "a_name", "b_name")):
        disease_genes[d].add(g)
        gene_diseases[g].add(d)

    hpo_genes: dict[str, set[str]] = defaultdict(set)
    for p, g in zip(*_cols(_edges(df, "phenotype_protein", "effect/phenotype", "gene/protein"), "a_id", "b_name")):
        hpo_genes[hpo.primary(norm_hpo(p))].add(g)

    return KG(dict(disease_hpo), dict(disease_genes), dict(gene_diseases), dict(hpo_genes))


def _cols(df: pd.DataFrame, *names: str):
    return [df[n].tolist() for n in names]
