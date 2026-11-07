"""Graph tools: find_entity, expand, find_paths.

find_entity: works for genes, diseases, and HPO terms. TODO: variants.
expand: works on PrimeKG gene, disease, phenotype, pathway, and drug nodes. TODO: variant
nodes once graph/build.py adds them.
find_paths: works for symptom -> gene paths. TODO: variant patterns like
variant -> protein -> symptom -> disease once variant nodes exist.

find_paths path types (most informative first):
  phenotype -> gene                          (phenotype_protein edge)
  phenotype -> disease -> gene               (patient term is annotated on the disease)
  phenotype -> [is_a ...] -> disease -> gene (partial match through the HPO tree)
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from ... import config
from ...models.phenotype_overlap import get_overlap_ranker
from ...data import HPO_ROOT, PHENO_ROOT, get_hpo, get_kg, norm_hpo, read_primekg
from . import stubs

# PrimeKG node type -> tool node type
NODE_TYPES = {"gene/protein": "gene", "disease": "disease", "effect/phenotype": "phenotype",
              "pathway": "pathway", "drug": "drug"}
TYPE_ORDER = {"disease": 0, "gene": 1, "phenotype": 2, "pathway": 3, "drug": 4}
# drug_drug (2.7M rows) and the anatomy / GO / exposure layers are left out to keep memory low.
EXPAND_RELATIONS = [
    "disease_phenotype_positive", "disease_phenotype_negative", "disease_protein", "disease_disease",
    "phenotype_protein", "phenotype_phenotype", "protein_protein", "pathway_protein", "pathway_pathway",
    "drug_protein", "drug_effect", "indication", "contraindication", "off-label use",
]
# Phenotypes above this percentile of phenotype degree (e.g. inheritance modes) count as generic hubs.
HUB_PERCENTILE = 99


def find_paths(hpo_ids: list[str], gene: str, max_paths: int = 5) -> list[dict]:
    """Returns [{path: [{node_id, node_type, name}], relations: [str]}]."""
    if config.use_stubs():
        return stubs.evidence_paths(gene)[:max_paths]
    kg, hpo, ic = get_kg(), get_hpo(), get_overlap_ranker().ic

    def pnode(t):
        return {"node_id": t, "node_type": "phenotype", "name": hpo.label(t)}

    gnode = {"node_id": gene, "node_type": "gene", "name": gene}
    scored: list[tuple[float, dict]] = []

    for t in dict.fromkeys(hpo.primary(x) for x in hpo_ids):
        if gene in kg.hpo_genes.get(t, ()):
            scored.append((ic.get(t, 0.0) + 1.0, {"path": [pnode(t), gnode], "relations": ["phenotype_protein"]}))

        for d in sorted(kg.gene_diseases.get(gene, ())):
            dnode = {"node_id": d, "node_type": "disease", "name": d}
            direct = kg.disease_hpo.get(d, set())
            if t in direct:
                scored.append((ic.get(t, 0.0), {
                    "path": [pnode(t), dnode, gnode],
                    "relations": ["disease_phenotype_positive", "disease_protein"],
                }))
                continue
            # Partial match: most informative ancestor of t shared with one of the disease's terms.
            # Ties go to the more specific term.
            best, best_key = None, None
            for a in sorted(hpo.ancestors(t)):
                if a in (HPO_ROOT, PHENO_ROOT):
                    continue
                key = (ic.get(a, 0.0), len(hpo.ancestors(a)))
                for u in sorted(direct):
                    if a in hpo.ancestors(u) and (best_key is None or key > best_key):
                        best, best_key = (a, u), key
            if best:
                a, u = best
                nodes, rels = [pnode(t)], []
                if a != t:
                    nodes.append(pnode(a))
                    rels.append("is_a")
                if a != u:
                    nodes.append(pnode(u))
                    rels.append("has_subtype")
                nodes += [dnode, gnode]
                rels += ["disease_phenotype_positive", "disease_protein"]
                scored.append((0.8 * ic.get(a, 0.0), {"path": nodes, "relations": rels}))

    scored.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for _, p in scored:
        key = tuple(n["node_id"] for n in p["path"])
        if key not in seen:
            seen.add(key)
            out.append(p)
        if len(out) >= max_paths:
            break
    return out


def find_entity(query: str) -> dict:
    """Look up a gene, disease, or HPO term by name or ID. Returns {query, matches: [{node_id, node_type, name}]}."""
    if config.use_stubs():
        return stubs.find_entity(query)
    kg, hpo = get_kg(), get_hpo()
    q = query.strip()
    ql = q.lower()
    matches = []
    if q.upper() in kg.gene_diseases:
        matches.append({"node_id": q.upper(), "node_type": "gene", "name": q.upper()})
    for d in kg.disease_hpo.keys() | kg.disease_genes.keys():
        if d.lower() == ql:
            matches.append({"node_id": d, "node_type": "disease", "name": d})
    for t, name in hpo.names.items():
        if t.lower() == ql or name.lower() == ql:
            matches.append({"node_id": t, "node_type": "phenotype", "name": name})
    return {"query": query, "matches": matches}


@dataclass
class _Adjacency:
    edges: pd.DataFrame  # src_type, src, dst_type, dst, relation; both directions
    rows: dict  # (type, key) -> row positions in edges
    degree: dict  # (type, key) -> number of distinct neighbors
    hub_degree: float


def _node_keys(ids: pd.Series, names: pd.Series, types: pd.Series) -> pd.Series:
    """Genes, diseases, pathways, and drugs are keyed by name, phenotypes by HPO id."""
    hpo = get_hpo()
    is_pheno = types == "phenotype"
    lookup = {i: hpo.primary(norm_hpo(i)) for i in pd.unique(ids[is_pheno])}
    return names.where(~is_pheno, ids.map(lookup))


@lru_cache(maxsize=1)
def _adjacency() -> _Adjacency:
    df = read_primekg(EXPAND_RELATIONS)
    df = df[df.x_type.isin(NODE_TYPES.keys()) & df.y_type.isin(NODE_TYPES.keys())]
    xt, yt = df.x_type.map(NODE_TYPES), df.y_type.map(NODE_TYPES)
    fwd = pd.DataFrame({
        "src_type": xt.values, "src": _node_keys(df.x_id, df.x_name, xt).values,
        "dst_type": yt.values, "dst": _node_keys(df.y_id, df.y_name, yt).values,
        "relation": df.relation.values,
    })
    rev = fwd.rename(columns={"src_type": "dst_type", "src": "dst", "dst_type": "src_type", "dst": "src"})
    edges = pd.concat([fwd, rev], ignore_index=True).drop_duplicates()
    edges = edges[~((edges.src_type == edges.dst_type) & (edges.src == edges.dst))].reset_index(drop=True)

    pairs = edges.drop_duplicates(["src_type", "src", "dst_type", "dst"])
    degree = pairs.groupby(["src_type", "src"]).size().to_dict()
    pheno_degrees = [d for (t, _), d in degree.items() if t == "phenotype"]
    hub = float(np.percentile(pheno_degrees, HUB_PERCENTILE)) if pheno_degrees else float("inf")
    rows = edges.groupby(["src_type", "src"]).indices
    return _Adjacency(edges, rows, degree, hub)


def _resolve(adj: _Adjacency, node_id: str) -> list[tuple[str, str]]:
    q = node_id.strip()
    candidates = [q, q.upper()]
    if q.upper().startswith("HP:"):
        candidates.append(get_hpo().primary(norm_hpo(q)))
    found = [(t, k) for t in TYPE_ORDER for k in dict.fromkeys(candidates) if (t, k) in adj.rows]
    if found:
        return found
    ql = q.lower()
    return [key for key in adj.rows if key[0] in ("disease", "pathway", "drug") and key[1].lower() == ql]


def _name(node_type: str, key: str) -> str:
    return get_hpo().label(key) if node_type == "phenotype" else key


def expand(node_id: str, node_type: str | None = None, relation: str | None = None, limit: int = 25) -> dict:
    """Neighbors of a node, optionally filtered by neighbor type and PrimeKG relation.

    Returns {node_id, node_type, name, found, total, neighbors: [{node_id, node_type, name, relation, degree}]}.
    At most `limit` neighbors. Generic hub phenotypes come last. Other phenotypes are ordered most
    specific first (lowest degree), other node types best-connected first.
    """
    if config.use_stubs():
        return stubs.expand(node_id, node_type)
    adj = _adjacency()
    matches = _resolve(adj, node_id)
    if not matches:
        return {"node_id": node_id, "found": False, "total": 0, "neighbors": []}
    src_type, src = matches[0]
    rows = adj.edges.iloc[np.concatenate([adj.rows[m] for m in matches])]
    if node_type:
        rows = rows[rows.dst_type == node_type]
    if relation:
        rows = rows[rows.relation == relation]

    neighbors = []
    for t, k, rel in rows[["dst_type", "dst", "relation"]].drop_duplicates().itertuples(index=False):
        deg = adj.degree.get((t, k), 0)
        hub = t == "phenotype" and deg >= adj.hub_degree
        order = deg if t == "phenotype" else -deg
        name = _name(t, k)
        neighbors.append(((hub, TYPE_ORDER.get(t, 9), order, name, rel),
                          {"node_id": k, "node_type": t, "name": name, "relation": rel, "degree": deg}))
    neighbors.sort(key=lambda x: x[0])
    return {
        "node_id": src,
        "node_type": src_type,
        "name": _name(src_type, src),
        "found": True,
        "total": len(neighbors),
        "neighbors": [n for _, n in neighbors[:max(0, int(limit))]],
    }
