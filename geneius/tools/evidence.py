"""get_evidence_paths: why is this gene a candidate? Real paths from the knowledge graph.

Path types (most informative first):
  phenotype -> gene                          (phenotype_protein edge)
  phenotype -> disease -> gene               (patient term is annotated on the disease)
  phenotype -> [is_a ...] -> disease -> gene (partial match through the HPO tree)
"""
from __future__ import annotations

from .. import config
from ..baseline_overlap import get_overlap_ranker
from ..data import HPO_ROOT, PHENO_ROOT, get_hpo, get_kg
from . import stubs


def get_evidence_paths(hpo_ids: list[str], gene: str, max_paths: int = 5) -> list[dict]:
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
