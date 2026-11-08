"""patient_rank: rank a patient's candidate variants by harm and by fit to their symptoms.

For each candidate:
  harmful          predictions.json `harmful` ("current model"), or with time_machine=True only
                   `harmful_2023_model` (trained on 2023 ClinVar).
  phenotype_match  phenotype-overlap score between the symptoms and the best-matching disease of
                   the variant's gene (models/phenotype_overlap.py), divided by the best score among
                   the candidates, so the best fit is 1.0.
  combined         sqrt(harmful * phenotype_match).

Variants are 'GENE:RefPosAlt' in canonical UniProt numbering (feature_table.csv pos).
"""
from __future__ import annotations

import math

from ...data import demo, get_hpo, get_kg
from ...models.phenotype_overlap import get_overlap_ranker


def _gene_fit(gene: str, disease_scores: dict[str, float]) -> tuple[float, str | None]:
    best, best_disease = 0.0, None
    for d in sorted(get_kg().gene_diseases.get(gene, ())):
        s = disease_scores.get(d, 0.0)
        if s > best:
            best, best_disease = s, d
    return best, best_disease


def patient_rank(hpo_ids: list[str], variants: list[str], time_machine: bool = False) -> dict:
    """Returns {hpo_terms, harmful_source, ranked: [...], not_found: [...], notes: [...]}."""
    hpo = get_hpo()
    terms = [hpo.primary(t) for t in hpo_ids]
    unknown = [t for t in terms if t not in hpo.names]
    preds, ft = demo.predictions(), demo.feature_table()
    field = "harmful_2023_model" if time_machine else "harmful"

    rows, not_found, notes, genes = [], [], [], set()
    for query in variants:
        ids = demo.resolve(query)
        if not ids:
            not_found.append(query)
            continue
        if len(ids) > 1:
            notes.append(f"{query} matches {len(ids)} ClinVar records (same protein change, different DNA "
                         f"changes): {', '.join(ids)}. All are listed.")
        for vid in ids:
            rec = preds.get(vid, {})
            gene = ft.at[vid, "gene"]
            genes.add(gene)
            rows.append({"query": query, "variant": ft.at[vid, "variant_id"], "variation_id": vid, "gene": gene,
                         "clinvar_change": demo.clinvar_changes().get(vid), "harmful": rec.get(field)})

    diseases = set().union(*(get_kg().gene_diseases.get(g, ()) for g in genes))
    disease_scores = get_overlap_ranker().score_diseases([t for t in terms if t in hpo.names], diseases=diseases)
    for r in rows:
        raw, disease = _gene_fit(r["gene"], disease_scores)
        r["phenotype_raw"] = round(raw, 4)
        r["best_matching_disease"] = disease

    top = max((r["phenotype_raw"] for r in rows), default=0.0)
    for r in rows:
        r["phenotype_match"] = round(r["phenotype_raw"] / top, 4) if top > 0 else 0.0
        h = r["harmful"]
        r["combined"] = round(math.sqrt(h * r["phenotype_match"]), 4) if h is not None else None
    rows.sort(key=lambda r: (r["combined"] is None, -(r["combined"] or 0.0), r["variant"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    if unknown:
        notes.append(f"HPO ids not in the ontology, ignored: {', '.join(unknown)}")
    return {
        "hpo_terms": [{"hpo_id": t, "label": hpo.label(t)} for t in terms if t in hpo.names],
        "harmful_source": "2023 model (time machine)" if time_machine else "current model",
        "ranked": rows,
        "not_found": [{"query": q, "status": "not found"} for q in not_found],
        "notes": notes,
    }
