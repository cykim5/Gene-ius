"""Variant-level demo tools: population_evidence, stability, mouse_evidence, and the abstain check.

population_evidence: 1000 Genomes phased panel carriers (kg1000_carriers.csv), matched on
genomic_key, children in trios excluded so related people aren't counted twice. The panel leaves
out singletons, so "not seen" never means "absent".
stability: FoldX mechanism from predictions.json (RHO and RPE65; GLA once foldx_GLA_results.csv exists).
mouse_evidence: MGI mouse phenotypes. Gene-level, not about the specific variant.
assess: when the agent should say it's unsure and suggest an assay.
"""
from __future__ import annotations

import re

from ... import config
from ...data import demo

SUPERPOPS = ["AFR", "EUR", "SAS", "EAS", "AMR"]
NOT_SEEN = "not seen in the 1000 Genomes phased panel (which excludes singletons)"


def _foldx_genes() -> set[str]:
    genes = {"RHO", "RPE65"}
    if (config.processed_dir() / "foldx_GLA_results.csv").exists():
        genes.add("GLA")
    return genes


def _population(vid: str) -> dict:
    ft = demo.feature_table()
    key, gene = ft.at[vid, "genomic_key"], ft.at[vid, "gene"]
    change = demo.clinvar_changes().get(vid)
    kg = demo.kg1000_carriers()
    in_trio_child = kg["is_child_in_trio"].astype(str).str.lower() == "true"
    hits = kg[(kg["key"] == key) & ~in_trio_child]
    samples = hits.drop_duplicates("sample")
    out = {"variant": ft.at[vid, "variant_id"], "variation_id": vid, "genomic_key": key, "clinvar_change": change}
    if samples.empty:
        out["summary"] = NOT_SEEN
        out["carriers"] = 0
    else:
        out["carriers"] = int(len(samples))
        out["homozygotes"] = int((samples["copies"] == 2).sum())
        counts = samples["superpop"].value_counts()
        out["by_superpop"] = {p: int(counts.get(p, 0)) for p in SUPERPOPS}
        out["summary"] = f"{out['carriers']} carriers ({out['homozygotes']} homozygous) in the 1000 Genomes phased panel"
    healthy = demo.kg1000_healthy()
    h = healthy[healthy["key"] == key]
    out["seen_in_apparently_healthy_people"] = not h.empty
    if not h.empty:
        out["healthy_carriers"] = {"carriers": int(h["carriers"].iloc[0]), "clinvar_stars": int(h["stars"].iloc[0])}
    two = demo.kg1000_two_hit()
    if change:
        hit = two[(two["gene"] == gene) & two["variants"].astype(str).apply(
            lambda s: change in re.split(r"[;,|\s]+", s))]
        if not hit.empty:
            out["two_hit_healthy_people"] = hit[["sample", "type", "variants"]].to_dict("records")
    return out


def population_evidence(variant: str) -> dict:
    """1000 Genomes carriers for a variant ('GENE:RefPosAlt'). Returns {query, records: [{carriers,
    homozygotes, by_superpop, seen_in_apparently_healthy_people, two_hit_healthy_people, summary}]}."""
    ids = demo.resolve(variant)
    if not ids:
        return {"query": variant, "status": "not found", "records": []}
    return {"query": variant, "records": [_population(v) for v in ids]}


def stability(variant: str) -> dict:
    """FoldX stability (ddG) and mechanism label. Returns {query, records: [{ddg_kcal_mol, label, dominant_term}]}."""
    ids = demo.resolve(variant)
    if not ids:
        return {"query": variant, "status": "not found", "records": []}
    records = []
    for vid in ids:
        rec = demo.predictions().get(vid, {})
        mech = rec.get("mechanism") or {}
        gene = demo.feature_table().at[vid, "gene"]
        base = {"variant": demo.feature_table().at[vid, "variant_id"], "variation_id": vid}
        if gene in _foldx_genes() and isinstance(mech, dict) and "ddg_kcal_mol" in mech:
            records.append({**base, "ddg_kcal_mol": mech["ddg_kcal_mol"], "label": mech.get("label"),
                            "dominant_term": mech.get("dominant_term") or None, "capped": mech.get("capped"),
                            "method": rec.get("mechanism_status")})
        else:
            records.append({**base, "status": "not computed",
                            "note": f"FoldX covers {', '.join(sorted(_foldx_genes()))} only"})
    return {"query": variant, "records": records}


def mouse_evidence(gene: str) -> dict:
    """MGI mouse phenotypes for a gene. Gene-level: not about any specific variant."""
    g = gene.strip().upper()
    text = demo.mouse_mgi().get(g)
    if text is None:
        text = next((r.get("mouse_evidence") for r in demo.predictions().values()
                     if r.get("gene") == g and r.get("mouse_evidence")), None)
    if not text:
        return {"gene": g, "source": "MGI", "level": "gene-level", "status": "no MGI mouse phenotypes on file"}
    return {"gene": g, "source": "MGI", "level": "gene-level (not specific to this variant)", "evidence": text}


def assess(record: dict, population: dict) -> dict:
    """{unsure, reasons}. Unsure when computational evidence is indeterminate, the model is unsure, or
    evidence conflicts (PP3 but healthy carriers, or a healthy person carrying two hits)."""
    reasons = []
    comp = str(record.get("computational_evidence") or "")
    if comp == "indeterminate":
        reasons.append("computational evidence is indeterminate (between the calibrated PP3 and BP4 cutoffs)")
    if record.get("confidence") == "unsure":
        reasons.append("the model's confidence is 'unsure'")
    if comp.startswith("PP3") and population.get("seen_in_apparently_healthy_people"):
        reasons.append(f"conflict: {comp} says damaging, but the variant is seen in apparently healthy people")
    if population.get("two_hit_healthy_people"):
        reasons.append("conflict: a healthy person in 1000 Genomes carries it as a recessive two-hit "
                       f"({population['two_hit_healthy_people'][0]['type']})")
    return {"unsure": bool(reasons), "reasons": reasons}
