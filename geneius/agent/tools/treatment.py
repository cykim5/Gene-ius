"""suggest_treatment and suggest_assay. Rule tables from the plan.

Treatments are hypotheses, not recommendations.
"""
from __future__ import annotations

TREATMENT_BY_MECHANISM = {
    "loss_of_function": {
        "strategy": "gene_augmentation",
        "why": "Adding a working copy can restore function.",
        "example": "Luxturna for RPE65",
    },
    "dominant_negative": {
        "strategy": "silencing_knockdown_replace",
        "why": "The faulty protein disrupts the healthy copy, so adding a copy isn't enough.",
        "example": "Some RHO and PRPH2 variants",
    },
    "gain_of_function": {
        "strategy": "silencing_knockdown_replace",
        "why": "The protein does too much, so it needs to be silenced or blocked.",
        "example": "Allele-specific silencing",
    },
    "misfolding": {
        "strategy": "chaperone",
        "why": "Approaches that help folding can rescue some variants.",
        "example": "Migalastat for amenable GLA variants",
    },
}

CHANNEL_GENES = {"SCN5A", "KCNQ1", "KCNH2", "KCNJ2", "CACNA1C"}
RECEPTOR_GENES = {"RHO"}
ENZYME_GENES = {"GLA", "RPE65"}


def suggest_treatment(mechanism: str, gene: str = "", disease: str = "") -> dict:
    """Returns {mechanism, strategy, why, example, note}."""
    if gene.upper() in CHANNEL_GENES and mechanism == "gain_of_function":
        rule = {
            "strategy": "channel_blocker",
            "why": "Drugs that block the overactive current.",
            "example": "Mexiletine for long QT 3 (SCN5A)",
        }
    else:
        rule = TREATMENT_BY_MECHANISM.get(mechanism)
    if not rule:
        return {"mechanism": mechanism, "strategy": None, "why": "No rule for this mechanism.", "example": None,
                "note": "Hypothesis only."}
    return {"mechanism": mechanism, **rule, "note": "Hypothesis only."}


def suggest_assay(gene: str) -> dict:
    """The experiment that would settle an uncertain case. Returns {gene, protein_type, assay, evidence_code}."""
    g = gene.upper()
    if g in CHANNEL_GENES:
        ptype, assay = "ion channel", "Patch clamp electrophysiology (measures the current, shows gain vs loss)"
    elif g in RECEPTOR_GENES:
        ptype, assay = "membrane receptor", "Cell assays for protein trafficking and activation"
    elif g in ENZYME_GENES:
        ptype = "enzyme"
        assay = "Enzyme activity assay" + ("; chaperone-response assay for migalastat" if g == "GLA" else "")
    else:
        ptype, assay = "other", "Protein expression and stability assays"
    return {"gene": g, "protein_type": ptype, "assay": assay, "evidence_code": "PS3/BS3 if the result is clear"}
