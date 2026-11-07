"""Fake tool outputs with the exact interface shapes.

Used automatically when data isn't downloaded (or GENEIUS_USE_STUBS=1), so the agent,
demo app, and eval scripts can be built in parallel before the real tools are ready.
"""

MAP_TO_HPO = [
    {"hpo_id": "HP:0001250", "label": "Seizure", "matched_text": "seizures"},
    {"hpo_id": "HP:0001252", "label": "Hypotonia", "matched_text": "low muscle tone"},
    {"hpo_id": "HP:0000750", "label": "Delayed speech and language development", "matched_text": "speech delay"},
]

RANK_CANDIDATES = [
    {"gene": "SCN1A", "disease": "Dravet syndrome", "score": 0.91},
    {"gene": "STXBP1", "disease": "developmental and epileptic encephalopathy 4", "score": 0.74},
    {"gene": "CDKL5", "disease": "developmental and epileptic encephalopathy 2", "score": 0.69},
    {"gene": "UBE3A", "disease": "Angelman syndrome", "score": 0.66},
    {"gene": "SCN2A", "disease": "developmental and epileptic encephalopathy 11", "score": 0.61},
]


def evidence_paths(gene: str) -> list[dict]:
    return [
        {
            "path": [
                {"node_id": "HP:0001250", "node_type": "phenotype", "name": "Seizure"},
                {"node_id": "Dravet syndrome", "node_type": "disease", "name": "Dravet syndrome"},
                {"node_id": gene, "node_type": "gene", "name": gene},
            ],
            "relations": ["disease_phenotype_positive", "disease_protein"],
        }
    ]


def mechanism(disease: str) -> dict:
    return {
        "disease": disease,
        "mechanism_summary": "STUB: mechanism lookup (DisMech) not implemented yet.",
        "key_phenotypes": [],
        "source": "stub",
    }


FOLLOWUP = {
    "question": "Are the seizures triggered by fever?",
    "hpo_id": "HP:0002373",
    "distinguishes": ["Dravet syndrome"],
}
