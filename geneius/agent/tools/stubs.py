"""Fake tool outputs with the right shapes.

Used until the real tool exists (or data is missing, or GENEIUS_USE_STUBS=1), so the
agent and demo app can be built in parallel. Every stub result has "stub": True.
"""

MAP_SYMPTOMS = [
    {"hpo_id": "HP:0000510", "label": "Rod-cone dystrophy", "matched_text": "retinitis pigmentosa"},
    {"hpo_id": "HP:0000662", "label": "Nyctalopia", "matched_text": "night blindness"},
]


def evidence_paths(gene: str) -> list[dict]:
    return [
        {
            "path": [
                {"node_id": "HP:0000510", "node_type": "phenotype", "name": "Rod-cone dystrophy"},
                {"node_id": "retinitis pigmentosa", "node_type": "disease", "name": "retinitis pigmentosa"},
                {"node_id": gene, "node_type": "gene", "name": gene},
            ],
            "relations": ["disease_phenotype_positive", "disease_protein"],
        }
    ]


def find_entity(query: str) -> dict:
    return {"query": query, "matches": [{"node_id": "RHO", "node_type": "gene", "name": "RHO"}], "stub": True}


def expand(node_id: str, node_type: str | None) -> dict:
    return {
        "node_id": node_id,
        "neighbors": [{"node_id": "retinitis pigmentosa", "node_type": "disease", "relation": "disease_protein"}],
        "stub": True,
    }


def structural_neighbors(variant: str) -> dict:
    return {
        "variant": variant,
        "neighbors_3d": [{"variant": "RHO p.P23H", "distance": 6.2, "label": "pathogenic", "mechanism": "misfolding"}],
        "related_protein": [],
        "stub": True,
    }


def gnn_predict(variant: str) -> dict:
    return {
        "variant": variant,
        "harmful": 0.91,
        "mechanism": {"misfolding": 0.72, "gain_of_function": 0.12, "loss_of_function": 0.10, "dominant_negative": 0.06},
        "disease": {"retinitis pigmentosa": 0.81, "congenital stationary night blindness": 0.14},
        "treatment": "silencing_knockdown_replace",
        "confidence": "high",
        "stub": True,
    }


def gnn_score(source: str, target: str) -> dict:
    return {"source": source, "target": target, "score": 0.5, "stub": True}


def evidence_by_source(variant: str) -> dict:
    return {
        "variant": variant,
        "contributions": {"clinvar": 0.35, "structure": 0.30, "alphamissense": 0.20, "mouse": 0.10, "hpo": 0.05},
        "stub": True,
    }


def evidence_report(variant: str) -> dict:
    return {"variant": variant, "report_markdown": "STUB: evidence report not written yet.", "stub": True}


def show_structure(variant: str) -> dict:
    return {"variant": variant, "uniprot_id": "P08100", "highlight": [23], "neighbors": [], "stub": True}
