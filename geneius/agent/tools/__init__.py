"""The agent's tools (Stage 7). All read-only, all return JSON-serializable data.

| Tool                 | Status                          |
| map_symptoms         | works (dictionary matcher)      |
| find_entity          | works for genes, diseases, HPO  |
| expand               | works on PrimeKG (no variants)  |
| find_paths           | works for symptom -> gene paths |
| structural_neighbors | stub                            |
| gnn_predict          | stub                            |
| gnn_score            | stub                            |
| evidence_by_source   | stub                            |
| suggest_treatment    | works (rule table)              |
| suggest_assay        | works (rule table)              |
| evidence_report      | works (predictions.json)        |
| show_structure       | stub                            |
| patient_rank         | works (harm x phenotype fit)    |
"""
from .evidence import evidence_by_source, evidence_report
from .graph_tools import expand, find_entity, find_paths
from .patient import patient_rank
from .predict import gnn_predict, gnn_score
from .structure import show_structure, structural_neighbors
from .symptoms import map_symptoms
from .treatment import suggest_assay, suggest_treatment

TOOL_FUNCTIONS = {
    "map_symptoms": map_symptoms,
    "find_entity": find_entity,
    "expand": expand,
    "find_paths": find_paths,
    "structural_neighbors": structural_neighbors,
    "gnn_predict": gnn_predict,
    "gnn_score": gnn_score,
    "evidence_by_source": evidence_by_source,
    "suggest_treatment": suggest_treatment,
    "suggest_assay": suggest_assay,
    "evidence_report": evidence_report,
    "show_structure": show_structure,
    "patient_rank": patient_rank,
}

__all__ = list(TOOL_FUNCTIONS) + ["TOOL_FUNCTIONS"]
