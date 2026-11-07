"""The five agent tools. Everyone builds against these signatures.

map_to_hpo(text) -> [{hpo_id, label, matched_text}]
rank_candidates(hpo_ids, top_k=10) -> [{gene, disease, score}]
get_evidence_paths(hpo_ids, gene, max_paths=5) -> [{path: [{node_id, node_type, name}], relations}]
get_mechanism(disease) -> {disease, mechanism_summary, key_phenotypes, source}
suggest_followup(candidates, known_hpo_ids) -> {question, hpo_id, distinguishes}

All return plain JSON-serializable data. HPO IDs look like "HP:0001250", genes are HGNC symbols.
"""
from .evidence import get_evidence_paths
from .followup import suggest_followup
from .map_to_hpo import map_to_hpo
from .mechanism import get_mechanism
from .rank_candidates import rank_candidates

TOOL_FUNCTIONS = {
    "map_to_hpo": map_to_hpo,
    "rank_candidates": rank_candidates,
    "get_evidence_paths": get_evidence_paths,
    "get_mechanism": get_mechanism,
    "suggest_followup": suggest_followup,
}

__all__ = list(TOOL_FUNCTIONS) + ["TOOL_FUNCTIONS"]
