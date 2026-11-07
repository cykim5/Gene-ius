import sys
from pathlib import Path

from geneius.gnn.graph_build import training_edges
from geneius.tools import TOOL_FUNCTIONS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import score  # noqa: E402
import simulate_patients  # noqa: E402


def test_stub_shapes(monkeypatch):
    monkeypatch.setenv("GENEIUS_USE_STUBS", "1")
    assert set(TOOL_FUNCTIONS["map_to_hpo"]("x")[0]) == {"hpo_id", "label", "matched_text"}
    assert set(TOOL_FUNCTIONS["rank_candidates"](["HP:0001250"])[0]) == {"gene", "disease", "score"}
    assert set(TOOL_FUNCTIONS["get_evidence_paths"](["HP:0001250"], "SCN1A")[0]) == {"path", "relations"}
    assert set(TOOL_FUNCTIONS["suggest_followup"]([], [])) == {"question", "hpo_id", "distinguishes"}


def test_training_edges_remove_leaks():
    df = training_edges({"Dravet syndrome"})
    dravet_pheno = df[(df.relation == "disease_phenotype_positive")
                      & ((df.x_name == "Dravet syndrome") | (df.y_name == "Dravet syndrome"))]
    assert dravet_pheno.empty
    assert (df.relation == "phenotype_protein").sum() == 0
    assert (df.relation == "disease_protein").sum() == 5
    assert (df.x_name == "Angelman syndrome").sum() == 4


def test_simulate_and_score(tmp_path):
    out, held = tmp_path / "sim.jsonl", tmp_path / "held.txt"
    simulate_patients.main(["--n", "3", "--min-terms", "2", "--out", str(out), "--heldout-out", str(held)])
    result = score.main(["--patients", str(out), "--heldout", str(held)])["overlap"]
    assert result["n"] == 3
    assert 0.0 <= result["hits@1"] <= result["hits@10"] <= 1.0
