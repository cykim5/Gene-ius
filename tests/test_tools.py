import json
import sys
from pathlib import Path

from geneius.agent.tools import (
    TOOL_FUNCTIONS,
    expand,
    find_entity,
    find_paths,
    map_symptoms,
    suggest_assay,
    suggest_treatment,
)
from geneius.data import get_hpo, get_kg, load_variants, norm_hpo
from geneius.data.clinvar import parse_protein_change, review_stars
from geneius.graph.leakage import remove_heldout_disease_edges
from geneius.labels import harmful_label
from geneius.models.phenotype_overlap import get_overlap_ranker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from time_saved import found_in_top_k, random_expected  # noqa: E402


def test_norm_hpo():
    assert norm_hpo("1250") == "HP:0001250"
    assert norm_hpo("1250.0") == "HP:0001250"
    assert norm_hpo("HP:0001250") == "HP:0001250"


def test_hpo_parsing():
    hpo = get_hpo()
    assert hpo.label("HP:0001250") == "Seizure"
    assert "HP:9999999" not in hpo.names
    assert hpo.primary("HP:0002279") == "HP:0001250"
    assert "HP:0000118" in hpo.ancestors("HP:0002373")


def test_kg_loads_both_edge_directions():
    kg = get_kg()
    assert "HP:0001252" in kg.disease_hpo["Dravet syndrome"]
    assert kg.disease_genes["retinitis pigmentosa"] == {"RHO", "USH2A"}
    assert kg.hpo_genes["HP:0001250"] == {"SCN1A"}


def test_map_symptoms():
    out = map_symptoms("4 year old with seizures, low muscle tone and speech delay")
    assert [t["hpo_id"] for t in out] == ["HP:0001250", "HP:0001252", "HP:0000750"]
    assert out[1]["matched_text"] == "low muscle tone"


def test_phenotype_overlap_baseline():
    out = get_overlap_ranker().rank_genes(["HP:0002373", "HP:0001252"], top_k=3)
    assert out[0]["gene"] == "SCN1A"


def test_find_paths():
    paths = find_paths(["HP:0002373", "HP:0001250"], "SCN1A")
    assert paths
    for p in paths:
        assert len(p["relations"]) == len(p["path"]) - 1
        assert p["path"][-1]["node_id"] == "SCN1A"


def test_find_entity():
    assert find_entity("rho")["matches"][0] == {"node_id": "RHO", "node_type": "gene", "name": "RHO"}
    assert find_entity("Seizure")["matches"][0]["node_id"] == "HP:0001250"


def test_expand():
    # Fixture: Dravet syndrome has 4 phenotypes, gene SCN1A, and drug Stiripentol. Seizure is shared
    # with Angelman syndrome and SCN1A (degree 3), the top phenotype degree, so it is the generic hub.
    out = expand("Dravet syndrome")
    assert out["found"] and out["node_type"] == "disease" and out["total"] == 6
    names = [n["name"] for n in out["neighbors"]]
    assert names == ["SCN1A", "Febrile seizure", "Hypotonia", "Global developmental delay", "Stiripentol", "Seizure"]
    assert set(out["neighbors"][0]) == {"node_id", "node_type", "name", "relation", "degree"}
    assert out["neighbors"][-1] == {"node_id": "HP:0001250", "node_type": "phenotype", "name": "Seizure",
                                    "relation": "disease_phenotype_positive", "degree": 3}

    phenos = expand("Dravet syndrome", node_type="phenotype", limit=2)
    assert phenos["total"] == 4
    assert [n["node_id"] for n in phenos["neighbors"]] == ["HP:0002373", "HP:0001252"]

    assert [n["name"] for n in expand("Dravet syndrome", relation="indication")["neighbors"]] == ["Stiripentol"]
    assert {n["name"] for n in expand("rho", node_type="disease")["neighbors"]} == {"retinitis pigmentosa"}
    assert expand("HP:0001250", node_type="disease")["total"] == 2
    assert expand("not a node") == {"node_id": "not a node", "found": False, "total": 0, "neighbors": []}

    empty = expand("rho", node_type="gene", relation="disease_protein")
    assert empty["neighbors"] == [] and empty["available_node_types"] == {"disease": 1}
    assert empty["available_relations"] == ["disease_protein"]


def test_treatment_and_assay_rules():
    assert suggest_treatment("loss_of_function")["strategy"] == "gene_augmentation"
    assert suggest_treatment("gain_of_function", gene="SCN5A")["strategy"] == "channel_blocker"
    assert suggest_treatment("unknown")["strategy"] is None
    assert suggest_assay("KCNQ1")["protein_type"] == "ion channel"
    assert "chaperone" in suggest_assay("GLA")["assay"]


def test_every_tool_returns_json():
    args = {
        "map_symptoms": {"text": "seizures"},
        "find_entity": {"query": "RHO"},
        "expand": {"node_id": "RHO"},
        "find_paths": {"hpo_ids": ["HP:0001250"], "gene": "SCN1A"},
        "structural_neighbors": {"variant": "RHO p.Pro23His"},
        "gnn_predict": {"variant": "RHO p.Pro23His"},
        "gnn_score": {"source": "RHO", "target": "retinitis pigmentosa"},
        "evidence_by_source": {"variant": "RHO p.Pro23His"},
        "suggest_treatment": {"mechanism": "misfolding"},
        "suggest_assay": {"gene": "RHO"},
        "evidence_report": {"variant": "RHO p.Pro23His"},
        "show_structure": {"variant": "RHO p.Pro23His"},
        "patient_rank": {"hpo_ids": ["HP:0000510"], "variants": ["RHO:P23H"]},
        "population_evidence": {"variant": "RHO:C187S"},
        "stability": {"variant": "RHO:P23H"},
        "mouse_evidence": {"gene": "RHO"},
    }
    assert set(args) == set(TOOL_FUNCTIONS)
    for name, kw in args.items():
        json.dumps(TOOL_FUNCTIONS[name](**kw))


def test_clinvar_parsing():
    assert parse_protein_change("NM_000539.3(RHO):c.68C>A (p.Pro23His)") == ("P", 23, "H")
    assert parse_protein_change("NM_000539.3(RHO):c.1A>G (p.Met1=)") is None
    assert parse_protein_change("NM_000539.3(RHO):c.100C>T (p.Gln34Ter)") is None
    assert review_stars("reviewed by expert panel") == 3
    assert review_stars("no assertion criteria provided") == 0


def test_load_variants_hides_test_labels(tmp_path, monkeypatch):
    csv = tmp_path / "variants.csv"
    csv.write_text("variant_id,gene,class_2023,class_now,stars_now,split\n1,RHO,VUS,P,2,test\n")
    monkeypatch.setenv("GENEIUS_VARIANTS_CSV", str(csv))
    assert list(load_variants().columns) == ["variant_id", "gene", "class_2023", "split"]
    assert load_variants(with_test_labels=True)["class_now"].tolist() == ["P"]


def test_harmful_label():
    assert harmful_label("Likely pathogenic") == 1
    assert harmful_label("Benign/Likely benign") == 0
    assert harmful_label("Uncertain significance") is None


def test_leakage_removes_heldout_edges():
    df = remove_heldout_disease_edges({"Dravet syndrome"})
    dravet = df[(df.relation == "disease_phenotype_positive")
                & ((df.x_name == "Dravet syndrome") | (df.y_name == "Dravet syndrome"))]
    assert dravet.empty
    assert (df.relation == "phenotype_protein").sum() == 0
    assert (df.relation == "disease_protein").sum() == 5


def test_time_saved_curve():
    truth = [1, 0, 1, 0, 0, 1]
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
    assert found_in_top_k(truth, scores, ks=(1, 3, 6)) == {1: 1, 3: 2, 6: 3}
    assert random_expected(truth, ks=(2,)) == {2: 1.0}
