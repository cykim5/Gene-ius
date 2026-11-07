import json

from geneius.data import get_hpo, get_kg, norm_hpo
from geneius.pipeline import run_pipeline
from geneius.tools import (
    get_evidence_paths,
    get_mechanism,
    map_to_hpo,
    rank_candidates,
    suggest_followup,
)


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


def test_map_to_hpo():
    out = map_to_hpo("4 year old with seizures, low muscle tone and speech delay")
    assert [t["hpo_id"] for t in out] == ["HP:0001250", "HP:0001252", "HP:0000750"]
    assert out[1]["matched_text"] == "low muscle tone"


def test_rank_candidates():
    out = rank_candidates(["HP:0002373", "HP:0001252"], top_k=3)
    assert out[0]["gene"] == "SCN1A"
    assert out[0]["disease"] == "Dravet syndrome"
    assert set(out[0]) == {"gene", "disease", "score"}


def test_evidence_paths():
    paths = get_evidence_paths(["HP:0002373", "HP:0001250"], "SCN1A")
    assert paths
    for p in paths:
        assert len(p["relations"]) == len(p["path"]) - 1
        assert p["path"][-1]["node_id"] == "SCN1A"


def test_followup_splits_candidates():
    cands = [{"gene": "SCN1A", "disease": "Dravet syndrome"}, {"gene": "UBE3A", "disease": "Angelman syndrome"}]
    out = suggest_followup(cands, ["HP:0001250"])
    assert out["hpo_id"] in {"HP:0002373", "HP:0001252", "HP:0000750"}
    assert len(out["distinguishes"]) == 1


def test_all_outputs_are_json():
    json.dumps(get_mechanism("Dravet syndrome"))
    json.dumps(run_pipeline("febrile seizures and hypotonia"))


def test_pipeline_end_to_end():
    result = run_pipeline("febrile seizures and hypotonia")
    assert result["candidates"][0]["gene"] == "SCN1A"
    assert result["candidates"][0]["evidence"]
