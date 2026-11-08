"""patient_rank and evidence_report on fixture demo data (tests/fixtures/processed) and the fixture graph."""
import math

from geneius.agent.tools import evidence_report, patient_rank
from geneius.data import demo


def test_resolve_on_canonical_position():
    assert demo.resolve("RHO:P23H") == ["13013"]
    assert demo.resolve("RHO p.Pro23His") == ["13013"]
    assert sorted(demo.resolve("USH2A:C759F")) == ["2356", "2357"]  # two DNA changes, one protein change
    assert demo.resolve("SCN1A:A900V") == []  # unmappable rows have no canonical residue
    assert demo.resolve("RHO:P999H") == [] and demo.resolve("nonsense") == []


def test_patient_rank_scores_and_order():
    # Rod-cone dystrophy fits retinitis pigmentosa (RHO, USH2A) and Usher syndrome (USH2A), not Dravet (SCN1A).
    out = patient_rank(["HP:0000510"], ["SCN1A:R101Q", "RHO:P23H", "USH2A:C759F", "RHO:P999H"])
    ranked = out["ranked"]
    assert out["harmful_source"] == "current model"
    assert [r["variation_id"] for r in ranked][:1] == ["13013"]
    assert max(r["phenotype_match"] for r in ranked) == 1.0
    scn1a = next(r for r in ranked if r["gene"] == "SCN1A")
    assert scn1a["phenotype_match"] == 0.0 and scn1a["combined"] == 0.0 and scn1a["rank"] == len(ranked)
    for r in ranked:
        assert r["combined"] == round(math.sqrt(r["harmful"] * r["phenotype_match"]), 4)
    assert {r["variation_id"] for r in ranked if r["gene"] == "USH2A"} == {"2356", "2357"}
    assert any("2 ClinVar records" in n for n in out["notes"])
    assert out["not_found"] == [{"query": "RHO:P999H", "status": "not found"}]


def test_time_machine_uses_only_2023_model():
    now = patient_rank(["HP:0000510"], ["RHO:P23H"])["ranked"][0]
    then = patient_rank(["HP:0000510"], ["RHO:P23H"], time_machine=True)["ranked"][0]
    assert now["harmful"] == 0.98 and then["harmful"] == 0.95


def test_evidence_report_fields_and_time_machine():
    rec = evidence_report("RHO:P23H")["records"][0]
    assert rec["computational_evidence"] == "PP3_Strong" and rec["suggested_assay"] == "rhodopsin trafficking"
    assert rec["mechanism"]["ddg_kcal_mol"] == 3.2 and rec["clinvar_change"] == "p.P23H"
    old = evidence_report("RHO:P23H", time_machine=True)["records"][0]
    assert set(old) == {"variant", "variation_id", "gene", "clinvar_change", "harmful_2023_model", "evidence_2023", "note"}
    assert evidence_report("RHO:P999H")["status"] == "not found"
