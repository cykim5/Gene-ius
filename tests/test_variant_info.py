"""population_evidence, stability, mouse_evidence, show_structure, and the abstain check on fixture demo data."""
from geneius.agent.tools import evidence_report, mouse_evidence, population_evidence, show_structure, stability
from geneius.agent.tools.variant_info import NOT_SEEN


def test_population_counts_unique_non_trio_carriers():
    rec = population_evidence("RHO:C187S")["records"][0]
    # S1 (EUR, het), S2 (AFR, hom, listed twice), S3 is a child in a trio and is excluded.
    assert rec["carriers"] == 2 and rec["homozygotes"] == 1
    assert rec["by_superpop"] == {"AFR": 1, "EUR": 1, "SAS": 0, "EAS": 0, "AMR": 0}
    assert rec["seen_in_apparently_healthy_people"] is True and rec["healthy_carriers"]["carriers"] == 2


def test_population_not_seen_wording_and_two_hit_match():
    rec = population_evidence("RHO:P23H")["records"][0]
    assert rec["summary"] == NOT_SEEN and rec["carriers"] == 0
    assert rec["seen_in_apparently_healthy_people"] is False and "two_hit_healthy_people" not in rec
    ush = population_evidence("USH2A:C759F")["records"]
    assert len(ush) == 2 and all(r["two_hit_healthy_people"][0]["sample"] == "S9" for r in ush)
    assert population_evidence("RHO:P999H")["status"] == "not found"


def test_stability_only_where_foldx_ran():
    assert stability("RHO:P23H")["records"][0]["ddg_kcal_mol"] == 3.2
    assert stability("RHO:C187S")["records"][0]["status"] == "not computed"  # RHO, but no FoldX result
    assert stability("SCN1A:R101Q")["records"][0]["status"] == "not computed"


def test_mouse_evidence_is_gene_level_mgi():
    out = mouse_evidence("rho")
    assert out["source"] == "MGI" and "gene-level" in out["level"] and "vision/eye" in out["evidence"]
    assert mouse_evidence("SCN1A")["status"] == "no MGI mouse phenotypes on file"


def test_show_structure_demo_variants_only():
    assert show_structure("RHO:C187S")["viewer_file"].endswith("viewer_RHO_pC187S.html")
    assert show_structure("RHO:P23H")["status"] == "3D view available for demo variants only."


def test_abstain_reasons():
    sure = evidence_report("RHO:P23H")["records"][0]["assessment"]
    assert sure == {"unsure": False, "reasons": []}
    conflict = evidence_report("RHO:C187S")["records"][0]["assessment"]
    assert conflict["unsure"] and any("healthy people" in r for r in conflict["reasons"])
    ush = evidence_report("USH2A:C759F")["records"][0]["assessment"]
    assert ush["unsure"] and len(ush["reasons"]) == 3  # indeterminate, unsure, two-hit healthy carrier
    assert "suggest_assay" in ush["next_step"]
    tm = evidence_report("USH2A:C759F", time_machine=True)["records"][0]
    assert "assessment" not in tm and "computational_evidence" not in tm
