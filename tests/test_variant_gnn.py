"""Output A on a tiny variant table over the fixture graph. No PrimeKG or real variant table needed."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from geneius.agent.tools import gnn_predict  # noqa: E402
from geneius.agent.tools import predict as predict_tool  # noqa: E402
from geneius.features.variant_features import (  # noqa: E402
    AA,
    BLOSUM62,
    PLACEHOLDER_COLUMNS,
    all_variant_features,
    model_columns,
    model_matrix,
    variant_features,
)
from geneius.data.paralogs import align_map  # noqa: E402
from geneius.features.graph_features import (  # noqa: E402
    canonical_residues,
    paralog_features,
    paralog_links,
    residue_positions,
    structure_features,
    structure_pairs,
)
from geneius.graph.build import USED_COLUMNS, VARIANT_DISEASE, VARIANT_GENE, build_graph  # noqa: E402
from geneius.models import gnn  # noqa: E402

ROWS = [
    # variant_id, gene, position, ref, alt, class_2023, split, disease_2023, class_now
    (1, "RHO", 23, "P", "H", "P", "train", "retinitis pigmentosa|not provided", "P"),
    (2, "RHO", 90, "G", "D", "P", "train", "retinitis pigmentosa", "P"),
    (3, "USH2A", 759, "C", "F", "P", "train", "Usher syndrome", "P"),
    (4, "USH2A", 34, "R", "Q", "B", "train", "not provided", "B"),
    (5, "SCN1A", 101, "R", "Q", "P", "train", "Dravet syndrome", "P"),
    (6, "SCN1A", 200, "A", "V", "B", "train", "not specified", "B"),
    (7, "UBE3A", 50, "L", "P", "P", "train", "Angelman syndrome", "P"),
    (8, "UBE3A", 60, "K", "R", "B", "train", "not provided", "B"),
    (9, "RHO", 28, "Q", "R", "VUS", "test", "retinitis pigmentosa", "P"),
    (10, "USH2A", 100, "S", "F", "VUS", "unlabeled", "Usher syndrome", "VUS"),
    (11, "UBE3A", 70, "D", "N", "conflicting", "exclude", "Angelman syndrome", "P"),
]


def variants() -> pd.DataFrame:
    df = pd.DataFrame(ROWS, columns=["variant_id", "gene", "position", "ref_aa", "alt_aa", "class_2023", "split",
                                     "disease_2023", "class_now"])
    df["stars_2023"] = 2
    df["pheno_ids_2023"] = "MedGen:CN517202"
    df["uniprot_pos"] = df["position"].astype(float)
    df["uniprot_ref"] = df["ref_aa"]
    df["stars_now"] = 2
    df["disease_now"] = df["disease_2023"]
    df["pheno_ids_now"] = "MONDO:MONDO:0019200"
    return df


def _graph(df=None):
    return build_graph(variants() if df is None else df, cache=False, verbose=False)


def test_variant_features():
    assert all(BLOSUM62[(a, b)] == BLOSUM62[(b, a)] for a in AA for b in AA)
    f = variant_features(variants())
    assert f.loc[0, "blosum62"] == BLOSUM62[("P", "H")]
    assert f.loc[3, "charge_change"] == -1.0  # R -> Q
    assert f.loc[6, "to_proline"] == 1.0 and f.loc[0, "from_proline"] == 1.0
    assert f[PLACEHOLDER_COLUMNS].isna().all().all()
    assert not set(PLACEHOLDER_COLUMNS) & set(model_columns(f))


def test_feature_table_fills_placeholders_and_ignores_post_2023_columns(tmp_path, monkeypatch):
    table = pd.DataFrame({
        "variation_id": [1, 9], "variant_id": ["RHO:P23H", "RHO:Q28R"], "split": ["train", "test"],
        "cls_2023": ["P", "VUS"], "cls_now": ["P", "P"], "stars_now": [3, 2], "diseases_now": ["x", "y"],
        "am_score": [0.9, np.nan], "plddt": [91.0, 80.0], "in_domain": [1, 0], "transmembrane": [1, 1],
        "active_site": [0, 0], "binding_site": [0, 1], "disulfide_cys": [0, 0], "topo_extracellular": [1, 0],
        "topo_cytoplasmic": [0, 1], "topo_other": [0, 0], "dist_to_site": [4.2, 30.0],
    })
    path = tmp_path / "feature_table.csv"
    table.to_csv(path, index=False)
    monkeypatch.setenv("GENEIUS_FEATURE_TABLE", str(path))
    f = all_variant_features(variants())
    assert f.loc[0, "alphamissense"] == 0.9 and f.loc[0, "plddt"] == 91.0 and f.loc[0, "dist_to_site"] == 4.2
    assert np.isnan(f.loc[8, "alphamissense"]) and f.loc[8, "in_binding_site"] == 1.0
    assert np.isnan(f.loc[1, "plddt"])  # variant 2 isn't in the table
    assert f.loc[0, "blosum62"] == BLOSUM62[("P", "H")]  # computed features kept
    assert not [c for c in f.columns if c.endswith("_now") or c in ("split", "cls_2023", "cls_now")]
    g = build_graph(variants(), cache=False, verbose=False)
    assert "alphamissense" in g.feature_columns and "in_modified_residue" not in g.feature_columns
    assert "alphamissense" not in build_graph(variants(), cache=False, verbose=False,
                                              drop_features=("alphamissense",)).feature_columns


def test_align_map_and_paralog_links():
    m = align_map("MKTAYIAK", "MKTAYAK")  # paralog lacks the I at panel residue 6
    assert m[1] == 1 and m[5] == 5 and 6 not in m and m[7] == 6 and m[8] == 7
    panel = pd.DataFrame({"gene": ["RHO", "RHO", "RHO"], "uniprot_pos": [5.0, 7.0, np.nan],
                          "uniprot_ref": ["Y", "A", None], "position": [5, 7, 99], "ref_aa": ["Y", "A", "Q"]})
    residues = residue_positions(panel, {"RHO": "MKTAYIAK"})
    assert residues.tolist() == [5, 7, 0]
    para = pd.DataFrame({"variant_id": ["p1", "p2", "p3"], "gene": ["OPN1LW"] * 3, "pos": [5, 6, 4],
                         "ref": ["Y", "A", "A"], "alt": ["C", "P", "V"], "stars": [1, 2, 1]})
    links = paralog_links(panel, residues, para, {("RHO", "OPN1LW"): m})
    got = sorted(map(tuple, links.to_numpy().tolist()))
    # panel 5 -> paralog 5 (same p1, near p2 at 6 and p3 at 4); panel 7 -> paralog 6 (same p2, near p1 at 5)
    assert got == [(0, 0, "same"), (0, 1, "near"), (0, 2, "near"), (1, 0, "near"), (1, 1, "same")]
    f = paralog_features(3, links)
    assert f["paralog_p_same"].tolist() == [1.0, 1.0, 0.0] and f["paralog_p_near"].tolist() == [2.0, 1.0, 0.0]


def test_canonical_residues_from_feature_table(tmp_path, monkeypatch):
    table = pd.DataFrame({"variation_id": [1, 2, 3, 4], "pos": [5, 7, 2, 6], "ref": ["Y", "A", "K", "Q"],
                          "numbering": ["uniprot_canonical", "mapped_via_transcript", "unmappable", "uniprot_canonical"]})
    path = tmp_path / "feature_table.csv"
    table.to_csv(path, index=False)
    monkeypatch.setenv("GENEIUS_FEATURE_TABLE", str(path))
    v = pd.DataFrame({"variant_id": [1, 2, 3, 4, 5], "gene": ["RHO"] * 5})
    # 2 is re-mapped and used; 3 is unmappable; 4's ref doesn't match the sequence; 5 isn't in the table.
    assert canonical_residues(v, {"RHO": "MKTAYIAK"}).tolist() == [5, 7, 0, 0, 0]
    monkeypatch.setenv("GENEIUS_FEATURE_TABLE", str(tmp_path / "missing.csv"))
    assert canonical_residues(v, {"RHO": "MKTAYIAK"}) is None


def test_structure_pairs_and_label_counts():
    # Residues 1-3 within 10 A of each other; residue 4 is close but low confidence; residue 5 is far.
    coords = {"RHO": {1: (0, 0, 0, 90), 2: (3, 0, 0, 90), 3: (6, 0, 0, 80), 4: (1, 1, 0, 50), 5: (40, 0, 0, 95)}}
    v = pd.DataFrame({"gene": ["RHO"] * 6})
    residues = np.array([1, 2, 3, 4, 5, 1])  # variant 5 sits on residue 1 too
    pairs = structure_pairs(v, residues, coords)
    got = {(int(a), int(b)) for a, b in zip(pairs["a"], pairs["b"])}
    assert got == {(a, b) for a in (0, 1, 2, 5) for b in (0, 1, 2, 5) if a != b}
    assert pairs.loc[(pairs["a"] == 0) & (pairs["b"] == 2), "distance"].item() == pytest.approx(6.0)
    # Labels: variant 1 pathogenic training, variant 2 benign training, variant 5 held out (NaN).
    f = structure_features(6, pairs, np.array([np.nan, 1.0, 0.0, np.nan, np.nan, np.nan]))
    assert f.loc[0, "struct_p_10a"] == 1 and f.loc[0, "struct_b_10a"] == 1 and f.loc[0, "struct_p_frac_10a"] == 0.5
    assert f.loc[1, "struct_p_10a"] == 0  # never counts itself
    assert f.loc[3, "struct_p_10a"] == 0 and np.isnan(f.loc[3, "struct_p_frac_10a"])  # low pLDDT: no edges


def test_holdout_gene_loses_labels_and_disease_edges():
    df = variants()
    g = build_graph(df, cache=False, verbose=False, holdout_genes=("RHO",))
    rho_train = ((df["gene"] == "RHO") & (df["split"] == "train")).to_numpy()
    assert not g.data["variant"].train_mask[torch.tensor(rho_train)].any()
    assert torch.isnan(g.data["variant"].y[torch.tensor(rho_train)]).all()
    linked = set(g.data[VARIANT_DISEASE].edge_index[0].tolist())
    assert not linked & set(np.flatnonzero(rho_train))
    assert sorted(g.data[VARIANT_GENE].edge_index[0].tolist()) == list(range(len(df)))  # still in the graph


def test_graph_with_3d_edges(tmp_path, monkeypatch):
    import json
    import pickle

    seq = "M" * 300
    (tmp_path / "seqs.json").write_text(json.dumps({g: {"uniprot": "X", "sequence": seq}
                                                     for g in ("RHO", "USH2A", "SCN1A", "UBE3A")}))
    coords = {"RHO": {23: (0, 0, 0, 90), 90: (5, 0, 0, 90), 28: (2, 0, 0, 90)}}
    with open(tmp_path / "ca.pkl", "wb") as f:
        pickle.dump(coords, f)
    monkeypatch.setenv("GENEIUS_PARALOG_SEQUENCES", str(tmp_path / "seqs.json"))
    monkeypatch.setenv("GENEIUS_CA_COORDS", str(tmp_path / "ca.pkl"))
    from geneius.data.paralogs import load_sequences

    load_sequences.cache_clear()
    df = variants()
    df["ref_aa"] = "M"
    df["uniprot_ref"] = "M"
    g = build_graph(df, cache=False, verbose=False, structure=True)
    load_sequences.cache_clear()
    ei = g.data[("variant", "near_3d", "variant")].edge_index
    assert {(int(a), int(b)) for a, b in ei.T} == {(a, b) for a in (0, 1, 8) for b in (0, 1, 8) if a != b}
    attr = g.data[("variant", "near_3d", "variant")].edge_attr
    assert attr.shape == (6, 1) and float(attr.max()) == pytest.approx(5.0)  # not doubled by ToUndirected
    assert {"struct_p_10a", "struct_b_10a", "struct_p_frac_10a"} <= set(g.feature_columns)


def test_model_matrix_uses_train_medians_and_optional_am_indicator():
    feats = pd.DataFrame({"alphamissense": [0.2, np.nan, 0.8, 0.0], "plddt": [50.0, 70.0, np.nan, 99.0]})
    is_train = np.array([True, True, True, False])
    a = model_matrix(feats, ["alphamissense", "plddt"], is_train, am_indicator=True)
    assert a["alphamissense"].tolist() == [0.2, 0.5, 0.8, 0.0]  # train median of 0.2 and 0.8
    assert a["plddt"].tolist() == [50.0, 70.0, 60.0, 99.0]
    assert a["am_missing"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert "am_missing" not in model_matrix(feats, ["alphamissense", "plddt"], is_train, am_indicator=False)


def test_graph_reads_no_post_2023_columns():
    assert "split" in USED_COLUMNS
    assert not [c for c in USED_COLUMNS if c.endswith("_now")]
    g = _graph()
    assert not [c for c in g.feature_columns if c.endswith("_now") or c == "split"]
    assert set(g.data["variant"].keys()) == {"x", "y", "train_mask", "num_nodes"}

    scrambled = variants()
    for col in [c for c in scrambled if c.endswith("_now")]:
        scrambled[col] = scrambled[col].sample(frac=1.0, random_state=1).to_numpy()
    scrambled["class_now"] = "B"
    h = _graph(scrambled)
    assert torch.equal(g.data["variant"].x, h.data["variant"].x)
    assert torch.equal(torch.nan_to_num(g.data["variant"].y, -1), torch.nan_to_num(h.data["variant"].y, -1))
    for etype in g.data.edge_types:
        assert torch.equal(g.data[etype].edge_index, h.data[etype].edge_index)


def test_only_train_variants_get_labels_and_disease_edges():
    df = variants()
    g = _graph(df)
    data = g.data
    is_train = (df["split"] == "train").to_numpy()
    assert data["variant"].train_mask.numpy().tolist() == is_train.tolist()
    assert torch.isnan(data["variant"].y[~torch.tensor(is_train)]).all()
    assert data["variant"].y[torch.tensor(is_train)].tolist() == [1, 1, 1, 0, 1, 0, 1, 0]

    assert sorted(data[VARIANT_GENE].edge_index[0].tolist()) == list(range(len(df)))
    linked = set(data[VARIANT_DISEASE].edge_index[0].tolist())
    assert linked <= set(np.flatnonzero(is_train))
    # Variant 9 is a test VUS whose 2023 condition names a fixture disease. It must not be linked.
    assert 8 not in linked
    assert g.disease_match["matched"] == 5 and g.disease_match["by_name"] == 5


def test_train_and_predict(tmp_path, monkeypatch):
    monkeypatch.setenv("GENEIUS_GNN_A_DIR", str(tmp_path))
    df = variants()
    metrics = gnn.train(_graph(df), df["gene"].tolist(), out_dir=tmp_path, hidden=8, epochs=3, log_every=0)
    assert metrics["epochs_run"] >= 1 and metrics["n_val"] >= 1
    assert set(metrics["val_genes"]) <= {"USH2A", "SCN1A", "UBE3A", "RHO"}
    out = gnn.predict(["9", "999"])
    assert set(out[0]) == {"variant", "harmful", "mechanism", "disease", "treatment", "confidence"}
    assert 0.0 <= out[0]["harmful"] <= 1.0 and out[0]["confidence"] in {"high", "medium", "low"}
    assert out[0]["mechanism"] is None and out[1]["harmful"] is None


def test_gnn_predict_tool(tmp_path, monkeypatch):
    csv = tmp_path / "variants.csv"
    df = variants()
    df.to_csv(csv, index=False)
    monkeypatch.setenv("GENEIUS_VARIANTS_CSV", str(csv))
    monkeypatch.setenv("GENEIUS_GNN_A_DIR", str(tmp_path / "gnn_a"))
    predict_tool._variant_index.cache_clear()
    gnn.train(_graph(df), df["gene"].tolist(), out_dir=tmp_path / "gnn_a", hidden=8, epochs=2, log_every=0)

    for name in ("RHO p.Gln28Arg", "RHO Q28R", "9"):
        out = gnn_predict(name)
        assert out["variant"] == "RHO p.Q28R" and out["gene"] == "RHO"
        assert out["clinvar_class_2023"] == "VUS" and isinstance(out["harmful"], float)
        assert "class_now" not in out and "stub" not in out
    assert "error" in gnn_predict("RHO p.Pro999His")
    predict_tool._variant_index.cache_clear()
    gnn.clear_cache()
