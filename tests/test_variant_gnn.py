"""Output A on a tiny variant table over the fixture graph. No PrimeKG or real variant table needed."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from geneius.agent.tools import gnn_predict  # noqa: E402
from geneius.agent.tools import predict as predict_tool  # noqa: E402
from geneius.features.variant_features import AA, BLOSUM62, PLACEHOLDER_COLUMNS, model_columns, variant_features  # noqa: E402
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
