import sys
from pathlib import Path

import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from geneius.gnn import predict, train  # noqa: E402
from geneius.tools import rank_candidates  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
import score  # noqa: E402
import simulate_patients  # noqa: E402

TINY = ["--hidden", "16", "--epochs", "5", "--log-every", "0"]


@pytest.fixture
def gnn_dir(tmp_path, monkeypatch):
    out = tmp_path / "gnn"
    monkeypatch.setenv("GENEIUS_GNN_DIR", str(out))
    predict.clear_cache()
    yield out
    predict.clear_cache()


@pytest.fixture
def trained(tmp_path, gnn_dir):
    held = tmp_path / "held.txt"
    held.write_text("Dravet syndrome\n")
    metrics = train.main(["--heldout", str(held), *TINY])
    return metrics


def test_train_writes_checkpoint(trained, gnn_dir):
    for name in ("model.pt", "embeddings.pt", "index.json", "metrics.json"):
        assert (gnn_dir / name).exists()
    assert trained["epochs_run"] >= 1
    assert trained["heldout_diseases"] == 1
    assert predict.get_gnn_ranker("heldout").heldout == {"Dravet syndrome"}


def test_validation_path_and_dot_decoder(tmp_path, gnn_dir):
    # No held-out list: all 10 fixture edges stay, so one becomes a validation edge.
    variants = (
        [],
        ["--decoder", "cosine", "--negatives", "10", "--neg-sampling", "degree"],
        ["--no-pair-norm", "--stop-on", "auroc"],
    )
    for extra in variants:
        metrics = train.main(["--heldout", str(tmp_path / "missing.txt"), *TINY, *extra])
        assert 0.0 <= metrics["best_val"]["hits@10"] <= 1.0
        assert 0.0 < metrics["best_val"]["mrr"] <= 1.0
        predict.clear_cache()
        assert predict.rank_genes(["HP:0001250"], graph="full")


def test_filtered_ranking_ignores_other_true_diseases():
    class Identity:
        def project(self, x):
            return x

    # Phenotype 0 belongs to diseases 0 and 1. Disease 1 outscores disease 0, but it is a known
    # answer, so the validation edge (0, 0) still ranks first. Disease 2 is not, and would beat it.
    h = {"disease": torch.tensor([[1.0], [3.0], [0.5]]), "phenotype": torch.tensor([[1.0]])}
    val_pos = torch.tensor([[0], [0]])
    all_pos = torch.tensor([[0, 1], [0, 0]])
    known = train._known_diseases(val_pos, all_pos)
    assert train._val_ranking(Identity(), h, val_pos, known) == (1.0, 1.0)
    h["disease"][2] = 2.0
    assert train._val_ranking(Identity(), h, val_pos, known) == (1.0, 0.5)


def test_disease_nodes_share_one_vector(trained):
    model = predict.load_model()
    embeddings = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Embedding)}
    assert embeddings == {"gene", "phenotype"}
    assert tuple(model.disease.shape) == (1, 16)


def test_heldout_disease_still_scored(trained):
    out = predict.rank_genes(["HP:0002373", "HP:0001252"], top_k=10, graph="heldout")
    assert out and all(set(c) == {"gene", "disease", "score"} for c in out)
    assert all(0.0 < c["score"] < 1.0 for c in out)
    assert [c["score"] for c in out] == sorted((c["score"] for c in out), reverse=True)
    by_gene = {c["gene"]: c["disease"] for c in out}
    assert by_gene["SCN1A"] == "Dravet syndrome"


def test_unknown_terms_are_skipped(trained):
    assert predict.rank_genes(["HP:9999999"], graph="full") == []


@pytest.mark.parametrize("ranker", ["gnn", "hybrid"])
def test_rank_candidates_contract(trained, monkeypatch, ranker):
    monkeypatch.setenv("GENEIUS_RANKER", ranker)
    out = rank_candidates(["HP:0002373", "HP:0001252"], top_k=3)
    assert 0 < len(out) <= 3
    assert all(set(c) == {"gene", "disease", "score"} for c in out)


def test_fuse_is_reciprocal_rank_fusion():
    a = [{"gene": "A", "disease": "a1"}, {"gene": "B", "disease": "b1"}]
    b = [{"gene": "B", "disease": "b2"}, {"gene": "C", "disease": "c2"}]
    out = predict.fuse([a, b], top_k=3)
    assert [c["gene"] for c in out] == ["B", "A", "C"]
    assert out[0]["disease"] == "b2"
    assert out[0]["score"] == round(1 / 62 + 1 / 61, 6)


def test_score_all_methods_both_modes(tmp_path, gnn_dir):
    sim, held = tmp_path / "sim.jsonl", tmp_path / "held.txt"
    simulate_patients.main(["--n", "3", "--min-terms", "2", "--out", str(sim), "--heldout-out", str(held)])
    train.main(["--heldout", str(held), *TINY])
    for mode in ("heldout", "transductive"):
        results = score.main(["--patients", str(sim), "--heldout", str(held), "--method", "all", "--mode", mode])
        assert set(results) == {"overlap", "gnn", "hybrid"}
        for r in results.values():
            assert r["n"] == 3
            assert 0.0 <= r["hits@1"] <= r["hits@5"] <= r["hits@10"] <= 1.0


def test_heldout_eval_rejects_leaked_diseases(tmp_path, trained):
    held = tmp_path / "other.txt"
    held.write_text("Angelman syndrome\n")
    sim = tmp_path / "sim.jsonl"
    sim.write_text('{"hpo_ids": ["HP:0001250"], "true_genes": ["UBE3A"]}\n')
    with pytest.raises(ValueError, match="not held out"):
        score.main(["--patients", str(sim), "--heldout", str(held), "--method", "gnn"])
