"""Ablation and held-out-gene evaluation for Output A.

Configs: base, +paralog, +3D, +both. Each runs in two settings:
  standard   train on split=train, score split=test (headline) and split=test_1star
  held-out   for each gene with >= MIN_TEST test variants, remove all of that gene's variants from
             training (they stay in the graph unlabeled), score that gene's split=test variants, and
             pool across those genes. A per-gene prior is impossible here: the gene has no labels.
             Most of these genes' test variants are all pathogenic, so the same held-out models are
             also scored on the gene's 2023-labeled (split=train) variants, which they never saw.

GNN and XGBoost get the same feature matrix: amino-acid change features, the feature table,
am_missing, and the config's hand-built paralog and 3D counts. The GNN also gets the edges.

    python eval/ablation.py --train   # train every GNN run (one subprocess each, resumable)
    python eval/ablation.py           # score everything into eval/results/ablation_A.*

This script never imports torch: xgboost and torch load different OpenMP runtimes and crash
together on macOS. GNN runs go through `python -m geneius.models.gnn` subprocesses.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from geneius import config  # noqa: E402
from geneius.data import load_variants  # noqa: E402
from geneius.features.graph_features import extra_links  # noqa: E402
from geneius.features.variant_features import all_variant_features, model_columns, model_matrix  # noqa: E402
from reclassification import SEED, gnn_scores, metric_with_ci, xgboost_scores  # noqa: E402

CONFIGS = {"base": [], "+paralog": ["--paralog"], "+3D": ["--structure"], "+both": ["--paralog", "--structure"]}
MIN_TEST = 10
RESULTS = ROOT / "eval" / "results"


def runs_dir() -> Path:
    return config.gnn_a_dir().with_name("gnn_runs_v2")


def heldout_genes(variants: pd.DataFrame) -> list[str]:
    test = variants[(variants["split"] == "test") & variants["class_now"].isin(["P", "B"])]
    counts = test.groupby("gene").size()
    return sorted(counts[counts >= MIN_TEST].index)


def run_name(cfg: str, gene: str | None) -> str:
    tag = cfg.strip("+").replace("3D", "3d")
    return f"std_{tag}" if gene is None else f"ho_{gene}_{tag}"


def train_all(genes: list[str]) -> None:
    jobs = [(cfg, None) for cfg in CONFIGS] + [(cfg, g) for g in genes for cfg in CONFIGS]
    for i, (cfg, gene) in enumerate(jobs, 1):
        out = runs_dir() / run_name(cfg, gene)
        if (out / "predictions.csv").exists():
            print(f"[{i}/{len(jobs)}] {out.name}: done already", flush=True)
            continue
        cmd = [sys.executable, "-W", "ignore", "-m", "geneius.models.gnn", *CONFIGS[cfg], "--out", str(out)]
        if gene:
            cmd += ["--holdout-gene", gene]
        print(f"[{i}/{len(jobs)}] {out.name}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def xgb_matrix(variants: pd.DataFrame, feats: pd.DataFrame, is_train: np.ndarray, cfg: str) -> pd.DataFrame:
    flags = CONFIGS[cfg]
    extra = extra_links(variants, is_train, "--paralog" in flags, "--structure" in flags)
    cols = model_columns(feats)
    if extra:
        feats = pd.concat([feats, extra["features"]], axis=1)
        cols += list(extra["features"].columns)
    return model_matrix(feats, cols, is_train, am_indicator=True)


def score_rows(name: str, y: np.ndarray, p: np.ndarray, rng, **meta) -> dict:
    row = {**meta, "method": name, "n": int(len(y)), "n_benign": int((y == 0).sum())}
    if 0 < y.sum() < len(y):
        row.update(metric_with_ci(y, p, rng))
    else:
        row.update({"auroc": None, "auprc": None, "note": "one class only, AUROC undefined"})
    return row


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train missing GNN runs, then exit")
    args = ap.parse_args(argv)

    variants = load_variants(with_test_labels=True)
    variants["variant_id"] = variants["variant_id"].astype(str)
    genes = heldout_genes(variants)
    if args.train:
        print(f"held-out genes (>= {MIN_TEST} test variants): {genes}", flush=True)
        train_all(genes)
        return []

    feats = all_variant_features(variants)
    label_2023 = (variants["class_2023"] == "P").to_numpy(dtype=int)
    truth = (variants["class_now"] == "P").to_numpy(dtype=int)
    scored = variants["class_now"].isin(["P", "B"]).to_numpy()
    rng = np.random.default_rng(SEED)
    rows = []

    base_train = (variants["split"] == "train").to_numpy()
    for cfg in CONFIGS:
        x = xgb_matrix(variants, feats, base_train, cfg)
        xgb = xgboost_scores(x[base_train], label_2023[base_train], x)
        gnn = gnn_scores(runs_dir() / run_name(cfg, None) / "predictions.csv")
        for split in ("test", "test_1star"):
            m = (variants["split"] == split).to_numpy() & scored
            if gnn is not None:
                rows.append(score_rows("gnn", truth[m], variants.loc[m, "variant_id"].map(gnn).to_numpy(dtype=float),
                                       rng, setting="standard", split=split, config=cfg))
            rows.append(score_rows("xgboost", truth[m], xgb[m], rng, setting="standard", split=split, config=cfg))

    in_gene_train = base_train & variants["class_2023"].isin(["P", "B"]).to_numpy()
    for cfg in CONFIGS:
        pooled = {(s, mth): ([], []) for s in ("test", "2023 labels") for mth in ("gnn", "xgboost")}
        for gene in genes:
            is_train = base_train & (variants["gene"] != gene).to_numpy()
            x = xgb_matrix(variants, feats, is_train, cfg)
            xgb = xgboost_scores(x[is_train], label_2023[is_train], x)
            gnn = gnn_scores(runs_dir() / run_name(cfg, gene) / "predictions.csv")
            in_gene = (variants["gene"] == gene).to_numpy()
            targets = {
                "test": ((variants["split"] == "test").to_numpy() & scored & in_gene, truth),
                "2023 labels": (in_gene_train & in_gene, label_2023),
            }
            for split, (m, y) in targets.items():
                preds = {"xgboost": xgb[m]}
                if gnn is not None:
                    preds["gnn"] = variants.loc[m, "variant_id"].map(gnn).to_numpy(dtype=float)
                for method, p in preds.items():
                    rows.append(score_rows(method, y[m], p, rng, setting="held-out gene", split=split,
                                           config=cfg, gene=gene))
                    pooled[(split, method)][0].append(y[m])
                    pooled[(split, method)][1].append(p)
        for (split, method), (ys, ps) in pooled.items():
            if ys:
                rows.append(score_rows(method, np.concatenate(ys), np.concatenate(ps), rng,
                                       setting="held-out gene", split=split, config=cfg, gene="pooled"))

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "ablation_A.csv", index=False)
    (RESULTS / "ablation_A.json").write_text(json.dumps({"heldout_genes": genes, "rows": rows}, indent=2, default=str))
    _print(rows)
    return rows


def _fmt(r: dict) -> str:
    if r.get("auroc") is None:
        return f"{r['n']:>4} {r['n_benign']:>3}   {r.get('note', '')}"
    return (f"{r['n']:>4} {r['n_benign']:>3}  {r['auroc']:.3f} {str(r['auroc_ci']):>15}  "
            f"{r['auprc']:.3f} {str(r['auprc_ci']):>15}")


def _print(rows: list[dict]) -> None:
    print(f"{'setting':<14} {'target':<12} {'gene':<7} {'config':<9} {'method':<8} {'n':>4} {'B':>3}  "
          f"AUROC {'95% CI':>15}  AUPRC {'95% CI':>15}")
    for r in rows:
        print(f"{r['setting']:<14} {r['split']:<12} {r.get('gene') or '-':<7} {r['config']:<9} {r['method']:<8} {_fmt(r)}")


if __name__ == "__main__":
    main()
