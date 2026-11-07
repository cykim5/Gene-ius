"""Headline test (Stage 8, tests 1 and 2).

Score A, B, C on 2023 VUS that experts reclassified since (2+ stars or expert panel).
Run twice: full model, and the model without AlphaMissense.

Metrics: A AUROC and AUPRC, B macro-F1 vs LoGoFunc, C top-1 and top-3 among the gene's diseases.

Done: A on split=test (headline, 2+ stars) and split=test_1star (secondary), comparing the GNN,
XGBoost on the same variant features, and a per-gene prior (fraction pathogenic in the gene's
train labels). AlphaMissense joins as a baseline when its data lands.

This is the only place class_now is read. It's the answer key, never an input.

    python -m geneius.models.gnn       # train first
    python eval/reclassification.py    # writes eval/results/reclassification_A.json and .csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Don't import torch here: xgboost and torch load separate OpenMP runtimes, which crashes on macOS.
from geneius import config  # noqa: E402
from geneius.data import load_variants  # noqa: E402
from geneius.features.variant_features import model_columns, variant_features  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
TEST_SPLITS = {"test": "headline: 2023 VUS reclassified to P/B with 2+ stars",
               "test_1star": "secondary: 2023 VUS reclassified to P/B with 1 star"}
N_BOOT = 1000
SEED = 0


def xgboost_scores(train: pd.DataFrame, score: pd.DataFrame, cols: list[str]) -> np.ndarray:
    import xgboost as xgb

    y = (train["class_2023"] == "P").astype(int)
    model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                              colsample_bytree=0.9, random_state=SEED, eval_metric="aucpr",
                              scale_pos_weight=float((y == 0).sum()) / max(int(y.sum()), 1))
    model.fit(train[cols], y)
    return model.predict_proba(score[cols])[:, 1]


def gene_prior(train: pd.DataFrame, score: pd.DataFrame) -> np.ndarray:
    rate = (train["class_2023"] == "P").groupby(train["gene"]).mean()
    overall = float((train["class_2023"] == "P").mean())
    return score["gene"].map(rate).fillna(overall).to_numpy(dtype=float)


def metric_with_ci(y: np.ndarray, p: np.ndarray, rng: np.random.Generator) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    out = {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}
    boots = {"auroc": [], "auprc": []}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boots["auroc"].append(roc_auc_score(y[i], p[i]))
            boots["auprc"].append(average_precision_score(y[i], p[i]))
    for k, vals in boots.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out[f"{k}_ci"] = [round(float(lo), 3), round(float(hi), 3)]
        out[k] = round(out[k], 3)
    return out


def main() -> dict:
    variants = load_variants(with_test_labels=True)
    feats = variant_features(variants)
    cols = model_columns(feats)
    table = pd.concat([variants[["variant_id", "gene", "class_2023", "class_now", "split"]], feats[cols]], axis=1)
    table["variant_id"] = table["variant_id"].astype(str)
    train = table[table["split"] == "train"]

    preds = pd.read_csv(config.gnn_a_dir() / "predictions.csv", dtype={"variant_id": str})
    gnn = dict(zip(preds["variant_id"], preds["harmful"]))

    rng = np.random.default_rng(SEED)
    results, rows = {"features": cols, "splits": {}}, []
    for split, description in TEST_SPLITS.items():
        test = table[(table["split"] == split) & table["class_now"].isin(["P", "B"])].copy()
        y = (test["class_now"] == "P").to_numpy(dtype=int)
        scores = {
            "gnn": test["variant_id"].map(gnn).to_numpy(dtype=float),
            "xgboost": xgboost_scores(train, test, cols),
            "gene_prior": gene_prior(train, test),
        }
        split_out = {"description": description, "n": int(len(y)), "n_pathogenic": int(y.sum()),
                     "n_benign": int(len(y) - y.sum()), "random_auprc": round(float(y.mean()), 3),
                     "per_gene_n": test.groupby("gene").size().to_dict(), "methods": {}}
        for method, p in scores.items():
            m = metric_with_ci(y, p, rng)
            split_out["methods"][method] = m
            rows.append({"split": split, "method": method, "n": len(y), "n_benign": int(len(y) - y.sum()), **m})
        results["splits"][split] = split_out

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "reclassification_A.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(rows).to_csv(RESULTS / "reclassification_A.csv", index=False)

    for split, s in results["splits"].items():
        print(f"\n{split} ({s['description']}): n={s['n']} ({s['n_pathogenic']} P, {s['n_benign']} B), "
              f"random AUPRC {s['random_auprc']}")
        print(f"{'method':<11} {'AUROC':>6} {'95% CI':>15} {'AUPRC':>6} {'95% CI':>15}")
        for method, m in s["methods"].items():
            print(f"{method:<11} {m['auroc']:>6.3f} {str(m['auroc_ci']):>15} {m['auprc']:>6.3f} {str(m['auprc_ci']):>15}")
        print("per-gene n: " + ", ".join(f"{g} {n}" for g, n in sorted(s["per_gene_n"].items(), key=lambda kv: -kv[1])))
    return results


if __name__ == "__main__":
    main()
