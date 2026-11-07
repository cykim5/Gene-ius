"""Headline test (Stage 8, tests 1 and 2).

Score A, B, C on 2023 VUS that experts reclassified since (2+ stars or expert panel).
Run twice: full model, and the model without AlphaMissense.

Metrics: A AUROC and AUPRC, B macro-F1 vs LoGoFunc, C top-1 and top-3 among the gene's diseases.

Done: A on split=test (headline, 2+ stars) and split=test_1star (secondary), comparing the GNN
and XGBoost on the same variant features (each also without AlphaMissense), AlphaMissense alone,
a per-gene prior (fraction pathogenic in the gene's train labels), and the teammates' 2023 model
(predictions.csv harmful_2023_model). Its `harmful` column is not used: that model was trained on
current ClinVar labels, including every test variant.

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
from geneius.features.variant_features import (  # noqa: E402
    NO_ALPHAMISSENSE,
    all_variant_features,
    model_columns,
    model_matrix,
)

RESULTS = Path(__file__).resolve().parent / "results"
TEST_SPLITS = {"test": "headline: 2023 VUS reclassified to P/B with 2+ stars",
               "test_1star": "secondary: 2023 VUS reclassified to P/B with 1 star"}
N_BOOT = 1000
SEED = 0


def xgboost_scores(x_train: pd.DataFrame, y_train: np.ndarray, x_score: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb

    model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                              colsample_bytree=0.9, random_state=SEED, eval_metric="aucpr",
                              scale_pos_weight=float((y_train == 0).sum()) / max(int(y_train.sum()), 1))
    model.fit(x_train, y_train)
    return model.predict_proba(x_score)[:, 1]


def gnn_scores(path: Path) -> dict | None:
    if not path.exists():
        return None
    preds = pd.read_csv(path, dtype={"variant_id": str})
    return dict(zip(preds["variant_id"], preds["harmful"]))


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
    variants["variant_id"] = variants["variant_id"].astype(str)
    feats = all_variant_features(variants)
    cols = model_columns(feats)
    cols_no_am = [c for c in cols if c in NO_ALPHAMISSENSE]
    is_train = (variants["split"] == "train").to_numpy()
    y_train = (variants.loc[is_train, "class_2023"] == "P").to_numpy(dtype=int)
    # Same matrices the GNN gets: (a) train-median fill plus am_missing, (b) train-median fill only.
    matrices = {
        "a": model_matrix(feats, cols, is_train, am_indicator=True),
        "b": model_matrix(feats, cols, is_train, am_indicator=False),
        "no_am": model_matrix(feats, cols_no_am, is_train, am_indicator=False),
    }
    gnn_dir = config.gnn_a_dir()
    gnn_runs = {
        "gnn (a) am_missing indicator": gnn_scores(gnn_dir / "predictions.csv"),
        "gnn (b) am imputed": gnn_scores(gnn_dir.with_name("gnn_a_am_impute") / "predictions.csv"),
        "gnn no AM": gnn_scores(gnn_dir.with_name("gnn_a_no_alphamissense") / "predictions.csv"),
    }
    team = None
    if config.baseline_predictions_csv().exists():
        tp = pd.read_csv(config.baseline_predictions_csv(), usecols=["variation_id", "harmful_2023_model"])
        team = dict(zip(tp["variation_id"].astype(str), tp["harmful_2023_model"]))
    am = feats["alphamissense"] if "alphamissense" in cols else None
    train_table = variants.loc[is_train, ["gene", "class_2023"]]

    rng = np.random.default_rng(SEED)
    results, rows = {"features": cols, "splits": {}}, []
    for split, description in TEST_SPLITS.items():
        mask = ((variants["split"] == split) & variants["class_now"].isin(["P", "B"])).to_numpy()
        test = variants[mask]
        y = (test["class_now"] == "P").to_numpy(dtype=int)
        scores = {}
        for name, preds in gnn_runs.items():
            if preds is not None:
                scores[name] = test["variant_id"].map(preds).to_numpy(dtype=float)
        scores["xgboost (a) am_missing indicator"] = xgboost_scores(matrices["a"][is_train], y_train, matrices["a"][mask])
        scores["xgboost (b) am imputed"] = xgboost_scores(matrices["b"][is_train], y_train, matrices["b"][mask])
        scores["xgboost no AM"] = xgboost_scores(matrices["no_am"][is_train], y_train, matrices["no_am"][mask])
        if am is not None:
            scores["alphamissense (imputed)"] = matrices["b"].loc[mask, "alphamissense"].to_numpy(dtype=float)
        scores["gene_prior"] = gene_prior(train_table, test)
        if team is not None:
            scores["team_2023_model"] = test["variant_id"].map(team).to_numpy(dtype=float)
        split_out = {"description": description, "n": int(len(y)), "n_pathogenic": int(y.sum()),
                     "n_benign": int(len(y) - y.sum()), "random_auprc": round(float(y.mean()), 3),
                     "per_gene_n": test.groupby("gene").size().to_dict(), "methods": {}}
        for method, p in scores.items():
            m = metric_with_ci(y, p, rng)
            split_out["methods"][method] = m
            rows.append({"split": split, "method": method, "n": len(y), "n_benign": int(len(y) - y.sum()), **m})
        if am is not None:
            scored = am[mask].notna().to_numpy()
            split_out["alphamissense_missing"] = {"n": int((~scored).sum()), "benign": int(((~scored) & (y == 0)).sum())}
            m = metric_with_ci(y[scored], am[mask].to_numpy(dtype=float)[scored], rng)
            m["n"] = int(scored.sum())
            split_out["methods"]["alphamissense (scored only)"] = m
            rows.append({"split": split, "method": "alphamissense (scored only)", "n": int(scored.sum()),
                         "n_benign": int((y[scored] == 0).sum()), **m})
        results["splits"][split] = split_out

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "reclassification_A.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(rows).to_csv(RESULTS / "reclassification_A.csv", index=False)

    for split, s in results["splits"].items():
        print(f"\n{split} ({s['description']}): n={s['n']} ({s['n_pathogenic']} P, {s['n_benign']} B), "
              f"random AUPRC {s['random_auprc']}")
        print(f"{'method':<34} {'n':>4} {'AUROC':>6} {'95% CI':>15} {'AUPRC':>6} {'95% CI':>15}")
        for method, m in s["methods"].items():
            print(f"{method:<34} {m.get('n', s['n']):>4} {m['auroc']:>6.3f} {str(m['auroc_ci']):>15} "
                  f"{m['auprc']:>6.3f} {str(m['auprc_ci']):>15}")
        if s.get("alphamissense_missing"):
            miss = s["alphamissense_missing"]
            print(f"AlphaMissense missing for {miss['n']} variants, {miss['benign']} of them benign")
        print("per-gene n: " + ", ".join(f"{g} {n}" for g, n in sorted(s["per_gene_n"].items(), key=lambda kv: -kv[1])))
    return results


if __name__ == "__main__":
    main()
