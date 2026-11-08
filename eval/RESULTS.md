# Output A

**These numbers predate the canonical-position fix and need a rerun.** A local rebuild was stopped partway; mixed checkpoints were discarded. Do not treat them as a result.

On a GPU box, from the repo root:

```bash
bash scripts/retrain_all.sh
```

That rebuilds the graph on `feature_table.csv` `pos`, retrains headline + ablation + held-out genes, and writes `eval/results/`. Then replace the tables below, especially anything involving **CACNA1C**.

## What the rerun must pick up

Residue numbering for paralog and 3D edges now comes from `feature_table.csv` `pos` (Omar remapped 822 variants). Rows with `numbering == "unmappable"` are not placed. Join key is `variation_id`, never `variant_id`.

Compared with the old `uniprot_pos` / ClinVar `position` fallback:

| gene | variants whose residue changed | of |
|---|---:|---:|
| CACNA1C | 471 | 800 |
| OPA1 | 4 |  |
| RPGR | 2 |  |
| PAX6 | 1 |  |

CACNA1C headline-test variants (`split=test`, n=7: 1 P / 6 B) went from **0 placed** to **7 placed**. Same-gene 3D edges (`≤10 Å`, pLDDT ≥ 70) went from 522 to 1024; paralog links from 61 to 117.

2023-split training and eval still use only `*_2023` columns, `split`, AlphaMissense, pLDDT, UniProt features, and `paralog_pathogenic_2023.csv`. They do not read `predictions.json` `harmful` / `clinvar_today` / `evidence` / `computational_evidence`, `cls_now`, or `diseases_now`.

## Headline (pre-fix, n=144, 124 P / 20 B)

GNN Output A here did not use paralog or 3D edges. The 0.910 figure is still from **stale CACNA1C placement**. After the rerun, check CACNA1C’s 7 test variants separately (both classes present).

| method | AUROC | 95% CI | AUPRC |
|---|---:|---|---:|
| team 2023 model | 0.981 | [0.959, 0.997] | 0.997 |
| XGBoost + `am_missing` | 0.957 | [0.918, 0.983] | 0.993 |
| GNN + `am_missing` | 0.910 | [0.853, 0.958] | 0.985 |
| gene prior | 0.774 | [0.685, 0.852] | 0.962 |
| AlphaMissense (imputed) | 0.742 | [0.599, 0.872] | 0.946 |

AlphaMissense is missing for 10 headline variants, 8 of them benign.

## Ablation (pre-fix)

| config | GNN AUROC | XGBoost AUROC |
|---|---:|---:|
| base | 0.910 | 0.957 |
| +paralog | 0.939 | 0.955 |
| +3D | 0.900 | 0.961 |
| +both | 0.889 | 0.963 |

## Held-out genes (pre-fix; ABCA4, GUCY2D, MYH7, RPE65, USH2A)

Test AUROC is undefined for every held-out gene except GUCY2D (all pathogenic except GUCY2D’s 5 benign). The table is GNN AUROC on that gene’s 2023-labeled variants (never used as labels), pooled n=1077.

| config | GNN AUROC | XGBoost AUROC |
|---|---:|---:|
| base | 0.920 | 0.931 |
| +paralog | 0.836 | 0.930 |
| +3D | 0.869 | 0.933 |
| +both | 0.775 | 0.932 |

## Claim (until the rerun)

GNN Output A matched a strong tabular baseline and did not beat it, **on a graph that left most CACNA1C variants unplaced**. That is the only claim these numbers support.
