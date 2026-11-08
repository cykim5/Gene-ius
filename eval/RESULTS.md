# Output A after canonical residue placement

Residue numbering for paralog and 3D edges now comes from `feature_table.csv` `pos` (Omar remapped 822 variants, mostly CACNA1C). Rows with `numbering == "unmappable"` are not placed. 2023-split training and eval still use only `*_2023` columns, `split`, AlphaMissense, pLDDT, UniProt features, and `paralog_pathogenic_2023.csv`. They do not read `predictions.json` `harmful` / `clinvar_today` / `evidence` / `computational_evidence`, `cls_now`, or `diseases_now`.

## What was stale

The previous graph placed residues with `uniprot_pos` when that amino acid matched the sequence, else ClinVar `position`. That left **478 variants** on the wrong residue or unplaced:

| gene | changed | of |
|---|---:|---:|
| CACNA1C | 471 | 800 |
| OPA1 | 4 |  |
| RPGR | 2 |  |
| PAX6 | 1 |  |

CACNA1C headline-test variants (`split=test`, n=7: 1 P / 6 B) went from **0 placed** to **7 placed**. Same-gene 3D edges (`≤10 Å`, pLDDT ≥ 70) went from 522 to 1024; paralog links from 61 to 117.

## Headline (n=144, 124 P / 20 B)

GNN Output A does not use paralog or 3D edges, so it should not move. It did not.

| method | AUROC | 95% CI | AUPRC |
|---|---:|---|---:|
| team 2023 model | 0.981 | [0.959, 0.997] | 0.997 |
| XGBoost + `am_missing` | 0.957 | [0.918, 0.983] | 0.993 |
| GNN + `am_missing` | 0.910 | [0.853, 0.958] | 0.985 |
| gene prior | 0.774 | [0.685, 0.852] | 0.962 |
| AlphaMissense (imputed) | 0.742 | [0.599, 0.872] | 0.946 |

AlphaMissense is missing for 10 headline variants, 8 of them benign.

## Ablation on the same 144 (GNN vs XGBoost)

| config | GNN AUROC (was) | XGBoost AUROC (was) |
|---|---|---|
| base | 0.910 (0.910) | 0.957 (0.957) |
| +paralog | 0.940 (0.939) | 0.957 (0.955) |
| +3D | 0.903 (0.900) | 0.962 (0.961) |
| +both | 0.889 (0.889) | 0.961 (0.963) |

Headline movement is inside the bootstrap noise. XGBoost still matches or beats the GNN with the same features. Adding both paralog and 3D still hurts the GNN.

## Held-out genes (ABCA4, GUCY2D, MYH7, RPE65, USH2A)

Test AUROC is undefined for every held-out gene except GUCY2D (all pathogenic except GUCY2D's 5 benign). Scoring the same models on that gene's 2023-labeled variants (never used as labels):

| config | GNN AUROC pooled n=1077 (was) | XGBoost (was) |
|---|---|---|
| base | 0.920 (0.920) | 0.931 (0.931) |
| +paralog | 0.831 (0.836) | 0.931 (0.930) |
| +3D | 0.917 (0.869) | 0.934 (0.933) |
| +both | 0.802 (0.775) | 0.933 (0.932) |

Held-out genes were not remapped. The +3D GNN bump (0.869 → 0.917, CIs barely disjoint) is from retraining those runs on the rebuilt graph, not from CACNA1C test variants. Do not read it as “3D edges now win on unseen genes.” XGBoost is flat.

## CACNA1C

All 7 headline-test variants are now on canonical residues and 6 of them have a 3D neighbor. Rank order is unchanged: the one pathogenic (ClinVar 67554) still scores above the six benign (AUROC 1.0, n=7). Pathogenic +3D score 0.31 vs benign mean ~0.001. Small n, both classes present, not a headline claim.

## Claim

GNN Output A still matches a strong tabular baseline and still does not beat it. Canonical CACNA1C placement fixes the graph for that gene; it does not change the headline ranking result.
