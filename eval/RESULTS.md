# Output A after the canonical-position rerun

Reran on `geneiusgpu` (L40S, CUDA) with `scripts/retrain_all.sh`. Residue placement uses `feature_table.csv` `pos`. Rows with `numbering == "unmappable"` are not placed. Join key is `variation_id`.

**Before the position fix** the graph used `uniprot_pos` when it matched the sequence, otherwise ClinVar `position`. That left 478 variants unplaced or on the wrong residue, 471 of them CACNA1C (of 800). All 7 CACNA1C headline-test variants were unplaced. Those pre-fix numbers are in parentheses below.

2023-split training and eval still use only `*_2023` columns, `split`, AlphaMissense, pLDDT, UniProt features, and `paralog_pathogenic_2023.csv`. They do not read `predictions.json` `harmful` / `clinvar_today` / `evidence` / `computational_evidence`, `cls_now`, or `diseases_now`.

## Headline (n=144, 124 P / 20 B)

| method | AUROC | 95% CI | before |
|---|---:|---|---:|
| team 2023 model | 0.981 | [0.959, 0.997] | 0.981 |
| XGBoost + `am_missing` | 0.953 | [0.913, 0.981] | 0.957 |
| GNN + `am_missing` | 0.931 | [0.878, 0.974] | 0.910 |
| GNN, AM imputed | 0.797 | [0.724, 0.866] | 0.827 |
| GNN, no AlphaMissense | 0.800 | [0.708, 0.886] | 0.783 |
| gene prior | 0.774 | [0.685, 0.852] | 0.774 |
| AlphaMissense (imputed) | 0.742 | [0.599, 0.872] | 0.742 |

AlphaMissense is missing for 10 headline variants, 8 of them benign. The GNN move from 0.910 to 0.931 sits inside the old confidence interval. XGBoost is still ahead.

## 1-star test (n=181, 147 P / 34 B)

| method | AUROC | 95% CI | before |
|---|---:|---|---:|
| team 2023 model | 0.986 | [0.968, 0.998] | 0.988 |
| XGBoost + `am_missing` | 0.982 | [0.959, 0.999] | 0.983 |
| GNN + `am_missing` | 0.975 | [0.935, 0.998] | 0.973 |
| GNN, AM imputed | 0.904 | [0.851, 0.945] | 0.931 |
| GNN, no AlphaMissense | 0.835 | [0.749, 0.912] | 0.820 |

## Ablation on the headline 144

| config | GNN AUROC (before) | XGBoost AUROC (before) |
|---|---|---|
| base | 0.931 (0.910) | 0.953 (0.957) |
| +paralog | 0.940 (0.939) | 0.956 (0.955) |
| +3D | 0.895 (0.900) | 0.961 (0.961) |
| +both | 0.909 (0.889) | 0.961 (0.963) |

## Held-out genes (ABCA4, GUCY2D, MYH7, RPE65, USH2A)

Test AUROC is undefined for every held-out gene except GUCY2D. Pooled 2023-labeled variants of those genes (n=1077, never used as labels):

| config | GNN AUROC (before) | XGBoost (before) |
|---|---|---|
| base | 0.916 (0.920) | 0.929 (0.931) |
| +paralog | 0.837 (0.836) | 0.932 (0.930) |
| +3D | 0.729 (0.869) | 0.930 (0.933) |
| +both | 0.797 (0.775) | 0.933 (0.932) |

The +3D GNN drop (0.869 → 0.729) is the one change whose confidence intervals do not overlap. Those five genes were not remapped. The new 3D edges are inside CACNA1C, so this is the shared GNN weights refit on that graph, not a direct score of CACNA1C. XGBoost, which sees only that gene's own neighbor counts, stayed flat.

## CACNA1C

Headline test: 7 variants, 1 pathogenic (ClinVar 67554) and 6 benign. All seven are now on canonical residues. AUROC is 1.0 for the headline GNN and for base, +paralog, +3D, and +both. On +3D the pathogenic score is 0.44 and the benign mean is 0.003. Before the fix those seven variants had no 3D neighbors and the pathogenic +3D score was 0.31, with the same rank order.

## Claim

GNN Output A still matches a strong tabular baseline and still does not beat it. Canonical CACNA1C placement keeps that gene's seven test variants in the right order. It does not make the 3D edges win on the headline set or on the held-out genes.
