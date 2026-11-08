# Evaluation

| File | What it does |
|---|---|
| `RESULTS.md` | GNN results: headline test, 1-star test, ablations, held-out genes, CACNA1C check |
| `reclassification.py` | Output A on 2023 VUS that experts reclassified later, GNN vs XGBoost vs AlphaMissense |
| `ablation.py` | One layer at a time (paralog and 3D edges) and the held-out-gene test |
| `time_saved.py` | How many real disease variants land in the top k when a lab tests in ranked order |

The main model's time-machine test, REVEL and CADD benchmarks, disease prediction, and ACMG calibration are in `notebooks/omar_clinvar_vus.ipynb`.

Every number comes with its n and a bootstrap confidence interval, and weak results stay next to the baselines.
