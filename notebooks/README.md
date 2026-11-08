# Notebooks

`omar_clinvar_vus.ipynb` is the main pipeline, top to bottom:

1. ClinVar January 2023 and current releases, filtered to the 23-gene panel
2. The time-machine split: train on 2023 labels, test on what experts decided later
3. Disease groups, AlphaMissense, gnomAD benign negatives, nearby known variants, related genes
4. The transparent model, layer ablations, and the ranked list of today's uncertain variants
5. Disease prediction from the nearest known variants
6. FoldX on RHO, RPE65 and GLA, and AlphaFold 3D neighbors
7. ACMG PP3/BP4 calibration, one-page evidence reports, REVEL and CADD benchmarks
8. 1000 Genomes healthy carriers and MGI mouse phenotypes

It ran in Google Colab and downloads its own public data.
