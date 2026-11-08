# Gene-ius: Mechanism-Aware VUS Interpretation Agent

Nucleate Pittsburgh BioHack 2026, BioTech track, NVIDIA problem statement 2: "How can we prototype agentic frameworks that leverage GNNs to parse phenotype-genotype knowledge graphs?"

**Live demo:** https://gene-ius-app.vercel.app

## The problem

When someone gets genetic testing, the report often comes back with a variant of uncertain significance (VUS): a change in a gene that nobody can yet call harmful or harmless. About a third of the variants in ClinVar carry this label. A VUS gives the patient no diagnosis, no trial eligibility, and nothing to test their family for, and labs can't run experiments on all of them.

Knowing a variant is harmful often isn't enough either. In the same gene, different mutations cause different diseases and need different treatments. In RHO, a misfolding mutation causes progressive blindness, while one that switches the protein on causes night blindness that doesn't progress.

## What it does

For each missense VUS in 23 retina and heart genes, Gene-ius tells a lab which variants to test first and answers four questions, with the evidence for each:

| | Question | How |
|---|---|---|
| A | Is it harmful? | A transparent model over evidence layers, calibrated to ACMG PP3/BP4 strength |
| B | How does it break the protein? | FoldX stability physics on AlphaFold models (RHO, RPE65, GLA), plus ESMFold on RHO |
| C | Which disease follows? | The nearest known disease variants in the same gene and in related genes |
| D | What strategy fits? | A gene and mechanism rule table, stated as a hypothesis to test |

An agent (NVIDIA Nemotron on Nebius) explains each result using only tools that return real evidence: knowledge graph paths from PrimeKG, nearby known variants, related-gene residues, FoldX, 1000 Genomes carriers, and MGI mouse phenotypes. It says when it's unsure and names the experiment that would settle the question.

## How we tested it: the time machine

The model only sees data from January 2023. We then grade it on variants that were uncertain then and that experts resolved later. A variant is never used as evidence for itself.

| Claim | Test | Result |
|---|---|---|
| Finds real disease variants first | Top 500 of 10,173 variants uncertain in 2023 | 38 later proven pathogenic vs 18 for AlphaMissense, about 6 by chance |
| Beats the tools labs use | REVEL, CADD, AlphaMissense on the same variants | AUROC 0.953 vs 0.893 for REVEL |
| Related genes fill gaps | Cross-validation on 2023 channel variants | CACNA1C accuracy 0.76 to 0.92 |
| Predicts the right disease | 48 test variants with one disease | 81% vs 65% for the gene's most common disease |
| Physics agrees, without labels | FoldX on labeled RHO variants | Median +4.9 kcal/mol for disease vs +0.3 for benign |
| Does a GNN do better? | Same 144 test variants | No: GNN 0.931 vs XGBoost 0.953 vs our model 0.981 |

We report what didn't work too: 3D structure and UniProt regions didn't improve the ranking, so the model leaves them out, and graph edges hurt the GNN on genes it hadn't seen. Details and confidence intervals are in `notebooks/omar_clinvar_vus.ipynb` and `eval/RESULTS.md`.

## Where things are

| Path | What's in it |
|---|---|
| `notebooks/omar_clinvar_vus.ipynb` | The main pipeline: ClinVar 2023 vs today, features, the transparent model, the time-machine test, disease prediction, related genes, FoldX, ACMG calibration, REVEL/CADD benchmarks, 1000 Genomes |
| `geneius/agent/` | The agent loop and its tools |
| `geneius/graph/`, `geneius/models/` | Variant graph, GNN, and baselines |
| `geneius/data/`, `geneius/features/` | PrimeKG, HPO, and ClinVar loaders, gene panels, variant features, gnomAD frequencies |
| `eval/` | GNN evaluation, ablations, held-out genes (`eval/RESULTS.md`) |
| `app/` | The website: variant lookup, A to D, 3D structure, ranked list, agent |
| `data/processed/` | The variant table (15,905 missense variants, 2023 and current labels) |
| `tests/` | Tests on small fixtures, no download needed |

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # add NEBIUS_API_KEY for the agent
bash scripts/download_data.sh # PrimeKG, HPO, ClinVar, AlphaMissense, MGI
python -m pytest -q
```

```bash
python -m geneius ask "Interpret RHO:C187S."
python -m geneius symptoms "nystagmus and visual impairment"
python app/dev_server.py      # the website at http://localhost:8000
```

The data is several GB. Set `GENEIUS_DATA_DIR` in `.env` to keep it outside synced folders.

## Leakage rules

1. Train only on releases from before the January 2023 cutoff (ClinVar, HPO, UniProt).
2. Nearby-variant and related-gene evidence uses only what was known in January 2023, never the variant itself.
3. When a gene-disease link is held out, its symptom edges are removed too.
4. Test only on reclassifications with review stars or from an expert panel.
5. Always also report results without AlphaMissense.
6. Tune only on 2023 data, never on the test set.

## Limits

- Missense variants only. Splicing and truncating variants aren't covered.
- FoldX ran on three proteins and ESMFold on one.
- 62% of uncertain variants have no known disease variant within 5 residues, and those stay hard.
- Validated only on variants experts could already resolve. Performance on a live backlog will be lower.
- Research and decision support only. Under ACMG/AMP rules a computational prediction is supporting evidence, not a classification. Treatment strategies are hypotheses, not recommendations.

## Data sources

All public, no patient data: ClinVar (NCBI), PrimeKG (Harvard), HPO, MONDO, UniProt, gnomAD, 1000 Genomes, MGI, REVEL, and CADD. AlphaFold structures are from the AlphaFold Protein Structure Database (CC BY 4.0). AlphaMissense scores are from Google DeepMind (CC BY-NC-SA 4.0, non-commercial use).

## Team

Omar Baltagi, Claire Kim, Shank Sri Shasanka Madduri, Yihyun (Ellie) Kim, Nathaniel Girmay
