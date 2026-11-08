# Mechanism-Aware VUS Interpretation Agent

Nucleate Pittsburgh BioHack 2026, BioTech track, NVIDIA problem statement 2: "How can we prototype agentic frameworks that leverage GNNs to parse phenotype-genotype knowledge graphs?"

When a patient gets genetic testing, labs often find a variant they can't classify as harmful or harmless. These are variants of uncertain significance (VUS). In many genes, different mutations cause different diseases, so a yes/no harmful score isn't enough. What matters is how the mutation breaks the protein, because that decides which disease follows and which treatment could work.

For each variant the plan is four linked outputs. **Output A is trained and scored. Outputs B, C, and D are specified and unimplemented.**

| | Question | Status |
|---|---|---|
| A | Is it harmful or benign? | Trained and scored. See `eval/RESULTS.md`. |
| B | How does it break the protein? (gain, loss, dominant-negative, misfolding) | Unimplemented. Demo mechanism text comes from FoldX on the evidence report, for the variants that have it. |
| C | Which disease and symptoms follow? Eye-only or syndromic? | Unimplemented as a GNN head. The agent reads disease groups from the evidence report and symptom paths from PrimeKG. |
| D | Which treatment strategy fits? (augmentation, silencing, chaperone) | Unimplemented. `suggest_treatment` is a gene/mechanism rule table. |

An LLM agent (Nemotron via Nebius) calls graph and evidence tools, explains with the paths those tools return, says when it's unsure, and can suggest an assay. Symptom-to-gene ranking for a patient is `patient_rank`: the current-model harmful score times a phenotype-overlap score against that gene's PrimeKG diseases.

We test output A by training only on 2023 labels and scoring VUS that experts reclassified afterward.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add your API key
bash scripts/download_data.sh # PrimeKG, HPO, ClinVar, AlphaMissense, MGI
python -m pytest -q
```

```bash
python -m geneius variant RHO p.Pro23His
python -m geneius ask "What does RHO P23H do?"
python -m geneius symptoms "night blindness and retinitis pigmentosa"
```

Tools that aren't built yet return fake outputs with the right shape (marked `"stub": true`), so the agent and app can be built in parallel.

The data is several GB. If this folder syncs to OneDrive or Dropbox, set `GENEIUS_DATA_DIR` in `.env` to a folder outside it.

## Repo layout

Folders follow the stages in the plan.

| Folder | Stage | What's in it |
|---|---|---|
| `geneius/config.py` | 0 | paths and the 2023 `CUTOFF` constant |
| `geneius/data/` | 1, 2 | PrimeKG and HPO loaders, ClinVar parsing, UniProt mapping, gene panels |
| `geneius/features/` | 3 | AlphaMissense, ESM-2, AlphaFold, UniProt region features |
| `geneius/graph/` | 0, 4 | graph build, 3D and related-protein edges, leakage rules |
| `geneius/labels/` | 5 | label definitions and cleaning |
| `geneius/models/` | 6 | GNN, baselines, calibration |
| `geneius/agent/` | 7 | agent loop and its 16 tools |
| `eval/` | 8, 9 | go/no-go, reclassification test, time-saved curve, today's ranked list |
| `app/` | 10 | web app and 3D viewer |
| `docs/` | 11 | pitch, judge questions |
| `notebooks/` | | scratch work, prefix files with your name |
| `tests/` | | runs on a tiny fixture graph, no download needed |

## What works today

| Piece | Status |
|---|---|
| PrimeKG and HPO 2023 loaders | works |
| Variant table, 2023 vs current ClinVar, leakage splits | works (`data/processed/variants.csv`, 15,905 missense rows) |
| Graph build, output A GNN, XGBoost and AlphaMissense baselines | trained; numbers in `eval/RESULTS.md` |
| `gnn_predict` | output A only, from `data/gnn_a/predictions.csv` |
| `map_symptoms` | dictionary matcher on HPO names and synonyms. Embedding matcher is not built. Negation is ignored. |
| `find_entity`, `expand`, `find_paths` | works on PrimeKG |
| `evidence_report`, `patient_rank` | works from `predictions.json` and phenotype overlap |
| `population_evidence` | 1000 Genomes phased panel. A miss means "not seen in that panel." |
| `stability` | FoldX for RHO and RPE65 demo variants |
| `mouse_evidence` | MGI, gene-level |
| `show_structure` | the 7 demo variants |
| `suggest_treatment`, `suggest_assay` | rule tables |
| Agent loop | works with `NEBIUS_API_KEY` (`python -m geneius ask ...`) |
| `gnn_score`, `structural_neighbors`, `evidence_by_source` | stubs (`"stub": true`) |
| Web app | static site on branch `claire/app`, live at https://gene-ius-app.vercel.app. Not merged here. |

## Numbers to cite

Use these. They are the variant-table splits and `eval/RESULTS.md` after the canonical-position rerun.

| Quantity | Number |
|---|---|
| Headline test: 2023 VUS later P/B at 2+ stars | **144** (124 pathogenic, 20 benign) |
| Secondary test: same, at exactly 1 star | **181** (147 pathogenic, 34 benign) |
| 1-star or higher, both tests | **325** |
| 2023 labeled training variants (P or B) | **2,790** |
| Unlabeled rows (still no P/B train or test label) | **10,685** |
| Headline AUROC | team 2023 model **0.981**, XGBoost **0.953**, GNN **0.931**, GNN without AlphaMissense **0.800**, AlphaMissense imputed **0.742** |

The GNN matches the tabular baseline and does not beat it. 3D edges do not win on the headline set or the held-out genes. Full table, confidence intervals, and the CACNA1C placement check are in `eval/RESULTS.md`.

## Demo

Patient with nystagmus, visual impairment, and an abnormal electroretinogram (`HP:0000639`, `HP:0000505`, `HP:0000512`) and candidates `RPE65:H68P`, `ABCA4:R2040P`, `MYH7:R403P`, `KCNQ1:R591P`. `patient_rank` orders them RPE65, ABCA4, MYH7, KCNQ1. Single-variant agent demo: `RHO:C187S` (current-model harmful 1.0 and GNN output A 1.0).

```bash
python -m geneius ask "Interpret RHO:C187S. Report the current-model harmful score and the GNN score."
python -m geneius symptoms "nystagmus and visual impairment"
```

## Build order

Original plan. Current status is the table above.

1. ClinVar 2023 vs current comparison and VUS counts (go/no-go)
2. UniProt mapping, residue check, AlphaMissense, UniProt region features
3. AlphaFold features (pLDDT, burial, distances)
4. Graph with PrimeKG and variants, XGBoost baselines, GNN output A, time-saved curve (first complete version)
5. Agent with core tools, 3D viewer, deployment skeleton
6. Outputs B and C, mechanism and disease tests
7. 3D-neighbor and related-protein edges, mouse layer, ESM-2
8. Output D and the Fabry case study, evidence reports, calibration, ArgosX scan
9. Cardiac panel results, agent benchmark, remaining tests

## Leakage rules

1. Train only on releases from before the cutoff: ClinVar, HPO, UniProt humsavar.
2. When a gene-disease link is held out, also remove symptom edges derived from that disease.
3. Labels on 3D and related-protein edges come from training data only.
4. Mechanism labels from gene-disease rules are for training only.
5. VUS stay in the graph as unlabeled nodes, with their VUS-to-disease edges removed.
6. Test only on reclassifications with 2+ stars or from an expert panel.
7. Always also run the model without AlphaMissense.
8. Confirm AlphaMissense, ESM-2, and AlphaFold weren't trained on ClinVar labels.
9. Tune only on a validation split from 2023 data, never on the test set.

## How we work

Pull before you start, work on your own branch, and open a pull request when it works.

```bash
git pull origin main
git checkout -b yourname/what-you-are-doing
```

```bash
python -m pytest -q
git add <your files>
git commit -m "Short description"
git push -u origin yourname/what-you-are-doing
```

Rules:

- Don't change a tool's inputs or outputs without telling the team.
- Run the tests before you push.
- Never commit `.env`, API keys, or anything in `data/`. Public data only, no patient data.
- Feature freeze Saturday 11pm. After that, fixes only.

Research use only. Supports, but does not replace, expert variant classification.
