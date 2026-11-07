# Mechanism-Aware VUS Interpretation Agent

Nucleate Pittsburgh BioHack 2026, BioTech track, NVIDIA problem statement 2: "How can we prototype agentic frameworks that leverage GNNs to parse phenotype-genotype knowledge graphs?"

When a patient gets genetic testing, labs often find a variant they can't classify as harmful or harmless. These are variants of uncertain significance (VUS). In many genes, different mutations cause different diseases, so a yes/no harmful score isn't enough. What matters is how the mutation breaks the protein, because that decides which disease follows and which treatment could work.

For each variant, a GNN over a phenotype-genotype knowledge graph predicts four linked things:

| | Question |
|---|---|
| A | Is it harmful or benign? |
| B | How does it break the protein? (gain, loss, dominant-negative, misfolding) |
| C | Which disease and symptoms follow? Eye-only or syndromic? |
| D | Which treatment strategy fits? (augmentation, silencing, chaperone) |

An LLM agent uses the GNN to search the graph, explains every prediction with real evidence, says when it's unsure, and suggests the experiment that would settle it.

We test it by training only on 2023 data and scoring it on VUS that experts reclassified afterward.

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
| `geneius/agent/` | 7 | agent loop and its 12 tools |
| `eval/` | 8, 9 | go/no-go, reclassification test, time-saved curve, today's ranked list |
| `app/` | 10 | web app and 3D viewer |
| `docs/` | 11 | pitch, judge questions |
| `notebooks/` | | scratch work, prefix files with your name |
| `tests/` | | runs on a tiny fixture graph, no download needed |

## What works today

| Piece | Status |
|---|---|
| PrimeKG and HPO loaders | works |
| ClinVar protein change and review star parsing | works |
| Leakage removal for held-out diseases | works |
| `map_symptoms`, `find_entity`, `find_paths` | works |
| `suggest_treatment`, `suggest_assay` | works (rule tables from the plan) |
| Phenotype overlap baseline | works |
| Time-saved curve math | works |
| Agent loop | written, not run yet (needs API key) |
| Go/no-go count | to do, do this first |
| Features, graph build, GNN, baselines, calibration | to do |
| `expand`, `structural_neighbors`, `gnn_predict`, `gnn_score`, `evidence_by_source`, `evidence_report`, `show_structure` | stubs |
| Web app | to do |

## Build order

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
