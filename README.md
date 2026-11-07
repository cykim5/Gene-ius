# GENEIUS

Rare disease diagnosis copilot. Built at Nucleate Pittsburgh BioHack 2026, BioTech track #2 (NVIDIA).

Rare disease patients often wait years for a diagnosis. Clinicians have the symptoms but can't link them to one of thousands of rare genetic disorders. GENEIUS takes symptoms in plain English, ranks likely genes using a GNN over the PrimeKG knowledge graph, and explains each candidate with the graph evidence behind it.

## How it works

```
symptoms (free text)
  > map to HPO terms
  > rank candidate genes on PrimeKG (GNN, or the overlap baseline)
  > pull evidence paths for the top candidates
  > if candidates are close: suggest a follow-up question, check mechanisms
  > ranked answer with evidence
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add your API key
bash scripts/download_data.sh # PrimeKG + HPO
python -m geneius "4 year old with seizures, low muscle tone and speech delay"
```

Use `--agent` to run the LLM agent instead of the fixed pipeline, and `--json` for raw output.

No data yet? Everything still runs. The tools return fake outputs with the right shape until the data is downloaded, so the agent and demo can be built in parallel.

PrimeKG is large. If this folder syncs to OneDrive or Dropbox, set `GENEIUS_DATA_DIR` in `.env` to a folder outside it.

## Tools

Everyone builds against these. All return JSON. HPO IDs look like `HP:0001250`, genes are HGNC symbols like `SCN1A`.

| Tool | Returns | Status |
|---|---|---|
| `map_to_hpo(text)` | `[{hpo_id, label, matched_text}]` | Dictionary matcher works. Embedding matcher to do. |
| `rank_candidates(hpo_ids, top_k=10)` | `[{gene, disease, score}]` | Overlap baseline works. GNN to do. |
| `get_evidence_paths(hpo_ids, gene, max_paths=5)` | `[{path: [{node_id, node_type, name}], relations}]` | Works |
| `get_mechanism(disease)` | `{disease, mechanism_summary, key_phenotypes, source}` | Stub. DisMech to do. |
| `suggest_followup(candidates, known_hpo_ids)` | `{question, hpo_id, distinguishes}` | Works |

## Repo layout

```
geneius/
  tools/             the five agent tools
  baseline_overlap.py  phenotype overlap ranker (first baseline)
  data.py            PrimeKG and HPO loaders
  pipeline.py        fixed pipeline, no LLM
  agent.py           LLM agent loop
  gnn/               GNN training and inference
eval/
  simulate_patients.py
  score.py           hits@1, hits@5, hits@10
  data/              real case reports go here
scripts/download_data.sh
tests/               runs on a tiny fixture graph, no download needed
```

## Evaluation

```bash
python eval/simulate_patients.py --n 200
python eval/score.py --method overlap
```

Simulated patients keep 60% of a disease's HPO terms and add 2 noise terms. Their diseases are held out: phenotype edges are removed from the GNN's training graph, and hidden from the overlap baseline too, so the comparison is fair.

Real cases go in `eval/data/real_cases.jsonl`, one per line, following `real_cases.template.jsonl`. Score them with `--mode transductive`.

Baselines: phenotype overlap, LLM only, GNN only, full agent.

## Known gaps

- `map_to_hpo` ignores negation, so "no seizures" still maps to Seizure.
- HPO derives most gene-phenotype links from disease annotations. `gnn/graph_build.py` drops these for held-out diseases. Keep that in mind when adding new edge types.
- This is decision support, not a diagnosis.

## Team

- Sri: agent loop, GNN, tools, demo app
- Comp bio: data loading, simulated patients, scoring
- PhD 1: real case reports, sanity checks on rankings
- PhD 2: mechanism lookup, follow-up logic, judge questions
- Pitch: problem story, slides, demo script

## Development

```bash
python -m pytest -q
```

Work on your own branch and open a pull request. Never commit `.env` or anything in `data/`.
