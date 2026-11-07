#!/usr/bin/env bash
# Downloads the Stage 1 data into $GENEIUS_DATA_DIR (default: ./data).
# Safe to re-run: skips files that already exist.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi
DIR="${GENEIUS_DATA_DIR:-$REPO/data}"
CUTOFF="2023-01"   # keep in sync with CUTOFF in geneius/config.py
mkdir -p "$DIR"
cd "$DIR"

dl() {
  if [ -s "$2" ]; then
    echo "already have $2"
  else
    echo "downloading $2 ..."
    curl -fL --retry 3 -o "$2.part" "$1"
    mv "$2.part" "$2"
  fi
}

HPO_REL="https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2023-01-27"
CLINVAR="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited"

# Knowledge graph (Harvard Dataverse, doi:10.7910/DVN/IXA7BM)
dl https://dataverse.harvard.edu/api/access/datafile/6180620 kg.csv

# HPO: 2023 snapshot for training, current for the time-based test
dl "$HPO_REL/hp.obo" "hp_${CUTOFF}.obo"
dl "$HPO_REL/phenotype.hpoa" "phenotype_${CUTOFF}.hpoa"
dl https://purl.obolibrary.org/obo/hp.obo hp_current.obo
dl https://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa phenotype_current.hpoa

# ClinVar: cutoff release (train) and current release (test)
dl "$CLINVAR/archive/2023/variant_summary_${CUTOFF}.txt.gz" "variant_summary_${CUTOFF}.txt.gz"
dl "$CLINVAR/variant_summary.txt.gz" variant_summary_current.txt.gz

# AlphaMissense precomputed scores (large)
dl https://storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz AlphaMissense_aa_substitutions.tsv.gz

# MGI: human gene -> mouse phenotypes
dl https://www.informatics.jax.org/downloads/reports/HMD_HumanPhenotype.rpt HMD_HumanPhenotype.rpt

# Not scripted yet: UniProt (ID mapping, features, 2023 humsavar), AlphaFold structures,
# LoGoFunc labels, RHO mechanism classes, migalastat list, RetNet gene list.

echo "Done. Data is in $DIR"
