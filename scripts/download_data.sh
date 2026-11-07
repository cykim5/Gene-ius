#!/usr/bin/env bash
# Downloads PrimeKG + HPO into $GENEIUS_DATA_DIR (default: ./data).
# Safe to re-run: skips files that already exist.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi
DIR="${GENEIUS_DATA_DIR:-$REPO/data}"
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

# PrimeKG (Harvard Dataverse, doi:10.7910/DVN/IXA7BM). Large file, takes a while.
dl https://dataverse.harvard.edu/api/access/datafile/6180620 kg.csv
# HPO ontology (term names, synonyms, is_a tree)
dl https://purl.obolibrary.org/obo/hp.obo hp.obo
# HPO disease annotations (disease -> phenotype, with frequencies)
dl https://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa phenotype.hpoa

echo "Done. Data is in $DIR"
