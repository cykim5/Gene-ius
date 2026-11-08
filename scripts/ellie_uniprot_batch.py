
import csv
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode

# 1. Target gene list
genes = [
    "RHO", "ABCA4", "GUCY2D", "CRX", "PRPH2",
    "RPE65", "CEP290", "BEST1", "NR2E3",
    "RPGR", "OPA1", "PAX6", "USH2A",
    "SCN5A", "KCNQ1", "KCNH2", "KCNJ2",
    "CACNA1C", "MYH7", "MYBPC3",
    "LMNA", "PKP2", "GLA"
]


# 2. File locations
root = Path(__file__).resolve().parent.parent

raw_dir = root / "data" / "raw" / "uniprot"
processed_dir = root / "data" / "processed"

raw_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)


# 3. Requested UniProt annotations
wanted_types = {
    "Domain",
    "Transmembrane",
    "Topological domain",
    "Active site",
    "Binding site",
    "Disulfide bond"
}

# 4. Helper: download JSON from UniProt
def get_json(url):
    request = Request(
        url,
        headers={"User-Agent": "GENEIUS-BioHack/1.0"}
    )
    with urlopen(request, timeout=45) as response:
        return json.load(response)

# 5. Helper: identify the primary gene name
def primary_gene(entry):
    for gene in entry.get("genes", []):
        name = gene.get("geneName", {}).get("value")
        if name:
            return name
    return None

# 6. Prepare output tables
mapping_rows = []
feature_rows = []

# 7. Process each gene
for gene in genes:
    print(f"\nProcessing {gene}...")

    try:
        query = (
            f"gene_exact:{gene} AND "
            "organism_id:9606 AND reviewed:true"
        )
        params = urlencode({
            "query": query,
            "format": "json",
            "size": 100
        })

        search_url = (
            "https://rest.uniprot.org/uniprotkb/search?"
            + params
        )

        results = get_json(search_url).get("results", [])

        # Only accept an exact primary gene match
        matches = [
            entry for entry in results
            if primary_gene(entry) == gene
            and entry.get("entryType") ==
                "UniProtKB reviewed (Swiss-Prot)"
        ]

        if len(matches) != 1:
            print(
                f"REVIEW NEEDED: {len(matches)} "
                f"exact entries for {gene}"
            )
            mapping_rows.append({
                "gene": gene,
                "uniprot_id": "",
                "protein_length": "",
                "sequence": "",
                "reviewed": "",
                "entry_version": "",
                "annotation_date": "",
                "status": "REVIEW NEEDED"
            })
            continue

        accession = matches[0]["primaryAccession"]

        # Load local copy if already downloaded
        json_path = raw_dir / f"{accession}.json"

        if json_path.exists():
            with open(
                json_path, "r", encoding="utf-8"
            ) as f:
                protein = json.load(f)
        else:
            protein = get_json(
                "https://rest.uniprot.org/"
                f"uniprotkb/{accession}.json"
            )
            with open(
                json_path, "w", encoding="utf-8"
            ) as f:
                json.dump(protein, f, indent=2)

        sequence = protein["sequence"]["value"]

        # Mapping table: one row per gene
        mapping_rows.append({
            "gene": gene,
            "uniprot_id": accession,
            "protein_length": len(sequence),
            "sequence": sequence,
            "reviewed": "Yes",
            "entry_version": protein.get(
                "entryAudit", {}
            ).get("entryVersion", ""),
            "annotation_date": protein.get(
                "entryAudit", {}
            ).get("lastAnnotationUpdateDate", ""),
            "status": "OK"
        })

        # Feature table: one row per annotation
        count = 0

        for feature in protein.get("features", []):
            if feature.get("type") not in wanted_types:
                continue

            location = feature.get("location", {})
            start = location.get("start", {})
            end = location.get("end", {})

            feature_rows.append({
                "gene": gene,
                "uniprot_id": accession,
                "feature_type": feature["type"],
                "description": feature.get(
                    "description", ""
                ),
                "start": start.get("value", ""),
                "end": end.get("value", ""),
                "start_modifier": start.get(
                    "modifier", ""
                ),
                "end_modifier": end.get(
                    "modifier", ""
                ),
                "feature_id": feature.get(
                    "featureId", ""
                )
            })
            count += 1

        print(
            f"OK: {gene} -> {accession}, "
            f"{len(sequence)} aa, {count} features"
        )

        # Avoid making requests too rapidly
        time.sleep(0.2)

    except Exception as e:
        print(f"ERROR for {gene}: {e}")
        mapping_rows.append({
            "gene": gene,
            "uniprot_id": "",
            "protein_length": "",
            "sequence": "",
            "reviewed": "",
            "entry_version": "",
            "annotation_date": "",
            "status": f"ERROR: {e}"
        })

# 8. Save the mapping table
mapping_path = processed_dir / "uniprot_mapping.csv"

with open(
    mapping_path, "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=[
            "gene", "uniprot_id", "protein_length",
            "sequence", "reviewed", "entry_version",
            "annotation_date", "status"
        ]
    )
    writer.writeheader()
    writer.writerows(mapping_rows)

# 9. Save the feature table
features_path = processed_dir / "uniprot_features.csv"

with open(
    features_path, "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=[
            "gene", "uniprot_id", "feature_type",
            "description", "start", "end",
            "start_modifier", "end_modifier",
            "feature_id"
        ]
    )
    writer.writeheader()
    writer.writerows(feature_rows)

print("\nFINISHED!")
print("Mapping table:", mapping_path)
print("Feature table:", features_path)

successful = sum(
    row["status"] == "OK" for row in mapping_rows
)
print(f"Successfully processed: {successful}/{len(genes)}")
print(f"Extracted feature annotations: {len(feature_rows)}")
