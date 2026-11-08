"""Turn the team's predictions.json into the small per-gene files the web app loads.

    python app/build_data.py path/to/predictions.json

Writes app/data/index.json (genes, demo variants), app/data/genes/{GENE}.json, and the
AlphaFold structures in app/data/structures/ (downloaded once, then reused).
The app only shows what's in predictions.json. Re-run this whenever predictions.json changes.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).parent / "data"

UNIPROT = {
    "RHO": ("P08100", 348), "ABCA4": ("P78363", 2273), "GUCY2D": ("Q02846", 1103),
    "CRX": ("O43186", 299), "PRPH2": ("P23942", 346), "RPE65": ("Q16518", 533),
    "CEP290": ("O15078", 2479), "BEST1": ("O76090", 585), "NR2E3": ("Q9Y5X4", 410),
    "RPGR": ("Q92834", 1020), "OPA1": ("O60313", 960), "PAX6": ("P26367", 422),
    "USH2A": ("O75445", 5202), "SCN5A": ("Q14524", 2016), "KCNQ1": ("P51787", 676),
    "KCNH2": ("Q12809", 1159), "KCNJ2": ("P63252", 427), "CACNA1C": ("Q13936", 2221),
    "MYH7": ("P12883", 1935), "MYBPC3": ("Q14896", 1274), "LMNA": ("P02545", 664),
    "PKP2": ("Q99959", 881), "GLA": ("P06280", 429),
}
RETINA = {"RHO", "CEP290", "USH2A", "RPE65", "ABCA4", "GUCY2D", "CRX", "PRPH2", "BEST1", "NR2E3", "RPGR",
          "OPA1", "PAX6"}
CHANNELS = {"SCN5A", "KCNQ1", "KCNH2", "KCNJ2", "CACNA1C"}

# Demo order from the pitch. Only the ones present in predictions.json are shown.
DEMOS = [
    ("RHO:C187S", "Steve's variant"),
    ("KCNQ1:R591C", "Time machine"),
    ("KCNQ1:R591P", "Same residue, uncertain today"),
    ("CACNA1C:G402R", "Related genes"),
]
UNSURE_GENES = ["RHO", "KCNQ1", "RPE65", "SCN5A"]  # pick the unsure demo from a gene with a 3D model
DROP = {"disease", "disclaimer", "in_training_today", "mouse_evidence"}  # unused or gene-level

_VARIANT = re.compile(r"^([A-Z0-9]+):([A-Z])(\d+)([A-Z])$")


def strategy(gene: str, mechanism: str) -> dict | None:
    """Treatment strategy hypothesis from the mechanism. Same rules as the agent's suggest_treatment."""
    m = (mechanism or "").lower().replace(" ", "_").replace("-", "_")
    if gene == "RHO" and m in ("misfolding", "dominant_negative"):
        return {"strategy": "Knockdown-and-replace", "why": "Misfolded rhodopsin harms the healthy copy, so "
                "adding a working copy alone isn't enough."}
    if gene in CHANNELS and m == "gain_of_function":
        return {"strategy": "Channel blocker", "why": "Block the overactive current."}
    rules = {
        "loss_of_function": ("Gene augmentation", "Adding a working copy can restore function."),
        "dominant_negative": ("Knockdown-and-replace", "The faulty protein disrupts the healthy copy."),
        "gain_of_function": ("Silencing", "The protein does too much, so it needs to be silenced."),
        "misfolding": ("Folding chaperone", "Helping the protein fold can rescue some variants."),
    }
    if m not in rules:
        return None
    return {"strategy": rules[m][0], "why": rules[m][1]}


def fetch_structure(uniprot: str) -> Path | None:
    """AlphaFold model for the canonical sequence, saved once to data/structures/.

    Served from our own site, so visitors' browsers never call another server for it.
    Returns None when AlphaFold only has fragments or isoforms (USH2A).
    """
    path = OUT / "structures" / f"{uniprot}.pdb"
    if path.exists():
        return path
    headers = {"User-Agent": "gene-ius build script"}

    def get(url: str) -> bytes:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
            return r.read()

    models = json.loads(get(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"))
    model = next((m for m in models if m["modelEntityId"] == f"AF-{uniprot}-F1"), None)
    if model is None:
        return None
    lines = [ln for ln in get(model["pdbUrl"]).decode().splitlines() if ln.startswith(("ATOM", "TER"))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + ["END"]) + "\n")
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("predictions")
    args = ap.parse_args(argv)

    rows = json.loads(Path(args.predictions).read_text())
    by_gene = defaultdict(list)
    mouse = {}
    for r in rows:
        m = _VARIANT.match(str(r.get("variant", "")))
        if not m or m.group(1) not in UNIPROT:
            continue
        gene, ref, pos, alt = m.groups()
        if r.get("mouse_evidence"):
            mouse[gene] = r["mouse_evidence"]
        r = {k: v for k, v in r.items() if k not in DROP}
        r.update(gene=gene, ref=ref, pos=int(pos), alt=alt, variation_id=str(r.get("variation_id", "")))
        # The team's gene-level treatment note wins. Otherwise fall back to the mechanism rule.
        if not r.get("treatment"):
            rule = strategy(gene, (r.get("mechanism") or {}).get("label", "").split(" (")[0])
            r["treatment"] = f"{rule['strategy']}: {rule['why']}" if rule else ""
        by_gene[gene].append(r)

    genes_dir = OUT / "genes"
    genes_dir.mkdir(parents=True, exist_ok=True)
    for old in genes_dir.glob("*.json"):
        old.unlink()
    for gene, recs in by_gene.items():
        recs.sort(key=lambda r: (r["pos"], r["alt"]))
        (genes_dir / f"{gene}.json").write_text(json.dumps(recs, separators=(",", ":")))

    present = {f"{g}:{r['ref']}{r['pos']}{r['alt']}": r for g, recs in by_gene.items() for r in recs}
    demos = [{"variant": v, "label": label} for v, label in DEMOS if v in present]
    unsure = [k for k, r in present.items() if r.get("confidence") == "unsure" and r.get("clinvar_today") == "VUS"]
    unsure.sort(key=lambda k: (UNSURE_GENES.index(k.split(":")[0]) if k.split(":")[0] in UNSURE_GENES else 99, k))
    if unsure:
        demos.append({"variant": unsure[0], "label": "Unsure case"})

    structures = {g: fetch_structure(UNIPROT[g][0]) for g in sorted(by_gene)}
    index = {
        "genes": {g: {"uniprot": UNIPROT[g][0], "length": UNIPROT[g][1], "variants": len(by_gene[g]),
                      "mouse": mouse.get(g, ""),
                      "structure": f"/data/structures/{structures[g].name}" if structures[g] else None,
                      "panel": "retina" if g in RETINA else "cardiac" if g in CHANNELS or g in
                      {"MYH7", "MYBPC3", "LMNA", "PKP2"} else "other"}
                  for g in sorted(by_gene)},
        "demos": demos,
    }
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{sum(len(v) for v in by_gene.values())} variants in {len(by_gene)} genes, {len(demos)} demos")


if __name__ == "__main__":
    main()
