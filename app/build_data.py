"""Turn the team's predictions.json into the small per-gene files the web app loads.

    python app/build_data.py path/to/predictions.json

Writes app/data/index.json (genes, demo variants) and app/data/genes/{GENE}.json.
The app only shows what's in predictions.json. Re-run this whenever predictions.json changes.
"""
from __future__ import annotations

import argparse
import json
import re
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
    ("CACNA1C:G402R", "Related genes"),
]

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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("predictions")
    args = ap.parse_args(argv)

    rows = json.loads(Path(args.predictions).read_text())
    by_gene = defaultdict(list)
    for r in rows:
        m = _VARIANT.match(str(r.get("variant", "")))
        if not m or m.group(1) not in UNIPROT:
            continue
        gene, ref, pos, alt = m.groups()
        r = dict(r, gene=gene, ref=ref, pos=int(pos), alt=alt, variation_id=str(r.get("variation_id", "")))
        r["treatment"] = strategy(gene, (r.get("mechanism") or {}).get("label", ""))
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
    unsure = next((k for k, r in present.items() if r.get("confidence") == "unsure"), None)
    if unsure:
        demos.append({"variant": unsure, "label": "Unsure case"})

    index = {
        "genes": {g: {"uniprot": UNIPROT[g][0], "length": UNIPROT[g][1], "variants": len(by_gene[g]),
                      "panel": "retina" if g in RETINA else "cardiac" if g in CHANNELS or g in
                      {"MYH7", "MYBPC3", "LMNA", "PKP2"} else "other"}
                  for g in sorted(by_gene)},
        "demos": demos,
    }
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{sum(len(v) for v in by_gene.values())} variants in {len(by_gene)} genes, {len(demos)} demos")


if __name__ == "__main__":
    main()
