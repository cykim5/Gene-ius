"""gnomAD population frequencies and gene constraint for the panel genes.

A variant that's common in people without severe genetic disease is very unlikely to
cause a rare disease (ACMG BA1/BS1). We use gnomAD two ways: as a feature for every
variant, and as a source of extra benign training examples.

Versions: v2.1.1 (125,748 exomes + 15,708 genomes) and v3.1.2 (76,156 genomes). Both
were released before the 2023 cutoff. Don't use v4 (late 2023).

Matching to the variant table:
  v3 is on GRCh38, same as ClinVar, so we match on chrom-pos-ref-alt (exact).
  v2 is on GRCh37, so we match on the protein change (gene + ref + position + alt) using
  gnomAD's canonical transcript. The script reports how often the two methods agree.

    python -m geneius.features.gnomad

Writes to data/processed/:
  gnomad_variants.csv        one row per variant in variants.csv
  gnomad_common_missense.csv gnomAD missense variants not in ClinVar with max_af >= 0.1%
  gnomad_constraint.csv      one row per gene
Raw API responses are cached in $GENEIUS_DATA_DIR/raw/gnomad/ so reruns are instant.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .. import config
from ..data.clinvar import AA3

API = "https://gnomad.broadinstitute.org/api"
PROCESSED = config.REPO_ROOT / "data" / "processed"
COMMON_AF = 0.001  # 0.1%, for gnomad_common_missense.csv

_FREQ = "ac an af homozygote_count"
QUERIES = {
    "r3": f"""query($g: String!) {{ gene(gene_symbol: $g, reference_genome: GRCh38) {{
        variants(dataset: gnomad_r3) {{ variant_id consequence hgvsp
          genome {{ {_FREQ} faf95 {{ popmax popmax_population }} }} }} }} }}""",
    "r2": f"""query($g: String!) {{ gene(gene_symbol: $g, reference_genome: GRCh37) {{
        gnomad_constraint {{ pli oe_lof_upper mis_z oe_mis }}
        variants(dataset: gnomad_r2_1) {{ variant_id consequence hgvsp
          exome {{ {_FREQ} faf95 {{ popmax popmax_population }} }}
          genome {{ {_FREQ} }} }} }} }}""",
}
_HGVSP = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$")


def parse_hgvsp(hgvsp: str | None) -> tuple[str, int, str] | None:
    """'p.Arg591Cys' -> ('R', 591, 'C'). None if not a simple missense change."""
    m = _HGVSP.match(hgvsp or "")
    if not m or m.group(1) not in AA3 or m.group(3) not in AA3:
        return None
    return AA3[m.group(1)], int(m.group(2)), AA3[m.group(3)]


def fetch_gene(gene: str, release: str, retries: int = 5) -> dict:
    """gnomAD API response for one gene and release ('r2' or 'r3'), cached on disk."""
    cache = config.data_dir() / "raw" / "gnomad" / f"{gene}_{release}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    body = json.dumps({"query": QUERIES[release], "variables": {"g": gene}}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read())
            if out.get("errors"):
                raise RuntimeError(out["errors"][0].get("message"))
            gene_data = out["data"]["gene"]
            if gene_data is None:
                raise RuntimeError(f"gnomAD has no gene {gene}")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(gene_data))
            return gene_data
        except Exception as e:  # rate limits and timeouts: back off and retry
            wait = 10 * (attempt + 1)
            print(f"[gnomad] {gene} {release}: {e}. Retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"gnomAD request failed for {gene} {release}")


def _sum_freq(*blocks) -> tuple[int, int, int]:
    """Add allele count, allele number, and homozygote count across exome/genome blocks."""
    ac = an = hom = 0
    for b in blocks:
        if b:
            ac += b.get("ac") or 0
            an += b.get("an") or 0
            hom += b.get("homozygote_count") or 0
    return ac, an, hom


def _missense_rows(gene: str, release: str, data: dict) -> list[dict]:
    rows = []
    for v in data["variants"]:
        if v["consequence"] != "missense_variant":
            continue
        change = parse_hgvsp(v["hgvsp"])
        if not change:
            continue
        if release == "r3":
            g = v.get("genome") or {}
            ac, an, hom = _sum_freq(g)
            popmax = (g.get("faf95") or {}).get("popmax")
        else:
            ex = v.get("exome") or {}
            ac, an, hom = _sum_freq(ex, v.get("genome"))
            popmax = (ex.get("faf95") or {}).get("popmax")
        chrom, pos, ref, alt = v["variant_id"].split("-")
        rows.append({
            "gene": gene, "release": release, "gnomad_id": v["variant_id"],
            "key": f"chr{chrom}:{pos}:{ref}:{alt}",
            "ref_aa": change[0], "position": change[1], "alt_aa": change[2],
            "ac": ac, "an": an, "af": ac / an if an else 0.0, "hom": hom, "faf95_popmax": popmax,
        })
    return rows


def pull(genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All gnomAD missense variants for the genes (both releases) and the constraint table."""
    rows, constraint = [], []
    for gene in genes:
        for release in ("r3", "r2"):
            data = fetch_gene(gene, release)
            rows += _missense_rows(gene, release, data)
            if release == "r2":
                constraint.append({"gene": gene, **(data.get("gnomad_constraint") or {})})
        print(f"[gnomad] {gene} done", file=sys.stderr)
    return pd.DataFrame(rows), pd.DataFrame(constraint)


def match(variants: pd.DataFrame, gn: pd.DataFrame) -> pd.DataFrame:
    """One row per variant in the table with its gnomAD frequencies (0 when absent)."""
    out = variants[["variant_id", "gene", "position", "ref_aa", "alt_aa", "key", "protein_variant"]].copy()

    r3 = gn[gn.release == "r3"].drop_duplicates("key").set_index("key")
    hit3 = out.key.map(r3.af)
    out["r3_af"] = hit3.fillna(0.0)
    out["r3_ac"] = out.key.map(r3.ac).fillna(0).astype(int)
    out["r3_hom"] = out.key.map(r3.hom).fillna(0).astype(int)
    out["r3_found"] = hit3.notna()

    r2 = gn[gn.release == "r2"].copy()
    r2["pkey"] = r2.gene + ":" + r2.ref_aa + r2.position.astype(str) + r2.alt_aa
    # Two DNA changes can give the same protein change. Keep the more common one.
    r2 = r2.sort_values("af", ascending=False).drop_duplicates("pkey").set_index("pkey")
    pkey = out.gene + ":" + out.ref_aa + out.position.astype(str) + out.alt_aa
    hit2 = pkey.map(r2.af)
    out["r2_af"] = hit2.fillna(0.0)
    out["r2_ac"] = pkey.map(r2.ac).fillna(0).astype(int)
    out["r2_hom"] = pkey.map(r2.hom).fillna(0).astype(int)
    out["r2_faf95_popmax"] = pkey.map(r2.faf95_popmax)
    out["r2_found"] = hit2.notna()

    out["max_af"] = out[["r3_af", "r2_af"]].max(axis=1)
    out["hom_total"] = out.r3_hom + out.r2_hom
    out["in_gnomad"] = out.r3_found | out.r2_found
    return out


def common_not_in_clinvar(variants: pd.DataFrame, gn: pd.DataFrame, min_af: float = COMMON_AF) -> pd.DataFrame:
    """Common gnomAD missense variants that aren't in the ClinVar table at all.

    Candidate extra benign examples. Positions use gnomAD's canonical transcript.
    """
    seen = set(variants.gene + ":" + variants.ref_aa + variants.position.astype(str) + variants.alt_aa)
    gn = gn.copy()
    gn["pkey"] = gn.gene + ":" + gn.ref_aa + gn.position.astype(str) + gn.alt_aa
    best = (gn.sort_values("af", ascending=False).drop_duplicates("pkey"))
    best = best[(best.af >= min_af) & ~best.pkey.isin(seen)]
    return best[["gene", "ref_aa", "position", "alt_aa", "release", "gnomad_id", "af", "ac", "an", "hom"]]


def agreement(matched: pd.DataFrame) -> dict:
    """How often v2 (protein match) and v3 (exact DNA match) agree, where both found the variant."""
    both = matched[matched.r3_found & matched.r2_found]
    if both.empty:
        return {"both_found": 0}
    ratio = (both.r2_af / both.r3_af).clip(upper=1e6)
    return {
        "both_found": len(both),
        "within_5x": float(((ratio > 0.2) & (ratio < 5)).mean()),
        "only_r3": int((matched.r3_found & ~matched.r2_found).sum()),
        "only_r2": int((matched.r2_found & ~matched.r3_found).sum()),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", default=str(PROCESSED / "variants.csv"))
    ap.add_argument("--out-dir", default=str(PROCESSED))
    args = ap.parse_args(argv)

    variants = pd.read_csv(args.variants)
    genes = sorted(variants.gene.unique())
    gn, constraint = pull(genes)
    matched = match(variants, gn)
    extra = common_not_in_clinvar(variants, gn)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    matched.to_csv(out / "gnomad_variants.csv", index=False)
    extra.to_csv(out / "gnomad_common_missense.csv", index=False)
    constraint.to_csv(out / "gnomad_constraint.csv", index=False)

    print(f"variants in table: {len(matched)}, found in gnomAD: {int(matched.in_gnomad.sum())}")
    print(f"v2/v3 agreement: {agreement(matched)}")
    print(f"common gnomAD missense not in ClinVar (max AF >= {COMMON_AF:.1%}): {len(extra)}")


if __name__ == "__main__":
    main()
