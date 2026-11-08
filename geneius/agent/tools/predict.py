"""gnn_predict and gnn_score.

gnn_predict: Output A (harmful) from models/gnn.py for variants in the variant table. Mechanism,
disease, and treatment are None until outputs B to D exist. Falls back to the stub when the
data or the trained model is missing.
gnn_score: stub.
"""
from __future__ import annotations

import re
from functools import lru_cache

from ... import config
from ...data.clinvar import AA3
from . import stubs

_NAMED = re.compile(r"^\s*([A-Za-z0-9-]+)[\s:]+(?:p\.)?([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})\s*$")


@lru_cache(maxsize=1)
def _variant_index() -> dict:
    """(gene, ref, pos, alt) and variant_id -> row of 2023-safe columns."""
    from ...data import load_variants

    df = load_variants()[["variant_id", "gene", "position", "ref_aa", "alt_aa", "class_2023", "stars_2023"]]
    rows = {}
    for r in df.itertuples(index=False):
        rec = r._asdict()
        rows[str(r.variant_id)] = rec
        rows[(str(r.gene).upper(), str(r.ref_aa), int(r.position), str(r.alt_aa))] = rec
    return rows


def _aa(code: str) -> str | None:
    if len(code) == 1:
        return code.upper()
    return AA3.get(code.capitalize())


def resolve_variant(variant: str) -> dict | None:
    """'RHO p.Pro23His', 'RHO P23H', 'RHO:P23H', or a ClinVar variant id -> table row, or None."""
    index = _variant_index()
    q = variant.strip()
    if q.isdigit():
        return index.get(q)
    m = _NAMED.match(q)
    if not m:
        return None
    gene, ref, pos, alt = m.groups()
    ref, alt = _aa(ref), _aa(alt)
    if not ref or not alt:
        return None
    return index.get((gene.upper(), ref, int(pos), alt))


def gnn_predict(variant: str) -> dict:
    """Outputs A to D with confidence.
    Returns {variant, harmful, mechanism: {class: p}, disease: {name: p}, treatment, confidence}."""
    if config.use_stubs() or not (config.gnn_a_dir() / "predictions.csv").exists():
        return stubs.gnn_predict(variant)
    try:
        row = resolve_variant(variant)
    except FileNotFoundError:
        return stubs.gnn_predict(variant)
    if row is None:
        return {"variant": variant, "error": "not in the variant table (try 'GENE p.Pro23His' or a ClinVar id)"}
    from ...models.gnn import predict

    out = predict([str(row["variant_id"])])[0]
    out.update({
        "variant": f"{row['gene']} p.{row['ref_aa']}{row['position']}{row['alt_aa']}",
        "variant_id": str(row["variant_id"]),
        "gene": row["gene"],
        "clinvar_class_2023": row["class_2023"],
        "model": "GNN output A (harmful vs benign); mechanism, disease, and treatment outputs aren't built yet",
    })
    return out


def gnn_score(source: str, target: str) -> dict:
    """GNN score for any proposed link. Used to guide the agent's graph search.
    Returns {source, target, score}."""
    return stubs.gnn_score(source, target)
