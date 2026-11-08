"""evidence_by_source and evidence_report.

evidence_report reads the teammates' predictions.json for one variant: the evidence line, paralog
evidence, nearest known variants, FoldX mechanism, MGI mouse evidence, recalibrated
computational evidence (PP3/BP4), and the suggested assay. With time_machine=True it returns only
the 2023 model's score and the 2023 evidence line.

evidence_by_source is still a stub.
"""
from __future__ import annotations

from ... import config
from ...data import demo
from . import stubs

DISCLAIMER = "Research use only. Supports, but does not replace, expert variant classification."
REPORT_FIELDS = ["harmful", "confidence", "computational_evidence", "evidence", "paralog_evidence", "nearest_known",
                 "mechanism", "mechanism_status", "mouse_evidence", "suggested_assay", "disease_groups"]


def evidence_by_source(variant: str) -> dict:
    """How much each data source contributed. Returns {variant, contributions: {source: share}}."""
    return stubs.evidence_by_source(variant)


def _record(vid: str, time_machine: bool) -> dict:
    rec = demo.predictions().get(vid, {})
    ft = demo.feature_table()
    out = {"variant": ft.at[vid, "variant_id"], "variation_id": vid, "gene": ft.at[vid, "gene"],
           "clinvar_change": demo.clinvar_changes().get(vid)}
    if time_machine:
        out.update({"harmful_2023_model": rec.get("harmful_2023_model"), "evidence_2023": rec.get("evidence_2023"),
                    "note": "time machine: only the 2023 model and 2023 evidence"})
    else:
        out.update({k: rec.get(k) for k in REPORT_FIELDS})
        out["harmful_source"] = "current model"
    return out


def _markdown(r: dict) -> str:
    lines = [f"**{r['gene']} {r['variant'].split(':')[1]}** (ClinVar {r.get('clinvar_change')}, id {r['variation_id']})"]
    for k, v in r.items():
        if k in {"variant", "variation_id", "gene", "clinvar_change"} or v in (None, "", {}, []):
            continue
        lines.append(f"- {k.replace('_', ' ')}: {v}")
    lines.append(f"\n_{DISCLAIMER}_")
    return "\n".join(lines)


def evidence_report(variant: str, time_machine: bool = False) -> dict:
    """Evidence for one variant ('GENE:RefPosAlt', canonical numbering). Returns
    {query, records: [{variant, variation_id, gene, clinvar_change, harmful, confidence, computational_evidence,
    evidence, paralog_evidence, nearest_known, mechanism, mouse_evidence, suggested_assay, ...}], report_markdown}."""
    if not (config.processed_dir() / "predictions.json").exists():
        return stubs.evidence_report(variant)
    ids = demo.resolve(variant)
    if not ids:
        return {"query": variant, "status": "not found", "records": []}
    records = [_record(vid, time_machine) for vid in ids]
    if not time_machine:
        from .variant_info import _population, assess

        for r in records:
            pop = _population(r["variation_id"])
            r["population"] = pop["summary"] + ("; seen in apparently healthy people"
                                                 if pop["seen_in_apparently_healthy_people"] else "")
            r["assessment"] = assess(r, pop)
            if r["assessment"]["unsure"]:
                r["assessment"]["next_step"] = "say you're unsure, explain why, and call suggest_assay"
    out = {"query": variant, "records": records, "report_markdown": "\n\n".join(_markdown(r) for r in records),
           "disclaimer": DISCLAIMER}
    if len(ids) > 1:
        out["note"] = f"{len(ids)} ClinVar records share this protein change (different DNA changes)."
    return out
