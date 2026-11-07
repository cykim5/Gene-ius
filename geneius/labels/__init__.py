"""Stage 5: labels.

| Output             | Train (2023 and earlier)                 | Test (after 2023)                      |
| A harmful          | ClinVar P/LP vs B/LB, humsavar benign    | 2023 VUS reclassified since (2+ stars) |
| B mechanism        | LoGoFunc, RHO classes, gene-disease rules | curated labels only                   |
| C disease          | ClinVar variant -> specific disease      | reclassified VUS with a disease        |
| D treatment        | rules from mechanism + inheritance       | migalastat list, approved therapies    |

Rules: merge P with LP and B with LB, drop conflicting, drop vague diseases for C.
Labels derived from gene-disease rules are for training only.
"""

HARMFUL = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
BENIGN = {"benign", "likely benign", "benign/likely benign"}
VUS = {"uncertain significance"}

MECHANISMS = ["gain_of_function", "loss_of_function", "dominant_negative", "misfolding", "no_effect"]
TREATMENTS = ["gene_augmentation", "silencing_knockdown_replace", "chaperone", "channel_blocker"]


def harmful_label(classification: str) -> int | None:
    """1 harmful, 0 benign, None for VUS, conflicting, or anything else."""
    c = (classification or "").strip().lower()
    if c in HARMFUL:
        return 1
    if c in BENIGN:
        return 0
    return None
