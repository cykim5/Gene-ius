"""Go/no-go check (Stage 2). Run this first.

Counts, per gene: labeled training variants (2023), 2023 VUS reclassified since
with 2+ stars, and how many of those have a specific disease.

If there are fewer than about 50 reclassified VUS across the pool, widen to whole
gene families (RetNet, sodium and potassium channels) or move CUTOFF earlier.

    python eval/go_no_go.py
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Needs data/clinvar.py load_missense() and compare_releases().")


if __name__ == "__main__":
    main()
