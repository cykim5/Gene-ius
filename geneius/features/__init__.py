"""Stage 3: variant features. What each mutation does to the protein.

| Feature                    | Source             | File            |
| damage score               | AlphaMissense      | alphamissense.py |
| effect score, embedding    | ESM-2              | esm2.py          |
| pLDDT, burial, distances   | AlphaFold          | alphafold.py     |
| protein region flags       | UniProt features   | regions.py       |
| position, change type      | sequence           | regions.py       |
| population frequency       | gnomAD v2.1.1, v3.1.2 | gnomad.py     |

Every function takes variant rows (gene, uniprot_id, pos, ref, alt) and returns
one column per feature, so the feature table can be built by joining them.
"""
