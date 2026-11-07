"""Build the heterogeneous graph for the GNN (Stage 4).

Steps:
  1. Load PrimeKG. Keep gene/protein, disease, phenotype, pathway, drug nodes.
  2. Add HPO is_a edges from the 2023 ontology if PrimeKG lacks them.
  3. Add variant nodes (labeled + VUS) with Stage 3 features.
  4. Variant -> gene for all. Variant -> disease and -> phenotype for labeled 2023 variants only.
  5. Mouse phenotype nodes and gene -> mouse phenotype edges (MGI).
  6. 3D edges: variant residues within 8-10 A, skip pLDDT < 70 (structural_edges.py).
  7. Related-protein edges at aligned residues (structural_edges.py).
  8. Mechanism and treatment strategy nodes and edges.
  9. Apply the leakage rules (leakage.py).
  10. Convert to PyG HeteroData, add reverse edges (ToUndirected).
  11. Export hand-built graph features for XGBoost.
"""
from __future__ import annotations


def build_graph():
    raise NotImplementedError
