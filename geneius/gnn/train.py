"""Train the GNN ranker. Not written yet.

Plan:
  1. edges = graph_build.training_edges(heldout) using eval/data/heldout_diseases.txt
  2. Build a PyG HeteroData: node types gene/protein, disease, effect/phenotype, drug, ...
     Map each node type's ids to 0..n-1. Add reverse edges (T.ToUndirected).
  3. Model: 2-layer GraphSAGE wrapped with to_hetero, hidden size 128.
  4. Task: link prediction on phenotype -> gene/protein (and disease -> gene/protein).
     Negative sampling, binary cross entropy. Hold out 10% of edges for validation.
  5. Patient score for gene g: mean over patient terms t of sigmoid(dot(z_t, z_g)).
  6. Save node embeddings + id maps to checkpoints/, then implement predict.rank_genes.

Freeze the model by Saturday evening. After that, treat it as a fixed tool.
"""


def main() -> None:
    raise NotImplementedError("GNN training not written yet. See the plan in this file's docstring.")


if __name__ == "__main__":
    main()
