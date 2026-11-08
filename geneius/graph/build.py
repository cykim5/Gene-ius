"""Build the heterogeneous graph for the GNN (Stage 4).

Done (minimal version for Output A):
  1. PrimeKG gene/protein, disease, and phenotype nodes, with disease-phenotype, disease-gene,
     phenotype-gene, phenotype is_a, and disease-disease edges, via the filtered loader.
  2. Leakage rules for held-out diseases (leakage.py).
  3. Variant nodes for every variant in the table, with amino-acid change features.
  4. Variant -> gene edges for all variants.
  5. Variant -> disease edges for train variants only, from pheno_ids_2023 (MONDO ids matched to
     PrimeKG disease ids), falling back to disease_2023 names.
  6. Labels (from class_2023) on train variants only.
  7. PyG HeteroData with reverse edges (ToUndirected), cached on disk.

Leakage: only the columns in USED_COLUMNS are read. `split` picks which rows get labels and
disease edges and is never stored on the graph. *_now columns are never read.

3D and related-protein links are in features/graph_features.py. Not built: mouse phenotype,
mechanism, treatment, pathway, and drug nodes.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from .. import config
from ..data import get_hpo, norm_hpo
from ..data.kg import _edges
from ..features.graph_features import extra_links
from ..features.variant_features import CHANGE_COLUMNS, all_variant_features, model_columns, model_matrix, variant_features
from .leakage import remove_heldout_disease_edges

USED_COLUMNS = ["variant_id", "gene", "ref_aa", "alt_aa", "class_2023", "split", "pheno_ids_2023", "disease_2023",
                "position", "uniprot_pos", "uniprot_ref"]
LABELS_2023 = {"P": 1.0, "B": 0.0}

KG_RELATIONS = {
    "disease_phenotype_positive": ("disease", "effect/phenotype", ("disease", "has_phenotype", "phenotype")),
    "disease_protein": ("disease", "gene/protein", ("disease", "associated_with", "gene")),
    "phenotype_protein": ("effect/phenotype", "gene/protein", ("phenotype", "associated_with", "gene")),
    "phenotype_phenotype": ("effect/phenotype", "effect/phenotype", ("phenotype", "is_a", "phenotype")),
    "disease_disease": ("disease", "disease", ("disease", "related_to", "disease")),
}
VARIANT_GENE = ("variant", "in_gene", "gene")
VARIANT_DISEASE = ("variant", "linked_to", "disease")
PARALOG_SAME = ("variant", "paralog_same", "paralog_variant")
PARALOG_NEAR = ("variant", "paralog_near", "paralog_variant")
PARALOG_GENE = ("paralog_variant", "in_gene", "gene")
NEAR_3D = ("variant", "near_3d", "variant")
NODE_TYPES = ("gene", "disease", "phenotype", "variant")
CACHE_VERSION = "a5"

_MONDO = re.compile(r"MONDO:MONDO:(\d+)")


@dataclass
class VariantGraph:
    data: HeteroData
    variant_ids: list[str]  # variant node order
    is_train: np.ndarray  # rows that carry a label; from `split`, for row selection only
    feature_columns: list[str]
    index: dict[str, list[str]]  # node type -> keys in node order
    disease_match: dict = field(default_factory=dict)


def _kg_edges(heldout: set[str]) -> dict[tuple, pd.DataFrame]:
    hpo = get_hpo()
    df = remove_heldout_disease_edges(heldout, list(KG_RELATIONS))
    out = {}
    for rel, (x_type, y_type, etype) in KG_RELATIONS.items():
        e = _edges(df, rel, x_type, y_type)
        keys = []
        for ids, names, node_type in ((e.a_id, e.a_name, etype[0]), (e.b_id, e.b_name, etype[2])):
            if node_type == "phenotype":
                lookup = {i: hpo.primary(norm_hpo(i)) for i in pd.unique(ids)}
                keys.append(ids.map(lookup).to_numpy(dtype=object))
            else:
                keys.append(names.to_numpy(dtype=object))
        out[etype] = pd.DataFrame({"src": keys[0], "dst": keys[1]}).drop_duplicates()
    return out


def _disease_lookup(heldout: set[str]) -> tuple[dict[int, str], dict[str, str]]:
    """PrimeKG disease name by MONDO number and by lowercased name."""
    df = remove_heldout_disease_edges(heldout, ["disease_protein", "disease_phenotype_positive", "disease_disease"])
    by_mondo, by_name = {}, {}
    for ids, names, types in ((df.x_id, df.x_name, df.x_type), (df.y_id, df.y_name, df.y_type)):
        sub = pd.DataFrame({"id": ids, "name": names})[types.values == "disease"].drop_duplicates()
        for raw, name in zip(sub["id"], sub["name"]):
            by_name.setdefault(str(name).lower(), name)
            for part in str(raw).split("_"):
                if part.isdigit():
                    by_mondo.setdefault(int(part), name)
    return by_mondo, by_name


def match_diseases(row: pd.Series, by_mondo: dict[int, str], by_name: dict[str, str]) -> tuple[list[str], str]:
    """PrimeKG diseases for one variant's 2023 conditions, and how they were matched."""
    ids = [int(m) for m in _MONDO.findall(str(row["pheno_ids_2023"] or ""))]
    hits = list(dict.fromkeys(by_mondo[i] for i in ids if i in by_mondo))
    if hits:
        return hits, "mondo"
    names = re.split(r"[|;]", str(row["disease_2023"] or ""))
    hits = list(dict.fromkeys(by_name[n.strip().lower()] for n in names if n.strip().lower() in by_name))
    return hits, ("name" if hits else "none")


def _cache_key(variants: pd.DataFrame, heldout: set[str], options: dict) -> str:
    h = hashlib.sha1(CACHE_VERSION.encode())
    h.update(pd.util.hash_pandas_object(variants, index=False).values.tobytes())
    h.update("|".join(sorted(heldout)).encode())
    h.update(repr(sorted(options.items())).encode())
    h.update(str(config.primekg_csv().stat().st_size).encode())
    table = config.feature_table_csv()
    if table.exists():
        h.update(f"{table.stat().st_size}:{table.stat().st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def build_graph(variants: pd.DataFrame | None = None, heldout_diseases: set[str] | frozenset = frozenset(),
                cache: bool = True, verbose: bool = True, drop_features: tuple[str, ...] = (),
                feature_table: pd.DataFrame | None = None, am_indicator: bool = True, paralog: bool = False,
                structure: bool = False, holdout_genes: tuple[str, ...] = ()) -> VariantGraph:
    """drop_features removes variant features, e.g. ("alphamissense",) for the run without AlphaMissense.
    am_indicator adds an am_missing feature (see features.variant_features.model_matrix).
    paralog adds paralog-variant nodes, same/near residue edges, and the paralog count features.
    structure adds 10 A C-alpha edges between variants and the 3D neighbor-label count features.
    holdout_genes: these genes' variants get no labels, no disease edges, and add nothing to
    label-based features, but stay in the graph.
    feature_table overrides feature_table.csv (tests)."""
    if variants is None:
        from ..data import load_variants

        variants = load_variants()
    variants = variants[USED_COLUMNS].reset_index(drop=True)
    heldout = set(heldout_diseases)
    options = {"drop_features": tuple(sorted(drop_features)), "am_indicator": am_indicator, "paralog": paralog,
               "structure": structure, "holdout_genes": tuple(sorted(holdout_genes))}

    cache_path = config.data_dir() / "cache" / f"variant_graph_{_cache_key(variants, heldout, options)}.pt"
    if cache and cache_path.exists():
        graph = torch.load(cache_path, weights_only=False)
        if verbose:
            _print_counts(graph)
        return graph

    kg = _kg_edges(heldout)
    is_train = ((variants["split"] == "train") & ~variants["gene"].isin(holdout_genes)).to_numpy()
    by_mondo, by_name = _disease_lookup(heldout)
    variant_disease, methods = [], []
    for i in np.flatnonzero(is_train):
        hits, how = match_diseases(variants.iloc[i], by_mondo, by_name)
        methods.append((variants.at[i, "class_2023"], how))
        variant_disease += [(str(variants.at[i, "variant_id"]), d) for d in hits]

    keys: dict[str, set] = {t: set() for t in NODE_TYPES}
    for (s, _, d), e in kg.items():
        keys[s].update(e.src)
        keys[d].update(e.dst)
    keys["gene"].update(variants["gene"])
    extra = extra_links(variants, is_train, paralog, structure)
    if paralog:
        keys["gene"].update(extra["paralog_variants"]["gene"])
    index = {t: sorted(k) for t, k in keys.items() if t != "variant"}
    index["variant"] = variants["variant_id"].astype(str).tolist()
    if paralog:
        index["paralog_variant"] = extra["paralog_variants"]["variant_id"].tolist()
    row = {t: {k: i for i, k in enumerate(ks)} for t, ks in index.items()}

    data = HeteroData()
    for t in index:
        data[t].num_nodes = len(index[t])

    def put(etype, pairs):
        if len(pairs):
            src = [row[etype[0]][a] for a, _ in pairs]
            dst = [row[etype[2]][b] for _, b in pairs]
            data[etype].edge_index = torch.tensor([src, dst], dtype=torch.long).unique(dim=1)
        else:
            data[etype].edge_index = torch.empty((2, 0), dtype=torch.long)

    for etype, e in kg.items():
        put(etype, list(zip(e.src, e.dst)))
    put(VARIANT_GENE, list(zip(index["variant"], variants["gene"])))
    put(VARIANT_DISEASE, variant_disease)
    if paralog:
        pv, links = extra["paralog_variants"], extra["paralog_links"]
        for kind, etype in (("same", PARALOG_SAME), ("near", PARALOG_NEAR)):
            sub = links[links["kind"] == kind]
            put(etype, list(zip([index["variant"][i] for i in sub["variant_row"]],
                                [index["paralog_variant"][j] for j in sub["paralog_row"]])))
        put(PARALOG_GENE, list(zip(pv["variant_id"], pv["gene"])))
        px = variant_features(pv.rename(columns={"ref": "ref_aa", "alt": "alt_aa"}))[CHANGE_COLUMNS]
        px = px.fillna(0.0).to_numpy(dtype=np.float32)
        data["paralog_variant"].x = torch.from_numpy((px - px.mean(0)) / np.where(px.std(0) > 0, px.std(0), 1.0))
    if structure:
        pairs = extra["structure_pairs"]
        data[NEAR_3D].edge_index = torch.tensor(pairs[["a", "b"]].to_numpy(dtype=np.int64).T.copy())
        data[NEAR_3D].edge_attr = torch.tensor(pairs["distance"].to_numpy(dtype=np.float32)).unsqueeze(-1)

    feats = all_variant_features(variants, feature_table)
    if paralog or structure:
        feats = pd.concat([feats, extra["features"]], axis=1)
    base_cols = [c for c in model_columns(feats) if c not in drop_features]
    base_cols += [c for c in extra.get("features", pd.DataFrame()).columns if c not in base_cols]
    mat = model_matrix(feats, base_cols, is_train, am_indicator)
    cols = list(mat.columns)
    x = mat.to_numpy(dtype=np.float32)
    mean = x[is_train].mean(axis=0) if is_train.any() else np.zeros(len(cols), dtype=np.float32)
    std = x[is_train].std(axis=0) if is_train.any() else np.ones(len(cols), dtype=np.float32)
    x = (x - mean) / np.where(std > 0, std, 1.0)
    data["variant"].x = torch.from_numpy(x.astype(np.float32))
    y = variants["class_2023"].map(LABELS_2023).to_numpy(dtype=np.float32, na_value=np.nan)
    data["variant"].y = torch.from_numpy(np.where(is_train, y, np.nan).astype(np.float32))
    data["variant"].train_mask = torch.from_numpy(is_train & ~np.isnan(y))
    data = T.ToUndirected(reduce="mean")(data)

    m = pd.DataFrame(methods, columns=["label", "how"])
    match = {
        "train_variants": int(is_train.sum()),
        "matched": int((m.how != "none").sum()),
        "by_mondo": int((m.how == "mondo").sum()),
        "by_name": int((m.how == "name").sum()),
        "match_rate": round(float((m.how != "none").mean()), 4) if len(m) else 0.0,
        "match_rate_by_label": {k: round(float((g.how != "none").mean()), 4) for k, g in m.groupby("label")},
        "variant_disease_edges": int(data[VARIANT_DISEASE].edge_index.size(1)),
    }
    graph = VariantGraph(data, index["variant"], is_train, cols, index, match)
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, cache_path)
    if verbose:
        _print_counts(graph)
    return graph


def _print_counts(graph: VariantGraph) -> None:
    d = graph.data
    print("[graph] nodes " + ", ".join(f"{t} {d[t].num_nodes}" for t in d.node_types), flush=True)
    print("[graph] edges " + ", ".join(f"{'__'.join(e)} {d[e].edge_index.size(1)}" for e in d.edge_types), flush=True)
    print(f"[graph] variant features {graph.feature_columns}", flush=True)
    print(f"[graph] disease match {graph.disease_match}", flush=True)
