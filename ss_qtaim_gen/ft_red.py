import pandas as pd
import numpy as np
import json
import os

QTAIM_FILES = "/ocean/projects/che250019p/fihiri/qtaim_graph_gen/omdb_PBE_CPProp"

DROP_NODE = ["cp_num", "element", "number", "pos_ang"]
DROP_EDGE = ["cp_num", "connected_bond_paths", "pos_ang"]


def iter_qtaim_pairs(root_dir):
    """Yields (cod_id, bcp_path, ncp_path) tuples."""
    for root, _, files in os.walk(root_dir):
        bcp = ncp = None
        for f in files:
            if f.endswith("_qtaim_bcp.json"):
                bcp = os.path.join(root, f)
            elif f.endswith("_qtaim_ncp.json"):
                ncp = os.path.join(root, f)
        if bcp and ncp:
            cod_id = os.path.basename(bcp).split('_')[0]
            yield cod_id, bcp, ncp


def extract_features(root_dir, batch_size=500):
    node_chunks, edge_chunks = [], []
    node_batch, edge_batch = [], []

    for cod_id, bcp_path, ncp_path in iter_qtaim_pairs(root_dir):
        with open(bcp_path) as f:
            bcp_data = json.load(f)
        with open(ncp_path) as f:
            ncp_data = json.load(f)
        for rec in ncp_data[cod_id].values():
            node_batch.append(rec)
        for rec in bcp_data[cod_id].values():
            edge_batch.append(rec)
        if len(node_batch) >= batch_size:
            node_chunks.append(pd.DataFrame(node_batch))
            node_batch = []
        if len(edge_batch) >= batch_size:
            edge_chunks.append(pd.DataFrame(edge_batch))
            edge_batch = []

    if node_batch:
        node_chunks.append(pd.DataFrame(node_batch))
    if edge_batch:
        edge_chunks.append(pd.DataFrame(edge_batch))

    df_nodes = pd.concat(node_chunks, ignore_index=True) if node_chunks else pd.DataFrame()
    df_edges = pd.concat(edge_chunks, ignore_index=True) if edge_chunks else pd.DataFrame()
    df_nodes.drop(columns=DROP_NODE, inplace=True, errors="ignore")
    df_edges.drop(columns=DROP_EDGE, inplace=True, errors="ignore")
    return df_nodes, df_edges


def reduce_features(df, threshold=0.9):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return df
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    return df.drop(columns=to_drop)


def save_feature_list(df_nodes, df_edges, out_dir):
    with open(os.path.join(out_dir, "node_features.json"), "w") as f:
        json.dump(df_nodes.columns.tolist(), f)
    with open(os.path.join(out_dir, "edge_features.json"), "w") as f:
        json.dump(df_edges.columns.tolist(), f)


if __name__ == "__main__":
    df_nodes, df_edges = extract_features(QTAIM_FILES)
    df_nodes = reduce_features(df_nodes)
    df_edges = reduce_features(df_edges)
    save_feature_list(df_nodes, df_edges, QTAIM_FILES)