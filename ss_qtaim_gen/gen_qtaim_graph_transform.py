import os
import json
import re
import logging
import numpy as np
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

MAIN_DIR = "/ocean/projects/che250019p/fihiri/qtaim_graph_gen/omdb_PBE_CPProp"
# MAIN_DIR = "/ocean/projects/che250019p/fihiri/PAH101_data/pah101_pbe_CPProp"

NUM_WORKERS = 1

DEFAULT_NODE_FEATURES = [
    'e_density', 'iri', 'sign_lambda2_rho', 'delta_g_promolecular',
    'delta_g_hirsh', 'e_loc_func', 'esp_nuc', 'esp_e', 'grad_norm', 'ellip_e_dens'
    ]
DEFAULT_EDGE_FEATURES = [
    'Lagrangian_K', 'e_density', 'iri', 'sign_lambda2_rho', 'ave_loc_ion_E',
    'delta_g_promolecular', 'e_loc_func', 'esp_nuc', 'esp_e', 'grad_norm',
    'det_hessian', 'ellip_e_dens', 'eta'
    ]
BCP_CATEGORIES = ["standard", "orphaned", "hydrogen_bond"]

NODE_LOG_TRANSFORM_KEYS = {'esp_nuc', 'e_density', 'sign_lambda2_rho', 'iri', 'grad_norm'}
EDGE_LOG_TRANSFORM_KEYS = {'det_hessian', 'iri', 'grad_norm'}


def load_features(feature_dir):
    node_file = os.path.join(feature_dir, "node_features.json")
    edge_file = os.path.join(feature_dir, "edge_features.json")
    with open(node_file) as f:
        node_features = json.load(f)
    with open(edge_file) as f:
        edge_features = json.load(f)
    return DEFAULT_NODE_FEATURES, DEFAULT_EDGE_FEATURES


def extract_structure_id(filename):
    basename = os.path.basename(filename)
    match = re.match(r"(ROY_\d+|[^_]+)_", basename) #change here based on dataset name
    return match.group(1) if match else basename.split('_')[0]


def find_qtaim_files(sub_dir):
    bcp_file = ncp_file = None
    for fname in os.listdir(sub_dir):
        path = os.path.join(sub_dir, fname)
        if fname.endswith("_qtaim_bcp.json"):
            bcp_file = path
        elif fname.endswith("_qtaim_ncp.json"):
            ncp_file = path
            if bcp_file and ncp_file:
                break
    return bcp_file, ncp_file


def sanitize(value):
    if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
        return 0.0
    return value


def signed_log_transform(value):
    if value == 0:
        return 0.0
    return float(np.sign(value) * np.log1p(np.abs(value)))


def transform_descriptor(feat_name, value, transform_keys):
    value = sanitize(value)
    if feat_name in transform_keys:
        value = signed_log_transform(value)
    return value


def classify_bcp(connected_atoms, bcp_data, G):
    has_hydrogen = any(
        G.nodes[aid].get('element') == 'H'
        for aid in connected_atoms if aid in G.nodes()
    )
    if has_hydrogen and bcp_data.get('e_density', 1.0) < 0.1:
        return "hydrogen_bond"
    return "standard"


def build_graph(qtaim_ncp, qtaim_bcp, cod_id, node_features, edge_features):
    G = nx.Graph()
    atom_positions = {}
        
    for atom_data in qtaim_ncp[cod_id].values():
        atom_id = int(atom_data["number"])
        atom_positions[atom_id] = atom_data["pos_ang"]

        attrs = {"element": atom_data["element"], "pos": atom_data["pos_ang"]}
        for feat in node_features:
            if feat in atom_data:
                attrs[feat] = transform_descriptor(feat, atom_data[feat], NODE_LOG_TRANSFORM_KEYS)
        G.add_node(atom_id, **attrs)

    for bcp_key, bcp_data in qtaim_bcp[cod_id].items():
        attrs = {}
        for feat in edge_features:
            if feat in bcp_data:
                attrs[feat] = transform_descriptor(feat, bcp_data[feat], EDGE_LOG_TRANSFORM_KEYS)

            if "pos_ang" in bcp_data:
                attrs["bcp_pos"] = bcp_data["pos_ang"]
                bcp_pos = np.array(bcp_data["pos_ang"])
            else:
                bcp_pos = None
            
        category_onehot = {f"bcp_category_{cat}": 0 for cat in BCP_CATEGORIES}
            
        if "connected_bond_paths" in bcp_data:
            src, dst = bcp_data["connected_bond_paths"]
            category = classify_bcp([src, dst], bcp_data, G)
            category_onehot[f"bcp_category_{category}"] = 1
            attrs.update(category_onehot)

            if src in atom_positions and dst in atom_positions:
                dist = np.linalg.norm(
                    np.array(atom_positions[src]) - np.array(atom_positions[dst])
                )
                attrs["bond_distance"] = float(dist)

            G.add_edge(src, dst, **attrs)
        else:
            # Orphaned BCP. Create self-loop on nearest atom
            category_onehot["bcp_category_orphaned"] = 1
            attrs.update(category_onehot)

            if bcp_pos is not None:
                closest, min_dist = None, float('inf')
                for aid, apos in atom_positions.items():
                    d = np.linalg.norm(bcp_pos - np.array(apos))
                    if d < min_dist:
                        min_dist, closest = d, aid

                if closest is not None:
                    attrs["bond_distance"] = float(min_dist)
                    G.add_edge(closest, closest, **attrs)
            else:
                if atom_positions:
                    closest = min(atom_positions.keys())
                    G.add_edge(closest, closest, **attrs)

    return G


def validate_graph(G, cod_id):
    isolated = list(nx.isolates(G))
    if isolated:
        logging.warning(f"{cod_id}: {len(isolated)} isolated nodes")
    if G.number_of_nodes() > 0 and not nx.is_connected(G):
        n_comp = nx.number_connected_components(G)
        logging.warning(f"{cod_id}: graph has {n_comp} disconnected components")


def process_subdirectory(sub_dir, node_features, edge_features):
    bcp_file, ncp_file = find_qtaim_files(sub_dir)
    if not bcp_file or not ncp_file:
        logging.warning(f"Missing QTAIM files in {sub_dir}")
        return

    cod_id = extract_structure_id(bcp_file)

    with open(bcp_file) as f:
        qtaim_bcp = json.load(f)
    with open(ncp_file) as f:
        qtaim_ncp = json.load(f)

    G = build_graph(qtaim_ncp, qtaim_bcp, cod_id, node_features, edge_features)
    validate_graph(G, cod_id)

    metadata = {
        "structure_id": cod_id,
        "node_features": node_features,
        "edge_features": edge_features,
        "transforms": {
            "method": "signed_log1p",
            "formula": "sign(x) * log1p(|x|)",
            "node_transformed": list(NODE_LOG_TRANSFORM_KEYS),
            "edge_transformed": list(EDGE_LOG_TRANSFORM_KEYS)
        }
    }
    with open(os.path.join(sub_dir, f"{cod_id}_graph_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    gml_path = os.path.join(sub_dir, f"{cod_id}_qtaim_graph.gml")
    nx.write_gml(G, gml_path)


def main():
    node_features, edge_features = load_features(MAIN_DIR)
    
    sub_dirs = [
        os.path.join(MAIN_DIR, d) for d in os.listdir(MAIN_DIR)
        if os.path.isdir(os.path.join(MAIN_DIR, d))
    ]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_subdirectory, sd, node_features, edge_features): sd
            for sd in sub_dirs
        }
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
