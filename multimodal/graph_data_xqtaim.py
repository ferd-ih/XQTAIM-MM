import os
import random
import torch
import numpy as np
from torch_geometric.data import Data
import pandas as pd
import networkx as nx
from math import isnan
from torch.nn.functional import one_hot

class QTAIMGraph:
    # Meta keys to exclude from feature extraction
    NODE_META_KEYS = set(['element', 'pos', 'rdg', 'id', 'index', 'name', 'label'])
    EDGE_META_KEYS = set(['source', 'target', 'bcp_pos','rdg', 'bond_distance'])
    
    # Default edge descriptor keys (used when include_edge_descriptors is None)
    DEFAULT_EDGE_DESCRIPTORS = [
        'Lagrangian_K', 'ave_loc_ion_E', 'bond_distance', 
        'delta_g_promolecular', 'det_hessian', 'e_density', 'e_loc_func', 
        'ellip_e_dens', 'esp_e', 'eta', 'sign_lambda2_rho', 'iri',
        'grad_norm', 'esp_nuc', 'delta_g_hirsh'
    ]
    
    def __init__(self, root_dir, id_prop_dir=None, random_seed=123,
                 include_node_descriptors=None, exclude_node_descriptors=None,
                 include_edge_descriptors=None, exclude_edge_descriptors=None):

        self.root_dir = root_dir
        self.ptable = CustomPeriodicTable()
        
        self.include_node_descriptors = include_node_descriptors
        self.exclude_node_descriptors = exclude_node_descriptors or []
        self.include_edge_descriptors = include_edge_descriptors
        self.exclude_edge_descriptors = exclude_edge_descriptors or []
        
        self.node_descriptors = None
        self.edge_descriptors = None
        
        if id_prop_dir is not None:
            id_prop_file = os.path.join(id_prop_dir, 'id_prop.csv')
            assert os.path.exists(id_prop_file), f"Error: '{id_prop_file}' does not exist!"
        else:
            id_prop_file = os.path.join(root_dir, 'id_prop.csv')
            assert os.path.exists(id_prop_file), f"Error: '{id_prop_file}' does not exist!"
        self.id_prop_df = pd.read_csv(id_prop_file, header=None, names=["id", "target", "graph_level_att"])
        self.data_list = self.id_prop_df["id"].astype(str).tolist()
        self.target_list = self.id_prop_df["target"].tolist()
        if "graph_level_att" in self.id_prop_df.columns:
            self.global_value = self.id_prop_df["graph_level_att"].tolist()
        else:
            self.global_value = [0.0] * len(self.data_list)

        if include_edge_descriptors is None:
            self._discover_descriptors()
        
        if self.include_edge_descriptors is not None:
            self._edge_descriptor_keys = [k for k in self.include_edge_descriptors 
                                          if k not in self.exclude_edge_descriptors]
        elif self.edge_descriptors is not None:
            self._edge_descriptor_keys = [k for k in self.edge_descriptors 
                                          if k not in self.exclude_edge_descriptors]
        else:
            self._edge_descriptor_keys = [k for k in self.DEFAULT_EDGE_DESCRIPTORS 
                                          if k not in self.exclude_edge_descriptors]
        
        print(f"[QTAIMGraph] Descriptor masking configuration:")
        print(f"  Node descriptors: include={self.include_node_descriptors}, exclude={self.exclude_node_descriptors}")
        print(f"  Edge descriptors ({len(self._edge_descriptor_keys)}): {self._edge_descriptor_keys}")

        random.seed(random_seed)
    
    def _discover_descriptors(self):
        """Discover available descriptors from the first graph in the dataset."""
        if len(self.data_list) == 0:
            return
        
        for cod_id in self.data_list:
            gml_path = os.path.join(self.root_dir, f"{cod_id}_qtaim_graph.gml")
            if os.path.exists(gml_path):
                try:
                    graph = nx.read_gml(gml_path)
                    node_descriptors = set()
                    for n in graph.nodes():
                        for key, val in graph.nodes[n].items():
                            if key not in self.NODE_META_KEYS:
                                if isinstance(val, (int, float, np.floating, np.integer)):
                                    node_descriptors.add(key)
                    
                    edge_descriptors = set()
                    for u, v, data in graph.edges(data=True):
                        for key, val in data.items():
                            if key not in self.EDGE_META_KEYS:
                                if isinstance(val, (int, float, np.floating, np.integer)):
                                    edge_descriptors.add(key)
                    
                    self.node_descriptors = sorted(node_descriptors)
                    self.edge_descriptors = sorted(edge_descriptors)
                    break
                except Exception:
                    continue
    
    def __len__(self):
        return len(self.data_list)
    
    def get(self, cod_id):
        cod_id = str(cod_id)
        
        try:
            idx = self.data_list.index(cod_id)
        except ValueError:
            raise ValueError(f"ID {cod_id} not found in dataset")
            
        target = torch.tensor([self.target_list[idx]], dtype=torch.float)
        gml_path = os.path.join(self.root_dir, f"{cod_id}_qtaim_graph.gml")
        
        if not os.path.exists(gml_path):
            raise FileNotFoundError(f"GML file not found: {gml_path}")
            
        graph = nx.read_gml(gml_path)

        try:
            value = self.global_value[idx]
            if value is None or (isinstance(value, float) and isnan(value)):
                raise ValueError("Invalid global value")
            u = torch.tensor([[value]], dtype=torch.float)
        except (IndexError, ValueError):
            u = np.zeros((3))
            u = torch.Tensor(u[np.newaxis, ...])

        qtaim_features, element_features, edge_index, edge_features, edge_weights, atomic_pos = self.nx_to_pyg(graph)
        
        # Combine one-hot encoded atomic features with continuous QTAIM features
        # One-hot encoded atomic features come first (119 dimensions), then continuous QTAIM features
        if qtaim_features.size(1) > 0:
            node_features = torch.cat([element_features, qtaim_features], dim=1)
        else:
            node_features = element_features
        node_features = torch.nan_to_num(node_features, nan=0.0, posinf=1e6, neginf=-1e6)
            
        return Data(x=node_features, z=element_features, edge_index=edge_index, 
                   edge_attr=edge_features, edge_weight=edge_weights, 
                   y=target, id=cod_id, u=u, pos=atomic_pos)

    def nx_to_pyg(self, graph):
        qtaim_features, element_features, node_mapping, atomic_pos = self._collect_node_features(graph)
        edge_index, edge_features, edge_weights = self._collect_edge_features(graph, node_mapping)
        return qtaim_features, element_features, edge_index, edge_features, edge_weights, atomic_pos

    def _collect_node_features(self, graph):
        if self.ptable is None:
            self.ptable = CustomPeriodicTable()
        atomic_numbers = []
        extra_features_list = []
        node_mapping = {}
        pos_list = []

        for idx, n in enumerate(graph.nodes()):
            node_data = graph.nodes[n]
            element = node_data.get('element', '')
            atomic_number = self.ptable.GetAtomicNumber(element)
            atomic_numbers.append(atomic_number)
            pos = node_data.get('pos', [0, 0, 0])
            # Ensure pos is a list of 3 floats (NetworkX may store as list from GML)
            if isinstance(pos, list) and len(pos) >= 3:
                pos = [float(pos[0]), float(pos[1]), float(pos[2])]
            else:
                pos = [0.0, 0.0, 0.0]
            pos_list.append(pos)

            numeric_items = {}
            for key, raw_val in node_data.items():
                if key in self.NODE_META_KEYS:
                    continue
                if self.node_descriptors and key not in self.node_descriptors:
                    continue
                    
                if self.include_node_descriptors is not None:
                    if key not in self.include_node_descriptors:
                        continue
                if key in self.exclude_node_descriptors:
                    continue
                    
                if isinstance(raw_val, (int, float, np.floating, np.integer)):
                    valf = float(raw_val)
                else:
                    continue
                numeric_items[key] = valf

            if self.node_descriptors:
                features = [numeric_items.get(k, 0.0) for k in self.node_descriptors]
            elif numeric_items:
                ordered_keys = sorted(numeric_items.keys())
                features = [numeric_items[k] for k in ordered_keys]
            else:
                features = []

            extra_features_list.append(features)

            node_mapping[n] = idx

        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.int64)
        atomic_onehots = one_hot(atomic_numbers, num_classes=119).float()

        # Store qtaim features separately from element features
        if extra_features_list and len(extra_features_list[0]) > 0:
            max_len = max(len(row) for row in extra_features_list)
            if any(len(row) != max_len for row in extra_features_list):
                print(f"WARNING: Node features have different lengths. Padding with zeros.")
                padded = []
                for row in extra_features_list:
                    if len(row) < max_len:
                        row = row + [0.0] * (max_len - len(row))
                    padded.append(row)
                extra_features_list = padded
            extra_features_tensor = torch.tensor(extra_features_list, dtype=torch.float)
            extra_features_tensor = torch.nan_to_num(extra_features_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
            qtaim_features = extra_features_tensor
        else:
            qtaim_features = torch.zeros((len(atomic_numbers), 0), dtype=torch.float)
        atomic_pos = torch.tensor(pos_list, dtype=torch.float)
        return qtaim_features, atomic_onehots, node_mapping, atomic_pos

    
    def _collect_edge_features(self, graph, node_mapping):
        edge_index = []
        edge_features = []
        edge_weights = []
        
        edge_descriptor_keys = self._edge_descriptor_keys
        
        for u, v, edge_data in graph.edges(data=True):
            if u == v:  # Skip self-loops for DMPNN compatibility
                continue
                
            feature_dict = {}
            for key, value in edge_data.items():
                if isinstance(value, (int, float)):
                    feature_dict[key] = float(value)

            features = []
            for key in edge_descriptor_keys:
                raw_val = feature_dict.get(key, 0.0)
                try:
                    valf = float(raw_val)
                except Exception:
                    valf = 0.0
                features.append(valf)
            pos_u = graph.nodes[u].get('pos', [0, 0, 0])
            pos_v = graph.nodes[v].get('pos', [0, 0, 0])
            if isinstance(pos_u, list) and len(pos_u) >= 3:
                pos_u = [float(pos_u[0]), float(pos_u[1]), float(pos_u[2])]
            else:
                pos_u = [0.0, 0.0, 0.0]
            if isinstance(pos_v, list) and len(pos_v) >= 3:
                pos_v = [float(pos_v[0]), float(pos_v[1]), float(pos_v[2])]
            else:
                pos_v = [0.0, 0.0, 0.0]
            dist = np.linalg.norm(np.array(pos_u) - np.array(pos_v))

            edge_index.append([node_mapping[u], node_mapping[v]])
            edge_features.append(features.copy())
            edge_weights.append(dist)
            edge_index.append([node_mapping[v], node_mapping[u]])
            edge_features.append(features.copy())
            edge_weights.append(dist)

        if edge_features:
            expected_len = len(edge_descriptor_keys)
            for i, feat in enumerate(edge_features):
                if len(feat) != expected_len:
                    edge_features[i] = feat[:expected_len] + [0.0] * (expected_len - len(feat))
            
            edge_features = np.array(edge_features, dtype=float)
            edge_features = np.nan_to_num(edge_features, nan=0.0, posinf=1e6, neginf=-1e6)
            edge_features = torch.tensor(edge_features, dtype=torch.float)
        else:
            if edge_descriptor_keys:
                edge_features = torch.zeros((0, len(edge_descriptor_keys)), dtype=torch.float)
            else:
                edge_features = torch.empty((0,), dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float) if edge_weights else torch.empty((0,))

        return edge_index, edge_features, edge_weights

    def get_graph_info(self):
        return {
            "data_list": self.data_list,
            "target_list": self.target_list,
            "global_value": self.global_value
        }
    
    def get_descriptor_config(self):
        return {
            "include_node_descriptors": self.include_node_descriptors,
            "exclude_node_descriptors": self.exclude_node_descriptors,
            "include_edge_descriptors": self.include_edge_descriptors,
            "exclude_edge_descriptors": self.exclude_edge_descriptors,
            "active_edge_descriptors": self._edge_descriptor_keys,
        }
    
    @classmethod
    def get_available_edge_descriptors(cls):
        return cls.DEFAULT_EDGE_DESCRIPTORS.copy()
    
    @staticmethod
    def get_available_node_descriptors_from_gml(gml_path):
        import networkx as nx
        graph = nx.read_gml(gml_path)
        meta_keys = set(['element', 'pos', 'rdg', 'id', 'index', 'name'])
        descriptor_names = set()
        
        for n in graph.nodes():
            node_data = graph.nodes[n]
            for key, value in node_data.items():
                if key not in meta_keys and isinstance(value, (int, float)):
                    descriptor_names.add(key)
        
        return sorted(list(descriptor_names))

class CustomPeriodicTable:
    """A picklable alternative to RDKit's Periodic Table.""" #for hpo
    def __init__(self):
        self.element_to_num = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
            'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
            'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
            'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
            'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
            'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
            'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
            'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
            'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
            'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118,
            '': 0 
        }
        
    def GetAtomicNumber(self, element):
        """Get the atomic number for a given element symbol."""
        if element and isinstance(element, str):
            element = element.capitalize()
        return self.element_to_num.get(element, 0)
