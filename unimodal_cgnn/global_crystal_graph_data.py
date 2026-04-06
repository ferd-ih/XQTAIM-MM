import os
import csv
import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
from torch.nn.functional import one_hot
from ase import io
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifParser
from scipy.stats import rankdata
from torch_geometric.utils import dense_to_sparse, add_self_loops
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering as SPCL
from collections import Counter
from math import isnan

class CrystalGraph(Dataset):
    def __init__(self, root_dir, id_prop_dir=None, r_max=8.0, n_neighbors=12, cls_num=3, transform=None, pre_transform=None, random_seed=123):
        super().__init__(root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.r_max = r_max
        self.n_neighbors = n_neighbors

        if id_prop_dir is not None:
            id_prop_file = os.path.join(id_prop_dir, 'id_prop.csv')
        else:
            id_prop_file = os.path.join(root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), f"Error: '{id_prop_file}' does not exist!"
        self.id_prop_df = pd.read_csv(id_prop_file, header=None, names=["id", "target", "graph_level_att"])
        self.data_list = self.id_prop_df["id"].tolist()
        self.target_list = self.id_prop_df["target"].tolist()
        self.global_value = self.id_prop_df["graph_level_att"].tolist() # should be [[xx],[xx]]
        self.encoder_elem = ELEM_Encoder()
        self.clusterizer = SPCL(n_clusters=cls_num, random_state=None,assign_labels='discretize')
        self.clusterizer2  = KMeans(n_clusters=cls_num, random_state=None)
        random.seed(random_seed)

    def len(self):
        return len(self.data_list)

    def get(self, cod_id):
        idx = self.data_list.index(cod_id)
        target = torch.tensor([self.target_list[idx]], dtype=torch.float)

        cif_path = os.path.join(self.root_dir, f"{cod_id}.cif")
        # try:
        #     crystal = Structure.from_file(cif_path)
        #     enc_compo = self.encoder_elem.encode(crystal.composition)
        # except:
        #     print(f"Error: {cif_path} is not a valid CIF file to encode")
        #     enc_compo = torch.zeros(1, 113)

        # ase_crystal = io.read(cif_path)
        # crystal = Structure.from_file(cif_path)
        parser = CifParser(cif_path, occupancy_tolerance=5.0)
        crystal = parser.parse_structures(primitive=False)[0]
        ase_crystal = AseAtomsAdaptor.get_atoms(crystal)
        enc_compo = self.encoder_elem.encode(ase_crystal)
        node_feats = self._get_node_features(ase_crystal)
        # print("Somehow we got here")
        edge_index, edge_weight = self._get_adjacency_info(ase_crystal)
        edge_attr = self._get_edge_features(edge_weight)
        try:
            value = self.global_value[idx]
            if value is None or (isinstance(value, float) and isnan(value)):
                raise ValueError("Invalid global value")
            # u = torch.tensor([[0.0]], dtype=torch.float)
        except (IndexError, ValueError):
            u = np.zeros((3))
            u = torch.Tensor(u[np.newaxis, ...])

        # enc_compo = self.encoder_elem.encode(crystal.composition)
        g_coords = crystal.cart_coords
        groups = [0]*len(g_coords)
        if len(g_coords) > 2:
            try: groups = self.clusterizer.fit_predict(g_coords)
            except: groups = self.clusterizer2.fit_predict(g_coords)
        groups = torch.tensor(groups).long()
        return Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr, global_feature=enc_compo, edge_weight=edge_weight, y=target, id=cod_id, u=u, cluster=groups)

    def _get_node_features(self, ase_crystal):
        atomic_numbers = torch.tensor(ase_crystal.get_atomic_numbers()).to(torch.int64)
        node_features = one_hot(atomic_numbers, num_classes=113)
        return node_features.to(torch.float)
    
    def _trimming(self, matrix, r_max, n_neighbors):
        mask = matrix > r_max
        matrix_masked = np.ma.array(matrix, mask=mask)
        matrix_trimmed = rankdata(matrix_masked, method="ordinal", axis=1)
        matrix_trimmed = np.nan_to_num(np.where(mask, np.nan, matrix_trimmed))
        matrix_trimmed[matrix_trimmed > n_neighbors + 1] = 0
        matrix_trimmed = np.where(matrix_trimmed == 0, matrix_trimmed, matrix)
        return matrix_trimmed
    
    def _get_adjacency_info(self, ase_crystal):
        # print("Starting _get_adjacency_info")
        dist_matrix = ase_crystal.get_all_distances(mic=True)
        # print("Distance matrix calculated")
        dist_matrix_trimmed = self._trimming(
            matrix=dist_matrix,
            r_max=self.r_max,
            n_neighbors=self.n_neighbors)
        # print("Distance matrix trimmed")
        dist_matrix_trimmed = torch.tensor(dist_matrix_trimmed)
        out = dense_to_sparse(dist_matrix_trimmed)
        edge_index = out[0]
        edge_weight = out[1]
        self_loops = True
        if self_loops:
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, num_nodes=len(ase_crystal), fill_value=0
            )
        return edge_index, edge_weight

    def _get_edge_features(self, edge_weight):
        edge_attr = self.distance_gaussian(edge_weight)
        return edge_attr.to(torch.float)


    def distance_gaussian(self, edge_weight, start=0.0, stop=5.0, resolution=50, coef=0.5):
        offset = torch.linspace(start, stop, resolution)
        edge_weight = edge_weight.unsqueeze(-1) - offset.view(1, -1)
        # return torch.exp(-1 * edge_weight.pow(2) / coef ** 2)
        return torch.exp(-1*torch.pow(edge_weight, 2)/coef**2)

    def get_graph_info(self):
        return {
            "data_list": self.data_list,
            "target_list": self.target_list,
            "global_value": self.global_value
        }

class ELEM_Encoder:
    def __init__(self):
        self.elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 
                        'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
                        'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 
                        'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 
                        'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 
                        'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
        self.e_arr = np.array(self.elements)

    def encode(self, ase_atoms):
        symbols = ase_atoms.get_chemical_symbols()
        composition_dict = Counter(symbols)

        answer = [0] * len(self.elements)

        total = sum(composition_dict.values())
        for elem, count in composition_dict.items():
            if elem not in self.elements:
                raise ValueError(f"Element '{elem}' is not in the encoder's supported list.")
            idx_e = self.elements.index(elem)
            answer[idx_e] = count / total

        return torch.tensor(answer).float().view(1, -1)

    def decode_pymatgen_num(self, tensor_idx):
        idx = (tensor_idx - 1).cpu().tolist()
        return self.e_arr[idx]