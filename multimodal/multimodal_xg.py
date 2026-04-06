import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from dataloader_crystal_graph import collect_graph_files, Normalizer
from graph_data_xqtaim import QTAIMGraph
from models.pyg_chemprop import RevIndexedData
import numpy as np

class MultimodalCrystalQTAIMDataset(Dataset):
    def __init__(self, crystal_graph_dir, qtaim_data_dir, id_prop_dir=None, split='train', split_file_dir=None, 
                 crystal_normalizer=None, qtaim_normalizer=None, random_seed=42,
                 normalize_edge_weight=False, use_dmpnn=True,
                 include_node_descriptors=None, exclude_node_descriptors=None,
                 include_edge_descriptors=None, exclude_edge_descriptors=None):
        """
        Combined dataset for crystal graphs and QTAIM graphs.
        
        Notes:
            normalize_edge_weight: Whether to normalize edge_weight (interatomic distances).
                NOTE: For SchNet-like models, you typically want *raw distances* so
                Gaussian smearing is physically meaningful. Set to False for SchNet.
            use_dmpnn: Whether to wrap QTAIM data in RevIndexedData for DMPNN.
                Set to False for SchNet-based models.
            include_node_descriptors: List of QTAIM node descriptor names to include (None = all).
                For ablation studies, e.g., ['e_density', 'Lagrangian_K'].
            exclude_node_descriptors: List of QTAIM node descriptor names to exclude (None = none).
            include_edge_descriptors: List of QTAIM edge descriptor names to include (None = all).
            exclude_edge_descriptors: List of QTAIM edge descriptor names to exclude (None = none).
        """
        super().__init__()
        
        self.normalize_edge_weight = normalize_edge_weight
        self.use_dmpnn = use_dmpnn
        self.valid_pairs = self._find_matching_pairs(crystal_graph_dir, qtaim_data_dir, split, split_file_dir, id_prop_dir)
        
        if len(self.valid_pairs) == 0:
            raise ValueError(f"No matching pairs found between crystal and QTAIM data for {split} split")
        
        print(f"Found {len(self.valid_pairs)} matching crystal-QTAIM pairs for {split} split")
        
        self.crystal_graphs = self._load_crystal_graphs(crystal_graph_dir, self.valid_pairs)
        self.qtaim_loader = QTAIMGraph(
            qtaim_data_dir,
            id_prop_dir,
            random_seed,
            include_node_descriptors=include_node_descriptors,
            exclude_node_descriptors=exclude_node_descriptors,
            include_edge_descriptors=include_edge_descriptors,
            exclude_edge_descriptors=exclude_edge_descriptors
        )
        self.crystal_normalizer = crystal_normalizer
        self.qtaim_normalizer = qtaim_normalizer
        
    def _find_matching_pairs(self, crystal_dir, qtaim_dir, split, split_file_dir=None, id_prop_dir=None):
        crystal_files = collect_graph_files(crystal_dir)
        crystal_ids = set([os.path.basename(f).replace('.pt', '') for f in crystal_files])

        if split in ['train', 'val', 'test']:
            if id_prop_dir and os.path.exists(os.path.join(id_prop_dir, f'{split}_id_prop.csv')):
                split_file = os.path.join(id_prop_dir, f'{split}_id_prop.csv')
            elif split_file_dir:
                split_file = os.path.join(split_file_dir, f'{split}_id_prop.csv')
            else:
                split_file = os.path.join(qtaim_dir, f'{split}_id_prop.csv')
        else:
            if id_prop_dir and os.path.exists(os.path.join(id_prop_dir, 'id_prop.csv')):
                split_file = os.path.join(id_prop_dir, 'id_prop.csv')
            else:
                split_file = os.path.join(qtaim_dir, 'id_prop.csv')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        df = pd.read_csv(split_file, header=None, names=['id', 'target', 'extra'])
        qtaim_ids = set([str(id) for id in df['id'].tolist()])
        matched_ids = crystal_ids.intersection(qtaim_ids)
        return sorted(list(matched_ids))
    
    def _load_crystal_graphs(self, crystal_dir, valid_ids):
        crystal_graphs = {}
        for cod_id in valid_ids:
            crystal_path = os.path.join(crystal_dir, f"{cod_id}.pt")
            if os.path.exists(crystal_path):
                data = torch.load(crystal_path, weights_only=False)
                if torch.isnan(data.u).any():
                    data.u = torch.Tensor(np.zeros((3))[np.newaxis, ...])
                crystal_graphs[cod_id] = data
            else:
                print(f"Warning: Crystal graph file not found: {crystal_path}")
        return crystal_graphs
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        cod_id = self.valid_pairs[idx]
        crystal_data = self.crystal_graphs[cod_id] 
        qtaim_data = self.qtaim_loader.get(cod_id)
        
        if self.qtaim_normalizer:
            qtaim_data.edge_attr = self.qtaim_normalizer["edge"].norm(qtaim_data.edge_attr)
            qtaim_data.y = self.qtaim_normalizer["target"].norm(qtaim_data.y)
            qtaim_data.u = self.qtaim_normalizer["u"].norm(qtaim_data.u)
            if self.normalize_edge_weight and "edge_w" in self.qtaim_normalizer and self.qtaim_normalizer["edge_w"] is not None:
                qtaim_data.edge_weight = self.qtaim_normalizer["edge_w"].norm(qtaim_data.edge_weight)
            if qtaim_data.x.shape[1] > 119 and "node" in self.qtaim_normalizer and self.qtaim_normalizer["node"] is not None:
                qtaim_data.x = torch.cat([
                    qtaim_data.x[:, :119],
                    self.qtaim_normalizer["node"].norm(qtaim_data.x[:, 119:])
                ], dim=1)

        try:
            debug_norm = os.environ.get("MM_DEBUG_NORM", "0") not in ("0", "false", "False", None)
        except Exception:
            debug_norm = False
        if debug_norm and idx < 3:
            if hasattr(crystal_data, 'x') and crystal_data.x is not None and crystal_data.x.numel() > 0:
                one_hot_row_sums = crystal_data.x.sum(dim=1)
                one_hot_mean_sum = float(one_hot_row_sums.mean().item())
                one_hot_min_sum = float(one_hot_row_sums.min().item())
                one_hot_max_sum = float(one_hot_row_sums.max().item())
                print(f"[DEBUG_NORM] crystal one-hot row-sum mean/min/max: {one_hot_mean_sum:.3f}/{one_hot_min_sum:.3f}/{one_hot_max_sum:.3f}")
            if hasattr(qtaim_data, 'x') and qtaim_data.x is not None and qtaim_data.x.shape[1] > 119:
                cont = qtaim_data.x[:, 119:]
                cont_mean = float(cont.mean().item())
                cont_std = float(cont.std().item())
                cont_min = float(cont.min().item())
                cont_max = float(cont.max().item())
                print(f"[DEBUG_NORM] qtaim node cont mean/std/min/max: {cont_mean:.4f}/{cont_std:.4f}/{cont_min:.2f}/{cont_max:.2f}")
            if hasattr(qtaim_data, 'edge_attr') and qtaim_data.edge_attr is not None and qtaim_data.edge_attr.numel() > 0:
                e = qtaim_data.edge_attr
                e_mean = float(e.mean().item())
                e_std = float(e.std().item())
                print(f"[DEBUG_NORM] qtaim edge mean/std: {e_mean:.4f}/{e_std:.4f}")
        
        qtaim_output = RevIndexedData(qtaim_data) if self.use_dmpnn else qtaim_data
        
        return {
            'crystal': crystal_data,
            'qtaim': qtaim_output,
            'target': qtaim_data.y.squeeze(),
            'cod_id': cod_id
        }


def multimodal_collate_fn(batch):
    crystal_list = [item['crystal'] for item in batch]
    qtaim_list = [item['qtaim'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    cod_ids = [item['cod_id'] for item in batch]
    crystal_batch = Batch.from_data_list(crystal_list)
    qtaim_batch = Batch.from_data_list(qtaim_list)
    return {
        'crystal': crystal_batch,
        'qtaim': qtaim_batch,
        'targets': targets,
        'cod_ids': cod_ids
    }

def create_multimodal_normalizers(crystal_dir, qtaim_dir, split_file_dir=None, id_prop_dir=None, random_seed=42, 
                                   normalize_edge_weight=True, use_dmpnn=True,
                                   include_node_descriptors=None, exclude_node_descriptors=None,
                                   include_edge_descriptors=None, exclude_edge_descriptors=None):
    print("Creating normalizers from training data...")

    train_dataset = MultimodalCrystalQTAIMDataset(
        crystal_graph_dir=crystal_dir,
        qtaim_data_dir=qtaim_dir,
        id_prop_dir=id_prop_dir,
        split="train",
        split_file_dir=split_file_dir,
        crystal_normalizer=None,
        qtaim_normalizer=None,
        random_seed=random_seed,
        normalize_edge_weight=False,  # Don't normalize during normalizer creation
        use_dmpnn=use_dmpnn,
        include_node_descriptors=include_node_descriptors,
        exclude_node_descriptors=exclude_node_descriptors,
        include_edge_descriptors=include_edge_descriptors,
        exclude_edge_descriptors=exclude_edge_descriptors,
    )

    print(f"Computing normalizers from {len(train_dataset)} training samples...")

    crystal_edge, crystal_target, crystal_u, crystal_global = [], [], [], []
    qtaim_edge, qtaim_edge_w, qtaim_target, qtaim_u, qtaim_node_cont = [], [], [], [], []

    for sample in train_dataset:
        c, q = sample["crystal"], sample["qtaim"]

        crystal_edge.append(c.edge_attr)
        crystal_target.append(c.y)
        crystal_u.append(c.u)
        crystal_global.append(c.global_feature)

        qtaim_edge.append(q.edge_attr)
        qtaim_target.append(q.y)
        qtaim_u.append(q.u)
        
        if hasattr(q, 'edge_weight') and q.edge_weight is not None:
            qtaim_edge_w.append(q.edge_weight)

        if q.x.shape[1] > 119:
            qtaim_node_cont.append(q.x[:, 119:])

    def make_normalizer(tensors):
        return Normalizer(torch.cat(tensors, dim=0)) if tensors else None

    crystal_normalizer = {
        "edge": make_normalizer(crystal_edge),
        "target": make_normalizer(crystal_target),
        "u": make_normalizer(crystal_u),
        "global_feature": make_normalizer(crystal_global),
    }

    qtaim_normalizer = {
        "edge": make_normalizer(qtaim_edge),
        "edge_w": make_normalizer(qtaim_edge_w) if normalize_edge_weight else None,
        "target": make_normalizer(qtaim_target),
        "u": make_normalizer(qtaim_u),
        "node": make_normalizer(qtaim_node_cont),
    }

    print("Normalizers created successfully!")
    return crystal_normalizer, qtaim_normalizer

