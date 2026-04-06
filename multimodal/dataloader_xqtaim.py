import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import networkx as nx
from tqdm import tqdm
from graph_data_xqtaim import QTAIMGraph
import os
import pandas as pd
import numpy as np
import traceback
from sklearn.model_selection import train_test_split
from models.pyg_chemprop import RevIndexedData, RevIndexedDataset


class QTAIMGDataset(Dataset):
    def __init__(self, data_dir, split, graph_files=None, normalizer=None, random_seed=123, use_dmpnn=False):

        if split == 'train':
            id_prop_file = os.path.join(data_dir, 'train_id_prop.csv')
        elif split == 'val':
            id_prop_file = os.path.join(data_dir, 'val_id_prop.csv')
        elif split == 'test':
            id_prop_file = os.path.join(data_dir, 'test_id_prop.csv')
        elif split == 'testt':
            id_prop_file = os.path.join(data_dir, 'testt_id_prop.csv')
        else:
            id_prop_file = os.path.join(data_dir, 'id_prop.csv')
        split_df = pd.read_csv(id_prop_file, header=None, names=["id", "target", "graph_level_att"])
        split_df["id"] = split_df["id"].astype(str)
        split_ids = set(split_df["id"].tolist())
        
        self.graphs = []
        graph_maker = QTAIMGraph(data_dir, random_seed)
        graph_data = graph_maker.get_graph_info()
        
        filtered_data = [(cod_id, target_val) for cod_id, target_val in 
                         zip(graph_data["data_list"], graph_data["target_list"]) 
                         if cod_id in split_ids]
        for cod_id, target_val in filtered_data:
            data = graph_maker.get(cod_id)
            data.y = torch.tensor([[target_val]], dtype=torch.float)  # Shape: [1, 1] to match model output
            data.id = cod_id
            self.graphs.append(data)
            
        self.use_dmpnn = use_dmpnn
        
        if len(self.graphs) == 0:
            raise ValueError(f"No graphs were loaded for {split} split. Check that the data files exist and contain valid IDs.")
            
        if normalizer is None:
            # Separate one-hot encoded atomic features (first 119 columns) from continuous features
            all_edge_features = torch.cat([g.edge_attr for g in self.graphs], dim=0)
            all_edge_weights = torch.cat([g.edge_weight for g in self.graphs], dim=0)
            all_targets = torch.cat([g.y for g in self.graphs], dim=0)
            all_global_values = torch.cat([g.u for g in self.graphs], dim=0)
            
            # Get continuous node features (if any exist after the one-hot encoding)
            # The first 119 columns are one-hot encoded atomic features, remaining are continuous QTAIM features
            if self.graphs[0].x.shape[1] > 119:  # If there are features beyond one-hot encoding
                all_continuous_node_features = torch.cat([g.x[:, 119:] for g in self.graphs], dim=0)
                self.node_normalizer = Normalizer(all_continuous_node_features)
            else:
                self.node_normalizer = None

            self.edge_normalizer = Normalizer(all_edge_features)
            self.edge_w_normalizer = Normalizer(all_edge_weights)
            self.target_normalizer = Normalizer(all_targets)
            self.global_value_normalizer = Normalizer(all_global_values)
        else:
            self.node_normalizer = normalizer.get("node", None)
            self.edge_normalizer = normalizer.get("edge", None)
            self.edge_w_normalizer = normalizer["edge_w"]
            self.target_normalizer = normalizer["target"]
            self.global_value_normalizer = normalizer["u"]

        for g in self.graphs:
            if self.node_normalizer is not None and g.x.shape[1] > 119:
                g.x = torch.cat([
                    g.x[:, :119],
                    self.node_normalizer.norm(g.x[:, 119:])
                ], dim=1)
            
            g.edge_attr = self.edge_normalizer.norm(g.edge_attr)
            g.edge_weight = self.edge_w_normalizer.norm(g.edge_weight)
            g.y = self.target_normalizer.norm(g.y)
            g.u = self.global_value_normalizer.norm(g.u)

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        if self.use_dmpnn:
            return RevIndexedData(self.graphs[idx])
        else:
            return self.graphs[idx]
    
    def get_normalizer(self):
        normalizer = {
            "node": self.node_normalizer,
            "edge": self.edge_normalizer,
            "edge_w": self.edge_w_normalizer,
            "target": self.target_normalizer,
            "u": self.global_value_normalizer
        }
        return normalizer
    
def create_splits(data_dir, split_file_dir=None, test_size=0.15, val_size=0.15, random_seed=42):
    if split_file_dir is None:
        split_file_dir = data_dir
    else:
        os.makedirs(split_file_dir, exist_ok=True)
        
    train_file = os.path.join(split_file_dir, 'train_id_prop.csv')
    test_file = os.path.join(split_file_dir, 'test_id_prop.csv' if val_size is not None else 'testt_id_prop.csv')
    val_file = os.path.join(split_file_dir, 'val_id_prop.csv') if val_size is not None else None
    
    split_type = "train/val/test" if val_size is not None else "train/test"
    print(f"Creating {split_type} splits...")
    
    id_prop_file = os.path.join(data_dir, 'id_prop.csv')
    if not os.path.exists(id_prop_file):
        raise FileNotFoundError(f"id_prop.csv file not found: {id_prop_file}")
        
    id_prop_df = pd.read_csv(id_prop_file, header=None, names=["id", "target", "graph_level_att"])
    id_prop_df["id"] = id_prop_df["id"].astype(str)
    all_ids = id_prop_df["id"].tolist()
    print(f"Total samples in id_prop.csv: {len(all_ids)}")
    qtaim_files = [f.split('_qtaim_graph.gml')[0] for f in os.listdir(data_dir) if f.endswith('_qtaim_graph.gml')]
    valid_ids = [id for id in all_ids if id in qtaim_files]
    
    if not valid_ids:
        raise ValueError("No matching QTAIM graph files found for the IDs in id_prop.csv")
    
    if val_size is not None:
        train_ids, temp_ids = train_test_split(valid_ids, test_size=test_size + val_size, random_state=random_seed)
        val_ids, test_ids = train_test_split(temp_ids, test_size=test_size / (test_size + val_size), random_state=random_seed)
        print(f"Train: {len(train_ids)}, Validation: {len(val_ids)}, Test: {len(test_ids)}")
    else:
        train_ids, test_ids = train_test_split(valid_ids, test_size=test_size, random_state=random_seed)
        val_ids = []
        print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")
    
    train_df = id_prop_df[id_prop_df["id"].isin(train_ids)]
    test_df = id_prop_df[id_prop_df["id"].isin(test_ids)]
    
    train_df.to_csv(train_file, header=False, index=False)
    test_df.to_csv(test_file, header=False, index=False)
    
    split_files = {
        "train": train_file,
        "test": test_file
    }
    
    if val_size is not None:
        val_df = id_prop_df[id_prop_df["id"].isin(val_ids)]
        val_df.to_csv(val_file, header=False, index=False)
        split_files["val"] = val_file
        print(f"Split files created: {train_file}, {val_file}, {test_file}")
    else:
        print(f"Split files created: {train_file}, {test_file}")
    
    return split_files

def load_datasets(data_dir, splits=['train', 'val', 'test'], normalizer=None, random_seed=42, use_dmpnn=False):
    datasets = {}
    
    if 'train' in splits:
        print("Loading training set...")
        train_dataset = QTAIMGDataset(
            data_dir=data_dir, 
            split='train',
            normalizer=normalizer,
            random_seed=random_seed,
            use_dmpnn=use_dmpnn
        )
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty after loading!")
            
        print(f"Successfully loaded {len(train_dataset)} training samples")
        if normalizer is None:
            normalizer = train_dataset.get_normalizer()
        datasets['train'] = train_dataset
    
    for split in splits:
        if split != 'train' and split in ['val', 'test', 'testt']:
            print(f"Loading {split} set...")
            dataset = QTAIMGDataset(
                data_dir=data_dir, 
                split=split,
                normalizer=normalizer,
                random_seed=random_seed,
                use_dmpnn=use_dmpnn
            )
            print(f"Successfully loaded {len(dataset)} {split} samples")
            datasets[split] = dataset
    
    return datasets, normalizer

def create_dataloaders(datasets, batch_size=8, shuffle_train=True):
    dataloaders = {}
    
    for split, dataset in datasets.items():
        shuffle = shuffle_train and split == 'train'
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloaders

def load_full_dataset(data_dir, batch_size=8, normalizer=None, random_seed=42, use_dmpnn=False):
    print("Loading full dataset...")
    dataset = QTAIMGDataset(
        data_dir, 
        split='all', 
        normalizer=normalizer,
        random_seed=random_seed, 
        use_dmpnn=use_dmpnn
    )
    
    if normalizer is None:
        normalizer = dataset.get_normalizer()
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return data_loader, dataset, normalizer

class Normalizer(object):
    def __init__(self, tensor, eps=1e-6):
        self.mean = torch.mean(tensor, dim=0, keepdim=True)
        self.std = torch.std(tensor, dim=0, keepdim=True) + eps

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self
    
    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]

