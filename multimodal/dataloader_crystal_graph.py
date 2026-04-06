import os
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

class CrystalGraphDataset(Dataset):
    def __init__(self, graph_files, normalizer=None):
        super().__init__()
        self.graph_files = graph_files
        
        self.graphs = []
        for file_path in graph_files:
            data = torch.load(file_path)
            self.graphs.append(data)

        for g in self.graphs:
            if torch.isnan(g.u).any():
                import numpy as np
                g.u = torch.Tensor(np.zeros((3))[np.newaxis, ...])
        if normalizer is None:
            all_edge_attr = torch.cat([g.edge_attr for g in self.graphs], dim=0)
            all_y = torch.cat([g.y for g in self.graphs], dim=0)
            all_u = torch.cat([g.u for g in self.graphs], dim=0)
            all_global_feature = torch.cat([g.global_feature for g in self.graphs], dim=0)
            
            self.edge_normalizer = Normalizer(all_edge_attr)
            self.target_normalizer = Normalizer(all_y)
            self.u_normalizer = Normalizer(all_u)
            self.global_feature_normalizer = Normalizer(all_global_feature)
        else:
            self.edge_normalizer = normalizer["edge"]
            self.target_normalizer = normalizer["target"]
            self.u_normalizer = normalizer["u"]
            self.global_feature_normalizer = normalizer["global_feature"]
        for g in self.graphs:
            g.edge_attr = self.edge_normalizer.norm(g.edge_attr)
            g.y = self.target_normalizer.norm(g.y)
            g.u = self.u_normalizer.norm(g.u)
            g.global_feature = self.global_feature_normalizer.norm(g.global_feature)
    
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        data.x = data.x.to(torch.float32)
        data.u = data.u.to(torch.float32)
        data.edge_attr = data.edge_attr.to(torch.float32)
        data.edge_weight = data.edge_weight.to(torch.float32)
        data.global_feature = data.global_feature.to(torch.float32)
        return data

def collect_graph_files(directory: str) -> list:
    """Collect all .pt graph files in the given directory."""
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) 
                  if f.endswith('.pt')])

def split_crystal_datasets(graph_dir, batch_size=8, test_size=0.15, val_size=0.15, random_seed=42):
    all_files = collect_graph_files(graph_dir)
    
    train_files, temp_files = train_test_split(
        all_files, test_size=test_size + val_size, random_state=random_seed
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=test_size / (test_size + val_size), random_state=random_seed
    )
    
    train_dataset = CrystalGraphDataset(train_files)
    normalizer = {
        "edge": train_dataset.edge_normalizer,
        "target": train_dataset.target_normalizer,
        "u": train_dataset.u_normalizer,
        "global_feature": train_dataset.global_feature_normalizer
    }
    
    val_dataset = CrystalGraphDataset(val_files, normalizer=normalizer)
    test_dataset = CrystalGraphDataset(test_files, normalizer=normalizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, test_dataset, normalizer

def split_crystal_datasets_tt(graph_dir, batch_size=8, test_size=0.20, random_seed=42):
    all_files = collect_graph_files(graph_dir)
    
    train_files, test_files = train_test_split(
        all_files, test_size=test_size, random_state=random_seed
    )
    
    train_dataset = CrystalGraphDataset(train_files)
    normalizer = {
        "edge": train_dataset.edge_normalizer,
        "target": train_dataset.target_normalizer,
        "u": train_dataset.u_normalizer,
        "global_feature": train_dataset.global_feature_normalizer
    }
    
    test_dataset = CrystalGraphDataset(test_files, normalizer=normalizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_full_dataset(graph_dir, batch_size=8, normalizer=None):
    all_files = collect_graph_files(graph_dir)
    dataset = CrystalGraphDataset(all_files, normalizer=normalizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader, dataset

class Normalizer:
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

def load_crystal_datasets_from_splits(graph_dir, splits_dir, batch_size=8, 
                                     splits=['train', 'val', 'test']):
    all_graph_files = collect_graph_files(graph_dir)
    available_ids = set([os.path.basename(f).replace('.pt', '') for f in all_graph_files])
    
    datasets = {}
    split_files = {}
    
    split_file_mapping = {
        'train': 'train_id_prop.csv',
        'val': 'val_id_prop.csv', 
        'test': 'test_id_prop.csv',
        'testt': 'testt_id_prop.csv'
    }
    
    for split in splits:
        split_file = os.path.join(splits_dir, split_file_mapping[split])
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        df = pd.read_csv(split_file, header=None, names=['id', 'target', 'extra'] if split_file_mapping[split] else ['id', 'target'])
        df['id'] = df['id'].astype(str)
        
        valid_ids = df[df['id'].isin(available_ids)]['id'].tolist()
        
        if not valid_ids:
            raise ValueError(f"No valid graph files found for {split} split. "
                           f"Check that graph files exist in {graph_dir} for IDs in {split_file}")
        
        split_graph_files = [os.path.join(graph_dir, f"{id}.pt") for id in valid_ids]
        split_files[split] = split_graph_files
        
        print(f"Loaded {len(valid_ids)} {split} samples from {split_file}")
    
    normalizer = None
    if 'train' in splits:
        print("Creating training dataset...")
        train_dataset = CrystalGraphDataset(split_files['train'])
        datasets['train'] = train_dataset
        
        normalizer = {
            "edge": train_dataset.edge_normalizer,
            "target": train_dataset.target_normalizer,
            "u": train_dataset.u_normalizer,
            "global_feature": train_dataset.global_feature_normalizer
        }
        print(f"Training dataset created with {len(train_dataset)} samples")
    
    for split in splits:
        if split != 'train':
            print(f"Creating {split} dataset...")
            dataset = CrystalGraphDataset(split_files[split], normalizer=normalizer)
            datasets[split] = dataset
            print(f"{split.capitalize()} dataset created with {len(dataset)} samples")
    
    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True) if 'train' in datasets else None
    val_loader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=False) if 'val' in datasets else None
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False) if 'test' in datasets else None
    testt_loader = DataLoader(datasets['testt'], batch_size=batch_size, shuffle=False) if 'testt' in datasets else None
    
    test_dataset = datasets.get('test', datasets.get('testt', None))
    
    if 'val' in splits and 'test' in splits:
        return train_loader, val_loader, test_loader, test_dataset, normalizer
    elif 'testt' in splits:
        return train_loader, testt_loader, normalizer
    else:
        return train_loader, val_loader, test_loader, test_dataset, normalizer


def create_crystal_dataloaders(datasets, batch_size=8, shuffle_train=True):
    dataloaders = {}
    
    for split, dataset in datasets.items():
        shuffle = shuffle_train and split == 'train'
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        print(f"Created {split} dataloader with batch_size={batch_size}, shuffle={shuffle}")
    
    return dataloaders 