import json
import os
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from sklearn.metrics import r2_score
import torch
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded config from: {config_path}")
    return config

def config_overwrite(config, args):

    config_ns = SimpleNamespace(**config)
    params = ['crystal_dir', 'qtaim_dir', 'split_file_dir', 'lr', 'epochs', 'patience']
    
    for param in params:
        if hasattr(args, param) and getattr(args, param) is not None:
            setattr(config_ns, param, getattr(args, param))
            print(f"Overriding {param}: {getattr(config_ns, param)}")
    
    return config_ns

def create_default_config():
    return {
        "crystal_dir": None,
        "qtaim_dir": None,
        "split_file_dir": None,
        "batch_size": 32,
        "epochs": 10,
        "hidden_size": 300,
        "fusion_size": 512,
        "dropout": 0.1,
        "lr": 0.001,
        "patience": 20,
        "weight_decay": 0.0,
        "grad_clip": 10.0,
        "warm_start_epochs": 0,
        "warm_freeze_branch": "crystal",
        "log_adaptive_weights": False,
        "init_adaptive": "neutral",
        "adaptive_temp": 1.0,
        "entropy_reg": 0.0,
        "unfreeze_lr_factor": 0.5,
        "unfreeze_lr_epochs": 3,
        "scheduler": "plateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,
        "scheduler_min_lr": 1e-6,
        "cosine_T0": 10,
        "cosine_Tmult": 2
    }

def plot_training_curves(train_losses, val_losses, best_epoch, save_path=f'multimodal_training_loss_{os.environ["SLURM_JOB_ID"]}.png'):

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=best_epoch-1, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Multimodal Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_parity_plot(targets, predictions, save_path=f'multimodal_predictions_{os.environ["SLURM_JOB_ID"]}.png'):

    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.6, s=30)
    lo = float(min(targets.min(), predictions.min()))
    hi = float(max(targets.max(), predictions.max()))
    plt.plot([lo, hi], [lo, hi], 'r--', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Multimodal Model - Parity Plot: Prediction vs Target')
    plt.text(0.05, 0.95, f'R²: {r2:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}',
             ha='left', va='top', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mse, mae, rmse, r2

def analyze_data_scales(train_loader, device='cpu'):
    print("\n=== DATA SCALE ANALYSIS ===")
    
    crystal_stats = {"x": [], "edge_attr": [], "y": [], "u": [], "global_feature": []}
    qtaim_stats = {"x": [], "edge_attr": [], "y": [], "u": []}
    
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break
        crystal_data = batch['crystal'].to(device)
        qtaim_data = batch['qtaim'].to(device)
        
        crystal_stats["x"].append(crystal_data.x.mean().item())
        crystal_stats["edge_attr"].append(crystal_data.edge_attr.mean().item())
        crystal_stats["y"].append(crystal_data.y.mean().item())
        crystal_stats["u"].append(crystal_data.u.mean().item())
        crystal_stats["global_feature"].append(crystal_data.global_feature.mean().item())
        
        qtaim_stats["x"].append(qtaim_data.x.mean().item())
        qtaim_stats["edge_attr"].append(qtaim_data.edge_attr.mean().item())
        qtaim_stats["y"].append(qtaim_data.y.mean().item())
        qtaim_stats["u"].append(qtaim_data.u.mean().item())
    
    print("Crystal Graph Features:")
    for key, values in crystal_stats.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {key}: mean={mean_val:.6f}, std={std_val:.6f}")
    
    print("\nQTAIM Graph Features:")
    for key, values in qtaim_stats.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {key}: mean={mean_val:.6f}, std={std_val:.6f}")
    
    print("=" * 30)

def seed_everything(seed: int):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def make_seed_worker(seed):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return seed_worker

def make_loaders(batch_size: int, seed: int, 
                 train_dataset, val_dataset, test_dataset, 
                 multimodal_collate_fn,
                 num_workers: int = 4):
    g_train = torch.Generator().manual_seed(seed)
    g_val   = torch.Generator().manual_seed(seed + 1)
    g_test  = torch.Generator().manual_seed(seed + 2)

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, collate_fn=multimodal_collate_fn,
        generator=g_train, worker_init_fn=make_seed_worker(seed),
        num_workers=num_workers, drop_last=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size, shuffle=False, collate_fn=multimodal_collate_fn,
        generator=g_val, worker_init_fn=make_seed_worker(seed+1000),
        num_workers=num_workers, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size, shuffle=False, collate_fn=multimodal_collate_fn,
        generator=g_test, worker_init_fn=make_seed_worker(seed+2000),
        num_workers=num_workers, persistent_workers=True
    )
    return train_loader, val_loader, test_loader

def save_config(config, save_path):

    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    

def load_best_config(config_dir):

    best_config_path = os.path.join(config_dir, 'best_config.json')
    if os.path.exists(best_config_path):
        with open(best_config_path, 'r') as f:
            best_result = json.load(f)
        return best_result.get('config')
    return None

def create_input_stats_dicts():
    crystal_input_stats = {"x_mean": [], "x_std": [], "edge_attr_mean": [], "edge_attr_std": []}
    qtaim_input_stats = {"x_mean": [], "x_std": [], "edge_attr_mean": [], "edge_attr_std": []}
    return crystal_input_stats, qtaim_input_stats


def log_input_scale_statistics(epoch, crystal_input_stats, qtaim_input_stats, verbose=False):

    if verbose:
        print(f"Epoch {epoch+1} Input Scales:")
        print(f"  Crystal - x: mean={np.mean(crystal_input_stats['x_mean']):.4f}, std={np.mean(crystal_input_stats['x_std']):.4f}")
        print(f"  Crystal - edge_attr: mean={np.mean(crystal_input_stats['edge_attr_mean']):.4f}, std={np.mean(crystal_input_stats['edge_attr_std']):.4f}")
        print(f"  QTAIM - x: mean={np.mean(qtaim_input_stats['x_mean']):.4f}, std={np.mean(qtaim_input_stats['x_std']):.4f}")
        print(f"  QTAIM - edge_attr: mean={np.mean(qtaim_input_stats['edge_attr_mean']):.4f}, std={np.mean(qtaim_input_stats['edge_attr_std']):.4f}")

def clamp_input_features(crystal_data, qtaim_data, min_val=-10.0, max_val=10.0):

    crystal_data.edge_attr = torch.clamp(crystal_data.edge_attr, min=min_val, max=max_val)
    qtaim_data.x = torch.clamp(qtaim_data.x, min=min_val, max=max_val)
    qtaim_data.edge_attr = torch.clamp(qtaim_data.edge_attr, min=min_val, max=max_val)

def check_nan_inf(tensor, name, batch_idx=None):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        batch_info = f" at batch {batch_idx}" if batch_idx is not None else ""
        print(f"WARNING: {name} contains NaN/Inf{batch_info}")
        print(f"{name} stats: mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}, max={tensor.max().item():.6f}, min={tensor.min().item():.6f}")
        return True
    return False

