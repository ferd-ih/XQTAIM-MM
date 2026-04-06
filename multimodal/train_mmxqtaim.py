import argparse
import sys
import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from types import SimpleNamespace
from models.xqtaim_multimodal_fusion import Gated_DGL_MM_Fusion
from graph_data_xqtaim import QTAIMGraph
import shutil
from multimodal_xg import MultimodalCrystalQTAIMDataset, multimodal_collate_fn, create_multimodal_normalizers
from dataloader_xqtaim import create_splits
from utils import *
import random
from collections import defaultdict

def split_id_props(id_props_path, split_file_dir=None, test_size=0.15, val_size=0.15, random_seed=42):
    return create_splits(id_props_path, split_file_dir=split_file_dir, test_size=test_size, val_size=val_size, random_seed=random_seed)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=0.001,
    weight_decay=0.0,
    grad_clip=10.0,
    device='cpu',
    patience=20,
    return_history=False,
    scheduler_type='plateau',
    scheduler_factor=0.5,
    scheduler_patience=5,
    scheduler_min_lr=1e-6,
    cosine_T0=10,
    cosine_Tmult=2,
    debug_val=False,
    val_normalizer=None,
    unimodal_reg_weight=0.1,
    enc_lr=0.0007,
    crystal_enc_lr=None,
    qtaim_enc_lr=None,
    checkpoint_path=None,

):
    model = model.to(device)
    
    c_params = list(model.crystal_encoder.parameters())
    q_params = list(model.qtaim_encoder.parameters())

    fusion_params = [
        p for n, p in model.named_parameters()
        if not n.startswith('crystal_encoder.')
        and not n.startswith('qtaim_encoder.')
    ]
    
    crystal_lr = crystal_enc_lr if crystal_enc_lr is not None else enc_lr
    qtaim_lr = qtaim_enc_lr if qtaim_enc_lr is not None else enc_lr
    
    optimizer = optim.Adam([
        {"params": c_params, "lr": crystal_lr},
        {"params": q_params, "lr": qtaim_lr },
        {"params": fusion_params, "lr": lr, "weight_decay": weight_decay},
    ])
    
    print(f"Learning rates - Crystal encoder: {crystal_lr}, QTAIM encoder: {qtaim_lr}, Fusion: {lr}")
    loss_fn = nn.MSELoss()
    mm_loss_fn = loss_fn
    schedulers = []
    if scheduler_type == 'plateau':
        temp_opt = optim.Adam([optimizer.param_groups[2]], lr=optimizer.param_groups[2]['lr'])
        schedulers.append(optim.lr_scheduler.ReduceLROnPlateau(
            temp_opt, mode='min', factor=scheduler_factor, patience=scheduler_patience,
            min_lr=scheduler_min_lr
        ))
    elif scheduler_type == 'cosine':
        temp_opt = optim.Adam([optimizer.param_groups[2]], lr=optimizer.param_groups[2]['lr'])
        schedulers.append(optim.lr_scheduler.CosineAnnealingWarmRestarts(
            temp_opt, T_0=cosine_T0, T_mult=cosine_Tmult
        ))
    else:
        schedulers = None
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_state_dict = None
    start_epoch = 0
    
    print(f"Starting training with {sum(p.numel() for p in model.parameters())} parameters")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        best_epoch = checkpoint.get('best_epoch', 0)
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Resuming training from epoch {start_epoch + 1}")
        print(f"Previous best validation loss: {best_val_loss:.4f}")
    else:
        start_epoch = 0
    
    
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        train_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            crystal_data = batch['crystal'].to(device)
            qtaim_data = batch['qtaim'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            out = model(crystal_data, qtaim_data)
            targets = targets.view(-1, 1).to(out.dtype)
            
            if check_nan_inf(out, "model train output", batch_idx):
                continue
            if check_nan_inf(targets, "targets train", batch_idx):
                continue

            multimodal_loss = mm_loss_fn(out, targets)
            unimodal_loss = 0.0
            crystal_unimodal_loss, qtaim_unimodal_loss = 0.0, 0.0
            if unimodal_reg_weight > 0:
                c_pred, q_pred = model.get_unimodal_predictions()
                crystal_unimodal_loss = loss_fn(c_pred, targets)
                qtaim_unimodal_loss   = loss_fn(q_pred, targets)
                unimodal_loss = unimodal_reg_weight * (crystal_unimodal_loss + qtaim_unimodal_loss)

                unimodal_loss.backward(retain_graph=True)

                for name, param in model.named_parameters():
                    if "fusion_initial" in name or "fusion_final" in name:
                        param.grad = None

            multimodal_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            if batch_idx == len(train_loader) - 1:
                if unimodal_reg_weight > 0:
                    print(f"Crystal unimodal: {crystal_unimodal_loss.item():.4f}, "
                        f"QTAIM unimodal: {qtaim_unimodal_loss.item():.4f}")
                print(f"Multimodal: {multimodal_loss.item():.4f}")
            total_loss += multimodal_loss.item() * targets.size(0)
            train_count += targets.size(0)
        
        model.eval()
        total_val_loss = 0
        val_count = 0
        with torch.no_grad():
            val_preds_denorm = []
            val_targets_denorm = []
            for batch_idx, batch in enumerate(val_loader):
                crystal_data = batch['crystal'].to(device)
                qtaim_data = batch['qtaim'].to(device)
                targets = batch['targets'].to(device)

                out = model(crystal_data, qtaim_data)
                targets = targets.view(-1, 1).to(out.dtype)
                if check_nan_inf(out, "model val output", batch_idx):
                    continue
                if check_nan_inf(targets, "targets val", batch_idx):
                    continue

                val_loss = mm_loss_fn(out, targets)
                total_val_loss += val_loss.item() * targets.size(0)
                val_count += targets.size(0)
                if debug_val and val_normalizer is not None and "target" in val_normalizer:
                    pred_denorm = val_normalizer["target"].denorm(out.detach().cpu())
                    target_denorm = val_normalizer["target"].denorm(targets.detach().cpu())
                    val_preds_denorm.extend(pred_denorm.numpy())
                    val_targets_denorm.extend(target_denorm.numpy())
        
        avg_train_loss = total_loss / max(train_count, 1)
        if val_count == 0:
            print("No valid validation batches this epoch; skipping early stopping update.")
            continue
        avg_val_loss = total_val_loss / val_count
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"Epoch {epoch+1}: Train {avg_train_loss:.4f}, Val {avg_val_loss:.4f} (Best)")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: Train {avg_train_loss:.4f}, Val {avg_val_loss:.4f}")

        if checkpoint_path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
            }, checkpoint_path)
        
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
        if schedulers is not None:
            for scheduler in schedulers:
                if scheduler_type == 'plateau':
                    scheduler.step(avg_val_loss)
                elif scheduler_type == 'cosine':
                    scheduler.step(epoch + 1)
                else:
                    scheduler.step()
                optimizer.param_groups[2]['lr'] = scheduler.optimizer.param_groups[0]['lr']
            crystal_lr = optimizer.param_groups[0]['lr']
            qtaim_lr = optimizer.param_groups[1]['lr']
            fusion_lr = optimizer.param_groups[2]['lr']
            print(f"LRs - Crystal encoder: {crystal_lr:.6f}, QTAIM encoder: {qtaim_lr:.6f}, Fusion: {fusion_lr:.6f}")
        
        if debug_val and val_normalizer is not None and len(val_preds_denorm) > 0:
            v_targets = np.array(val_targets_denorm).reshape(-1)
            v_preds = np.array(val_preds_denorm).reshape(-1)
            finite_mask = np.isfinite(v_targets) & np.isfinite(v_preds)
            v_targets = v_targets[finite_mask]
            v_preds = v_preds[finite_mask]
            if v_targets.size > 0:
                v_mse = np.mean((v_preds - v_targets) ** 2)
                v_mae = np.mean(np.abs(v_preds - v_targets))
                v_rmse = np.sqrt(v_mse)
                v_r2 = r2_score(v_targets, v_preds)
                print(f"[Val Debug] Denorm MSE={v_mse:.4f} MAE={v_mae:.4f} RMSE={v_rmse:.4f} R²={v_r2:.4f}")
    
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    plot_training_curves(train_losses, val_losses, best_epoch)
    
    if return_history:
        history = {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        return model, history
    return model


def evaluate_model(model, test_loader, device='cpu', normalizer=None, debug_eval=False, debug_samples=5):
    model.eval()
    predictions = []
    targets = []
    mm_loss_fn = nn.MSELoss()
    total_loss = 0
    
    with torch.no_grad():
        printed_header = False
        for batch_idx, batch in enumerate(test_loader):
            crystal_data = batch['crystal'].to(device)
            qtaim_data = batch['qtaim'].to(device)
            batch_targets = batch['targets'].to(device)
            
            if debug_eval and not printed_header:
                print(f"[Eval Debug] Using pure multimodal model (both modalities active)")
                printed_header = True

            out = model(crystal_data, qtaim_data)
            batch_targets = batch_targets.view(-1, 1).to(out.dtype)

            if check_nan_inf(out, "model test output", batch_idx):
                continue
            if check_nan_inf(batch_targets, "targets test", batch_idx):
                continue
            
            loss = mm_loss_fn(out, batch_targets)
            total_loss += loss.item() * batch_targets.size(0)
            
            if normalizer and "target" in normalizer:
                pred_denorm = normalizer["target"].denorm(out.cpu())
                target_denorm = normalizer["target"].denorm(batch_targets.cpu())
                predictions.extend(pred_denorm.numpy())
                targets.extend(target_denorm.numpy())
                if debug_eval and batch_idx == 0:
                    p_norm = out.detach().cpu().numpy().reshape(-1)
                    t_norm = batch_targets.detach().cpu().numpy().reshape(-1)
                    p_den = pred_denorm.numpy().reshape(-1)
                    t_den = target_denorm.numpy().reshape(-1)
                    n = min(debug_samples, p_norm.shape[0])
                    print("[Eval Debug] First samples (norm -> denorm):")
                    for i in range(n):
                        print(f"  {i}: pred {p_norm[i]:.4f} -> {p_den[i]:.4f} | target {t_norm[i]:.4f} -> {t_den[i]:.4f}")
            else:
                predictions.extend(out.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
    
    targets = np.array(targets).reshape(-1)
    predictions = np.array(predictions).reshape(-1)
    
    finite_mask = np.isfinite(targets) & np.isfinite(predictions)
    targets = targets[finite_mask]
    predictions = predictions[finite_mask]
    

    if targets.size == 0 or predictions.size == 0:
        print("\nNo valid test samples after filtering non-finite outputs. Skipping metrics.")
        return None

    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    avg_test_loss = total_loss / max(len(test_loader.dataset), 1)
    
    print(f"\nFinal Test Results:")
    print(f"  Test Loss (normalized): {avg_test_loss:.4f}")
    print(f"  Test MSE: {mse:.4f}")
    print(f"  Test MAE: {mae:.4f}")
    print(f"  Test RMSE: {rmse:.4f}")
    print(f"  Test R²: {r2:.4f}")
    plot_parity_plot(targets, predictions)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'test_loss': avg_test_loss,
        'predictions': predictions,
        'targets': targets
    }


def main():
    parser = argparse.ArgumentParser(description='Train multimodal fusion model')
    
    parser.add_argument('--config', type=str, default=None, help='Path to configuration JSON file')
    parser.add_argument('--crystal_dir', type=str, default=None, help='Path to crystal graph data (overrides config)')
    parser.add_argument('--qtaim_dir', type=str, default=None, help='Path to QTAIM data (overrides config)')
    parser.add_argument('--split_file_dir', type=str, default=None, help='Path to split files (overrides config)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides config)')
    parser.add_argument('--enc_lr', type=float, default=None, help='Encoder learning rate (overrides config, used as fallback if individual LRs not set)')
    parser.add_argument('--crystal_enc_lr', type=float, default=None, help='Crystal encoder learning rate (overrides config)')
    parser.add_argument('--qtaim_enc_lr', type=float, default=None, help='QTAIM encoder learning rate (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    parser.add_argument('--patience', type=int, default=None, help='Early stopping patience (overrides config)')
    parser.add_argument('--debug_val', action='store_true', help='Print detailed validation metrics and masks each epoch')
    parser.add_argument('--debug_eval', action='store_true', help='Print detailed evaluation metrics, masks, and samples')
    parser.add_argument('--debug_samples', type=int, default=5, help='Number of sample predictions to print in debug eval')
    parser.add_argument('--gradient_truncation', action='store_true', help='Enable gradient truncation for expressiveness')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to resume training from')
    parser.add_argument('--include_node_descriptors', type=str, default=None, 
                        help='Comma-separated list of QTAIM node descriptors to include (e.g., "e_density,Lagrangian_K"). Use "none" for element features only.')
    parser.add_argument('--exclude_node_descriptors', type=str, default=None,
                        help='Comma-separated list of QTAIM node descriptors to exclude')
    parser.add_argument('--include_edge_descriptors', type=str, default=None,
                        help='Comma-separated list of QTAIM edge descriptors to include (e.g., "bond_distance,e_density")')
    parser.add_argument('--exclude_edge_descriptors', type=str, default=None,
                        help='Comma-separated list of QTAIM edge descriptors to exclude')
    parser.add_argument('--list_descriptors', action='store_true',
                        help='List available edge descriptors and exit')
    args = parser.parse_args()
    
    if args.config:
        config = load_config(args.config)
    else:
        print("No config file provided, using default configuration")
        config = create_default_config()
    
    config = config_overwrite(config, args)
    
    if config.crystal_dir is None:
        raise ValueError("crystal_dir must be provided either in config file or as command line argument")
    if config.qtaim_dir is None:
        raise ValueError("qtaim_dir must be provided either in config file or as command line argument")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    if device.type == 'cuda':
        print(f"Device info: {torch.cuda.get_device_name(device)}")

    if args.list_descriptors:
        print("\nAvailable QTAIM edge descriptors:")
        for desc in QTAIMGraph.get_available_edge_descriptors():
            print(f"  - {desc}")
        print("\nTo discover node descriptors, use QTAIMGraph.get_available_node_descriptors_from_gml(path_to_gml)")
        print("\nExample usage:")
        print("  --include_edge_descriptors 'bond_distance,e_density'")
        print("  --exclude_node_descriptors 'esp_nuc,iri'")
        print("  --include_node_descriptors 'none'  # Use only element features")
        return

    def parse_descriptor_list(arg_value):
        if arg_value is None:
            return None
        if arg_value.lower() == 'none':
            return []
        return [s.strip() for s in arg_value.split(',') if s.strip()]
    
    include_node_descriptors = parse_descriptor_list(args.include_node_descriptors)
    exclude_node_descriptors = parse_descriptor_list(args.exclude_node_descriptors)
    include_edge_descriptors = parse_descriptor_list(args.include_edge_descriptors)
    exclude_edge_descriptors = parse_descriptor_list(args.exclude_edge_descriptors)
    
    if any([include_node_descriptors is not None, exclude_node_descriptors, 
            include_edge_descriptors is not None, exclude_edge_descriptors]):
        print("\n" + "="*60)
        print("ABLATION STUDY: Descriptor Masking Configuration")
        print("="*60)
        if include_node_descriptors is not None:
            print(f"  Include node descriptors: {include_node_descriptors if include_node_descriptors else 'NONE (element features only)'}")
        if exclude_node_descriptors:
            print(f"  Exclude node descriptors: {exclude_node_descriptors}")
        if include_edge_descriptors is not None:
            print(f"  Include edge descriptors: {include_edge_descriptors if include_edge_descriptors else 'NONE'}")
        if exclude_edge_descriptors:
            print(f"  Exclude edge descriptors: {exclude_edge_descriptors}")
        print("="*60 + "\n")


    def create_model(crystal_data, qtaim_data, config):
        return Gated_DGL_MM_Fusion(
            crystal_data=crystal_data,
            qtaim_data=qtaim_data,
            hidden_size=config.hidden_size,
            fusion_size=config.fusion_size,
            dropout=config.dropout,
            adaptive_temp=config.adaptive_temp,
            gradient_truncation=args.gradient_truncation,
            schnet_dim1=config.schnet_dim1,
            schnet_dim2=config.schnet_dim2,
            schnet_dim3=config.schnet_dim3,
            schnet_cutoff=config.schnet_cutoff,
            schnet_pre_fc_count=config.schnet_pre_fc_count,
            schnet_gc_count=config.schnet_gc_count,
            schnet_post_fc_count=config.schnet_post_fc_count,
            schnet_num_gaussians=config.schnet_num_gaussians,
            schnet_use_edge_descriptors=config.schnet_use_edge_descriptors,
            schnet_pool=config.schnet_pool,
            schnet_pool_order=config.schnet_pool_order,
            schnet_batch_norm=config.schnet_batch_norm,
            schnet_act=config.schnet_act,
            schnet_dropout_rate=config.schnet_dropout_rate,
        )
    
    train_config_dict = {
        'epochs': config.epochs,
        'lr': config.lr,
        'weight_decay': config.weight_decay,
        'grad_clip': config.grad_clip,
        'patience': config.patience,
        'scheduler_type': config.scheduler,
        'scheduler_factor': config.scheduler_factor,
        'scheduler_patience': config.scheduler_patience,
        'scheduler_min_lr': config.scheduler_min_lr,
        'cosine_T0': config.cosine_T0,
        'cosine_Tmult': config.cosine_Tmult,
        'debug_val': args.debug_val,
        'unimodal_reg_weight': config.unimodal_reg_weight,
        'enc_lr': getattr(config, 'enc_lr', 0.0007),
        'crystal_enc_lr': getattr(config, 'crystal_enc_lr', None),
        'qtaim_enc_lr': getattr(config, 'qtaim_enc_lr', None),
        'checkpoint_path': args.checkpoint,
    }
    
    seed = args.seed
    
    split_files = split_id_props(config.qtaim_dir,
                                split_file_dir=config.split_file_dir,
                                test_size=0.15, 
                                val_size=0.15, 
                                random_seed=seed)
    seed_everything(seed)
    print(f"Seed: {seed}")
    
    crystal_normalizer, qtaim_normalizer = create_multimodal_normalizers(
        config.crystal_dir, config.qtaim_dir, split_file_dir=config.split_file_dir, random_seed=seed,
        normalize_edge_weight=False,
        use_dmpnn=False,
        include_node_descriptors=include_node_descriptors,
        exclude_node_descriptors=exclude_node_descriptors,
        include_edge_descriptors=include_edge_descriptors,
        exclude_edge_descriptors=exclude_edge_descriptors,
    )
    train_dataset = MultimodalCrystalQTAIMDataset(
        config.crystal_dir, config.qtaim_dir, split='train', split_file_dir=config.split_file_dir, 
        crystal_normalizer=crystal_normalizer, qtaim_normalizer=qtaim_normalizer,
        normalize_edge_weight=False, use_dmpnn=False,
        include_node_descriptors=include_node_descriptors,
        exclude_node_descriptors=exclude_node_descriptors,
        include_edge_descriptors=include_edge_descriptors,
        exclude_edge_descriptors=exclude_edge_descriptors,
    )
    val_dataset = MultimodalCrystalQTAIMDataset(
        config.crystal_dir, config.qtaim_dir, split='val', split_file_dir=config.split_file_dir, 
        crystal_normalizer=crystal_normalizer, qtaim_normalizer=qtaim_normalizer,
        normalize_edge_weight=False, use_dmpnn=False,
        include_node_descriptors=include_node_descriptors,
        exclude_node_descriptors=exclude_node_descriptors,
        include_edge_descriptors=include_edge_descriptors,
        exclude_edge_descriptors=exclude_edge_descriptors,
    )
    test_dataset = MultimodalCrystalQTAIMDataset(
        config.crystal_dir, config.qtaim_dir, split='test', split_file_dir=config.split_file_dir, 
        crystal_normalizer=crystal_normalizer, qtaim_normalizer=qtaim_normalizer,
        normalize_edge_weight=False, use_dmpnn=False,
        include_node_descriptors=include_node_descriptors,
        exclude_node_descriptors=exclude_node_descriptors,
        include_edge_descriptors=include_edge_descriptors,
        exclude_edge_descriptors=exclude_edge_descriptors,
    )

    train_loader, val_loader, test_loader = make_loaders(config.batch_size, seed, 
                                                        train_dataset, val_dataset, test_dataset, 
                                                        multimodal_collate_fn)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    sample = train_dataset[0]
    crystal_data = sample['crystal']
    qtaim_data = sample['qtaim']

    model = create_model(crystal_data, qtaim_data, config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Model config parameters: {config}")

    trained_model = train_model(
        model, train_loader, val_loader, 
        device=device,
        **train_config_dict
    )
    
    evaluate_model(
        trained_model,
        test_loader,
        device=device,
        normalizer=crystal_normalizer,
        debug_eval=args.debug_eval,
        debug_samples=args.debug_samples,
    )
    
    if 'SLURM_JOB_ID' in os.environ:
        model_save_path = f"best_model_multimodal_{os.environ['SLURM_JOB_ID']}.pt"
    else:
        model_save_path = "best_model_multimodal.pt"
    
    torch.save({
        'model': trained_model.state_dict(),
        'normalizers': {key: norm.state_dict() for key, norm in crystal_normalizer.items()},
    }, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    if 'SLURM_JOB_ID' in os.environ:
        os.makedirs(f'results/{os.environ["SLURM_JOB_ID"]}', exist_ok=True)
        shutil.move(f'multimodal_training_loss_{os.environ["SLURM_JOB_ID"]}.png', f'results/{os.environ["SLURM_JOB_ID"]}/multimodal_training_loss.png')
        shutil.move(f'multimodal_predictions_{os.environ["SLURM_JOB_ID"]}.png', f'results/{os.environ["SLURM_JOB_ID"]}/multimodal_predictions.png')
        shutil.move(model_save_path, f'results/{os.environ["SLURM_JOB_ID"]}/best_model_multimodal.pt')

if __name__ == "__main__":
    main()