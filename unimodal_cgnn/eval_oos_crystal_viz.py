import os
import torch
import argparse
import pandas as pd
from dataloader_crystal_graph import split_crystal_datasets, load_full_dataset, Normalizer
from models.cgcnn import CGCNN
from models.megnet import MEGNet
from models.mpnn import MPNN
from models.schnet import SchNet
from models.gatgnn import GATGNN
from models.gat import GATNet
from models.ginconv import GINConvNet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.cm as cm

def evaluate_model(model, test_loader, device='cpu', normalizer=None, model_name="model"):
    model.to(device)
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0
    predictions, true_values = [], []
    target_normalizer = normalizer["target"]
    target_normalizer = target_normalizer.to(device)

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item()
            output_denorm = target_normalizer.denorm(output)
            true_denorm = target_normalizer.denorm(data.y)
            predictions.extend(output_denorm.cpu().numpy())
            true_values.extend(true_denorm.cpu().numpy())


    avg_test_loss = total_loss / len(test_loader.dataset)
    print(f"Test Loss (normalized): {avg_test_loss:.4f}")

    predictions = np.array(predictions)
    targets = np.array(true_values)

    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    print(f"Denormalized MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Predictions vs True")
    plt.text(0.05, 0.95, f'R²: {r2:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}',
             transform=plt.gca().transAxes, ha='left', va='top')
    plt.savefig(f'predictions_oos_{model_name}.png')
    plt.close()
    errors = np.abs(np.array(true_values) - np.array(predictions))

    return avg_test_loss, predictions, true_values

def visualize_embeddings(model, data_loader, device='cpu', method='tsne', model_name="model"):
    model.eval()
    model = model.to(device)
    embeddings = []
    targets = []
    
    # Register a hook to get embeddings from the model
    embedding_container = []
    
    def hook_fn(module, input, output):
        embedding_container.append(output.detach().cpu().numpy())

    # print("Model architecture:")
    # for name, module in model.named_modules():
    #     print(f"  {name}") 
    #     if 'fc' in name and 'list' not in name:
    #         hook = module.register_forward_hook(hook_fn)
    #         print(f"Registered hook on {name}")
    #         found_hook = True
    #         break   

    hook = model.post_lin_list[-1].register_forward_hook(hook_fn)
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            _ = model(data)
            targets.extend(data.y.cpu().numpy())
    
    hook.remove()
    
    if not embedding_container:
        print("No embeddings were captured. The hook might not have been triggered correctly.")
        return
    
    try:
        all_embeddings = np.vstack(embedding_container)
        print(f"Extracted embeddings shape: {all_embeddings.shape}")
        
        if method == 'tsne':
            reduced_embeddings = TSNE(n_components=2, random_state=42).fit_transform(all_embeddings)
        else:
            reduced_embeddings = PCA(n_components=2, random_state=42).fit_transform(all_embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced_embeddings[:, 0], 
            reduced_embeddings[:, 1], 
            c=targets, 
            cmap=cm.viridis, 
            alpha=0.8,
            s=50
        )
        
        plt.colorbar(scatter, label='Band Gap (eV)')
        plt.xlabel(f'{method.upper()} Dimension 1')
        plt.ylabel(f'{method.upper()} Dimension 2')
        plt.title(f'Crystal Embeddings of CGCNN trained model ({method.upper()})')
        plt.tight_layout()
        plt.savefig(f'crystal_embeddings_{method}_{model_name}_omdb.png', dpi=300)
        plt.close()
        
        print(f"Embeddings visualization saved to crystal_embeddings_{method}_{model_name}_omdb.png")
    except Exception as e:
        print(f"Error during embedding visualization: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test a saved crystal graph model on a new dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model (.pth file).")
    parser.add_argument("--model_type", type=str, required=True, 
                        choices=["CGCNN", "MPNN", "MEGNet", "SchNet", "GATGNN", "GAT", "GINConv"],
                        help="Type of model architecture.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the crystal graph dataset to test on.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for testing.")
    args = parser.parse_args()

    saved_chk = torch.load(args.model_path, map_location="cpu")
    normalizer = {}
    for key, state in saved_chk['normalizers'].items():
        dummy = torch.empty_like(state['mean'])
        norm = Normalizer(dummy)
        norm.load_state_dict(state)
        normalizer[key] = norm
        
    print(f"Normalizer: {normalizer}")
    print(f"Loading crystal graph dataset from {args.data_dir}")
    

    data_loader, dataset = load_full_dataset(
        args.data_dir,
        batch_size=args.batch_size,
        normalizer=normalizer
    )

    # train_loader, val_loader, test_loader, _, _ = split_crystal_datasets(
    #     graph_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     test_size=0.15,
    #     val_size=0.15,
    #     random_seed=17
    # )

    print(f"Using full dataset: {len(dataset)} samples")

    first_batch = next(iter(data_loader))

    if args.model_type == "CGCNN":
        model = CGCNN(first_batch, dim1=64, dim2=128, pre_fc_count=1, gc_count=5, post_fc_count=2, pool="set2set", dropout_rate=0.06455285352866819)
    elif args.model_type == "MEGNet":
        model = MEGNet(first_batch, dim1=128, dim2=32, dim3=256, pre_fc_count=1, gc_count=4, gc_fc_count=2, post_fc_count=2, pool="global_max_pool", dropout_rate=0.08919207357951375)
    elif args.model_type == "MPNN":
        model = MPNN(first_batch, dim1=64, dim2=128, dim3=128, pre_fc_count=1, gc_count=2, gc_fc_count=2, post_fc_count=4, pool="global_max_pool", dropout_rate=0.114362)
    # elif args.model_type == "GCN":
    #     model = GCN(first_batch)
    elif args.model_type == "SchNet":
        model = SchNet(first_batch, dim1=256, dim2=128, dim3=64, pre_fc_count=1, gc_count=4, gc_fc_count=None, post_fc_count=4, pool="global_mean_pool", dropout_rate=0.3497866040911963)
    elif args.model_type == "GATGNN":
        model = GATGNN(first_batch, heads=3, neurons=256, nl=5, global_attention="composition", unpooling_technique="learnable", concat_comp=False)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    device = torch.device(args.device if torch.cuda.is_available()  else "cpu")
    if torch.cuda.is_available() and args.device == "cuda":
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Using GPU: {device_name} (ID: {current_device})")
    else:
        print("Using CPU.")

    print(f"Loading model weights from {args.model_path}")
    model.load_state_dict(saved_chk['model'])

    
    print(f"Testing crystal graph model...")
    evaluate_model(model, data_loader, device=device, normalizer=normalizer, model_name=args.model_type)
    
    
    # print("Visualizing embeddings with t-SNE...")
    # visualize_embeddings(model, data_loader, device=device, method='tsne', model_name=args.model_type)

if __name__ == "__main__":
    main() 