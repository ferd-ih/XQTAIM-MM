import os
import torch
import argparse
from torch.optim import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from dataloader_crystal_graph_v2 import split_crystal_datasets
# from x_graph_utils.dataloader_crystal_graph import split_crystal_datasets
# from x_graph_utils.dataloader_crystal_graph2 import split_crystal_datasets
from dataloader_crystal_graph import split_crystal_datasets, load_crystal_datasets_from_splits
from models.cgcnn import CGCNN
from models.megnet import MEGNet
from models.mpnn import MPNN
from models.schnet import SchNet
from models.gatgnn import GATGNN
from models.deep_gatgnn import DEEP_GATGNN
from models.gat import GATNet
from models.ginconv import GINConvNet
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from sklearn.metrics import r2_score
def train(model, train_loader, val_loader, normalizer, device, epochs=20, lr=0.001, patience=5, checkpoint_path=None, model_name="model",weight_decay=0.0):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    
    print(f"\nTraining Parameters:")
    print(f"Model: {model_name}")
    print(f"Learning Rate: {lr}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Epochs: {epochs}")
    print(f"Early Stopping Patience: {patience}")
    print(f"Batch Size: {next(iter(train_loader)).num_graphs}")
    print(f"Device: {device}")

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        total_train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            if torch.isnan(loss):
                print("NaN loss detected :C")
                return
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data)
                val_loss = criterion(output, data.y)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}", flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # torch.save(model.state_dict(), f"best_model_xgraph_{model_name}.pth")  # Save only the model state
        else:
            patience_counter += 1


        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, f"checkpoint_xgraph_{model_name}.pth")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    plot_loss(train_losses, val_losses, model_name)
    return model

def plot_loss(train_losses, val_losses, model_name):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs. Validation Loss")
    plt.grid()
    plt.savefig(f"training_curve_xgraph_{model_name}.png")

def test(model, test_loader, normalizer, device, save_csv=True, model_name="model"):
    # model.load_state_dict(torch.load(f"best_model_xgraph_{model_name}.pth"))
    model.to(device)
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0
    predictions, true_values = [], []

    target_normalizer = normalizer["target"].to(device)
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

    avg_test_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    # Calculate MAE
    mae = np.mean(np.abs(np.array(predictions) - np.array(true_values)))
    print(f"Test MAE: {mae:.4f}")
    
    # if save_csv:
    #     # df = pd.DataFrame({"True": true_values, "Predicted": predictions})
    #     df = pd.DataFrame(zip(true_values, predictions))
    #     df.to_csv(f"test_predictions_xgraph_{model_name}.csv", index=False, header=False)
    r2 = r2_score(true_values, predictions)
    rmse = np.sqrt(np.mean((np.array(true_values) - np.array(predictions)) ** 2))
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, predictions, alpha=0.5)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red', linewidth=2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f'Crystal Graphs{model_name} model - Parity Plot: Predicted vs DFT Band Gap (eV)')
    plt.text(0.05, 0.95, f'R2: {r2:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}',
             ha='left', va='top', transform=plt.gca().transAxes)
    plt.savefig(f"predictions_xgraph_{model_name}.png")
    plt.close()
    return mae
def visualize_embeddings(model, data_loader, device='cpu', method='tsne', model_name=None):
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
    if model_name == "GATGNN":
        hook = model.linear1.register_forward_hook(hook_fn)
    elif model_name == "DEEP_GATGNN":
        # DEEP_GATGNN uses post_lin_list (if exists) or lin_out
        if hasattr(model, 'post_lin_list') and len(model.post_lin_list) > 0:
            hook = model.post_lin_list[-1].register_forward_hook(hook_fn)
        elif hasattr(model, 'lin_out'):
            hook = model.lin_out.register_forward_hook(hook_fn)
        else:
            raise AttributeError(f"DEEP_GATGNN model doesn't have post_lin_list or lin_out for visualization")
    else:
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
            print("Applying t-SNE...")
            reduced_embeddings = TSNE(n_components=2, random_state=42).fit_transform(all_embeddings)
        else:
            print("Applying PCA...")
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
        plt.title(f'Crystal Embeddings of {model_name} trained model ({method.upper()})')
        plt.tight_layout()
        plt.savefig(f'crystal_embeddings_{method}_{model_name}.png', dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error during embedding visualization: {str(e)}")
        import traceback
        traceback.print_exc()

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    # XGRAPH_DIR = "/ocean/projects/che250019p/fihiri/OCELOT_data/xtal_graphs2"
    XGRAPH_DIR = "/ocean/projects/che250019p/fihiri/OMDB_data/OMDB_xtal_graphs2"
    
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--model", type=str, required=True, choices=["CGCNN", "MPNN", "MEGNet", "SchNet", "GATGNN", "DEEP_GATGNN", 
                                                                     "GAT", "GINConv", "TiraCGCNNPyG"], help="Model to use for training.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for training (default: cuda).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint file to resume training.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Using GPU: {device_name} (ID: {current_device})")
    else:
        print("CUDA is not available. Using CPU.")

    # batch_size = 30 #MPNN 1
    # batch_size = 204 #CGCNN og
    # batch_size = 16 #CGCNN latest
    # batch_size = 16 #SchNet
    # batch_size = 192 #GATGNN
    batch_size = 64
    # batch_size = 32

    seed = 42
    train_loader, val_loader, test_loader, test_dataset, normalizer = split_crystal_datasets(
        XGRAPH_DIR, 
        batch_size=batch_size,
        test_size=0.15,
        val_size=0.15,
        random_seed=seed
    )    
    # train_loader, val_loader, test_loader, test_dataset, normalizer = load_crystal_datasets_from_splits(
    #     XGRAPH_DIR, 
    #     splits_dir = "qtaim_graph/multigraphs2",
    #     batch_size=batch_size,
    #     splits=["train", "val", "test"]
    # )
    print(f"Training using random seed: {seed}")
    
    first_batch = next(iter(train_loader))
    input_dim = first_batch.x.shape[1]
    edge_dim = first_batch.edge_attr.shape[1]
    hidden_dim = 64
    print(f"Input Dim: {input_dim}, Edge Dim: {edge_dim}, Hidden Dim: {hidden_dim}")
    # # Print graph size information
    # print("\nGraph Size Information:")
    # print(f"Number of nodes: {first_batch.x.size(0)}")
    # print(f"Number of edges: {first_batch.edge_index.size(1)}")
    # print(f"Batch size: {batch_size}")
    # print(f"Features per node: {first_batch.x.size(1)}")
    # print(f"Features per edge: {first_batch.edge_attr.size(1)}")
    # print(f"Total number of graphs in batch: {first_batch.num_graphs}")
    # # print information about a graph, including target y value
    # print(f"Graph ID: {first_batch.id}")
    # print(f"Graph size: {first_batch.x.size(0)}")
    # print(f"Graph edge_index: {first_batch.edge_index}")
    # print(f"Graph edge_attr: {first_batch.edge_attr}")
    # print(f"Graph y: {first_batch.y}")
    # print(f"Graph u: {first_batch.u}")

    if args.model == "CGCNN":
        model = CGCNN(first_batch, dim1=64, dim2=128, pre_fc_count=1, gc_count=5, post_fc_count=2, pool="set2set", dropout_rate=0.06455285352866819)
        # model = CGCNN(first_batch, dim1=128, dim2=64, pre_fc_count=1, gc_count=4, post_fc_count=4, pool="global_mean_pool", dropout_rate=0.13722044694804686)
    elif args.model == "MEGNet":
        model = MEGNet(first_batch, dim1=128, dim2=32, dim3=256, pre_fc_count=1, gc_count=4, gc_fc_count=2, post_fc_count=2, pool="global_max_pool", dropout_rate=0.08919207357951375)
    elif args.model == "MPNN":
        model = MPNN(first_batch, dim1=64, dim2=128, dim3=128, pre_fc_count=1, gc_count=2, gc_fc_count=2, post_fc_count=4, pool="global_max_pool", dropout_rate=0.114362) #MPNN 1
    elif args.model == "SchNet":
        model = SchNet(first_batch, dim1=256, dim2=128, dim3=64, pre_fc_count=1, gc_count=4, gc_fc_count=None, post_fc_count=4, pool="global_mean_pool", dropout_rate=0.3497866040911963)
    elif args.model == "GATGNN":
        # model = GATGNN(first_batch, global_attention="cluster", unpooling_technique="learnable")
        model = GATGNN(first_batch, heads=3, neurons=256, nl=5, global_attention="composition", unpooling_technique="learnable", concat_comp=False)
        #ocelot hpo
        # model = GATGNN(first_batch, heads=5, neurons=64, nl=3, global_attention="cluster", unpooling_technique="learnable", concat_comp=False)

    elif args.model == "DEEP_GATGNN":
        # model = DEEP_GATGNN(first_batch, dim1=32, dim2=64, pre_fc_count=1, gc_count=4, post_fc_count=1, pool="global_add_pool", dropout_rate=0.04379118030158097)
        model = DEEP_GATGNN(first_batch, dim1=32, dim2=128, pre_fc_count=2, gc_count=4, post_fc_count=2, pool="global_add_pool", dropout_rate=0.03297089058595759)

    elif args.model == "GAT":
        model = GATNet(input_dim)
    elif args.model == "GINConv":
        model = GINConvNet(input_dim)

    # elif args.model == "TiraCGCNNPyG":
    #     model = TiraCGCNNPyG(first_batch)
    else:
        raise ValueError("Invalid model choice.")

    model.apply(weights_init)

    print(f"Training Crystal Graphs on {args.model} using {device}")

    # lr=0.00358466 #MPNN 1
    # lr = 0.011231107851831906 #CGCNN
    # lr = 0.0021763932215795624 #CGCNN latest
    # lr = 0.00018440471073394508 #SchNet
    lr = 0.000746275 #GATGNN
    # lr = 0.005658299683505883 #DEEP_GATGNN

    trained_model = train(model, train_loader, val_loader, normalizer, device, epochs=250, lr=lr, patience=50, checkpoint_path=args.checkpoint, model_name=args.model)


    test(trained_model, test_loader, normalizer, device, model_name=args.model)

    print("Visualizing embeddings with t-SNE...")
    visualize_embeddings(trained_model, test_loader, device=device, method='tsne', model_name=args.model)

    torch.save({
        'model': trained_model.state_dict(),
        'normalizers': {key: norm.state_dict() for key, norm in normalizer.items()},
    }, f"best_model_xgraph_{args.model}.pt")
    # os.remove("best_model_xgraph.pth")
if __name__ == "__main__":
    main()