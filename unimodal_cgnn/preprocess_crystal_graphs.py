import os
import torch
from tqdm import tqdm
from global_crystal_graph_data import CrystalGraph

def preprocess_crystal_graphs(cif_dir: str, output_dir: str, id_prop_dir=None, r_max: float = 8.0, n_neighbors: int = 12):
    """
    Process all CIF files and save their crystal graphs.
    
    Args:
        cif_dir (str): Directory containing CIF files and id_prop.csv
        output_dir (str): Directory to save processed graph files
        r_max (float): Maximum radius for neighbor search
        n_neighbors (int): Maximum number of neighbors to consider
    """
    os.makedirs(output_dir, exist_ok=True)
    
    crystal_graph_maker = CrystalGraph(root_dir=cif_dir, id_prop_dir=id_prop_dir, r_max=r_max, n_neighbors=n_neighbors)
    
    for idx in tqdm(range(len(crystal_graph_maker.data_list)), desc="Processing crystal structures"):
        cod_id = crystal_graph_maker.data_list[idx]
        target = crystal_graph_maker.target_list[idx]
        global_val = crystal_graph_maker.global_value[idx]
        
        try:
            data = crystal_graph_maker.get(cod_id)
            data.y = torch.tensor([target], dtype=torch.float)
            data.u = torch.tensor([[global_val]], dtype=torch.float)
            output_file = os.path.join(output_dir, f"{cod_id}.pt")
            torch.save(data, output_file)
            
        except Exception as e:
            print(f"Error processing {cod_id}: {str(e)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess crystal structures into graph files")
    parser.add_argument("--cif_dir", type=str, required=True, help="Directory containing CIF files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for graph files")
    parser.add_argument("--id_prop_dir", type=str, default=None, help="Directory containing id_prop.csv")
    parser.add_argument("--r_max", type=float, default=8.0, help="Maximum radius for neighbor search")
    parser.add_argument("--n_neighbors", type=int, default=12, help="Maximum number of neighbors")
    
    args = parser.parse_args()
    preprocess_crystal_graphs(args.cif_dir, args.output_dir, args.id_prop_dir, args.r_max, args.n_neighbors) 