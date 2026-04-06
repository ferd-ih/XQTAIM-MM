---
noteId: "4936d9c0316711f1a37b67cfbea112b9"
tags: []

---

# XQTAIM-MM Training

This directory contains the scripts for training XQTAIM-MM, a multimodal machine learning model that combines Quantum Theory of Atoms in Molecules (QTAIM) graph features with crystal graph representations.

## Script Overview

- **`train_mmxqtaim.py`**: The main entry point for training. It handles the training loop, validation, and logging. It relies on a JSON configuration file to set hyperparameters and dataset paths.
- **`graph_data_xqtaim.py`**: Processes the QTAIM `.gml` files into PyTorch Geometric graph objects. It handles parsing the graphs, extracting node/edge features, and applying any descriptor filtering.
- **`multimodal_xg.py`**: Defines the actual neural network architectures, including the unimodal encoders and the fusion layers that combine them.
- **`dataloader_crystal_graph.py` / `dataloader_xqtaim.py`**: Custom data loaders for batching the graph data efficiently.

## Configuration

Training is controlled by a JSON config file (e.g., `config/config_best_hpo_gat_gate_omdball.json`). This file stores your model hyperparameters (identified a priori via hyperparameter optimization) as well as your data paths.

**Important:** Before training, change the directory paths to point to your actual data locations:
- `"crystal_dir"`: Path to your crystal graph data.
- `"qtaim_dir"`: Path to your QTAIM `.gml` graphs.
- `"split_file_dir"`: Path to the directory containing your train/val/test split files.

## Managing Descriptors (Include/Exclude)

The QTAIM graphs contain a lot of node and edge features. Instead of hardcoding which ones to use, the data loader allows you to dynamically filter them at runtime. This is useful for ablation studies or removing noisy features.

By default, the script automatically discovers and uses all numeric features. this behavior is overidden using the following arguments:

- **`include_node_descriptors`**: List of node descriptor names to include (default: all). If provided, *only* these descriptors will be used.
- **`exclude_node_descriptors`**: List of node descriptor names to exclude (default: none). Applied *after* the include filter.
- **`include_edge_descriptors`**: List of edge descriptor names to include (default: `DEFAULT_EDGE_DESCRIPTORS` as defined in graph_data_xqtaim.py). If provided, *only* these descriptors will be used.
- **`exclude_edge_descriptors`**: List of edge descriptor names to exclude (default: none). Applied *after* the include filter.

**Examples of Descriptor Masking:**

```python
# Use all descriptors (default behavior)
graph = QTAIMGraph(root_dir)

# Ablation: exclude electron density features
graph = QTAIMGraph(root_dir, exclude_edge_descriptors=['e_density', 'e_loc_func'])

# Ablation: only use bond distance and electron density
graph = QTAIMGraph(root_dir, include_edge_descriptors=['bond_distance', 'e_density'])

# Ablation: no node descriptors (element features only)
graph = QTAIMGraph(root_dir, include_node_descriptors=[])
```

When running the training script (`train_mmxqtaim.py`), these filters are passed as comma-separated strings.

## Running a Sample Training


```bash
# 1. Activate your environment
conda activate /ocean/projects/che250019p/fihiri/env/chem_ml

# 2. Define any descriptors you want to drop (comma-separated, no spaces)

# Note: The following is what used in the paper associated with this code
EXCLUDE_EDGE_DESCRIPTORS='bcp_category_hydrogen_bond,bcp_category_orphaned,bcp_category_standard'

# 3. Run the training script
python train_mmxqtaim.py \
    --config config/config_best_hpo_gat_gate_omdb.json \
    --gradient_truncation \
    --epochs 250 \
    --seed 73 \
    --exclude_edge_descriptors $EXCLUDE_EDGE_DESCRIPTORS
```
