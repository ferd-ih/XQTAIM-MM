# Unimodal Crystal Graph Models

This directory contains scripts and models for training and evaluating unimodal crystal graph neural networks.

## Contents
- `models/`: Architectures for various crystal GNNs (GATGNN, CGCNN, MEGNet, SchNet, MPNN, GINConv).
- `train_xgraphs.py`: Main training script.
- `eval_oos_crystal_viz.py`: Script for out-of-sample evaluation and visualization.
- `dataloader_crystal_graph.py`: Dataset and dataloader utilities.
- `preprocess_crystal_graphs.py` & `global_crystal_graph_data.py`: Data preprocessing utilities.

## Usage

### Training
To train a model (e.g., GATGNN) on a GPU:
```bash
python train_xgraphs.py --model GATGNN --device cuda
```
*(Check `models/` for other supported model types).*

### Evaluation
To evaluate a trained model on an out-of-sample dataset (e.g., PAH or ROY datasets):
```bash
python eval_oos_crystal_viz.py \
    --model_path best_model_xgraph_GATGNN.pt \
    --model_type GATGNN \
    --data_dir /path/to/oos_data/xtal_graphs \
    --batch_size 4
```