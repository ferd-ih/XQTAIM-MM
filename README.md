# Quantum-Chemically Informed Crystal Graph Neural Networks via Multimodal Learning

This repository contains the machine learning pipelines for predicting crystal properties using Quantum Theory of Atoms in Molecules (QTAIM) features and Crystal Graph Neural Networks (CGNN).

## Project Structure

- **`multimodal/`**: Contains the multimodal architecture (XQTAIM-MM) that fuses QTAIM graph representations with crystal graph representations. See `multimodal/README.md` for detailed training instructions, configuration details, and descriptor masking.
- **`unimodal_cgnn/`**: Contains the baseline unimodal Crystal Graph Neural Network models.

## Prerequisites

To run the scripts in this project, you will need the following system requirements:
- **Python 3.x**
- **Anaconda / Miniconda** for environment management
- **CUDA Toolkit**: Version **12.4.0** is recommended and used in the study.
- **GPU**: An NVIDIA GPU (e.g., H100, V100) is highly recommended for training.

## Environment Setup

The models are built primarily using PyTorch and PyTorch Geometric. 
Clone the repository and install the exact project dependencies by executing the following commands in the root of the repository:

```bash
# 1. Clone the repository
git clone <repository-url>
cd <repository-directory>

# 2. Create a new conda environment
conda create -n chem_ml python=3.10 -y
conda activate chem_ml

# 3. Install PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Install PyTorch Geometric and its dependencies
pip install torch_geometric==2.6.1
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# 5. Install remaining required packages
pip install -r requirements.txt
```

## Usage

Each sub-project (`multimodal` and `unimodal_cgnn`) has its own set of training scripts, configurations, and data loaders. 

**General Workflow:**
1. Navigate to the desired model directory (e.g., `cd multimodal/`).
2. Update the JSON configuration file with your specific dataset paths (e.g., `crystal_dir`, `qtaim_dir`, `split_file_dir`).
3. Execute the training script directly or submit an HPC job script.

For detailed instructions on how to run a sample training session, filter node/edge descriptors, and manage configurations for the multimodal model, please refer to the [Multimodal README](multimodal/README.md).