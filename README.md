# Nichecompass_Analysis

## 📖 Overview
Nichecompass_Analysis

## 🚀 How to run
### build the nichecompass conda environment

```bash
conda create -n nichecompass python=3.10 -y
conda activate nichecompass
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
pip install torch-geometric nichecompass "numpy<2" scanpy jupyter matplotlib
pip install igraph leidenalg
```
### download the official Nichecompass respository
```bash
git clone https://github.com/Lotfollahi-lab/nichecompass.git
```
### train your model
```bash
conda activate nichecompass
python run_niche_auto_train_stable.py
```
### remember to change the path to your own path
```bash
# 1. Path to gene ortholog mapping file (OmniPath Human -> Mouse)
# official nichecompass respository
ORTHOLOG_FILE = ".../nichecompass/data/gene_annotations/human_mouse_gene_orthologs.csv"

# 2. Directory containing input data (.h5ad files)
DATA_DIR = ".../spatial_data/bin20_h5ad_10.30"

# 3. Base output directory for results
# your own directory
OUTPUT_BASE_DIR = ".../"
```

### results analysis
```bash
python run_single_analysis.py
```
### remember to change the path to your own path
```bash
# Path to the specific model folder you want to analyze
# Example: /root/autodl-tmp/BGI/STOmics/BM11_..._nichecompass_model
MODEL_DIR = "your_model_path"
```





