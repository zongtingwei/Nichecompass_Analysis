# Nichecompass_Analysis

## 📖 Overview
Nichecompass_Analysis

## 🚀 How to run
### build the cellphonedb env

```bash
conda create -n nichecompass python=3.10 -y
conda activate nichecompass
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
pip install torch-geometric nichecompass "numpy<2" scanpy jupyter matplotlib
pip install igraph leidenalg
```

