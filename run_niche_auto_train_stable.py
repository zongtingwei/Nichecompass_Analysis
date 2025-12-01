import os
import sys
import glob
import subprocess
import time

# ==========================================================
# 🛠️ User Configuration
# ==========================================================

# 1. Path to gene ortholog mapping file (OmniPath Human -> Mouse)
# official nichecompass respository
ORTHOLOG_FILE = ".../nichecompass/data/gene_annotations/human_mouse_gene_orthologs.csv"

# 2. Directory containing input data (.h5ad files)
DATA_DIR = ".../spatial_data/bin20_h5ad_10.30"

# 3. Base output directory for results
# your own directory
OUTPUT_BASE_DIR = ".../"

# 4. Training parameters (Reduce BATCH_SIZE if OOM occurs)
MAX_EPOCHS = 200
# Note: A lower batch size helps with GPU memory, but system RAM is critical for loading large datasets.
# Recommendation: 128 or 256 for large samples to be safe.
BATCH_SIZE = 128  
LR = 0.001

# ==========================================================
# 👷 Worker Logic (Process a Single File)
# ==========================================================
def worker_logic(adata_path):
    import scanpy as sc
    import nichecompass as nc
    from nichecompass.models import NicheCompass
    from nichecompass.utils import extract_gp_dict_from_omnipath_lr_interactions, add_gps_from_gp_dict_to_adata
    import torch
    import numpy as np
    import gc
    import warnings
    
    warnings.filterwarnings("ignore")
    
    # --- Helper Functions ---
    def convert_to_float32(mat):
        """Converts matrix to float32 precision."""
        if hasattr(mat, "tocsr"): return mat.astype(np.float32).tocsr()
        return mat.astype(np.float32)

    def generate_sample_info(file_path):
        """Generates sample ID from filename."""
        filename = os.path.basename(file_path)
        name_stem = filename.replace(".h5ad", "")
        parts = name_stem.split("_")
        if len(parts) >= 2: short_id = f"{parts[0]}_{parts[1]}"
        else: short_id = name_stem
        return short_id, name_stem

    sample_short_name, sample_unique_name = generate_sample_info(adata_path)
    save_dir = os.path.join(OUTPUT_BASE_DIR, f"{sample_unique_name}_nichecompass_model")
    
    # Check for existing results (Breakpoint Resume)
    if os.path.exists(os.path.join(save_dir, f"{sample_unique_name}_analyzed.h5ad")):
        print(f"⏩ [Worker] {sample_short_name} already exists. Skipping.")
        return

    os.makedirs(save_dir, exist_ok=True)
    print(f"\n🚀 [Worker] Starting process: {sample_short_name} (PID: {os.getpid()})")
    print(f"📂 File: {os.path.basename(adata_path)}")

    try:
        # 1. Load Data
        print(">>> [1/5] Loading data...")
        adata = sc.read_h5ad(adata_path)
        adata.var_names_make_unique()
        
        # 2. Memory Optimization: Convert precision to float32
        adata.X = convert_to_float32(adata.X)
        if "counts" not in adata.layers: adata.layers["counts"] = adata.X.copy()
        else: adata.layers["counts"] = convert_to_float32(adata.layers["counts"])
        
        sc.pp.filter_genes(adata, min_cells=10)

        # 3. Construct Spatial Graph
        print(">>> [2/5] Constructing spatial neighbor graph...")
        sc.pp.neighbors(adata, use_rep="spatial", n_neighbors=4, key_added="spatial")
        if "spatial_connectivities" in adata.obsp:
            adata.obsp["spatial_connectivities"] = adata.obsp["spatial_connectivities"].maximum(adata.obsp["spatial_connectivities"].T)

        # 4. Inject Knowledge Base
        print(">>> [3/5] Injecting knowledge base...")
        gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
            species="mouse", min_curation_effort=0, load_from_disk=False, save_to_disk=False,
            plot_gp_gene_count_distributions=False, gene_orthologs_mapping_file_path=ORTHOLOG_FILE
        )
        add_gps_from_gp_dict_to_adata(
            gp_dict=gp_dict, adata=adata, gp_targets_mask_key="nichecompass_gp_targets",
            gp_sources_mask_key="nichecompass_gp_sources", gp_names_key="nichecompass_gp_names",
            min_genes_per_gp=2, min_source_genes_per_gp=1, min_target_genes_per_gp=1
        )

        # 5. Initialize Model
        print(">>> [4/5] Initializing model...")
        model = NicheCompass(
            adata, counts_key="counts", adj_key="spatial_connectivities",
            gp_names_key="nichecompass_gp_names", active_gp_names_key="nichecompass_active_gp_names",
            gp_targets_mask_key="nichecompass_gp_targets", gp_sources_mask_key="nichecompass_gp_sources",
            latent_key="nichecompass_latent", conv_layer_encoder="gcnconv", active_gp_thresh_ratio=0.01
        )

        # 🔥 [CRITICAL OPTIMIZATION] 🔥
        # The model creates an internal copy of adata. 
        # Delete the original 'adata' object immediately to free up ~50% system RAM!
        print(">>> 🗑️ [Memory Optimization] Deleting original adata object...")
        del adata
        gc.collect()

        # 6. Train
        print(f">>> [5/5] Starting training (Batch Size: {BATCH_SIZE})...")
        model.train(
            n_epochs=MAX_EPOCHS, lr=LR, lambda_edge_recon=500000.0, lambda_gene_expr_recon=300.0,
            edge_batch_size=BATCH_SIZE, 
            use_cuda_if_available=True, verbose=True
        )

        # 7. Save & Cluster
        print(">>> Saving model and performing clustering...")
        # Save base model first to prevent data loss on clustering error
        model.save(dir_path=save_dir, overwrite=True, save_adata=True, adata_file_name="adata_result.h5ad")
        
        # Perform Leiden clustering on latent space
        sc.pp.neighbors(model.adata, use_rep="nichecompass_latent", key_added="nichecompass_latent")
        sc.tl.leiden(model.adata, resolution=0.4, key_added="nichecompass_niches", neighbors_key="nichecompass_latent")
        
        # Save final analyzed file
        final_path = os.path.join(save_dir, f"{sample_unique_name}_analyzed.h5ad")
        model.adata.write(final_path)
        print(f"✅ [Worker] Successfully completed: {final_path}")

    except Exception as e:
        print(f"❌ [Worker] Error: {e}")
        sys.exit(1) # Return error code 1 to notify main process

# ==========================================================
# 🧠 Main Controller Logic (Scheduler)
# ==========================================================
if __name__ == "__main__":
    # Check if script is running as a worker subprocess
    if len(sys.argv) > 1 and sys.argv[1] == "--target":
        worker_logic(sys.argv[2])
        sys.exit(0)

    # Otherwise, run as the main process manager
    print(f"🔧 Starting Robust Batch Training Manager...")
    
    if not os.path.exists(ORTHOLOG_FILE):
        print("❌ Error: Gene ortholog mapping file not found!")
        sys.exit(1)

    # 1. Scan for files
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5ad")))
    # Filter out result files to avoid reprocessing
    valid_files = [f for f in files if "result" not in f and "nichecompass" not in f]
    
    print(f"📋 Found {len(valid_files)} files to process:")
    for f in valid_files:
        print(f"  - {os.path.basename(f)}")
    print("-" * 40)

    # 2. Launch worker processes sequentially
    for i, file_path in enumerate(valid_files):
        print(f"\n>>>>>> Processing File {i+1}/{len(valid_files)} <<<<<<")
        
        # Use subprocess to launch a fresh, clean Python process for each sample.
        # This ensures 100% memory reclamation by the OS after each sample finishes.
        start_time = time.time()
        
        process = subprocess.run(
            [sys.executable, __file__, "--target", file_path],
            capture_output=False # Stream child process output directly to console
        )
        
        duration = time.time() - start_time
        if process.returncode == 0:
            print(f"✨ Sample processed successfully (Time: {duration:.1f}s)")
        else:
            print(f"⚠️ Sample processing failed (Exit Code: {process.returncode}). Proceeding to next...")
            # Optional: time.sleep(5) to let system stabilize

    print("\n🎉🎉🎉 All tasks completed!")
