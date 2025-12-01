import scanpy as sc
import nichecompass as nc
from nichecompass.models import NicheCompass
from nichecompass.utils import (
    create_new_color_dict,
    generate_enriched_gp_info_plots,
    compute_communication_gp_network,
    visualize_communication_gp_network
)
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import warnings
import matplotlib
import sys

# Set headless mode (prevent server display errors)
matplotlib.use('Agg')
warnings.filterwarnings("ignore")

# ==========================================================
# 🛠️ User Configuration (Modify this section)
# ==========================================================

# Path to the specific model folder you want to analyze
# Example: /root/autodl-tmp/BGI/STOmics/BM11_..._nichecompass_model
MODEL_DIR = "your_model_path"

# Key parameters (Usually kept default)
GP_NAMES_KEY = "nichecompass_gp_names"
LATENT_CLUSTER_KEY = "nichecompass_niches" # Column name for Niche clusters
LATENT_KEY = "nichecompass_latent"
SAMPLE_KEY = "batch"
SPOT_SIZE = 30 
DIFFERENTIAL_GP_TEST_KEY = "nichecompass_differential_gp_test_results"

# ==========================================================
# 🚀 Analysis Logic
# ==========================================================

def find_analyzed_file(model_dir):
    """Automatically search for the h5ad file in the directory"""
    # 1. Prioritize file with '_analyzed' suffix (contains clustering results)
    candidates = glob.glob(os.path.join(model_dir, "*_analyzed.h5ad"))
    if candidates: return candidates[0]
    
    # 2. Fallback to raw model output 'adata_result.h5ad'
    fallback = os.path.join(model_dir, "adata_result.h5ad")
    if os.path.exists(fallback): return fallback
        
    return None

if not os.path.exists(MODEL_DIR):
    print(f"❌ Error: Directory not found {MODEL_DIR}")
    sys.exit(1)

figure_folder = os.path.join(MODEL_DIR, "figures")
os.makedirs(figure_folder, exist_ok=True)

print(f"\n{'='*60}")
print(f"🚀 Starting Model Analysis: {os.path.basename(MODEL_DIR)}")
print(f"📂 Output Directory: {figure_folder}")
print(f"{'='*60}")

# --- 1. Load Data ---
h5ad_path = find_analyzed_file(MODEL_DIR)
if not h5ad_path:
    print("❌ Error: No .h5ad file found in the directory!")
    sys.exit(1)

print(f"Loading data from: {os.path.basename(h5ad_path)}")
adata = sc.read_h5ad(h5ad_path)

# Load model (weights only)
print("Loading model weights...")
model = NicheCompass.load(dir_path=MODEL_DIR, adata=adata, gp_names_key=GP_NAMES_KEY)

# --- 2. Check and Fix Data ---
print(">>> [1/5] Checking data integrity...")

# 2.1 Fix coordinate format (Solve Pandas Index Error)
if 'spatial' in model.adata.uns:
    del model.adata.uns['spatial']

if isinstance(model.adata.obsm['spatial'], pd.DataFrame):
    print("   ⚠️ Fix: Converting spatial coordinates from DataFrame to NumPy Array")
    model.adata.obsm['spatial'] = model.adata.obsm['spatial'].values
model.adata.obsm['spatial'] = np.array(model.adata.obsm['spatial'])

# 2.2 Ensure Latent Features exist
if LATENT_KEY not in model.adata.obsm:
    print("   ⚠️ Fix: Re-extracting Latent Representation...")
    model.adata.obsm[LATENT_KEY] = model.get_latent_representation()

# 2.3 Ensure Niche Clustering exists (Solve KeyError: nichecompass_niches)
if LATENT_CLUSTER_KEY not in model.adata.obs:
    print("   ⚠️ Fix: Niche clusters not found, running Leiden clustering...")
    sc.pp.neighbors(model.adata, use_rep=LATENT_KEY, key_added=LATENT_KEY)
    sc.tl.leiden(model.adata, resolution=0.4, key_added=LATENT_CLUSTER_KEY, neighbors_key=LATENT_KEY)
    
    # Save the fixed file so we don't need to re-run next time
    new_save_path = h5ad_path.replace(".h5ad", "_fixed.h5ad")
    if "_analyzed" not in h5ad_path:
        new_save_path = os.path.join(MODEL_DIR, "adata_result_analyzed.h5ad")
    model.adata.write(new_save_path)
    print(f"   ✅ Fixed data saved to: {os.path.basename(new_save_path)}")

# 2.4 Ensure Sample Key exists
if SAMPLE_KEY not in model.adata.obs:
    model.adata.obs[SAMPLE_KEY] = "sample_1"

# --- 3. Visualize Niches ---
print(">>> [2/5] Plotting Niche Spatial Distribution...")
samples = model.adata.obs[SAMPLE_KEY].unique().tolist()
latent_cluster_colors = create_new_color_dict(adata=model.adata, cat_key=LATENT_CLUSTER_KEY)

file_path = os.path.join(figure_folder, "niches_latent_physical_space.png")
fig = plt.figure(figsize=(12, 14))
title = fig.suptitle(t="NicheCompass Niches in Latent & Physical Space", y=0.96, x=0.55, fontsize=16)

spec1 = gridspec.GridSpec(ncols=1, nrows=2, width_ratios=[1], height_ratios=[3, 2])
spec2 = gridspec.GridSpec(ncols=len(samples), nrows=2, width_ratios=[1] * len(samples), height_ratios=[3, 2])

axs = []
axs.append(fig.add_subplot(spec1[0]))

if "X_umap" not in model.adata.obsm:
    print("   -> Computing UMAP...")
    sc.tl.umap(model.adata, neighbors_key=LATENT_KEY)

sc.pl.umap(adata=model.adata, color=[LATENT_CLUSTER_KEY], palette=latent_cluster_colors, 
           title="Latent Space", ax=axs[0], show=False)

for idx, s in enumerate(samples):
    axs.append(fig.add_subplot(spec2[len(samples) + idx]))
    subset = model.adata[model.adata.obs[SAMPLE_KEY] == s]
    # Clean subset uns
    if 'spatial' in subset.uns: del subset.uns['spatial']
    subset.obsm['spatial'] = np.array(subset.obsm['spatial'])
    
    sc.pl.spatial(adata=subset, color=[LATENT_CLUSTER_KEY], palette=latent_cluster_colors, 
                  spot_size=SPOT_SIZE, title=f"Physical Space\n({s})", 
                  legend_loc=None, ax=axs[idx+1], show=False)

try:
    handles, labels = axs[0].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.98, 0.5))
    axs[0].get_legend().remove()
    fig.savefig(file_path, bbox_extra_artists=(lgd, title), bbox_inches="tight")
except:
    fig.savefig(file_path, bbox_inches="tight")
plt.close(fig)
print(f"   ✅ Saved: {os.path.basename(file_path)}")

# --- 4. Niche Composition ---
print(">>> [3/5] Plotting Niche Composition...")

if "cluster" in model.adata.obs:
    df_counts = (model.adata.obs.groupby([LATENT_CLUSTER_KEY, "cluster"]).size().unstack())
    
    # 1. Plot Counts
    ax = df_counts.plot(kind="bar", stacked=True, figsize=(12, 6))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Cell Type")
    plt.title("Cell Type Composition of Niches (Counts)")
    plt.xlabel("Niche")
    plt.ylabel("Cell Counts")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "niche_composition_counts.png"))
    plt.close()
    print(f"   ✅ Saved: niche_composition_counts.png")

    # 2. Plot Proportions
    df_counts_norm = df_counts.div(df_counts.sum(axis=1), axis=0)
    ax = df_counts_norm.plot(kind="bar", stacked=True, figsize=(12, 6))
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Cell Type")
    plt.title("Cell Type Composition of Niches (Proportion)")
    plt.xlabel("Niche")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "niche_composition_proportion.png"))
    plt.close()
    print(f"   ✅ Saved: niche_composition_proportion.png")
else:
    print("   ⚠️ Skipped: No 'cluster' column found.")

# --- 5. Differential GP Analysis (Heatmap: Enriched + Depleted) ---
print(">>> [4/5] Performing Differential GP Analysis...")
if DIFFERENTIAL_GP_TEST_KEY in model.adata.uns:
    print("   -> Found existing results, skipping calculation.")
    results_df = model.adata.uns[DIFFERENTIAL_GP_TEST_KEY]
else:
    print("   -> Running differential test (this may take a moment)...")
    model.run_differential_gp_tests(
        cat_key=LATENT_CLUSTER_KEY,
        selected_cats=None,
        comparison_cats="rest",
        log_bayes_factor_thresh=2.3
    )
    results_df = model.adata.uns[DIFFERENTIAL_GP_TEST_KEY]

# [FIX] Standardize column name: Ensure 'gp_name' exists
if "gp_name" not in results_df.columns:
    if "gene_program" in results_df.columns:
        results_df["gp_name"] = results_df["gene_program"]
    else:
        results_df["gp_name"] = results_df.index

# Save CSV
results_df.to_csv(os.path.join(figure_folder, "differential_gp_results.csv"))

# [Heatmap Logic] Filter Top 50 significant GPs (Sorted by Absolute LogBF)
print("   -> Filtering significant GPs (Abs(LogBF) > 2.3, Top 50)...")

# 1. Filter Significant (Absolute value > 2.3)
sig_mask = np.abs(results_df["log_bayes_factor"]) > 2.3
sig_df = results_df[sig_mask].copy()

# 2. Sort by absolute significance (descending)
sig_df["abs_log_bf"] = sig_df["log_bayes_factor"].abs()
sig_df = sig_df.sort_values(by="abs_log_bf", ascending=False)

# 3. Take top 50 unique GP names
heatmap_gps = sig_df["gp_name"].unique().tolist()[:50]

if len(heatmap_gps) > 0:
    df = model.adata.obs[[LATENT_CLUSTER_KEY] + heatmap_gps].groupby(LATENT_CLUSTER_KEY).mean()
    scaler = MinMaxScaler()
    norm_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    
    # Adjust figure size for 50 labels
    plt.figure(figsize=(20, 10)) 
    
    # xticklabels=True: Force display of all x-axis labels
    sns.heatmap(norm_df, cmap='coolwarm', center=0.5, annot=False, xticklabels=True)
    
    plt.title(f"Top {len(heatmap_gps)} Significant GPs (Enriched & Depleted)")
    plt.xlabel("Gene Programs")
    plt.xticks(rotation=45, ha='right', fontsize=10) # Rotate labels
    
    plt.savefig(os.path.join(figure_folder, "enriched_gps_heatmap.png"), bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved Heatmap (Count: {len(heatmap_gps)})")
    print(f"   ✅ Saved CSV")

# --- 6. Communication Networks (Plot: Enriched Only) ---
print(">>> [5/5] Plotting Communication Networks (Top 10 Enriched Only)...")

# 1. Filter POSITIVE significant only (LogBF > 2.3)
positive_gps_df = results_df[results_df["log_bayes_factor"] > 2.3]

if len(positive_gps_df) == 0:
    print("   ⚠️ No significantly enriched GPs found. Skipping network plots.")
else:
    # Extract GP names
    candidates = positive_gps_df["gp_name"].unique().tolist()
    
    # Take top 10 enriched
    candidates = candidates[:10]
    print(f"   -> Candidates (Enriched): {candidates}")

    count = 0
    for gp_name in candidates:
        try:
            network_df = compute_communication_gp_network(
                gp_list=[gp_name],
                model=model,
                group_key=LATENT_CLUSTER_KEY,
                n_neighbors=4
            )
            
            if len(network_df) > 0:
                out_path = os.path.join(figure_folder, f"gp_network_{gp_name}.png")
                visualize_communication_gp_network(
                    adata=model.adata,
                    network_df=network_df,
                    figsize=(10, 7),
                    cat_colors=latent_cluster_colors,
                    cat_key=LATENT_CLUSTER_KEY,
                    save=True,
                    save_path=out_path,
                )
                print(f"      ✅ Saved: gp_network_{gp_name}.png")
                count += 1
            else:
                print(f"      ⚠️ Skipped {gp_name} (Empty network)")
        except Exception as e:
            print(f"      ❌ Error plotting {gp_name}: {e}")

    print(f"   -> Generated {count} network plots.")

print("\n🎉 Analysis Completed Successfully!")
