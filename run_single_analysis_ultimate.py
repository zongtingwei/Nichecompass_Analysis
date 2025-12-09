#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scanpy as sc
import nichecompass as nc
from nichecompass.models import NicheCompass
from nichecompass.utils import (
    create_new_color_dict,
    generate_enriched_gp_info_plots,
    compute_communication_gp_network,
    visualize_communication_gp_network,
)

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import warnings
import sys
import math
from pathlib import Path
from typing import List, Optional, Dict

warnings.filterwarnings("ignore")

# ==========================================================
# 🛠️ User Configuration
# ==========================================================

MODEL_DIR = "your_path"

GP_NAMES_KEY = "nichecompass_gp_names"
LATENT_CLUSTER_KEY = "nichecompass_niches"
LATENT_KEY = "nichecompass_latent"
SAMPLE_KEY = "batch"
SPOT_SIZE = 30
DIFFERENTIAL_GP_TEST_KEY = "nichecompass_differential_gp_test_results"

# ==========================================================
# 🎨 Plotting Utils
# ==========================================================

def _to_numpy2(x) -> np.ndarray:
    if isinstance(x, (pd.DataFrame, pd.Series)):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)
    return np.asarray(arr, dtype=np.float32)


def autocolors(n: int):
    cmap1 = plt.get_cmap("tab20")
    cmap2 = plt.get_cmap("hsv")
    return [cmap1(i % 20) if i < 20 else cmap2((i - 20) / max(1, n - 20)) for i in range(n)]


def _save_both(fig, png_path: str, pdf_path: Optional[str] = None, dpi: int = 320):
    fig.savefig(png_path, dpi=dpi, facecolor=fig.get_facecolor())
    if pdf_path is None:
        pdf_path = str(Path(png_path).with_suffix(".pdf"))
    fig.savefig(pdf_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def rasterize_labels(coords: np.ndarray, labels: pd.Series):
    """
    把离散点 rasterize 成网格，保持 labels 的 category 顺序（0,1,2,...）
    """
    x, y = coords[:, 0], coords[:, 1]
    ux, uy = np.unique(x), np.unique(y)
    dx = np.median(np.diff(np.sort(ux))) if len(ux) > 1 else 1.0
    dy = np.median(np.diff(np.sort(uy))) if len(uy) > 1 else 1.0

    ix = np.rint((x - x.min()) / max(dx, 1e-6)).astype(int)
    iy = np.rint((y - y.min()) / max(dy, 1e-6)).astype(int)

    W, H = ix.max() + 1, iy.max() + 1

    if hasattr(labels, "cat"):
        cats = [str(c) for c in labels.cat.categories]
        labels_str = labels.astype(str)
    else:
        labels_str = labels.astype(str)
        try:
            cats = sorted(labels_str.unique(), key=lambda x: int(x))
        except ValueError:
            cats = sorted(labels_str.unique())

    code = pd.Categorical(labels_str, categories=cats).codes

    img = np.full((H, W), fill_value=-1, dtype=np.int32)
    img[iy, ix] = code
    cover = np.full((H, W), np.nan, dtype=float)
    cover[iy, ix] = 1.0
    extent = (x.min() - dx / 2, x.max() + dx / 2,
              y.min() - dy / 2, y.max() + dy / 2)
    return img, cover, extent, cats


def pick_legend_loc(img: np.ndarray, corner_frac: float = 0.18) -> str:
    H, W = img.shape
    h = max(1, int(round(H * corner_frac)))
    w = max(1, int(round(W * corner_frac)))
    mask = (img >= 0)
    occ_tl = float(mask[0:h, 0:w].mean()) if h > 0 and w > 0 else 1.0
    occ_tr = float(mask[0:h, W - w:W].mean()) if h > 0 and w > 0 else 1.0
    return "upper left" if occ_tl < occ_tr else "upper right"


def plot_overlay_imshow(
    img, extent, cats: List[str], out_png: str,
    origin_upper=True, dpi=320,
    fig_bg="#ffffff", corner_frac=0.18,
    sample_title: Optional[str] = None,
    custom_colors: Optional[List[str]] = None,
):
    K = len(cats)
    if custom_colors and len(custom_colors) >= K:
        colors = custom_colors[:K]
    else:
        colors = autocolors(K)

    cmap = ListedColormap(colors)
    img_masked = np.ma.masked_where(img < 0, img)
    cmap.set_bad(alpha=0.0)
    vmin, vmax = 0, max(0, K - 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(fig_bg)
    ax.set_facecolor(fig_bg)

    ax.imshow(
        img_masked,
        cmap=cmap,
        origin=("upper" if origin_upper else "lower"),
        interpolation="nearest",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    if sample_title:
        ax.set_title(sample_title, fontsize=15, pad=10)

    handles = [Patch(facecolor=colors[i], edgecolor="none", label=cats[i]) for i in range(K)]
    loc = pick_legend_loc(img, corner_frac=corner_frac)
    anchor_map = {"upper left": (0.0, 1.0), "upper right": (1.0, 1.0)}
    anchor = anchor_map.get(loc, (1.0, 1.0))

    leg = ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=anchor,
        borderaxespad=0.2,
        fontsize=10,
        frameon=True,
        title=f"Niches: {K}",
        labelspacing=0.3,
        handlelength=1.1,
        handletextpad=0.4,
        borderpad=0.4,
    )
    fr = leg.get_frame()
    fr.set_facecolor((1, 1, 1, 0.90))
    fr.set_edgecolor("black")
    fr.set_linewidth(0.5)

    plt.tight_layout()
    _save_both(fig, out_png, dpi=dpi)


def plot_facets_imshow(
    img, cover, extent, cats: List[str], out_png: str,
    ncols=None, dpi=320, origin_upper=True,
    fig_bg="#ffffff", sample_bg="#e6e6e6",
    sample_title: Optional[str] = None,
    custom_colors: Optional[List[str]] = None,
):
    K = len(cats)
    if K == 0:
        return
    if ncols is None:
        ncols = int(math.ceil(math.sqrt(K)))
    nrows = int(math.ceil(K / ncols))

    if custom_colors and len(custom_colors) >= K:
        base_colors = custom_colors[:K]
    else:
        base_colors = autocolors(K)

    fig_w, fig_h = max(8.0, 3.5 * ncols), max(6.0, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)
    fig.patch.set_facecolor(fig_bg)
    cover_cmap = ListedColormap([sample_bg])

    for i, (cat, ax) in enumerate(zip(cats, axes)):
        ax.set_facecolor(fig_bg)
        cover_masked = np.ma.masked_invalid(cover)
        cover_cmap.set_bad(alpha=0.0)
        ax.imshow(
            cover_masked,
            cmap=cover_cmap,
            origin=("upper" if origin_upper else "lower"),
            interpolation="nearest",
            extent=extent,
            vmin=0,
            vmax=1,
        )

        layer = np.where(img == i, 1.0, np.nan)
        hl_cmap = ListedColormap([base_colors[i]])
        hl_cmap.set_bad(alpha=0.0)
        ax.imshow(
            layer,
            cmap=hl_cmap,
            origin=("upper" if origin_upper else "lower"),
            interpolation="nearest",
            extent=extent,
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"{cat}", fontsize=12)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    for j in range(K, len(axes)):
        axes[j].axis("off")
    if sample_title:
        fig.suptitle(f"{sample_title}", fontsize=16, y=0.995)
    plt.tight_layout(pad=1.0, rect=[0, 0, 1, 0.98])
    _save_both(fig, out_png, dpi=dpi)

# ==========================================================
# 🚀 Analysis Logic
# ==========================================================

def find_analyzed_file(model_dir):
    candidates = glob.glob(os.path.join(model_dir, "*_analyzed.h5ad"))
    if candidates:
        return candidates[0]
    fallback = os.path.join(model_dir, "adata_result.h5ad")
    if os.path.exists(fallback):
        return fallback
    return None


def get_gp_display_name(model, gp_name):
    try:
        gp_summary = model.get_gp_summary()
        if "gp_name" not in gp_summary.columns:
            gp_summary["gp_name"] = gp_summary.index
        row = gp_summary[gp_summary["gp_name"] == gp_name]
        if len(row) == 0:
            return gp_name
        sources = row["gp_source_genes"].values[0]
        targets = row["gp_target_genes"].values[0]
        if isinstance(sources, list):
            sources = " & ".join(sources)
        if isinstance(targets, list):
            targets = " & ".join(targets)
        if len(sources) > 15:
            sources = sources[:12] + ".."
        if len(targets) > 15:
            targets = targets[:12] + ".."
        return f"{sources} -> {targets}"
    except Exception:
        return gp_name


# ----------------------------------------------------------

if not os.path.exists(MODEL_DIR):
    print(f"❌ Error: Directory not found {MODEL_DIR}")
    sys.exit(1)

figure_folder = os.path.join(MODEL_DIR, "figures")
os.makedirs(figure_folder, exist_ok=True)

print("\n" + "=" * 60)
print(f"🚀 Starting Advanced Analysis: {os.path.basename(MODEL_DIR)}")
print("=" * 60)

# --- 1. Load Data ---
h5ad_path = find_analyzed_file(MODEL_DIR)
if not h5ad_path:
    print("❌ Error: No .h5ad file found!")
    sys.exit(1)

print(f"Loading data from: {os.path.basename(h5ad_path)}")
adata = sc.read_h5ad(h5ad_path)
print("Loading model weights...")
model = NicheCompass.load(dir_path=MODEL_DIR, adata=adata, gp_names_key=GP_NAMES_KEY)

# --- 2. Fix Data ---
print(">>> [1/6] Checking data integrity...")
if "spatial" in model.adata.uns:
    del model.adata.uns["spatial"]
if isinstance(model.adata.obsm["spatial"], pd.DataFrame):
    model.adata.obsm["spatial"] = model.adata.obsm["spatial"].values
model.adata.obsm["spatial"] = np.array(model.adata.obsm["spatial"])

if LATENT_KEY not in model.adata.obsm:
    model.adata.obsm[LATENT_KEY] = model.get_latent_representation()

if LATENT_CLUSTER_KEY not in model.adata.obs:
    print("   ⚠️ Re-running Leiden clustering...")
    sc.pp.neighbors(model.adata, use_rep=LATENT_KEY, key_added=LATENT_KEY)
    sc.tl.leiden(
        model.adata,
        resolution=0.4,
        key_added=LATENT_CLUSTER_KEY,
        neighbors_key=LATENT_KEY,
    )

if SAMPLE_KEY not in model.adata.obs:
    model.adata.obs[SAMPLE_KEY] = "sample_1"

labels_str = model.adata.obs[LATENT_CLUSTER_KEY].astype(str)
try:
    ordered_cats = sorted(labels_str.unique(), key=lambda x: int(x))
except ValueError:
    ordered_cats = sorted(labels_str.unique())
model.adata.obs[LATENT_CLUSTER_KEY] = pd.Categorical(
    labels_str, categories=ordered_cats, ordered=True
)
samples = model.adata.obs[SAMPLE_KEY].unique().tolist()
latent_cluster_colors_dict = create_new_color_dict(
    adata=model.adata, cat_key=LATENT_CLUSTER_KEY
)


print(">>> [2/6] Plotting Latent & Physical Space figure...")
if "X_umap" not in model.adata.obsm:
    if LATENT_KEY not in model.adata.uns:
        sc.pp.neighbors(model.adata, use_rep=LATENT_KEY, key_added=LATENT_KEY)
    sc.tl.umap(model.adata, neighbors_key=LATENT_KEY)

file_path = os.path.join(figure_folder, "niches_latent_physical_space.png")

fig = plt.figure(figsize=(12, 14))
outer = gridspec.GridSpec(
    nrows=2, ncols=1, height_ratios=[3, 2], hspace=0.35
)

ax_latent = fig.add_subplot(outer[0, 0])
sc.pl.umap(
    adata=model.adata,
    color=[LATENT_CLUSTER_KEY],
    palette=latent_cluster_colors_dict,
    title="Latent Space",
    ax=ax_latent,
    show=False,
    legend_loc=None,  
)

bottom_gs = gridspec.GridSpecFromSubplotSpec(
    1, len(samples), subplot_spec=outer[1, 0], wspace=0.30
)

for idx, s in enumerate(samples):
    ax = fig.add_subplot(bottom_gs[0, idx])
    subset = model.adata[model.adata.obs[SAMPLE_KEY] == s].copy()
    if "spatial" in subset.uns:
        del subset.uns["spatial"]
    subset.obsm["spatial"] = np.array(subset.obsm["spatial"])

    sc.pl.spatial(
        adata=subset,
        color=[LATENT_CLUSTER_KEY],
        palette=latent_cluster_colors_dict,
        spot_size=SPOT_SIZE,
        title=f"Physical Space\n({s})",
        legend_loc=None,
        ax=ax,
        show=False,
    )

cats_all = model.adata.obs[LATENT_CLUSTER_KEY].cat.categories
handles = [
    Patch(
        facecolor=latent_cluster_colors_dict.get(str(c), "#333333"),
        edgecolor="none",
        label=str(c),
    )
    for c in cats_all
]

lgd = fig.legend(
    handles,
    [str(c) for c in cats_all],
    loc="center left",
    bbox_to_anchor=(0.98, 0.5),
    borderaxespad=0.6,
    frameon=True,
    title="Niches",
    fontsize=10,
)
frame = lgd.get_frame()
frame.set_facecolor((1, 1, 1, 0.9))
frame.set_edgecolor("black")
frame.set_linewidth(0.5)

title = fig.suptitle(
    "NicheCompass Niches in Latent & Physical Space",
    x=0.5,
    y=0.98,
    fontsize=16,
)

fig.savefig(
    file_path,
    dpi=320,
    bbox_inches="tight",
    bbox_extra_artists=(lgd, title),
)
plt.close(fig)
print(f"   ✅ Saved: {os.path.basename(file_path)}")


print(">>> [3/6] Plotting Advanced Spatial Maps (Rasterized)...")
for s in samples:
    print(f"   -> Processing sample: {s}")
    subset = model.adata[model.adata.obs[SAMPLE_KEY] == s]
    if "spatial" in subset.uns:
        del subset.uns["spatial"]
    coords = _to_numpy2(subset.obsm["spatial"])
    labels = subset.obs[LATENT_CLUSTER_KEY]

    img, cover, extent, cats = rasterize_labels(coords, labels)

    ordered_colors = [latent_cluster_colors_dict.get(c, "#333333") for c in cats]

    overlay_path = os.path.join(figure_folder, f"niches_spatial_overlay_{s}.png")
    plot_overlay_imshow(
        img,
        extent,
        cats,
        overlay_path,
        sample_title=f"Niche Distribution ({s})",
        origin_upper=True,
        custom_colors=ordered_colors,
    )

    facet_path = os.path.join(figure_folder, f"niches_spatial_facet_{s}.png")
    plot_facets_imshow(
        img,
        cover,
        extent,
        cats,
        facet_path,
        sample_title=f"Separate Niches ({s})",
        origin_upper=True,
        custom_colors=ordered_colors,
    )

print("   ✅ Saved high-res spatial plots (Overlay & Facet)")

# --- 5. Niche Composition ---
print(">>> [4/6] Plotting Niche Composition...")
if "cluster" in model.adata.obs:
    df_counts = (
        model.adata.obs.groupby([LATENT_CLUSTER_KEY, "cluster"])
        .size()
        .unstack()
    )

    ax = df_counts.plot(kind="bar", stacked=True, figsize=(12, 6), rot=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Cell Type")
    plt.title("Niche Composition (Counts)")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "niche_composition_counts.png"))
    plt.close()

    df_counts_norm = df_counts.div(df_counts.sum(axis=1), axis=0)
    ax = df_counts_norm.plot(kind="bar", stacked=True, figsize=(12, 6), rot=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Cell Type")
    plt.title("Niche Composition (Proportion)")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "niche_composition_proportion.png"))
    plt.close()
    print("   ✅ Saved composition plots")
else:
    print("   ⚠️ Skipped composition: No 'cluster' column.")

# --- 6. Differential GP Analysis ---
print(">>> [5/6] Performing Differential GP Analysis...")
if DIFFERENTIAL_GP_TEST_KEY in model.adata.uns:
    results_df = model.adata.uns[DIFFERENTIAL_GP_TEST_KEY]
else:
    model.run_differential_gp_tests(
        cat_key=LATENT_CLUSTER_KEY,
        selected_cats=None,
        comparison_cats="rest",
        log_bayes_factor_thresh=2.3,
    )
    results_df = model.adata.uns[DIFFERENTIAL_GP_TEST_KEY]

if "gp_name" not in results_df.columns:
    if "gene_program" in results_df.columns:
        results_df["gp_name"] = results_df["gene_program"]
    else:
        results_df["gp_name"] = results_df.index

results_df.to_csv(os.path.join(figure_folder, "differential_gp_results.csv"))

print("   -> Generating Heatmap with L-R names...")
sig_mask = np.abs(results_df["log_bayes_factor"]) > 2.3
sig_df = results_df[sig_mask].copy()
sig_df["abs_log_bf"] = sig_df["log_bayes_factor"].abs()
sig_df = sig_df.sort_values(by="abs_log_bf", ascending=False)

heatmap_gps = sig_df["gp_name"].unique().tolist()[:50]

if len(heatmap_gps) > 0:
    df = (
        model.adata.obs[[LATENT_CLUSTER_KEY] + heatmap_gps]
        .groupby(LATENT_CLUSTER_KEY)
        .mean()
    )
    scaler = MinMaxScaler()
    norm_df = pd.DataFrame(
        scaler.fit_transform(df), columns=df.columns, index=df.index
    )

    display_names = []
    for gp in norm_df.columns:
        display_name = get_gp_display_name(model, gp)
        display_names.append(display_name)
    norm_df.columns = display_names

    plt.figure(figsize=(20, 12))
    sns.heatmap(norm_df, cmap="coolwarm", center=0.5, annot=False, xticklabels=True)
    plt.title(f"Top {len(heatmap_gps)} Significant Pathways (Ligand -> Receptor)")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "enriched_gps_heatmap.png"), bbox_inches="tight")
    plt.close()
    print("   ✅ Saved Heatmap")

# --- 7. Communication Networks ---
print(">>> [6/6] Plotting Communication Networks (Enriched)...")
positive_gps_df = results_df[results_df["log_bayes_factor"] > 2.3]

if len(positive_gps_df) == 0:
    print("   ⚠️ No enriched GPs found.")
else:
    candidates = positive_gps_df["gp_name"].unique().tolist()[:10]
    print(f"   -> Processing top {len(candidates)} enriched pathways...")

    count = 0
    for gp_name in candidates:
        try:
            network_df = compute_communication_gp_network(
                gp_list=[gp_name],
                model=model,
                group_key=LATENT_CLUSTER_KEY,
                n_neighbors=4,
            )

            if len(network_df) > 0:
                out_path = os.path.join(figure_folder, f"gp_network_{gp_name}.png")
                visualize_communication_gp_network(
                    adata=model.adata,
                    network_df=network_df,
                    figsize=(10, 7),
                    cat_colors=latent_cluster_colors_dict,
                    cat_key=LATENT_CLUSTER_KEY,
                    save=True,
                    save_path=out_path,
                )
                print(f"      ✅ Saved: {os.path.basename(out_path)}")
                count += 1
            else:
                print(f"      ⚠️ Skipped {gp_name} (Empty)")
        except Exception as e:
            print(f"      ❌ Error plotting {gp_name}: {e}")

    print(f"   -> Generated {count} network plots.")

print("\n🎉 Advanced Analysis Completed!")
