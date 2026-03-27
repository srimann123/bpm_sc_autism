import time
import os
import math
import zlib
import sys

start = time.time()
import h5py
print("h5py import: ", time.time() - start, flush=True)

start = time.time()
import numpy as np
print("numpy import: ", time.time() - start, flush=True)

start = time.time()
import pandas as pd
print("pandas import: ", time.time() - start, flush=True)

start = time.time()
import warnings
print("warning import: ", time.time() - start, flush=True)

start = time.time()
from cupyx.scipy.sparse import csr_matrix, vstack, diags
from cupyx.scipy import sparse as cpx_sparse #### Added this line for covariate residualization
print("cupyx.scipy.sparse import: ", time.time() - start, flush=True)

start = time.time()
from cuml.linear_model import LinearRegression
print("cuml.linear_model import: ", time.time() - start, flush=True)

start = time.time()
from statsmodels import robust
print("statsmodels import: ", time.time() - start, flush=True)

# Only import what's needed from cupy
start = time.time()
import cupy as cp
print("cupy import: ", time.time() - start, flush=True)
import scanpy as sc

from anndata import AnnData
import rapids_singlecell as rsc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.sparse import issparse
import random  # at top with other imports
from matplotlib.colors import to_hex

start = time.time()
from scipy.io import mmread
print("scipy mmread import: ", time.time() - start, flush=True)

from scipy.sparse.csgraph import connected_components

# import cudf  # Only import if you're using DataFrame manipulation specific to cudf, # If you're not using all of cudf, avoid importing it unless needed

def set_global_seeds(seed: int):
    """
    Set global RNG seeds for Python, NumPy, and CuPy to make
    runs as reproducible as possible across processes.
    """
    random.seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)



def get_h5_files(file_dirs):
    output_files = []

    for file_dir in file_dirs:
        cell_ids_path = os.path.join(file_dir, "cell_ids.txt")   # nuclei (columns)
        gene_ids_path = os.path.join(file_dir, "gene_ids.txt")   # genes (rows)
        mtx_path      = os.path.join(file_dir, "count_matrix.mtx")
        output_file   = os.path.join(file_dir, "count_matrix.h5")

        if os.path.exists(output_file):
            output_files.append(output_file)
            continue

        cell_ids = pd.read_csv(cell_ids_path, header=None, sep=r"\s+", dtype=str)[0].to_numpy()
        gene_ids = pd.read_csv(gene_ids_path, header=None, sep=r"\s+", dtype=str)[0].to_numpy()

        X = mmread(mtx_path).tocsr()  # genes x nuclei

        # sanity check: genes x nuclei
        if X.shape != (len(gene_ids), len(cell_ids)):
            raise ValueError(
                f"Shape mismatch (expected genes x nuclei): "
                f"X={X.shape}, genes={len(gene_ids)}, nuclei={len(cell_ids)}"
            )

        with h5py.File(output_file, "w") as h5f:
            X_grp = h5f.create_group("X")
            X_grp.create_dataset("data",    data=X.data,    compression="gzip")
            X_grp.create_dataset("indices", data=X.indices, compression="gzip")
            X_grp.create_dataset("indptr",  data=X.indptr,  compression="gzip")
            X_grp.attrs["shape"] = X.shape

            # genes = rows, nuclei = columns
            h5f.create_dataset("obs/_index", data=np.asarray(gene_ids, dtype="S"))
            h5f.create_dataset("var/_index", data=np.asarray(cell_ids, dtype="S"))

            h5f.attrs["orientation"] = "genes_x_nuclei"
            h5f.attrs["sparse_format"] = "csr"
            h5f.attrs["source_format"] = "mtx"

        output_files.append(output_file)

    return output_files



def parse_conditions(conditions_row):
    condition_params = conditions_row.to_dict()
    # Split semicolon-separated strings into lists
    for key, value in condition_params.items():
        if isinstance(value, str) and ';' in value:
            condition_params[key] = [v.strip() for v in value.split(';') if v.strip() != ""]
    # Convert lists of numeric strings into lists of ints
    for key, value in condition_params.items():
        if isinstance(value, list) and all(isinstance(v, str) and v.isdigit() for v in value):
            condition_params[key] = [int(v) for v in value]
    # Handle numeric/non-string scalars: leave as-is (don’t wrap)
    for key, value in condition_params.items():
        # Use a safe scalar NaN check
        is_scalar = np.isscalar(value)
        is_nan = is_scalar and pd.isna(value)
        if isinstance(value, (int, float, np.integer, np.floating, bool)) or is_nan:
            continue
        # Wrap only non-numeric strings
        if not isinstance(value, list):
            condition_params[key] = [value]
    # Add n_<key> entries
    condition_params.update({
        f"n_{key}": 0 if (np.isscalar(value) and pd.isna(value)) else
                    (len(value) if isinstance(value, list) else 1)
        for key, value in condition_params.items()
    })
    return condition_params

def clear_graph_and_cluster_state(adata):
    for k in ["connectivities", "distances"]:
        if k in adata.obsp:
            del adata.obsp[k]

    if "neighbors" in adata.uns:
        del adata.uns["neighbors"]

    for k in ["X_pca", "X_umap", "X_tsne"]:
        if k in adata.obsm:
            del adata.obsm[k]

    for k in ["leiden", "louvain"]:
        if k in adata.obs:
            del adata.obs[k]

    return adata


def run_pca_and_neighbors(adata, condition_params):
    """
    Run PCA and kNN graph construction on GPU using RAPIDS.
    Assumes adata is already on GPU (rsc.get.anndata_to_GPU has been called).
    """
    n_pca = condition_params["n_pca_components"]
    n_neighbors = condition_params["n_neighbors"]
    seed = condition_params["random_state"]

    print("Before PCA:", flush = True)
    print(type(adata.X), flush = True)
    print(getattr(adata.X, "format", "not sparse"), flush = True)
    print(adata.X.shape, flush = True)

    # PCA on GPU
    rsc.pp.pca(
        adata,
        n_comps=n_pca,
        zero_center = True,
        chunked=True,
        chunk_size=1000,
        random_state=seed # Supposedly this is ignored in chunked mode, but maybe worth it considered memory tradeoffs of manual scaling.
    )

    # kNN graph on GPU
    rsc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pca,
        random_state=seed
    )

    C = adata.obsp["connectivities"]
    D = adata.obsp["distances"]

    print(adata.uns["neighbors"]["connectivities_key"], flush = True)
    print(adata.uns["neighbors"]["distances_key"], flush = True)

    print("Has X_umap:", "X_umap" in adata.obsm, flush = True)
    print("neighbors params:", adata.uns["neighbors"]["params"], flush = True)



    # degrees = row-sums of connectivities
    deg = np.asarray(C.sum(axis=1)).ravel()

    print("deg min/median/max:", deg.min(), np.median(deg), deg.max(), flush = True)
    print("deg==0 fraction:", np.mean(deg == 0), flush = True)
    print("deg<1e-6 fraction:", np.mean(deg < 1e-6), flush = True)
    print("nnz connectivities:", C.nnz, flush = True)
    print("nnz distances:", D.nnz, flush = True)

    C = adata.obsp["connectivities"]
    diff = (C - C.T).power(2).sum()
    print("symmetry error:", diff, flush = True)

    n_cc, labels = connected_components(C, directed=False)

    print("Connected components:", n_cc, flush = True)

    diag = C.diagonal()
    print("diag min/median/max:", diag.min(), np.median(diag), diag.max(), flush = True)
    print("fraction diag > 1:", np.mean(diag > 1), flush = True)


def run_clustering(adata, condition_params):
    """
    Run Leiden and Louvain clustering on GPU.
    """
    resolution = condition_params["resolution"]
    seed = condition_params["random_state"]

    rsc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=seed
    )

    sc.tl.louvain(
        adata,
        resolution=resolution
    )

    print(f"Leiden clusters: {adata.obs['leiden'].nunique()}", flush = True)
    print(f"Leiden Clustering counts: {adata.obs['leiden'].value_counts()}", flush = True)

    print(f"Louvain clusters: {adata.obs['louvain'].nunique()}", flush = True)
    print(f"Louvain Clustering counts: {adata.obs['louvain'].value_counts()}", flush = True)

    print(f"Connectivities shape: {adata.obsp['connectivities'].shape}", flush = True)
    print(f"Connectivities shape: {adata.obsp['connectivities'].nnz}", flush = True)   # number of edges
    print(f"Adata.uns: {adata.uns['neighbors']}", flush = True)




def run_embeddings(adata, condition_params):
    """
    Run UMAP and t-SNE on CPU via Scanpy for reproducible embeddings.
    Assumes PCA / neighbors are already computed (e.g., with RAPIDS)
    and stored in adata (X_pca + neighbors graph).
    """
    seed = condition_params["random_state"]

    # Make sure AnnData is on CPU (Scanpy operates on CPU)
    try:
        rsc.get.anndata_to_CPU(adata)
    except Exception:
        # If it's already on CPU, this will likely throw or be a no-op; that's fine.
        pass

    # --- UMAP (CPU, deterministic with random_state) ---
    sc.tl.umap(
        adata,
        spread=condition_params["spread"],
        min_dist=condition_params["min_dist"],
        random_state=seed,
        alpha = condition_params["alpha"]
    )
    #init_pos=condition_params["init_pos"][0],  # e.g. "spectral" or "random"

    # --- t-SNE (CPU, deterministic with random_state) ---
    sc.tl.tsne(
        adata,
        n_pcs=condition_params["n_tsnePCs"],
        use_rep="X_pca",
        perplexity=condition_params["perplex"],
        early_exaggeration=condition_params["early_exagg"],
        learning_rate=condition_params["learning_rate"],
        random_state=seed,
        n_jobs=1,  # ensures reproducibility
    )


"""

def plot_embeddings(adata, save_dir, file_labels):
    
    Plot UMAP and t-SNE colored by Leiden and Louvain.
    Assumes adata is on CPU (rsc.get.anndata_to_CPU already called).
    
    os.makedirs(save_dir, exist_ok=True)
    sc.set_figure_params(figsize=(6, 5), dpi=300)

    # UMAP - Leiden
    filename = os.path.join(save_dir, f"umap_leiden_{file_labels['cluster_label']}.png")
    sc.pl.umap(adata, color="leiden", frameon=False, legend_loc="right margin", show=False)
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close()

    # UMAP - Louvain
    filename = os.path.join(save_dir, f"umap_louvain_{file_labels['cluster_label']}.png")
    sc.pl.umap(adata, color="louvain", frameon=False, legend_loc="right margin", show=False)
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close()

    # t-SNE - Leiden
    filename = os.path.join(save_dir, f"tsne_leiden_{file_labels['cluster_label']}.png")
    sc.pl.tsne(adata, color="leiden", frameon=False, legend_loc="right margin", show=False)
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close()

    # t-SNE - Louvain
    filename = os.path.join(save_dir, f"tsne_louvain_{file_labels['cluster_label']}.png")
    sc.pl.tsne(adata, color="louvain", frameon=False, legend_loc="right margin", show=False)
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close()
"""


def summarize_hvg_variation_gpu_prescaled(
    hvg_sparse_gpu,
    gene_names,
    output_dir,
    zero_var_eps=1e-8,
):
    """
    Compute per-gene stats on a GPU sparse matrix *before scaling*.

    Parameters
    ----------
    hvg_sparse_gpu : cupyx.scipy.sparse.csr_matrix
        Shape: (n_genes, n_cells) on GPU.
    gene_names : list/array of length n_genes
    output_path : str or None
        If given, write CSV with one row per gene.
    zero_var_eps : float
        Threshold to flag near-zero variance genes.

    Returns
    -------
    qc_df : pandas.DataFrame with columns:
        ['gene', 'mean', 'variance', 'std', 'frac_zero', 'row_index']
    """

    n_genes, n_cells = hvg_sparse_gpu.shape

    # Sum and sum of squares per gene (rows)
    sums = cp.asarray(hvg_sparse_gpu.sum(axis=1)).ravel()

    squared = hvg_sparse_gpu.copy()
    squared.data = squared.data ** 2
    sums_sq = cp.asarray(squared.sum(axis=1)).ravel()

    means = sums / n_cells
    variances = (sums_sq / n_cells) - means ** 2
    stds = cp.sqrt(cp.maximum(variances, 0.0))

    # Fraction of zeros per gene
    nnz_per_gene = cp.diff(cp.asarray(hvg_sparse_gpu.indptr))
    frac_zero = 1.0 - (nnz_per_gene / n_cells)

    # Move to CPU
    qc_df = pd.DataFrame(
        {
            "gene": np.asarray(gene_names),
            "mean": means.get(),
            "variance": variances.get(),
            "std": stds.get(),
            "frac_zero": frac_zero.get(),
            "row_index": np.arange(n_genes),
        }
    )

    print("=== HVG QC (pre-scaled, GPU) ===")
    print(f"Total HVGs: {len(qc_df)}")
    print(f"Mean variance:   {qc_df['variance'].mean():.4f}")
    print(f"Median variance: {qc_df['variance'].median():.4f}")
    n_zero = (qc_df["variance"] <= zero_var_eps).sum()
    print(f"Genes with near-zero variance (≤ {zero_var_eps}): {n_zero}")

    qc_df.to_csv(os.path.join(output_dir, "hvg_qc_prescaled.csv"), index=False)

    return qc_df



def compute_cluster_qc(
    adata,
    cluster_key: str = "leiden",
    layer: str | None = None,
) -> pd.DataFrame:
    """
    Recompute cluster-level QC metrics on a (potentially filtered) AnnData.

    Returns
    -------
    cluster_summary : pd.DataFrame
        index: cluster labels
        columns:
            - cluster_size  (# cells per cluster)
            - cluster_means (mean of per-gene means in that cluster)
            - cluster_SDs   (SD of per-gene means in that cluster)
    """

    # --- Get data matrix (cells x genes) ---
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]

    if issparse(X):
        X = X.toarray()

    # genes x cells (to match how you did it in find_outliers)
    expr = X.T

    # --- Clusters and levels ---
    clusters = adata.obs[cluster_key].astype(str).values
    cluster_levels = np.sort(np.unique(clusters))

    cluster_size = pd.Series(index=cluster_levels, dtype=float)
    cluster_means = pd.Series(index=cluster_levels, dtype=float)
    cluster_SDs = pd.Series(index=cluster_levels, dtype=float)

    # --- Loop over clusters ---
    for cl in cluster_levels:
        sel = clusters == cl
        if not np.any(sel):
            # in case a cluster label exists but has no cells in this filtered adata
            cluster_size.loc[cl] = 0
            cluster_means.loc[cl] = np.nan
            cluster_SDs.loc[cl] = np.nan
            continue

        cluster_data = expr[:, sel]  # genes x cells_in_cluster

        # mean per gene within this cluster
        feature_means = cluster_data.mean(axis=1)  # genes

        cluster_size.loc[cl] = sel.sum()
        cluster_means.loc[cl] = feature_means.mean()
        cluster_SDs.loc[cl] = feature_means.std(ddof=1)

    cluster_summary = pd.DataFrame(
        {
            "cluster_size": cluster_size,
            "cluster_means": cluster_means,
            "cluster_SDs": cluster_SDs,
        }
    )
    cluster_summary.index.name = "cluster"

    return cluster_summary

def save_scanpy_plot(filename, dpi=300, right=0.80):
    """
    Call immediately after sc.pl.umap/tsne(..., show=False).
    Reserves right margin so 'right margin' legends aren't clipped.
    """
    fig = plt.gcf()

    # Make room for the legend in the saved canvas
    fig.subplots_adjust(right=right)

    # Reduce extra whitespace around axes
    ax = plt.gca()
    ax.margins(0.02)

    fig.savefig(filename, dpi=dpi, bbox_inches=None)
    plt.close(fig)


def plot_embeddings(adata, save_dir, file_labels):
    """
    Plot UMAP and t-SNE colored by Leiden and Louvain.
    Assumes adata is on CPU (rsc.get.anndata_to_CPU already called).
    """
    os.makedirs(save_dir, exist_ok=True)
    sc.set_figure_params(figsize=(6, 5), dpi=300)

    # --- Stabilize category order + colors for Leiden and Louvain ---
    for key in ["leiden", "louvain"]:
        if key in adata.obs:
            # Ensure categorical dtype
            cats = adata.obs[key].astype("category")

            # Sort categories numerically if they look like "0","1","2", else lexicographically
            try:
                sorted_cats = sorted(cats.cat.categories, key=lambda x: int(x))
            except (ValueError, TypeError):
                sorted_cats = sorted(cats.cat.categories)

            cats = cats.cat.reorder_categories(sorted_cats, ordered=True)
            adata.obs[key] = cats

            # Stable color map: cluster i -> color i in tab20 (cycling if > 20)
            cmap = plt.get_cmap("tab20")
            n = len(sorted_cats)
            colors = [to_hex(cmap(i % cmap.N)) for i in range(n)]
            adata.uns[f"{key}_colors"] = colors

    # ---------- PLOTS ----------


    # UMAP - Leiden
    print("\n--- Starting UMAP Leiden ---\n", flush = True)
    filename = os.path.join(save_dir, f"umap_leiden_{file_labels['cluster_label']}.png")
    sc.pl.umap(adata, color="leiden", frameon=False, legend_loc="right margin", show=False)
    save_scanpy_plot(filename)

    print("\n--- Starting UMAP Louvain ---\n", flush = True)
    # UMAP - Louvain
    filename = os.path.join(save_dir, f"umap_louvain_{file_labels['cluster_label']}.png")
    sc.pl.umap(adata, color="louvain", frameon=False, legend_loc="right margin", show=False)
    save_scanpy_plot(filename)

    print("\n--- Starting tSNE Leiden ---\n", flush = True)
    # t-SNE - Leiden
    filename = os.path.join(save_dir, f"tsne_leiden_{file_labels['cluster_label']}.png")
    sc.pl.tsne(adata, color="leiden", frameon=False, legend_loc="right margin", show=False)
    save_scanpy_plot(filename)

    print("\n--- Starting tSNE Louvain ---\n", flush = True)
    # t-SNE - Louvain
    filename = os.path.join(save_dir, f"tsne_louvain_{file_labels['cluster_label']}.png")
    sc.pl.tsne(adata, color="louvain", frameon=False, legend_loc="right margin", show=False)
    save_scanpy_plot(filename)



def run_before_after_embeddings(
    adata,
    condition_params,
    save_dir,
    file_labels,
    covariates_df,
    results_list,
    mad_thres
):
    """
    Runs RAPIDS PCA → neighbors → clustering → embeddings before and after
    outlier removal, and saves 4 plots:
        - umap_before
        - tsne_before
        - umap_after
        - tsne_after
    """

    # ============================
    # 1) BEFORE OUTLIER REMOVAL
    # ============================

    seed = condition_params["random_state"]
    set_global_seeds(seed)
    print("Set global seeds", flush=True)

    print("\n--- Running BEFORE outlier removal ---\n", flush = True)

    # Move to GPU
    rsc.get.anndata_to_GPU(adata)

    print("\n--- Starting PCA/neighbors ---\n", flush = True)
    # PCA + neighbors + clustering
    run_pca_and_neighbors(adata, condition_params)

    print("\n--- Starting Clustering ---\n", flush = True)
    run_clustering(adata, condition_params) # Maximize complexity heuristic, iteratively

    # Embeddings (UMAP + t-SNE)
    print("\n--- Starting Embeddings ---\n", flush = True)
    run_embeddings(adata, condition_params)

    # Bring to CPU for plotting
    rsc.get.anndata_to_CPU(adata)

    # Save BEFORE plots
    plot_embeddings(
        adata,
        save_dir=os.path.join(save_dir, "before_outlier_removal"),
        file_labels=file_labels,
    )

    
    # ============================
    # 2) OUTLIER REMOVAL
    # ============================

    print("\n--- Running Outlier Detection ---\n")

    results_list, drop_cells = find_outliers(
        adata,
        covariate_df=covariates_df,
        mad_thres=mad_thres,
        results_list=results_list,
        cluster_key="leiden",
        layer=None,
        cov_cells_col="number_of_cells",
    )

    print(f"Dropping {len(drop_cells)} outlier nuclei...")
    adata = adata[~adata.obs_names.isin(drop_cells)].copy()
    adata = clear_graph_and_cluster_state(adata)

    # ============================
    # 3) AFTER OUTLIER REMOVAL
    # ============================

    print("\n--- Running AFTER outlier removal ---\n")

    set_global_seeds(seed)

    # Move filtered data back to GPU
    rsc.get.anndata_to_GPU(adata)

    # PCA + neighbors + clustering
    run_pca_and_neighbors(adata, condition_params)
    run_clustering(adata, condition_params)

    # Embeddings (UMAP + t-SNE)
    run_embeddings(adata, condition_params)

    # Bring to CPU
    rsc.get.anndata_to_CPU(adata)

    # Save AFTER plots
    plot_embeddings(
        adata,
        save_dir=os.path.join(save_dir, "after_outlier_removal"),
        file_labels=file_labels,
    )
    csv_dir = os.path.join(save_dir, "results_csv")
    os.makedirs(csv_dir, exist_ok=True)

    for i, df in enumerate(results_list):
        filename = os.path.join(csv_dir, f"result_{i+1}.csv")
        df.to_csv(filename, index=True)
        print(f"Saved: {filename}")

    cluster_qc_post = compute_cluster_qc(
    adata,
    cluster_key="leiden",
    layer=None,  # or your scaled layer name if you use one
    )

    cluster_qc_post.to_csv(os.path.join(save_dir, "cluster_qc_post_outlier_removal.csv"))

    return adata, results_list



def mad(x: np.ndarray) -> float:
    """
    Median absolute deviation with constant = 1 (to match R's mad(..., constant = 1)).
    """
    x = np.asarray(x)
    med = np.median(x)
    return np.median(np.abs(x - med))

def find_outliers(
    adata,
    covariate_df: pd.DataFrame,
    mad_thres: int,
    results_list: list,
    cluster_key: str = "leiden",
    layer: str | None = None,
    cov_cells_col: str = "number_of_cells",
):
    """
    Python translation of the R find_outliers() for single-cell data.

    Parameters
    ----------
    adata : AnnData
        AnnData object with scaled expression.
    covariate_df : pd.DataFrame
        Covariate data with index matching adata.obs_names and a column
        giving the estimated number of cells per sample (cov_cells_col).
    mad_thres : int
        Baseline MAD threshold used in the original R code.
    results_list : list
        List to which summary DataFrames will be appended (like in R).
    cluster_key : str
        Column in adata.obs containing cluster labels (e.g. "seurat_clusters",
        "leiden", or "louvain").
    layer : str or None
        If not None, use adata.layers[layer] as the scaled matrix.
        Otherwise use adata.X.
    cov_cells_col : str
        Column in covariate_df with the 'Estimated.Number.of.Cells'.

    Returns
    -------
    results_list : list
        Updated list with 3 new entries:
        [cluster_summary_df, sample_summary_df, drop_cells_df]
    drop_cells : np.ndarray
        Array of cell IDs to drop.
    """

    # --- Get scaled data (genes x cells to match Seurat scale.data) ---
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]

    if issparse(X):
        X = X.toarray()
    # AnnData is cells x genes; R's scale.data is genes x cells
    expr = X.T  # genes x cells

    n_cells = expr.shape[1]

    # --- Clusters and cluster levels ---
    clusters = adata.obs[cluster_key].astype(str).values
    cluster_levels = np.sort(np.unique(clusters))

    # --- MAD thresholds: c(mad_thres - 1, seq(mad_thres, mad_thres + 2, by = 1)) ---
    test_mad_thres = np.concatenate(
        [[mad_thres - 1], np.arange(mad_thres, mad_thres + 3)]
    )
    k_thres = len(test_mad_thres)

    # --- Outlier matrix: cells x thresholds ---
    cell_ids = np.asarray(adata.obs_names)
    outliers = np.zeros((n_cells, k_thres), dtype=bool)
    outlier_cols = [f"mad_{t}" for t in test_mad_thres]

    # --- Cluster-level stats ---
    cluster_size = pd.Series(index=cluster_levels, dtype=float)
    cluster_means = pd.Series(index=cluster_levels, dtype=float)
    cluster_SDs = pd.Series(index=cluster_levels, dtype=float)

    # --- Derive sample IDs from cell IDs (prefix before last "_") ---
    first = cell_ids[0]
    parts = first.split("_")
    barcode_len = len(parts[-1])
    id_len = len(first)
    # This mimics substring(rownames(outliers), 1, id_length - (barcode_length+1))
    sample_ids = np.array([
        cid[: id_len - (barcode_len + 1)] for cid in cell_ids
    ])
    unique_sample_ids = pd.unique(sample_ids)

    # --- Complexity matrix: samples x clusters ---
    complexity = pd.DataFrame(
        0,
        index=unique_sample_ids,
        columns=[f"compl_{cl}" for cl in cluster_levels],
        dtype=float,
    )

    # --- Loop over clusters (Seurat: for each cluster) ---
    for cl in cluster_levels:
        sel = clusters == cl
        cluster_data = expr[:, sel]  # genes x cells_in_cluster

        # Row means over genes -> feature means
        feature_means = cluster_data.mean(axis=0 if cluster_data.ndim == 1 else 0)
        # (Be explicit: should be axis=1, but for safety:)
        feature_means = cluster_data.mean(axis=1)

        cluster_size.loc[cl] = sel.sum()
        cluster_means.loc[cl] = feature_means.mean()
        cluster_SDs.loc[cl] = feature_means.std(ddof=1)  # R's sd uses n-1

        # dev_scores: per-cell average absolute deviation from gene means
        # cluster_data: genes x cells_in_cluster
        dev_scores = np.mean(
            np.abs(cluster_data - feature_means[:, None]),
            axis=0,
        )

        med_dev = np.median(dev_scores)
        dev_mad = mad(dev_scores)  # constant = 1

        # scaled deviations; division by 0 -> inf (which will be considered outlier)
        scaled = (dev_scores - med_dev) / dev_mad if dev_mad != 0 else (
            dev_scores - med_dev
        ) / 0.0

        # fill outlier matrix for each threshold
        for j, thresh in enumerate(test_mad_thres):
            outliers[sel, j] = np.abs(scaled) > thresh

        # --- Complexity matrix for baseline threshold mad_thres ---
        j0 = int(np.where(test_mad_thres == mad_thres)[0][0])
        not_outlier_baseline = ~outliers[sel, j0]  # cells in this cluster

        df_temp = pd.DataFrame({
            "sample_id": sample_ids[sel],
            "not_outlier": not_outlier_baseline.astype(int),
        })
        temp = df_temp.groupby("sample_id")["not_outlier"].sum()

        # Update complexity (samples x clusters)
        complexity.loc[temp.index, f"compl_{cl}"] = temp.values

    # --- Cells to drop: baseline threshold ---
    j0 = int(np.where(test_mad_thres == mad_thres)[0][0])
    sel_drop = outliers[:, j0]
    drop_cells = cell_ids[sel_drop]

    # --- Outliers by sample x cluster (baseline threshold) ---
    j0 = int(np.where(test_mad_thres == mad_thres)[0][0])

    df_out = pd.DataFrame({
        "sample_id": sample_ids,
        "cluster": clusters.astype(str),
        "is_outlier": outliers[:, j0].astype(int),   # 1 if outlier at baseline threshold
    })

    out_by_cluster = (
        df_out.groupby(["sample_id", "cluster"])["is_outlier"]
        .sum()
        .unstack(fill_value=0)
    )

    # Rename columns to something explicit
    out_by_cluster.columns = [f"outl_{cl}" for cl in out_by_cluster.columns]
    out_dist = out_by_cluster.div(out_by_cluster.sum(axis=1).replace(0, np.nan), axis=0)
    out_dist.columns = [c.replace("outl_", "outlprop_") for c in out_dist.columns]



    # --- Covariate aggregation ---
    # Align covariate_df to cell order
    covariate_df = covariate_df.loc[cell_ids]
    ids_for_cov = sample_ids

    df_cov = covariate_df[[cov_cells_col]].copy()
    df_cov["sample_id"] = ids_for_cov

    # sample_covs: per-sample Estimated.Number.of.Cells (take first, like FUN = head, 1)
    sample_covs = df_cov.groupby("sample_id")[cov_cells_col].first()
    sample_covs.name = cov_cells_col  # ensures name isn't lost

    # --- n_sampleOutlier: # outliers per sample per threshold ---
    outlier_df = pd.DataFrame(outliers, columns=outlier_cols, index=cell_ids)
    outlier_df["sample_id"] = sample_ids
    n_sampleOutlier = outlier_df.groupby("sample_id")[outlier_cols].sum()

    # Proportion of outliers per sample per threshold
    prop_cells = n_sampleOutlier.div(sample_covs, axis=0)
    prop_cells.columns = [f"prop_{c}" for c in prop_cells.columns]

    # --- n_cells: # non-outlier cells per sample at baseline threshold ---
    df_ncells = pd.DataFrame({
        "sample_id": sample_ids,
        "non_outlier": (~outliers[:, j0]).astype(int),
    })
    n_cells_series = df_ncells.groupby("sample_id")["non_outlier"].sum()
    n_cells_series.name = "n_cells"

    # --- Align complexity rows to sample order ---
    complexity = complexity.loc[n_cells_series.index]
    complexity_prop = complexity.div(sample_covs, axis=0)

    complexity_nclust = (complexity > 0).sum(axis=1)
    complexity_sd = complexity_prop.std(axis=1, ddof=0)

    # --- Build result DataFrames like the R results_list entries ---

    # 1) Cluster-level summary
    cluster_summary = pd.DataFrame({
        "cluster_size": cluster_size,
        "cluster_means": cluster_means,
        "cluster_SDs": cluster_SDs,
    })
    cluster_summary.index.name = "cluster"

    # 2) Sample-level summary
    sample_summary = pd.concat(
        [
            n_cells_series,
            complexity_nclust.rename("complexity_nclust"),
            complexity_sd.rename("complexity_sd"),
            complexity, # compl_* (non-outliers per cluster)
            out_by_cluster, # outl_*  (outliers per cluster)
            n_sampleOutlier,
            prop_cells,
        ],
        axis=1,
    )
    sample_summary.index.name = "sample_id"
    sample_summary = sample_summary.reset_index()

    # 3) Dropped cells
    drop_cells_df = pd.DataFrame({"drop_cells": drop_cells})

    # Append to results_list like in R
    results_list = list(results_list)  # copy / ensure it's mutable
    results_list.append(cluster_summary)
    results_list.append(sample_summary)
    results_list.append(drop_cells_df)

    return results_list, drop_cells


# ---- precompute ONCE (outside the batch loop) ----

def factor2dummy_once(covariates_df: pd.DataFrame, covariate_names):
    cov = pd.DataFrame(covariates_df.loc[:, covariate_names])  # preserve order
    if cov.shape[1] == 1:
        cov = cov.copy()
        cov['extra_var'] = 1

    dummies = pd.DataFrame(index=cov.index)
    for col in cov.columns:
        if (cov[col].dtype == 'object') or pd.api.types.is_categorical_dtype(cov[col]):
            tmp = pd.get_dummies(cov[col], drop_first=True)
            if cov[col].isna().any():
                tmp = tmp.reindex(cov.index)
            dummies = pd.concat([dummies, tmp], axis=1)

    if not dummies.empty:
        categorical_cols = cov.select_dtypes(include=['object', 'category']).columns
        numeric_cols = cov.drop(columns=categorical_cols)
        cov = pd.concat([numeric_cols, dummies], axis=1)

    cov.columns = (
        cov.columns.str.replace('.', '_', regex=False)
                   .str.replace('-', '_', regex=False)
    )

    # Disallow NaN exactly as in the reference
    if cov.isna().any().any():
        raise ValueError("Missing values are not allowed in the covariates")

    # Return a float32 NumPy array (still on CPU at this point)
    return cov.to_numpy(dtype=np.float32)

def compute_Q_on_gpu(cov_numeric_np: np.ndarray, add_intercept: bool = True) -> cp.ndarray:
    # Move covariates to GPU
    X = cp.asarray(cov_numeric_np)                     # (n_nuclei, p’)
    if add_intercept:
        ones = cp.ones((X.shape[0], 1), dtype=cp.float32)
        X = cp.concatenate([ones, X], axis=1)          # (n_nuclei, k)

    # Thin QR on GPU: X = Q R, with Q (n_nuclei x k) orthonormal
    Q, _ = cp.linalg.qr(X, mode='reduced')
    return Q  # (n_nuclei, k)

# Example precompute (before your batching)
# cov_numeric_np = factor2dummy_once(covariates_df, covariate_names)
# Q_gpu = compute_Q_on_gpu(cov_numeric_np, add_intercept=True)


# ---- per-batch residualization (inside your batch loop) ----

def residualize_partial_batch_gpu(partial_sparse_array, Q_gpu: cp.ndarray, return_sparse: bool = False):
    """
    Faithful projection: Y_res = Y - Q(Q^T Y)
    Inputs:
      partial_sparse_array: cupyx CSR (n_genes_batch x n_nuclei)  OR SciPy CSR
      Q_gpu: cp.ndarray, shape (n_nuclei, k) from compute_Q_on_gpu(...)
    Returns:
      Dense cp.ndarray (n_genes_batch x n_nuclei), or GPU CSR if return_sparse=True
    """
    # Ensure GPU CSR
    if isinstance(partial_sparse_array, cpx_sparse.csr_matrix):
        Y_csr_gpu = partial_sparse_array
    else:
        # moves SciPy CSR to GPU
        Y_csr_gpu = cpx_sparse.csr_matrix(partial_sparse_array)

    # Densify as nuclei x genes (samples in rows), exactly like the reference
    Y = Y_csr_gpu.T.toarray().astype(cp.float32)        # (n_nuclei x n_genes_batch)

    # Faithful projection: Y - Q @ (Q.T @ Y)
    Y_res = Y - Q_gpu @ (Q_gpu.T @ Y)                   # (n_nuclei x n_genes_batch)

    R = Y_res.T                                         # (n_genes_batch x n_nuclei)

    if return_sparse:
        return cpx_sparse.csr_matrix(R)
    return R  # dense cp.ndarray


def _is_nan(x):
    try:
        return pd.isna(x) or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False

def _ensure_list(x):
    if x is None or _is_nan(x):
        return []
    return x if isinstance(x, (list, tuple)) else [x]

def _crc32_hex(s: str) -> str:
    return f"{zlib.crc32(s.encode('utf-8')) & 0xffffffff:08x}"

def build_cluster_label(condition_params):
    """
    Emulates your R scheme but:
      • Supports multiple files for data_file / covariates_file
      • Includes only the specified analysis settings below
    """

    # ----- files (can be lists) -----
    data_files       = _ensure_list(condition_params.get("data_file"))
    covariates_files = _ensure_list(condition_params.get("covariates_file"))

    # basenames; keep input order but drop Nones/NaNs; cast to str
    base_data = [os.path.basename(str(p)) for p in data_files if p is not None and not _is_nan(p)]
    base_cov  = [os.path.basename(str(p)) for p in covariates_files if p is not None and not _is_nan(p)]

    # file_hash: paste(c(basename(data_file), basename(covariates_file)), collapse = "\t")
    file_hash_input = "\t".join(base_data + base_cov)
    file_hash = _crc32_hex(file_hash_input)

    # ----- covariate sets -> cov_hash -----
    covariates  = _ensure_list(condition_params.get("covariates"))
    sampleCovs  = _ensure_list(condition_params.get("sampleCovs"))
    merged_covs = [str(c) for c in (covariates + sampleCovs) if c is not None and not _is_nan(c)]

    # cov_hash: digest(paste(sort(c(covariates, sampleCovs)), collapse = "\t"), "crc32")
    cov_hash_input = "\t".join(sorted(merged_covs))
    cov_hash = _crc32_hex(cov_hash_input)

    # ----- analysis settings (ONLY the ones you asked for) -----
    # resolution min_dist random_state n_neighbors batch_size exp_thresh remove_samples
    # covariates sampleCovs n_variableGenes n_pca_components mad_thres remove_clusters
    settings = {
        "alpha": condition_params.get("alpha"),
        "learning_rate":           condition_params.get("learning_rate"),
        "early_exagg":           condition_params.get("early_exagg"),
        "perplex":           condition_params.get("perplex"),
        "spread":           condition_params.get("spread"),
        "resolution":       condition_params.get("resolution"),
        "min_dist":         condition_params.get("min_dist"),
        "init_pos":         condition_params.get("init_pos"), # [0]
        "random_state":     condition_params.get("random_state"),
        "n_neighbors":      condition_params.get("n_neighbors"),
        "batch_size":       condition_params.get("batch_size"),
        "exp_thresh":       condition_params.get("exp_thresh"),
        "remove_samples":   condition_params.get("remove_samples"),
        "n_variableGenes":  condition_params.get("n_variableGenes"),
        "n_pca_components": condition_params.get("n_pca_components"),
        "mad_thres":        condition_params.get("mad_thres"),
        "remove_clusters":  condition_params.get("remove_clusters"),
        "hvg_var_ceiling": condition_params.get("hvg_var_ceiling"),
        "hvg_mad_thresh": condition_params.get("hvg_mad_thresh")
    }

    def fmt_val(v):
        if isinstance(v, (list, tuple)):
            parts = [str(x) for x in v if not _is_nan(x)]
            return "_".join(parts) if parts else "NA"
        if v is None or _is_nan(v):
            return "NA"
        return str(v)

    # Build cluster_label using ONLY requested analysis settings (+ file/cov hashes)
    # (You did not ask to include n_covariates or n_tsnePCs, so we omit them.)
    cluster_label = (
        f"f{file_hash}"
        f"_cov{cov_hash}"
        f"_alpha{fmt_val(settings['alpha'])}"
        f"_learning_rate{fmt_val(settings['learning_rate'])}"
        f"_early_exagg{fmt_val(settings['early_exagg'])}"
        f"_perplex{fmt_val(settings['perplex'])}"
        f"_spread{fmt_val(settings['spread'])}"
        f"_res{fmt_val(settings['resolution'])}"
        f"_init_pos{fmt_val(settings['init_pos'])}"
        f"_min{fmt_val(settings['min_dist'])}"
        f"_rs{fmt_val(settings['random_state'])}"
        f"_nn{fmt_val(settings['n_neighbors'])}"
        f"_bs{fmt_val(settings['batch_size'])}"
        f"_exp{fmt_val(settings['exp_thresh'])}"
        f"_rem{fmt_val(settings['remove_samples'])}"
        f"_varGen{fmt_val(settings['n_variableGenes'])}"
        f"_pca{fmt_val(settings['n_pca_components'])}"
        f"_mad{fmt_val(settings['mad_thres'])}"
        f"_rmCl{fmt_val(settings['remove_clusters'])}"
        f"_hvgCeil{fmt_val(settings['hvg_var_ceiling'])}"
        f"_hvg_mad_thresh{fmt_val(settings['hvg_mad_thresh'])}"
    )

    return {
        "file_hash": file_hash,
        "file_hash_input": file_hash_input,  # debug aid
        "cov_hash": cov_hash,
        "cov_hash_input": cov_hash_input,    # debug aid
        "cluster_label": cluster_label,
    }
def _cellranger_hvg(
    mean,
    mean_sq,
    n_genes,
    n_nuclei,
    n_top_genes=4000,
    gene_names=None,
    output_dir=None,
    hvg_mad_threshold=3,
    restricted_biotypes=None, # {"unprocessed_pseudogene", "transcribed_unprocessed_pseudogene", "IG_V_pseudogene"}
    restricted_chromosomes={"Y", "X", "MT"},
):
    """
    CellRanger-style HVG selection with BioMart-based RESTRICTION filtering.

    Parameters
    ----------
    mean : np.ndarray or cp.ndarray
        Mean expression per gene
    mean_sq : np.ndarray or cp.ndarray
        Mean squared expression per gene
    n_genes : int
        Number of genes
    n_nuclei : int
        Number of nuclei (cells)
    n_top_genes : int
        Number of HVGs to select (after filtering)
    gene_names : array-like or None
        Gene identifiers
    output_dir : str
        Output directory
    restricted_biotypes : set or None
        Gene biotypes to EXCLUDE (BioMart gene_biotype)
    restricted_chromosomes : set or None
        Chromosomes to EXCLUDE (BioMart chromosome_name)

    Returns
    -------
    np.ndarray
        Boolean mask of selected HVGs (aligned to input gene order)
    """

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    def to_numpy(x):
        return x.get() if cp is not None and isinstance(x, cp.ndarray) else x

    # -----------------------------
    # Input preparation
    # -----------------------------
    mean = mean.copy()
    mean[mean == 0] = 1e-12  # avoid divide-by-zero

    variance = mean_sq - mean ** 2
    variance *= n_genes / (n_nuclei - 1)
    dispersion = variance / mean

    if gene_names is None:
        gene_names = np.arange(mean.shape[0])

    mean = to_numpy(mean)
    variance = to_numpy(variance)
    dispersion = to_numpy(dispersion)
    gene_names = to_numpy(gene_names)

    # -----------------------------
    # Base dataframe
    # -----------------------------
    df = pd.DataFrame(
        {
            "gene": gene_names,
            "mean": mean,
            "variance": variance,
            "dispersion": dispersion,
        }
    )

    # -----------------------------
    # Mean binning (CellRanger)
    # -----------------------------
    df["mean_bin"] = pd.cut(
        df["mean"],
        np.r_[
            -np.inf,
            np.percentile(df["mean"], np.arange(10, 105, 10)),
            np.inf,
        ],
    )

    # -----------------------------
    # Normalize dispersion within bins
    # -----------------------------
    disp_grouped = df.groupby("mean_bin")["dispersion"]
    disp_median_bin = disp_grouped.median()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disp_mad_bin = disp_grouped.apply(robust.mad)

    df["dispersion_norm"] = (
        (df["dispersion"].values - disp_median_bin[df["mean_bin"].values].values)
        / disp_mad_bin[df["mean_bin"].values].values
    )

    # -----------------------------
    # Write unannotated QC output
    # -----------------------------
    df.to_csv(
        os.path.join(output_dir, "all_genes_unannotated.csv"),
        index=False,
    )

    # -----------------------------
    # Merge BioMart annotations (ALL genes)
    # -----------------------------
    biomart_path = os.path.join(output_dir, "all_genes_annotated_biomart.csv")
    annotated_df = df.copy()

    if os.path.exists(biomart_path):
        biomart_df = pd.read_csv(biomart_path)

        annotated_df = annotated_df.merge(
            biomart_df,
            left_on="gene",
            right_on="gene_id",
            how="left",
            sort=False,
        )

        annotated_df.to_csv(
            os.path.join(output_dir, "all_genes_annotated.csv"),
            index=False,
        )
    else:
        print(
            "Warning: all_genes_annotated_biomart.csv not found — skipping annotation merge."
        )

    # -----------------------------
    # Apply RESTRICTION filters
    # -----------------------------
    filtered_df = annotated_df.copy()

    if restricted_biotypes is not None:
        print("Assessing restricted biotypes", flush = True)
        n_biotypes_before = len(filtered_df)
        filtered_df = filtered_df[
            ~filtered_df["gene_biotype"].isin(restricted_biotypes)
        ]
        print("Before and after of genes: ", len(filtered_df) - n_biotypes_before, flush = True)


    if restricted_chromosomes is not None:
        print("Assessing restricted chrosomes", flush = True)
        n_genes_before = len(filtered_df)
        filtered_df = filtered_df[
            ~filtered_df["chromosome_name"].isin(restricted_chromosomes)
        ]
        print("Before and after of genes: ", len(filtered_df) - n_genes_before, flush = True)

    # Drop NaNs from normalization
    filtered_df = filtered_df.dropna(subset=["dispersion_norm"])

    ### MAD outlier detections


    disp = filtered_df["dispersion_norm"].values
    disp_median = np.median(disp)
    disp_mad = robust.mad(disp)

    if disp_mad == 0:
        print(
            "Warning: MAD is zero — skipping MAD-based filtering.",
            flush=True,
        )
    else:
        scaled = (disp - disp_median) / disp_mad
        n_before = len(filtered_df)
        filtered_df = filtered_df.loc[np.abs(scaled) <= hvg_mad_threshold]
        print(
            f"MAD filter removed {n_before - len(filtered_df)} genes",
            flush=True,
        )

    # -----------------------------
    # Select top genes by normalized dispersion
    # -----------------------------
    filtered_df = filtered_df.sort_values(
        by="dispersion_norm",
        ascending=False,
    )

    if n_top_genes > len(filtered_df):
        n_top_genes = len(filtered_df)

    top_df = filtered_df.head(n_top_genes)

    top_df.to_csv(
        os.path.join(output_dir, "hvg_only_filtered_annotated.csv"),
        index=False,
    )

    # -----------------------------
    # Final boolean mask (original order)
    # -----------------------------

    selected_genes = set(top_df["gene"])
    df["is_hvg_filtered"] = df["gene"].isin(selected_genes)

    print("Number of filtered HVGs:", df["is_hvg_filtered"].sum())

    return df["is_hvg_filtered"].values


"""
def _cellranger_hvg(mean, mean_sq, n_genes, n_nuclei, n_top_genes=4000):  # mean, mean_sq, genes, n_cells, n_top_genes
    if n_top_genes is None:
        n_top_genes = n_genes.shape[0] // 10  # n_top_genes = genes.shape[0] // 10

    mean[mean == 0] = 1e-12
    variance = mean_sq - mean ** 2
    variance *= n_genes / (n_nuclei - 1)
    dispersion = variance / mean

    df = pd.DataFrame()
    # Note - can be replaced with cudf once 'cut' is added in 21.08
    # df['genes'] = genes.to_array()
    df['means'] = mean.tolist()
    df['dispersions'] = dispersion.tolist()

    if n_top_genes > df.shape[0]:
        n_top_genes = df.shape[0]

    t = cp.unique(mean)
    print(len(t))

    # Below code assigns a bin for each corresponding value in mean_bin (ex: ((1.3, 2.5]))
    df['mean_bin'] = pd.cut(
        df['means'],
        np.r_[-np.inf, np.percentile(df['means'], np.arange(10, 105, 10)), np.inf],
    )  # CHANGED np.arange from (10, 105, 5) to (10, 105, 10) increasing % jump and bin width

    # All the Below code Normalize dispersion within bins
    disp_grouped = df.groupby('mean_bin')['dispersions']
    disp_median_bin = disp_grouped.median()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        disp_mad_bin = disp_grouped.apply(robust.mad)
        df['dispersions_norm'] = (
                                         df['dispersions'].values - disp_median_bin[df['mean_bin'].values].values
                                 ) / disp_mad_bin[df['mean_bin'].values].values

    # Select Top N HVGs, using normalized dispersion values
    dispersion_norm = df['dispersions_norm'].values
    dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
    dispersion_norm[::-1].sort()

    # Order is preserved: df['disperson_norms'] and variable_genes mask are still in the same order as the original mean and mean_sq list
    disp_cut_off = dispersion_norm[n_top_genes - 1]
    variable_genes = np.nan_to_num(df['dispersions_norm'].values) >= disp_cut_off
    print(sum(variable_genes))  # Correctly outputs 4000, as defined above
    return variable_genes
"""


def get_shared_genes(all_files, gene_key="/gene_ids", nuclei_key="/cell_ids"):
    """
    Find genes shared across all files (intersection) and collect nuclei IDs.

    Returns
    -------
    shared_genes_ordered : list of str
        Genes present in every file, ordered as in the first file.
    nuclei_names_list : list of str
        Concatenated nuclei IDs from all files, in file order.
    """
    nuclei_names_list = []

    shared_genes = None          # set of genes common to all files
    first_gene_order = None      # preserves order from the first file

    for path in all_files:
        with h5py.File(path, "r") as h5f:
            gene_names = h5f[gene_key][:].astype("U") #str
            nuc_names = h5f[nuclei_key][:].astype("U") #str

        # accumulate nuclei IDs in the order they appear
        nuclei_names_list.extend(list(nuc_names))

        # initialize on first file
        if shared_genes is None:
            shared_genes = set(gene_names)
            first_gene_order = list(gene_names)
        else:
            # intersection with genes from this file
            shared_genes &= set(gene_names)

    if not shared_genes:
        raise ValueError("No shared genes found across the provided files.")

    # preserve the ordering from the first file, but only keep intersecting genes
    shared_genes_ordered = [g for g in first_gene_order if g in shared_genes]

    return shared_genes_ordered, nuclei_names_list

def process_cov_files(all_cov_files, nuclei_names):
    all_covariates = []
    for path in all_cov_files:
        df = pd.read_csv(path, index_col=0)
        all_covariates.append(df)

    covariates_df = pd.concat(all_covariates)
    covariates_df = covariates_df.loc[nuclei_names] # Ensure alignment with nuclei names for residualization step accuracy

    return covariates_df


def build_gene_index_map(h5_file, shared_genes, gene_key="/gene_ids"):
    """Build a mapping of gene names to indices."""
    with h5py.File(h5_file, "r") as h5f:
        gene_names = h5f[gene_key][:].astype(str)
        temp = {gene: idx for idx, gene in enumerate(gene_names)}
        subset = {gene:temp[gene] for gene in shared_genes}
    return subset



def residualize(partial_sparse_array, covariates_df):
    # Convert covariates to GPU arrays
    X = cp.asarray(covariates_df.to_numpy(), dtype=cp.float32)

    n_genes = partial_sparse_array.shape[0]
    residualized_rows = []

    model = LinearRegression(fit_intercept=True, algorithm="eig")

    for i in range(n_genes):
        y = partial_sparse_array[i, :].toarray().ravel()
        y = cp.asarray(y, dtype=cp.float32)

        if cp.all(y == 0):
            residualized_rows.append(cp.zeros_like(y))
            continue

        model.fit(X, y)
        y_hat = model.predict(X)
        residuals = y - y_hat
        residualized_rows.append(residuals)

    residual_matrix = cp.vstack(residualized_rows)
    residual_matrix = cp.sparse.csr_matrix(residual_matrix)

    return residual_matrix



def log_normalize_batch(batch): # reduce imbalances in data
    # Step 1: Compute total expression per nucleus (i.e., per column)
    col_sums = batch.sum(axis=0)  # shape: (1, n_nuclei)

    # Step 2: Prevent divide-by-zero
    col_sums = cp.asarray(col_sums).ravel()
    col_sums[col_sums == 0] = 1  # avoid division by 0

    # Step 3: Compute normalization factors
    scale_factor = 1e4  # like Seurat default
    norm_factors = scale_factor / col_sums  # shape: (n_nuclei,)

    # Step 4: Construct a diagonal matrix for scaling columns
    D = diags(norm_factors)  # shape: (n_nuclei, n_nuclei)

    # Step 5: Normalize by multiplying from the right (sparse × diagonal)
    normalized = batch @ D  # shape: (n_genes_batch, n_nuclei)

    # Step 6: Log1p transform (only modifies nonzero values)
    normalized.data = cp.log1p(normalized.data)

    return normalized


def scale_and_clip(partial_sparse_array, clip_thres=10):
    # Convert to dense temporarily for per-row ops
    dense = partial_sparse_array.toarray()  # shape: (n_genes_batch, n_nuclei)

    # Compute per-gene mean and std (axis=1 since genes are rows)
    gene_means = cp.mean(dense, axis=1, keepdims=True)
    gene_stds = cp.std(dense, axis=1, ddof=1, keepdims=True)

    # Avoid divide-by-zero by setting std=1 where it's 0
    gene_stds[gene_stds == 0] = 1

    # Z-score normalize per gene
    scaled = (dense - gene_means) / gene_stds

    # Clip extreme values
    scaled = cp.clip(scaled, -clip_thres, clip_thres)

    # Convert back to sparse if needed
    scaled_sparse = cp.sparse.csr_matrix(scaled)

    return scaled_sparse



def remove_lowexp_genes(partial_sparse_array, batch_genes, thresh):

    n_cells = partial_sparse_array.shape[1]

    gene_sums = cp.asarray(partial_sparse_array.sum(axis=1)).ravel()
    gene_means = gene_sums / n_cells
    expression_mask = gene_means > thresh

    print("CURRENT BATCH")
    print(partial_sparse_array.shape)

    partial_sparse_array = partial_sparse_array[expression_mask, :]

    print(partial_sparse_array.shape)
    print("DONE")

    genes_above_thresh = np.asarray(batch_genes)[expression_mask.get()]

    return partial_sparse_array, genes_above_thresh


def process_gene_batches(output_dir, datasets, gene_batches, gene_maps, Q_gpu, n_genes, total_nuclei, covariates_df, covariate_names, hvg_mad_threshold, data_key="/X/data", indptr_key="/X/indptr", indices_key = "/X/indices", n_top_genes = 100, thresh = 0.001):
    mean = []
    mean_sq = []
    all_genes_above_thresh = []


    for batch_idx, batch_genes in enumerate(gene_batches): # Keep this nested structure to avoid opening/closing IO ineffiiencies (i.e. looping through files first and then the genes withine each batch)
        combined_batch = []

        for file, gene_map in zip(datasets, gene_maps):
            with h5py.File(file, "r") as h5f:
                indptrs = h5f[indptr_key] # h5f[indptr_key][:] <-- this code ends up loding into memory; in this way, we avoid that
                data = h5f[data_key] # This code does not load into memory, which is good
                indices = h5f[indices_key]

                n_nuclei_actual = len(h5f["/cell_ids"])
                #print("Expected n_nuclei:", total_nuclei, "| Actual:", n_nuclei_actual)

                
                batch_rows = []
                for gene in batch_genes:
                    row_idx = gene_map[gene]
                    start = indptrs[row_idx] # here we are splicing without loading into memory
                    end = indptrs[row_idx + 1] # here we are splicing without loading into memory

                    if start == end:
                        #print(f"Skipping gene '{gene}' — no non-zero values")
                        row = csr_matrix((1, n_nuclei_actual), dtype=cp.float32) # Make empty row to ensure alignment
                    else:
                        row_data = cp.array(data[start:end], dtype=cp.float32)
                        row_indices = cp.array(indices[start:end], dtype=cp.int32)
                        row_indptr = cp.array([0, len(row_data)], dtype=cp.int32)
                        row = csr_matrix((row_data, row_indices, row_indptr), shape=(1, n_nuclei_actual))

                    batch_rows.append(row)

                dataset_matrix = (vstack(batch_rows)) # dataset_matrix = csr_matrix(cp.vstack(batch_rows)) - new change
                combined_batch.append(dataset_matrix)

        # Combine data across datasets
        partial_sparse_array = cp.sparse.hstack(combined_batch).tocsr() # hstack returns coo format, we need to convert back. This is your batch across all datasets
        assert partial_sparse_array.shape[1] == total_nuclei

        # REMOVE LOW EXPR genes by calculating the mean expression level for each gene across all nuclei
        partial_sparse_array, genes_above_thresh = remove_lowexp_genes(partial_sparse_array, batch_genes, thresh)
        print("Num genes above thresh: ", len(genes_above_thresh), flush = True)
        # Ensure alignment between matrix rows and gene list
        print("Edited sparse array has only genes above threshold: ", partial_sparse_array.shape[0] == len(genes_above_thresh), flush = True)
        assert partial_sparse_array.shape[0] == len(genes_above_thresh), (
            "Row count after low-exp filter must equal length of genes_above_thresh"
        )

        all_genes_above_thresh.extend(genes_above_thresh)

        # 1) normalize + log
        partial_sparse_array = log_normalize_batch(partial_sparse_array)

        """
        # 2) residualize
        residual = residualize_partial_batch_gpu(partial_sparse_array, Q_gpu, return_sparse=True)
        """
        residual = residualize_partial_batch_gpu(partial_sparse_array, Q_gpu, return_sparse=True)
        #residual = partial_sparse_array
        # 2b) HVG statistics from *residual* (pre-scaled)
        partial_mean = cp.asarray(residual.sum(axis=1)).ravel() / residual.shape[1]
        mean.append(partial_mean)

        residual_sq = residual.copy()
        residual_sq.data **= 2
        partial_mean_sq = cp.asarray(residual_sq.sum(axis=1)).ravel() / residual_sq.shape[1]
        mean_sq.append(partial_mean_sq)

        # 3) Now scale+clip for downstream PCA/UMAP, if you still want scaled input
        #partial_sparse_array = scale_and_clip(residual, clip_thres=10)

        

        # ******* Each value in mean and mean_sq corresponds to a gene, ideally and I belive in the same order as all_genes_above_thresh *******

    mean_lens = [len(i) for i in mean]
    print(mean_lens)

    print("Number of genes that passed the min expression threshold: ", len(all_genes_above_thresh), flush = True)

    mean = cp.concatenate([cp.atleast_1d(m) for m in mean]).ravel() # Double check this line
    mean_sq = cp.concatenate([cp.atleast_1d(msq) for msq in mean_sq]).ravel() # Double check this line

    # Basic sanity checks
    assert mean.shape == mean_sq.shape, "mean and mean_sq must have same shape"
    assert mean.ndim == 1, "mean must be 1D"
    assert mean.shape[0] == len(all_genes_above_thresh), (
        "Stats length must match number of genes collected"
    )

    n_genes_final = int(mean.shape[0])

    variable_genes = _cellranger_hvg(mean=mean,mean_sq=mean_sq,n_genes=n_genes_final,n_nuclei=total_nuclei,n_top_genes=n_top_genes,output_dir=output_dir, gene_names = all_genes_above_thresh, hvg_mad_threshold = hvg_mad_threshold)
    variable_gene_names = np.asarray(all_genes_above_thresh)[variable_genes] # all_genes_above_thresh[variable_genes]

        # Optionally print or return them
    print("Top variable genes:", variable_gene_names[:10])
    
    return variable_gene_names


def sparse_fmt(x):
    return getattr(x, "format", type(x))

def build_hvg_matrix(variable_gene_names, datasets, output_dir, gene_maps, total_nuclei, Q_gpu, data_key="/X/data", indptr_key="/X/indptr", indices_key="/X/indices", clip_thres = 10):
    all_datasets = []
    for file, gene_map in zip(datasets, gene_maps):
        with h5py.File(file, "r") as h5f:
            indptrs = h5f[indptr_key] # h5f[indptr_key][:] <-- this code ends up loding into memory; in this way, we avoid that
            data = h5f[data_key] # This code does not load into memory, which is good
            indices = h5f[indices_key]

            n_nuclei_actual = len(h5f["/cell_ids"])

            batch_rows = []

            for gene in variable_gene_names: # All genes in these batches will have already passed the mean threshold expression
                row_idx = gene_map[gene]
                start = indptrs[row_idx] # here we are splicing without loading into memory
                end = indptrs[row_idx + 1] # here we are splicing without loading into memory

                if start == end:
                    #print(f"Skipping gene '{gene}' — no non-zero values")
                    row = csr_matrix((1, n_nuclei_actual), dtype=cp.float32) # Make empty row to ensure alignment
                else:
                    row_data = cp.array(data[start:end], dtype=cp.float32)
                    row_indices = cp.array(indices[start:end], dtype=cp.int32)
                    row_indptr = cp.array([0, len(row_data)], dtype=cp.int32)
                    row = csr_matrix((row_data, row_indices, row_indptr), shape=(1, n_nuclei_actual))

                batch_rows.append(row)

            dataset_matrix = vstack(batch_rows)
            all_datasets.append(dataset_matrix)


    refined_sparse_array = cp.sparse.hstack(all_datasets, format="csr", dtype=cp.float32)
    print("after hstack:", type(refined_sparse_array), sparse_fmt(refined_sparse_array), flush=True)
    #refined_sparse_array = cp.sparse.hstack(all_datasets).tocsr()  # shape: (n_HVGs, total_nuclei)


    # Guardrail: confirm this matches what you think total_nuclei is
    assert refined_sparse_array.shape[0] == len(variable_gene_names), (
        f"Expected {len(variable_gene_names)} HVGs, got {refined_sparse_array.shape[0]}"
    )
    assert refined_sparse_array.shape[1] == total_nuclei, (
        f"Expected total_nuclei={total_nuclei}, got {refined_sparse_array.shape[1]}"
    )

    # 🔥 Apply the SAME pipeline as in process_gene_batches
    # 1) Normalize + log
    refined_sparse_array = log_normalize_batch(refined_sparse_array)
    print("after log_normalize_batch:", type(refined_sparse_array), sparse_fmt(refined_sparse_array), flush=True)

    # 2) Residualize
    if Q_gpu is None:
        raise ValueError("Q_gpu must be provided to apply residualization.")
    refined_sparse_array = residualize_partial_batch_gpu(
        refined_sparse_array, Q_gpu, return_sparse=True
    )
    print("after residualize:", type(refined_sparse_array), sparse_fmt(refined_sparse_array), flush=True)

    #summarize_hvg_variation_gpu_prescaled(refined_sparse_array,variable_gene_names,output_dir)


    # 3) Scale + clip
    #refined_sparse_array = scale_and_clip(refined_sparse_array, clip_thres=clip_thres)

    return refined_sparse_array


print("TESTING2")

"""
def remove_lowexp_genes(partial_sparse_array, total_nuclei, thresh): # dataset_files, gene_maps, common_genes, n_nuclei, thresh, data_key="/X/data", indptr_key="/X/indptr"

    gene_sums = {gene: 0.0 for gene in common_genes}

    for file, gene_map in zip(dataset_files, gene_maps):
        with h5py.File(file, "r") as h5f:
            indptrs = h5f[indptr_key]
            data = h5f[data_key]
            for gene in common_genes:
                row_idx = gene_map[gene]
                start = indptrs[row_idx]
                end = indptrs[row_idx + 1]
                row_data = cp.array(data[start:end], dtype=cp.float32)
                gene_sums[gene] += row_data.sum().item()

    filtered_genes = [gene for gene in common_genes if (gene_sums[gene] / n_nuclei) > thresh]
    return filtered_genes

OR YOU CAN DO THIS INSTEAD:

    filtered_genes = []
    for gene in common_genes:
        gene_total_expr = 0.0
        for file, gene_map in zip(dataset_files, gene_maps):
            with h5py.File(file, "r") as h5f:
                indptrs = h5f[indptr_key] # h5f[indptr_key][:] <-- this code ends up loding into memory; in this way, we avoid that
                data = h5f[data_key] # This code does not load into memory, which is good
                row_idx = gene_map[gene]
                
                start = indptrs[row_idx] # here we are splicing without loading into memory
                end = indptrs[row_idx + 1] # here we are splicing without loading into memory

                row_data = cp.array(data[start:end], dtype=cp.float32) # here we are splicing without loading into memory
                gene_total_expr += row_data.sum().item()
        mean_expr = gene_total_expr / n_nuclei
        if mean_expr > thresh:
            filtered_genes.append(gene)
    return filtered_genes

"""