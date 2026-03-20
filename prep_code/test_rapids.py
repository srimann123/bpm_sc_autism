import h5py
import scipy.sparse as sp
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import rapids_singlecell as rsc
import os
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)
sc.settings.figdir = "plots"

h5ad_path = "/lustre/home/ramachandruss/python_single_cell/test_interface/ctr_converted.h5ad"
adata = sc.read_h5ad(h5ad_path)
print(adata.shape)


rsc.get.anndata_to_GPU(adata)
#adata.layers["counts"] = adata.X.copy()


# Basic QC
rsc.pp.filter_cells(adata, min_genes=200)
rsc.pp.filter_genes(adata, min_cells=3)
print("Done with filtering", flush = True)


rsc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="cell_ranger") # Removed paramter [, layer="counts"], flavor was seurat_v3
print("Done calculating HVGs", flush = True)

rsc.pp.normalize_total(adata, target_sum=1e4)
rsc.pp.log1p(adata)
print("Done normalizing", flush = True)


adata.raw = adata
adata = adata[:, adata.var["highly_variable"]]

print(adata.shape)
rsc.pp.scale(adata, max_value=10)


rsc.pp.pca(adata, n_comps=100, mask_var="highly_variable") # Removed paramater [, use_highly_variable=True]
print("Done with PCA")

rsc.pp.neighbors(adata, n_neighbors=50, n_pcs=45)
rsc.tl.umap(adata, min_dist=1.0, alpha = 10)
rsc.tl.louvain(adata, resolution=0.5)
rsc.tl.leiden(adata, resolution=0.5)
print("Done with neighbors, umap, louvain, leiden")

print("Number of clusters:", adata.obs["leiden"].nunique())
print("Cluster size summary:")
print(adata.obs["leiden"].value_counts().describe())

rsc.get.anndata_to_CPU(adata)

sc.set_figure_params(figsize=(8,8), dpi=150)
sc.pl.pca_variance_ratio(adata, log=True, n_pcs=100, save="_pca_variance.png")

sc.pl.umap(
    adata,
    color="leiden",
    size=2.5,
    alpha=0.9,
    frameon=False,
    legend_loc="right margin"
)
plt.savefig("plots/umap_clusters.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done with plots")




"""
input_h5  = "/lustre/home/BPM/autism_brain/nucleiQC_noFeatureQC/ctr.h5"
output_h5ad = "/lustre/home/ramachandruss/python_single_cell/test_interface/ctr_converted.h5ad"

with h5py.File(input_h5, "r") as f:
    Xg = f["X"]

    data    = Xg["data"][:].astype(np.float32)
    indices = Xg["indices"][:]
    indptr  = Xg["indptr"][:]

    # Reconstruct as CSC (THIS is the fix)
    X = sp.csc_matrix((data, indices, indptr))

    gene_ids = f["gene_ids"][:].astype(str)
    cell_ids = f["cell_ids"][:].astype(str)

X = X.T.tocsr()   # now cells x genes, CSR
adata = ad.AnnData(
    X=X,
    obs=pd.DataFrame(index=cell_ids),
    var=pd.DataFrame(index=gene_ids),
)

adata.write(output_h5ad)

print(adata)
print(adata.X.shape)
print(len(adata.obs), len(adata.var)) # 67937 62710 (cells, genes)
print(sp.issparse(adata.X))
"""


"""
# Below files need to be tranposed
# /lustre/home/BPM/autism_brain/nucleiQC_noFeatureQC/ctr.h5
# /lustre/home/BPM/autism_brain/nucleiQC_noFeatureQC/epilepsy.h5
# /lustre/home/ramachandruss/python_single_cell/krasnow_hlca_10x.sparse.h5ad

print("Done reading")

h5ad_path = "/lustre/home/BPM/autism_brain/nucleiQC_noFeatureQC/ctr_converted.h5ad"
adata = sc.read_h5ad(h5ad_path)
rsc.get.anndata_to_GPU(adata)

print(adata.shape)

rsc.pp.normalize_total(adata, target_sum=1e4)
rsc.pp.log1p(adata)

rsc.pp.highly_variable_genes(
    adata, n_top_genes=5000, flavor="seurat_v3", layer="counts"
)

rsc.get.anndata_to_CPU(adata, layer="counts")
adata.raw = adata

adata = adata[:, adata.var["highly_variable"]]

print(adata.shape)
#rsc.pp.scale(adata, max_value=10)

rsc.pp.pca(adata, n_comps=100, use_highly_variable=False)
sc.pl.pca_variance_ratio(adata, log=True, n_pcs=100)

rsc.get.anndata_to_CPU(adata)
rsc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
rsc.tl.umap(adata, min_dist=0.3)

rsc.tl.louvain(adata, resolution=0.6)
rsc.tl.leiden(adata, resolution=1.0)

sc.pl.umap(adata, color=["louvain", "leiden"], legend_loc="on data")


"""
