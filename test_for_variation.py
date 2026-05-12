print("Python script started now", flush=True)

import time
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scanpy as sc

start = time.time()
import test_functions
print("test_functions import:", time.time() - start, flush=True)

start = time.time()
from anndata import AnnData
print("anndata import:", time.time() - start, flush=True)

start = time.time()
import rapids_singlecell as rsc
print("rapids_singlecell import:", time.time() - start, flush=True)

start = time.time()
import cupy as cp
print("cupy import:", time.time() - start, flush=True)

start = time.time()
import numpy as np
print("numpy import:", time.time() - start, flush=True)

start = time.time()
import pandas as pd
print("pandas import:", time.time() - start, flush=True)

import cudf, cuml, cugraph

print("cudf:", cudf.__version__, flush=True)
print("cuml:", cuml.__version__, flush=True)
print("cugraph:", cugraph.__version__, flush=True)
print("rapids-singlecell:", rsc.__version__, flush=True)

print("Started test.py", flush=True)

# -----------------------------
# Read conditions row
# -----------------------------
csv_path = sys.argv[1]
row_num = int(sys.argv[2]) - 1   # SLURM_ARRAY_TASK_ID is 1-based

conditions = pd.read_csv(csv_path)
row = conditions.iloc[row_num]
condition_params = test_functions.parse_conditions(row)

file_labels = test_functions.build_cluster_label(condition_params)

print("cluster_label:", file_labels["cluster_label"], flush=True)
print("data_file:", condition_params.get("data_file"), flush=True)
print("h5ad_file:", condition_params.get("h5ad_file"), flush=True)

# -----------------------------
# Output paths
# -----------------------------
output_dir = condition_params["output_dir"][0]
out_path = os.path.join(output_dir, "sc_autism.h5ad")

save_dir = os.path.join(
    output_dir,
    "plots",
    file_labels["cluster_label"],
    file_labels["run_id"]
)
os.makedirs(save_dir, exist_ok=False)

test_functions.save_run_metadata(save_dir, condition_params, file_labels)

# -----------------------------
# Load h5ad
# -----------------------------
h5ad_path = condition_params.get("h5ad_file", out_path)
if isinstance(h5ad_path, list):
    h5ad_path = h5ad_path[0]

print(f"Reading H5AD: {h5ad_path}", flush=True)
adata = sc.read_h5ad(h5ad_path)

print("Loaded adata shape:", adata.shape, flush=True)
print("adata.X type:", type(adata.X), flush=True)
print("adata.X format:", getattr(adata.X, "format", "not sparse"), flush=True)

# -----------------------------
# Run PCA / neighbors / clustering / embeddings / plots
# -----------------------------
adata_final, results_list = test_functions.run_before_after_embeddings(
    adata=adata,
    condition_params=condition_params,
    save_dir=save_dir,
    file_labels=file_labels,
    covariates_df=None,
    results_list=[],
    mad_thres=condition_params.get("mad_thres")
)

print("Finished successfully.", flush=True)
print("Final adata shape:", adata_final.shape, flush=True)
print("Plots/output saved to:", save_dir, flush=True)