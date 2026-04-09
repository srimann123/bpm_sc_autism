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
print("Test_functions import: ", time.time() - start, flush=True)

start = time.time()
from anndata import AnnData
print("anndata import: ", time.time() - start, flush=True)

start = time.time()
import rapids_singlecell as rsc
print("rapids_singlecell import: ", time.time() - start, flush=True)

start = time.time()
import cupy as cp
print("cupy import: ", time.time() - start, flush=True)

start = time.time()
import numpy as np
print("numpy import: ", time.time() - start, flush=True)

start = time.time()
import pandas as pd
print("pandas import: ", time.time() - start, flush=True)

import cudf, cuml, cugraph
import rapids_singlecell as rsc

print("cudf:", cudf.__version__, flush=True)
print("cuml:", cuml.__version__, flush=True)
print("cugraph:", cugraph.__version__, flush=True)
print("rapids-singlecell:", rsc.__version__, flush=True)


#start = time.time()
#from cupyx.scipy.sparse import csr_matrix
#print("cupyx.scipy.sparse import: ", time.time() - start, flush=True)

#end_time = time.time()
#print("Total time spent for importing:", end_time - start_time)

######TODO
# Figure out how to track gene names
# Move low_expression genes into process_gene_batches, so that youre only reading data once # Done (need to track the names though)
# Fix bugs in process_gene_batches (fixed partial_mean and partial_mean_sq calculations)
# Can process_gene_batches be modified so that youre only opening/closing each file once? (it's already doing that so we're good)
# _______ ^ DONE
# Using the highly variable genes, construct the new partial sparase array
# Empricially determine the thresh
# Is remove_lowexp genes actually speedin up analyses? Wouldnt HVG take care of this?

# PCA + Cluster
# Outlier (find in cluster.R)
# Handling the parameters in conditons
# Outliers and interface

# 7/24/25: (Handling parameters, shell scripting, otuliers)


print("Started test.py", flush = True)

csv_path = sys.argv[1]
row_num = int(sys.argv[2]) - 1 # SLURM_ARRAY_TASK_ID (1-based index)
conditions = pd.read_csv(csv_path)
row = conditions.iloc[row_num]
condition_params = test_functions.parse_conditions(row)


file_labels = test_functions.build_cluster_label(condition_params)

print(file_labels["cluster_label"], flush=True)


print(condition_params["data_file"])



result = test_functions.get_shared_genes(condition_params["data_file"]) # Genes that are shared genes, intersection across datasets
shared_genes = result[0]
nuclei_names_list = result[1]

print("Shared genes count: ", len(shared_genes), flush = True)

n_genes = len(shared_genes)
total_nuclei = len(nuclei_names_list)

print("Total genes: ", n_genes, flush = True)
print("Total nuclei: ", total_nuclei, flush = True)

covariates_df = test_functions.process_cov_files(condition_params["covariates_file"], nuclei_names_list)

"""

gene_maps = [test_functions.build_gene_index_map(file, shared_genes) for file in condition_params["data_file"]]
print("Shared genes: ", len(shared_genes), flush = True)


gene_batches = [shared_genes[i:i + condition_params["batch_size"]] for i in range(0, len(shared_genes), condition_params["batch_size"])]

print("Starting variable genes function", flush = True)

cov_numeric_np = test_functions.factor2dummy_once(covariates_df, condition_params["covariates"]) # Added code for covariates residualization
Q_gpu = test_functions.compute_Q_on_gpu(cov_numeric_np, add_intercept=True) # # Added code for covariates residualization


variable_gene_names = test_functions.process_gene_batches(condition_params["output_dir"][0], condition_params["data_file"], gene_batches, gene_maps, Q_gpu, n_genes = n_genes, total_nuclei = total_nuclei, covariates_df = covariates_df, covariate_names = condition_params["covariates"], thresh = condition_params["exp_thresh"], n_top_genes = condition_params["n_variableGenes"], hvg_mad_threshold = condition_params["hvg_mad_thresh"])
#np.savetxt("variable_gene_names_new.txt", variable_gene_names, fmt='%s', delimiter='\n')
print(variable_gene_names[0:5], flush = True)

refined_sparse_array = test_functions.build_hvg_matrix(variable_gene_names, condition_params["data_file"], condition_params["output_dir"][0], gene_maps, total_nuclei, Q_gpu, clip_thres = 10)



# Convert CuPy sparse matrix back to CPU (scipy sparse CSR)
refined_sparse_array_cpu = refined_sparse_array.get()

adata = AnnData(X=refined_sparse_array_cpu.T.tocsr())

# Assign correct names
adata.obs_names = nuclei_names_list           # rows = nuclei
adata.var_names = variable_gene_names         # columns = HVGs


"""

output_dir = condition_params["output_dir"][0]
out_path = os.path.join(output_dir, "sc_autism.h5ad")
#adata.write(out_path) # This line needs to be commented out



save_dir = os.path.join(condition_params["output_dir"][0], "plots", file_labels["cluster_label"])
os.makedirs(save_dir, exist_ok=True)  # Create the folder if it doesn’t exist


# 1) Read precomputed matrix (nuclei x HVGs) from H5AD ####### REMOVE THIS BLOCK OF CODE AFTER TESTING
h5ad_path = condition_params.get("h5ad_file", out_path)
# ---------- LOAD DATA ----------
print(f"Reading H5AD: {h5ad_path}", flush=True)
adata = sc.read_h5ad(h5ad_path)



adata_final, results_list = test_functions.run_before_after_embeddings(
    adata=adata,
    condition_params=condition_params,
    save_dir=save_dir,
    file_labels=file_labels,
    covariates_df=covariates_df,
    results_list=[],
    mad_thres=condition_params.get("mad_thres")
)

"""