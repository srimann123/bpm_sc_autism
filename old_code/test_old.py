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

# Testing github
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

adata = AnnData(X=refined_sparse_array_cpu.T) # .tocsr())

# Assign correct names
adata.obs_names = nuclei_names_list           # rows = nuclei
adata.var_names = variable_gene_names         # columns = HVGs




output_dir = condition_params["output_dir"][0]
out_path = os.path.join(output_dir, "sc_autism.h5ad")
adata.write(out_path) # This line needs to be commented out



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
"""
#---------------------------------------------------

# adata = AnnData(X=partial_sparse_array)
# rapids_singlecell.pp.pca
# rapids_singlecell.tl.louvain

"""

def preprocess_in_batches(input_file, min_genes_per_cell=200, max_genes_per_cell=6000, min_cells_per_gene=1,
                          target_sum=1e4, n_top_genes=4000):
    _data = '/X/data'
    _index = '/X/indices'  # columns in which data elements are found in
    _indptr = '/X/indptr'  # indices in data that represent where a new row starts (index 0 in indptor stored the position in data that corresonds to row 0)
    _nuclei = '/var/_index'  # column names
    _genes = '/obs/_index'  # row names
    gene_batch_size = 2000
    batches = []
    mean = []
    mean_sq = []

    input_file = "krasnow_transposed.h5ad"

    with h5py.File(input_file, 'r') as h5f:
        n_genes = len(h5f[_genes])
        n_nuclei = len(h5f[_nuclei])
        indptrs = h5f[_indptr][:]
        genes = h5f[_genes][:].astype(str)   

    print("n_genes: ", n_genes)
    print("n_nuclei: ", n_nuclei)

    print("Calculating highly variable genes.")
    for batch_start in range(0, n_genes,
                             gene_batch_size):  # LETS ASSUME THAT INCOMING DATA IN H5F has genes as rows and nuclei as columns
        # Get batch indices
        with h5py.File(input_file, 'r') as h5f:
            actual_batch_size = min(gene_batch_size, n_genes - batch_start)
            batch_end = batch_start + actual_batch_size
            start_ptr = indptrs[batch_start]
            end_ptr = indptrs[batch_end]
            # Read data and index of batch from hdf5
            sub_data = cp.array(h5f[_data][start_ptr:end_ptr])
            sub_indices = cp.array(h5f[_index][start_ptr:end_ptr])
            # recompute the row pointer for the partial dataset
            sub_indptrs = cp.array(indptrs[batch_start:(batch_end + 1)])
            sub_indptrs = sub_indptrs - sub_indptrs[0]
            # Reconstruct partial sparse array
            partial_sparse_array = cp.sparse.csr_matrix((sub_data, sub_indices, sub_indptrs),
                                                        shape=(batch_end - batch_start, n_nuclei))  #
            print(partial_sparse_array.shape)

        # rsc.pp.regress_out(adata, keys=["n_counts", "percent_MT"], inplace = True)

        partial_mean = partial_sparse_array.sum(axis=1) / partial_sparse_array.shape[1]  # Gives average expression of a gene per nuclei
        mean.append(partial_mean)

        # Calculate sq sum per gene - can batch across genes
        partial_sparse_array = partial_sparse_array.multiply(partial_sparse_array)
        partial_mean_sq = partial_sparse_array.sum(axis=1) / partial_sparse_array.shape[1]
        mean_sq.append(partial_mean_sq)

    mean_lens = [len(i) for i in mean]
    print(mean_lens)

    mean = cp.concatenate(mean).ravel()
    mean_sq = cp.concatenate(mean_sq).ravel()
    variable_genes = _cellranger_hvg(mean, mean_sq, n_genes, n_nuclei, n_top_genes)
    
    
    row_indices = np.where(variable_genes)[0]
    batches = []

    for row_index in row_indices:
        row_start = row_index
        row_end = row_index + 1
        start_ptr = indptrs[row_start]
        end_ptr = indptrs[row_end]  # We will be extracting one row at a time
        with h5py.File(input_file, 'r') as h5f:
            sub_data = cp.array(h5f[_data][start_ptr:end_ptr])
            sub_indices = cp.array(h5f[_index][start_ptr:end_ptr])
            sub_indptrs = cp.array(indptrs[row_start:(row_end + 1)])
            sub_indptrs = sub_indptrs - sub_indptrs[0]

            partial_sparse_array = cp.sparse.csr_matrix((sub_data, sub_indices, sub_indptrs),
                                                        shape=(row_end - row_start, n_nuclei))
            batches.append(partial_sparse_array)

    sparse_gpu_array = cp.sparse.vstack([partial_sparse_array for partial_sparse_array in batches])
    genes_filtered = [genes[row_index] for row_index in row_indices]  # Unsure if this line works
    genes_filtered = sorted(genes_filtered)
    #genes_filtered = [gene.decode('utf-8') for gene in genes_filtered]

    print(sparse_gpu_array.shape)
    print(cp.min(sparse_gpu_array.data))
    print(sparse_gpu_array.nnz)
    print(sparse_gpu_array.nnz)

    print(len(genes_filtered))
    print(genes_filtered[0:5])
    
    f = open("hvg_sr.txt", "w")
    f.write(','.join(genes_filtered)[:-1])
    f.close()

    return sparse_gpu_array, genes_filtered
"""