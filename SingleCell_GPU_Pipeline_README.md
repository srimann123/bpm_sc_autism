
# GPU-Accelerated Single-Cell Clustering Pipeline
## Memory‑Efficient HVG Selection and Clustering for 1M+ Cells

---

## Overview

This pipeline performs **GPU-accelerated single-cell RNA-seq analysis** designed to scale to **>1 million nuclei** while minimizing:

- Memory usage  
- Execution time  
- Disk I/O overhead  

The workflow clusters nuclei into **cell types** using **highly variable genes (HVGs)** computed across one or more datasets.

### Pipeline Stages

1. **HVG Identification (memory-efficient, batched)**
2. **Clustering + Visualization (GPU accelerated)**

Outputs include:

- Leiden clusters  
- Louvain clusters  
- UMAP plots  
- tSNE plots  
- QC metrics  
- Outlier detection summaries  

---

# High-Level Workflow

```
Raw HDF5 matrices (genes × nuclei, CSR)
            │
            ▼
Step 1: HVG Selection (batched)
    remove low-expression genes
    normalize + log1p
    residualize covariates
    compute normalized dispersion
    select HVGs
            │
            ▼
Build HVG matrix (nuclei × HVGs)
            │
            ▼
Step 2: Clustering
    PCA
    neighbors graph
    Leiden / Louvain
    UMAP / tSNE
            │
            ▼
MAD Outlier Detection
            │
            ▼
Re-cluster
            │
            ▼
Final plots + QC
```

---

# Architecture

## Main Driver

```
test.py
```

Responsible for:

- reading conditions.csv
- loading datasets
- running HVG selection
- running clustering
- saving outputs

## Core Functions

```
test_functions.py
```

Contains:

- HVG selection logic  
- batch processing  
- sparse matrix construction  
- clustering wrappers  
- plotting utilities  
- outlier detection  
- QC metrics  

---

# Input Data Requirements

Input HDF5 files must be:

### Sparse Format
CSR (compressed sparse row)

### Matrix Orientation

```
genes × nuclei
```

This enables **fast gene batching**.

---

# Step 1 — HVG Identification

Genes are processed in **batches** to minimize memory usage.

Each batch:

1. Load gene batch from HDF5
2. Remove low expression genes
3. Normalize + log1p
4. Residualize covariates
5. Compute dispersion
6. Select HVGs

---

# CellRanger-style HVG Selection

The `_cellranger_hvg()` function:

1. bins genes by mean expression
2. computes median dispersion
3. computes MAD per bin
4. normalizes dispersion
5. filters genes
6. selects top N genes

Optional gene filtering:

- mitochondrial genes  
- sex chromosome genes  
- custom gene sets  

---

# HVG Matrix Construction

Function:

```
build_hvg_matrix()
```

Steps:

1. slice HVGs from HDF5
2. normalize
3. residualize
4. optional scaling
5. build sparse matrix

Matrix:

```
genes × nuclei  (CSR)
```

Transposed to:

```
nuclei × genes  (CSR)
```

Converted to AnnData.

---

# Step 2 — Clustering

Performed using **RAPIDS Single Cell**

## Pipeline

### PCA (GPU)

```
rsc.pp.pca()
```

### Neighbor Graph

```
rsc.pp.neighbors()
```

### Clustering

- Leiden  
- Louvain  

### Embeddings

- UMAP  
- tSNE  

---

# Outlier Detection

MAD-based detection within clusters:

1. compute cluster center
2. compute cell deviation
3. compute MAD
4. remove outliers

Pipeline reruns after removal.

---

# Outputs

Plots (before & after):

- UMAP Leiden  
- UMAP Louvain  
- tSNE Leiden  
- tSNE Louvain  

QC Files:

- cluster_qc_post_outlier_removal.csv  
- result_1.csv  
- result_2.csv  
- result_3.csv  

---

# conditions.csv Interface

Each row defines one run.

Example parameters:

```
data_file
covariates_file
n_variableGenes
n_pca_components
n_neighbors
resolution
exp_thresh
batch_size
random_state
```

---

# Parallel Execution

```
submit_array.sh
    ↓
run_array.sh
    ↓
test.py
```

Each SLURM job runs one row.

---

# Sparse Format Notes

Required:

```
CSR
genes × nuclei
```

After HVG build:

```
CSR
nuclei × genes
```

---

# Multi‑Dataset Integration

Pipeline:

1. find shared genes
2. align datasets
3. concatenate nuclei
4. cluster jointly

---

# Optional: Skip HVG Selection

Use precomputed matrix:

```
h5ad_file parameter
```

---

# Performance Features

- GPU accelerated PCA
- GPU clustering
- HDF5 slicing
- gene batching
- sparse matrices
- >1M cell scalability

---

# Summary

This pipeline enables:

- scalable single-cell clustering
- GPU acceleration
- low memory usage
- flexible parameter sweeps
- multi dataset integration
- robust outlier removal

