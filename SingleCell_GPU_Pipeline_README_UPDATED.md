
# GPU-Accelerated Single-Cell Clustering Pipeline
## Memory-Efficient HVG Selection and Clustering for 1M+ Cells

---

## Overview

This pipeline performs **GPU-accelerated single-cell RNA-seq analysis** designed to scale efficiently to **>1 million nuclei** while minimizing:

- memory footprint  
- disk I/O  
- runtime  

The workflow clusters nuclei into **cell types** using **highly variable genes (HVGs)**.

Pipeline stages:

1. HVG identification (memory-efficient batching)  
2. Clustering + visualization (RAPIDS GPU accelerated)  

---

# Pipeline Flowchart

```
        ┌────────────────────────────┐
        │  Input HDF5 (CSR)          │
        │  genes × nuclei            │
        └──────────────┬─────────────┘
                       │
                       ▼
        ┌────────────────────────────┐
        │  Shared Gene Intersection  │
        └──────────────┬─────────────┘
                       │
                       ▼
        ┌────────────────────────────┐
        │  Batch HVG Selection       │
        │  - remove low exp genes    │
        │  - normalize + log1p       │
        │  - residualize covariates  │
        │  - compute dispersion      │
        └──────────────┬─────────────┘
                       │
                       ▼
        ┌────────────────────────────┐
        │  Select HVGs               │
        └──────────────┬─────────────┘
                       │
                       ▼
        ┌────────────────────────────┐
        │  Build HVG Matrix (CSR)    │
        │  genes × nuclei            │
        └──────────────┬─────────────┘
                       │
                       ▼
        ┌────────────────────────────┐
        │  Transpose                 │
        │  nuclei × genes            │
        └──────────────┬─────────────┘
                       │
                       ▼
        ┌────────────────────────────┐
        │  AnnData                   │
        └──────────────┬─────────────┘
                       │
                       ▼
        ┌────────────────────────────┐
        │  PCA (GPU)                 │
        └──────────────┬─────────────┘
                       │
                       ▼
        ┌────────────────────────────┐
        │  Neighbor Graph            │
        └──────────────┬─────────────┘
                       │
                       ▼
        ┌──────────────┴─────────────┐
        │ Leiden        Louvain      │
        └──────────────┬─────────────┘
                       │
                       ▼
        ┌────────────────────────────┐
        │  UMAP / tSNE               │
        └──────────────┬─────────────┘
                       │
                       ▼
        ┌────────────────────────────┐
        │  MAD Outlier Removal       │
        └──────────────┬─────────────┘
                       │
                       ▼
        ┌────────────────────────────┐
        │  Recluster                 │
        └──────────────┬─────────────┘
                       │
                       ▼
        ┌────────────────────────────┐
        │  Final Outputs             │
        └────────────────────────────┘
```

---

# Architecture

### Driver Script

```
test.py
```

Responsibilities:

- read conditions.csv  
- select run parameters  
- HVG selection  
- clustering  
- save outputs  

### Core Functions

```
test_functions.py
```

Contains:

- HVG batching logic  
- sparse matrix construction  
- clustering  
- plotting  
- QC metrics  
- outlier detection  

---

# Data Flow Through Pipeline

| Stage | Format | Shape | Device |
|------|-------|------|------|
Raw HDF5 | CSR | genes × nuclei | disk |
Gene batch | CSR | batch × nuclei | GPU |
Residualized batch | dense | batch × nuclei | GPU |
HVG matrix | CSR | genes × nuclei | GPU |
Transposed | CSR | nuclei × genes | GPU |
AnnData | CSR | nuclei × genes | CPU |
PCA | dense | nuclei × PCs | GPU |
Neighbors | graph | nuclei × k | GPU |

---

# Step 1 — HVG Selection

Genes are processed **in batches** to avoid loading full matrices into memory.

Each batch:

1. Load genes from HDF5  
2. Remove low expression genes  
3. Normalize counts  
4. Log1p transform  
5. Residualize covariates  
6. Compute dispersion  
7. Select HVGs  

---

# HVG Algorithm (Technical)

For each gene:

Mean:

μ_g = mean(expression)

Variance:

σ²_g = var(expression)

Dispersion:

d_g = σ²_g / μ_g

Genes are binned by mean expression.

Within each bin:

normalized dispersion:

d_norm = (d_g − median_bin) / MAD_bin

Top N genes selected.

Optional gene filtering:

- mitochondrial genes  
- sex chromosomes  
- user-defined exclusions  

---

# Step 2 — Clustering

Performed using RAPIDS Single Cell.

Pipeline:

1. PCA (GPU)
2. Neighbor graph
3. Leiden clustering
4. Louvain clustering
5. UMAP
6. tSNE

---

# Sparse Matrix Strategy

Input format:

CSR  
genes × nuclei  

Why:

fast gene batching

After HVG selection:

CSR  
genes × nuclei  

Matrix is then transposed:

CSR  
nuclei × genes  

Reason:

AnnData expects cells as rows.

RAPIDS also prefers CSR format.

---

# Memory Efficiency Design

Without batching:

1M cells × 30k genes → too large for memory

With batching:

process only subset of genes at a time

Benefits:

- low RAM usage
- low GPU memory
- scalable to millions of cells
- streaming computation

---

# conditions.csv Reference

Each row defines one run.

| Parameter | Description |
|----------|-------------|
data_file | input HDF5 files |
covariates_file | covariate matrix |
n_variableGenes | number of HVGs |
exp_thresh | low expression threshold |
batch_size | genes per batch |
n_neighbors | neighbor graph size |
resolution | clustering resolution |
mad_thres | outlier threshold |
random_state | reproducibility seed |

---

# Outputs

Plots:

- UMAP Leiden  
- UMAP Louvain  
- tSNE Leiden  
- tSNE Louvain  

Before and after outlier removal.

QC Files:

- cluster_qc_post_outlier_removal.csv  
- result_1.csv  
- result_2.csv  
- result_3.csv  

---

# Developer Notes

### CSR vs CSC

CSR used for:

- fast gene slicing
- RAPIDS compatibility

CSC not used due to slow row access.

### GPU Sparse Matrices

Pipeline uses:

- CuPy CSR
- GPU residualization
- GPU PCA

### AnnData Conversion

Matrix transposed to:

cells × genes

before creating AnnData.

### RAPIDS Requirements

Preferred:

CSR sparse matrix  
float32 precision  

### Covariate Residualization

Performed using QR decomposition on GPU.

Removes:

- batch effects
- covariate bias
- sample effects

---

# Summary

This pipeline enables:

- scalable single-cell clustering
- GPU acceleration
- memory efficient HVG selection
- multi dataset integration
- robust outlier removal
