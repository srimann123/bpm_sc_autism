
# GPU-Accelerated Single Cell Analysis Pipeline
## Memory-Efficient HVG Selection and Clustering for 1M+ Cells

---

# Overview

This pipeline performs **GPU-accelerated single-cell analysis** designed to scale efficiently to **1M+ nuclei** while minimizing:

- memory usage
- execution time
- intermediate storage

The goal of the pipeline is to **cluster nuclei by cell type** using **highly variable genes (HVGs)**.

The workflow consists of two major stages:

### Stage 1 — HVG Identification
Identify highly variable genes using **memory-efficient batched processing**.

### Stage 2 — Clustering
Cluster nuclei using only the HVG matrix.

The output of Stage 1 is an **AnnData object containing nuclei × HVGs**, which is used as the input for Stage 2.

---

# Pipeline Flowchart

```
Input HDF5 (CSR genes × nuclei)
            │
            ▼
Shared Gene Intersection
            │
            ▼
Batch HVG Selection
 ├── Remove low expression genes
 ├── Normalize + log1p
 ├── Residualize covariates
 └── Compute dispersion
            │
            ▼
Select HVGs
            │
            ▼
Build HVG Matrix (HDF5 splicing)
            │
            ▼
Transpose → nuclei × genes
            │
            ▼
AnnData Object
            │
            ▼
PCA (GPU)
            │
            ▼
Neighbor Graph
            │
            ▼
Leiden / Louvain
            │
            ▼
UMAP / tSNE
            │
            ▼
MAD Outlier Detection
            │
            ▼
Remove Outliers
            │
            ▼
Re-run Clustering
            │
            ▼
Final Outputs
```

---

# Architecture

The pipeline consists of **two main Python files**:

### Driver Script
Responsible for:

- reading conditions.csv
- selecting run parameters
- orchestrating HVG selection
- running clustering
- saving outputs

### Function Library

Contains:

- HVG selection functions
- batching logic
- sparse matrix construction
- clustering utilities
- plotting
- QC metrics
- outlier detection

---

# Input Requirements

Input HDF5 matrices must be stored as:

CSR sparse format  
genes × nuclei  

This format is required because:

- CSR enables fast row access
- genes are processed in batches
- improves HDF5 slicing performance

---

# Multi-Dataset Integration

Multiple single-cell datasets can be integrated in a single run.

The pipeline:

1. Finds shared genes across datasets
2. Aligns gene indices
3. Streams batches from each dataset
4. Concatenates nuclei
5. Computes HVGs jointly

This enables:

- multi-sample clustering
- dataset integration
- batch-aware analysis

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

# Stage 1 — HVG Selection

HVG selection is performed **in gene batches** using HDF5 splicing.

This ensures only a subset of genes is in memory at any time.

For each batch:

1. Load gene batch from HDF5
2. Remove low expression genes
3. Normalize expression
4. Log1p transform
5. Residualize covariates
6. Compute mean and variance
7. Compute dispersion
8. Select HVGs

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

Normalized dispersion:

d_norm = (d_g − median_bin) / MAD_bin

Top N genes are selected.

Optional filtering:

- mitochondrial genes
- sex chromosome genes
- user-defined exclusions

---

# HDF5 Splicing Design

The HVG matrix is constructed using **HDF5 splicing**.

This means:

- no intermediate matrix stored
- genes streamed directly from disk
- no memory traces from prior steps
- scalable to millions of cells

This design is critical for:

- memory efficiency
- GPU compatibility
- large dataset support

---

# Sparse Matrix Strategy

Input:

CSR  
genes × nuclei

During HVG construction:

CSR  
genes × nuclei

After construction:

Transpose → CSR nuclei × genes

This is required because:

AnnData expects:

cells × genes

CSR format is retained to support RAPIDS GPU operations.

---

# Stage 2 — Clustering

Stage 2 operates on the HVG-only AnnData matrix.

Pipeline:

1. PCA (GPU)
2. Compute neighbor graph
3. Leiden clustering
4. Louvain clustering
5. UMAP embedding
6. tSNE embedding

Leiden is recommended for single-cell analyses.

---

# Outlier Detection

Outliers are detected using **MAD-based deviation** within clusters.

Procedure:

1. Compute cluster center
2. Compute cell distance
3. Compute MAD
4. Flag outliers
5. Remove outliers

After removal, clustering is re-run.

Two complete runs are produced:

Run 1 — Original clustering  
Run 2 — Outliers removed  

---

# conditions.csv Interface

Each row represents a standalone run.

Parameters include:

| Parameter | Description |
|----------|-------------|
data_file | input HDF5 datasets |
covariates_file | covariate matrix |
n_variableGenes | number of HVGs |
exp_thresh | low expression threshold |
batch_size | genes per batch |
n_neighbors | neighbor graph size |
resolution | clustering resolution |
mad_thres | outlier threshold |
random_state | reproducibility seed |

---

# Parallel Execution

The pipeline supports SLURM array jobs.

Execution flow:

conditions.csv  
→ submit_array.sh  
→ run_array.sh  
→ main python script  

Each row in conditions.csv produces:

- one independent run
- one output directory

A unique hash is generated for each run.

---

# Optional: Skip HVG Selection

Users may provide a precomputed HDF5 / h5ad file.

This allows:

- testing clustering parameters
- skipping HVG recomputation
- faster iteration

---

# Outputs

For each run:

UMAP plots:

- Leiden
- Louvain

tSNE plots:

- Leiden
- Louvain

Both before and after outlier removal.

Additional files:

- QC metrics
- cluster assignments
- outlier reports
- run hash metadata

---

# Memory Efficiency Design

Without batching:

1M cells × 30k genes → extremely large

With batching:

only subset of genes loaded

Benefits:

- low RAM usage
- low GPU memory
- scalable to large datasets
- no intermediate matrices

---

# Developer Notes

### CSR vs CSC

CSR used for fast gene slicing.

CSC not used due to slow row access.

### GPU Sparse Matrices

Pipeline uses:

- CuPy CSR
- GPU PCA
- GPU clustering

### AnnData Conversion

Matrix transposed to:

cells × genes

before creating AnnData.

### Covariate Residualization

Performed using QR decomposition on GPU.

Removes:

- batch effects
- covariate bias
- sample effects

### Implementation Inspiration

Stage 1 HVG batching functions are inspired by NVIDIA GPU
single-cell analysis workflows but adapted for:

- HDF5 splicing
- multi-dataset integration
- CSR sparse operations

---

# Summary

This pipeline enables:

- GPU accelerated single-cell clustering
- memory efficient HVG selection
- 1M+ cell scalability
- multi dataset integration
- robust outlier removal
