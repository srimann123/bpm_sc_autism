
# GPU-Accelerated Single Cell Analysis Pipeline
## Memory‑Efficient HVG Selection and Clustering for 1M+ Cells

---

# Overview

This pipeline performs **GPU‑accelerated single‑cell analysis** designed to scale to **>1 million nuclei** while minimizing memory usage and runtime.  
The goal is to cluster nuclei by cell type using **highly variable genes (HVGs)**.

The workflow consists of two stages:

**Stage 1 — HVG Identification**  
Memory‑efficient batched computation of highly variable genes.

**Stage 2 — Clustering**  
PCA → neighbor graph → Leiden/Louvain clustering → UMAP/tSNE visualization.

The output of Stage 1 is an **AnnData object (nuclei × HVGs)** which is used for clustering.

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
        │
        ▼
Build HVG Matrix (HDF5 splicing)
        │
        ▼
Transpose → nuclei × genes
        │
        ▼
AnnData
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
MAD Outlier Removal
        │
        ▼
Re‑cluster
        │
        ▼
Outputs
```

---

# Running the Pipeline

## Step 1 — Create `conditions.csv`

Each row represents one run.

Example:

```csv
data_file,covariates_file,n_variableGenes,n_pca_components,resolution,n_neighbors,batch_size
dataset.h5,covariates.csv,3000,50,1.0,30,1000
```

---

## Step 2 — Submit Array Job

```bash
sbatch submit_array.sh conditions.csv
```

Each row in `conditions.csv` launches one independent run.

---

# Job Execution Workflow

```
conditions.csv
     │
     ▼
submit_array.sh
     │
     ▼
SLURM array jobs
     │
     ▼
run_array.sh
     │
     ▼
python test.py conditions.csv $SLURM_ARRAY_TASK_ID
     │
     ▼
select row
     │
     ▼
run pipeline
     │
     ▼
output_dir/run_hash/
```

### Row Mapping

SLURM uses 1‑based indexing:

```
job 1 → row 1
job 2 → row 2
job 3 → row 3
```

---

# conditions.csv Parameters

## Core Analysis Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
n_variableGenes | number of HVGs | 2000–5000 |
n_pca_components | PCA dimensions | 30–100 |
n_neighbors | kNN graph size | 15–50 |
resolution | Leiden resolution | 0.4–1.5 |
batch_size | gene batch size | 500–5000 |
exp_thresh | low expression cutoff | 0.001–0.01 |
mad_thres | outlier MAD threshold | 3–5 |

## UMAP / tSNE Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
perplex | tSNE perplexity | 30–200 |
learning_rate | tSNE learning rate | 200–2000 |
early_exagg | tSNE exaggeration | 12–24 |
min_dist | UMAP min distance | 0.1–0.5 |
spread | UMAP spread | 1–2 |

---

# Data Flow Through Pipeline

| Stage | Format | Shape | Device |
|------|-------|------|------|
Input | CSR | genes × nuclei | disk |
Batch | CSR | batch × nuclei | GPU |
HVG matrix | CSR | genes × nuclei | GPU |
Transposed | CSR | nuclei × genes | GPU |
AnnData | CSR | nuclei × genes | CPU |
PCA | dense | nuclei × PCs | GPU |
Neighbors | graph | nuclei × k | GPU |

---

# Stage 1 — HVG Selection

Genes are processed in batches using HDF5 splicing:

1. load gene batch
2. remove low expression genes
3. normalize + log1p
4. residualize covariates
5. compute variance/dispersion
6. select HVGs

This avoids loading the full matrix into memory.

---

# HVG Algorithm

For each gene:

Mean:

μ = mean(expression)

Variance:

σ² = var(expression)

Dispersion:

σ² / μ

Genes are binned by mean expression and normalized using MAD within bins.  
Top N genes are selected.

Optional filters:

- mitochondrial genes
- sex chromosome genes
- user exclusions

---

# Sparse Matrix Strategy

Input:

CSR (genes × nuclei)

After HVG construction:

CSR (genes × nuclei)

Transpose:

CSR (nuclei × genes)

This format is required for:

- AnnData
- RAPIDS GPU clustering

---

# Stage 2 — Clustering

Performed using RAPIDS:

1. PCA (GPU)
2. neighbor graph
3. Leiden clustering
4. Louvain clustering
5. UMAP
6. tSNE

Leiden is recommended.

---

# Outlier Removal

MAD‑based outlier detection is performed within clusters.

Two runs are produced:

1. original clustering
2. outliers removed clustering

---

# Multi‑Dataset Integration

Multiple HDF5 datasets may be provided.

Pipeline:

1. find shared genes
2. align datasets
3. stream batches
4. concatenate nuclei
5. cluster jointly

---

# Memory Efficiency Design

Batch processing ensures:

- low RAM usage
- low GPU memory
- no intermediate matrices
- scalability to millions of cells

---

# Outputs

Each run produces:

- UMAP plots (Leiden + Louvain)
- tSNE plots (Leiden + Louvain)
- cluster assignments
- QC metrics
- outlier reports

---

# Developer Notes

### Sparse format

CSR used for fast gene batching and RAPIDS compatibility.

### Covariate residualization

Performed using GPU QR decomposition.

### Implementation

HVG batching inspired by NVIDIA GPU single‑cell workflows with
custom HDF5 splicing and multi‑dataset integration.

