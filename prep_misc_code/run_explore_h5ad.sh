#!/usr/bin/env bash
#SBATCH --job-name=run_explore_h5ad
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=ramachandruss@vcu.edu   # (or whatever your HPC email is)
# Ensure logs directory exists
mkdir -p logs
# Load R if needed for rpy2


# Activate your environment
#source /lustre/home/ramachandruss/python_single_cell/rsc_upgrade_test/bin/activate
# Run the Python script
#pip install h5py numpy pandas scipy statsmodels matplotlib anndata scanpy
#conda create -n /lustre/home/ramachandruss/python_single_cell_updated python=3.12 -y
module load miniforge3/23.3.1
#conda create -p /lustre/home/ramachandruss/python_single_cell_updated python=3.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate /lustre/home/ramachandruss/python_single_cell_updated
#python --version

#conda install -c conda-forge numpy pandas scipy scanpy anndata matplotlib scikit-learn h5py statsmodels -y
#conda install -c rapidsai -c conda-forge -c nvidia cudf cuml cugraph cuda-version=12 -y

#pip install rapids-singlecell-cu12
#conda install -c conda-forge numpy pandas scipy scanpy anndata matplotlib scikit-learn h5py statsmodels -y
#conda install -c rapidsai -c conda-forge -c nvidia cudf cuml cugraph cuda-version=12 -y
#pip install --pre 'rapids-singlecell-cu12[rapids]' --extra-index-url=https://pypi.nvidia.com

python convert_sparse_matrix_format.py 
#python data_explore.py
#python check_rapids_version.py
#python explore_h5ad.py /lustre/home/BPM/autism_brain/cluster/krasnow_transposed_hvg.h5ad
#python merge_cluster_runs.py \
#  --path-list /lustre/home/ramachandruss/python_single_cell/test_interface/nuclei_cluster_paths.txt \
#  --reference-index 0 \
#  --output /lustre/home/ramachandruss/python_single_cell/test_interface/merged_clusters.csv \
#  --output-nonmajority /lustre/home/ramachandruss/python_single_cell/test_interface/nonmajority_stats.csv
