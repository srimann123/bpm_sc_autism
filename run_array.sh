#!/bin/bash
/bin/hostname

python_script=$1
array_file=$2
work_dir=$3

#module purge
#module load miniconda3/py39_23.9.0
#eval "$(conda shell.bash hook)"
#conda activate $HOME/conda-envs/rapids-24.08
module load miniforge3/23.3.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/home/ramachandruss/python_single_cell_updated

#source /lustre/home/ramachandruss/python_single_cell_updated/bin/activate
cd "$work_dir"

# Optional: read the current line (only if your python script needs it)
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$array_file")

# EITHER of these works:
# Plain python (works fine)
python "$python_script" "$array_file" "$SLURM_ARRAY_TASK_ID" "$work_dir"

# Or prefer srun (best practice under SLURM)
# srun python "$python_script" "$array_file" "$SLURM_ARRAY_TASK_ID"
