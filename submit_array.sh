#!/bin/bash
#SBATCH --output=main-%j-%a.out
#SBATCH --error=main-%j-%a.err

work_dir=$(pwd)
log_dir="$work_dir/logs"
mkdir -p "$log_dir"


array_file="$1"
run_array="/lustre/home/ramachandruss/python_single_cell/test_interface/run_array.sh"
python_script="/lustre/home/ramachandruss/python_single_cell/test_interface/test.py" # driver.py
name="array_job"

array_jobid=$(sbatch --job-name="$name" \
               --array=1-$(wc -l < "$array_file") \
               --partition=gpu \
               --gres=gpu:1 \
               --mem=100G \
               --time=04:00:00 \
               --output="$log_dir/${name}_%A_%a.out" \
               --error="$log_dir/${name}_%A_%a.err" \
               "$run_array" "$python_script" "$array_file" "$work_dir" \
               | awk '{print $NF}')

echo "Submitted array job with ID: $array_jobid"
