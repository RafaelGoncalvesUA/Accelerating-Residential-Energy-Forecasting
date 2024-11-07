#!/bin/bash

#SBATCH --job-name=energy_locality_vs_globality
#SBATCH --output=benchmark.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # 1 process per node
#SBATCH --cpus-per-task=8           # 8 CPUs per task
#SBATCH --gres=gpu:1                # 1 GPU per node
#SBATCH --time=7-00:00:00           # Maximum execution time
#SBATCH --partition=gpuPartition    # Default partition

SCRIPT_NAME="benchmark.py"

source ./venv/bin/activate
srun python ${SCRIPT_NAME}
