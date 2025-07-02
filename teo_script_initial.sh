#!/bin/bash

#SBATCH --job-name=Relaxed_Full      ## Name of the job
#SBATCH --output=Relaxed_Full.out    ## Output file
#SBATCH --error=Relaxed_Full.err
#SBATCH --time=8:00:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=2      ## The number of threads the code will use

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200
#SBATCH --mem=8GB

echo "run_cifar10_relaxed_full"

eval "$(conda shell.bash hook)"

conda deactivate
conda activate newcap


./run_cifar10_relaxed_full.sh

conda deactivate