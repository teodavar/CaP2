#!/bin/bash

#SBATCH --job-name=Testing      ## Name of the job
#SBATCH --output=Testing.out    ## Output file
#SBATCH --error=Testing.err
#SBATCH --time=1:00:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=2      ## The number of threads the code will use

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200
#SBATCH --mem=4GB

echo "XXXXx"

eval "$(conda shell.bash hook)"

conda deactivate
conda activate newcap


./run_cifar10.sh

conda deactivate