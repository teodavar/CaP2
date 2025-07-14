#!/bin/bash

#SBATCH --time=4:00:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=2      ## The number of threads the code will use

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200
#SBATCH --mem=8GB

eval "$(conda shell.bash hook)"

conda deactivate
conda activate newcap

echo 'Runing for: '
echo $1

echo 'Configuration file: '
cfg_file="config/"$1".yaml"
echo $cfg_file

echo 'Log file: '
log_file="experiment_logs/"$1"_log.out"
echo $log_file


prune_finetune() {
    python -m source.core.run_partition \
           -cfg $cfg_file 
}

#prune_finetune() {
#    python -m source.core.run_partition \
#           -cfg $cfg_file \
#           >experiment_logs/${log_file}.out
#}

prune_finetune

conda deactivate

# run sbatch as following:

# cifar10 Uniform

#sbatch --job-name cifar10_original_aggregate_partition_uniform teo_script.sh cifar10_original_aggregate_partition
#sbatch --job-name cifar10_original_full_partition_uniform teo_script.sh cifar10_original_full_partition 
#sbatch --job-name cifar10_original_full_kernel_uniform teo_script.sh cifar10_original_full_kernel 

# cifar100 Uniform

#sbatch --job-name cifar100_original_aggregate_partition_uniform teo_script.sh cifar100_original_aggregate_partition_uniform 
#sbatch --job-name cifar100_original_full_partition_uniform teo_script.sh cifar100_original_full_partition_uniform 
#sbatch --job-name cifar100_original_full_kernel_uniform teo_script.sh cifar100_original_full_kernel_uniform 

# cifar100 watts

#sbatch --job-name cifar100_original_aggregate_partition_watts teo_script.sh cifar100_original_aggregate_partition_watts 
#sbatch --job-name cifar100_original_full_partition_watts teo_script.sh cifar100_original_full_partition_watts 
#sbatch --job-name cifar100_original_full_kernel_watts teo_script.sh cifar100_original_full_kernel_watts 

# cifar100 barabasi

#sbatch --job-name cifar100_original_aggregate_partition_barabasi teo_script.sh cifar100_original_aggregate_partition_barabasi 
#sbatch --job-name cifar100_original_full_partition_barabasi teo_script.sh cifar100_original_full_partition_barabasi
#sbatch --job-name cifar100_original_full_kernel_barabasi teo_script.sh cifar100_original_full_kernel_barabasi 

# cifar100 Abilene

#sbatch --job-name cifar100_original_aggregate_partition_abilene teo_script.sh cifar100_original_aggregate_partition_abilene 
#sbatch --job-name cifar100_original_full_partition_abilene teo_script.sh cifar100_original_full_partition_abilene
#sbatch --job-name cifar100_original_full_kernel_abilene teo_script.sh cifar100_original_full_kernel_abilene 

# cifar100 DTelecom

#sbatch --job-name cifar100_original_aggregate_partition_dtelecom teo_script.sh cifar100_original_aggregate_partition_dtelecom 
#sbatch --job-name cifar100_original_full_partition_dtelecom teo_script.sh cifar100_original_full_partition_dtelecom
#sbatch --job-name cifar100_original_full_kernel_dtelecom teo_script.sh cifar100_original_full_kernel_dtelecom 


# srun --partition=gpu --nodes=1 --gres=gpu:h200:1 --cpus-per-task=2 --mem=4GB --time=03:00:00 --pty /bin/bash

#sbatch --job-name cifar10_original_full_partition_uniform --output=experiment_logs/cifar10_original_full_partition.out --error=experiment_logs/cifar10_original_full_partition.err teo_script.sh cifar10_original_aggregate_partition 
#sbatch --job-name cifar10 teo_script.sh cifar10 

# cifar100 general
#sbatch --job-name cifar100_original_aggregate_partition_dteleom_SES teo_script.sh cifar100_original_aggregate_partition 
#sbatch --job-name cifar100_original_full_partition_dteleom_SES teo_script.sh cifar100_original_full_partition 
#sbatch --job-name cifar100_original_full_kernel teo_script.sh cifar100_original_full_kernel 



