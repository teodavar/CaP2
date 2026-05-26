#!/bin/bash
#SBATCH --job-name=sensitivity_analysis
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sensitivity_%j_%x.out
#SBATCH --error=logs/sensitivity_%j_%x.err

#
# CPU-only sensitivity analysis — no GPU requested.
#
# Usage:
#   sbatch submit_sensitivity.sh [TOPOLOGY]
#
# TOPOLOGY choices: Dtelekom_cost (default), Abilene_cost,
#                   BarabasiAlbert_cost, WattsStrogatz_cost
#
#
# Examples:
#   sbatch submit_sensitivity.sh
#   sbatch --export=ALL,TOPOLOGY=Abilene_cost submit_sensitivity.sh

# ── topology (from CLI arg or $TOPOLOGY env var, default Dtelekom_cost) ───────

TOPOLOGY="${1:-${TOPOLOGY:-Dtelekom_cost}}"

case "$TOPOLOGY" in
    Dtelekom_cost)
        BASE_DIR=experiment_logs_dtelecom_cost
        NP=68
        ;;
    Abilene_cost)
        BASE_DIR=experiment_logs_abilene_cost
        NP=11
        ;;
    BarabasiAlbert_cost)
        BASE_DIR=experiment_logs_barabasi_albert_cost
        NP=10
        ;;
    WattsStrogatz_cost)
        BASE_DIR=experiment_logs_watts_strogatz_costs
        NP=10
        ;;
    *)
        echo "ERROR: Unknown topology '$TOPOLOGY'."
        echo "Valid choices: Dtelekom_cost, Abilene_cost, BarabasiAlbert_cost, WattsStrogatz_cost"
        exit 1
        ;;
esac

CAP_FOLDER="${BASE_DIR}/cifar100_resnet101_pr0.75_np${NP}_kernel_${TOPOLOGY}_fixed_original_full_60"
P_TCC_FOLDER="${BASE_DIR}/cifar100_resnet101_pr0.75_np${NP}_partition_row_${TOPOLOGY}_rsgn5_original_full_100"
P_AOP_FOLDER="${BASE_DIR}/cifar100_resnet101_pr0.75_np${NP}_partition_row_${TOPOLOGY}_rsgn5_original_aggregate_partition_rows_100"
DENSE_FOLDER="${BASE_DIR}/cifar100_resnet101_pr1.0_np${NP}_kernel_${TOPOLOGY}_fixed_original_full_60"

echo "================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Topology : $TOPOLOGY"
echo "CAP      : $CAP_FOLDER"
echo "TCC      : $P_TCC_FOLDER"
echo "AOP      : $P_AOP_FOLDER"
echo "Dense    : $DENSE_FOLDER"
echo "================================================"
echo

# ── conda (sbatch does not source ~/.bashrc, so initialize explicitly) ────────

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || { echo "ERROR: conda init script not found."; exit 1; }

conda activate newcap

# ── create log dir if needed ──────────────────────────────────────────────────

mkdir -p logs sensitivity_out delays_out

# ── run ───────────────────────────────────────────────────────────────────────

echo ">>> Starting eval ..."
python -u sensitivity_analysis.py \
    --mode eval                    \
    --camp_tcc_dir "$P_TCC_FOLDER" \
    --camp_aop_dir "$P_AOP_FOLDER" \
    --cap_dir      "$CAP_FOLDER"   \
    --dense_dir    "$DENSE_FOLDER" \
    --topology     "$TOPOLOGY"     \
    --pr 0.75 --n_draws 10         \
    --device cpu                   \
    --out_dir sensitivity_out

echo ">>> Starting delays ..."
python -u sensitivity_analysis.py \
    --mode delays                  \
    --camp_tcc_dir "$P_TCC_FOLDER" \
    --camp_aop_dir "$P_AOP_FOLDER" \
    --cap_dir      "$CAP_FOLDER"   \
    --dense_dir    "$DENSE_FOLDER" \
    --topology     "$TOPOLOGY"     \
    --pr 0.75 --n_draws 10         \
    --device cpu                   \
    --out_dir delays_out

echo ">>> Done."

conda deactivate
