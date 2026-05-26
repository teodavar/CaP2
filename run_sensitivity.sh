#!/bin/bash
#
# Sensitivity analysis — CPU only (no GPU needed for --mode eval / delays).
#
# Usage:
#   bash run_sensitivity.sh [TOPOLOGY]
#
# TOPOLOGY choices: Dtelekom_cost (default), Abilene_cost,
#                   BarabasiAlbert_cost, WattsStrogatz_cost
#
# Examples:
#   bash run_sensitivity.sh
#   bash run_sensitivity.sh Abilene_cost

conda deactivate
conda activate newcap

# ── topology selection ────────────────────────────────────────────────────────

TOPOLOGY="${1:-Dtelekom_cost}"

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

echo "Topology : $TOPOLOGY"
echo "CAP      : $CAP_FOLDER"
echo "TCC      : $P_TCC_FOLDER"
echo "AOP      : $P_AOP_FOLDER"
echo "Dense    : $DENSE_FOLDER"
echo

# ── functions ─────────────────────────────────────────────────────────────────

run_analysis() {
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
}

run_analysis_delays() {
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
}

# ── main ──────────────────────────────────────────────────────────────────────

run_analysis
run_analysis_delays

conda deactivate

# To run it:
# bash run_sensitivity.sh
# bash run_sensitivity.sh Abilene_cost
