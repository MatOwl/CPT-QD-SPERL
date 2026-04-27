#!/bin/bash
# Paper Tables 1/2 SPERL sweep: 10 cells (2 CPT regimes x 5 p_win) x 10 seeds.
# Each cell uses the per-cell Pareto-optimal filter from paper §C.4 Tables 3/4 *-mark
# AND the paper §C.2.5 treshRatio (0.5 if filter<1, 0 if filter=1).
# Other hyperparams match runs/results_p066_10s_both_acceptInf/config.json + paper §C.2.5.
#
# v2 (2026-04-28): added per-cell treshRatio per paper §C.2.5 (was inf in v1).
#
# Usage:
#   bash scripts/sweep_paper_tables_1_2.sh
#
# Outputs: runs/results_paper_tables_1_2/barberis_sperl_p<p>_cpt_a<a>_r<r>_l<l>/

set -e

PYTHON="/c/Users/Jingxiang Tang/FNN/Scripts/python.exe"
RESULTS_DIR="./runs/results_paper_tables_1_2_v2"

# Paper §C.2.5 hyperparams
SEEDS=10
TRAIN_EPS=15000        # M = 15000
BATCH=5
SUPPORT=50             # K = 50
CRITIC_LR=0.04         # ϱ = .04
EXPLORE_EPS=0.3        # ξ = .3
HORIZON=5              # T̄ = T+1 = 6, so T = 5
SPE_ROLLOUTS=300
EVAL_PER_STATE=100

run_cell() {
    local regime=$1  # CPT88 or CPT95
    local p=$2
    local filter=$3

    # Paper §C.2.5: treshRatio = 0.5 if filter < 1, else 0 (== "no filter")
    local accept_ratio
    if [ "$filter" = "1.00" ]; then
        accept_ratio=0
    else
        accept_ratio=0.5
    fi

    local alpha rho1 lmbd
    if [ "$regime" = "CPT88" ]; then
        alpha=0.88; rho1=0.65; lmbd=2.25
    else
        alpha=0.95; rho1=0.5;  lmbd=1.5
    fi

    echo "===== $regime  p=$p  filter=$filter  treshRatio=$accept_ratio ====="
    "$PYTHON" agents/run_paper_eval.py --env barberis \
        --seeds $SEEDS --horizon $HORIZON --p-win $p \
        --alpha $alpha --rho1 $rho1 --rho2 $rho1 --lmbd $lmbd \
        --train-eps $TRAIN_EPS --batch $BATCH --support-size $SUPPORT \
        --critic-lr $CRITIC_LR --eps $EXPLORE_EPS \
        --spe-rollouts $SPE_ROLLOUTS --eval-per-state $EVAL_PER_STATE \
        --sticky-policy --filter-thresh $filter --filter-accept-ratio $accept_ratio \
        --results-dir $RESULTS_DIR \
        2>&1 | grep -v "Gym has\|Please upgrade\|Users of this\|See the migration"
    echo
}

# Per-cell filter mapping from paper §C.4 Tables 3/4 *-mark
# v2: re-runs ALL 10 cells (incl CPT88/0.66 — old data used wrong treshRatio=inf)
run_cell CPT88 0.72 0.95
run_cell CPT88 0.66 0.90
run_cell CPT88 0.60 0.85
run_cell CPT88 0.54 0.80
run_cell CPT88 0.48 1.00

# CPT95 cells
run_cell CPT95 0.72 0.95
run_cell CPT95 0.66 1.00
run_cell CPT95 0.60 0.80
run_cell CPT95 0.54 0.85
run_cell CPT95 0.48 0.90

echo
echo "===== sweep complete ====="
echo "Aggregate JSONs at $RESULTS_DIR/barberis_sperl_p*_cpt_*/aggregate.json"
