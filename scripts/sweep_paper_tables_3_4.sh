#!/bin/bash
# Paper Tables 3/4 SPERL filter-sweep: 10 cells x 6 filter values x 10 seeds = 600 runs.
# Per paper §C.4: filterTresh ∈ {1.0, .95, .9, .85, .8, .75}; treshRatio = 0.5 (filter<1) or 0 (filter=1).
# Other hyperparams per paper §C.2.5 (M=15000, K=50, ϱ=.04, ξ=.3).
#
# Estimated wall time: ~90 sec/cell × 60 = ~90 min.
#
# Usage:
#   bash scripts/sweep_paper_tables_3_4.sh
#
# Outputs: runs/results_paper_tables_3_4/barberis_sperl_p<p>_cpt_*/

set -e

PYTHON="/c/Users/Jingxiang Tang/FNN/Scripts/python.exe"
RESULTS_DIR="./runs/results_paper_tables_3_4_v3"

# Paper §C.2.5 hyperparams
SEEDS=10
TRAIN_EPS=15000
BATCH=5
SUPPORT=50
CRITIC_LR=0.04
EXPLORE_EPS=0.3
HORIZON=5
SPE_ROLLOUTS=2000  # paper Algorithm 5 M = 2000 (was 300; bumped 2026-04-28)
EVAL_PER_STATE=100

run_cell() {
    local regime=$1
    local p=$2
    local filter=$3

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

    local subdir="filt${filter}"
    echo "===== $regime  p=$p  filter=$filter  treshRatio=$accept_ratio ====="
    "$PYTHON" agents/run_paper_eval.py --env barberis \
        --seeds $SEEDS --horizon $HORIZON --p-win $p \
        --alpha $alpha --rho1 $rho1 --rho2 $rho1 --lmbd $lmbd \
        --train-eps $TRAIN_EPS --batch $BATCH --support-size $SUPPORT \
        --critic-lr $CRITIC_LR --eps $EXPLORE_EPS \
        --spe-rollouts $SPE_ROLLOUTS --eval-per-state $EVAL_PER_STATE \
        --sticky-policy --filter-thresh $filter --filter-accept-ratio $accept_ratio \
        --results-dir "$RESULTS_DIR/$subdir" \
        2>&1 | grep -v "Gym has\|Please upgrade\|Users of this\|See the migration"
    echo
}

for FILTER in 1.00 0.95 0.90 0.85 0.80 0.75; do
    for REGIME in CPT88 CPT95; do
        for P in 0.72 0.66 0.60 0.54 0.48; do
            run_cell $REGIME $P $FILTER
        done
    done
done

echo
echo "===== Tables 3/4 sweep complete ====="
