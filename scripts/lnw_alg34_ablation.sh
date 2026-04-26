#!/bin/bash
# Algorithm 3/4 ablation on LNW abandonment env.
#
# Runs sticky-policy / quantile-filter combinations on the hardest LNW
# config (T=7, p=0.6, CPT95 — chose this for highest baseline Disagree
# rate 4.2/28 from grid sweep) plus the stay-out config (T=5, p=0.6,
# CPT88) to see whether Alg 3/4 helps in CPT-induced edge cases.
#
# 5 seeds x 5k eps per cell. ~5 min wall time.

set -e

PYTHON="/c/Users/Jingxiang Tang/FNN/Scripts/python.exe"
SEEDS=5
EPS=5000
BATCH=1
SUPPORT=50
CRITIC_LR=0.04
EXPLORE_EPS=0.3
SPE_ROLLOUTS=2000
EVAL_PER_STATE=100

run_cell() {
    local label=$1
    local T=$2
    local p=$3
    local x1=$4
    local alpha=$5
    local rho1=$6
    local rho2=$7
    local lmbd=$8
    shift 8
    local extra_args="$@"

    echo "===== $label ====="
    "$PYTHON" agents/run_paper_eval.py --env abandonment --no-save \
        --seeds $SEEDS --horizon $T --x1 $x1 --c 11 --delta 10 \
        --p-win $p --alpha $alpha --rho1 $rho1 --rho2 $rho2 --lmbd $lmbd \
        --train-eps $EPS --batch $BATCH --support-size $SUPPORT \
        --critic-lr $CRITIC_LR --eps $EXPLORE_EPS \
        --spe-rollouts $SPE_ROLLOUTS --eval-per-state $EVAL_PER_STATE \
        $extra_args \
        2>&1 | grep -E "(Aggregate|Policy Error|Value Error|Optimality|Social Welfare|SPE Welfare)" \
             | grep -v "Gym\|Please\|Users\|See"
    echo
}

# Hardest config (T=7 p=0.6 CPT95, baseline Disagree 4.2/28 from grid)
HARD_T=7; HARD_P=0.6; HARD_X=70
HARD_A=0.95; HARD_R1=0.5; HARD_R2=0.5; HARD_L=1.5

# Stay-out config (T=5 p=0.6 CPT88, abandonment-ladder pattern)
STAY_T=5; STAY_P=0.6; STAY_X=50
STAY_A=0.88; STAY_R1=0.65; STAY_R2=0.65; STAY_L=2.25

# Helper to run all 4 ablation cells per config
run_ablation_set() {
    local label=$1
    shift
    local args=("$@")
    run_cell "$label baseline (no sticky, no filter)"           "${args[@]}"
    run_cell "$label sticky-only"                                "${args[@]}" --sticky-policy
    run_cell "$label filter=0.9 only"                            "${args[@]}" --filter-thresh 0.9
    run_cell "$label sticky + filter=0.9"                        "${args[@]}" --sticky-policy --filter-thresh 0.9
    run_cell "$label sticky + filter=0.75"                       "${args[@]}" --sticky-policy --filter-thresh 0.75
}

echo "######### HARDEST CONFIG: T=7 p=0.6 CPT95 #########"
run_ablation_set "HARD" $HARD_T $HARD_P $HARD_X $HARD_A $HARD_R1 $HARD_R2 $HARD_L

echo "######### STAY-OUT CONFIG: T=5 p=0.6 CPT88 #########"
run_ablation_set "STAYOUT" $STAY_T $STAY_P $STAY_X $STAY_A $STAY_R1 $STAY_R2 $STAY_L
