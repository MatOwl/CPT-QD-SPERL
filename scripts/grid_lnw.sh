#!/bin/bash
# LNW grid sweep: T x p_win x CPT_regime, 5 seeds x 5k eps each.
# ~12 min wall time on default Windows venv.
#
# Usage:
#   bash scripts/grid_lnw.sh
#
# Outputs go to runs/abandonment_sperl_T*_x1*_c11_d10_p*_cpt_*/

set -e

PYTHON="/c/Users/Jingxiang Tang/FNN/Scripts/python.exe"

# Common args
SEEDS=5
EPS=5000
BATCH=1
SUPPORT=50
CRITIC_LR=0.04
EXPLORE_EPS=0.3
SPE_ROLLOUTS=2000
EVAL_PER_STATE=100

# CPT regimes
declare -A CPT88=( [alpha]=0.88 [rho1]=0.65 [rho2]=0.65 [lmbd]=2.25 )
declare -A CPT95=( [alpha]=0.95 [rho1]=0.5  [rho2]=0.5  [lmbd]=1.5  )

run_cell() {
    local T=$1
    local p=$2
    local regime=$3  # CPT88 or CPT95
    local x1=$4

    local alpha rho1 rho2 lmbd
    if [ "$regime" = "CPT88" ]; then
        alpha=0.88; rho1=0.65; rho2=0.65; lmbd=2.25
    else
        alpha=0.95; rho1=0.5; rho2=0.5; lmbd=1.5
    fi

    echo "===== T=$T  p=$p  $regime  x1=$x1 ====="
    "$PYTHON" agents/run_paper_eval.py --env abandonment \
        --seeds $SEEDS --horizon $T --x1 $x1 --c 11 --delta 10 \
        --p-win $p --alpha $alpha --rho1 $rho1 --rho2 $rho2 --lmbd $lmbd \
        --train-eps $EPS --batch $BATCH --support-size $SUPPORT \
        --critic-lr $CRITIC_LR --eps $EXPLORE_EPS \
        --spe-rollouts $SPE_ROLLOUTS --eval-per-state $EVAL_PER_STATE \
        2>&1 | grep -E "(\[seed |Aggregate|Policy Error|Value Error|Optimality|Social Welfare|SPE Welfare)" \
             | grep -v "Gym\|Please\|Users\|See"
    echo
}

# Grid: T in {5, 7}, p in {0.6, 0.72}, CPT in {CPT88, CPT95}, x1 = unprofitable initial
# (paper convention: x1 = (T_paper-1)*delta = (T+1-1)*delta = (T+0)*delta if env.T=T_paper-1?
# Wait: x1 unprofitable = (T_paper-1)*delta. With env.T = T_paper-1, x1 = env.T * delta.)
for T in 5 7; do
    X1=$(( T * 10 ))  # unprofitable initial
    for P in 0.6 0.72; do
        for REGIME in CPT88 CPT95; do
            run_cell $T $P $REGIME $X1
        done
    done
done
