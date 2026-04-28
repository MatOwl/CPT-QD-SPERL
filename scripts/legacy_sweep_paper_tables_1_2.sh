#!/bin/bash
# Sweep paper Tables 1/2 with the LEGACY SPERL agent
# (``agents/rerun_GreedySPERL_QR__main.py``).
#
# Wraps the parameterized legacy main + the legacy_paper_eval.py post-processor
# to reproduce paper §C.4 *-mark filter assignments for each (CPT regime, p_win)
# cell with legacy production hp (ε=0.6, batch=1, M=30k).
#
# Wall time est: ~1.5 min/seed × 10 seeds × 10 cells = ~150 min.
#
# Usage:
#   bash scripts/legacy_sweep_paper_tables_1_2.sh
#
# Outputs:
#   - Legacy raw CSVs at agents/barberis/results/static/SPERL_aggr_*_*.csv
#   - Refactor-style aggregate.json at runs/results_legacy_tables_1_2/<cell>/

set -e

PYTHON="/c/Users/Jingxiang Tang/FNN/Scripts/python.exe"
LEGACY_DIR="agents/barberis/results/static"
RESULTS_DIR="runs/results_legacy_tables_1_2"

# Paper §C.2.5 alignment (ξ=0.3, M=15000), legacy n_batch=1 (paper doesn't
# specify batch size). train_num × 2 × n_batch = total episodes; with
# n_batch=1 and train_num=7500 → 15000 episodes (= paper M).
SEED_LO=5
SEED_HI=15        # exclusive → 10 seeds
EPS=0.3
N_BATCH=1
TRAIN_NUM=7500    # 7500 × 2 × 1 = 15000 episodes (paper M=15000)

run_cell() {
    local regime=$1
    local p=$2
    local filter=$3

    # Paper §C.2.5: treshRatio = 0.5 if filter < 1, else 0
    local tresh
    local lbub
    if [ "$filter" = "1.0" ]; then
        tresh=0
        lbub=0    # legacy: lbub=0 → no filter; aggregate filename gets "0"
    else
        tresh=0.5
        lbub=1
    fi

    local alpha rho lmbd
    if [ "$regime" = "CPT88" ]; then
        alpha=0.88; rho=0.65; lmbd=2.25
    else
        alpha=0.95; rho=0.5;  lmbd=1.5
    fi

    # Legacy aggregate-CSV filename uses lbub*p_filter as the "filter" token.
    # When lbub=0, the token is 0 (no filter). When lbub=1, it's the filter.
    local filter_token="$filter"
    if [ "$lbub" = "0" ]; then
        filter_token=0
    fi

    echo
    echo "===== $regime  p=$p  filter=$filter  treshRatio=$tresh  lbub=$lbub  ====="

    # 1. Clean stale legacy aggregate CSVs for this cell to prevent seed
    #    rows from accumulating across runs.
    local stem="$LEGACY_DIR/SPERL_aggr_${alpha}_${rho}_${lmbd}_${p}_50_${filter_token}_${tresh}_1_${EPS}"
    rm -f "${stem}_trainPol.csv" "${stem}_trainQ.csv" "${stem}_trueQPol.csv" "${stem}_visitFreq.csv"

    # 2. Train legacy SPERL for 10 seeds.
    (cd agents && \
        LEGACY_ALPHA=$alpha LEGACY_RHO1=$rho LEGACY_LMBD=$lmbd \
        LEGACY_PWIN=$p LEGACY_FILTER=$filter LEGACY_TRESHRATIO=$tresh \
        LEGACY_LBUB=$lbub LEGACY_FIRST_VISIT=1 \
        LEGACY_SEED_LO=$SEED_LO LEGACY_SEED_HI=$SEED_HI \
        LEGACY_EPS=$EPS LEGACY_N_BATCH=$N_BATCH LEGACY_TRAIN_NUM=$TRAIN_NUM \
        PYTHONPATH=. PYTHONIOENCODING=utf-8 \
        "$PYTHON" rerun_GreedySPERL_QR__main.py > /tmp/legacy_train.log 2>&1)
    echo "  legacy SPERL train done (${SEED_HI} - ${SEED_LO} = $((SEED_HI - SEED_LO)) seeds)"

    # 3. Post-process via refactor's eval pipeline (per-seed SPE oracle).
    PYTHONPATH=. PYTHONIOENCODING=utf-8 \
        "$PYTHON" agents/legacy_paper_eval.py \
            --alpha $alpha --rho1 $rho --lmbd $lmbd \
            --p-win $p --filter $filter_token --treshratio $tresh --eps $EPS \
            --legacy-dir "$LEGACY_DIR" \
            --results-dir "$RESULTS_DIR" \
            2>&1 | grep -v "Gym has\|Please upgrade\|Users of this\|See the migration"
}

# Per-cell filters from paper §C.4 Tables 3/4 *-mark
run_cell CPT88 0.72 0.95
run_cell CPT88 0.66 0.9
run_cell CPT88 0.6  0.85
run_cell CPT88 0.54 0.8
run_cell CPT88 0.48 1.0

run_cell CPT95 0.72 0.95
run_cell CPT95 0.66 1.0
run_cell CPT95 0.6  0.8
run_cell CPT95 0.54 0.85
run_cell CPT95 0.48 0.9

echo
echo "===== legacy Tables 1/2 sweep complete ====="
echo "Aggregate JSONs at $RESULTS_DIR/<cell>/aggregate.json"
