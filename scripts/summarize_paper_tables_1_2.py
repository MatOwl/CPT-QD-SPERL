"""Read all 10 paper-Tables-1/2 cells from refactor sweep + existing CPT88/0.66
and emit a 3-way comparison: Paper (PDF) vs Legacy (where available) vs Refactor.

Paper Tables 1/2 SPERL columns (verified extract from paperRef/MSci_MANUSCRIPT.pdf p.25):
  Table 1 Optimality (mean +/- stdev):
    CPT88: p=.72->7.91+/-1.64, p=.66->2.74+/-.95, p=.60->0+/-0, p=.54->0+/-0, p=.48->0+/-0
    CPT95: p=.72->-.03+/-.33, p=.66->0+/-0,    p=.60->0+/-0, p=.54->0+/-0, p=.48->0+/-0
  Table 2 Social Welfare (mean +/- stdev):
    CPT88: -26.03+/-3.63, -59.94+/-2.88, -80.09+/-4.22, -91.49+/-2.93, -104.41+/-2.13
    CPT95: -7.31+/-3.32, -17.56+/-1.2, -23.67+/-1.04, -30.36+/-.99, -35.61+/-2.37

Legacy data only available at CPT88/p=0.66/filter=0.9 (the verification run,
agents/barberis/results/static/SPERL_27042026{215842,...,223834}.{txt,csv}).

Usage:
    PYTHONPATH=. python scripts/summarize_paper_tables_1_2.py
"""

from __future__ import annotations

import json
import os
from math import sqrt

SWEEP_DIR_V1 = "runs/results_paper_tables_1_2"     # treshRatio=inf (wrong wrt paper)
SWEEP_DIR_V2 = "runs/results_paper_tables_1_2_v2"  # treshRatio=0.5 / 0 (paper §C.2.5)
P066_DIR     = "runs/results_p066_10s_both_acceptInf"
SWEEP_DIRS = [SWEEP_DIR_V2, SWEEP_DIR_V1, P066_DIR]  # prefer v2

# (CPT_label, alpha, rho, lmbd, p_win, filter,
#  paper_opt_mean, paper_opt_std, paper_sw_mean, paper_sw_std)
CELLS = [
    ("CPT88", 0.88, 0.65, 2.25, 0.72, 0.95,  7.91, 1.64,  -26.03, 3.63),
    ("CPT88", 0.88, 0.65, 2.25, 0.66, 0.90,  2.74, 0.95,  -59.94, 2.88),
    ("CPT88", 0.88, 0.65, 2.25, 0.60, 0.85,  0.00, 0.00,  -80.09, 4.22),
    ("CPT88", 0.88, 0.65, 2.25, 0.54, 0.80,  0.00, 0.00,  -91.49, 2.93),
    ("CPT88", 0.88, 0.65, 2.25, 0.48, 1.00,  0.00, 0.00, -104.41, 2.13),
    ("CPT95", 0.95, 0.50, 1.50, 0.72, 0.95, -0.03, 0.33,   -7.31, 3.32),
    ("CPT95", 0.95, 0.50, 1.50, 0.66, 1.00,  0.00, 0.00,  -17.56, 1.20),
    ("CPT95", 0.95, 0.50, 1.50, 0.60, 0.80,  0.00, 0.00,  -23.67, 1.04),
    ("CPT95", 0.95, 0.50, 1.50, 0.54, 0.85,  0.00, 0.00,  -30.36, 0.99),
    ("CPT95", 0.95, 0.50, 1.50, 0.48, 0.90,  0.00, 0.00,  -35.61, 2.37),
]

# Legacy data: only the verification cell (CPT88/p=0.66/filter=0.9) exists.
# Numbers from native run on 2026-04-27 (10 seeds, refactor-matched config),
# parsed via scripts/parse_legacy_native.py.
LEGACY = {
    ("CPT88", 0.66): {
        "opt_mu": 0.452, "opt_sd": 2.626,
        "dis_mu": 5.60,  "dis_sd": 1.50,
        "ve_mu": 27.00,
        "sw_mu": -67.46, "sw_sd": 7.77,
    },
}


def fmt_p(p):
    s = f"{p:.2f}"
    if s.endswith("0"):
        s = s[:-1]
    return s


def find_dir(alpha, rho, lmbd, p_win):
    p_str = fmt_p(p_win)
    target = f"barberis_sperl_p{p_str}_cpt_a{alpha}_r{rho}_l{lmbd}"
    for root in SWEEP_DIRS:
        cand = os.path.join(root, target)
        if os.path.exists(os.path.join(cand, "aggregate.json")):
            return cand
    return None


def zscore(mu1, sd1, mu2, sd2):
    pool_sd = sqrt((sd1**2 + sd2**2) / 2)
    if pool_sd == 0:
        return 0.0 if mu1 == mu2 else float("inf")
    return abs(mu1 - mu2) / pool_sd


def main():
    print("=" * 130)
    print("Paper Tables 1/2 SPERL: 3-way comparison (Paper vs Legacy vs Refactor)")
    print("=" * 130)

    print(f"\n--- Optimality ---")
    print(f"{'Cell':<12} {'flt':>4} | {'Paper':>15} | {'Legacy':>15} | {'Refactor':>15}"
          f" | {'|R-P|/s':>8}")
    print("-" * 95)

    opt_within_1s = 0
    opt_total = 0
    for (lbl, a, r, l, p, flt, p_mu, p_sd, _, _) in CELLS:
        d = find_dir(a, r, l, p)
        agg = json.load(open(os.path.join(d, "aggregate.json")))
        rf_mu = agg["optimality"]["mean"]
        rf_sd = agg["optimality"]["std"]

        leg = LEGACY.get((lbl, p))
        leg_str = (f"{leg['opt_mu']:>5.2f} +/- {leg['opt_sd']:.2f}"
                   if leg else "      n/a      ")

        z = zscore(rf_mu, rf_sd, p_mu, p_sd)
        ok = "*" if z <= 1.0 else " "
        if z <= 1.0:
            opt_within_1s += 1
        opt_total += 1

        print(f"{lbl}/{p:<5.2f} {flt:>4.2f} | "
              f"{p_mu:>5.2f} +/- {p_sd:>4.2f} | "
              f"{leg_str:>15} | "
              f"{rf_mu:>5.2f} +/- {rf_sd:>4.2f} | "
              f"{z:>6.2f}s {ok}")

    print(f"\nOptimality within 1s vs paper: {opt_within_1s}/{opt_total}")

    print(f"\n--- Social Welfare ---")
    print(f"{'Cell':<12} {'flt':>4} | {'Paper':>17} | {'Legacy':>17} | {'Refactor':>17}"
          f" | {'|R-P|/s':>8}")
    print("-" * 100)

    sw_within_1s = 0
    sw_total = 0
    for (lbl, a, r, l, p, flt, _, _, p_mu, p_sd) in CELLS:
        d = find_dir(a, r, l, p)
        agg = json.load(open(os.path.join(d, "aggregate.json")))
        rf_mu = agg["social_welfare"]["mean"]
        rf_sd = agg["social_welfare"]["std"]

        leg = LEGACY.get((lbl, p))
        leg_str = (f"{leg['sw_mu']:>6.2f} +/- {leg['sw_sd']:.2f}"
                   if leg else "       n/a       ")

        z = zscore(rf_mu, rf_sd, p_mu, p_sd)
        ok = "*" if z <= 1.0 else " "
        if z <= 1.0:
            sw_within_1s += 1
        sw_total += 1

        print(f"{lbl}/{p:<5.2f} {flt:>4.2f} | "
              f"{p_mu:>7.2f} +/- {p_sd:>4.2f} | "
              f"{leg_str:>17} | "
              f"{rf_mu:>7.2f} +/- {rf_sd:>4.2f} | "
              f"{z:>6.2f}s {ok}")

    print(f"\nSocial Welfare within 1s vs paper: {sw_within_1s}/{sw_total}")

    # 3-way at the verification cell
    print(f"\n--- 3-way at verification cell (CPT88/p=0.66/filter=0.9) ---")
    leg = LEGACY[("CPT88", 0.66)]
    p_cell = next(c for c in CELLS if c[0] == "CPT88" and c[4] == 0.66)
    d = find_dir(p_cell[1], p_cell[2], p_cell[3], p_cell[4])
    agg = json.load(open(os.path.join(d, "aggregate.json")))
    print(f"  Optimality:")
    print(f"    Paper    : {p_cell[6]:>6.2f} +/- {p_cell[7]:.2f}")
    print(f"    Legacy   : {leg['opt_mu']:>6.2f} +/- {leg['opt_sd']:.2f}")
    print(f"    Refactor : {agg['optimality']['mean']:>6.2f} +/- {agg['optimality']['std']:.2f}")
    print(f"    => Legacy vs Paper: {zscore(leg['opt_mu'], leg['opt_sd'], p_cell[6], p_cell[7]):.2f}s")
    print(f"    => Refac  vs Paper: {zscore(agg['optimality']['mean'], agg['optimality']['std'], p_cell[6], p_cell[7]):.2f}s")
    print(f"    => Refac  vs Legacy: {zscore(agg['optimality']['mean'], agg['optimality']['std'], leg['opt_mu'], leg['opt_sd']):.2f}s")
    print(f"  Social Welfare:")
    print(f"    Paper    : {p_cell[8]:>7.2f} +/- {p_cell[9]:.2f}")
    print(f"    Legacy   : {leg['sw_mu']:>7.2f} +/- {leg['sw_sd']:.2f}")
    print(f"    Refactor : {agg['social_welfare']['mean']:>7.2f} +/- {agg['social_welfare']['std']:.2f}")
    print(f"    => Legacy vs Paper: {zscore(leg['sw_mu'], leg['sw_sd'], p_cell[8], p_cell[9]):.2f}s")
    print(f"    => Refac  vs Paper: {zscore(agg['social_welfare']['mean'], agg['social_welfare']['std'], p_cell[8], p_cell[9]):.2f}s")
    print(f"    => Refac  vs Legacy: {zscore(agg['social_welfare']['mean'], agg['social_welfare']['std'], leg['sw_mu'], leg['sw_sd']):.2f}s")


if __name__ == "__main__":
    main()
