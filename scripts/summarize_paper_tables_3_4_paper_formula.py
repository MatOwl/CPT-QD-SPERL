"""Re-aggregate Tables 3/4 v3 sweep using paper §4.1 formulas (BUG #7 fix).

Paper §4.1 (page 23 of MSci_MANUSCRIPT.pdf) defines:

    Policy Error mean = sum_x | mu_seeds[pi_tilde(x)] - mu_seeds[pi_hat(x)] |
    Policy Error std  = (1/|X|) sum_x sigma_seeds[pi_hat(x)]

    Value Error mean  = sum_x | mu_seeds[V^pi_tilde(x)] - mu_seeds[V^pi_hat(x)] |
    Value Error std   = (1/|X|) sum_x sigma_seeds[V^pi_hat(x)]

KEY DIFFERENCE from the previous (incorrect) summarize script:
- Old aggregator: per-seed compute Sum_x |...|, then average across seeds.
  By Jensen / triangle inequality this OVERESTIMATES the paper formula
  (the per-seed |diff| picks up seed-to-seed noise that cancels when you
  take the seed-mean inside the |.|). Especially affects VE.
- New aggregator (this script): per-state, average V_tilde and V_hat over
  seeds first, THEN take |diff|. This matches paper §4.1.

Notes on the std formula:
- Paper averages sigma over states (not RMS), and uses sigma of the SPE
  benchmark side (pi_hat / V_hat), not SPERL. With our SPE oracle being
  near-deterministic per seed (each seed re-builds it via 2000-rollout MC,
  small noise), this std is small — explaining paper Table 3/4 std being
  in the 0.05-0.31 / 0.13-2.12 range.

Usage:
    PYTHONPATH=. python scripts/summarize_paper_tables_3_4_paper_formula.py
"""

from __future__ import annotations

import csv
import os
from math import sqrt

import numpy as np

from scripts.summarize_paper_tables_3_4 import (
    PAPER_T3_MEAN, PAPER_T4_MEAN, PAPER_PARETO, FILTERS, fmt_p, cpt_to_params,
)

SWEEP_DIR = "runs/results_paper_tables_3_4_v3"
N_SEEDS_EXPECTED = 10


def find_dir(cpt_lbl, p_win, filter):
    a, r, l = cpt_to_params(cpt_lbl)
    p_str = fmt_p(p_win)
    target = f"barberis_sperl_p{p_str}_cpt_a{a}_r{r}_l{l}"
    sub = f"filt{filter:.2f}"
    cand = os.path.join(SWEEP_DIR, sub, target)
    if os.path.exists(os.path.join(cand, "seed0", "per_state_values.csv")):
        return cand
    return None


def load_per_state_table(cell_dir):
    """Returns list-of-(state_str, a_tilde[seed], a_hat[seed], v_tilde[seed],
    v_hat[seed]) where each [seed] is an array of length n_seeds."""
    seed_dirs = sorted(d for d in os.listdir(cell_dir)
                       if d.startswith("seed") and
                       os.path.exists(os.path.join(cell_dir, d, "per_state_values.csv")))
    if not seed_dirs:
        return None
    n_seeds = len(seed_dirs)

    state_data = {}  # state_str -> {"a_tilde": [], "a_hat": [], "v_tilde": [], "v_hat": []}
    for seed_dir in seed_dirs:
        with open(os.path.join(cell_dir, seed_dir, "per_state_values.csv"),
                  newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = row["state"]
                d = state_data.setdefault(s, {"a_tilde": [], "a_hat": [],
                                              "v_tilde": [], "v_hat": []})
                d["a_tilde"].append(float(row["a_tilde"]))
                d["a_hat"].append(float(row["a_hat"]))
                d["v_tilde"].append(float(row["v_tilde"]))
                d["v_hat"].append(float(row["v_hat"]))
    # arrayify
    for s, d in state_data.items():
        for k in d:
            d[k] = np.asarray(d[k], dtype=np.float64)
    return state_data, n_seeds


def paper_metrics(state_data):
    """Apply paper §4.1 formulas:
        PE_mean = sum_x | mu_seeds[a_tilde(x)] - mu_seeds[a_hat(x)] |
        PE_std  = (1/|X|) sum_x sigma_seeds[a_hat(x)]
        VE_mean = sum_x | mu_seeds[v_tilde(x)] - mu_seeds[v_hat(x)] |
        VE_std  = (1/|X|) sum_x sigma_seeds[v_hat(x)]
    """
    states = list(state_data.keys())
    n_x = len(states)

    pe_mean_terms = []
    ve_mean_terms = []
    pe_std_terms = []
    ve_std_terms = []

    for s in states:
        d = state_data[s]
        pe_mean_terms.append(abs(d["a_tilde"].mean() - d["a_hat"].mean()))
        ve_mean_terms.append(abs(d["v_tilde"].mean() - d["v_hat"].mean()))
        # Paper std = sigma over seeds of the SPE side
        pe_std_terms.append(d["a_hat"].std())
        ve_std_terms.append(d["v_hat"].std())

    return {
        "pe_mean": float(sum(pe_mean_terms)),
        "pe_std": float(sum(pe_std_terms) / n_x),
        "ve_mean": float(sum(ve_mean_terms)),
        "ve_std": float(sum(ve_std_terms) / n_x),
        "n_states": n_x,
    }


def main():
    # Collect refactor v3 paper-formula stats
    data = {}
    for (cpt_lbl, p) in PAPER_T3_MEAN.keys():
        for f in FILTERS:
            cell_dir = find_dir(cpt_lbl, p, f)
            if cell_dir is None:
                data[(cpt_lbl, p, f)] = None
                continue
            sd, n_seeds = load_per_state_table(cell_dir)
            metrics = paper_metrics(sd)
            metrics["n_seeds"] = n_seeds
            data[(cpt_lbl, p, f)] = metrics

    # ---------- Per-cell tables ----------
    print("=" * 105)
    print("Paper Tables 3/4 vs refactor v3 (paper §4.1 formula: mean-of-seeds first, abs-diff later)")
    print("=" * 105)

    for (cpt_lbl, p) in PAPER_T3_MEAN.keys():
        print(f"\n--- {cpt_lbl}/p={p} ---")
        print(f"{'filter':>7} | {'PE refactor':>16} | {'PE paper mean':>14} | "
              f"{'VE refactor':>16} | {'VE paper mean':>14}")
        print("-" * 90)
        for f in FILTERS:
            d = data.get((cpt_lbl, p, f))
            if d is None:
                print(f"{f:>7.2f} | MISSING")
                continue
            star = " *" if PAPER_PARETO[(cpt_lbl, p)] == f else "  "
            print(f"{f:>6.2f}{star} | "
                  f"{d['pe_mean']:>5.2f} +/- {d['pe_std']:>5.3f}  | "
                  f"{PAPER_T3_MEAN[(cpt_lbl, p)][f]:>13.2f}  | "
                  f"{d['ve_mean']:>5.2f} +/- {d['ve_std']:>5.3f}  | "
                  f"{PAPER_T4_MEAN[(cpt_lbl, p)][f]:>13.2f}")

    # ---------- Mean comparison summary ----------
    print("\n" + "=" * 105)
    print("Direction summary: how often is refactor mean smaller than paper mean?")
    print("=" * 105)

    pe_smaller = pe_larger = ve_smaller = ve_larger = 0
    pe_diff_sum = ve_diff_sum = 0.0
    pe_diffs = []
    ve_diffs = []
    sum_R_VE = sum_P_VE = 0.0
    for (cpt_lbl, p) in PAPER_T3_MEAN.keys():
        for f in FILTERS:
            d = data.get((cpt_lbl, p, f))
            if d is None:
                continue
            r_pe, r_ve = d["pe_mean"], d["ve_mean"]
            p_pe = PAPER_T3_MEAN[(cpt_lbl, p)][f]
            p_ve = PAPER_T4_MEAN[(cpt_lbl, p)][f]
            if r_pe < p_pe: pe_smaller += 1
            else: pe_larger += 1
            if r_ve < p_ve: ve_smaller += 1
            else: ve_larger += 1
            pe_diff_sum += (r_pe - p_pe)
            ve_diff_sum += (r_ve - p_ve)
            pe_diffs.append(r_pe - p_pe)
            ve_diffs.append(r_ve - p_ve)
            sum_R_VE += r_ve
            sum_P_VE += p_ve

    n = pe_smaller + pe_larger
    print(f"Total cells: {n}")
    print(f"PE: refactor < paper in {pe_smaller}/{n}, refactor >= paper in {pe_larger}/{n}")
    print(f"PE: mean(R-P) = {pe_diff_sum / n:+.3f}, median(R-P) = {np.median(pe_diffs):+.3f},"
          f" max = {max(pe_diffs):+.3f}, min = {min(pe_diffs):+.3f}")
    print(f"VE: refactor < paper in {ve_smaller}/{n}, refactor >= paper in {ve_larger}/{n}")
    print(f"VE: mean(R-P) = {ve_diff_sum / n:+.3f}, median(R-P) = {np.median(ve_diffs):+.3f},"
          f" max = {max(ve_diffs):+.3f}, min = {min(ve_diffs):+.3f}")
    print(f"VE: aggregate ratio sum(R) / sum(P) = {sum_R_VE / sum_P_VE:.2f}x")

    # ---------- Compact 60-cell output for md ----------
    print("\n" + "=" * 105)
    print("60-cell compact: refactor (paper formula) vs paper")
    print("=" * 105)
    print()
    for kind, paper_dict, key in [
        ("Policy Error", PAPER_T3_MEAN, "pe_mean"),
        ("Value Error",  PAPER_T4_MEAN, "ve_mean"),
    ]:
        print(f"--- {kind} (paper formula) ---")
        print(f"{'Cell':<13}", end="")
        for f in FILTERS:
            print(f" |  flt={f:.2f}  ", end="")
        print()
        print("-" * (13 + 13 * 6))
        for (cpt_lbl, p) in paper_dict.keys():
            print(f"R {cpt_lbl}/{p:<5.2f}", end="")
            for f in FILTERS:
                d = data.get((cpt_lbl, p, f))
                if d is None:
                    print(" |   ----   ", end="")
                else:
                    print(f" |  {d[key]:5.2f}  ", end="")
            print()
            print(f"P {cpt_lbl}/{p:<5.2f}", end="")
            for f in FILTERS:
                print(f" |  {paper_dict[(cpt_lbl, p)][f]:5.2f}  ", end="")
            print()
            print()


if __name__ == "__main__":
    main()
