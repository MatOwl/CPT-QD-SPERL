"""Read all 60 cells (10 paper cells × 6 filter values) from
runs/results_paper_tables_3_4 and compare to paper Tables 3/4.

Paper Tables 3/4 SPERL data extracted from paperRef/MSci_MANUSCRIPT.pdf p.53-54.

⚠️ The PDF text extraction sometimes drops the leading "1." in std subscripts
(e.g., "8.82.12" might be 8.82 ± 1.12 not 0.12). This script therefore:
  1. Compares refactor MEAN to paper MEAN only (high-confidence transcription)
  2. Prints refactor raw (mean, std) so they can be visually inspected
  3. Recomputes Pareto-optimal *-mark from REFACTOR data and compares to paper

Usage:
    PYTHONPATH=. python scripts/summarize_paper_tables_3_4.py
"""

from __future__ import annotations

import json
import os
from math import sqrt

SWEEP_DIR = "runs/results_paper_tables_3_4_v3"

FILTERS = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75]

# Paper Table 3 (Policy Error MEAN only) — extracted from PDF p.53.
# Stds are sometimes ambiguous due to PDF leading-1 stripping; means are reliable.
PAPER_T3_MEAN = {
    ("CPT88", 0.72): {1.00: 0.89, 0.95: 0.44, 0.90: 0.67, 0.85: 0.78, 0.80: 1.33, 0.75: 1.11},
    ("CPT88", 0.66): {1.00: 5.11, 0.95: 3.44, 0.90: 2.89, 0.85: 3.00, 0.80: 3.00, 0.75: 3.11},
    ("CPT88", 0.60): {1.00: 1.22, 0.95: 1.44, 0.90: 0.78, 0.85: 0.67, 0.80: 1.11, 0.75: 0.78},
    ("CPT88", 0.54): {1.00: 1.67, 0.95: 1.56, 0.90: 1.11, 0.85: 1.33, 0.80: 0.56, 0.75: 1.00},
    ("CPT88", 0.48): {1.00: 1.67, 0.95: 2.11, 0.90: 3.33, 0.85: 2.44, 0.80: 2.78, 0.75: 2.33},
    ("CPT95", 0.72): {1.00: 1.11, 0.95: 0.88, 0.90: 1.11, 0.85: 1.22, 0.80: 1.11, 0.75: 1.44},
    ("CPT95", 0.66): {1.00: 0.22, 0.95: 0.44, 0.90: 0.78, 0.85: 0.56, 0.80: 0.22, 0.75: 1.00},
    ("CPT95", 0.60): {1.00: 0.22, 0.95: 0.11, 0.90: 0.22, 0.85: 0.22, 0.80: 0.11, 0.75: 0.11},
    ("CPT95", 0.54): {1.00: 0.67, 0.95: 0.78, 0.90: 0.78, 0.85: 0.44, 0.80: 0.67, 0.75: 0.89},
    ("CPT95", 0.48): {1.00: 0.22, 0.95: 0.44, 0.90: 0.22, 0.85: 0.56, 0.80: 0.56, 0.75: 0.78},
}

PAPER_T4_MEAN = {
    ("CPT88", 0.72): {1.00: 5.98, 0.95: 3.71, 0.90: 5.41, 0.85: 5.90, 0.80: 8.32, 0.75: 7.90},
    ("CPT88", 0.66): {1.00: 14.72, 0.95: 10.66, 0.90: 9.30, 0.85: 9.47, 0.80: 8.82, 0.75: 9.72},
    ("CPT88", 0.60): {1.00: 2.79, 0.95: 2.86, 0.90: 1.68, 0.85: 1.65, 0.80: 1.71, 0.75: 1.58},
    ("CPT88", 0.54): {1.00: 4.96, 0.95: 3.57, 0.90: 3.55, 0.85: 3.79, 0.80: 2.37, 0.75: 2.29},
    ("CPT88", 0.48): {1.00: 4.22, 0.95: 5.82, 0.90: 8.16, 0.85: 5.74, 0.80: 5.00, 0.75: 3.98},
    ("CPT95", 0.72): {1.00: 4.91, 0.95: 3.28, 0.90: 3.93, 0.85: 2.73, 0.80: 2.57, 0.75: 4.34},
    ("CPT95", 0.66): {1.00: 1.91, 0.95: 3.41, 0.90: 4.35, 0.85: 3.74, 0.80: 1.24, 0.75: 5.43},
    ("CPT95", 0.60): {1.00: 0.94, 0.95: 0.60, 0.90: 0.58, 0.85: 0.52, 0.80: 0.56, 0.75: 0.84},
    ("CPT95", 0.54): {1.00: 3.30, 0.95: 1.88, 0.90: 1.39, 0.85: 1.13, 0.80: 1.08, 0.75: 1.35},
    ("CPT95", 0.48): {1.00: 2.16, 0.95: 2.50, 0.90: 1.21, 0.85: 3.35, 0.80: 2.65, 0.75: 2.74},
}

# Paper *-mark Pareto-optimal filter (per row in Tables 3 and 4)
PAPER_PARETO = {
    ("CPT88", 0.72): 0.95, ("CPT88", 0.66): 0.90, ("CPT88", 0.60): 0.85,
    ("CPT88", 0.54): 0.80, ("CPT88", 0.48): 1.00,
    ("CPT95", 0.72): 0.95, ("CPT95", 0.66): 1.00, ("CPT95", 0.60): 0.80,
    ("CPT95", 0.54): 0.85, ("CPT95", 0.48): 0.90,
}


def fmt_p(p):
    s = f"{p:.2f}"
    if s.endswith("0"):
        s = s[:-1]
    return s


def cpt_to_params(lbl):
    return (0.88, 0.65, 2.25) if lbl == "CPT88" else (0.95, 0.5, 1.5)


def find_dir(cpt_lbl, p_win, filter):
    a, r, l = cpt_to_params(cpt_lbl)
    p_str = fmt_p(p_win)
    target = f"barberis_sperl_p{p_str}_cpt_a{a}_r{r}_l{l}"
    sub = f"filt{filter:.2f}"
    cand = os.path.join(SWEEP_DIR, sub, target)
    if os.path.exists(os.path.join(cand, "aggregate.json")):
        return cand
    return None


def is_pareto_dominated(this_mu, this_sd, others):
    """A point (mu, sd) is dominated iff some other has both ≤ AND at least
    one strict inequality."""
    for o_mu, o_sd in others:
        if (o_mu < this_mu and o_sd <= this_sd) or (o_mu <= this_mu and o_sd < this_sd):
            return True
    return False


def find_pareto(cell_data):
    """Given dict {filter: (mu, sd)}, return list of Pareto-optimal filters in
    (mean, stdev) space (both minimized)."""
    pareto = []
    items = list(cell_data.items())
    for filter, (mu, sd) in items:
        others = [(m, s) for f, (m, s) in items if f != filter]
        if not is_pareto_dominated(mu, sd, others):
            pareto.append(filter)
    return pareto


def collect_refactor_data():
    """Returns dict {(cpt_lbl, p, filter): {'pe_mu', 'pe_sd', 've_mu', 've_sd', 'opt_mu', 'sw_mu'}}.

    Uses paper §4.1 PE/VE formula (mean = sum_x |mu_seeds[π̃] - mu_seeds[π̂]|;
    std = (1/|X|) sum_x sigma_seeds[π̃(x)]) when present in aggregate.json,
    falling back to the old per-seed aggregator otherwise. Run
    ``scripts/reaggregate_paper_std.py <sweep_root>`` after sweep-time changes
    to refresh the paper-formula entries.
    """
    data = {}
    for (cpt_lbl, p), _ in PAPER_T3_MEAN.items():
        for filter in FILTERS:
            d = find_dir(cpt_lbl, p, filter)
            if d is None:
                data[(cpt_lbl, p, filter)] = None
                continue
            agg = json.load(open(os.path.join(d, "aggregate.json")))
            pe = agg.get("policy_error_paper") or agg["policy_disagree_total"]
            ve = agg.get("value_error_paper") or agg["value_error_total"]
            data[(cpt_lbl, p, filter)] = {
                "pe_mu": pe["mean"],
                "pe_sd": pe["std"],
                "ve_mu": ve["mean"],
                "ve_sd": ve["std"],
                "opt_mu": agg["optimality"]["mean"],
                "sw_mu": agg["social_welfare"]["mean"],
            }
    return data


def main():
    data = collect_refactor_data()

    print("=" * 105)
    print("Paper Tables 3/4 vs refactor sweep (10 cells × 6 filters × 10 seeds)")
    print("=" * 105)

    # ----------- Per-cell tables -----------
    for (cpt_lbl, p), _ in PAPER_T3_MEAN.items():
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
                  f"{d['pe_mu']:>5.2f} +/- {d['pe_sd']:>5.2f}  | "
                  f"{PAPER_T3_MEAN[(cpt_lbl, p)][f]:>13.2f}  | "
                  f"{d['ve_mu']:>5.2f} +/- {d['ve_sd']:>5.2f}  | "
                  f"{PAPER_T4_MEAN[(cpt_lbl, p)][f]:>13.2f}")

    # ----------- Mean-only summary -----------
    print("\n" + "=" * 105)
    print("Mean-only comparison (paper text high-confidence; refactor mean from sweep)")
    print("=" * 105)
    print(f"\n--- Policy Error MEAN ---")
    print(f"{'Cell':<13}", end="")
    for f in FILTERS:
        print(f" |  flt={f:.2f}  ", end="")
    print()
    print("-" * (13 + 13 * 6))

    pe_close = 0
    pe_total = 0
    for (cpt_lbl, p) in PAPER_T3_MEAN.keys():
        # Refactor row
        print(f"R {cpt_lbl}/{p:<5.2f}", end="")
        for f in FILTERS:
            d = data.get((cpt_lbl, p, f))
            if d is None:
                print(" |   ----   ", end="")
                continue
            print(f" |  {d['pe_mu']:5.2f}  ", end="")
        print()
        # Paper row
        print(f"P {cpt_lbl}/{p:<5.2f}", end="")
        for f in FILTERS:
            print(f" |  {PAPER_T3_MEAN[(cpt_lbl, p)][f]:5.2f}  ", end="")
        print()
        # |R-P| row
        print(f"  |R-P|     ", end="")
        for f in FILTERS:
            d = data.get((cpt_lbl, p, f))
            if d is None:
                print(" |   ----   ", end="")
                continue
            diff = abs(d['pe_mu'] - PAPER_T3_MEAN[(cpt_lbl, p)][f])
            tag = "*" if diff < 0.5 else " "
            if diff < 0.5:
                pe_close += 1
            pe_total += 1
            print(f" | {diff:5.2f}{tag}  ", end="")
        print()
        print()

    print(f"PE: {pe_close}/{pe_total} cells with |R-P| < 0.5")

    print(f"\n--- Value Error MEAN ---")
    print(f"{'Cell':<13}", end="")
    for f in FILTERS:
        print(f" |  flt={f:.2f}  ", end="")
    print()
    print("-" * (13 + 13 * 6))

    ve_close = 0
    ve_total = 0
    for (cpt_lbl, p) in PAPER_T4_MEAN.keys():
        print(f"R {cpt_lbl}/{p:<5.2f}", end="")
        for f in FILTERS:
            d = data.get((cpt_lbl, p, f))
            if d is None:
                print(" |   ----   ", end="")
                continue
            print(f" |  {d['ve_mu']:5.2f}  ", end="")
        print()
        print(f"P {cpt_lbl}/{p:<5.2f}", end="")
        for f in FILTERS:
            print(f" |  {PAPER_T4_MEAN[(cpt_lbl, p)][f]:5.2f}  ", end="")
        print()
        print(f"  |R-P|     ", end="")
        for f in FILTERS:
            d = data.get((cpt_lbl, p, f))
            if d is None:
                print(" |   ----   ", end="")
                continue
            diff = abs(d['ve_mu'] - PAPER_T4_MEAN[(cpt_lbl, p)][f])
            tag = "*" if diff < 2.0 else " "
            if diff < 2.0:
                ve_close += 1
            ve_total += 1
            print(f" | {diff:5.2f}{tag}  ", end="")
        print()
        print()

    print(f"VE: {ve_close}/{ve_total} cells with |R-P| < 2.0")

    # ----------- Pareto *-mark check (refactor recompute vs paper) -----------
    print("\n" + "=" * 105)
    print("Pareto-optimal *-mark verification (using refactor T3 (mean,stdev) Pareto)")
    print("=" * 105)
    print(f"{'Cell':<13} | {'Paper *':>9} | {'Refactor T3 Pareto':>22} | {'Refactor T4 Pareto':>22} | match?")
    print("-" * 95)
    paper_match = 0
    paper_total = 0
    for (cpt_lbl, p), paper_star in PAPER_PARETO.items():
        if any(data.get((cpt_lbl, p, f)) is None for f in FILTERS):
            print(f"{cpt_lbl}/{p:<5.2f} | incomplete")
            continue
        t3_data = {f: (data[(cpt_lbl, p, f)]["pe_mu"], data[(cpt_lbl, p, f)]["pe_sd"])
                   for f in FILTERS}
        t4_data = {f: (data[(cpt_lbl, p, f)]["ve_mu"], data[(cpt_lbl, p, f)]["ve_sd"])
                   for f in FILTERS}
        t3_pareto = sorted(find_pareto(t3_data))
        t4_pareto = sorted(find_pareto(t4_data))
        in_t3 = paper_star in t3_pareto
        in_t4 = paper_star in t4_pareto
        match_str = ("T3+T4" if in_t3 and in_t4 else
                     "T3-only" if in_t3 else "T4-only" if in_t4 else "neither")
        if in_t3 and in_t4:
            paper_match += 1
        paper_total += 1
        t3_str = ",".join(f"{f:.2f}" for f in t3_pareto)
        t4_str = ",".join(f"{f:.2f}" for f in t4_pareto)
        print(f"{cpt_lbl}/{p:<5.2f} | {paper_star:>9.2f} | {t3_str:>22} | {t4_str:>22} | {match_str}")

    print(f"\nPaper *-mark in BOTH refactor T3 AND T4 Pareto: {paper_match}/{paper_total}")


if __name__ == "__main__":
    main()
