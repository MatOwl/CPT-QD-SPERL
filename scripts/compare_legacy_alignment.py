"""Compare legacy-aligned refactor runs against:
  (a) legacy native (HANDOFF: PE 5.60, VE 27.00, Opt 0.45, SW -67.46)
  (b) original refactor v3 (paper-aligned algo)
  (c) paper Tables 1-4

Run this AFTER ``scripts/reaggregate_paper_std.py runs/results_legacy_aligned``.
"""
from __future__ import annotations

import json
import os


CONFIGS = [
    ("v3 baseline (paper-aligned algo)",
     "runs/results_paper_tables_3_4_v3/filt0.90/barberis_sperl_p0.66_cpt_a0.88_r0.65_l2.25/aggregate.json"),
    ("legacy-aligned algo, paper hyperparams (eps=.3, batch=5, M=15k)",
     "runs/results_legacy_aligned/eps03_batch5/barberis_sperl_p0.66_cpt_a0.88_r0.65_l2.25/aggregate.json"),
    ("legacy-aligned algo + legacy hyperparams (eps=.6, batch=1, M=30k)",
     "runs/results_legacy_aligned/eps06_batch1_30k/barberis_sperl_p0.66_cpt_a0.88_r0.65_l2.25/aggregate.json"),
]
LEGACY = dict(opt_mu=0.452, opt_sd=2.626, dis_mu=5.60, dis_sd=1.50,
              ve_mu=27.00, sw_mu=-67.46, sw_sd=7.77)
PAPER = dict(opt_mu=2.74, opt_sd=0.95, dis_mu=2.89, dis_sd=0.13,
             ve_mu=9.30, ve_sd=0.95, sw_mu=-59.94, sw_sd=2.88)


def main():
    print("CPT88/p=0.66/filter=0.9 — refactor convergence to legacy via NE-fixes")
    print()
    print("%-65s | %-13s | %-15s | %-13s | %-13s" % (
        "Config", "Optimality", "SW", "PE (paper-formula)", "VE (paper-formula)"))
    print("-" * 130)

    rows = []
    for label, path in CONFIGS:
        if not os.path.exists(path):
            print(f"{label:<65}  MISSING ({path})")
            continue
        a = json.load(open(path))
        opt = a.get("optimality", {})
        sw = a.get("social_welfare", {})
        pe = a.get("policy_error_paper", a.get("policy_disagree_total", {}))
        ve = a.get("value_error_paper", a.get("value_error_total", {}))
        rows.append((label, opt, sw, pe, ve))
        print("%-65s | %5.2f +- %4.2f | %6.2f +- %4.2f | %5.2f +- %4.2f | %5.2f +- %4.2f" % (
            label,
            opt.get("mean", float("nan")), opt.get("std", float("nan")),
            sw.get("mean", float("nan")), sw.get("std", float("nan")),
            pe.get("mean", float("nan")), pe.get("std", float("nan")),
            ve.get("mean", float("nan")), ve.get("std", float("nan"))))

    print("-" * 130)
    print("%-65s | %5.2f +- %4.2f | %6.2f +- %4.2f | %5.2f +- %4.2f | %5.2f       " % (
        "Legacy native (HANDOFF row 14, OLD aggregator)",
        LEGACY["opt_mu"], LEGACY["opt_sd"], LEGACY["sw_mu"], LEGACY["sw_sd"],
        LEGACY["dis_mu"], LEGACY["dis_sd"], LEGACY["ve_mu"]))
    print("%-65s | %5.2f +- %4.2f | %6.2f +- %4.2f | %5.2f +- %4.2f | %5.2f +- %4.2f" % (
        "Paper Tables 1-4",
        PAPER["opt_mu"], PAPER["opt_sd"], PAPER["sw_mu"], PAPER["sw_sd"],
        PAPER["dis_mu"], PAPER["dis_sd"], PAPER["ve_mu"], PAPER["ve_sd"]))
    print()
    print("NOTE: legacy native PE/VE are OLD aggregator (per-seed sum then std-across-seeds).")
    print("Refactor's policy_disagree_total mean is comparable to legacy's dis_mu.")
    print("Refactor's value_error_total mean is comparable to legacy's ve_mu.")


if __name__ == "__main__":
    main()
