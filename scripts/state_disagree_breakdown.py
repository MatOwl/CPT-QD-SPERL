"""Aggregate per-state policy disagreement and value-error across seeds.

Reads per_state_values.csv from a paper-eval results directory and reports
which states persistently disagree across seeds plus their contribution to
the cumulative value error. Helps localize where the SPERL learner is
struggling vs the SPE oracle.

Usage:
  python scripts/state_disagree_breakdown.py <results_dir> [--top N]

Example:
  python scripts/state_disagree_breakdown.py runs/results_p066_10s_both_acceptInf
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    p.add_argument("--top", type=int, default=15,
                   help="show top-N states by mean |v_tilde - v_hat|")
    args = p.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"not a directory: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    run_dirs = [
        d for d in os.listdir(args.results_dir)
        if os.path.isdir(os.path.join(args.results_dir, d))
    ]
    if not run_dirs:
        print(f"no run subdirs in {args.results_dir}", file=sys.stderr)
        sys.exit(1)
    run_root = os.path.join(args.results_dir, run_dirs[0])

    seed_dirs = sorted(d for d in os.listdir(run_root) if d.startswith("seed"))
    if not seed_dirs:
        print(f"no seed* subdirs in {run_root}", file=sys.stderr)
        sys.exit(1)

    print(f"reading {len(seed_dirs)} seeds from {run_root}")

    frames = []
    for sd in seed_dirs:
        path = os.path.join(run_root, sd, "per_state_values.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["seed"] = int(sd.replace("seed", ""))
        frames.append(df)

    big = pd.concat(frames, ignore_index=True)
    big["disagree"] = (big["a_tilde"] != big["a_hat"]).astype(int)
    big["abs_v_err"] = (big["v_tilde"] - big["v_hat"]).abs()

    g = big.groupby("state").agg(
        a_hat=("a_hat", "first"),
        n_disagree=("disagree", "sum"),
        n_seeds=("disagree", "size"),
        mean_abs_v_err=("abs_v_err", "mean"),
        mean_v_tilde=("v_tilde", "mean"),
        mean_v_hat=("v_hat", "mean"),
    )
    g["disagree_rate"] = g["n_disagree"] / g["n_seeds"]
    g = g.sort_values("mean_abs_v_err", ascending=False)

    print(f"\nTop-{args.top} states by mean |v_tilde - v_hat|:")
    print(g.head(args.top).to_string())

    total_ve = g["mean_abs_v_err"].sum()
    print(f"\nSum of mean per-state |Δv| = {total_ve:.2f}")
    print(f"Total disagree rate (state-avg): {g['disagree_rate'].mean():.2%}")
    print(f"Persistent disagreers (rate ≥ 0.5): "
          f"{(g['disagree_rate'] >= 0.5).sum()}/{len(g)}")


if __name__ == "__main__":
    main()
