"""Test whether OptEx policy disagreement concentrates on low-visit states.

Hypothesis: the 184 stable-but-mismatch states (cross-seed-stable SPERL choice
that disagrees with paper SPE oracle) might mostly be off the SPE-visited
sub-tree. If so, the 49% disagreement is largely off-equilibrium-path noise
rather than economically meaningful divergence.

Method:
  1. Build OptEx env + load SPE oracle (same as paper_eval).
  2. Roll out N=2000 trajectories from x_0 under SPE policy, count visits.
  3. Cross-tab visit count vs disagreement category from sticky+filter run.

Usage: PYTHONIOENCODING=utf-8 python scripts/visit_freq_check.py
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from lib.envs.optimal_execution import OptimalExecution
from lib.envs.optex_spe import SPEOracle


def state_key(s):
    return (int(s[0]), int(s[1]), int(s[2]))


def rollout_visits(env, oracle_pi, n_traj=2000, seed=0):
    rng = np.random.RandomState(seed)
    counts = Counter()
    for _ in range(n_traj):
        np.random.seed(int(rng.randint(0, 2**31 - 1)))
        s = env.reset()
        t = 0
        done = False
        while not done:
            counts[(t, state_key(s))] += 1
            a = oracle_pi(t, s)
            s, _, done, _ = env.step(int(a))
            t += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.015)
    ap.add_argument("--num-w", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--spe-file", default="CumSPERL_ref/SPE_OptEx_5_0.015_4.npy")
    ap.add_argument("--results-dir",
                    default="runs/results_alg34_optex/optex_sperl_sig0.015_numw4_cpt_a0.95_r0.5_l1.5",
                    help="dir containing seedN/per_state_values.csv (sticky+filter run)")
    ap.add_argument("--n-traj", type=int, default=2000)
    args = ap.parse_args()

    env = OptimalExecution(horizon=args.horizon, sigma=args.sigma, num_w=args.num_w)
    oracle = SPEOracle(env, args.spe_file, initial_states=[env.reset()])
    spe_pi = oracle.policy()
    print(f"[env] OptEx tree size = {len(oracle.node_df)}")

    counts = rollout_visits(env, spe_pi, n_traj=args.n_traj)
    print(f"[rollout] {args.n_traj} trajectories under SPE policy → "
          f"{len(counts)} distinct (t, state) visited")

    # Each row in per_state_values.csv corresponds 1:1 with a BFS node in
    # oracle.node_df (same iteration order — both call OptExTree on same env).
    # We use the row index as the join key so (t, state) is unambiguous.
    nd = oracle.node_df.copy()
    nd["s"] = list(zip(nd["s0"].astype(int), nd["s1"].astype(int), nd["s2"].astype(int)))
    nd["t"] = nd["t"].astype(int)

    # Aggregate stable-vs-disagree per BFS node (= row index)
    seed_rows = []
    for seed in range(3):
        df = pd.read_csv(Path(args.results_dir) / f"seed{seed}" / "per_state_values.csv")
        df = df.reset_index().rename(columns={"index": "node_id"})
        df["seed"] = seed
        seed_rows.append(df)
    full = pd.concat(seed_rows, ignore_index=True)

    if len(full) // 3 != len(nd):
        print(f"[warn] per_state rows/3 = {len(full)//3} != BFS nodes = {len(nd)}; "
              "row-index join may be off")

    grouped = full.groupby("node_id").agg(
        a_tildes=("a_tilde", lambda x: tuple(sorted(set(x)))),
        a_hat=("a_hat", "first"),
        state=("state", "first"),
    ).reset_index()

    # Attach (t, state) from BFS for unambiguous visit lookup
    grouped["t"] = grouped["node_id"].map(nd["t"])
    grouped["s"] = grouped["node_id"].map(nd["s"])
    grouped["stable"] = grouped["a_tildes"].apply(lambda t: len(t) == 1)
    grouped["match"] = grouped.apply(
        lambda r: r["stable"] and r["a_tildes"][0] == r["a_hat"], axis=1
    )

    def classify(r):
        if not r["stable"]:
            return "unstable"
        return "stable_match" if r["match"] else "stable_mismatch"

    grouped["cat"] = grouped.apply(classify, axis=1)
    grouped["visits"] = grouped.apply(
        lambda r: counts.get((int(r["t"]), r["s"]), 0), axis=1
    )

    # === Headline cross-tab ===
    print("\n=== Visit-count distribution by category ===")
    print(f"{'cat':<18} {'n_states':>10} {'visits=0':>10} {'visits<5':>10} "
          f"{'visits>=5':>10} {'mean_v':>10} {'max_v':>10}")
    for cat in ["stable_match", "stable_mismatch", "unstable"]:
        sub = grouped[grouped["cat"] == cat]
        n = len(sub)
        n0 = (sub["visits"] == 0).sum()
        n_lo = ((sub["visits"] > 0) & (sub["visits"] < 5)).sum()
        n_hi = (sub["visits"] >= 5).sum()
        print(f"{cat:<18} {n:>10} {n0:>10} {n_lo:>10} {n_hi:>10} "
              f"{sub['visits'].mean():>10.2f} {sub['visits'].max():>10}")

    print(f"\ntotal trajectories rolled = {args.n_traj}")
    print(f"max possible visits per state = {args.n_traj}")

    # === Visit-weighted disagreement ===
    print("\n=== Visit-weighted disagreement ===")
    total_visits = grouped["visits"].sum()
    mismatch_visits = grouped[grouped["cat"] == "stable_mismatch"]["visits"].sum()
    unstable_visits = grouped[grouped["cat"] == "unstable"]["visits"].sum()
    print(f"  raw count:        stable_mismatch = {(grouped['cat']=='stable_mismatch').sum()} / {len(grouped)}")
    print(f"  visit-weighted:   stable_mismatch visits / total = "
          f"{mismatch_visits}/{total_visits} = {mismatch_visits/max(total_visits,1):.4f}")
    print(f"  visit-weighted:   unstable visits / total = "
          f"{unstable_visits}/{total_visits} = {unstable_visits/max(total_visits,1):.4f}")

    # === Top mismatch states sorted by visits ===
    print("\n=== Top-10 mismatch states by visit count ===")
    sub = grouped[grouped["cat"] == "stable_mismatch"].sort_values(
        "visits", ascending=False
    ).head(10)
    print(sub[["state", "a_tildes", "a_hat", "visits"]].to_string(index=False))

    # save full table
    out = Path(args.results_dir) / "visit_freq_breakdown.csv"
    grouped[["state", "cat", "a_tildes", "a_hat", "visits"]].to_csv(out, index=False)
    print(f"\n[io] full breakdown -> {out}")


if __name__ == "__main__":
    main()
