"""Per-state analysis of SPERL vs SPE for OptEx.

Reads per_state_values.csv files written by agents/run_paper_eval.py
(one per seed) for a given OptEx run, and produces:

- Agreement breakdown by time-step t and by state depth
- Distribution of |a_tilde - a_hat| for disagreement states
- Distribution of |v_tilde - v_hat| at disagreements
- Per-seed vs cross-seed argmax stability (how often do 3 seeds pick the
  same a_tilde, even if it differs from a_hat?)
- (t x log-return-bin) heatmap of disagreement count
- Visit-frequency breakdown: % of disagreements that fall on states the
  SPE policy actually visits (vs off-equilibrium-path noise)

Usage:
    python scripts/analyze_optex.py <results_dir> [--n-traj 2000]

where <results_dir> points at e.g.
  results_optex/optex_sperl_sig0.015_numw4_cpt_a0.95_r0.5_l1.5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_state(s: str) -> tuple[int, int, int]:
    """Parse a state string like '(0, 10, 10)' into (log_bin, remain, X)."""
    m = re.match(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", s.strip())
    if not m:
        raise ValueError(f"cannot parse {s!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_seeds(run_dir: Path) -> list[pd.DataFrame]:
    dfs = []
    for seed_dir in sorted(run_dir.glob("seed*/")):
        csv = seed_dir / "per_state_values.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        parsed = df["state"].apply(parse_state)
        df["log_bin"] = [p[0] for p in parsed]
        df["remain"] = [p[1] for p in parsed]
        df["X"] = [p[2] for p in parsed]
        # OptEx default: horizon=5, n_bins_N=10 → remain_bin in [0, 10], step = 2 per t
        # t = (max_remain - remain) / 2  where max_remain = horizon * 2 = 10
        max_remain = int(df["remain"].max())  # assume initial state has max
        df["t"] = (max_remain - df["remain"]) // 2
        df["adiff"] = (df["a_tilde"] - df["a_hat"]).abs()
        df["vdiff"] = (df["v_tilde"] - df["v_hat"]).abs()
        df["disagree"] = (df["a_tilde"] != df["a_hat"]).astype(int)
        df["seed"] = int(seed_dir.name.replace("seed", ""))
        dfs.append(df)
    return dfs


def summarize(dfs: list[pd.DataFrame]) -> None:
    print(f"\n=== Seeds loaded: {len(dfs)} ===")
    for df in dfs:
        n_dis = df["disagree"].sum()
        print(
            f"  seed {df.seed.iloc[0]}: {n_dis}/{len(df)} disagree "
            f"  mean|dv|={df.vdiff.mean():.4f}  mean|da|={df.adiff.mean():.2f}"
        )

    combined = pd.concat(dfs, ignore_index=True)

    print("\n=== Disagreement by t (summed across seeds) ===")
    g = combined.groupby("t").agg(
        n_states=("state", "count"),
        n_disagree=("disagree", "sum"),
    )
    g["rate"] = g["n_disagree"] / g["n_states"]
    print(g.to_string())

    print("\n=== |a_tilde - a_hat| distribution (disagreements only) ===")
    print(combined[combined.disagree == 1].adiff.value_counts().sort_index().to_string())

    print("\n=== |v_tilde - v_hat| at disagreements ===")
    vd = combined[combined.disagree == 1].vdiff
    print(f"  mean={vd.mean():.4f}  median={vd.median():.4f}")
    print(f"  90p ={vd.quantile(0.9):.4f}  max   ={vd.max():.4f}")

    # Cross-seed argmax stability
    if len(dfs) >= 2:
        print("\n=== Cross-seed stability of a_tilde ===")
        wide = dfs[0][["state", "a_hat"]].copy()
        for df in dfs:
            wide[f"a{df.seed.iloc[0]}"] = df["a_tilde"].values
        seed_cols = [c for c in wide.columns if c.startswith("a") and c != "a_hat"]
        wide["all_equal"] = wide[seed_cols].nunique(axis=1) == 1
        wide["matches_hat"] = wide.apply(
            lambda row: all(row[c] == row["a_hat"] for c in seed_cols), axis=1,
        )
        print(f"  states where all {len(seed_cols)} seeds pick same a_tilde: "
              f"{wide.all_equal.sum()}/{len(wide)} "
              f"({100*wide.all_equal.mean():.1f}%)")
        print(f"  of those, matches SPE a_hat: "
              f"{(wide.all_equal & wide.matches_hat).sum()}")
        print(f"  states where seeds disagree among themselves: "
              f"{(~wide.all_equal).sum()}")


def visit_breakdown(dfs: list[pd.DataFrame], run_dir: Path, n_traj: int) -> None:
    """Cross-tab disagreement category vs visit count under SPE policy.

    Answers: of the ~half of states where SPERL disagrees with SPE, how many
    are actually on the SPE-visited equilibrium path? If most are off-path,
    the headline disagreement rate overstates the meaningful divergence.
    """
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        print("\n[visit-breakdown] no config.json — skipping")
        return
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("env") != "optex":
        return
    spe_file = cfg.get("spe_file")
    if not spe_file or not Path(spe_file).exists():
        print(f"\n[visit-breakdown] spe_file {spe_file!r} missing — skipping")
        return

    # Build env + oracle (must be on sys.path; assumes project root cwd)
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from lib.envs.optimal_execution import OptimalExecution
    from lib.envs.optex_spe import SPEOracle

    env = OptimalExecution(
        horizon=cfg["horizon"], sigma=cfg["sigma"], num_w=cfg["num_w"],
    )
    oracle = SPEOracle(env, spe_file, initial_states=[env.reset()])
    spe_pi = oracle.policy()

    # Roll out under SPE policy and count (t, state) visits
    rng = np.random.RandomState(0)
    counts: Counter = Counter()
    for _ in range(n_traj):
        np.random.seed(int(rng.randint(0, 2**31 - 1)))
        s = env.reset()
        t = 0
        done = False
        while not done:
            counts[(t, (int(s[0]), int(s[1]), int(s[2])))] += 1
            s, _, done, _ = env.step(int(spe_pi(t, s)))
            t += 1
    n_visited = len(counts)
    total_step_visits = sum(counts.values())

    # Map BFS node_id -> (t, state)
    nd = oracle.node_df.copy()
    nd["s"] = list(zip(nd["s0"].astype(int), nd["s1"].astype(int), nd["s2"].astype(int)))
    nd["t"] = nd["t"].astype(int)
    if len(nd) != len(dfs[0]):
        print(f"\n[visit-breakdown] BFS nodes ({len(nd)}) != per_state rows "
              f"({len(dfs[0])}) — env config may not match the run; skipping")
        return

    # Per BFS node: stable / mismatch classification across seeds
    seed_concat = []
    for df in dfs:
        d = df[["a_tilde", "a_hat"]].copy().reset_index().rename(columns={"index": "node_id"})
        d["seed"] = int(df["seed"].iloc[0])
        seed_concat.append(d)
    full = pd.concat(seed_concat, ignore_index=True)
    grouped = full.groupby("node_id").agg(
        a_tildes=("a_tilde", lambda x: tuple(sorted(set(x)))),
        a_hat=("a_hat", "first"),
    ).reset_index()
    grouped["t"] = grouped["node_id"].map(nd["t"])
    grouped["s"] = grouped["node_id"].map(nd["s"])
    grouped["stable"] = grouped["a_tildes"].apply(lambda x: len(x) == 1)

    def cat(r):
        if not r["stable"]:
            return "unstable"
        return "stable_match" if r["a_tildes"][0] == r["a_hat"] else "stable_mismatch"
    grouped["cat"] = grouped.apply(cat, axis=1)
    grouped["visits"] = grouped.apply(
        lambda r: counts.get((int(r["t"]), r["s"]), 0), axis=1
    )

    print(f"\n=== Visit-frequency breakdown (SPE policy, {n_traj} trajectories) ===")
    print(f"  BFS reachable     : {len(nd)} (t, state) pairs")
    print(f"  Visited under SPE : {n_visited} pairs ({total_step_visits} step-visits)")
    print(f"\n  {'category':<18} {'n_states':>10} {'visits=0':>10} "
          f"{'visits>=5':>10} {'mean_v':>10}")
    on_path_states = 0
    on_path_mismatch = 0
    for c in ["stable_match", "stable_mismatch", "unstable"]:
        sub = grouped[grouped["cat"] == c]
        n0 = (sub["visits"] == 0).sum()
        nhi = (sub["visits"] >= 5).sum()
        print(f"  {c:<18} {len(sub):>10} {n0:>10} {nhi:>10} {sub['visits'].mean():>10.2f}")
        on_path_states += (sub["visits"] > 0).sum()
        if c == "stable_mismatch":
            on_path_mismatch = (sub["visits"] > 0).sum()
    n_unstable_visited = (grouped[(grouped["cat"] == "unstable") & (grouped["visits"] > 0)]).shape[0]

    # Headline: on-path disagreement count
    raw_disagree = ((grouped["cat"] == "stable_mismatch") |
                    (grouped["cat"] == "unstable")).sum()
    on_path_disagree = on_path_mismatch + n_unstable_visited
    print(f"\n  raw disagree (stable_mismatch + unstable) : {raw_disagree}/{len(grouped)} "
          f"({100*raw_disagree/len(grouped):.1f}%)")
    print(f"  on-path disagree (visits>0)               : {on_path_disagree}/{n_visited} "
          f"({100*on_path_disagree/max(n_visited,1):.1f}%)")
    print(f"    of which unstable across seeds          : {n_unstable_visited}")
    print(f"    of which stable but mismatch SPE        : {on_path_mismatch}")

    out = run_dir / "visit_freq_breakdown.csv"
    grouped[["node_id", "t", "s", "cat", "a_tildes", "a_hat", "visits"]].to_csv(out, index=False)
    print(f"\n[io] visit breakdown -> {out}")


def plot_heatmap(dfs: list[pd.DataFrame], out_path: Path) -> None:
    """Disagreement-count heatmap in (t, log_bin) space, averaged over seeds."""
    combined = pd.concat(dfs, ignore_index=True)
    pivot = combined.groupby(["t", "log_bin"]).disagree.mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis",
                   extent=[pivot.columns.min(), pivot.columns.max(),
                           pivot.index.max() + 0.5, pivot.index.min() - 0.5])
    ax.set_xlabel("log-return bin")
    ax.set_ylabel("time t")
    ax.set_title(f"Per-state P(disagree) over {len(dfs)} seeds\n"
                 f"({out_path.parent.name})")
    plt.colorbar(im, ax=ax, label="P(a_tilde != a_hat)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n[plot] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--n-traj", type=int, default=2000,
                    help="trajectories under SPE for visit-frequency breakdown")
    ap.add_argument("--no-visit", action="store_true",
                    help="skip visit-frequency breakdown (saves ~10s)")
    args = ap.parse_args()

    if not args.run_dir.exists():
        print(f"not found: {args.run_dir}")
        sys.exit(1)

    dfs = load_seeds(args.run_dir)
    if not dfs:
        print(f"no per_state_values.csv files under {args.run_dir}")
        sys.exit(1)

    summarize(dfs)
    plot_heatmap(dfs, args.run_dir / "disagree_heatmap.png")
    if not args.no_visit:
        visit_breakdown(dfs, args.run_dir, n_traj=args.n_traj)
