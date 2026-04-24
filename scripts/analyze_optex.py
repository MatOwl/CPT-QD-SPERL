"""Per-state analysis of SPERL vs SPE for OptEx.

Reads per_state_values.csv files written by agents/run_paper_eval.py
(one per seed) for a given OptEx run, and produces:

- Agreement breakdown by time-step t and by state depth
- Distribution of |a_tilde - a_hat| for disagreement states
- Distribution of |v_tilde - v_hat| at disagreements
- Per-seed vs cross-seed argmax stability (how often do 3 seeds pick the
  same a_tilde, even if it differs from a_hat?)
- (t x log-return-bin) heatmap of disagreement count

Usage:
    python scripts/analyze_optex.py <results_dir>

where <results_dir> points at e.g.
  results_optex/optex_sperl_sig0.015_numw4_cpt_a0.95_r0.5_l1.5
"""
from __future__ import annotations

import sys
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
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"not found: {run_dir}")
        sys.exit(1)

    dfs = load_seeds(run_dir)
    if not dfs:
        print(f"no per_state_values.csv files under {run_dir}")
        sys.exit(1)

    summarize(dfs)
    plot_heatmap(dfs, run_dir / "disagree_heatmap.png")
