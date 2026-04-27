"""Per-state policy + value visualization for LNW abandonment env.

Reads per_state_values.csv files written by agents/run_paper_eval.py
(one per seed) for a given LNW run dir, and produces:

  - SPE oracle policy heatmap on (t, x_idx) grid
  - SPERL learned policy heatmap (cross-seed agreement frequency)
  - Disagreement breakdown by state
  - Per-state V_tilde vs V_hat scatter (with disagreement markers)

Output PNG files written to the run_dir.

Usage:
    python scripts/analyze_lnw.py <results_dir>

Example:
    python scripts/analyze_lnw.py \
        runs/abandonment_sperl_T5_x150_c11_d10_p0.72_cpt_a0.88_r0.65_l2.25
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_state(s: str) -> tuple[int, int]:
    """Parse '(t, x_idx)' string into (t, x_idx) tuple of ints."""
    m = re.match(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", s.strip())
    if not m:
        raise ValueError(f"cannot parse state string: {s!r}")
    return int(m.group(1)), int(m.group(2))


def load_seeds(run_dir: Path) -> list[pd.DataFrame]:
    """Load all seed CSVs into list of DataFrames with parsed (t, x_idx)."""
    dfs = []
    for seed_dir in sorted(run_dir.glob("seed*/")):
        csv = seed_dir / "per_state_values.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        states = df["state"].apply(parse_state)
        df["t"] = [s[0] for s in states]
        df["x_idx"] = [s[1] for s in states]
        df["seed"] = int(re.search(r"seed(\d+)", seed_dir.name).group(1))
        dfs.append(df)
    return dfs


def plot_policy_heatmap(run_dir: Path, dfs: list[pd.DataFrame], config: dict):
    """3-panel figure: SPE policy | SPERL policy (cross-seed mean) | Disagreement.

    Panel encoding:
      - SPE / SPERL: cell color = action probability (white=abandon, dark=continue);
        annotated with action symbol (A or C). For SPERL, also annotate cross-seed
        agreement % (e.g., "C 60%" = 60% of seeds chose continue).
      - Disagreement: cell color = number of seeds disagreeing with SPE.
    """
    if not dfs:
        print("No seed CSVs found in run dir")
        return

    T = int(max(df["t"].max() for df in dfs)) + 1
    x_min = int(min(df["x_idx"].min() for df in dfs))
    x_max = int(max(df["x_idx"].max() for df in dfs))

    n_seeds = len(dfs)
    # Build per-state {a_hat (SPE), n_continue (across seeds)}
    spe_action = {}
    sperl_n_continue = Counter()
    sperl_v_tilde_sum = Counter()
    sperl_n_states = Counter()
    spe_v = {}

    for df in dfs:
        for _, row in df.iterrows():
            key = (int(row["t"]), int(row["x_idx"]))
            spe_action[key] = int(row["a_hat"])
            spe_v[key] = float(row["v_hat"])
            if int(row["a_tilde"]) == 1:
                sperl_n_continue[key] += 1
            sperl_v_tilde_sum[key] += float(row["v_tilde"])
            sperl_n_states[key] += 1

    # Build matrices for plotting (t on x-axis, x_idx on y-axis, NaN for unreachable)
    def make_matrix(value_fn):
        mat = np.full((x_max - x_min + 1, T), np.nan)
        for (t, k), val in spe_action.items():
            row = (x_max - x_min) - (k - x_min)  # flip so high x at top
            col = t
            mat[row, col] = value_fn((t, k))
        return mat

    spe_mat = make_matrix(lambda key: float(spe_action[key]))
    sperl_freq_mat = make_matrix(lambda key: sperl_n_continue[key] / max(sperl_n_states[key], 1))
    disagree_mat = make_matrix(lambda key: sum(
        1 for df in dfs
        for _, row in df.iterrows()
        if (int(row["t"]), int(row["x_idx"])) == key
        and int(row["a_tilde"]) != int(row["a_hat"])
    ))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    extent = [-0.5, T - 0.5, x_min - 0.5, x_max + 0.5]
    cmap_action = "Greys"  # white = 0 (abandon), black = 1 (continue)
    cmap_disagree = "Reds"

    # --- Panel 1: SPE oracle policy ---
    ax = axes[0]
    im1 = ax.imshow(spe_mat, cmap=cmap_action, vmin=0, vmax=1,
                    extent=extent, aspect="auto", origin="upper")
    ax.set_title(f"SPE oracle policy\n(white=abandon, dark=continue)")
    ax.set_xlabel("Stage t")
    ax.set_ylabel("x_idx (project value offset)")
    # Annotate cells with A/C
    for (t, k), action in spe_action.items():
        sym = "C" if action == 1 else "A"
        col = "white" if action == 1 else "black"
        ax.text(t, k, sym, ha="center", va="center", color=col, fontsize=10, fontweight="bold")
    # SPE V annotations in top-right
    ax.text(0.02, 0.97, f"SPE Welfare = {sum(spe_v.values()):.1f}",
            transform=ax.transAxes, fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    # --- Panel 2: SPERL learned policy (cross-seed continue frequency) ---
    ax = axes[1]
    im2 = ax.imshow(sperl_freq_mat, cmap=cmap_action, vmin=0, vmax=1,
                    extent=extent, aspect="auto", origin="upper")
    ax.set_title(f"SPERL learned policy (mean over {n_seeds} seeds)\n"
                 "shade = % of seeds choosing continue")
    ax.set_xlabel("Stage t")
    ax.set_ylabel("x_idx")
    for (t, k), n in sperl_n_continue.items():
        total = sperl_n_states[(t, k)]
        frac = n / total
        sym = f"{int(round(frac*100))}%"
        col = "white" if frac > 0.5 else "black"
        ax.text(t, k, sym, ha="center", va="center", color=col, fontsize=8)
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # --- Panel 3: Disagreement count ---
    ax = axes[2]
    max_disagree = n_seeds
    im3 = ax.imshow(disagree_mat, cmap=cmap_disagree, vmin=0, vmax=max_disagree,
                    extent=extent, aspect="auto", origin="upper")
    ax.set_title(f"Disagreement count (max {n_seeds})\n"
                 "# seeds where SPERL action != SPE")
    ax.set_xlabel("Stage t")
    ax.set_ylabel("x_idx")
    for (t, k), action in spe_action.items():
        n_dis = int(disagree_mat[(x_max - x_min) - (k - x_min), t])
        if n_dis > 0:
            ax.text(t, k, str(n_dis), ha="center", va="center",
                    color="black" if n_dis < n_seeds * 0.6 else "white",
                    fontsize=10, fontweight="bold")
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

    suptitle = (f"LNW abandonment env: SPE vs SPERL policy structure\n"
                f"T={config.get('horizon', '?')}, p_win={config.get('p_win', '?')}, "
                f"x_1={config.get('x1', '?')}, "
                f"CPT(α={config.get('alpha', '?')}, λ={config.get('lmbd', '?')})")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()

    out_path = run_dir / "policy_heatmap.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  wrote {out_path}")
    plt.close(fig)


def plot_v_scatter(run_dir: Path, dfs: list[pd.DataFrame], config: dict):
    """Scatter: V_hat (SPE) vs V_tilde (SPERL), one point per (state, seed).

    Disagreement states colored separately.
    """
    if not dfs:
        return

    n_seeds = len(dfs)
    fig, ax = plt.subplots(figsize=(6, 6))

    all_pts_agree = []
    all_pts_disagree = []
    for df in dfs:
        for _, row in df.iterrows():
            pt = (float(row["v_hat"]), float(row["v_tilde"]))
            if int(row["a_tilde"]) == int(row["a_hat"]):
                all_pts_agree.append(pt)
            else:
                all_pts_disagree.append(pt)

    if all_pts_agree:
        xs, ys = zip(*all_pts_agree)
        ax.scatter(xs, ys, s=20, alpha=0.5, label=f"agree ({len(all_pts_agree)})", c="C0")
    if all_pts_disagree:
        xs, ys = zip(*all_pts_disagree)
        ax.scatter(xs, ys, s=40, alpha=0.7, label=f"disagree ({len(all_pts_disagree)})",
                   c="C3", marker="x")

    # 45-degree line
    all_v = [v for df in dfs for v in df["v_hat"].tolist() + df["v_tilde"].tolist()]
    if all_v:
        lo, hi = min(all_v), max(all_v)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="V_tilde = V_hat")

    ax.set_xlabel("V_hat (SPE oracle)")
    ax.set_ylabel("V_tilde (SPERL learned)")
    ax.set_title(f"LNW V-error scatter (per state × {n_seeds} seeds)\n"
                 f"T={config.get('horizon', '?')}, p={config.get('p_win', '?')}, "
                 f"CPT(α={config.get('alpha', '?')}, λ={config.get('lmbd', '?')})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = run_dir / "v_scatter.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  wrote {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path,
                   help="Path to runs/abandonment_sperl_T*_*/ directory")
    args = p.parse_args()

    if not args.run_dir.is_dir():
        print(f"Not a directory: {args.run_dir}")
        sys.exit(1)

    config = {}
    config_path = args.run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())

    print(f"Loading seeds from {args.run_dir}...")
    dfs = load_seeds(args.run_dir)
    print(f"  loaded {len(dfs)} seeds")
    if not dfs:
        sys.exit(1)
    print(f"  {len(dfs[0])} states per seed")

    print("Plotting policy heatmap...")
    plot_policy_heatmap(args.run_dir, dfs, config)

    print("Plotting V scatter...")
    plot_v_scatter(args.run_dir, dfs, config)


if __name__ == "__main__":
    main()
