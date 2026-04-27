"""LNW SPERL convergence curve generator.

Trains GreedySPERL on LNW with frequent eval and plots CPT(x_0) trajectories
across seeds. Output: runs/<run>/convergence.png + convergence.npz.

Plots two configs side-by-side:
  (a) Headline: T=5 p=0.72 CPT88 (SPE chooses continue at (0,0), V≈5)
  (b) Stay-out: T=5 p=0.6  CPT88 (SPE chooses abandon at (0,0), V=0)

Usage:
    python scripts/lnw_convergence.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

from lib.cpt import CPTParams
from lib.envs.registry import make_env, make_featurizer
from agents.sperl_qr_generic import GreedySPERL


def train_and_get_curves(env_kwargs, cpt: CPTParams, n_seeds: int,
                        train_eps: int, eval_freq: int, n_eval_eps: int = 50):
    """Train n_seeds agents and return matrix of cpt_rewards trajectories.

    Returns:
        curves: np.ndarray of shape (n_seeds, n_eval_points)
        x: np.ndarray of episode indices for each eval point
    """
    curves = []
    for seed in range(n_seeds):
        print(f"  seed {seed}...", end=" ", flush=True)
        env = make_env("abandonment", **env_kwargs)
        feat = make_featurizer("abandonment", env)
        agent = GreedySPERL(
            env, feat, cpt,
            support_size=50,
            critic_lr=0.04,
            exploration={"type": "eps-greedy", "params": [0.3]},
            target_type="TD",
            order="bwd",
            seed=seed,
        )
        agent.learn(
            n_train_eps=train_eps,
            n_batch=1,
            n_eval_eps=n_eval_eps,
            eval_freq=eval_freq,
            verbose=0,
        )
        curves.append(np.asarray(agent.stats["cpt_rewards"]))
        print(f"final cpt={curves[-1][-1]:.3f}")

    # Pad to max length (shouldn't differ but just in case)
    max_len = max(len(c) for c in curves)
    arr = np.full((n_seeds, max_len), np.nan)
    for i, c in enumerate(curves):
        arr[i, :len(c)] = c

    n_pts = arr.shape[1]
    x = np.arange(n_pts) * eval_freq
    return arr, x


def plot_curves(ax, curves, x, title: str, color="C0", spe_v: float = None):
    """Plot mean ± 1σ band of curves vs x (episode index)."""
    mean = np.nanmean(curves, axis=0)
    std = np.nanstd(curves, axis=0)
    ax.plot(x, mean, color=color, lw=2, label=f"SPERL CPT(x_0) mean (n={curves.shape[0]} seeds)")
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2,
                    label="±1σ across seeds")
    if spe_v is not None:
        ax.axhline(spe_v, color="k", ls="--", lw=1, alpha=0.7,
                   label=f"SPE oracle V(x_0) = {spe_v:.2f}")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("CPT(x_0)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)


def main():
    out_dir = Path("runs/_lnw_convergence")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_eps = 8000
    eval_freq = 200
    n_seeds = 5

    cpt88 = CPTParams(0.88, 0.65, 0.65, 2.25)
    env_T5 = dict(p=0.72, delta=10, c=11, T=5, x1=50)
    env_T5_p06 = dict(p=0.6, delta=10, c=11, T=5, x1=50)

    print(f"=== Config 1: T=5 p=0.72 CPT88 (headline, SPE V_x0 ~ 5.0) ===")
    curves_a, x_a = train_and_get_curves(env_T5, cpt88, n_seeds, train_eps, eval_freq)
    print(f"=== Config 2: T=5 p=0.6  CPT88 (stay-out, SPE V_x0 = 0) ===")
    curves_b, x_b = train_and_get_curves(env_T5_p06, cpt88, n_seeds, train_eps, eval_freq)

    np.savez(out_dir / "convergence.npz",
             curves_p072=curves_a, x_p072=x_a,
             curves_p06=curves_b, x_p06=x_b,
             train_eps=train_eps, eval_freq=eval_freq, n_seeds=n_seeds)
    print(f"Saved {out_dir / 'convergence.npz'}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

    plot_curves(axes[0], curves_a, x_a,
                title="LNW T=5, p=0.72, CPT88 — SPE chooses continue at start",
                color="C0", spe_v=5.0)
    plot_curves(axes[1], curves_b, x_b,
                title="LNW T=5, p=0.6, CPT88 — CPT-induced stay-out",
                color="C3", spe_v=0.0)

    fig.suptitle("SPERL convergence on LNW abandonment env\n"
                 f"({n_seeds} seeds × {train_eps} episodes, eval every {eval_freq})",
                 fontsize=12)
    fig.tight_layout()

    out_path = out_dir / "convergence.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
