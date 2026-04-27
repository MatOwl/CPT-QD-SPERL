"""Lightweight persistence for runs of the generic framework.

Directory layout (default root: ``runs/``, set via ``--results-dir``):
    runs/{run_name}/
        config.json                # all hyperparameters, for reproducibility
        seed{i}/
            stats.npz              # training curves (mean/std/cpt_rewards)
            qtile_theta.npy        # critic params (SPERL) — optional
            greedy_action.npy      # SPERL learned policy
            theta.npy              # SPSA learned policy params — optional
            per_state_values.csv   # x, a_tilde, v_tilde, a_hat, v_hat
            metrics.json           # 4 paper metrics
        aggregate.json             # (mean, std) across seeds
        aggregate.csv              # one row per metric
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def run_name_from_args(args, env):
    """Build a deterministic results-dir name from common CLI args."""
    parts = [env, args.algo if hasattr(args, "algo") else "sperl"]
    if env == "barberis":
        parts.append(f"p{args.p_win}")
    elif env == "optex":
        parts.append(f"sig{args.sigma}_numw{args.num_w}")
    elif env == "abandonment":
        parts.append(
            f"T{args.horizon}_x1{args.x1}_c{args.c}_d{args.delta}_p{args.p_win}"
        )
    elif env == "bln":
        parts.append(
            f"T{args.horizon}_nW{args.n_W}_nR{args.n_R}"
            f"_g{args.gamma}_dc{args.delta_c}"
        )
    parts.append(f"cpt_a{args.alpha}_r{args.rho1}_l{args.lmbd}")
    return "_".join(str(p) for p in parts)


def save_config(root, config: dict):
    _ensure_dir(root)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=_json_default)


def save_seed_result(root, seed: int, agent, metrics: dict):
    """Persist everything from one training+eval run.

    Works for both ``GreedySPERL`` and ``SPSAAgent`` (the presence of
    ``agent.critic`` / ``agent.policy.theta`` drives what gets saved).
    """
    seed_dir = _ensure_dir(os.path.join(root, f"seed{seed}"))

    # learning curves
    np.savez(
        os.path.join(seed_dir, "stats.npz"),
        mean_rewards=np.asarray(agent.stats.get("mean_rewards", [])),
        std_rewards=np.asarray(agent.stats.get("std_rewards", [])),
        cpt_rewards=np.asarray(agent.stats.get("cpt_rewards", [])),
    )

    # policy + critic params
    if hasattr(agent, "critic") and hasattr(agent.critic, "qtile_theta"):
        np.save(os.path.join(seed_dir, "qtile_theta.npy"),
                agent.critic.qtile_theta)
    if hasattr(agent, "policy"):
        if hasattr(agent.policy, "greedy_action"):
            np.save(os.path.join(seed_dir, "greedy_action.npy"),
                    agent.policy.greedy_action)
        if hasattr(agent.policy, "theta"):
            np.save(os.path.join(seed_dir, "theta.npy"), agent.policy.theta)

    # per-state value table
    per_state = metrics.get("per_state", [])
    if per_state:
        with open(os.path.join(seed_dir, "per_state_values.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["state", "a_tilde", "a_hat", "v_tilde", "v_hat"])
            for row in per_state:
                w.writerow([str(row["state"]), row["a_tilde"],
                            row["a_hat"], row["v_tilde"], row["v_hat"]])

    # scalar metrics
    scalar = {k: v for k, v in metrics.items() if k != "per_state"}
    with open(os.path.join(seed_dir, "metrics.json"), "w") as f:
        json.dump(scalar, f, indent=2, default=_json_default)

    return seed_dir


def save_aggregate(root, all_metrics: list[dict]):
    """Aggregate scalar metrics across seeds; write JSON + CSV."""
    keys = [
        "policy_disagree_total",
        "value_error_total",
        "optimality",
        "social_welfare",
        "social_welfare_spe",
    ]
    agg = {}
    for k in keys:
        vals = [float(m[k]) for m in all_metrics
                if m.get(k) is not None]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)),
                      "std": float(np.std(vals)),
                      "n": len(vals)}

    with open(os.path.join(root, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)

    with open(os.path.join(root, "aggregate.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std", "n"])
        for k, v in agg.items():
            w.writerow([k, v["mean"], v["std"], v["n"]])

    return agg


def _json_default(o):
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def load_seed(seed_dir):
    """Load a previously saved seed's stats + metrics (for re-plotting etc.)."""
    out = {}
    stats_path = os.path.join(seed_dir, "stats.npz")
    if os.path.exists(stats_path):
        out["stats"] = dict(np.load(stats_path))
    metrics_path = os.path.join(seed_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            out["metrics"] = json.load(f)
    return out
