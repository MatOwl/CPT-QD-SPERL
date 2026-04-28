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
    """Aggregate scalar metrics across seeds; write JSON + CSV.

    Records two flavors of Policy Error / Value Error:

    (a) ``policy_disagree_total`` / ``value_error_total`` (per-seed counts and
        per-state |diff| sums, averaged across seeds). This is what was used
        before the 2026-04-28 BUG #7 finding.

    (b) ``policy_error_paper`` / ``value_error_paper`` — Paper §4.1 formula:

            PE = sum_x | mu_seeds[a_tilde(x)] - mu_seeds[a_hat(x)] |
            VE = sum_x | mu_seeds[V_tilde(x)]  - mu_seeds[V_hat(x)]  |

        i.e. average each side over seeds first, take |diff| afterward.
        By Jensen / triangle inequality, (b) <= per-seed average of |...|;
        (a) systematically over-estimates the paper formula. Use (b) when
        comparing to paper Tables 3/4 numbers directly.

        Paper §4.1 (tex:936-937) defines the σ side as
            σ_PE = (1/|X|) Σ_x σ_seeds[π̂(x)]
            σ_VE = (1/|X|) Σ_x σ_seeds[V^π̂(x)]
        i.e. σ on the SPE oracle (π̂, V̂), NOT the learned (π̃, Ṽ). To populate
        these σ terms with non-zero values, ``run_paper_eval.py`` rebuilds the
        SPE oracle per seed (different MC-rollout RNG → different π̂, V̂).
        The 2026-04-28 ``BUG #8 fix`` that flipped to (a_tilde, v_tilde) was a
        retracted misreading: it was a workaround for the cached-once oracle
        producing σ_hat ≈ 0, but it changes the formula's semantics. We have
        reverted it here and addressed the σ_hat ≈ 0 issue at the oracle build
        site instead.
    """
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

    # Paper §4.1 formula: average per-state across seeds, then |diff|.
    paper_metrics = _aggregate_paper_formula(all_metrics)
    if paper_metrics is not None:
        agg["policy_error_paper"] = paper_metrics["policy_error_paper"]
        agg["value_error_paper"] = paper_metrics["value_error_paper"]

    with open(os.path.join(root, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)

    with open(os.path.join(root, "aggregate.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std", "n"])
        for k, v in agg.items():
            n = v.get("n", v.get("n_states"))
            w.writerow([k, v["mean"], v["std"], n])

    return agg


def _aggregate_paper_formula(all_metrics: list[dict]) -> dict | None:
    """Compute paper §4.1 PE/VE from per_state tables across seeds.

    Returns ``None`` if any seed lacks per_state. Each seed's ``per_state`` is
    a list of {state, a_tilde, a_hat, v_tilde, v_hat}. We join across seeds by
    ``state`` (str-cast tuple) and apply the paper aggregator per state.
    """
    if not all_metrics:
        return None
    if not all(m.get("per_state") for m in all_metrics):
        return None

    # Collect per-state arrays across seeds
    state_keys = [str(row["state"]) for row in all_metrics[0]["per_state"]]
    state_data = {s: {"a_tilde": [], "a_hat": [], "v_tilde": [], "v_hat": []}
                  for s in state_keys}
    for m in all_metrics:
        seen = set()
        for row in m["per_state"]:
            s = str(row["state"])
            if s not in state_data:
                state_data[s] = {"a_tilde": [], "a_hat": [],
                                 "v_tilde": [], "v_hat": []}
            state_data[s]["a_tilde"].append(float(row["a_tilde"]))
            state_data[s]["a_hat"].append(float(row["a_hat"]))
            state_data[s]["v_tilde"].append(float(row["v_tilde"]))
            state_data[s]["v_hat"].append(float(row["v_hat"]))
            seen.add(s)

    pe_mean_terms = []
    ve_mean_terms = []
    pe_std_terms = []
    ve_std_terms = []
    for s, d in state_data.items():
        if not d["a_tilde"]:
            continue
        a_tilde = np.asarray(d["a_tilde"], dtype=np.float64)
        a_hat = np.asarray(d["a_hat"], dtype=np.float64)
        v_tilde = np.asarray(d["v_tilde"], dtype=np.float64)
        v_hat = np.asarray(d["v_hat"], dtype=np.float64)
        pe_mean_terms.append(abs(a_tilde.mean() - a_hat.mean()))
        ve_mean_terms.append(abs(v_tilde.mean() - v_hat.mean()))
        # Paper §4.1 (tex:936-937): σ on π̂ / V^π̂ (the SPE oracle), not π̃.
        # Requires per-seed SPE oracle rebuild (see run_paper_eval.py) to be
        # non-zero; cached-once oracle gives σ ≈ 0.
        pe_std_terms.append(a_hat.std())
        ve_std_terms.append(v_hat.std())
    n_x = len(pe_mean_terms)
    if n_x == 0:
        return None

    return {
        "policy_error_paper": {
            "mean": float(sum(pe_mean_terms)),
            "std": float(sum(pe_std_terms) / n_x),
            "n_states": n_x,
        },
        "value_error_paper": {
            "mean": float(sum(ve_mean_terms)),
            "std": float(sum(ve_std_terms) / n_x),
            "n_states": n_x,
        },
    }


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
