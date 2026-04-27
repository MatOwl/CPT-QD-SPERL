"""Parse `agents/barberis/results/static/SPERL_<datetime>.csv` outputs from
native legacy runs and re-evaluate each seed's final policy with our
`compute_paper_metrics` against the same SPE oracle the generic / Phase C
pipeline uses.

This bridges the legacy native record_csv format to our Phase C eval
protocol so numbers are directly comparable.

Usage:
    PYTHONPATH=. python scripts/parse_legacy_native.py \\
        --static-dir agents/barberis/results/static \\
        --p-win-cells 0.36 0.3 0.42

Reads each `SPERL_<datetime>.csv` (the per-seed result file), extracts the
FINAL `pi_i` row as the trained policy, runs paper-metrics against the
SPE oracle for that (alpha, rho1, rho2, lmbd, p_win) cell.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

from lib.cpt import CPTParams
from lib.envs.registry import make_env, make_featurizer
from lib.envs.barberis_spe import compute_spe_policy, spe_policy_fn
from lib.paper_eval import compute_paper_metrics


def parse_param_txt(txt_path):
    """Read SPERL_<datetime>.txt and return a dict of (key, str-value)."""
    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    out = {}
    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def extract_final_policy_from_csv(csv_path, T=5, bet=10):
    """Read SPERL_<runID>.csv (multi-index header) and return:
       theta : ndarray (n_states=66, n_actions=2) of one-hot probs
    where loc = 11*t + (T + z//bet).

    The CSV has header rows [t, z, a] and index rows [runID, kind, iter_num].
    The 'pi_i' rows are policy probabilities (0/1 per action). The final
    iter's pi_i row is the trained policy.
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=[0, 1, 2])
    # Index level 1 indicates kind: 'q_i', 'q_true', 'policy_i', 'visit_freq_total'
    pi_rows = df.xs("policy_i", level=1)
    # Take the last iteration's row
    final_row = pi_rows.iloc[-1]
    n_z = 2 * T + 1  # = 11
    n_states = (T + 1) * n_z  # = 66
    theta = np.zeros((n_states, 2), dtype=np.float64)
    for col, val in final_row.items():
        t_str, z_str, a_str = col
        try:
            t = int(t_str)
            z = int(z_str)
            a = int(a_str)
        except ValueError:
            continue
        loc = n_z * t + T + z // bet
        if 0 <= loc < n_states:
            theta[loc, a] = float(val)
    return theta


def cpt_from_param_txt(params):
    return CPTParams(
        alpha=float(params["alpha"]),
        rho1=float(params["delta"].strip("()").split(",")[0]),
        rho2=float(params["delta"].strip("()").split(",")[1]),
        lmbd=float(params["lmbd"]),
    )


def evaluate_native_outputs(static_dir, p_win_cells, eval_per_state=100,
                            spe_rollouts=300):
    files = sorted(os.listdir(static_dir))
    runs = []  # list of (datetime_id, params, theta)

    for f in files:
        # Match SPERL_<datetime>.txt
        if not f.startswith("SPERL_") or not f.endswith(".txt"):
            continue
        if "QFDyn" in f or "aggr" in f:
            continue
        datetime_id = f[len("SPERL_"):-len(".txt")]
        csv_path = os.path.join(static_dir, f"SPERL_{datetime_id}.csv")
        if not os.path.exists(csv_path):
            continue
        params = parse_param_txt(os.path.join(static_dir, f))
        try:
            p_win = float(params["p_win"])
        except KeyError:
            continue
        if p_win_cells and p_win not in p_win_cells:
            continue
        try:
            theta = extract_final_policy_from_csv(csv_path)
        except Exception as e:
            print(f"[{datetime_id}] parse failed: {e}")
            continue
        runs.append((datetime_id, params, theta))

    print(f"Loaded {len(runs)} native runs from {static_dir}")

    # Group by p_win
    cells = {}
    for datetime_id, params, theta in runs:
        p_win = float(params["p_win"])
        cells.setdefault(p_win, []).append((datetime_id, params, theta))

    summary = []
    for p_win, group in sorted(cells.items()):
        print(f"\n=== Cell p_win={p_win} ({len(group)} seeds) ===")
        cpt = cpt_from_param_txt(group[0][1])

        ref_env = make_env("barberis", p=p_win, bet=10, T=5)
        spe_dict = compute_spe_policy(ref_env, cpt, n_eval_eps=spe_rollouts)
        reference_pi = spe_policy_fn(spe_dict)

        eval_env = make_env("barberis", p=p_win, bet=10, T=5)
        featurizer = make_featurizer("barberis", eval_env)

        cell_metrics = []
        for datetime_id, params, theta in group:
            shim = _ShimAgent(eval_env, cpt, theta)
            learned_pi = _policy_fn_from_theta(theta, eval_env)
            m = compute_paper_metrics(
                shim, eval_env, featurizer, learned_pi, reference_pi, cpt,
                n_eps_per_state=eval_per_state,
            )
            print(f"  seed={params.get('seed','?'):>3s}: disagree={m['policy_disagree_total']}/"
                  f"{m['n_states']}, VE={m['value_error_total']:.2f}, "
                  f"Opt={m['optimality']:.3f}, SW={m['social_welfare']:.2f}")
            cell_metrics.append(m)

        opt_arr = np.array([float(m["optimality"]) if m["optimality"] is not None else np.nan
                            for m in cell_metrics])
        disagree_arr = np.array([float(m["policy_disagree_total"]) for m in cell_metrics])
        ve_arr = np.array([float(m["value_error_total"]) for m in cell_metrics])
        sw_arr = np.array([float(m["social_welfare"]) for m in cell_metrics])

        print(f"  Aggregate ({len(group)} seeds):")
        print(f"    Optimality: {np.nanmean(opt_arr):.3f} +/- {np.nanstd(opt_arr):.3f}")
        print(f"    Disagree:   {disagree_arr.mean():.2f} +/- {disagree_arr.std():.2f}")
        print(f"    VE:         {ve_arr.mean():.2f} +/- {ve_arr.std():.2f}")
        print(f"    SW:         {sw_arr.mean():.2f} +/- {sw_arr.std():.2f}")
        summary.append({
            "p_win": p_win,
            "n_seeds": len(group),
            "optimality_mean": float(np.nanmean(opt_arr)),
            "optimality_std": float(np.nanstd(opt_arr)),
            "disagree_mean": float(disagree_arr.mean()),
            "disagree_std": float(disagree_arr.std()),
            "ve_mean": float(ve_arr.mean()),
            "ve_std": float(ve_arr.std()),
            "sw_mean": float(sw_arr.mean()),
            "sw_std": float(sw_arr.std()),
        })

    print("\n=== Summary ===")
    print(pd.DataFrame(summary).to_string(index=False))
    return summary


# Shim used by compute_paper_metrics

def _policy_fn_from_theta(theta, env):
    T = env.T; bet = env.bet
    n_z = env.observation_space.spaces[1].n

    def pi(_t, state):
        t, z = int(state[0]), int(state[1])
        loc = n_z * t + T + z // bet
        return int(np.argmax(theta[loc, :]))
    return pi


class _PolicyShim:
    def __init__(self, theta, env):
        self.theta = theta
        self.T = env.T
        self.bet = env.bet
        self.n_z = env.observation_space.spaces[1].n

    def predict(self, obs, deterministic=False):
        t, z = int(obs[0]), int(obs[1])
        loc = self.n_z * t + self.T + z // self.bet
        return self.theta[loc, :].copy()


class _ShimAgent:
    def __init__(self, env, cpt, theta):
        self.env = env
        self.policy = _PolicyShim(theta, env)

        class _C:
            def __init__(self, cp):
                self.cpt_params = cp
        self.critic = _C(cpt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--static-dir", type=str,
                   default="agents/barberis/results/static")
    p.add_argument("--p-win-cells", type=float, nargs="*",
                   default=[0.36, 0.3, 0.42])
    p.add_argument("--eval-per-state", type=int, default=100)
    p.add_argument("--spe-rollouts", type=int, default=300)
    args = p.parse_args()

    evaluate_native_outputs(args.static_dir, args.p_win_cells,
                            eval_per_state=args.eval_per_state,
                            spe_rollouts=args.spe_rollouts)


if __name__ == "__main__":
    main()
