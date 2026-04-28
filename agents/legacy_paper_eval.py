"""Post-process legacy ``rerun_GreedySPERL_QR__main.py`` SPERL CSVs into the
4 paper §4.1 metrics, using the refactor's SPE oracle + eval pipeline.

Why: legacy outputs ``SPERL_aggr_*_trainPol.csv`` (per-seed 0/1 action choice
per (t, z)) but no Optimality / Social Welfare / Policy Error / Value Error
directly. To compare apples-to-apples with the refactor v3/v4 sweep, we wrap
each legacy seed's policy as a ``(t, state) -> action`` callable and feed it
into ``lib.paper_eval.compute_paper_metrics`` along with the same per-seed
SPE oracle (``lib.envs.barberis_spe.compute_spe_policy(seed=...)``) used by
the refactor sweep. Identical eval code → only difference is which agent
generated the policy.

Usage:
    PYTHONPATH=. python agents/legacy_paper_eval.py \\
        --alpha 0.88 --rho1 0.65 --lmbd 2.25 \\
        --p-win 0.66 --filter 0.9 --treshratio 0.5 --eps 0.6 \\
        --legacy-dir agents/barberis/results/static \\
        --results-dir runs/results_legacy_paper

Reads the legacy aggregate CSV matching the (alpha, rho, lmbd, pwin, K=50,
filter, treshratio, ss_inverted=1, eps) tuple, iterates over each seed row,
and writes a refactor-style ``aggregate.json`` (paper-formula PE/VE + the
old totals) alongside per-seed metrics. Use the existing
``scripts/summarize_paper_tables_*.py`` to compare to paper.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Make the repo root importable when invoked from the agents/ subdir.
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

from lib.cpt import CPTParams
from lib.envs.barberis_spe import compute_spe_policy, spe_policy_fn
from lib.envs.registry import make_env, make_featurizer
from lib.io import save_aggregate, save_seed_result
from lib.paper_eval import compute_paper_metrics


def _legacy_filename(legacy_dir: str, alpha, rho, lmbd, pwin, p_filter,
                     treshratio, eps, kind: str) -> str:
    """Match legacy ``record_csv.record_results`` aggregate naming.

    Format: SPERL_aggr_{alpha}_{rho}_{lmbd}_{pwin}_{K=50}_{lbub*p_filter}_
            {smoothen=treshratio}_{ss_inverted=1}_{eps}_{kind}.csv

    Legacy formats numbers with Python's default repr, so 0.5 → "0.5",
    1.0 → "1.0", 0.66 → "0.66". We use the same.
    """
    K = 50
    ss = 1
    return os.path.join(
        legacy_dir,
        f"SPERL_aggr_{alpha}_{rho}_{lmbd}_{pwin}_{K}_{p_filter}_"
        f"{treshratio}_{ss}_{eps}_{kind}.csv",
    )


def load_legacy_seed_policies(legacy_dir, alpha, rho, lmbd, pwin, p_filter,
                              treshratio, eps):
    """Return list of ``policy_fn`` callables, one per seed row in the legacy
    aggregate ``trainPol.csv``. Each row is the SPERL policy at end-of-training.
    """
    path = _legacy_filename(legacy_dir, alpha, rho, lmbd, pwin, p_filter,
                            treshratio, eps, "trainPol")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, header=[0, 1, 2], index_col=[0, 1, 2])

    # Columns are MultiIndex (t_str, z_str, action_str). Each (t, z) has
    # both action 0 and 1 columns; we want argmax over actions.
    policies = []
    seeds = []
    for idx, row in df.iterrows():
        runID, seed, iter_num = idx
        # Group columns by (t, z) and pick action with value 1.
        policy_dict = {}
        # Build {(t, z): {action: indicator}}
        by_state: dict[tuple, dict[int, float]] = {}
        for col, v in row.items():
            t_str, z_str, a_str = col
            t = int(t_str)
            z = int(z_str)
            a = int(a_str)
            by_state.setdefault((t, z), {})[a] = float(v)
        for (t, z), actions in by_state.items():
            # argmax across actions; tie → action 0 (paper-default ties go to
            # exit a.k.a. action 0 in barberis_casino; legacy's policy_val sets
            # exactly one action to 1 via np.argmax).
            policy_dict[(t, z)] = int(max(actions, key=lambda k: actions[k]))
        policies.append(policy_dict)
        seeds.append(int(seed))
    return seeds, policies


def policy_fn_from_dict(policy_dict):
    """Wrap legacy {(t, z): action} dict into a (t, state) -> action callable
    matching refactor's ``learned_policy_fn``."""
    def pi(_t, state):
        t, z = int(state[0]), int(state[1])
        return int(policy_dict.get((t, z), 0))  # exit by default
    return pi


def build_env(args):
    return make_env("barberis", p=args.p_win, bet=10, T=args.horizon)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--rho1", type=float, required=True)
    parser.add_argument("--lmbd", type=float, required=True)
    parser.add_argument("--p-win", type=float, required=True)
    parser.add_argument("--filter", type=float, required=True,
                        dest="p_filter",
                        help="paper §C.4 filterTresh; legacy: lbub*p_filter")
    parser.add_argument("--treshratio", type=float, default=0.5,
                        help="paper §C.2.5 treshRatio (legacy 'smoothen')")
    parser.add_argument("--eps", type=float, default=0.6,
                        help="exploration epsilon (legacy default 0.6)")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--spe-rollouts", type=int, default=2000,
                        help="paper Algorithm 5 M = 2000")
    parser.add_argument("--eval-per-state", type=int, default=100)
    parser.add_argument("--legacy-dir", type=str,
                        default="agents/barberis/results/static")
    parser.add_argument("--results-dir", type=str,
                        default="runs/results_legacy_paper")
    parser.add_argument("--cell-name", type=str, default=None,
                        help="output subdir name; defaults to "
                             "barberis_sperl_p{pwin}_cpt_a{a}_r{r}_l{l}")
    args = parser.parse_args()

    seeds, policies = load_legacy_seed_policies(
        args.legacy_dir, args.alpha, args.rho1, args.lmbd, args.p_win,
        args.p_filter, args.treshratio, args.eps,
    )
    print(f"[legacy-eval] loaded {len(policies)} seeds for "
          f"CPT(α={args.alpha},δ={args.rho1},λ={args.lmbd}) p_win={args.p_win} "
          f"filter={args.p_filter} treshRatio={args.treshratio} eps={args.eps}")

    cpt = CPTParams(args.alpha, args.rho1, args.rho1, args.lmbd)

    cell_name = args.cell_name or (
        f"barberis_sperl_p{args.p_win}_cpt_a{args.alpha}_r{args.rho1}_l{args.lmbd}"
    )
    run_root = os.path.join(args.results_dir, cell_name)
    os.makedirs(run_root, exist_ok=True)

    all_metrics = []
    for seed, policy_dict in zip(seeds, policies):
        env = build_env(args)
        featurizer = make_featurizer("barberis", env)

        # Per-seed SPE oracle (matches refactor v4 methodology): seed offset
        # 1_000_003 keeps oracle RNG decoupled from the agent's training.
        ref_env = build_env(args)
        spe_dict = compute_spe_policy(
            ref_env, cpt, n_eval_eps=args.spe_rollouts,
            seed=seed + 1_000_003,
        )
        reference_pi = spe_policy_fn(spe_dict)

        learned_pi = policy_fn_from_dict(policy_dict)

        # Stub agent shim so save_seed_result + compute_paper_metrics can read
        # ``agent.policy.predict``. We only need a deterministic predict.
        class _LegacyAgentShim:
            def __init__(self):
                self.stats = {"mean_rewards": [], "std_rewards": [],
                              "cpt_rewards": []}

                class _Pol:
                    @staticmethod
                    def predict(state, deterministic=True):
                        a = learned_pi(0, state)
                        out = np.zeros(env.action_space.n)
                        out[a] = 1.0
                        return out

                self.policy = _Pol()

        agent = _LegacyAgentShim()

        m = compute_paper_metrics(
            agent, env, featurizer, learned_pi, reference_pi,
            cpt, n_eps_per_state=args.eval_per_state,
        )
        print(f"[seed {seed}] disagree={m['policy_disagree_total']}/"
              f"{m['n_states']}, value_err={m['value_error_total']:.4f}, "
              f"opt={m['optimality']:.4f}, sw={m['social_welfare']:.4f}")
        save_seed_result(run_root, seed, agent, m)
        all_metrics.append(m)

    save_aggregate(run_root, all_metrics)
    print(f"[io] aggregate saved to {run_root}/aggregate.{{json,csv}}")


if __name__ == "__main__":
    main()
