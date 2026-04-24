"""Paper-style evaluation (MSci_MANUSCRIPT §4.1–4.2).

Trains a Greedy-SPERL agent and reports:
  * Policy disagreement count (vs reference SPE policy)
  * Value error Σ_x |V_tilde(x) − V_hat(x)|
  * Optimality V_tilde(x_0)
  * Social welfare Σ_x V_tilde(x)

Reference SPE policy:
  * barberis → backward-induction solver (``lib.envs.barberis_spe``)
  * optex    → preloaded .npy oracle (``lib.envs.optex_spe``)

Aggregates μ, σ of each metric across seeds.

Examples:
    python agents/run_paper_eval.py --env barberis --p-win 0.6 --seeds 3 \\
        --train-eps 2000 --eval-per-state 100 --spe-rollouts 300

    python agents/run_paper_eval.py --env optex --sigma 0.015 --num-w 4 \\
        --seeds 2 --train-eps 500 --eval-per-state 50 \\
        --spe-file ../../../CumSPERL_ref/SPE_OptEx_5_0.015_4.npy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)
if ".." not in sys.path:
    sys.path.append("..")

from lib.cpt import CPTParams
from lib.envs.registry import make_env, make_featurizer, REGISTERED_ENVS
from lib.paper_eval import compute_paper_metrics, learned_policy_fn
from lib.io import (
    run_name_from_args, save_config, save_seed_result, save_aggregate,
)
from agents.sperl_qr_generic import GreedySPERL


def build_env(name, args):
    if name == "barberis":
        return make_env(name, p=args.p_win, bet=10, T=args.horizon)
    if name == "optex":
        return make_env(
            name,
            horizon=args.horizon,
            sigma=args.sigma,
            num_w=args.num_w,
            action_space_size=args.action_space_size,
        )
    raise ValueError(name)


def get_reference_policy(args, env):
    if args.env == "barberis":
        from lib.envs.barberis_spe import compute_spe_policy, spe_policy_fn
        print(f"[ref] solving Barberis SPE via backward induction "
              f"(n_eval_eps={args.spe_rollouts})")
        spe_dict = compute_spe_policy(
            env, CPTParams(args.alpha, args.rho1, args.rho2, args.lmbd),
            n_eval_eps=args.spe_rollouts,
        )
        return spe_policy_fn(spe_dict)
    if args.env == "optex":
        if not args.spe_file:
            raise ValueError("--spe-file required for optex paper-eval")
        from lib.envs.optex_spe import SPEOracle
        state = env.reset()
        oracle = SPEOracle(env, args.spe_file, initial_states=[state])
        print(f"[ref] OptEx SPE oracle loaded, tree size = "
              f"{len(oracle.node_df)} nodes")
        return oracle.policy()
    raise ValueError(args.env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=REGISTERED_ENVS, default="barberis")
    parser.add_argument("--seeds", type=int, default=3)

    # training
    parser.add_argument("--train-eps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--support-size", type=int, default=20)
    parser.add_argument("--critic-lr", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.3)

    # CPT
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--rho1", type=float, default=0.5)
    parser.add_argument("--rho2", type=float, default=0.5)
    parser.add_argument("--lmbd", type=float, default=1.5)

    # env
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--p-win", type=float, default=0.6)
    parser.add_argument("--sigma", type=float, default=0.019)
    parser.add_argument("--num-w", type=int, default=4)
    parser.add_argument("--action-space-size", type=int, default=11)

    # eval
    parser.add_argument("--eval-per-state", type=int, default=100,
                        help="# rollouts per state for V^pi(x)")
    parser.add_argument("--spe-rollouts", type=int, default=300,
                        help="# rollouts per (s,a) for backward-induction SPE"
                             " (barberis only)")
    parser.add_argument("--spe-file", type=str, default=None,
                        help="path to SPE_OptEx_*.npy (optex only)")

    # paper Alg 3 — consistent tie-break / is-better guard
    parser.add_argument("--sticky-policy", action="store_true",
                        help="Alg 3: only update policy when new argmax"
                             " strictly beats old at this state (paper §C.2).")
    parser.add_argument("--tie-thresh", type=float, default=0.0,
                        help="Randomize tie-break within tie-thresh of max"
                             " CPT. Default 0 = exact ties only.")

    # paper Alg 4 — quantile filter
    parser.add_argument("--filter-thresh", type=float, default=None,
                        help="Alg 4: gap-quantile threshold (filterTresh),"
                             " e.g. 0.75. None disables filtering.")
    parser.add_argument("--filter-accept-ratio", type=float,
                        default=float("inf"),
                        help="Alg 4: max relative CPT deviation to accept"
                             " filtered quantiles (treshRatio). Default inf"
                             " = trust filter unconditionally.")

    # persistence
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="root directory for saved artifacts")
    parser.add_argument("--no-save", action="store_true",
                        help="disable writing any files")

    args = parser.parse_args()
    # synthetic field for run_name_from_args compat
    args.algo = "sperl"

    cpt = CPTParams(args.alpha, args.rho1, args.rho2, args.lmbd)

    # Reference policy (depends only on env, not seed)
    ref_env = build_env(args.env, args)
    reference_pi = get_reference_policy(args, ref_env)

    # Persistence setup
    run_root = None
    if not args.no_save:
        run_root = os.path.join(args.results_dir, run_name_from_args(args, args.env))
        save_config(run_root, vars(args))
        print(f"[io] writing to {run_root}")

    all_metrics = []
    for seed in range(args.seeds):
        print(f"\n=== seed {seed} ===")
        env = build_env(args.env, args)
        featurizer = make_featurizer(args.env, env)
        agent = GreedySPERL(
            env, featurizer, cpt,
            support_size=args.support_size, critic_lr=args.critic_lr,
            exploration={"type": "eps-greedy", "params": [args.eps]},
            target_type="TD", order="bwd", seed=seed,
            sticky_policy=args.sticky_policy,
            tie_thresh=args.tie_thresh,
            filter_thresh=args.filter_thresh,
            filter_accept_ratio=args.filter_accept_ratio,
        )
        agent.learn(
            n_train_eps=args.train_eps, n_batch=args.batch,
            n_eval_eps=50, eval_freq=args.train_eps, verbose=0,
        )
        print(f"[seed {seed}] training done")

        learned_pi = learned_policy_fn(agent)
        m = compute_paper_metrics(
            agent, env, featurizer, learned_pi, reference_pi,
            cpt, n_eps_per_state=args.eval_per_state,
        )
        print(f"[seed {seed}] policy_disagree={m['policy_disagree_total']}/"
              f"{m['n_states']}, value_err={m['value_error_total']:.4f}, "
              f"optimality={m['optimality']:.4f}, "
              f"social_welfare={m['social_welfare']:.4f}")
        if run_root is not None:
            save_seed_result(run_root, seed, agent, m)
        all_metrics.append(m)

    # Aggregate
    def agg(key):
        vals = np.array([float(m[key]) if m[key] is not None else np.nan
                         for m in all_metrics])
        return np.nanmean(vals), np.nanstd(vals)

    print("\n=== Aggregate across seeds ===")
    print(f"  Policy Error     : {agg('policy_disagree_total')}")
    print(f"  Value Error      : {agg('value_error_total')}")
    print(f"  Optimality       : {agg('optimality')}")
    print(f"  Social Welfare   : {agg('social_welfare')}")
    sw_spe = np.nanmean([m["social_welfare_spe"] for m in all_metrics])
    print(f"  SPE Welfare (ref): {sw_spe:.4f}")

    if run_root is not None:
        save_aggregate(run_root, all_metrics)
        print(f"[io] aggregate saved to {run_root}/aggregate.{{json,csv}}")


if __name__ == "__main__":
    main()
