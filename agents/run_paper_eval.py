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
    if name == "abandonment":
        return make_env(
            name,
            p=args.p_win,
            delta=args.delta,
            c=args.c,
            T=args.horizon,
            x1=args.x1,
        )
    if name == "bln":
        return make_env(
            name,
            T=args.horizon,
            n_W=args.n_W,
            n_R=args.n_R,
            gamma=args.gamma,
            delta_c=args.delta_c,
            pi_fixed=args.pi_fixed,
            mu=args.mu_stock,
            sigma=args.sigma_stock,
            r=args.r_free,
            reward_scale=args.reward_scale,
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
    if args.env == "abandonment":
        from lib.envs.abandonment_spe import compute_spe_policy, spe_policy_fn
        print(f"[ref] solving LNW abandonment SPE via backward induction "
              f"(n_eval_eps={args.spe_rollouts})")
        spe_dict = compute_spe_policy(
            env, CPTParams(args.alpha, args.rho1, args.rho2, args.lmbd),
            n_eval_eps=args.spe_rollouts,
        )
        return spe_policy_fn(spe_dict)
    if args.env == "bln":
        from lib.envs.bln_spe import compute_spe_policy, spe_policy_fn
        print(f"[ref] solving BLN consumption SPE via backward induction "
              f"(n_eval_eps={args.spe_rollouts})")
        spe_dict = compute_spe_policy(
            env, CPTParams(args.alpha, args.rho1, args.rho2, args.lmbd),
            n_eval_eps=args.spe_rollouts,
        )
        return spe_policy_fn(spe_dict)
    raise ValueError(args.env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=REGISTERED_ENVS, default="barberis")
    parser.add_argument("--seeds", type=int, default=3)

    # training -- defaults match paper Section C.2.5 (M=15000, K=50, rho=.04, xi=.3)
    parser.add_argument("--train-eps", type=int, default=15000,
                        help="paper C.2.5: M = 15000")
    parser.add_argument("--batch", type=int, default=5,
                        help="paper C.2.5 doesn't list n_batch; we use 5"
                             " (matches legacy verification cell)")
    parser.add_argument("--support-size", type=int, default=50,
                        help="paper C.2.5: K = 50")
    parser.add_argument("--critic-lr", type=float, default=0.04,
                        help="paper C.2.5: rho = .04")
    parser.add_argument("--eps", type=float, default=0.3,
                        help="paper C.2.5: xi = .3")

    # CPT
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--rho1", type=float, default=0.5)
    parser.add_argument("--rho2", type=float, default=0.5)
    parser.add_argument("--lmbd", type=float, default=1.5)

    # env
    parser.add_argument("--horizon", type=int, default=5,
                        help="[barberis] T = #decisions; "
                             "[abandonment] T = #decisions = paper's T-1; "
                             "[optex] horizon")
    parser.add_argument("--p-win", type=float, default=0.6,
                        help="[barberis/abandonment] up-move probability")
    parser.add_argument("--sigma", type=float, default=0.019)
    parser.add_argument("--num-w", type=int, default=4)
    parser.add_argument("--action-space-size", type=int, default=11)

    # abandonment-specific
    parser.add_argument("--delta", type=int, default=10,
                        help="[abandonment] random-walk step size")
    parser.add_argument("--c", type=int, default=11,
                        help="[abandonment] continuation cost per step")
    parser.add_argument("--x1", type=int, default=50,
                        help="[abandonment] initial project value")

    # bln-specific
    parser.add_argument("--n-W", type=int, default=10,
                        help="[bln] # wealth bins (log-spaced grid)")
    parser.add_argument("--n-R", type=int, default=6,
                        help="[bln] # reference bins (linear grid)")
    parser.add_argument("--gamma", type=float, default=0.3,
                        help="[bln] reference EWMA persistence (0=constant)")
    parser.add_argument("--delta-c", type=float, default=0.25,
                        help="[bln] consumption delta per step (c=R*(1+a*delta_c))")
    parser.add_argument("--pi-fixed", type=float, default=0.5,
                        help="[bln] fixed risky-asset fraction (no decision)")
    parser.add_argument("--mu-stock", type=float, default=0.05,
                        help="[bln] log-stock-return mean (annual)")
    parser.add_argument("--sigma-stock", type=float, default=0.20,
                        help="[bln] log-stock-return vol (annual)")
    parser.add_argument("--r-free", type=float, default=0.01,
                        help="[bln] risk-free annual rate")
    parser.add_argument("--reward-scale", type=float, default=100.0,
                        help="[bln] multiplier on per-step reward (c-R)*scale "
                             "for hyperparam compatibility with Barberis")

    # eval
    parser.add_argument("--eval-per-state", type=int, default=100,
                        help="# rollouts per state for V^pi(x)")
    parser.add_argument("--spe-rollouts", type=int, default=2000,
                        help="paper Algorithm 5: M = 2000 rollouts per (s,a)"
                             " for backward-induction SPE oracle.")
    parser.add_argument("--spe-file", type=str, default=None,
                        help="path to SPE_OptEx_*.npy (optex only)")

    # paper Alg 3 -- consistent tie-break / is-better guard
    parser.add_argument("--sticky-policy", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Alg 3 (paper C.2.2 default): only update policy"
                             " when new argmax STRICTLY beats old. Pass"
                             " --no-sticky-policy to disable.")
    parser.add_argument("--tie-thresh", type=float, default=0.0,
                        help="Randomize tie-break within tie-thresh of max"
                             " CPT. Default 0 = exact ties only.")

    # paper Alg 4 -- quantile filter
    parser.add_argument("--filter-thresh", type=float, default=None,
                        help="Alg 4: gap-quantile threshold (filterTresh),"
                             " e.g. 0.75. None disables filtering. Per-cell"
                             " Pareto-optimal values from paper C.4 Tables 3/4.")
    parser.add_argument("--filter-accept-ratio", type=float,
                        default=0.5,
                        help="Alg 4 line 20: treshRatio. Paper C.2.5"
                             " default = 0.5 when filter<1, else 0. Pass inf"
                             " to disable the gate (legacy 'trust filter').")
    parser.add_argument("--filter-gate-mode", choices=["relative", "absolute"],
                        default="absolute",
                        help="Alg 4 line 20 gate semantics. Paper text reads"
                             " |Q_filt - Q| > treshRatio (absolute) -- the"
                             " refactor default. 'relative' matches legacy"
                             " rerun_GreedySPERL_QR__main.py (|dQ|/|Q|).")

    # persistence
    parser.add_argument("--results-dir", type=str, default="./runs",
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
            filter_gate_mode=args.filter_gate_mode,
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
