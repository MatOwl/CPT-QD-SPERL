"""Micro-benchmark for training + paper-eval hot paths.

Measures wall-time for:
  (a) GreedySPERL training on a configurable env
  (b) compute_paper_metrics over all featurizer.iter_states() states

Usage:
    python scripts/benchmark.py --env optex --sigma 0.015 --num-w 4 \\
        --train-eps 100 --eval-per-state 20 \\
        --spe-file ../CumSPERL_ref/SPE_OptEx_5_0.015_4.npy

    python scripts/benchmark.py --env barberis --p-win 0.6 \\
        --train-eps 200 --eval-per-state 50 --spe-rollouts 100

Add ``--profile`` to also dump a cProfile top-15 breakdown.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

from lib.cpt import CPTParams
from lib.envs.registry import make_env, make_featurizer, REGISTERED_ENVS
from lib.paper_eval import compute_paper_metrics, learned_policy_fn
from agents.sperl_qr_generic import GreedySPERL


def build_env(args):
    if args.env == "barberis":
        return make_env("barberis", p=args.p_win, bet=10, T=args.horizon)
    return make_env(
        "optex",
        horizon=args.horizon,
        sigma=args.sigma,
        num_w=args.num_w,
        action_space_size=args.action_space_size,
    )


def get_reference_pi(args, env, cpt):
    if args.env == "barberis":
        from lib.envs.barberis_spe import compute_spe_policy, spe_policy_fn
        spe_dict = compute_spe_policy(env, cpt, n_eval_eps=args.spe_rollouts)
        return spe_policy_fn(spe_dict)
    from lib.envs.optex_spe import SPEOracle
    if not args.spe_file:
        raise SystemExit("--spe-file is required for optex")
    return SPEOracle(env, args.spe_file, initial_states=[env.reset()]).policy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=REGISTERED_ENVS, default="optex")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-eps", type=int, default=100)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--support-size", type=int, default=20)
    p.add_argument("--eval-per-state", type=int, default=20)
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--p-win", type=float, default=0.6)
    p.add_argument("--sigma", type=float, default=0.015)
    p.add_argument("--num-w", type=int, default=4)
    p.add_argument("--action-space-size", type=int, default=11)
    p.add_argument("--spe-rollouts", type=int, default=200)
    p.add_argument("--spe-file", type=str, default=None)
    p.add_argument("--profile", action="store_true",
                   help="also print cProfile top-15 for eval")
    args = p.parse_args()

    cpt = CPTParams()
    env = build_env(args)
    feat = make_featurizer(args.env, env)

    agent = GreedySPERL(
        env, feat, cpt,
        support_size=args.support_size, seed=args.seed,
        exploration={"type": "eps-greedy", "params": [args.eps]},
    )

    print(f"[bench] env={args.env} n_states={feat.n_states} nA={env.action_space.n}")

    t0 = time.time()
    agent.learn(
        n_train_eps=args.train_eps, n_batch=args.batch,
        n_eval_eps=20, eval_freq=args.train_eps + 1, verbose=0,
    )
    t_train = time.time() - t0
    print(f"[bench] train({args.train_eps} eps) : {t_train:.2f}s "
          f"({args.train_eps / t_train:.1f} eps/s)")

    ref_pi = get_reference_pi(args, env, cpt)
    learned_pi = learned_policy_fn(agent)

    pr = cProfile.Profile() if args.profile else None
    t0 = time.time()
    if pr: pr.enable()
    m = compute_paper_metrics(
        agent, env, feat, learned_pi, ref_pi, cpt,
        n_eps_per_state=args.eval_per_state,
    )
    if pr: pr.disable()
    t_eval = time.time() - t0
    print(f"[bench] eval({m['n_states']} states × {args.eval_per_state} eps × 2 policies)"
          f" : {t_eval:.2f}s ({t_eval / m['n_states'] * 1000:.1f}ms/state)")
    print(f"[result] policy_disagree={m['policy_disagree_total']}/{m['n_states']}, "
          f"val_err={m['value_error_total']:.3f}, "
          f"opt={m['optimality']:.3f}, welfare={m['social_welfare']:.3f}")

    if pr:
        print("\n--- cProfile top 15 (eval) ---")
        pstats.Stats(pr).sort_stats("cumulative").print_stats(15)


if __name__ == "__main__":
    main()
