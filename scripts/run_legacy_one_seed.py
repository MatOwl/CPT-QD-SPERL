"""Run legacy `QPG_CPT` on one Barberis seed with paper hyperparams.

Phase C of refactor verification: legacy as ground truth. Saves the
trained policy + qtile_theta + an evaluator-friendly policy callable
so we can run `compute_paper_metrics` against the same SPE oracle that
generic uses.

Usage:
    PYTHONPATH=. python scripts/run_legacy_one_seed.py --seed 0 \\
        --p-win 0.66 --alpha 0.88 --rho1 0.65 --rho2 0.65 --lmbd 2.25 \\
        --train-eps 15000 --filter 0.9 --tresh-ratio 0.5

Output (to --out):
    qtile_theta.npy, theta.npy (one-hot policy probs), policy_dict.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

from scripts.legacy_extract import get_legacy_namespace, inject_closures


def build_legacy_model(ns, env, seed, args, support_size=50, critic_lr=0.04,
                       eps=0.3, target_type="TD", order="bwd", lbub=1,
                       tresh_ratio=0.5, step_size=5.0):
    QPG_CPT = ns["QPG_CPT"]
    CPTCritic = ns["CPTCritic"]
    GreedyPolicy = ns["GreedyPolicy"]

    model = QPG_CPT(
        env,
        estimator_critic=CPTCritic,
        estimator_policy=GreedyPolicy,
        exploration={"type": "eps-greedy", "params": [eps]},
        seed=seed,
        target_type=target_type,
        order=order,
        state_only=False,
        support_size=support_size,
        with_quantile=True,
        with_huber=False,
        lbub=lbub,
        treshRatio=tresh_ratio,
        param_init=0.0,
        critic_lr=critic_lr,
        empty_memory=True,
        step_size=step_size,
    )
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-win", type=float, default=0.66)
    p.add_argument("--alpha", type=float, default=0.88)
    p.add_argument("--rho1", type=float, default=0.65)
    p.add_argument("--rho2", type=float, default=0.65)
    p.add_argument("--lmbd", type=float, default=2.25)
    p.add_argument("--train-eps", type=int, default=15000)
    p.add_argument("--n-batch", type=int, default=5)
    p.add_argument("--n-eval-eps", type=int, default=50)
    p.add_argument("--eval-freq", type=int, default=15000)
    p.add_argument("--support-size", type=int, default=50)
    p.add_argument("--critic-lr-base", type=float, default=2.0,
                   help="legacy convention: actual lr = base / support_size")
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--filter", type=float, default=0.9, dest="p_filter")
    p.add_argument("--tresh-ratio", type=float, default=0.5)
    p.add_argument("--lbub", type=int, default=1, help="1 = filter on, 0 = off")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    if args.lbub == 0:
        args.p_filter = 1.0
        args.tresh_ratio = 0.0

    ns = get_legacy_namespace()
    inject_closures(ns, args.alpha, args.rho1, args.rho2, args.lmbd,
                    args.p_filter)

    barberisCasino = ns["barberisCasino"]
    env = barberisCasino(p=args.p_win)
    # Legacy module-level functions (zloc_to_zobs, barberisFeaturize) reference
    # a global `env`; inject it so they resolve at runtime.
    ns["env"] = env
    # evaluate_critic_ references envID/algoID/runID set by the runner block
    # we're skipping. Provide stub strings — record_csv stub ignores them.
    ns["envID"] = "barberis"
    ns["algoID"] = "SPERL"
    ns["runID"] = "phase_c_legacy"

    critic_lr = args.critic_lr_base / args.support_size

    model = build_legacy_model(
        ns, env, args.seed, args,
        support_size=args.support_size, critic_lr=critic_lr, eps=args.eps,
        lbub=args.lbub, tresh_ratio=args.tresh_ratio,
    )

    print(f"[legacy] seed={args.seed} p_win={args.p_win} "
          f"K={args.support_size} lr={critic_lr} eps={args.eps} "
          f"filter={args.p_filter} treshRatio={args.tresh_ratio}")
    print(f"[legacy] training {args.train_eps} episodes...")

    model.learn(
        n_train_eps=args.train_eps,
        n_batch=args.n_batch,
        n_eval_eps=args.n_eval_eps,
        eval_freq=args.eval_freq,
    )

    print(f"[legacy] training done.")

    out = args.out
    if out is None:
        out = (f"./runs/results_legacy_seed{args.seed}_p{args.p_win}_"
               f"f{args.p_filter}_tr{args.tresh_ratio}")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "qtile_theta.npy"), model.critic.qtile_theta)
    np.save(os.path.join(out, "policy_theta.npy"), model.policy.theta)
    print(f"[legacy] saved {out}")


if __name__ == "__main__":
    main()
