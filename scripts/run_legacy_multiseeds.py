"""Phase C: train legacy QPG_CPT on N seeds with paper hyperparams,
evaluate with the same `compute_paper_metrics` we use on generic, and
aggregate. Compares directly to generic's `runs/results_p066_10s_*` outputs.

Trains seeds sequentially (legacy uses global numpy RNG, so parallel
runs would interfere). On a typical laptop one seed × 15k eps ≈ 3 min.

Usage:
    PYTHONPATH=. python scripts/run_legacy_multiseeds.py \\
        --seeds 10 --p-win 0.66 --filter 0.9 --tresh-ratio 0.5 \\
        --out ./runs/results_legacy_10s_p066
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

from scripts.legacy_extract import get_legacy_namespace, inject_closures
from scripts.run_legacy_one_seed import build_legacy_model

from lib.cpt import CPTParams
from lib.envs.registry import make_env, make_featurizer
from lib.envs.barberis_spe import compute_spe_policy, spe_policy_fn
from lib.paper_eval import compute_paper_metrics


def policy_fn_from_legacy_theta(theta, env_for_loc):
    """Build a callable `pi(t, state) -> action` from legacy's policy.theta.

    Legacy's `theta` is shape (n_states, n_actions) one-hot probs.
    Indexing convention follows `barberisFeaturize`:
        loc = 11 * t + (T + z // bet)
    where T = env.T, bet = env.bet (T=5, bet=10 → 11 z-bins -50..50).
    """
    T = env_for_loc.T
    bet = env_for_loc.bet
    n_z = env_for_loc.observation_space.spaces[1].n  # = 11

    def loc_of(state):
        t, z = int(state[0]), int(state[1])
        return n_z * t + T + z // bet

    def pi(_t_iter, state):
        loc = loc_of(state)
        return int(np.argmax(theta[loc, :]))

    return pi


def train_one_seed(args, seed):
    ns = get_legacy_namespace()
    inject_closures(ns, args.alpha, args.rho1, args.rho2, args.lmbd,
                    args.p_filter)
    env = ns["barberisCasino"](p=args.p_win)
    ns["env"] = env
    ns["envID"] = "barberis"
    ns["algoID"] = "SPERL"
    ns["runID"] = f"seed{seed}"

    class _A: pass
    a = _A()
    critic_lr = args.critic_lr_base / args.support_size
    model = build_legacy_model(
        ns, env, seed, a,
        support_size=args.support_size, critic_lr=critic_lr, eps=args.eps,
        lbub=args.lbub, tresh_ratio=args.tresh_ratio,
        step_size=args.step_size,
    )
    t0 = time.time()
    model.learn(
        n_train_eps=args.train_eps,
        n_batch=args.n_batch,
        n_eval_eps=args.n_eval_eps,
        eval_freq=args.eval_freq,
    )
    train_t = time.time() - t0

    return model, env, train_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
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
    p.add_argument("--critic-lr-base", type=float, default=2.0)
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--filter", type=float, default=0.9, dest="p_filter")
    p.add_argument("--tresh-ratio", type=float, default=0.5)
    p.add_argument("--lbub", type=int, default=1)
    p.add_argument("--step-size", type=float, default=5.0)
    p.add_argument("--eval-per-state", type=int, default=100)
    p.add_argument("--spe-rollouts", type=int, default=300)
    p.add_argument("--out", type=str, default="./runs/results_legacy_phase_c")
    args = p.parse_args()

    if args.lbub == 0:
        args.p_filter = 1.0
        args.tresh_ratio = 0.0

    os.makedirs(args.out, exist_ok=True)

    # SPE oracle (same one generic uses)
    cpt = CPTParams(args.alpha, args.rho1, args.rho2, args.lmbd)
    ref_env = make_env("barberis", p=args.p_win, bet=10, T=5)
    print(f"[ref] solving Barberis SPE (n_eval_eps={args.spe_rollouts})")
    spe_dict = compute_spe_policy(ref_env, cpt, n_eval_eps=args.spe_rollouts)
    reference_pi = spe_policy_fn(spe_dict)

    # Generic eval env + featurizer (need an actual env for compute_paper_metrics)
    eval_env = make_env("barberis", p=args.p_win, bet=10, T=5)
    featurizer = make_featurizer("barberis", eval_env)

    # We need to pass an "agent" to compute_paper_metrics; it uses
    # agent.evaluate_under_policy(policy_fn, n_eval_eps). Build a thin shim.

    class LegacyPolicyShim:
        """Quack like generic GreedyPolicy.predict(obs, deterministic=True) →
        one-hot probs, sourced from legacy's `model.policy.theta`."""

        def __init__(self, theta, env_for_loc):
            self.theta = theta
            self.T = env_for_loc.T
            self.bet = env_for_loc.bet
            self.n_z = env_for_loc.observation_space.spaces[1].n

        def predict(self, obs, deterministic=False):
            t, z = int(obs[0]), int(obs[1])
            loc = self.n_z * t + self.T + z // self.bet
            return self.theta[loc, :].copy()

    class LegacyAgentShim:
        def __init__(self, env_, cpt_params_, theta):
            self.env = env_
            self._cpt = cpt_params_
            self.policy = LegacyPolicyShim(theta, env_)

            class _C:
                def __init__(self, cp):
                    self.cpt_params = cp
            self.critic = _C(cpt_params_)

    all_metrics = []
    for seed in range(args.seeds):
        print(f"\n=== seed {seed} ===")
        model, train_env, train_t = train_one_seed(args, seed)
        print(f"[seed {seed}] training {args.train_eps} eps in {train_t:.1f}s")

        learned_pi = policy_fn_from_legacy_theta(model.policy.theta, train_env)
        shim = LegacyAgentShim(eval_env, cpt, model.policy.theta)
        m = compute_paper_metrics(
            shim, eval_env, featurizer, learned_pi, reference_pi,
            cpt, n_eps_per_state=args.eval_per_state,
        )
        print(f"[seed {seed}] disagree={m['policy_disagree_total']}/{m['n_states']}, "
              f"VE={m['value_error_total']:.3f}, Opt={m['optimality']:.3f}, "
              f"SW={m['social_welfare']:.3f}")

        # Persist seed outputs
        seed_dir = os.path.join(args.out, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        np.save(os.path.join(seed_dir, "qtile_theta.npy"),
                model.critic.qtile_theta)
        np.save(os.path.join(seed_dir, "policy_theta.npy"),
                model.policy.theta)
        with open(os.path.join(seed_dir, "metrics.json"), "w") as f:
            json.dump({k: (float(v) if isinstance(v, (int, float, np.floating))
                            else v)
                       for k, v in m.items()}, f, indent=2)

        all_metrics.append(m)

    # Aggregate
    def agg(key):
        vals = np.array([float(m[key]) if m[key] is not None else np.nan
                         for m in all_metrics])
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    print("\n=== Aggregate (legacy QPG_CPT, {} seeds) ===".format(args.seeds))
    print(f"  Policy Error : {agg('policy_disagree_total')}")
    print(f"  Value Error  : {agg('value_error_total')}")
    print(f"  Optimality   : {agg('optimality')}")
    print(f"  SW           : {agg('social_welfare')}")
    sw_spe = float(np.nanmean([m['social_welfare_spe'] for m in all_metrics]))
    print(f"  SPE Welfare  : {sw_spe:.4f}")

    summary = {
        "config": {k: getattr(args, k) for k in vars(args)},
        "aggregate": {
            "policy_disagree_mean_std": agg("policy_disagree_total"),
            "value_error_mean_std": agg("value_error_total"),
            "optimality_mean_std": agg("optimality"),
            "social_welfare_mean_std": agg("social_welfare"),
            "social_welfare_spe": sw_spe,
        },
    }
    with open(os.path.join(args.out, "aggregate.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
