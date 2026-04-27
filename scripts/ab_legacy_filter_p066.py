"""A/B test: monkey-patch generic SPERL's `filter_quantiles` with the
legacy-faithful version, rerun CPT88/p=0.66 paper-eval at the best-known
config (sticky + filter=0.9 + accept=inf), and report 10-seed metrics.

Compare the resulting Optimality / Disagree / VE against
`runs/results_p066_10s_both_acceptInf` (generic-paper-pseudocode-faithful
filter): if numbers move toward paper 2.74±0.95, legacy's cascading +
signed-distance behavior is the missing piece for paper-table reproduction.

Run:
    PYTHONPATH=. python scripts/ab_legacy_filter_p066.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

from scripts.legacy_faithful_filter import legacy_faithful_filter

# Monkey-patch BEFORE importing the rest of the pipeline so QRCritic picks up
# the patched symbol on first reference.
import agents.sperl_qr_generic as _sqg
_sqg.filter_quantiles = legacy_faithful_filter

from lib.cpt import CPTParams
from lib.envs.registry import make_env, make_featurizer
from lib.envs.barberis_spe import compute_spe_policy, spe_policy_fn
from lib.paper_eval import compute_paper_metrics, learned_policy_fn
from lib.io import (
    run_name_from_args, save_config, save_seed_result, save_aggregate,
)
from agents.sperl_qr_generic import GreedySPERL


class Args:
    env = "barberis"
    seeds = 10
    train_eps = 15000
    batch = 5
    support_size = 50
    critic_lr = 0.04
    eps = 0.3
    alpha = 0.88
    rho1 = 0.65
    rho2 = 0.65
    lmbd = 2.25
    horizon = 5
    p_win = 0.66
    eval_per_state = 100
    spe_rollouts = 300
    sticky_policy = True
    tie_thresh = 0.0
    filter_thresh = 0.9
    filter_accept_ratio = float("inf")
    results_dir = "./runs/results_p066_10s_legacyfilt"
    no_save = False
    algo = "sperl"


def main():
    args = Args()
    cpt = CPTParams(args.alpha, args.rho1, args.rho2, args.lmbd)

    ref_env = make_env(args.env, p=args.p_win, bet=10, T=args.horizon)
    print(f"[ref] solving Barberis SPE (n_eval_eps={args.spe_rollouts})")
    spe_dict = compute_spe_policy(ref_env, cpt, n_eval_eps=args.spe_rollouts)
    reference_pi = spe_policy_fn(spe_dict)

    run_root = os.path.join(args.results_dir, run_name_from_args(args, args.env))
    save_config(run_root, {k: v for k, v in vars(Args).items()
                            if not k.startswith("_")})
    print(f"[io] writing to {run_root}")
    print(f"[patch] using legacy-faithful filter (cascading lb + signed dist)")

    all_metrics = []
    for seed in range(args.seeds):
        print(f"\n=== seed {seed} ===")
        env = make_env(args.env, p=args.p_win, bet=10, T=args.horizon)
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

        learned_pi = learned_policy_fn(agent)
        m = compute_paper_metrics(
            agent, env, featurizer, learned_pi, reference_pi,
            cpt, n_eps_per_state=args.eval_per_state,
        )
        print(f"[seed {seed}] disagree={m['policy_disagree_total']}/{m['n_states']}, "
              f"VE={m['value_error_total']:.3f}, "
              f"Opt={m['optimality']:.3f}, SW={m['social_welfare']:.3f}")
        save_seed_result(run_root, seed, agent, m)
        all_metrics.append(m)

    def agg(key):
        vals = np.array([float(m[key]) if m[key] is not None else np.nan
                         for m in all_metrics])
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    print("\n=== Aggregate (legacy-faithful filter, 10 seeds) ===")
    print(f"  Policy Error : {agg('policy_disagree_total')}")
    print(f"  Value Error  : {agg('value_error_total')}")
    print(f"  Optimality   : {agg('optimality')}")
    print(f"  SW           : {agg('social_welfare')}")
    sw_spe = float(np.nanmean([m['social_welfare_spe'] for m in all_metrics]))
    print(f"  SPE Welfare  : {sw_spe:.4f}")

    save_aggregate(run_root, all_metrics)
    print(f"[io] aggregate saved to {run_root}")


if __name__ == "__main__":
    main()
