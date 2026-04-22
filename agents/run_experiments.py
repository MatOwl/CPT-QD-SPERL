"""Unified experiment entry point.

Examples:
    python run_experiments.py --env barberis --algo sperl --seed 7 --train-eps 2000
    python run_experiments.py --env optex    --algo sperl --seed 0 --train-eps 500 \\
                              --support-size 20 --batch 20

The legacy per-algorithm scripts (``rerun_GreedySPERL_QR__main.py`` and
``rerun_SPSA_QRMC__main.py``) remain untouched for reproducing the paper's
Barberis results.
"""

from __future__ import annotations

import argparse
import os
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)
if ".." not in sys.path:
    sys.path.append("..")

from lib.cpt import CPTParams
from lib.envs.registry import make_env, make_featurizer, REGISTERED_ENVS
from agents.sperl_qr_generic import GreedySPERL
from agents.spsa_generic import SPSAAgent


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=REGISTERED_ENVS, default="barberis")
    parser.add_argument("--algo", choices=["sperl", "spsa"], default="sperl",
                        help="sperl: Greedy-SPERL w/ QR critic (time-consistent); "
                             "spsa: SPSA baseline (precommitment, s_0 only).")
    parser.add_argument("--seed", type=int, default=0)

    # shared hyperparams
    parser.add_argument("--train-eps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--eval-eps", type=int, default=200)
    parser.add_argument("--eval-freq", type=int, default=100)
    parser.add_argument("--support-size", type=int, default=50)
    parser.add_argument("--critic-lr", type=float, default=0.1)
    parser.add_argument("--critic-clip", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="[sperl] optional legacy lbub filter: clip "
                             "quantile rows to [LO, HI] after each update. "
                             "Useful on envs where known reward bounds exist "
                             "and default lr/param_init lets quantiles run away.")
    parser.add_argument("--target", choices=["TD", "MC"], default="TD")
    parser.add_argument("--order", choices=["fwd", "bwd"], default="bwd")

    # CPT hyperparams (Barberis reference defaults)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--rho1", type=float, default=0.5)
    parser.add_argument("--rho2", type=float, default=0.5)
    parser.add_argument("--lmbd", type=float, default=1.5)

    # env-specific
    parser.add_argument("--p-win", type=float, default=0.5,
                        help="[barberis] win probability")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-space-size", type=int, default=11,
                        help="[optex] action discretisation size")
    parser.add_argument("--sigma", type=float, default=0.019,
                        help="[optex] volatility; must match SPE file if provided")
    parser.add_argument("--num-w", type=int, default=4,
                        help="[optex] market-shock bins")

    # exploration
    parser.add_argument("--eps", type=float, default=0.1,
                        help="[sperl] eps-greedy exploration rate")

    # SPSA hyperparams
    parser.add_argument("--spsa-step", type=float, default=5.0,
                        help="[spsa] base step size a_0")
    parser.add_argument("--spsa-c", type=float, default=1.9,
                        help="[spsa] base perturbation c_0")
    parser.add_argument("--spsa-beta", type=float, default=1.0,
                        help="[spsa] softmax temperature")
    parser.add_argument("--theta-min", type=float, default=0.1)
    parser.add_argument("--theta-max", type=float, default=2.0)

    # optional SPE oracle evaluation
    parser.add_argument("--spe-file", type=str, default=None,
                        help="[optex] path to a CumSPERL SPE_OptEx_*.npy file; "
                             "if set, the oracle policy is also evaluated for "
                             "benchmarking.")
    parser.add_argument("--spe-eval-eps", type=int, default=500,
                        help="rollouts used to evaluate the SPE oracle")

    args = parser.parse_args()

    env = build_env(args.env, args)
    featurizer = make_featurizer(args.env, env)

    cpt = CPTParams(args.alpha, args.rho1, args.rho2, args.lmbd)

    if args.algo == "sperl":
        # NOTE: QRCritic uses env-sensitive tricks (first-visit mean init,
        # MC targets that collapse all quantiles to a scalar, etc.; see
        # QRCritic docstring). The defaults --critic-lr=0.1,
        # --support-size=50 are tuned for Barberis (rewards O(10)).
        # OptEx rewards are ~1e-3..1e-2 — if convergence looks weak,
        # retune critic-lr (try 0.01) and/or param_init before blaming the
        # algorithm. The legacy CumSPERL lbub-filter is not yet ported.
        exploration = {"type": "eps-greedy", "params": [args.eps]}
        agent = GreedySPERL(
            env, featurizer, cpt,
            support_size=args.support_size,
            critic_lr=args.critic_lr,
            exploration=exploration,
            target_type=args.target,
            order=args.order,
            seed=args.seed,
            critic_clip_bounds=tuple(args.critic_clip) if args.critic_clip else None,
        )
        print(f"[config] env={args.env} algo=sperl "
              f"n_states={featurizer.n_states} nA={env.action_space.n} "
              f"target={args.target} order={args.order} I={args.support_size}")
    else:  # spsa
        agent = SPSAAgent(
            env, featurizer, cpt,
            beta=args.spsa_beta,
            step_size=args.spsa_step,
            perturb_const=args.spsa_c,
            theta_min=args.theta_min, theta_max=args.theta_max,
            seed=args.seed,
        )
        print(f"[config] env={args.env} algo=spsa "
              f"n_states={featurizer.n_states} nA={env.action_space.n} "
              f"step={args.spsa_step} c={args.spsa_c} beta={args.spsa_beta}")

    agent.learn(
        n_train_eps=args.train_eps,
        n_batch=args.batch,
        n_eval_eps=args.eval_eps,
        eval_freq=args.eval_freq,
    )
    print("[done] final stats:")
    for k, v in agent.stats.items():
        print(f"  {k}: {v[-3:] if len(v) >= 3 else v}")

    if args.spe_file and args.env == "optex":
        from lib.envs.optex_spe import SPEOracle

        print(f"[spe] loading oracle from {args.spe_file}")
        oracle = SPEOracle(env, args.spe_file, initial_states=[env.reset()])
        print(f"[spe] tree size = {len(oracle.node_df)} nodes")
        mean_r, std_r, cpt_r = agent.evaluate_under_policy(
            oracle.policy(), args.spe_eval_eps
        )
        print(f"[spe] oracle eval over {args.spe_eval_eps} eps: "
              f"mean={mean_r:.6f} std={std_r:.6f} cpt={cpt_r:.6f}")


if __name__ == "__main__":
    main()
