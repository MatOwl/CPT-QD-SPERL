"""Dynamically-Optimal (DO) naive baseline policy for LNW abandonment env.

Computes the **expected-value-maximizing** policy via standard Bellman
backward induction (no CPT applied during planning). When this policy is
later evaluated under CPT preferences, the gap between V^DO and V^SPE
quantifies the cost of ignoring loss-aversion / probability-weighting
when planning.

Same MC backward-induction structure as ``compute_spe_policy`` but
aggregates rollouts via ``np.mean`` instead of ``cpt_params.compute``.

This is the "naive" precommitment-at-each-state baseline: at each state,
the agent maximizes expected future absolute payoff, oblivious to any
CPT distortion of the payoff distribution.
"""

from __future__ import annotations

import numpy as np


def compute_do_policy(env, n_eval_eps: int = 500, rng=None):
    """Return ``policy[(t, x_idx)] = a*`` maximizing E[absolute payoff]."""
    rng = rng if rng is not None else np.random.default_rng(0)
    T = env.T
    policy: dict = {}

    def follow(state):
        t, x_idx = int(state[0]), int(state[1])
        return policy.get((t, x_idx), 0)

    x1 = env.x1
    delta = env.delta

    for t in range(T - 1, -1, -1):
        for x_idx in range(-t, t + 1, 2):
            # State-dependent absolute-payoff offset, same as SPE oracle.
            x_t = x1 + x_idx * delta
            q_vals = []
            for a in range(env.action_space.n):
                rewards = []
                for _ in range(n_eval_eps):
                    env.reset(init_time=t, init_x_idx=x_idx)
                    total = 0.0
                    action = a
                    while True:
                        nxt, r, done, _ = env.step(int(action))
                        total += r
                        if done:
                            break
                        action = follow(nxt)
                    rewards.append(total + x_t)
                # MEAN, not CPT — that's the only difference from SPE oracle.
                q_vals.append(float(np.mean(rewards)))
            policy[(t, x_idx)] = int(np.argmax(q_vals))

    return policy


def do_policy_fn(policy_dict):
    """Wrap dict into a ``(t, state) -> action`` callable for paper_eval."""
    def pi(_t, state):
        t, x_idx = int(state[0]), int(state[1])
        return int(policy_dict.get((t, x_idx), 0))
    return pi
