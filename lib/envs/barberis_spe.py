"""Backward-induction SPE solver for Barberis casino.

At each (t, z) in backward order, for each action a, estimate
``Q^pi(t, z, a) = CPT[Σ_τ r_τ | start (t,z), first action a, follow π for τ>t]``
by Monte-Carlo rollouts, where π is the policy already filled in for τ>t.
Then set ``π(t, z) = argmax_a Q(t, z, a)``.

Mirrors ``QPG_CPT.compute_SPE`` in ``rerun_GreedySPERL_QR__main.py`` but is
standalone (no agent bookkeeping).
"""

from __future__ import annotations

import itertools

import numpy as np

from lib.cpt import CPTParams


def compute_spe_policy(env, cpt_params: CPTParams, n_eval_eps=500, rng=None):
    """Return a dict ``policy[(t, z)] = a*`` solving the backward-induction
    SPE for the Barberis casino with the given CPT parameters."""
    rng = rng if rng is not None else np.random.default_rng(0)
    T = env.T
    bet = env.bet
    policy = {}

    def follow(state):
        t, z = int(state[0]), int(state[1])
        return policy.get((t, z), 0)  # default to exit for t not yet set

    for t in range(T, -1, -1):
        for k in range(-t, t + 1):
            z = k * bet
            q_vals = []
            for a in range(env.action_space.n):
                rewards = []
                for _ in range(n_eval_eps):
                    state = env.reset(init_time=t, init_wealth=z)
                    total = 0.0
                    action = a  # forced first action
                    tau = t
                    while True:
                        nxt, r, done, _ = env.step(int(action))
                        total += r
                        tau += 1
                        if done:
                            break
                        action = follow(nxt)
                    rewards.append(total + z)
                q_vals.append(cpt_params.compute(rewards))
            policy[(t, z)] = int(np.argmax(q_vals))

    return policy


def spe_policy_fn(policy_dict):
    """Wrap dict into a ``(t, state) -> action`` callable for paper_eval."""
    def pi(_t, state):
        t, z = int(state[0]), int(state[1])
        return int(policy_dict.get((t, z), 0))
    return pi
