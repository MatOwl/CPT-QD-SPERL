"""Backward-induction SPE solver for the LNW abandonment env.

For each (t, x_idx) in backward order, MC-estimate
``Q^pi(t, x_idx, a) = CPT[Σ_τ r_τ | start (t, x_idx), first action a, follow π for τ>t]``
and set ``π(t, x_idx) = argmax_a Q``. Mirrors `barberis_spe` but with no wealth
offset because the LNW cumulative reward IS the CPT input (sunk costs at
``init_time`` are already excluded by construction — they were paid before the
rollout starts).

Iterates the parity-correct grid only (``x_idx`` shares parity with ``t``);
non-reachable states are skipped because the env never produces them.
"""

from __future__ import annotations

import numpy as np

from lib.cpt import CPTParams


def compute_spe_policy(env, cpt_params: CPTParams, n_eval_eps: int = 500,
                       rng=None):
    """Return a dict ``policy[(t, x_idx)] = a*`` for the LNW abandonment env."""
    rng = rng if rng is not None else np.random.default_rng(0)
    T = env.T
    policy: dict = {}

    def follow(state):
        t, x_idx = int(state[0]), int(state[1])
        return policy.get((t, x_idx), 0)  # default: abandon

    x1 = env.x1
    delta = env.delta

    # Backward induction over decision stages t = T-1, ..., 0.
    for t in range(T - 1, -1, -1):
        # Parity-correct grid: x_idx in {-t, -t+2, ..., t-2, t}
        for x_idx in range(-t, t + 1, 2):
            # State-dependent offset: current project value x_t = x_1 + x_idx*delta
            # (analogous to Barberis using current wealth z as offset).
            x_t = x1 + x_idx * delta
            q_vals = []
            for a in range(env.action_space.n):
                rewards = []
                for _ in range(n_eval_eps):
                    env.reset(init_time=t, init_x_idx=x_idx)
                    total = 0.0
                    action = a  # forced first action
                    while True:
                        nxt, r, done, _ = env.step(int(action))
                        total += r
                        if done:
                            break
                        action = follow(nxt)
                    # CPT input = current project value + future cumulative
                    # = full payoff from this state onward.
                    rewards.append(total + x_t)
                q_vals.append(cpt_params.compute(rewards))
            policy[(t, x_idx)] = int(np.argmax(q_vals))

    return policy


def spe_policy_fn(policy_dict):
    """Wrap dict into a ``(t, state) -> action`` callable for paper_eval."""
    def pi(_t, state):
        t, x_idx = int(state[0]), int(state[1])
        return int(policy_dict.get((t, x_idx), 0))
    return pi
