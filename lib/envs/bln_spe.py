"""Backward-induction SPE solver for the BLN consumption env.

For each (t, W_idx, R_idx) in backward order, MC-estimate
``Q^pi(t, W_idx, R_idx, a) = CPT[Σ_τ r_τ | start s, first action a, follow π]``
and set ``π(s) = argmax_a Q``. Mirrors `abandonment_spe` but with a 3D state
grid and ``cpt_offset = 0`` (BLN's per-step reward r_t = c_t - R_t is already
centered on the reference, so the cumulative is the CPT-relevant quantity
directly — no terminal-wealth offset like Barberis or LNW).
"""

from __future__ import annotations

import numpy as np

from lib.cpt import CPTParams


def compute_spe_policy(env, cpt_params: CPTParams, n_eval_eps: int = 500,
                       rng=None):
    """Return ``policy[(t, W_idx, R_idx)] = a*`` for the BLN consumption env."""
    rng = rng if rng is not None else np.random.default_rng(0)
    T = env.T
    n_W = env.n_W
    n_R = env.n_R
    n_actions = env.action_space.n
    policy: dict = {}

    def follow(state):
        t, w_idx, r_idx = (
            int(state[0]), int(state[1]), int(state[2])
        )
        # Default: action 0 (decrease c) — most conservative under
        # loss-aversion, since c < R guarantees feasibility.
        return policy.get((t, w_idx, r_idx), 0)

    # Backward induction over decision stages t = T-1, ..., 0.
    for t in range(T - 1, -1, -1):
        for w_idx in range(n_W):
            for r_idx in range(n_R):
                q_vals = []
                for a in range(n_actions):
                    rewards = []
                    for _ in range(n_eval_eps):
                        env.reset(
                            init_time=t,
                            init_W_idx=w_idx,
                            init_R_idx=r_idx,
                        )
                        total = 0.0
                        action = a  # forced first action
                        while True:
                            nxt, r, done, _ = env.step(int(action))
                            total += r
                            if done:
                                break
                            action = follow(nxt)
                        # cpt_offset = 0 for BLN; cumulative IS the CPT input.
                        rewards.append(total)
                    q_vals.append(cpt_params.compute(rewards))
                policy[(t, w_idx, r_idx)] = int(np.argmax(q_vals))

    return policy


def spe_policy_fn(policy_dict):
    """Wrap dict into a ``(t, state) -> action`` callable for paper_eval."""
    def pi(_t, state):
        t, w_idx, r_idx = (
            int(state[0]), int(state[1]), int(state[2])
        )
        return int(policy_dict.get((t, w_idx, r_idx), 0))
    return pi
