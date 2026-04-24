"""Per-environment featurizers for tabular FA critic/actor.

Each featurizer converts an env observation into:
  - ``loc``: an integer index in ``[0, n_states)`` for direct tabular indexing
  - ``key``: a hashable representation of the state (used as dict key for
    logging, replay, and (state, action) -> value bookkeeping)

The indexing-based interface avoids allocating full one-hot arrays on every
prediction (needed for environments like OptEx with O(10^4) states).
"""

from __future__ import annotations

import numpy as np


class Featurizer:
    n_states: int

    def loc(self, obs) -> int:
        raise NotImplementedError

    def key(self, obs):
        raise NotImplementedError

    def cpt_offset(self, obs) -> float:
        """Additive constant to shift critic quantiles by before CPT evaluation.

        Needed when the env's step reward is an increment relative to an
        embedded state coordinate (Barberis: wealth z) but the CPT reference is
        at the game's start (0). Default 0 — envs whose Σr is already the
        CPT-relevant quantity leave this untouched.
        """
        return 0.0


class BarberisFeaturizer(Featurizer):
    """Barberis casino: obs = (t, z), z \\in {-T*bet, ..., T*bet}."""

    def __init__(self, env):
        self.T = env.T
        self.bet = env.bet
        self.n_t = env.observation_space.spaces[0].n
        self.n_z = env.observation_space.spaces[1].n
        self.n_states = self.n_t * self.n_z

    def loc(self, obs) -> int:
        t, z = obs
        return int(self.n_z * int(t) + self.T + int(z) // self.bet)

    def key(self, obs):
        t, z = obs
        return (int(t), int(z))

    def cpt_offset(self, obs) -> float:
        """Barberis: CPT is on terminal wealth = z + Σr, so offset = z."""
        _, z = obs
        return float(z)

    def iter_states(self):
        """Enumerate reachable (t, z) states. z must lie in the support
        {-t*bet, ..., -bet, 0, bet, ..., t*bet} at each t."""
        for t in range(self.n_t):
            for k in range(-t, t + 1):
                yield (t, k * self.bet)


class OptExFeaturizer(Featurizer):
    """Optimal execution: obs = (logRet_bin, remain_bin, X_bin).

    Uses a 3D -> 1D flat index. Unseen log-return bins are clipped to the
    configured range (discretisation may produce values outside [-n_bins_price,
    n_bins_price] under extreme price moves).
    """

    def __init__(self, env):
        self.n_bins_price = env.n_bins_price
        self.n_bins_N = env.n_bins_N
        self.n_bins_X = env.n_bins_X
        # log-return bin is in [-n_bins_price, n_bins_price], so 2*n_bins_price+1 buckets
        self.price_span = 2 * self.n_bins_price + 1
        self.n_states = self.price_span * (self.n_bins_N + 1) * (self.n_bins_X + 1)

    def _clip(self, obs):
        p, n, x = int(obs[0]), int(obs[1]), int(obs[2])
        p = max(-self.n_bins_price, min(self.n_bins_price, p))
        n = max(0, min(self.n_bins_N, n))
        x = max(0, min(self.n_bins_X, x))
        return p, n, x

    def loc(self, obs) -> int:
        p, n, x = self._clip(obs)
        return (
            (p + self.n_bins_price) * (self.n_bins_N + 1) * (self.n_bins_X + 1)
            + n * (self.n_bins_X + 1)
            + x
        )

    def key(self, obs):
        return self._clip(obs)

    def iter_states(self, env=None):
        """OptEx reachable states — built via the BFS tree in ``optex_spe``.
        Requires passing the env so we can call ``env.next_state``.
        """
        if env is None:
            raise ValueError("OptExFeaturizer.iter_states needs the env instance.")
        from .optex_spe import OptExTree
        tree = OptExTree(env, [env.reset()], env.action_space_size)
        for (_, _), node in tree.state_time_space.items():
            yield tuple(int(v) for v in node.state)
