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
        """Enumerate decision states reachable from x_0=(0,0).

        Each step changes wealth by +/-bet, so z/bet must share parity with t.
        Terminal step t=T is excluded (no decision is made there). Yielding
        only these states matches paper Definition 5's domain for SW(pi)
        (verified: reproduces paper's SPE Welfare to within 1 std).
        """
        for t in range(min(self.n_t, self.T)):
            for k in range(-t, t + 1, 2):
                yield (t, k * self.bet)


class AbandonmentFeaturizer(Featurizer):
    """LNW abandonment: obs = (t, x_idx), x_idx in {-T, ..., T}.

    Parity-correct grid: at time t, reachable x_idx in {-t, -t+2, ..., t-2, t}.
    """

    def __init__(self, env):
        self.T = env.T
        self.delta = env.delta
        self.c = env.c
        self.x1 = env.x1
        self.n_t = env.observation_space.spaces[0].n  # T+1
        self.n_x = env.observation_space.spaces[1].n  # 2T+1
        self.n_states = self.n_t * self.n_x

    def loc(self, obs) -> int:
        t, x_idx = int(obs[0]), int(obs[1])
        return self.n_x * t + (self.T + x_idx)

    def key(self, obs):
        t, x_idx = obs
        return (int(t), int(x_idx))

    def cpt_offset(self, obs) -> float:
        """LNW: state-dependent offset = current project value x_t (analogous
        to Barberis cpt_offset = z, the current wealth). Mathematically:

            CPT input = (cumulative future reward) + x_t
                      = full payoff from this state onward
                        (zero on abandon, x_T - (T-t)*c on full continuation)

        This places the SPE-V at any abandon state at CPT(0) = 0, which matches
        LNW's "abandon → 0 future payoff" semantics."""
        _, x_idx = obs
        return float(self.x1 + int(x_idx) * self.delta)

    def iter_states(self):
        """Decision states reachable from the default init (t=0, x_idx=0).

        Each continue moves x_idx by ±1, so x_idx must share parity with t.
        Terminal step t=T is excluded (no decision is made there).
        """
        for t in range(self.T):
            for k in range(-t, t + 1, 2):
                yield (t, k)


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
