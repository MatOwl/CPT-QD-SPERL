"""Loss-averse life-cycle consumption env (van Bilsen, Laeven, Nijman 2020).

A managerial / household-flavored CPT-SPERL environment based on van Bilsen,
Laeven & Nijman 2020 *Management Science* 66(9):3927-3955. Each year the agent
decides how much to consume relative to a slowly-adapting reference level
``R_t``. The reference is *endogenous*: it is a γ-EWMA of past consumption,
so the current decision shapes the future loss/gain threshold.

Adaptation to SPERL framework
-----------------------------
The original BLN paper uses continuous-time + per-period loss-averse utility
``u(c_t, R_t)``. SPERL applies CPT once at terminal on cumulative scalar
reward. We adapt by setting the per-step signed reward to

    r_t = c_t - R_t

so the SPERL agent maximises ``CPT(Σ_t (c_t - R_t))``, the loss-averse
distortion of cumulative *excess* consumption. This preserves BLN's
endogenous-reference mechanism (the key contribution) while fitting the
cumulative-distributional CPT operator. We deliberately do NOT match BLN's
welfare numbers — see paper §4 for the framing.

MDP simplifications (MVP scope)
-------------------------------
- Action: ``Discrete(3)`` with ``c_t = R_t * (1 + Δ * (a-1))`` for a in
  {0=decrease, 1=maintain, 2=increase}, default Δ=0.25.
- Portfolio share π fixed (default 0.5 in risky asset). Adding a portfolio
  decision is a post-MVP extension.
- Stock returns: 2-point binomial matched to lognormal (μ=5%, σ=20%):
  ``R_up = exp(μ+σ)``, ``R_down = exp(μ-σ)``, ``p_up = 0.5``.
- Wealth + reference snapped to discretization grids after each step.

State
-----
``(t, W_idx, R_idx)``
    * t in {0, 1, ..., T} — current decision stage
    * W_idx in {0, ..., n_W-1} — wealth bin (log-spaced grid)
    * R_idx in {0, ..., n_R-1} — reference bin (linear grid)

Reward
------
Per-step ``r_t = (c_t − R_t) * reward_scale``. ``cpt_offset`` is 0 (the
``r_t`` stream is already centered on the reference, so no terminal-wealth
offset is needed — unlike Barberis where CPT is on terminal wealth = z + Σr).

The optional ``reward_scale`` (default 100) lifts the per-step magnitude
from natural BLN units (≈ ±0.25) into Barberis-comparable units (≈ ±25),
so default SPERL hyperparameters (critic-lr 0.04, support-size 50) work
without retuning.
"""

from __future__ import annotations

import numpy as np
import gym
from gym import spaces


class BLNConsumption(gym.Env):
    """Loss-averse consumption + endogenous-reference env [BLN 2020]."""

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        T: int = 5,
        n_W: int = 10,
        n_R: int = 6,
        W_min: float = 0.5,
        W_max: float = 10.0,
        R_min: float = 0.3,
        R_max: float = 1.5,
        W_init: float = 5.0,
        R_init: float = 1.0,
        r: float = 0.01,
        mu: float = 0.05,
        sigma: float = 0.20,
        gamma: float = 0.3,
        delta_c: float = 0.25,
        pi_fixed: float = 0.5,
        reward_scale: float = 100.0,
    ):
        super().__init__()

        # ==== Horizon / discretization ====
        self.T = int(T)
        self.n_W = int(n_W)
        self.n_R = int(n_R)

        # ==== Dynamics parameters ====
        self.r = float(r)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.delta_c = float(delta_c)
        self.pi_fixed = float(pi_fixed)
        self.reward_scale = float(reward_scale)

        # ==== Grids ====
        self.W_grid = np.geomspace(W_min, W_max, self.n_W)
        self.R_grid = np.linspace(R_min, R_max, self.n_R)

        # ==== Initial state indices ====
        self.W_init_idx = int(np.argmin(np.abs(self.W_grid - W_init)))
        self.R_init_idx = int(np.argmin(np.abs(self.R_grid - R_init)))

        # ==== Stock return support (2-point binomial, matched moments) ====
        self.R_up = float(np.exp(self.mu + self.sigma))
        self.R_down = float(np.exp(self.mu - self.sigma))

        # ==== Action: ternary Δc ====
        # 0 = decrease c by Δ (loss-domain step), 1 = maintain c=R,
        # 2 = increase c by Δ (gain-domain step).
        self.action_space = spaces.Discrete(3)

        # ==== Observation: (t, W_idx, R_idx) ====
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.T + 1),
            spaces.Discrete(self.n_W),
            spaces.Discrete(self.n_R),
        ))

        self.time = None
        self.W_idx = None
        self.R_idx = None
        self.prev_time = None
        self.prev_W_idx = None
        self.prev_R_idx = None

    def seed(self, seed=None):
        # gym 0.26 deprecation; use reset(seed=seed) instead
        pass

    def reset(self, init_time=None, init_W_idx=None, init_R_idx=None,
              seed=None, return_info=False):
        """Reset to default initial state, or to a custom (t, W_idx, R_idx)
        for backward-induction SPE rollouts."""
        if seed is not None and hasattr(super(), "seed"):
            super().seed(seed)

        self.time = 0
        self.W_idx = self.W_init_idx
        self.R_idx = self.R_init_idx

        if init_time is not None:
            self.time = int(init_time)
        if init_W_idx is not None:
            self.W_idx = int(init_W_idx)
        if init_R_idx is not None:
            self.R_idx = int(init_R_idx)

        return np.array([self.time, self.W_idx, self.R_idx], dtype=np.float32)

    def _snap_W(self, W: float) -> int:
        W = max(float(self.W_grid[0]), min(float(self.W_grid[-1]), float(W)))
        return int(np.argmin(np.abs(self.W_grid - W)))

    def _snap_R(self, R: float) -> int:
        R = max(float(self.R_grid[0]), min(float(self.R_grid[-1]), float(R)))
        return int(np.argmin(np.abs(self.R_grid - R)))

    def step(self, action):
        """One env step. ``action`` in {0=decrease c, 1=maintain c, 2=increase c}.

        Reward = (c_t − R_t) * reward_scale. Idempotent at t=T (any action
        returns done=True with reward 0)."""
        self.prev_time = self.time
        self.prev_W_idx = self.W_idx
        self.prev_R_idx = self.R_idx

        if self.time >= self.T:
            done = True
            self.time += 1
            return self._get_obs(), 0.0, done, {}

        # Decode action
        if action == 0:
            c_factor = 1.0 - self.delta_c
        elif action == 1:
            c_factor = 1.0
        elif action == 2:
            c_factor = 1.0 + self.delta_c
        else:
            raise ValueError(f"Invalid action {action}; expected 0, 1, or 2.")

        # Current state values (continuous, from grids)
        W = float(self.W_grid[self.W_idx])
        R = float(self.R_grid[self.R_idx])

        # Feasible consumption (clip to wealth-1% buffer for solvency safety)
        c_target = R * c_factor
        c = min(c_target, W * 0.99)
        c = max(c, 0.0)

        # Per-step reward (signed excess consumption, scaled)
        reward = float((c - R) * self.reward_scale)

        # Stock realisation (binomial)
        R_stock = self.R_up if np.random.random() < 0.5 else self.R_down
        portfolio_return = (
            self.pi_fixed * R_stock + (1.0 - self.pi_fixed) * (1.0 + self.r)
        )

        # Wealth + reference dynamics
        savings = W - c
        W_next = savings * portfolio_return
        R_next = (1.0 - self.gamma) * R + self.gamma * c

        # Snap to grids
        self.W_idx = self._snap_W(W_next)
        self.R_idx = self._snap_R(R_next)
        self.time += 1

        done = (self.time >= self.T)
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.array(
            [self.time, self.W_idx, self.R_idx], dtype=np.float32
        )

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        W = float(self.W_grid[self.W_idx]) if self.W_idx is not None else None
        R = float(self.R_grid[self.R_idx]) if self.R_idx is not None else None
        print(f"t={self.time} W={W:.3f} R={R:.3f}")

    def close(self):
        pass
