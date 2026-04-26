"""Multistage project abandonment env (Long, Nasiry, Wu 2020 MS).

Operations-flavored counterpart of `barberis_casino`. The PM observes the
current project value and at each stage decides ``continue`` (pay cost,
advance one step) or ``abandon`` (terminate, receive 0).

Dynamics
--------
Project value follows a binary random walk:

    x_{t+1} = x_t + delta   w.p. p
              x_t - delta   w.p. 1 - p

Reward stream (delta-style, mirrors Barberis casino):
    continue : r = delta * outcome - c       (project value change minus cost)
    abandon at time t : r = -x_t             (recovery: cancel BOTH accumulated
                                              value change AND x_1 offset, so
                                              cumulative+offset = -t*c, which
                                              matches paper "abandon → 0 payoff"
                                              semantics)

This delta-style reward (instead of "lump sum at terminal continue") makes per-
step magnitudes comparable to Barberis ±bet, which is critical for QR critic
training: with first-visit-mean quantile init, an always-negative reward stream
(-c every step) causes Q(continue) to start far below Q(abandon)=0 and the
agent can get stuck in cold-start abandon. Spreading the value-change over
steps gives the critic non-trivial early signal.

Cumulative reward over an episode (CPT input via featurizer.cpt_offset = x1):
    full continuation : x_1 + (x_T - x_1) - T*c = x_T - T*c
    abandon at stage k: x_1 + (x_k - x_1) - k*c + (-(x_k - x_1)) = x_1 - k*c

Note: when CPT-evaluated, the "abandon" cumulative still represents the LNW
paper's semantics (PM forfeits project, only loses sunk costs) because the
recovery term cancels out the project value at abandon time, leaving only the
cost stream. The cpt_offset = x_1 places the CPT reference at the initial
project value (analogous to Barberis cpt_offset = z_init).

State: ``(t, x_idx)``
    * ``t`` in {0, 1, ..., T} — current stage (T = terminal)
    * ``x_idx`` in {-T, ..., T} — project value index, x_t = x1 + x_idx * delta
      Parity-correct grid: at time ``t``, reachable x_idx in {-t, -t+2, ..., t-2, t}.

Horizon convention
------------------
``env.T`` = number of decision stages (matches barberis_casino convention).
The LNW paper uses T_paper = number of stages (decisions + 1 terminal stage).
To replicate paper experiments with T_paper in {4, 6, 8, 10}, pass
``T = T_paper - 1`` in {3, 5, 7, 9}.

CPT framing
-----------
CPT is computed on terminal cumulative + initial project value:
``CPT input = x_1 + Σ r_t``. Same structure as Barberis (CPT on terminal
wealth = z_init + Σ r_t).
``featurizer.cpt_offset = x_1`` and SPE rollouts compute
``cpt_params.compute(total + x_1)``.

Note on the paper's behavioral model
------------------------------------
The LNW paper fits experimental data with a three-component behavioral model
(reference dependence + sunk cost bias + status quo bias). This env intentionally
exposes only the *raw* managerial setup (random walk + continuation cost +
terminal payoff). The CPT preferences (Tversky-Kahneman alpha/rho/lambda) are
applied at the algorithm level via lib.cpt — they replace, not extend, the
paper's behavioral model.
"""

from __future__ import annotations

import numpy as np
import gym
from gym import spaces


class AbandonmentProject(gym.Env):
    """Multistage project abandonment env [Long-Nasiry-Wu 2020]."""

    metadata = {"render.modes": ["console"]}

    def __init__(self, p: float = 0.5, delta: int = 10, c: int = 11,
                 T: int = 5, x1: int = 50):
        super().__init__()

        # ==== Dynamics / Reward parameters ====
        self.p = p          # probability of upward random-walk move
        self.delta = delta  # ±delta per continue
        self.c = c          # continuation cost per step
        self.T = T          # number of decision stages (= paper's T - 1)
        self.x1 = x1        # initial project value at t=0

        # ==== Action: 0 = abandon, 1 = continue ====
        self.action_space = spaces.Discrete(2)

        # ==== State: (t, x_idx) ====
        # t in {0, 1, ..., T} -> Discrete(T+1)
        # x_idx in {-T, ..., T} -> Discrete(2T+1) (offset by +T at indexing)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.T + 1),
            spaces.Discrete(2 * self.T + 1),
        ))

        self.time = None
        self.x_idx = None
        self.prev_time = None
        self.prev_x_idx = None

    def seed(self, seed=None):
        # gym 0.26 deprecation; use reset(seed=seed) instead
        pass

    def reset(self, init_time=None, init_x_idx=None,
              seed=None, return_info=False):
        """Reset to (t=0, x=x1) by default; allow custom (init_time, init_x_idx)
        for backward-induction SPE rollouts."""
        if seed is not None and hasattr(super(), "seed"):
            super().seed(seed)

        self.time = 0
        self.x_idx = 0  # x = x1 at start (x_idx 0 means no offset)

        if init_time is not None and init_x_idx is not None:
            self.time = int(init_time)
            self.x_idx = int(init_x_idx)

        return np.array([self.time, self.x_idx], dtype=np.float32)

    def step(self, action, debug=False):
        """One env step. ``action`` in {0=abandon, 1=continue}.

        Delta-style rewards (continue : delta*outcome - c; abandon : recovery
        term so cumulative = -t*c). See module docstring for derivation.

        Idempotent at t=T: any action returns done=True with reward 0.
        """
        self.prev_time, self.prev_x_idx = self.time, self.x_idx

        if self.time >= self.T:
            # Already terminal — idempotent done.
            done = True
            reward = 0.0
            self.time += 1
            info = {}
            return self._get_obs(), reward, done, info

        if action == 0:  # abandon
            done = True
            # Recovery cancels both the accumulated project-value change
            # (delta * x_idx) AND the x_1 offset that featurizer.cpt_offset
            # adds back at CPT time, so that CPT input = -t*c (LNW's
            # "abandon → 0 payoff, sunk costs only" semantics).
            x_t = self.x1 + self.x_idx * self.delta
            reward = float(-x_t)
            self.time = self.T  # force terminal

        else:  # action == 1, continue
            outcome = +1 if np.random.random() < self.p else -1
            self.x_idx += outcome
            self.time += 1
            # Reward = project-value change - continuation cost.
            reward = float(self.delta * outcome - self.c)
            done = (self.time >= self.T)

        info = {}
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return self.time, self.x_idx

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()

    def close(self):
        pass
