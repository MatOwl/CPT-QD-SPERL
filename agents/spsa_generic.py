"""Env-agnostic SPSA baseline for CPT policy optimisation.

Mirrors the precommitment-only SPSA actor used in
``rerun_SPSA_QRMC__main.py`` (Prashant / Bhatnagar style), but decoupled from
Barberis-specific featurising. Each SPSA gradient step consumes two batches of
episodes (one at ``theta + c*Delta``, one at ``theta - c*Delta``) and updates
``theta`` with ``(J_plus - J_minus) / (2c) * Delta^{-1}``.

This is a **baseline**: it only optimises the CPT value at the reset state
s_0, producing a precommitment policy — not a time-consistent SPE.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import scipy.special as sp

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)
if ".." not in sys.path:
    sys.path.append("..")

from lib.cpt import CPTParams


class SPSAActor:
    """Tabular softmax policy with ± simultaneous perturbation.

    ``theta[loc, a]`` is clipped to ``[theta_min, theta_max]``. Perturbations
    are refreshed each gradient step (every 2nd batch).
    """

    def __init__(self, featurizer, n_actions, beta=1.0,
                 step_size=5.0, perturb_const=1.9,
                 theta_min=0.1, theta_max=2.0, rng=None):
        self.featurizer = featurizer
        self.n_actions = n_actions
        self.beta = beta

        self.ss_const = step_size
        self.pc_const = perturb_const
        self.theta_min = theta_min
        self.theta_max = theta_max

        self.rng = rng if rng is not None else np.random.default_rng()

        self.theta = self.rng.uniform(
            theta_min, theta_max, size=(featurizer.n_states, n_actions)
        )
        self.init_theta = self.theta.copy()

        # State machine: is_plus flips between True/False across batches.
        # When is_plus is True we evaluate at theta + c*Delta; when False at
        # theta - c*Delta. Delta/step/c are refreshed after each full +/- pair.
        self.is_plus = True
        self.step_idx = 0
        self.step_size = step_size
        self.perturb_c = perturb_const
        self.delta = self._sample_delta()
        self.J_pair = [0.0, 0.0]  # [J_plus, J_minus]

    # ---- prediction ----
    def _theta_eval(self):
        """Perturbed theta used for rollouts."""
        sign = 1.0 if self.is_plus else -1.0
        return self.theta + sign * self.perturb_c * self.delta

    def predict(self, obs, deterministic=False):
        theta_use = self.theta if deterministic else self._theta_eval()
        loc = self.featurizer.loc(obs)
        logits = theta_use[loc, :] * self.beta
        return sp.softmax(logits)

    # ---- perturbation housekeeping ----
    def _sample_delta(self):
        return self.rng.choice([-1.0, 1.0], size=self.theta.shape)

    def _refresh_schedule(self):
        """Spall (A3)-compatible step/perturb schedule."""
        n = self.step_idx + 1
        self.step_size = self.ss_const / n
        self.perturb_c = self.pc_const / (n ** 0.101)

    def _clip(self):
        np.clip(self.theta, self.theta_min, self.theta_max, out=self.theta)

    # ---- gradient step ----
    def record_batch_value(self, J):
        """Log J_+ or J_- from the just-finished batch, then toggle."""
        idx = 0 if self.is_plus else 1
        self.J_pair[idx] = float(J)

        if not self.is_plus:
            # both legs done -> update theta
            J_plus, J_minus = self.J_pair
            grad = (J_plus - J_minus) / (2.0 * self.perturb_c) / self.delta
            if self.ss_const > 1e-16:
                self.theta = self.theta + self.step_size * grad
                self._clip()
            # refresh for next pair
            self.step_idx += 1
            self._refresh_schedule()
            self.delta = self._sample_delta()

        self.is_plus = not self.is_plus


class SPSAAgent:
    """Precommitment SPSA agent — optimises CPT at s_0 via perturbed rollouts."""

    def __init__(self, env, featurizer, cpt_params: CPTParams,
                 beta=1.0, step_size=5.0, perturb_const=1.9,
                 theta_min=0.1, theta_max=2.0, seed=None):
        self.env = env
        self.featurizer = featurizer
        self.cpt_params = cpt_params
        self.horizon = getattr(env, "horizon", None) or env.T

        self.rng = np.random.default_rng(seed)
        if seed is not None:
            np.random.seed(seed)
            self.env.action_space.seed(seed)

        self.policy = SPSAActor(
            featurizer, env.action_space.n,
            beta=beta, step_size=step_size, perturb_const=perturb_const,
            theta_min=theta_min, theta_max=theta_max, rng=self.rng,
        )
        self.stats = {"mean_rewards": [], "std_rewards": [], "cpt_rewards": []}

    # ---- rollout under the current (perturbed) policy ----
    def _rollout_return(self):
        state = self.env.reset()
        offset = float(self.featurizer.cpt_offset(state))
        total = 0.0
        for _ in itertools.count():
            probs = self.policy.predict(state)
            a = int(self.rng.choice(self.n_actions, p=probs))
            state, r, done, _ = self.env.step(a)
            total += r
            if done:
                break
        # CPT input = initial-state offset + cumulative reward (mirrors SPERL
        # QR critic; for Barberis & OptEx the offset is 0, for LNW it's x_t).
        return total + offset

    # ---- deterministic (unperturbed) evaluation ----
    def evaluate(self, n_eval_eps):
        rewards = []
        for _ in range(n_eval_eps):
            state = self.env.reset()
            offset = float(self.featurizer.cpt_offset(state))
            total = 0.0
            for _ in itertools.count():
                probs = self.policy.predict(state, deterministic=True)
                a = int(np.argmax(probs))
                state, r, done, _ = self.env.step(a)
                total += r
                if done:
                    break
            rewards.append(total + offset)
        rewards = np.asarray(rewards)
        return rewards.mean(), rewards.std(), self.cpt_params.compute(list(rewards))

    @property
    def n_actions(self):
        return self.env.action_space.n

    def evaluate_under_policy(self, policy_fn, n_eval_eps):
        """Evaluate an external ``policy_fn(time, state) -> action`` (for
        benchmarking against the SPE oracle)."""
        rewards = []
        for _ in range(n_eval_eps):
            state = self.env.reset()
            offset = float(self.featurizer.cpt_offset(state))
            total = 0.0
            t = 0
            while True:
                a = int(policy_fn(t, state))
                state, r, done, _ = self.env.step(a)
                total += r
                t += 1
                if done:
                    break
            rewards.append(total + offset)
        rewards = np.asarray(rewards)
        return rewards.mean(), rewards.std(), self.cpt_params.compute(list(rewards))

    # ---- main loop ----
    def learn(self, n_train_eps, n_batch=50, n_eval_eps=200,
              eval_freq=100, verbose=1):
        """One ``i_ep`` produces one rollout; every ``n_batch`` rollouts the
        SPSA actor consumes the batch CPT value as a J_+/J_- leg."""
        returns_this_batch = []

        for i_ep in range(1, n_train_eps + 2):
            if (i_ep - 1) % eval_freq == 0:
                mean_r, std_r, cpt_r = self.evaluate(n_eval_eps)
                self.stats["mean_rewards"].append(mean_r)
                self.stats["std_rewards"].append(std_r)
                self.stats["cpt_rewards"].append(cpt_r)
                if verbose:
                    print(f"[ep {i_ep}/{n_train_eps}] "
                          f"mean={mean_r:.4f} std={std_r:.4f} cpt={cpt_r:.4f}")

            R = self._rollout_return()
            returns_this_batch.append(R)

            if len(returns_this_batch) == n_batch:
                J = self.cpt_params.compute(returns_this_batch)
                self.policy.record_batch_value(J)
                returns_this_batch = []

        return self
