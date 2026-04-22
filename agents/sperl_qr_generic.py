"""Env-agnostic Greedy-SPERL with QR critic.

Generalises ``rerun_GreedySPERL_QR__main.py`` by:
  * depending on an injected ``Featurizer`` (int loc + hashable key) instead of
    the hard-coded ``barberisFeaturize``;
  * storing qtile values as a dense ``ndarray[n_states, nA, I]`` indexed
    directly by ``loc`` (no per-predict one-hot allocation);
  * lazy state enumeration — ``vf_dict_init`` no longer preallocates the
    full Barberis state set; (state_key, action) entries are added on first
    visit.

Barberis-specific analytical oracles (``assign_GainExit`` / ``assign_Precomm``
/ ``compute_SPE``) are **not** included here; add them alongside per-env SPE
solvers when needed.
"""

from __future__ import annotations

import collections
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


class QRCritic:
    """Quantile-regression CPT critic with tabular FA.

    ``qtile_theta[loc, a, i]`` holds the i-th quantile of return under taking
    action ``a`` at the state indexed by ``loc``.

    ------------------------------------------------------------------
    Env-sensitive tricks (not pure QR — may need retuning per env!)
    ------------------------------------------------------------------
    1. **First-visit mean init** (``update``): all I quantiles are overwritten
       with the batch mean on first visit to (loc, a). Warm-starts away from
       ``param_init`` but couples the critic to the reward scale — on envs
       with O(1) rewards vs O(1e-2) rewards (Barberis vs OptEx) the same
       ``lr`` behaves very differently.
    2. **MC targets collapse all quantiles to the scalar return-to-go**
       (see ``GreedySPERL._train_critic_MC``). That's point-estimate
       regression wearing a QR coat — distributional info is lost. TD is
       the "real" QR path; MC is mainly an init baseline.
    3. **Terminal-step targets collapse to the realised reward** (TD path).
       Fine when terminal value is deterministic; revisit if you add a
       stochastic terminal reward.
    4. **Greedy-SPERL opponent uses deterministic argmax** on the learned
       policy when bootstrapping (``_train_critic_TD``). Correct for SPE
       induction, wrong for precommitment critics.

    Legacy tricks from the reference CumSPERL code that are **not** ported
    here (may matter for OptEx convergence):
      - lower/upper bound (lbub) filtering on quantile updates;
      - a more aggressive first-visit scheme.

    If you change the env, start by inspecting reward scale vs ``lr`` and
    ``param_init``, and consider whether MC targets are good enough or you
    need real per-quantile samples.
    """

    def __init__(self, featurizer, n_actions, support_size=50, lr=0.1,
                 param_init=0.0, cpt_params: CPTParams | None = None,
                 clip_bounds: tuple[float, float] | None = None):
        """``clip_bounds=(lo, hi)`` enables the legacy CumSPERL lbub filter:
        after each per-slice update the quantile row is clipped to
        ``[lo, hi]``. Leave ``None`` for pure QR (current default). Useful
        when porting an env with known reward bounds and the default
        ``lr``/``param_init`` produces runaway quantiles."""
        self.featurizer = featurizer
        self.n_actions = n_actions
        self.I = support_size
        self.lr = lr
        self.cpt_params = cpt_params or CPTParams()
        self.clip_bounds = clip_bounds

        shape = (featurizer.n_states, n_actions, support_size)
        self.qtile_theta = np.full(shape, param_init, dtype=np.float64)
        # Track (loc, a) pairs that have been updated at least once; used to
        # trigger first-visit mean-init below.
        self._visited = np.zeros((featurizer.n_states, n_actions), dtype=bool)

    # ---- prediction ----
    def qtile(self, obs, action, i=None):
        loc = self.featurizer.loc(obs)
        if i is None:
            return self.qtile_theta[loc, action, :]
        return float(self.qtile_theta[loc, action, i])

    def cpt_value(self, obs, action):
        loc = self.featurizer.loc(obs)
        return self.cpt_params.compute(
            list(self.qtile_theta[loc, action, :]), sort=True
        )

    def cpt_values_all_actions(self, obs):
        loc = self.featurizer.loc(obs)
        return [
            self.cpt_params.compute(list(self.qtile_theta[loc, a, :]), sort=True)
            for a in range(self.n_actions)
        ]

    # ---- update ----
    def update(self, data: dict):
        """``data``: {(loc, action): [quantile_targets...]}.

        Vectorised per-slice update — no full-matrix allocation, no
        Python-level loop over quantile indices. Previous version allocated
        ``ndarray[n_states, nA, I]`` zeros on every call, which dominated
        training cost on OptEx-size state spaces.
        """
        # Pre-compute the midpoint τ̃_i = (i/I + (i+1)/I)/2 = (2i+1)/(2I)
        if not hasattr(self, "_mids"):
            self._mids = (np.arange(self.I, dtype=np.float64) * 2 + 1) / (2 * self.I)

        mids = self._mids  # shape (I,)

        for (loc, a), targets in data.items():
            targets = np.asarray(targets, dtype=np.float64)  # shape (M,)

            if not self._visited[loc, a]:
                self.qtile_theta[loc, a, :] = targets.mean()
                self._visited[loc, a] = True

            current = self.qtile_theta[loc, a, :]  # shape (I,), view

            # cmp[i, k] = 1 if targets[k] < current[i]; shape (I, M)
            cmp = (targets[None, :] < current[:, None]).astype(np.float64)
            # grad_i = sum_k (mids[i] - cmp[i, k])
            grad_slice = mids * targets.size - cmp.sum(axis=1)

            # In-place per-slice update
            current += self.lr * grad_slice

            if self.clip_bounds is not None:
                np.clip(current, self.clip_bounds[0], self.clip_bounds[1],
                        out=current)


class GreedyPolicy:
    """Greedy policy driven directly by the critic's CPT-over-quantiles."""

    def __init__(self, featurizer, n_actions, exploration=None):
        self.featurizer = featurizer
        self.n_actions = n_actions
        exploration = exploration or {"type": "eps-greedy", "params": [0.1]}
        self.explore_type = exploration["type"]
        self.explore_param = exploration["params"][0]

        # Per-state greedy action index. -1 = undecided -> uniform.
        self.greedy_action = -np.ones(featurizer.n_states, dtype=np.int64)

    def predict(self, obs, deterministic=False):
        loc = self.featurizer.loc(obs)
        a_star = self.greedy_action[loc]

        if a_star < 0:
            # Uniform until critic produces a preference
            return np.full(self.n_actions, 1.0 / self.n_actions)

        base = np.zeros(self.n_actions)
        base[a_star] = 1.0

        if deterministic:
            return base

        if self.explore_type == "softmax":
            scores = np.zeros(self.n_actions)
            scores[a_star] = 1.0
            return sp.softmax(scores * self.explore_param)
        # eps-greedy
        eps = self.explore_param
        return base * (1 - eps) + np.full(self.n_actions, eps / self.n_actions)

    def update_from_critic_values(self, obs, cpt_values):
        loc = self.featurizer.loc(obs)
        self.greedy_action[loc] = int(np.argmax(cpt_values))


class GreedySPERL:
    """Greedy-SPERL with QR critic, env-agnostic.

    Supports MC and TD critic targets; 'bwd' (backward) player iteration is the
    canonical SPE-inducing order.
    """

    Transition = collections.namedtuple(
        "Transition", ["state", "action", "reward", "next_state", "done"]
    )

    def __init__(self, env, featurizer, cpt_params: CPTParams,
                 support_size=50, critic_lr=0.1,
                 exploration=None, target_type="TD", order="bwd",
                 discount=1.0, seed=None, empty_memory=True,
                 critic_clip_bounds: tuple[float, float] | None = None):
        self.env = env
        self.featurizer = featurizer
        self.horizon = getattr(env, "horizon", None) or env.T
        self.target_type = target_type
        self.order = order
        self.gamma = discount
        self.empty_memory = empty_memory

        if seed is not None:
            np.random.seed(seed)
            self.env.action_space.seed(seed)

        self.critic = QRCritic(
            featurizer, env.action_space.n,
            support_size=support_size, lr=critic_lr, cpt_params=cpt_params,
            clip_bounds=critic_clip_bounds,
        )
        self.policy = GreedyPolicy(
            featurizer, env.action_space.n, exploration=exploration,
        )

        # Per-step bookkeeping (sparse: only visited (key, action) pairs)
        self.visit_counts = collections.defaultdict(int)
        self.stats = {"mean_rewards": [], "std_rewards": [], "cpt_rewards": []}

        if order == "fwd":
            self.player_set = list(range(self.horizon + 1))
        elif order == "bwd":
            self.player_set = list(range(self.horizon, -1, -1))
        else:
            raise ValueError(f"order must be 'fwd' or 'bwd', got {order!r}")

        self.n_batch = None
        self.n_eval_eps = None

    # -------- rollout helpers --------
    def _rollout(self):
        """One episode, returning list[Transition | None] of length horizon+1."""
        state = self.env.reset()
        ep = [None] * (self.horizon + 1)
        for t in itertools.count():
            if t >= self.horizon + 1:
                break
            action_probs = self.policy.predict(state)
            action = np.random.choice(self.env.action_space.n, p=action_probs)
            next_state, reward, done, _ = self.env.step(action)
            self.visit_counts[(self.featurizer.key(state), action)] += 1
            ep[t] = self.Transition(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
        return ep

    # -------- critic training --------
    def _train_critic_TD(self, aggr_episodes):
        """Greedy-SPERL TD targets: r + gamma * opponent-action quantiles."""
        for m in range(self.n_batch):
            for t in self.player_set:
                tr = aggr_episodes[m][t]
                if tr is None:
                    continue
                state, action, reward, next_state, done = tr
                loc_s = self.featurizer.loc(state)

                if done:
                    # Terminal: quantiles collapse to the realised reward
                    targets = [reward] * self.critic.I
                else:
                    opp_probs = self.policy.predict(next_state, deterministic=True)
                    opp_action = int(np.argmax(opp_probs))
                    next_q = self.critic.qtile(next_state, opp_action)
                    targets = [reward + self.gamma * q for q in next_q]

                data = {(loc_s, action): targets}
                self.critic.update(data)

                # Policy improvement at this state
                cpt_vals = self.critic.cpt_values_all_actions(state)
                cur_best = int(np.argmax(cpt_vals))
                self.policy.update_from_critic_values(state, cpt_vals)

    def _train_critic_MC(self, aggr_episodes):
        """MC targets: Monte-Carlo accumulated reward from t onward (same for
        all quantiles — a crude but workable initialisation baseline)."""
        for m in range(self.n_batch):
            accum = 0.0
            # compute returns-to-go going backward
            returns_to_go = [None] * (self.horizon + 1)
            for t in range(self.horizon, -1, -1):
                tr = aggr_episodes[m][t]
                if tr is None:
                    continue
                accum = tr.reward + self.gamma * accum
                returns_to_go[t] = accum

            for t in self.player_set:
                tr = aggr_episodes[m][t]
                if tr is None:
                    continue
                state, action, *_ = tr
                loc_s = self.featurizer.loc(state)
                targets = [returns_to_go[t]] * self.critic.I
                self.critic.update({(loc_s, action): targets})

                cpt_vals = self.critic.cpt_values_all_actions(state)
                self.policy.update_from_critic_values(state, cpt_vals)

    # -------- evaluation --------
    def evaluate(self, n_eval_eps):
        rewards = []
        for _ in range(n_eval_eps):
            state = self.env.reset()
            total = 0.0
            while True:
                a_probs = self.policy.predict(state, deterministic=True)
                a = int(np.argmax(a_probs))
                state, r, done, _ = self.env.step(a)
                total += r
                if done:
                    break
            rewards.append(total)
        rewards = np.asarray(rewards)
        cpt_val = self.critic.cpt_params.compute(list(rewards))
        return rewards.mean(), rewards.std(), cpt_val

    def evaluate_under_policy(self, policy_fn, n_eval_eps):
        """Evaluate an external deterministic ``policy_fn(time, state) -> action``.

        Used to benchmark learned policies against a pre-computed SPE oracle
        (see ``lib.envs.optex_spe.SPEOracle``).
        """
        rewards = []
        for _ in range(n_eval_eps):
            state = self.env.reset()
            total = 0.0
            t = 0
            while True:
                a = int(policy_fn(t, state))
                state, r, done, _ = self.env.step(a)
                total += r
                t += 1
                if done:
                    break
            rewards.append(total)
        rewards = np.asarray(rewards)
        cpt_val = self.critic.cpt_params.compute(list(rewards))
        return rewards.mean(), rewards.std(), cpt_val

    # -------- main loop --------
    def learn(self, n_train_eps, n_batch=5, n_eval_eps=100, eval_freq=50, verbose=1):
        self.n_batch = n_batch
        self.n_eval_eps = n_eval_eps

        aggr = None
        m_id = 0

        for i_ep in range(1, n_train_eps + 2):

            if (i_ep - 1) % eval_freq == 0:
                mean_r, std_r, cpt_r = self.evaluate(n_eval_eps)
                self.stats["mean_rewards"].append(mean_r)
                self.stats["std_rewards"].append(std_r)
                self.stats["cpt_rewards"].append(cpt_r)
                if verbose:
                    print(f"[ep {i_ep}/{n_train_eps}] "
                          f"mean={mean_r:.4f} std={std_r:.4f} cpt={cpt_r:.4f}")

            if (i_ep - 1) % n_batch == 0:
                aggr = [[None] * (self.horizon + 1) for _ in range(n_batch)]
                m_id = 0

            aggr[m_id] = self._rollout()

            if m_id == n_batch - 1:
                if self.target_type == "TD":
                    self._train_critic_TD(aggr)
                elif self.target_type == "MC":
                    self._train_critic_MC(aggr)
                else:
                    raise ValueError(f"Unknown target_type: {self.target_type}")

            m_id += 1

        return self
