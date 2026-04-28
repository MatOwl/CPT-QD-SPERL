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


# ========== Algorithm 4 (paper §C.2, Appendix): quantile filter ==========

def filter_quantiles(q, p_filter: float = 0.75) -> np.ndarray:
    """Smooth out crossing / outlier quantiles (paper Algorithm 4).

    Given sorted quantile estimates ``q`` (shape ``(I,)``), treat a quantile
    as "valid" iff the gap leading into it is non-negative and below the
    ``p_filter``-quantile of all gaps. Invalid quantiles are snapped to the
    closer of their nearest valid neighbours.

    ``p_filter=0.75`` matches the paper's default ``filterTresh=[0.75, 1.0]``.
    """
    q = np.asarray(q, dtype=np.float64)
    I = q.size
    if I < 2:
        return q.copy()

    gaps = np.diff(q)  # shape (I-1,)
    # "higher" interpolation matches legacy's np.quantile(..., interpolation='higher')
    d_star = np.quantile(gaps, p_filter, method="higher")

    # Paper Algorithm 4 line 6: bool_j = 1{0 ≤ d_j ≤ d*}. Negative gaps mark
    # crossing quantiles (paper §C.2.3, "Crossing quantiles" paragraph) and
    # MUST be invalidated so the snap-to-neighbour branch fires. Tolerance
    # 1e-6 matches legacy `rerun_GreedySPERL_QR__main.py:1151-1152` for fp
    # noise; tighter than that (e.g. 1e-9) hid genuine crossings in legacy.
    valid_gap = (gaps >= -1e-6) & (gaps <= d_star + 1e-6)

    valid = np.empty(I, dtype=bool)
    valid[0] = valid_gap[0]  # tie first to second (legacy line 1158-1161)
    valid[1:] = valid_gap

    if valid.all():
        return q.copy()
    if not valid.any():
        return q.copy()  # nothing to pin against; give up cleanly

    q_new = q.copy()
    valid_idx = np.flatnonzero(valid)
    for j in range(I):
        if valid[j]:
            continue
        # Nearest valid neighbours via searchsorted over the valid indices.
        pos = np.searchsorted(valid_idx, j)
        lb = valid_idx[pos - 1] if pos > 0 else None
        ub = valid_idx[pos] if pos < valid_idx.size else None
        if lb is None:
            q_new[j] = q[ub]
        elif ub is None:
            q_new[j] = q[lb]
        else:
            dist_lb = abs(q[j] - q[lb])
            dist_ub = abs(q[ub] - q[j])
            q_new[j] = q[lb] if dist_lb < dist_ub else q[ub]
    return q_new


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
                 clip_bounds: tuple[float, float] | None = None,
                 filter_thresh: float | None = None,
                 filter_accept_ratio: float = float("inf"),
                 filter_gate_mode: str = "relative"):
        """``clip_bounds=(lo, hi)`` enables the legacy CumSPERL lbub filter:
        after each per-slice update the quantile row is clipped to
        ``[lo, hi]``. Leave ``None`` for pure QR (current default). Useful
        when porting an env with known reward bounds and the default
        ``lr``/``param_init`` produces runaway quantiles.

        ``filter_thresh`` enables Algorithm 4 (quantile filter). ``None``
        disables filtering entirely; a float in ``[0.5, 1.0]`` is used as the
        gap-quantile threshold (paper: ``filterTresh=0.75``). When enabled,
        each CPT evaluation computes both filtered and unfiltered values;
        the filtered one is used unless the gate (per ``filter_gate_mode``)
        rejects it. Paper §C.2.5: ``filter_accept_ratio=0.5``.

        ``filter_gate_mode`` selects gate semantics:
          - ``"relative"`` (default, matches legacy code): reject filter if
            ``|v_filt - v_ori| / |v_ori| > filter_accept_ratio``.
          - ``"absolute"`` (paper Algorithm 4 line 20 text): reject filter if
            ``|v_filt - v_ori| > filter_accept_ratio``.
        """
        self.featurizer = featurizer
        self.n_actions = n_actions
        self.I = support_size
        self.lr = lr
        self.cpt_params = cpt_params or CPTParams()
        self.clip_bounds = clip_bounds
        self.filter_thresh = filter_thresh
        self.filter_accept_ratio = float(filter_accept_ratio)
        if filter_gate_mode not in ("relative", "absolute"):
            raise ValueError(f"filter_gate_mode must be 'relative' or 'absolute', got {filter_gate_mode!r}")
        self.filter_gate_mode = filter_gate_mode

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

    def _cpt_from_quantiles(self, q_arr: np.ndarray) -> float:
        """Apply Alg 4 filter + acceptance gate, then compute CPT.

        ``q_arr`` is a (I,) array of quantile estimates. Paper §C.2.3 lists
        two motivations for Algorithm 4 — discrete return distributions AND
        crossing quantiles. The crossing-quantile detection (``d_j < 0`` in
        Algorithm 4 line 6) requires the RAW quantile order; pre-sorting
        would hide all crossings and silently disable that branch. We pass
        ``q_arr`` through as-is and let ``compute_CPT(sort=True)`` handle
        ordering for the CPT integral itself (which is order-invariant).
        """
        v_ori = self.cpt_params.compute(list(q_arr), sort=True)
        if self.filter_thresh is None:
            return v_ori
        q_filt = filter_quantiles(q_arr, p_filter=self.filter_thresh)
        v_filt = self.cpt_params.compute(list(q_filt), sort=True)
        # Acceptance gate: fall back to unfiltered if filter moved CPT too much.
        if self.filter_gate_mode == "absolute":
            # Paper Algorithm 4 line 20: |v_filt - v_ori| > treshRatio
            if abs(v_filt - v_ori) > self.filter_accept_ratio:
                return v_ori
        else:  # relative (legacy/refactor default)
            if v_ori == 0.0 or abs((v_filt - v_ori) / v_ori) > self.filter_accept_ratio:
                return v_ori
        return v_filt

    def cpt_value(self, obs, action):
        loc = self.featurizer.loc(obs)
        offset = float(self.featurizer.cpt_offset(obs))
        return self._cpt_from_quantiles(self.qtile_theta[loc, action, :] + offset)

    def cpt_values_all_actions(self, obs):
        loc = self.featurizer.loc(obs)
        offset = float(self.featurizer.cpt_offset(obs))
        return [
            self._cpt_from_quantiles(self.qtile_theta[loc, a, :] + offset)
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
    """Greedy policy driven directly by the critic's CPT-over-quantiles.

    Legacy-aligned semantics (matches `rerun_GreedySPERL_QR__main.py`):
      * Initial commitment at every state: action ``init_action`` (default 0,
        which is "exit" in Barberis). Legacy's ``_get_theta`` initialiser
        does the same — with all qtiles=0, ``argmax([0,0]) = 0`` for every
        state — so before any visit, the policy is uniformly committed to
        action 0. Refactor's earlier ``-1 = uncommitted -> uniform`` branch
        is removed since it changed first-visit dynamics: legacy "stays at
        exit until evidence drives CPT(gamble) > CPT(exit) strictly", and
        the refactor variant "commits randomly on first-visit tie", which
        broke that bootstrap cascade.
      * Tie-break: ``np.argmax`` (deterministic first-index). Legacy uses
        ``np.argmax`` everywhere. Pass ``tie_thresh > 0`` together with an
        ``rng`` to opt into approximate-tie randomization.
      * Sticky update: strict-improve only (paper Algorithm 3 + legacy
        line 765-768). The "first-visit unconditional commit" branch is
        removed — with ``init_action=0`` ``prev_a`` is never -1.
    """

    def __init__(self, featurizer, n_actions, exploration=None,
                 init_action: int = 0):
        self.featurizer = featurizer
        self.n_actions = n_actions
        exploration = exploration or {"type": "eps-greedy", "params": [0.1]}
        self.explore_type = exploration["type"]
        self.explore_param = exploration["params"][0]

        # Legacy-aligned: every state starts committed to ``init_action``
        # (0 = exit in Barberis), matching `policy._get_theta`'s initial
        # ``theta[loc, argmax([0,0])=0] = 1`` pattern.
        self.greedy_action = np.full(featurizer.n_states, int(init_action),
                                     dtype=np.int64)

    def predict(self, obs, deterministic=False):
        loc = self.featurizer.loc(obs)
        a_star = int(self.greedy_action[loc])

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

    def update_from_critic_values(self, obs, cpt_values,
                                  sticky: bool = False,
                                  tie_thresh: float = 0.0,
                                  rng: np.random.Generator | None = None):
        """Greedy policy update.

        ``sticky=False``: plain argmax, overwrites any previous action.

        ``sticky=True`` (paper Algorithm 3 / legacy line 765-768): keep the
        previous action unless the new argmax is *strictly* better in CPT
        value. With the legacy-aligned ``init_action=0`` constructor, every
        state starts at action 0, so the strict-improve check is the only
        path to flip — there is no "first-visit unconditional commit" branch.

        Default tie-break is ``np.argmax`` (deterministic first-index, matches
        legacy). Pass ``tie_thresh > 0`` with an ``rng`` to randomize among
        approximate-tie candidates instead.
        """
        loc = self.featurizer.loc(obs)
        cpt_values = np.asarray(cpt_values, dtype=np.float64)

        if tie_thresh > 0 and rng is not None:
            v_max = cpt_values.max()
            candidates = np.flatnonzero(cpt_values >= v_max - tie_thresh)
            new_a = int(rng.choice(candidates)) if candidates.size > 1 \
                else int(candidates[0])
        else:
            # Legacy default: deterministic ``np.argmax`` (first-index ties).
            new_a = int(np.argmax(cpt_values))

        if sticky:
            prev_a = int(self.greedy_action[loc])
            if cpt_values[new_a] > cpt_values[prev_a]:
                self.greedy_action[loc] = new_a
            # else: keep prev_a (strict-improve sticky)
        else:
            self.greedy_action[loc] = new_a


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
                 critic_clip_bounds: tuple[float, float] | None = None,
                 # Paper Alg 3 — consistent tie-break / is-better guard:
                 sticky_policy: bool = False,
                 tie_thresh: float = 0.0,
                 # Paper Alg 4 — quantile filter:
                 filter_thresh: float | None = None,
                 filter_accept_ratio: float = float("inf"),
                 filter_gate_mode: str = "relative"):
        self.env = env
        self.featurizer = featurizer
        self.horizon = getattr(env, "horizon", None) or env.T
        self.target_type = target_type
        self.order = order
        self.gamma = discount
        self.empty_memory = empty_memory
        self.sticky_policy = sticky_policy
        self.tie_thresh = float(tie_thresh)

        if seed is not None:
            np.random.seed(seed)
            self.env.action_space.seed(seed)
        self._rng = np.random.default_rng(seed)

        self.critic = QRCritic(
            featurizer, env.action_space.n,
            support_size=support_size, lr=critic_lr, cpt_params=cpt_params,
            clip_bounds=critic_clip_bounds,
            filter_thresh=filter_thresh,
            filter_accept_ratio=filter_accept_ratio,
            filter_gate_mode=filter_gate_mode,
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
        """Greedy-SPERL TD targets: r + gamma * opponent-action quantiles.

        Legacy-aligned: at the terminal step ``t == self.horizon`` (=T) we
        update qtile for ALL actions with the same target, mirroring
        ``rerun_GreedySPERL_QR__main.py:733-740``. Rationale: at terminal,
        the value is action-independent (reward=0, no further decisions),
        so visiting one action is enough to fix all of them. Refactor's
        earlier "only the sampled action" code left unvisited terminal
        actions at the qtile=0 default which is fine via cpt_offset, but
        legacy's broadcast version yields a tighter QR sample shape.
        """
        n_actions = self.env.action_space.n
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

                if t == self.horizon:
                    # Legacy line 733-740: broadcast target to ALL actions at t=T.
                    data = {(loc_s, a): targets for a in range(n_actions)}
                else:
                    data = {(loc_s, action): targets}
                self.critic.update(data)

                # Policy improvement at this state (paper Alg 3 when sticky=True)
                cpt_vals = self.critic.cpt_values_all_actions(state)
                self.policy.update_from_critic_values(
                    state, cpt_vals,
                    sticky=self.sticky_policy,
                    tie_thresh=self.tie_thresh,
                    rng=self._rng,
                )

    def _train_critic_MC(self, aggr_episodes):
        """MC targets: Monte-Carlo accumulated reward from t onward (same for
        all quantiles — a crude but workable initialisation baseline).

        Legacy-aligned t=T broadcast like ``_train_critic_TD``.
        """
        n_actions = self.env.action_space.n
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
                if t == self.horizon:
                    self.critic.update({(loc_s, a): targets for a in range(n_actions)})
                else:
                    self.critic.update({(loc_s, action): targets})

                cpt_vals = self.critic.cpt_values_all_actions(state)
                self.policy.update_from_critic_values(
                    state, cpt_vals,
                    sticky=self.sticky_policy,
                    tie_thresh=self.tie_thresh,
                    rng=self._rng,
                )

    # -------- evaluation --------
    def evaluate(self, n_eval_eps):
        rewards = []
        for _ in range(n_eval_eps):
            state = self.env.reset()
            offset = float(self.featurizer.cpt_offset(state))
            total = 0.0
            while True:
                a_probs = self.policy.predict(state, deterministic=True)
                a = int(np.argmax(a_probs))
                state, r, done, _ = self.env.step(a)
                total += r
                if done:
                    break
            # CPT input = init-state offset + cumulative reward (Barberis & OptEx
            # have offset=0 at reset; LNW has offset=x_1 -> matches paper-eval
            # rollout_cpt_from_state semantics).
            rewards.append(total + offset)
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
