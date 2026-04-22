"""Paper-style evaluation: Policy Error, Value Error, Optimality, Social
Welfare, as defined in Section 4.1–4.2 of MSci_MANUSCRIPT.

For each state x in ``featurizer.iter_states()``:
  V^pi(x) = CPT_{rollouts from x under pi}[cumulative reward]
  pi(x)   = deterministic greedy action under pi at x

Aggregated metrics (per-seed):
  Policy Error  = sum_x |1{pi_learned(x) != pi_spe(x)}| * (indicator mean)
                  Paper's form: sum_x |mu[pi_tilde(x)] - mu[pi_hat(x)]|
                  — for single-seed, the |mu[..]| reduces to the 0/1 policy
                  disagreement indicator.
  Value Error   = sum_x |V^pi_learned(x) - V^pi_spe(x)|
  Optimality    = V^pi(x_0)
  Social Welfare= sum_x V^pi(x)
"""

from __future__ import annotations

from typing import Callable, Iterable

import inspect
import itertools
import numpy as np

from lib.cpt import CPTParams


def rollout_cpt_from_state(
    env,
    policy_fn: Callable,
    init_state,
    n_eps: int,
    cpt_params: CPTParams,
    reset_kwargs_for_state: Callable | None = None,
):
    """Run ``n_eps`` episodes starting from ``init_state`` under ``policy_fn``
    and return the CPT of total rewards.

    ``policy_fn(time, state) -> int``.
    ``reset_kwargs_for_state(init_state) -> dict`` produces the kwargs that
    ``env.reset`` needs to land at ``init_state`` (Barberis and OptEx differ in
    reset signature).
    """
    rewards = []
    for _ in range(n_eps):
        if reset_kwargs_for_state is not None:
            state = env.reset(**reset_kwargs_for_state(init_state))
        else:
            state = env.reset()
        t = _state_time(init_state, env)
        total = 0.0
        for _ in itertools.count():
            a = int(policy_fn(t, state))
            state, r, done, _ = env.step(a)
            total += r
            t += 1
            if done:
                break
        rewards.append(total)
    return cpt_params.compute(list(rewards))


def _state_time(state, env):
    """Extract the current time-step from an env state for different envs."""
    # Barberis: state = (t, z)
    if hasattr(env, "bet") and hasattr(env, "T") and len(state) == 2:
        return int(state[0])
    # OptEx: state = (logRet_bin, remain_bin, X_bin); t = N - remain_bin * N / n_bins_N
    if hasattr(env, "remaining_num_trade"):
        try:
            remain_bin = int(state[1])
            return int(env.N - remain_bin / env.n_bins_N * env.N)
        except Exception:
            return 0
    return 0


def barberis_reset_kwargs(init_state):
    t, z = init_state
    return {"init_time": int(t), "init_wealth": int(z)}


def optex_reset_kwargs(init_state):
    return {"init_state": tuple(int(v) for v in init_state)}


def reset_kwargs_builder(env):
    """Return the appropriate reset-kwargs builder for ``env``."""
    if hasattr(env, "bet") and hasattr(env, "T"):
        return barberis_reset_kwargs
    return optex_reset_kwargs


# ------------------ policy helpers ------------------

def learned_policy_fn(agent):
    """Wrap ``GreedySPERL`` / ``SPSAAgent`` as ``(t, state) -> action``."""
    def pi(_t, state):
        probs = agent.policy.predict(state, deterministic=True)
        return int(np.argmax(probs))
    return pi


# ------------------ paper metrics ------------------

def compute_paper_metrics(
    agent,
    env,
    featurizer,
    learned_pi: Callable,
    reference_pi: Callable,
    cpt_params: CPTParams,
    n_eps_per_state: int = 200,
    states: Iterable | None = None,
    reset_kwargs_for_state: Callable | None = None,
):
    """Compute the 4 paper metrics for a single seed/run.

    Returns dict with keys:
      policy_disagree_total : int, sum_x 1{pi_tilde(x) != pi_hat(x)}
      value_error_total     : float, sum_x |V_tilde(x) - V_hat(x)|
      optimality            : float, V_tilde(x_0)
      social_welfare        : float, sum_x V_tilde(x)
      per_state             : list of dicts with detailed entries
    """
    if states is None:
        if not hasattr(featurizer, "iter_states"):
            raise ValueError("featurizer has no iter_states method")
        sig = inspect.signature(featurizer.iter_states)
        if "env" in sig.parameters:
            states = list(featurizer.iter_states(env))
        else:
            states = list(featurizer.iter_states())

    if reset_kwargs_for_state is None:
        reset_kwargs_for_state = reset_kwargs_builder(env)

    per_state = []
    v_tilde_total = 0.0
    v_hat_total = 0.0
    value_err_total = 0.0
    policy_disagree = 0
    v_tilde_x0 = None

    x0_raw = env.reset()
    x0 = tuple(x0_raw.tolist()) if hasattr(x0_raw, "tolist") else tuple(x0_raw)

    for x in states:
        probs_tilde = agent.policy.predict(_to_obs(x, env), deterministic=True)
        a_tilde = int(np.argmax(probs_tilde))
        a_hat = int(reference_pi(_state_time(x, env), _to_obs(x, env)))

        v_tilde = rollout_cpt_from_state(
            env, learned_pi, x, n_eps_per_state, cpt_params, reset_kwargs_for_state
        )
        v_hat = rollout_cpt_from_state(
            env, reference_pi, x, n_eps_per_state, cpt_params, reset_kwargs_for_state
        )

        per_state.append({
            "state": x,
            "a_tilde": a_tilde,
            "a_hat": a_hat,
            "v_tilde": v_tilde,
            "v_hat": v_hat,
        })
        value_err_total += abs(v_tilde - v_hat)
        v_tilde_total += v_tilde
        v_hat_total += v_hat
        if a_tilde != a_hat:
            policy_disagree += 1

        # Python int/float equality makes (0, 0) == (0.0, 0.0) → True,
        # so this handles both Barberis (int tuple) and OptEx (int vector)
        # without a type-specific fallback.
        if tuple(x) == x0:
            v_tilde_x0 = v_tilde

    return {
        "policy_disagree_total": policy_disagree,
        "value_error_total": value_err_total,
        "optimality": v_tilde_x0,
        "social_welfare": v_tilde_total,
        "social_welfare_spe": v_hat_total,
        "per_state": per_state,
        "n_states": len(per_state),
    }


def _to_obs(x, env):
    """Convert a state tuple back to the observation format env expects."""
    if hasattr(env, "bet") and len(x) == 2:
        return np.array([x[0], x[1]], dtype=np.float32)
    return np.asarray(x, dtype=np.intc)
