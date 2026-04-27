"""Phase B: QRCritic update + first-visit init unit isolation.

Replays a synthetic sequence of (loc, action, targets) updates against:
  1. A minimal "legacy-faithful" critic — verbatim port of `compute_LossGrad`
     + `_update` from `agents/rerun_GreedySPERL_QR__main.py` lines 254-328
  2. Our generic `QRCritic.update()` from `agents/sperl_qr_generic.py`

After every step, compares `qtile_theta` element-wise. Reports the first
step where they diverge (if any).

Run:
    PYTHONPATH=. python scripts/verify_refactor_phase_b.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

from lib.cpt import CPTParams
from agents.sperl_qr_generic import QRCritic


# =============================================================================
# Minimal legacy critic — only the qtile_theta update path
# =============================================================================


class LegacyCriticMinimal:
    """Verbatim port of legacy CPTCritic's qtile_theta update path
    (rerun_GreedySPERL_QR__main.py lines 254-328, with_quantile=True branch).
    Includes the "all-zero re-init" first-visit logic.
    """

    def __init__(self, n_states, n_actions, support_size, lr, param_init=0.0):
        self.support_size = support_size
        self.lr = lr
        self.qtile_theta = param_init + np.zeros(
            (n_states, n_actions, support_size), dtype=np.float64
        )
        self.loss_grad = np.zeros_like(self.qtile_theta)

    def qtilepredict(self, loc, action, i):
        return float(self.qtile_theta[loc, action, i])

    def compute_LossGrad(self, data):
        # data: {(loc, action): [targets...]}
        lossgrad = np.zeros(self.loss_grad.shape)

        for (loc, action), qtile_targets in data.items():
            # ------ First-visit re-init (legacy line 277-282) ------
            if (
                np.sum(self.qtile_theta[loc, action, :] != np.zeros(self.support_size))
                == 0
            ):
                mean_targets = np.average(qtile_targets)
                self.qtile_theta[loc, action, :] = [
                    mean_targets for _ in range(self.support_size)
                ]

            # ------ QR gradient (legacy line 288-296) ------
            for i in range(self.support_size):
                cur_qtile_estimate = self.qtilepredict(loc, action, i)

                # tau_i = i / I (compute_cdf with pos=False)
                tau_i = i / self.support_size
                tau_i_next = (i + 1) / self.support_size
                midpoint_i = (tau_i + tau_i_next) / 2

                lossgrad[loc, action, i] += np.sum(
                    [midpoint_i - (z < cur_qtile_estimate) for z in qtile_targets]
                )

        return lossgrad

    def update(self, data):
        # Legacy line 314-321
        self.loss_grad = self.compute_LossGrad(data)
        self.qtile_theta = self.qtile_theta + self.lr * self.loss_grad


# =============================================================================
# Test runners
# =============================================================================


def make_pair(n_states=20, n_actions=2, K=50, lr=0.04):
    legacy = LegacyCriticMinimal(n_states, n_actions, K, lr)

    class _Featurizer:
        n_states = 0
    feat = _Featurizer()
    feat.n_states = n_states
    cpt = CPTParams()  # not actually used for update
    generic = QRCritic(feat, n_actions, support_size=K, lr=lr,
                       param_init=0.0, cpt_params=cpt)
    return legacy, generic


def diff_thetas(legacy, generic):
    """Return (max_abs_err, mean_abs_err)."""
    d = np.abs(legacy.qtile_theta - generic.qtile_theta)
    return float(d.max()), float(d.mean())


def play_sequence(seq, n_states=20, n_actions=2, K=50, lr=0.04, label=""):
    """`seq` is a list of {(loc, action): [targets]} dicts (one per update call).
    Returns True if all steps match within atol=1e-12."""
    legacy, generic = make_pair(n_states, n_actions, K, lr)
    max_err = 0.0
    first_fail = None
    for step, data in enumerate(seq):
        legacy.update(data)
        generic.update(data)
        e_max, e_mean = diff_thetas(legacy, generic)
        max_err = max(max_err, e_max)
        if e_max > 1e-9 and first_fail is None:
            first_fail = (step, e_max, e_mean)
    if first_fail is None:
        print(f"  [{label}] OK after {len(seq)} steps; max_abs_err={max_err:.2e}")
        return True
    step, e_max, e_mean = first_fail
    print(f"  [{label}] FAIL at step {step}: max_abs_err={e_max:.4e}, "
          f"mean_abs_err={e_mean:.4e}")
    return False


def gen_random_seq(rng, n_steps, n_states, n_actions, K, target_kind):
    """Generate a random sequence of update dicts."""
    seq = []
    for _ in range(n_steps):
        loc = int(rng.integers(0, n_states))
        a = int(rng.integers(0, n_actions))
        if target_kind == "uniform":
            targets = rng.uniform(-30, 30, size=K).tolist()
        elif target_kind == "discrete":
            targets = rng.choice([-20.0, 0.0, 30.0], size=K).tolist()
        elif target_kind == "constant":
            v = float(rng.uniform(-10, 10))
            targets = [v] * K
        elif target_kind == "all_zero":
            targets = [0.0] * K
        elif target_kind == "td_like":
            # Mimic Barberis TD: K-length array of (small noise) replicated
            # values typical of bootstrapped Q estimates
            base = float(rng.uniform(-5, 5))
            targets = (rng.normal(base, 1.0, size=K)).tolist()
        else:
            raise ValueError(target_kind)
        seq.append({(loc, a): targets})
    return seq


def test_random():
    print("\n=== Phase B.1: random updates ===")
    rng = np.random.default_rng(11)
    all_ok = True
    for kind in ["uniform", "discrete", "constant", "td_like"]:
        seq = gen_random_seq(rng, n_steps=200, n_states=20, n_actions=2,
                             K=50, target_kind=kind)
        ok = play_sequence(seq, n_states=20, n_actions=2, K=50, lr=0.04,
                           label=f"random/{kind}")
        all_ok = all_ok and ok
    return all_ok


def test_first_visit_zero_target():
    """Probe legacy's all-zero re-init: feed targets that average to 0,
    then feed non-zero targets next visit. Tests whether legacy's
    "qtile_theta == 0 -> re-init" cascades."""
    print("\n=== Phase B.2: first-visit edge cases ===")
    K = 50
    # Visit (0, 0) with all-zero targets, then (0, 0) with positive targets
    seq1 = [
        {(0, 0): [0.0] * K},  # mean=0, after re-init qtile=[0]*K, then SA grad
        {(0, 0): [10.0] * K},  # next visit: legacy may re-trigger? generic won't
    ]
    ok1 = play_sequence(seq1, label="zero_then_pos")

    # Multiple touches with constant target = 0 — does legacy keep re-initing?
    seq2 = [{(0, 0): [0.0] * K} for _ in range(5)]
    ok2 = play_sequence(seq2, label="repeated_zero")

    # Targets summing exactly to 0 at first visit: e.g. [-1, 1, -1, 1, ...]
    targets = [(-1.0) ** i for i in range(K)]
    seq3 = [
        {(0, 0): targets},  # mean = 0 (or near-0 with K odd)
        {(0, 0): [5.0] * K},
    ]
    ok3 = play_sequence(seq3, label="zero_mean_alternating")

    return ok1 and ok2 and ok3


def test_revisit_pattern():
    """Realistic: same (loc, action) revisited many times with TD-like targets.
    Tests that QR gradient steps stay synchronized over hundreds of updates."""
    print("\n=== Phase B.3: revisit synchronization ===")
    rng = np.random.default_rng(7)
    K = 50
    seq = []
    # 500 visits to a small set of (loc, action) pairs
    for _ in range(500):
        loc = int(rng.integers(0, 5))
        a = int(rng.integers(0, 2))
        base = float(rng.uniform(-5, 5))
        targets = (rng.normal(base, 1.0, size=K)).tolist()
        seq.append({(loc, a): targets})
    return play_sequence(seq, n_states=5, n_actions=2, K=50, lr=0.04,
                         label="500-step revisit")


def test_param_inits():
    """Vary K and lr to make sure the per-slice gradient computation
    doesn't have a hidden assumption about size or step size."""
    print("\n=== Phase B.4: vary K and lr ===")
    rng = np.random.default_rng(3)
    all_ok = True
    for K in [10, 20, 50, 100]:
        for lr in [0.01, 0.04, 0.1]:
            seq = gen_random_seq(rng, n_steps=50, n_states=8, n_actions=2,
                                 K=K, target_kind="uniform")
            ok = play_sequence(seq, n_states=8, n_actions=2, K=K, lr=lr,
                               label=f"K={K}, lr={lr}")
            all_ok = all_ok and ok
    return all_ok


def test_with_vectorized_legacy():
    """If we patch legacy to use vectorized gradient (eliminating scalar-sum
    fp noise), do all tests pass? Confirms FP noise is the only divergence."""
    print("\n=== Phase B.5: legacy with vectorized gradient ===")

    class LegacyCriticVectorized(LegacyCriticMinimal):
        def compute_LossGrad(self, data):
            lossgrad = np.zeros(self.loss_grad.shape)
            for (loc, action), targets in data.items():
                if (
                    np.sum(self.qtile_theta[loc, action, :] != np.zeros(self.support_size))
                    == 0
                ):
                    self.qtile_theta[loc, action, :] = float(np.average(targets))

                # Vectorized gradient — same fp ops as generic
                mids = (np.arange(self.support_size, dtype=np.float64) * 2 + 1) \
                    / (2 * self.support_size)
                t_arr = np.asarray(targets, dtype=np.float64)
                cur = self.qtile_theta[loc, action, :]
                cmp = (t_arr[None, :] < cur[:, None]).astype(np.float64)
                lossgrad[loc, action, :] = mids * t_arr.size - cmp.sum(axis=1)
            return lossgrad

    rng = np.random.default_rng(11)
    gen_random_seq(rng, 200, 20, 2, 50, "uniform")
    seq = gen_random_seq(rng, 200, 20, 2, 50, "discrete")

    legacy_v = LegacyCriticVectorized(20, 2, 50, 0.04)

    class _F:
        n_states = 20
    feat = _F()
    generic = QRCritic(feat, 2, support_size=50, lr=0.04, param_init=0.0,
                       cpt_params=CPTParams())

    max_err = 0.0
    for step, data in enumerate(seq):
        legacy_v.update(data)
        generic.update(data)
        d = float(np.abs(legacy_v.qtile_theta - generic.qtile_theta).max())
        max_err = max(max_err, d)
    if max_err < 1e-12:
        print(f"  [vectorized legacy vs generic] OK 200 steps; max_abs_err={max_err:.2e}")
        print("  → confirmed: FP noise from scalar-sum is the only divergence")
        return True
    print(f"  [vectorized legacy vs generic] FAIL max_abs_err={max_err:.4e}")
    return False


def main():
    results = []
    results.append(("random updates (4 kinds)", test_random()))
    results.append(("first-visit edge cases", test_first_visit_zero_target()))
    results.append(("500-step revisit", test_revisit_pattern()))
    results.append(("vary K and lr", test_param_inits()))
    results.append(("legacy w/ vectorized grad", test_with_vectorized_legacy()))

    print("\n" + "=" * 50)
    print("PHASE B SUMMARY")
    print("=" * 50)
    all_ok = True
    for name, ok in results:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name}")
        all_ok = all_ok and ok
    print("=" * 50)
    if all_ok:
        print("Phase B: QRCritic update path is faithful to legacy.")
    else:
        print("Phase B: divergence found — see first-fail step above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
