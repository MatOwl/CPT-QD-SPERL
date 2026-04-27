"""Phase A unit-level isolation: legacy vs generic pure functions.

Verifies that the refactored CPT compute and quantile filter produce numerically
identical outputs vs the legacy reference (`agents/rerun_GreedySPERL_QR__main.py`).
The reference functions are duplicated verbatim here (they're closures inside the
legacy script's loop body, so we lift them into top-level callables for testing).

If any assertion fails, the bug is localized to the diverging function and the
heavier Phase B/C work isn't needed yet.

Run:
    PYTHONPATH=. python scripts/verify_refactor_phase_a.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

from lib.cpt import compute_CPT as generic_compute_CPT, CPTParams
from agents.sperl_qr_generic import filter_quantiles as generic_filter


# =============================================================================
# Legacy reference functions, lifted verbatim from agents/rerun_GreedySPERL_QR__main.py
# (lines 1113-1218 closure; see legacy file for original location).
# Closure parameters (alpha, rho1, rho2, lmbd, p_filter) made explicit.
# =============================================================================


def legacy_compute_cdf(k, q, pos):
    if pos is True:
        return 1 - k / q
    else:
        return k / q


def legacy_prob_weight(F, pos, rho1, rho2):
    if pos is True:
        return F ** rho1 / ((F ** rho1 + (1 - F) ** rho1) ** (1 / rho1))
    else:
        return F ** rho2 / ((F ** rho2 + (1 - F) ** rho2) ** (1 / rho2))


def legacy_utility(x, pos, alpha, lmbd):
    if pos is True:
        return x ** alpha
    else:
        return -lmbd * (-x) ** alpha


def legacy_filtering(quantiles, p_filter):
    qval_gaps = np.array(quantiles)[1:] - np.array(quantiles)[:-1]
    tresh_ = np.quantile(qval_gaps, p_filter, interpolation="higher")

    filter_ = qval_gaps <= tresh_ + 1e-6
    filter_ = np.multiply(
        filter_,
        1 - np.multiply(qval_gaps < -1e-6, np.abs(qval_gaps - tresh_) > 1e-6),
    )

    if filter_[0] != False:
        filter_ = np.append([True], filter_)
    else:
        filter_ = np.append([False], filter_)

    qval_filtered = np.multiply(filter_, np.array(quantiles))

    ubVal = None
    qval_filtered = list(qval_filtered) + [np.inf]

    for i in range(len(qval_filtered)):
        if abs(qval_filtered[i]) >= 1e-6:
            ubVal = None
            continue
        if i - 1 < 0:
            ubID = next(
                j
                for j, x in enumerate(qval_filtered)
                if j > i and abs(x) >= 1e-6
            )
            if ubID > len(quantiles) - 1:
                qval_filtered[i] = 0.0
                continue
            ubVal = quantiles[ubID]
            qval_filtered[i] = ubVal
        else:
            lbVal = qval_filtered[i - 1]
            if ubVal is None:
                ubID = next(
                    j
                    for j, x in enumerate(qval_filtered)
                    if j > i and abs(x) >= 1e-6
                )
                if ubID > len(quantiles) - 1:
                    qval_filtered[i] = lbVal
                    continue
                ubVal = quantiles[ubID]

            distlb = quantiles[i] - lbVal
            distub = ubVal - quantiles[i]

            if distlb < distub:
                qval_filtered[i] = lbVal
            else:
                qval_filtered[i] = ubVal

    qval_filtered = qval_filtered[:-1]
    return qval_filtered


def legacy_compute_CPT(XList, alpha, rho1, rho2, lmbd, sort=True, lbub=0,
                       p_filter=0.75):
    if sort:
        XList = sorted(XList)
    if lbub == 1:
        XList = legacy_filtering(XList, p_filter)
    m = len(XList)

    CPT_val_pos = 0
    CPT_val_neg = 0

    for id_, x in enumerate(XList):
        i = id_ + 1
        if x >= 0:
            dF_i_pos = (
                legacy_prob_weight(legacy_compute_cdf(i - 1, m, True), True, rho1, rho2)
                - legacy_prob_weight(legacy_compute_cdf(i, m, True), True, rho1, rho2)
            )
            CPT_val_pos += legacy_utility(x, True, alpha, lmbd) * dF_i_pos
        elif x < 0:
            dF_i_neg = (
                legacy_prob_weight(legacy_compute_cdf(i, m, False), False, rho1, rho2)
                - legacy_prob_weight(
                    legacy_compute_cdf(i - 1, m, False), False, rho1, rho2
                )
            )
            CPT_val_neg += legacy_utility(x, False, alpha, lmbd) * dF_i_neg

    return CPT_val_pos + CPT_val_neg


# =============================================================================
# Test harness
# =============================================================================


CPT_CONFIGS = [
    # (label, alpha, rho1, rho2, lmbd)
    ("CPT88", 0.88, 0.65, 0.65, 2.25),
    ("CPT95", 0.95, 0.5, 0.5, 1.5),
    ("Prashanth", 0.88, 0.61, 0.69, 2.25),
]


def random_quantiles(rng, n, kind):
    """Generate sorted-ish quantile arrays of various shapes."""
    if kind == "all_pos":
        return np.sort(rng.uniform(0, 50, size=n))
    if kind == "all_neg":
        return np.sort(rng.uniform(-50, 0, size=n))
    if kind == "mixed":
        return np.sort(rng.uniform(-50, 50, size=n))
    if kind == "with_zero":
        v = np.sort(rng.uniform(-50, 50, size=n))
        v[n // 2] = 0.0
        return v
    if kind == "crossing":
        # unsorted with small noise — for filter testing
        v = np.sort(rng.uniform(-30, 30, size=n))
        idx = rng.choice(n, size=n // 5, replace=False)
        v[idx] += rng.uniform(-3, 3, size=idx.size)
        return v
    if kind == "discrete_like":
        # mass at 3 points + small noise
        base = rng.choice([-20.0, 0.0, 30.0], size=n)
        return np.sort(base + rng.normal(0, 0.5, size=n))
    raise ValueError(kind)


def test_compute_CPT(verbose=True):
    print("\n=== Phase A.1: compute_CPT  legacy vs generic ===")
    rng = np.random.default_rng(42)
    n_trials = 50
    K = 50
    max_err = 0.0
    fails = 0

    for cfg_label, alpha, rho1, rho2, lmbd in CPT_CONFIGS:
        for kind in ["all_pos", "all_neg", "mixed", "with_zero"]:
            for _ in range(n_trials):
                q = random_quantiles(rng, K, kind)
                v_legacy = legacy_compute_CPT(
                    q.tolist(), alpha, rho1, rho2, lmbd, sort=False, lbub=0
                )
                v_generic = generic_compute_CPT(
                    q.tolist(), alpha=alpha, rho1=rho1, rho2=rho2, lmbd=lmbd,
                    sort=False,
                )
                err = abs(v_legacy - v_generic)
                max_err = max(max_err, err)
                if not np.isclose(v_legacy, v_generic, rtol=1e-12, atol=1e-12):
                    fails += 1
                    if verbose and fails <= 3:
                        print(f"  MISMATCH ({cfg_label}, {kind}): legacy={v_legacy:.10f} "
                              f"generic={v_generic:.10f} diff={err:.2e}")

    total = len(CPT_CONFIGS) * 4 * n_trials
    print(f"  total trials: {total}, failures: {fails}, max abs err: {max_err:.2e}")
    if fails == 0:
        print("  OK: compute_CPT identical")
    else:
        print(f"  ERR: {fails}/{total} mismatches")
    return fails == 0


def test_filtering(verbose=True):
    print("\n=== Phase A.2: filter_quantiles  legacy vs generic ===")
    rng = np.random.default_rng(13)
    n_trials = 50
    K = 50
    max_err = 0.0
    fails = 0

    for p_filter in [0.75, 0.85, 0.9, 0.95, 1.0]:
        for kind in ["mixed", "crossing", "discrete_like", "all_pos"]:
            for _ in range(n_trials):
                q_in = random_quantiles(rng, K, kind)
                q_legacy = np.asarray(legacy_filtering(q_in.tolist(), p_filter),
                                      dtype=np.float64)
                q_generic = generic_filter(q_in, p_filter=p_filter)
                err = float(np.max(np.abs(q_legacy - q_generic)))
                max_err = max(max_err, err)
                if not np.allclose(q_legacy, q_generic, rtol=0, atol=1e-9):
                    fails += 1
                    if verbose and fails <= 3:
                        print(f"  MISMATCH (p={p_filter}, {kind}): max diff {err:.2e}")
                        print(f"    legacy:  {np.round(q_legacy[:8], 4)}")
                        print(f"    generic: {np.round(q_generic[:8], 4)}")

    total = 5 * 4 * n_trials
    print(f"  total trials: {total}, failures: {fails}, max abs err: {max_err:.2e}")
    if fails == 0:
        print("  OK: filter_quantiles identical")
    else:
        print(f"  ERR: {fails}/{total} mismatches")
    return fails == 0


def test_filter_then_cpt(verbose=True):
    """End-to-end of Alg 4: filter -> CPT. Both implementations chained."""
    print("\n=== Phase A.3: filter -> CPT chained ===")
    rng = np.random.default_rng(7)
    n_trials = 30
    K = 50
    max_err = 0.0
    fails = 0

    for cfg_label, alpha, rho1, rho2, lmbd in CPT_CONFIGS:
        for p_filter in [0.75, 0.9]:
            for kind in ["mixed", "discrete_like"]:
                for _ in range(n_trials):
                    q = random_quantiles(rng, K, kind)
                    q_legacy_filt = np.asarray(
                        legacy_filtering(q.tolist(), p_filter), dtype=np.float64
                    )
                    q_generic_filt = generic_filter(q, p_filter=p_filter)

                    v_legacy = legacy_compute_CPT(
                        q_legacy_filt.tolist(), alpha, rho1, rho2, lmbd,
                        sort=False, lbub=0,
                    )
                    v_generic = generic_compute_CPT(
                        q_generic_filt.tolist(), alpha=alpha, rho1=rho1,
                        rho2=rho2, lmbd=lmbd, sort=False,
                    )
                    err = abs(v_legacy - v_generic)
                    max_err = max(max_err, err)
                    if not np.isclose(v_legacy, v_generic, rtol=1e-9, atol=1e-9):
                        fails += 1
                        if verbose and fails <= 3:
                            print(f"  MISMATCH ({cfg_label}, p={p_filter}, {kind}): "
                                  f"legacy={v_legacy:.6f} generic={v_generic:.6f}")

    total = len(CPT_CONFIGS) * 2 * 2 * n_trials
    print(f"  total trials: {total}, failures: {fails}, max abs err: {max_err:.2e}")
    if fails == 0:
        print("  OK: filter->CPT pipeline identical")
    else:
        print(f"  ERR: {fails}/{total} mismatches")
    return fails == 0


def test_barberis_env_determinism():
    """Verify barberisCasino is deterministic given the same RNG seed + actions."""
    print("\n=== Phase A.4: barberisCasino determinism ===")
    from lib.envs.barberis_casino import barberisCasino

    rng = np.random.default_rng(0)
    actions = rng.integers(0, 2, size=10).tolist()

    np.random.seed(123)
    env1 = barberisCasino(p=0.66)
    s1 = env1.reset()
    traj1 = []
    for a in actions:
        s, r, done, _ = env1.step(int(a))
        traj1.append((tuple(s), float(r), bool(done)))
        if done:
            break

    np.random.seed(123)
    env2 = barberisCasino(p=0.66)
    s2 = env2.reset()
    traj2 = []
    for a in actions:
        s, r, done, _ = env2.step(int(a))
        traj2.append((tuple(s), float(r), bool(done)))
        if done:
            break

    if traj1 == traj2:
        print(f"  OK: identical {len(traj1)}-step trajectory under seed=123")
        return True
    print("  ERR: trajectories differ:")
    for i, (t1, t2) in enumerate(zip(traj1, traj2)):
        if t1 != t2:
            print(f"    step {i}: {t1} vs {t2}")
    return False


def main():
    results = []
    results.append(("compute_CPT", test_compute_CPT()))
    results.append(("filter_quantiles", test_filtering()))
    results.append(("filter->CPT pipeline", test_filter_then_cpt()))
    results.append(("barberis env determinism", test_barberis_env_determinism()))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_ok = True
    for name, ok in results:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name}")
        all_ok = all_ok and ok
    print("=" * 50)
    if all_ok:
        print("Phase A: all unit-level functions match. Move on to Phase B.")
    else:
        print("Phase A: at least one mismatch found. Inspect above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
