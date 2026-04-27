"""Load legacy `QPG_CPT` / `CPTCritic` / `GreedyPolicy` classes by slicing
`agents/rerun_GreedySPERL_QR__main.py` at line 1048 (before the unguarded
runner block) and exec'ing the prefix in a controlled namespace.

This avoids modifying the legacy file. Returns a dict-like namespace with
the class definitions and helper functions, suitable for instantiating
QPG_CPT for ground-truth comparison.

Caller must import `scripts.legacy_compat_stubs` first to make
`record_csv` and `stable_baselines` importable.
"""

from __future__ import annotations

import os

# Trigger sys.modules stubs
import scripts.legacy_compat_stubs  # noqa: F401


_LEGACY_PATH = os.path.join("agents", "rerun_GreedySPERL_QR__main.py")
_RUNNER_BOUNDARY_LINE = 1048  # 1-based; runner block starts here


def _load_class_namespace():
    with open(_LEGACY_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    prefix = "".join(lines[: _RUNNER_BOUNDARY_LINE - 1])

    ns = {"__name__": "legacy_module", "__file__": _LEGACY_PATH}
    code = compile(prefix, _LEGACY_PATH, "exec")
    exec(code, ns)
    return ns


_NS = None


def get_legacy_namespace():
    """Cached. Call once to load; returned dict has GreedyPolicy, CPTCritic,
    QPG_CPT, barberisFeaturize, etc. as keys."""
    global _NS
    if _NS is None:
        _NS = _load_class_namespace()
    return _NS


def inject_closures(ns, alpha, rho1, rho2, lmbd, p_filter):
    """Inject the closure functions defined inside the runner loop
    (lines 1113-1255 of legacy) into the given namespace, parameterized
    by the CPT and filter values for this run.

    These closures (compute_CPT, filtering, prob_weight, utility,
    compute_cdf) are referenced by CPTCritic.CPTpredict and the policy
    update path. Without injection, those classes raise NameError at
    runtime.

    Verbatim port of legacy code inside the runner; same semantics.
    """
    import numpy as _np

    def compute_cdf(k, q, pos):
        if pos is True:
            return 1 - k / q
        else:
            return k / q

    def prob_weight(F, pos, _rho1=rho1, _rho2=rho2):
        if pos is True:
            return F ** _rho1 / ((F ** _rho1 + (1 - F) ** _rho1) ** (1 / _rho1))
        else:
            return F ** _rho2 / ((F ** _rho2 + (1 - F) ** _rho2) ** (1 / _rho2))

    def utility(x, pos, _alpha=alpha, _lmbd=lmbd):
        if pos is True:
            return x ** _alpha
        else:
            return -_lmbd * (-x) ** _alpha

    def filtering(quantiles, _p_filter=p_filter):
        qval_gaps = _np.array(quantiles)[1:] - _np.array(quantiles)[:-1]
        tresh_ = _np.quantile(qval_gaps, _p_filter, method="higher")

        filter_ = qval_gaps <= tresh_ + 1e-6
        filter_ = _np.multiply(
            filter_,
            1 - _np.multiply(
                qval_gaps < -1e-6, _np.abs(qval_gaps - tresh_) > 1e-6
            ),
        )

        if filter_[0] != False:
            filter_ = _np.append([True], filter_)
        else:
            filter_ = _np.append([False], filter_)

        qval_filtered = _np.multiply(filter_, _np.array(quantiles))

        ubVal = None
        qval_filtered = list(qval_filtered) + [_np.inf]

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

    def compute_CPT(XList, sort=True, lbub=0):
        if sort:
            XList = sorted(XList)
        if lbub == 1:
            XList = filtering(XList)
        m = len(XList)

        CPT_val_pos = 0
        CPT_val_neg = 0

        for id_, x in enumerate(XList):
            i = id_ + 1
            if x >= 0:
                dF_i_pos = prob_weight(compute_cdf(i - 1, m, True), True) - \
                    prob_weight(compute_cdf(i, m, True), True)
                CPT_val_pos += utility(x, True) * dF_i_pos
            elif x < 0:
                dF_i_neg = prob_weight(compute_cdf(i, m, False), False) - \
                    prob_weight(compute_cdf(i - 1, m, False), False)
                CPT_val_neg += utility(x, False) * dF_i_neg

        return CPT_val_pos + CPT_val_neg

    ns["compute_cdf"] = compute_cdf
    ns["prob_weight"] = prob_weight
    ns["utility"] = utility
    ns["filtering"] = filtering
    ns["compute_CPT"] = compute_CPT


if __name__ == "__main__":
    ns = get_legacy_namespace()
    expected = ["GreedyPolicy", "CPTCritic", "QPG_CPT", "barberisFeaturize"]
    print("Loaded legacy namespace:")
    for k in expected:
        v = ns.get(k, "<MISSING>")
        print(f"  {k}: {type(v).__name__} {v}")
