"""Legacy-faithful version of `filter_quantiles` for refactor regression testing.

Mirrors `agents/rerun_GreedySPERL_QR__main.py` lines 1143-1218 verbatim:
  * `lbVal = qval_filtered[i-1]` — cascading lookback (uses just-snapped values),
    NOT the largest original-valid index.
  * `distlb = quantiles[i] - lbVal` — signed difference, can be negative under
    crossing quantiles, vs `abs(...)` in `filter_quantiles`.

Paper Algorithm 4 (Lines 11-15) describes the original-valid + abs version
(matching `agents.sperl_qr_generic.filter_quantiles`). Use this module to
A/B test whether legacy's deviation explains the paper-Optimality gap.
"""

import numpy as np


def legacy_faithful_filter(quantiles, p_filter):
    qval_gaps = np.array(quantiles)[1:] - np.array(quantiles)[:-1]
    tresh_ = np.quantile(qval_gaps, p_filter, method="higher")

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
    return np.asarray(qval_filtered, dtype=np.float64)
