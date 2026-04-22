"""CPT (Cumulative Prospect Theory) utilities — shared across agents."""

import numpy as np


def compute_cdf(k: int, q: int, pos: bool) -> float:
    if pos:
        return 1 - k / q
    return k / q


def prob_weight(F: float, pos: bool, rho1: float, rho2: float) -> float:
    rho = rho1 if pos else rho2
    if F <= 0:
        return 0.0
    if F >= 1:
        return 1.0
    return F ** rho / ((F ** rho + (1 - F) ** rho) ** (1 / rho))


def utility(x: float, pos: bool, alpha: float, lmbd: float) -> float:
    if pos:
        return x ** alpha if x >= 0 else 0.0
    return -lmbd * (-x) ** alpha if x <= 0 else 0.0


def compute_CPT(x_list, alpha=0.95, rho1=0.5, rho2=0.5, lmbd=1.5, sort=True):
    """Cumulative Prospect Theory value of samples ``x_list``."""
    if sort:
        x_list = sorted(x_list)
    m = len(x_list)
    if m == 0:
        return 0.0

    cpt_pos = 0.0
    cpt_neg = 0.0
    for id_, x in enumerate(x_list):
        i = id_ + 1
        if x >= 0:
            dF = prob_weight(compute_cdf(i - 1, m, True), True, rho1, rho2) - prob_weight(
                compute_cdf(i, m, True), True, rho1, rho2
            )
            cpt_pos += utility(x, True, alpha, lmbd) * dF
        else:
            dF = prob_weight(compute_cdf(i, m, False), False, rho1, rho2) - prob_weight(
                compute_cdf(i - 1, m, False), False, rho1, rho2
            )
            cpt_neg += utility(x, False, alpha, lmbd) * dF
    return cpt_pos + cpt_neg


class CPTParams:
    """Bundle CPT hyperparameters for passing around."""

    def __init__(self, alpha=0.95, rho1=0.5, rho2=0.5, lmbd=1.5):
        self.alpha = alpha
        self.rho1 = rho1
        self.rho2 = rho2
        self.lmbd = lmbd

    def compute(self, x_list, sort=True):
        return compute_CPT(
            x_list, self.alpha, self.rho1, self.rho2, self.lmbd, sort=sort
        )
