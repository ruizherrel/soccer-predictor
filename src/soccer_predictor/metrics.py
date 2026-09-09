"""Evaluation metrics for 1X2 probabilistic forecasts.

RPS (Ranked Probability Score) is the standard metric for football outcome
forecasts (Constantinou & Fenton) because it respects the natural ordering
Away < Draw < Home and penalizes confidently-wrong predictions more than
plain accuracy or a class-unaware Brier score would.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss as sk_log_loss

# Fixed class order used throughout this project's RPS/probability arrays.
CLASS_ORDER = ("away", "draw", "home")


def rps(probs: np.ndarray, outcome_idx: np.ndarray) -> np.ndarray:
    """Ranked Probability Score, one value per row.

    probs: (n, 3) array of probabilities in CLASS_ORDER (away, draw, home).
    outcome_idx: (n,) array of ints, 0=away, 1=draw, 2=home.
    Lower is better; 0 = perfect, 1 = maximally wrong for r=3.
    """
    probs = np.asarray(probs, dtype=float)
    n_classes = probs.shape[1]
    cum_probs = np.cumsum(probs, axis=1)

    outcome_onehot = np.zeros_like(probs)
    outcome_onehot[np.arange(len(outcome_idx)), outcome_idx] = 1.0
    cum_outcome = np.cumsum(outcome_onehot, axis=1)

    diffs_sq = (cum_probs - cum_outcome) ** 2
    return diffs_sq[:, :-1].sum(axis=1) / (n_classes - 1)


def mean_rps(probs: np.ndarray, outcome_idx: np.ndarray) -> float:
    return float(np.mean(rps(probs, outcome_idx)))


def multiclass_log_loss(probs: np.ndarray, outcome_idx: np.ndarray) -> float:
    return float(sk_log_loss(outcome_idx, probs, labels=list(range(probs.shape[1]))))
