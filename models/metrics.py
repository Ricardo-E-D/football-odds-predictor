"""Proper scoring rules for probability forecasts.

Both are "proper": the forecaster minimizes the expected score only by
reporting true beliefs. Accuracy is not proper — it ignores confidence.
Lower is better for both.
"""

import numpy as np

OUTCOMES = ["H", "D", "A"]


def outcome_index(results) -> np.ndarray:
    """Map 'H'/'D'/'A' results to 0/1/2 column indices."""
    lookup = {o: i for i, o in enumerate(OUTCOMES)}
    return np.array([lookup[r] for r in results])


def log_loss(y: np.ndarray, probs: np.ndarray, eps: float = 1e-12) -> float:
    """Mean of -log(probability assigned to the outcome that happened)."""
    p = np.clip(probs[np.arange(len(y)), y], eps, 1.0)
    return float(-np.mean(np.log(p)))


def brier_score(y: np.ndarray, probs: np.ndarray) -> float:
    """Mean squared error between the probability vector and the one-hot outcome."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
