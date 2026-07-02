import numpy as np

from models.metrics import brier_score, log_loss, outcome_index


def test_perfect_forecast_scores_zero():
    y = outcome_index(["H", "A"])
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert log_loss(y, probs) == 0.0
    assert brier_score(y, probs) == 0.0


def test_confident_wrong_beats_uniform_only_when_right():
    y = outcome_index(["H"])
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
    confident_wrong = np.array([[0.01, 0.01, 0.98]])
    assert log_loss(y, uniform) < log_loss(y, confident_wrong)
    assert brier_score(y, uniform) < brier_score(y, confident_wrong)


def test_better_calibrated_scores_lower():
    y = outcome_index(["H", "H", "H", "A"])
    sharp = np.tile([0.75, 0.15, 0.10], (4, 1))
    vague = np.tile([0.40, 0.30, 0.30], (4, 1))
    assert log_loss(y, sharp) < log_loss(y, vague)
