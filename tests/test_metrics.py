import numpy as np
import pytest

from soccer_predictor.metrics import mean_rps, multiclass_log_loss, rps

# CLASS_ORDER = (away, draw, home) -> indices 0, 1, 2


def test_rps_perfect_prediction_is_zero():
    probs = np.array([[0.0, 0.0, 1.0]])  # certain home win
    outcome_idx = np.array([2])  # home win
    assert rps(probs, outcome_idx)[0] == pytest.approx(0.0)


def test_rps_confidently_wrong_is_maximal():
    probs = np.array([[1.0, 0.0, 0.0]])  # certain away win
    outcome_idx = np.array([2])  # actually home win
    assert rps(probs, outcome_idx)[0] == pytest.approx(1.0)


def test_rps_uniform_prediction_matches_hand_calculation():
    probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
    outcome_idx = np.array([2])  # home win
    expected = ((1 / 3) ** 2 + (2 / 3) ** 2) / 2
    assert rps(probs, outcome_idx)[0] == pytest.approx(expected)


def test_rps_penalizes_being_wrong_by_more_than_one_class_more():
    # A draw prediction that's actually a home win should score better
    # (lower RPS) than an away-win prediction that's actually a home win,
    # since RPS respects the away < draw < home ordering.
    probs_draw_favored = np.array([[0.1, 0.8, 0.1]])
    probs_away_favored = np.array([[0.8, 0.1, 0.1]])
    outcome_idx = np.array([2])  # home win

    rps_draw = rps(probs_draw_favored, outcome_idx)[0]
    rps_away = rps(probs_away_favored, outcome_idx)[0]
    assert rps_draw < rps_away


def test_mean_rps_averages_across_rows():
    probs = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    outcome_idx = np.array([2, 2])  # first perfect, second maximally wrong
    assert mean_rps(probs, outcome_idx) == pytest.approx(0.5)


def test_multiclass_log_loss_runs_and_is_positive_for_imperfect_predictions():
    probs = np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    outcome_idx = np.array([2, 0])
    loss = multiclass_log_loss(probs, outcome_idx)
    assert loss > 0.0


def test_multiclass_log_loss_tolerates_tiny_negative_floating_point_noise():
    # sklearn's log_loss hard-rejects any negative probability, even a
    # ~1e-12 rounding artifact -- seen in production from the Poisson
    # model's ridge-regularized fallback fit on Copa Libertadores. Real
    # model output should never be more negative than this.
    probs = np.array([[-6.238881808718773e-12, 0.4, 0.6]])
    outcome_idx = np.array([2])
    loss = multiclass_log_loss(probs, outcome_idx)
    assert loss > 0.0
