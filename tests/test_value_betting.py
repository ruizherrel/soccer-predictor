import pytest

from soccer_predictor.value_betting import kelly_fraction


def test_kelly_fraction_zero_for_no_edge():
    # Fair odds for a 50% true probability: no edge, no stake.
    assert kelly_fraction(model_prob=0.5, decimal_odds=2.0) == pytest.approx(0.0)


def test_kelly_fraction_zero_for_negative_edge():
    # Model thinks 40%, market prices it at 50% (odds 2.0): negative edge.
    assert kelly_fraction(model_prob=0.4, decimal_odds=2.0) == pytest.approx(0.0)


def test_kelly_fraction_positive_for_genuine_edge():
    # Model thinks 60%, market prices it at 50% (odds 2.0): real edge.
    frac = kelly_fraction(model_prob=0.6, decimal_odds=2.0, cap=1.0)
    assert frac == pytest.approx(0.2)  # full Kelly: edge / b = (0.6*1 - 0.4) / 1


def test_kelly_fraction_respects_cap():
    frac = kelly_fraction(model_prob=0.9, decimal_odds=5.0, cap=0.05)
    assert frac == pytest.approx(0.05)


def test_kelly_fraction_zero_for_odds_at_or_below_evens_payout():
    # decimal_odds <= 1.0 means no payout above stake; never a valid bet.
    assert kelly_fraction(model_prob=0.9, decimal_odds=1.0) == pytest.approx(0.0)
