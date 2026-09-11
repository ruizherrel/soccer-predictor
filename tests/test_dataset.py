import numpy as np
import pandas as pd
import pytest

import config
from soccer_predictor.dataset import add_pythagorean_features, build_ratings_features


def _toy_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-04-01", "2025-04-02"]),
            "season": ["2025", "2025"],
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "home_goals": [3, 1],
            "away_goals": [2, 4],
            "result": ["H", "A"],
        }
    )


def test_league_without_overrides_uses_global_elo_defaults(monkeypatch):
    monkeypatch.setitem(config.LEAGUES, "TESTLEAGUE", {"name": "Test", "sport": "soccer", "source": "thesportsdb"})

    _ratings, elo, _pi = build_ratings_features(_toy_matches(), league="TESTLEAGUE")

    assert elo.k == config.ELO_K
    assert elo.home_adv == config.ELO_HOME_ADV


def test_league_with_overrides_uses_its_own_elo_hyperparameters(monkeypatch):
    monkeypatch.setitem(
        config.LEAGUES,
        "TESTLEAGUE",
        {
            "name": "Test",
            "sport": "baseball",
            "source": "thesportsdb",
            "elo_k": 4.0,
            "elo_home_adv": 24.0,
        },
    )

    _ratings, elo, _pi = build_ratings_features(_toy_matches(), league="TESTLEAGUE")

    assert elo.k == 4.0
    assert elo.home_adv == 24.0


def test_no_league_given_uses_global_elo_defaults():
    _ratings, elo, _pi = build_ratings_features(_toy_matches())

    assert elo.k == config.ELO_K
    assert elo.home_adv == config.ELO_HOME_ADV


def test_pythagorean_features_match_the_james_formula():
    df = pd.DataFrame(
        {
            "home_gf_last10": [5.0, 4.0],
            "home_ga_last10": [2.0, 4.0],
            "away_gf_last10": [3.0, 4.0],
            "away_ga_last10": [3.0, 4.0],
        }
    )
    out = add_pythagorean_features(df)

    expected_home_0 = 5.0**config.PYTH_EXPONENT / (5.0**config.PYTH_EXPONENT + 2.0**config.PYTH_EXPONENT)
    assert out["pyth_home_pct"].iloc[0] == pytest.approx(expected_home_0)
    # Equal goals for/against both ways round-trips to a coin flip (0.5) for
    # both sides, so the home/away difference is exactly 0.
    assert out["pyth_home_pct"].iloc[1] == pytest.approx(0.5)
    assert out["pyth_away_pct"].iloc[1] == pytest.approx(0.5)
    assert out["pyth_diff"].iloc[1] == pytest.approx(0.0)


def test_pythagorean_features_are_nan_with_no_scoring_history():
    df = pd.DataFrame(
        {
            "home_gf_last10": [np.nan],
            "home_ga_last10": [np.nan],
            "away_gf_last10": [0.0],
            "away_ga_last10": [0.0],
        }
    )
    out = add_pythagorean_features(df)

    assert pd.isna(out["pyth_home_pct"].iloc[0])
    # 0**exp / (0**exp + 0**exp) is 0/0, a NaN -- not a divide-by-zero crash.
    assert pd.isna(out["pyth_away_pct"].iloc[0])
