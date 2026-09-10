import pandas as pd

import config
from soccer_predictor.dataset import build_ratings_features


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
