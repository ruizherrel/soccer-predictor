import pandas as pd
import pytest

from soccer_predictor import odds


def test_is_configured_false_without_a_key(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setattr(odds, "_api_key", lambda: None)
    assert not odds.is_configured()


def test_fetch_odds_returns_empty_without_an_api_key(monkeypatch):
    monkeypatch.setattr(odds, "_api_key", lambda: None)
    result = odds.fetch_odds("E0")
    assert result.empty
    assert list(result.columns) == odds.ODDS_COLUMNS


def test_fetch_odds_returns_empty_for_an_unsupported_league(monkeypatch):
    monkeypatch.setattr(odds, "_api_key", lambda: "fake-key")
    result = odds.fetch_odds("MX2")  # not in SPORT_KEYS -- no provider coverage
    assert result.empty


def test_fetch_odds_never_raises_on_a_network_failure(monkeypatch):
    monkeypatch.setattr(odds, "_api_key", lambda: "fake-key")

    def _boom(*args, **kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr(odds.requests, "get", _boom)
    result = odds.fetch_odds("E0")
    assert result.empty


def test_fetch_odds_parses_the_first_bookmakers_h2h_market(monkeypatch):
    monkeypatch.setattr(odds, "_api_key", lambda: "fake-key")

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {
                    "commence_time": "2026-09-20T19:00:00Z",
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "bookmakers": [
                        {
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Arsenal", "price": 1.8},
                                        {"name": "Draw", "price": 3.6},
                                        {"name": "Chelsea", "price": 4.2},
                                    ],
                                }
                            ]
                        }
                    ],
                }
            ]

    monkeypatch.setattr(odds.requests, "get", lambda *a, **k: _FakeResponse())
    result = odds.fetch_odds("E0")

    assert len(result) == 1
    row = result.iloc[0]
    assert row["home_team"] == "Arsenal"
    assert row["odds_home"] == pytest.approx(1.8)
    assert row["odds_draw"] == pytest.approx(3.6)
    assert row["odds_away"] == pytest.approx(4.2)


def test_find_match_odds_matches_either_order():
    df = pd.DataFrame(
        {
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
            "odds_home": [1.8],
            "odds_draw": [3.6],
            "odds_away": [4.2],
        }
    )
    direct = odds.find_match_odds(df, "Arsenal", "Chelsea")
    assert direct == {"odds_home": 1.8, "odds_draw": 3.6, "odds_away": 4.2}

    # Reversed perspective: our "home_team" is the provider's away side, so
    # the odds must swap accordingly.
    swapped = odds.find_match_odds(df, "Chelsea", "Arsenal")
    assert swapped == {"odds_home": 4.2, "odds_draw": 3.6, "odds_away": 1.8}


def test_find_match_odds_returns_none_when_not_found():
    df = pd.DataFrame(columns=odds.ODDS_COLUMNS)
    assert odds.find_match_odds(df, "Arsenal", "Chelsea") is None
