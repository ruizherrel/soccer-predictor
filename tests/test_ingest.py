import pandas as pd

from soccer_predictor.ingest import _names_roughly_match, find_upcoming_fixture


def test_names_roughly_match_exact():
    assert _names_roughly_match("América", "América")


def test_names_roughly_match_substring_after_accent_stripping():
    assert _names_roughly_match("Alaves", "Deportivo Alavés")
    assert _names_roughly_match("Sociedad", "Real Sociedad")


def test_names_roughly_match_rejects_different_clubs_sharing_a_word():
    assert not _names_roughly_match("Real Madrid", "Real Sociedad")


def test_find_upcoming_fixture_matches_either_order():
    fixtures = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-20"]),
            "home_team": ["Deportivo Alavés"],
            "away_team": ["Real Madrid"],
        }
    )
    assert find_upcoming_fixture(fixtures, "Real Madrid", "Alaves") == pd.Timestamp("2026-09-20")
    assert find_upcoming_fixture(fixtures, "Alaves", "Real Madrid") == pd.Timestamp("2026-09-20")


def test_find_upcoming_fixture_returns_none_when_not_scheduled():
    fixtures = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-20"]),
            "home_team": ["Deportivo Alavés"],
            "away_team": ["Real Madrid"],
        }
    )
    assert find_upcoming_fixture(fixtures, "Barcelona", "Sevilla") is None


def test_find_upcoming_fixture_handles_empty_fixtures():
    fixtures = pd.DataFrame(columns=["date", "home_team", "away_team"])
    assert find_upcoming_fixture(fixtures, "Barcelona", "Sevilla") is None
