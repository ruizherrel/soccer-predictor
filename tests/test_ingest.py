import pandas as pd

from soccer_predictor.ingest import _names_roughly_match, _thesportsdb_normalize, find_upcoming_fixture


def test_names_roughly_match_exact():
    assert _names_roughly_match("América", "América")


def test_names_roughly_match_substring_after_accent_stripping():
    assert _names_roughly_match("Alaves", "Deportivo Alavés")
    assert _names_roughly_match("Sociedad", "Real Sociedad")


def test_names_roughly_match_rejects_different_clubs_sharing_a_word():
    assert not _names_roughly_match("Real Madrid", "Real Sociedad")


def test_names_roughly_match_known_cross_source_aliases():
    # football-data.co.uk's abbreviation vs. TheSportsDB's full name --
    # no shared substring even after accent-stripping, so these only match
    # via the explicit _KNOWN_NAME_ALIASES table, not the substring check.
    assert _names_roughly_match("Ath Madrid", "Atlético Madrid")
    assert _names_roughly_match("Atlético Madrid", "Ath Madrid")  # order shouldn't matter
    assert _names_roughly_match("Man United", "Manchester United")
    assert _names_roughly_match("Nott'm Forest", "Nottingham Forest")


def test_names_roughly_match_does_not_alias_unrelated_names():
    assert not _names_roughly_match("Ath Madrid", "Manchester United")


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


def test_thesportsdb_normalize_renames_known_teams_for_football_data_leagues():
    # Used when the current season is backfilled from TheSportsDB (see
    # ingest.refresh's football-data.co.uk/mirror fallback) -- these must
    # rename to football-data.co.uk's own established name, not just be
    # recognized as equivalent, so a team's Elo/Pi rating history doesn't
    # fragment into two different dict keys.
    assert _thesportsdb_normalize("SP1", "Atlético Madrid") == "Ath Madrid"
    assert _thesportsdb_normalize("E0", "Manchester United") == "Man United"
    assert _thesportsdb_normalize("D1", "Borussia Mönchengladbach") == "M'gladbach"
    assert _thesportsdb_normalize("F1", "Paris Saint-Germain") == "Paris SG"


def test_thesportsdb_normalize_disambiguates_inter_milan_from_ac_milan():
    # "Inter Milan" fuzzy-substring-matches both "Inter" and "Milan" -- a
    # real false-positive risk (AC Milan and Inter are different clubs).
    # This must resolve to "Inter", not "Milan".
    assert _thesportsdb_normalize("I1", "Inter Milan") == "Inter"
    assert _thesportsdb_normalize("I1", "AC Milan") == "Milan"


def test_thesportsdb_normalize_leaves_a_team_with_no_prior_history_unchanged():
    # Newly promoted/returned clubs with zero football-data.co.uk history
    # (this season: Coventry City, Racing de Santander, Elversberg, Le
    # Mans) have no established name to rename to -- kept as TheSportsDB's
    # own name so Elo/Pi's existing new-team seeding applies.
    assert _thesportsdb_normalize("SP1", "Racing de Santander") == "Racing de Santander"
