import pandas as pd

from soccer_predictor.form_features import build_form_table, build_venue_form_table, current_venue_form


def _toy_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
            "season": ["1920", "1920", "1920"],
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "C", "D"],
            "home_goals": [1, 2, 0],
            "away_goals": [0, 2, 5],
        }
    )


def _toy_matches_mixed_venue() -> pd.DataFrame:
    """Team A plays home, then away, then home again -- designed so a bug
    that mixes venues (instead of keeping each team's home and away
    appearances in separate series) is directly observable in the 3rd
    match's pre-match venue form."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
            "season": ["1920", "1920", "1920"],
            "home_team": ["A", "X", "A"],
            "away_team": ["B", "A", "Z"],
            # Match 1 (A home vs B): A scores 2, concedes 0.
            # Match 2 (X home vs A away): A scores 3, concedes 1.
            # Match 3 (A home vs Z): the row under test.
            "home_goals": [2, 1, 9],
            "away_goals": [0, 3, 9],
        }
    )


def test_rolling_form_excludes_the_match_itself():
    matches = _toy_matches()
    form = build_form_table(matches, windows=(2,))

    row = form[(form["team"] == "A") & (form["date"] == pd.Timestamp("2020-01-15"))].iloc[0]

    # Pre-match form for the 3rd fixture must reflect only the first two
    # results (3pts/1pt, 1gf avg 1.5, 0ga avg 1.0) -- never the 0-5 loss
    # that is about to happen in that same row's own match.
    assert row["ppg_last2"] == 2.0
    assert row["gf_last2"] == 1.5
    assert row["ga_last2"] == 1.0


def test_first_ever_match_has_no_prior_form():
    matches = _toy_matches()
    form = build_form_table(matches, windows=(2,))

    row = form[(form["team"] == "A") & (form["date"] == pd.Timestamp("2020-01-01"))].iloc[0]
    assert pd.isna(row["ppg_last2"])


def test_home_venue_form_ignores_that_teams_away_matches():
    matches = _toy_matches_mixed_venue()
    form = build_venue_form_table(matches, "home", windows=(1,))

    # A's 3rd match (home vs Z) pre-match home-venue form must reflect only
    # A's 1st match (home vs B, gf=2 ga=0) -- never the 2nd match (A away
    # at X, gf=3 ga=1), even though it's A's most recent match overall.
    row = form[(form["team"] == "A") & (form["date"] == pd.Timestamp("2020-01-15"))].iloc[0]
    assert row["gf_last1"] == 2.0
    assert row["ga_last1"] == 0.0


def test_away_venue_form_ignores_that_teams_home_matches():
    matches = _toy_matches_mixed_venue()
    form = build_venue_form_table(matches, "away", windows=(1,))

    # A's only away match (2nd, at X) has no prior away match of its own --
    # A's 1st match was at home, and must not leak into the away-only series.
    row = form[(form["team"] == "A") & (form["date"] == pd.Timestamp("2020-01-08"))].iloc[0]
    assert pd.isna(row["gf_last1"])


def test_venue_form_excludes_the_match_itself():
    matches = _toy_matches_mixed_venue()
    form = build_venue_form_table(matches, "home", windows=(1,))

    row = form[(form["team"] == "A") & (form["date"] == pd.Timestamp("2020-01-01"))].iloc[0]
    assert pd.isna(row["gf_last1"])  # A's very first home match: nothing prior


def test_current_venue_form_matches_the_teams_actual_venue_history():
    matches = _toy_matches_mixed_venue()

    home_snapshot = current_venue_form(matches, "A", "home", windows=(1,))
    assert home_snapshot["gf_last1"] == 9.0  # A's most recent home match (3rd)
    assert home_snapshot["ga_last1"] == 9.0

    away_snapshot = current_venue_form(matches, "A", "away", windows=(1,))
    assert away_snapshot["gf_last1"] == 3.0  # A's only away match (2nd)
    assert away_snapshot["ga_last1"] == 1.0


def test_current_venue_form_is_nan_for_a_team_with_no_history_at_that_venue():
    matches = _toy_matches_mixed_venue()
    # B only ever appears as the away team (match 1) in this toy set.
    snapshot = current_venue_form(matches, "B", "home", windows=(1,))
    assert pd.isna(snapshot["gf_last1"])
    assert pd.isna(snapshot["ga_last1"])
