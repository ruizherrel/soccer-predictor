import pandas as pd

from soccer_predictor.form_features import build_form_table


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
