"""Assembles the full per-match feature table: Elo, Pi ratings, rolling
form, and season-boundary Poisson expected goals, all computed causally
(each row only ever sees information available strictly before that match).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from .elo import EloRatingSystem
from .form_features import attach_form_features, current_form
from .pi_ratings import PiRatingSystem
from .poisson_model import PoissonGoalModel


def _season_order(matches: pd.DataFrame) -> list[str]:
    # matches is sorted by date, so first-appearance order of `season` is
    # already chronological.
    return matches["season"].drop_duplicates().tolist()


def build_ratings_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, EloRatingSystem, PiRatingSystem]:
    """One row per match with pre-match Elo and Pi rating snapshots, plus the
    fully-updated rating systems (state after every match has been applied)
    for use in live/current-day predictions."""
    elo = EloRatingSystem()
    pi = PiRatingSystem()

    current_season = None
    prev_teams: set[str] | None = None
    season_teams_cache: dict[str, set[str]] = {}

    rows = []
    for row in matches.itertuples(index=False):
        if row.season != current_season:
            if row.season not in season_teams_cache:
                season_df = matches[matches["season"] == row.season]
                season_teams_cache[row.season] = set(season_df["home_team"]) | set(season_df["away_team"])
            season_teams = season_teams_cache[row.season]

            if prev_teams is not None:
                for team in season_teams - prev_teams:
                    elo.ensure_seeded(team)
                    pi.ensure_seeded(team)
                elo.new_season_regression()
                pi.new_season_regression()

            prev_teams = season_teams
            current_season = row.season

        elo_home, elo_away = elo.snapshot(row.home_team, row.away_team)
        pi_home, pi_away = pi.snapshot(row.home_team, row.away_team)

        rows.append(
            {
                "date": row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "elo_home": elo_home,
                "elo_away": elo_away,
                "elo_diff": elo_home - elo_away,
                "pi_home": pi_home,
                "pi_away": pi_away,
                "pi_diff": pi_home - pi_away,
            }
        )

        elo.update(row.home_team, row.away_team, row.home_goals, row.away_goals)
        pi.update(row.home_team, row.away_team, row.home_goals, row.away_goals)

    return pd.DataFrame(rows), elo, pi


def build_poisson_features(matches: pd.DataFrame, seasons_order: list[str]) -> pd.DataFrame:
    """One row per match with season-boundary Poisson expected-goals features.

    The Poisson model is refit once per season using only prior seasons'
    matches (never mid-season), then used to score every fixture in that
    season. The very first season has no prior data, so its rows get NaN
    (XGBoost handles missing values natively).
    """
    frames = []
    for i, season in enumerate(seasons_order):
        season_matches = matches[matches["season"] == season]
        cols = ["date", "home_team", "away_team"]

        if i == 0:
            out = season_matches[cols].copy()
            for c in ("poisson_lambda_home", "poisson_lambda_away", "poisson_p_home", "poisson_p_draw", "poisson_p_away"):
                out[c] = np.nan
            frames.append(out)
            continue

        train_matches = matches[matches["season"].isin(seasons_order[:i])]
        model = PoissonGoalModel().fit(train_matches, as_of_date=season_matches["date"].min())

        lam, mu = model.predict_lambdas_bulk(season_matches["home_team"], season_matches["away_team"])
        p_home, p_draw, p_away = [], [], []
        for h, a, l, m in zip(season_matches["home_team"], season_matches["away_team"], lam, mu):
            ph, pd_, pa = model.match_probabilities(h, a, lam=l, mu=m)
            p_home.append(ph)
            p_draw.append(pd_)
            p_away.append(pa)

        out = season_matches[cols].copy()
        out["poisson_lambda_home"] = lam
        out["poisson_lambda_away"] = mu
        out["poisson_p_home"] = p_home
        out["poisson_p_draw"] = p_draw
        out["poisson_p_away"] = p_away
        frames.append(out)

    return pd.concat(frames, ignore_index=True)


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.sort_values("date").reset_index(drop=True)
    seasons_order = _season_order(matches)
    season_index = {s: i for i, s in enumerate(seasons_order)}

    ratings, _elo, _pi = build_ratings_features(matches)
    poisson_feats = build_poisson_features(matches, seasons_order)
    form = attach_form_features(matches)

    out = matches.merge(ratings, on=["date", "home_team", "away_team"], how="left")
    out = out.merge(poisson_feats, on=["date", "home_team", "away_team"], how="left")

    form_cols = [
        c
        for c in form.columns
        if (c.startswith("home_") or c.startswith("away_")) and c not in ("home_team", "away_team")
    ]
    out = out.merge(
        form[["date", "home_team", "away_team", *form_cols]],
        on=["date", "home_team", "away_team"],
        how="left",
    )

    out["season_index"] = out["season"].map(season_index)
    return out


def build_and_save_features() -> pd.DataFrame:
    from . import ingest

    matches = ingest.load_matches()
    features = build_features(matches)
    features.to_parquet(config.FEATURES_PATH, index=False)
    return features


def load_features() -> pd.DataFrame:
    if not config.FEATURES_PATH.exists():
        return build_and_save_features()
    return pd.read_parquet(config.FEATURES_PATH)


def build_live_features(
    matches: pd.DataFrame, home_team: str, away_team: str
) -> tuple[pd.DataFrame, PoissonGoalModel]:
    """Single-row feature table for a hypothetical fixture "as of today",
    for interactive predictions (the Streamlit app). Also returns the fitted
    Poisson model so the caller can render a scoreline probability grid."""
    matches = matches.sort_values("date").reset_index(drop=True)

    _ratings_df, elo, pi = build_ratings_features(matches)
    elo_home, elo_away = elo.snapshot(home_team, away_team)
    pi_home, pi_away = pi.snapshot(home_team, away_team)

    home_form = current_form(matches, home_team)
    away_form = current_form(matches, away_team)

    poisson_model = PoissonGoalModel().fit(matches, as_of_date=matches["date"].max())
    lam, mu = poisson_model.predict_lambdas(home_team, away_team)

    row = {
        "elo_home": elo_home,
        "elo_away": elo_away,
        "elo_diff": elo_home - elo_away,
        "pi_home": pi_home,
        "pi_away": pi_away,
        "pi_diff": pi_home - pi_away,
        "home_ppg_last5": home_form["ppg_last5"],
        "home_gf_last5": home_form["gf_last5"],
        "home_ga_last5": home_form["ga_last5"],
        "home_ppg_last10": home_form["ppg_last10"],
        "home_gf_last10": home_form["gf_last10"],
        "home_ga_last10": home_form["ga_last10"],
        "home_rest_days": home_form["rest_days"],
        "away_ppg_last5": away_form["ppg_last5"],
        "away_gf_last5": away_form["gf_last5"],
        "away_ga_last5": away_form["ga_last5"],
        "away_ppg_last10": away_form["ppg_last10"],
        "away_gf_last10": away_form["gf_last10"],
        "away_ga_last10": away_form["ga_last10"],
        "away_rest_days": away_form["rest_days"],
        "poisson_lambda_home": lam,
        "poisson_lambda_away": mu,
    }
    return pd.DataFrame([row]), poisson_model
