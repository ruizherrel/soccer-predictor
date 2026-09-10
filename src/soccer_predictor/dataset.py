"""Assembles the full per-match feature table: Elo, Pi ratings, rolling
form, and season-boundary Poisson expected goals, all computed causally
(each row only ever sees information available strictly before that match).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from . import geo
from .elo import EloRatingSystem
from .form_features import attach_form_features, current_form
from .pi_ratings import PiRatingSystem
from .poisson_model import PoissonGoalModel


def _season_order(matches: pd.DataFrame) -> list[str]:
    # matches is sorted by date, so first-appearance order of `season` is
    # already chronological.
    return matches["season"].drop_duplicates().tolist()


# Champions League teams cross-referenced against the domestic leagues this
# project covers: name -> (domestic league code, domestic team name).
# Built from an automated near-duplicate search (reusing
# ingest._names_roughly_match) over each league's real team list, THEN
# manually verified and corrected -- the automated pass alone produced two
# dangerous false positives ("Inter Milan" substring-matching Serie A's
# "Milan", which is actually AC Milan, a different club; "Inter Club
# d'Escaldes", a small Andorran side, matching "Inter" the same way), and
# missed real pairs entirely because neither name is a substring of the
# other (e.g. "Atlético Madrid" vs "Ath Madrid", "Paris Saint-Germain" vs
# "Paris SG"). Every entry below was checked against that league's actual
# team list, not assumed. Extend this if a new mapping is needed; leagues
# we don't cover (Portugal, Turkey, smaller UEFA associations, etc.) simply
# aren't representable here.
UCL_DOMESTIC_TEAM_MAP: dict[str, tuple[str, str]] = {
    "Arsenal": ("E0", "Arsenal"),
    "Aston Villa": ("E0", "Aston Villa"),
    "Chelsea": ("E0", "Chelsea"),
    "Liverpool": ("E0", "Liverpool"),
    "Newcastle United": ("E0", "Newcastle"),
    "Tottenham Hotspur": ("E0", "Tottenham"),
    "Atlético Madrid": ("SP1", "Ath Madrid"),
    "Athletic Bilbao": ("SP1", "Ath Bilbao"),
    "Barcelona": ("SP1", "Barcelona"),
    "Girona": ("SP1", "Girona"),
    "Real Betis": ("SP1", "Betis"),
    "Real Madrid": ("SP1", "Real Madrid"),
    "Villarreal": ("SP1", "Villarreal"),
    "Bayer Leverkusen": ("D1", "Leverkusen"),
    "Bayern Munich": ("D1", "Bayern Munich"),
    "Borussia Dortmund": ("D1", "Dortmund"),
    "Eintracht Frankfurt": ("D1", "Ein Frankfurt"),
    "RB Leipzig": ("D1", "RB Leipzig"),
    "Stuttgart": ("D1", "Stuttgart"),
    "AC Milan": ("I1", "Milan"),
    "Atalanta": ("I1", "Atalanta"),
    "Bologna": ("I1", "Bologna"),
    "Inter Milan": ("I1", "Inter"),
    "Juventus": ("I1", "Juventus"),
    "Napoli": ("I1", "Napoli"),
    "Roma": ("I1", "Roma"),
    "Brest": ("F1", "Brest"),
    "Lille": ("F1", "Lille"),
    "Lyon": ("F1", "Lyon"),
    "Marseille": ("F1", "Marseille"),
    "Monaco": ("F1", "Monaco"),
    "Paris Saint-Germain": ("F1", "Paris SG"),
}


def _cross_league_seed_ratings(league: str) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """For a competition mixing teams from leagues we already cover (right
    now: just Champions League), returns each team's latest domestic Elo/Pi
    rating to use as its starting rating in this competition instead of the
    generic newly-promoted default -- a debutant or long-absent big club
    (e.g. Roma returning to the Champions League after 7 years) is nothing
    like an actual newly-promoted team, and treating it that way wastes
    real signal we already have. Only covers teams from the domestic
    leagues this project has data for; every other team keeps the
    unchanged default seeding."""
    from . import ingest

    if league != "UCL":
        return {}, {}

    by_domestic_league: dict[str, list[tuple[str, str]]] = {}
    for comp_name, (domestic_league, domestic_name) in UCL_DOMESTIC_TEAM_MAP.items():
        by_domestic_league.setdefault(domestic_league, []).append((comp_name, domestic_name))

    elo_seeds: dict[str, float] = {}
    pi_seeds: dict[str, tuple[float, float]] = {}
    for domestic_league, pairs in by_domestic_league.items():
        domestic_matches = ingest.load_matches(domestic_league)
        _ratings, elo, pi = build_ratings_features(domestic_matches)
        for comp_name, domestic_name in pairs:
            if domestic_name in elo.ratings:
                elo_seeds[comp_name] = elo.ratings[domestic_name]
            if domestic_name in pi.ratings:
                pi_seeds[comp_name] = pi.ratings[domestic_name]
    return elo_seeds, pi_seeds


def build_ratings_features(
    matches: pd.DataFrame, league: str | None = None
) -> tuple[pd.DataFrame, EloRatingSystem, PiRatingSystem]:
    """One row per match with pre-match Elo and Pi rating snapshots, plus the
    fully-updated rating systems (state after every match has been applied)
    for use in live/current-day predictions. `league` enables cross-league
    seeding (see _cross_league_seed_ratings) for competitions that need it;
    omit it (as every plain domestic league does) for unchanged behavior."""
    elo_seeds, pi_seeds = _cross_league_seed_ratings(league) if league else ({}, {})
    elo = EloRatingSystem(seed_ratings=elo_seeds)
    pi = PiRatingSystem(seed_ratings=pi_seeds)

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


def build_features(matches: pd.DataFrame, league: str | None = None) -> pd.DataFrame:
    matches = matches.sort_values("date").reset_index(drop=True)
    seasons_order = _season_order(matches)
    season_index = {s: i for i, s in enumerate(seasons_order)}

    ratings, _elo, _pi = build_ratings_features(matches, league)
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

    # Altitude/travel are static per team pair (no match history involved),
    # so no leakage risk in computing them directly rather than causally.
    # Appended after the merges above (rather than added to `matches`
    # upfront) so they don't get swept up by attach_form_features's own
    # "starts with home_/away_" column-selection, which would otherwise
    # re-select and duplicate-merge them (pandas then suffixes both copies
    # _x/_y, silently breaking the plain "home_altitude_m" name).
    alt_df = pd.DataFrame(
        [geo.altitude_features(h, a) for h, a in zip(matches["home_team"], matches["away_team"])]
    )
    out = pd.concat([out.reset_index(drop=True), alt_df], axis=1)

    out["season_index"] = out["season"].map(season_index)
    return out


def build_and_save_features(league: str) -> pd.DataFrame:
    from . import ingest

    matches = ingest.load_matches(league)
    features = build_features(matches, league)
    features.to_parquet(config.features_path(league), index=False)
    return features


def load_features(league: str) -> pd.DataFrame:
    features_path = config.features_path(league)
    if not features_path.exists():
        return build_and_save_features(league)
    return pd.read_parquet(features_path)


def build_live_features(
    matches: pd.DataFrame, home_team: str, away_team: str, league: str | None = None
) -> tuple[pd.DataFrame, PoissonGoalModel]:
    """Single-row feature table for a hypothetical fixture "as of today",
    for interactive predictions (the Streamlit app). Also returns the fitted
    Poisson model so the caller can render a scoreline probability grid."""
    matches = matches.sort_values("date").reset_index(drop=True)

    _ratings_df, elo, pi = build_ratings_features(matches, league)
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
        **geo.altitude_features(home_team, away_team),
        "poisson_lambda_home": lam,
        "poisson_lambda_away": mu,
    }
    return pd.DataFrame([row]), poisson_model
