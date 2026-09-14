"""Rolling recent-form features: points-per-game, goals for/against, rest days.

Every feature is computed with `shift(1)` before the rolling window, so a
match's features only ever look at that team's matches strictly earlier in
time. This is the single highest-leakage-risk part of the whole pipeline;
`tests/test_form_features.py` checks it directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _long_format(matches: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "date": matches["date"],
            "season": matches["season"],
            "team": matches["home_team"],
            "goals_for": matches["home_goals"],
            "goals_against": matches["away_goals"],
        }
    )
    away = pd.DataFrame(
        {
            "date": matches["date"],
            "season": matches["season"],
            "team": matches["away_team"],
            "goals_for": matches["away_goals"],
            "goals_against": matches["home_goals"],
        }
    )
    for df in (home, away):
        df["points"] = np.select(
            [df["goals_for"] > df["goals_against"], df["goals_for"] == df["goals_against"]],
            [3, 1],
            default=0,
        )
    long_df = pd.concat([home, away], ignore_index=True)
    return long_df.sort_values(["team", "date"]).reset_index(drop=True)


def build_form_table(matches: pd.DataFrame, windows: tuple[int, ...] = config.FORM_WINDOWS) -> pd.DataFrame:
    """Returns one row per (date, team) with pre-match rolling form features."""
    long_df = _long_format(matches)
    grouped = long_df.groupby("team")

    for window in windows:
        long_df[f"ppg_last{window}"] = grouped["points"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        long_df[f"gf_last{window}"] = grouped["goals_for"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        long_df[f"ga_last{window}"] = grouped["goals_against"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

    long_df["rest_days"] = grouped["date"].transform(lambda s: (s - s.shift(1)).dt.days)

    feature_cols = [c for c in long_df.columns if c not in ("goals_for", "goals_against", "points", "season")]
    return long_df[feature_cols]


def current_form(
    matches: pd.DataFrame,
    team: str,
    windows: tuple[int, ...] = config.FORM_WINDOWS,
    as_of: pd.Timestamp | None = None,
) -> dict[str, float]:
    """Unshifted form snapshot for a team as of today, for live predictions
    on a hypothetical fixture (not yet in the historical match table).
    Unlike the historical per-row features, this includes the team's most
    recent completed match rather than excluding it."""
    as_of = as_of or pd.Timestamp.now().normalize()
    long_df = _long_format(matches)
    team_matches = long_df[long_df["team"] == team].sort_values("date")

    if team_matches.empty:
        feats = {f"{stat}_last{w}": np.nan for w in windows for stat in ("ppg", "gf", "ga")}
        feats["rest_days"] = np.nan
        return feats

    feats = {}
    for window in windows:
        tail = team_matches.tail(window)
        feats[f"ppg_last{window}"] = float(tail["points"].mean())
        feats[f"gf_last{window}"] = float(tail["goals_for"].mean())
        feats[f"ga_last{window}"] = float(tail["goals_against"].mean())

    last_date = team_matches["date"].max()
    feats["rest_days"] = float((as_of - last_date).days)
    return feats


def attach_form_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Merges home/away rolling-form features onto the match table."""
    form = build_form_table(matches)
    feature_cols = [c for c in form.columns if c not in ("date", "team")]

    home_form = form.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in feature_cols}})
    away_form = form.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in feature_cols}})

    out = matches.merge(home_form, on=["date", "home_team"], how="left")
    out = out.merge(away_form, on=["date", "away_team"], how="left")
    return out


def _venue_format(matches: pd.DataFrame, venue: str) -> pd.DataFrame:
    """One row per (date, team), restricted to matches that team played at
    `venue` ("home" or "away") -- the venue-specific counterpart to
    _long_format, which mixes both into one series per team. A rolling
    window computed from this only ever sees that team's own past matches
    at this same venue: a match they played at the other venue was never
    in this series to begin with, not merely excluded from the window."""
    team_col = "home_team" if venue == "home" else "away_team"
    for_col = "home_goals" if venue == "home" else "away_goals"
    against_col = "away_goals" if venue == "home" else "home_goals"
    df = pd.DataFrame(
        {
            "date": matches["date"],
            "team": matches[team_col],
            "goals_for": matches[for_col],
            "goals_against": matches[against_col],
        }
    )
    return df.sort_values(["team", "date"]).reset_index(drop=True)


def build_venue_form_table(
    matches: pd.DataFrame, venue: str, windows: tuple[int, ...] = config.FORM_WINDOWS
) -> pd.DataFrame:
    """Returns one row per (date, team) with pre-match rolling goals
    for/against, computed only from that team's past matches at `venue`
    ("home" or "away"). Distinct signal from build_form_table's mixed-venue
    rolling stats: e.g. a team that concedes far more specifically playing
    away than their overall average suggests (real cases seen: a visiting
    team with a strongly positive away goal difference overturning what
    the general table/an AI model's probability both suggested was a clear
    home favorite)."""
    df = _venue_format(matches, venue)
    grouped = df.groupby("team")

    for window in windows:
        df[f"gf_last{window}"] = grouped["goals_for"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"ga_last{window}"] = grouped["goals_against"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

    feature_cols = [c for c in df.columns if c not in ("goals_for", "goals_against")]
    return df[feature_cols]


def current_venue_form(
    matches: pd.DataFrame,
    team: str,
    venue: str,
    windows: tuple[int, ...] = config.FORM_WINDOWS,
) -> dict[str, float]:
    """Unshifted venue-specific form snapshot for a team as of today (for
    live predictions on a hypothetical fixture), mirroring current_form but
    restricted to that team's past matches at `venue` only."""
    df = _venue_format(matches, venue)
    team_matches = df[df["team"] == team].sort_values("date")

    if team_matches.empty:
        return {f"{stat}_last{w}": np.nan for w in windows for stat in ("gf", "ga")}

    feats = {}
    for window in windows:
        tail = team_matches.tail(window)
        feats[f"gf_last{window}"] = float(tail["goals_for"].mean())
        feats[f"ga_last{window}"] = float(tail["goals_against"].mean())
    return feats


def attach_venue_form_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Merges the home team's home-venue and the away team's away-venue
    rolling goals for/against onto the match table, prefixed
    home_venue_/away_venue_ respectively (distinct from attach_form_features'
    home_/away_ mixed-venue columns)."""
    home_form = build_venue_form_table(matches, "home")
    away_form = build_venue_form_table(matches, "away")

    home_cols = [c for c in home_form.columns if c not in ("date", "team")]
    away_cols = [c for c in away_form.columns if c not in ("date", "team")]

    home_renamed = home_form.rename(columns={"team": "home_team", **{c: f"home_venue_{c}" for c in home_cols}})
    away_renamed = away_form.rename(columns={"team": "away_team", **{c: f"away_venue_{c}" for c in away_cols}})

    out = matches.merge(home_renamed, on=["date", "home_team"], how="left")
    out = out.merge(away_renamed, on=["date", "away_team"], how="left")
    return out
