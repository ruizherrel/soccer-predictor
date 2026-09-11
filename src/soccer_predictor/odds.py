"""Live betting odds via The Odds API (the-odds-api.com), used to show
market comparison and +EV/Kelly stake sizing on the app's prediction
screen.

Sport keys below were confirmed live against The Odds API's own docs
(https://the-odds-api.com/sports-odds-data/soccer-odds.html and
.../sports-apis.html) on 2026-09-11 -- not assumed from memory. Liga de
Expansión MX (MX2, a lower Mexican division) has no listed key there and
is simply not covered by this provider; that's a real coverage gap, same
kind already documented elsewhere in this project for other data sources.

Requires an ODDS_API_KEY (free tier: 500 credits/month, no card required
per their pricing page at signup time -- verify current terms yourself
before relying on it). A single request for one sport costs
`regions x markets` credits regardless of how many matches it returns
(confirmed in their v4 docs), which is why this queries one market and one
region: 1 credit per league per cache refresh.

Every function here degrades to returning empty/None on any failure --
missing key, unsupported league, network error, unexpected response shape
-- and never raises, so a third-party odds provider being down or
unconfigured can never break the model's own prediction (same principle as
ingest.fetch_upcoming_fixtures).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import requests

from .ingest import _names_roughly_match

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

SPORT_KEYS: dict[str, str] = {
    "E0": "soccer_epl",
    "SP1": "soccer_spain_la_liga",
    "D1": "soccer_germany_bundesliga",
    "I1": "soccer_italy_serie_a",
    "F1": "soccer_france_ligue_one",
    "N1": "soccer_netherlands_eredivisie",
    "P1": "soccer_portugal_primeira_liga",
    "MEX": "soccer_mexico_ligamx",
    "MLS": "soccer_usa_mls",
    "UCL": "soccer_uefa_champs_league",
    "UECL": "soccer_uefa_europa_conference_league",
    "LIB": "soccer_conmebol_copa_libertadores",
    "MLB": "baseball_mlb",
}

# Bookmaker region to query per league. The Odds API's docs don't publish a
# region -> bookmaker coverage map beyond examples, so "eu" (broad
# international/European coverage) is the reasonable default for soccer,
# and "us" for MLB. Deliberately a single region: cost is regions x
# markets per call, so querying more regions multiplies the free tier's
# monthly quota for no benefit here (this project only needs one quoted
# price per outcome, not a cross-book comparison).
REGION_BY_LEAGUE = {"MLB": "us"}
DEFAULT_REGION = "eu"

ODDS_COLUMNS = ["date", "home_team", "away_team", "odds_home", "odds_draw", "odds_away"]


def _api_key() -> str | None:
    """Reads ODDS_API_KEY from Streamlit secrets (how this is configured on
    Streamlit Cloud and in local `.streamlit/secrets.toml`), falling back
    to a plain environment variable for use outside a Streamlit run (e.g.
    scripts, tests). Returns None rather than raising when unset."""
    try:
        import streamlit as st

        if "ODDS_API_KEY" in st.secrets:
            return st.secrets["ODDS_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ODDS_API_KEY")


def is_configured() -> bool:
    """Whether an API key is available. Lets callers (the app) show a
    helpful "you need to configure this" message instead of silently
    showing nothing, which would look like a bug rather than a setup step."""
    return _api_key() is not None


def fetch_odds(league: str, timeout: float = 10.0) -> pd.DataFrame:
    """Upcoming h2h (moneyline) odds for every scheduled match in `league`,
    one row per match with decimal odds. `odds_draw` is NaN for sports
    without a draw outcome (e.g. MLB) or when a particular bookmaker didn't
    quote one. Returns an empty DataFrame (never raises) when there's no
    API key configured, the league isn't covered by this provider, or the
    request fails for any reason."""
    empty = pd.DataFrame(columns=ODDS_COLUMNS)

    sport_key = SPORT_KEYS.get(league)
    api_key = _api_key()
    if sport_key is None or not api_key:
        return empty

    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/sports/{sport_key}/odds",
            params={
                "apiKey": api_key,
                "regions": REGION_BY_LEAGUE.get(league, DEFAULT_REGION),
                "markets": "h2h",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception:
        logging.getLogger(__name__).exception("fetch_odds failed for %s", league)
        return empty

    rows = []
    for event in events:
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        if not home_team or not away_team:
            continue

        # First bookmaker with a usable h2h market -- a single consistent
        # quoted price, not a cross-book average, matching the same
        # single-source assumption value_betting.py already makes with
        # historical Bet365-only closing odds.
        outcomes = None
        for bookmaker in event.get("bookmakers") or []:
            for market in bookmaker.get("markets", []):
                if market.get("key") == "h2h":
                    outcomes = market.get("outcomes")
                    break
            if outcomes:
                break
        if not outcomes:
            continue

        price_by_name = {o["name"]: o["price"] for o in outcomes}
        odds_home = price_by_name.get(home_team)
        odds_away = price_by_name.get(away_team)
        if odds_home is None or odds_away is None:
            continue

        rows.append(
            {
                "date": event.get("commence_time"),
                "home_team": home_team,
                "away_team": away_team,
                "odds_home": float(odds_home),
                "odds_draw": float(price_by_name["Draw"]) if "Draw" in price_by_name else np.nan,
                "odds_away": float(odds_away),
            }
        )

    if not rows:
        return empty
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def find_match_odds(odds_df: pd.DataFrame, home_team: str, away_team: str) -> dict | None:
    """Odds for the fixture between these two teams (in either order,
    matched fuzzily via ingest._names_roughly_match since The Odds API's
    team names don't necessarily match this project's own naming any more
    than TheSportsDB's do), oriented to (home_team, away_team) regardless
    of which side the provider called "home". None if no match is found."""
    for _, row in odds_df.iterrows():
        if _names_roughly_match(row["home_team"], home_team) and _names_roughly_match(row["away_team"], away_team):
            return {"odds_home": row["odds_home"], "odds_draw": row["odds_draw"], "odds_away": row["odds_away"]}
        if _names_roughly_match(row["home_team"], away_team) and _names_roughly_match(row["away_team"], home_team):
            return {"odds_home": row["odds_away"], "odds_draw": row["odds_draw"], "odds_away": row["odds_home"]}
    return None
