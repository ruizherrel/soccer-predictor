"""Download and cache historical match results from football-data.co.uk.

Design: the direct CSV download is the primary data path. It needs no API
key and no extra scraper dependency. Completed seasons are cached
immutably; only the most recent (in-progress) season is re-downloaded on
every refresh.

football-data.co.uk is a small, occasionally-down hobbyist site (we hit a
real outage while building this), so every fetch also falls back to
datasets/football-datasets on GitHub: an automated daily mirror sourced
from the same site, missing only the bookmaker-odds columns (already
optional everywhere downstream).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# datasets/football-datasets only covers these five leagues (one directory
# per league, not a generic code like football-data.co.uk's). Folder names
# verified directly against the repo (github.com/datasets/football-datasets)
# since they don't follow a guessable pattern. Extend this map if the
# project ever grows beyond these five.
GITHUB_MIRROR_LEAGUE_SLUGS = {
    "E0": "premier-league",
    "SP1": "la-liga",
    "I1": "serie-a",
    "D1": "bundesliga",
    "F1": "ligue-1",
}
GITHUB_MIRROR_URL = (
    "https://raw.githubusercontent.com/datasets/football-datasets/main/"
    "datasets/{slug}/season-{season}.csv"
)

# football-data.co.uk team names are already fairly consistent within the
# same league over time, but a handful of clubs get renamed/rebranded.
# Extend this map if a new mismatch turns up in future seasons.
TEAM_NAME_MAP: dict[str, str] = {
    "Man United": "Man United",
    "Man Utd": "Man United",
}

# TheSportsDB names the same Liga MX club differently across seasons/rounds
# (e.g. official sponsor name in one, short name in another) rather than
# football-data.co.uk's fairly stable naming — confirmed by inspecting the
# distinct team names actually returned across all fetched seasons. Left
# unmapped, these would fragment one team's Elo/Pi rating history in two.
MEXICO_TEAM_NAME_MAP: dict[str, str] = {
    "CF America": "América",
    "Leon": "León",
    "FC Juarez": "Juárez",
    "Queretaro FC": "Querétaro",
    "Pumas": "Pumas UNAM",
    "Santos": "Santos Laguna",
    "Tigres": "Tigres UANL",
}


def _normalize_mexico_team(name: str) -> str:
    return MEXICO_TEAM_NAME_MAP.get(name, name)

_RAW_COLUMNS = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
}

_ODDS_COLUMNS = {
    "B365H": "b365_home",
    "B365D": "b365_draw",
    "B365A": "b365_away",
    "BWH": "bw_home",
    "BWD": "bw_draw",
    "BWA": "bw_away",
    "PSH": "ps_home",
    "PSD": "ps_draw",
    "PSA": "ps_away",
    "WHH": "wh_home",
    "WHD": "wh_draw",
    "WHA": "wh_away",
}


def normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def _candidate_urls(season: str, league: str) -> list[str]:
    urls = [BASE_URL.format(season=season, league=league)]
    slug = GITHUB_MIRROR_LEAGUE_SLUGS.get(league)
    if slug:
        urls.append(GITHUB_MIRROR_URL.format(slug=slug, season=season))
    return urls


def fetch_raw_season_csv(season: str, league: str) -> pd.DataFrame:
    """Download one season's raw match CSV and normalize its columns.

    Tries football-data.co.uk first, then the GitHub mirror if that fails
    (network error, non-2xx status, or an unparsable response).
    """
    from io import StringIO

    raw = None
    last_error: Exception | None = None
    for url in _candidate_urls(season, league):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            # encoding='utf-8-sig' strips a BOM some seasons ship with (see
            # probberechts/soccerdata issue #750 for the same bug in that
            # library); it's a no-op for the GitHub mirror, which has none.
            raw = pd.read_csv(StringIO(resp.content.decode("utf-8-sig")))
            break
        except (requests.RequestException, pd.errors.ParserError, UnicodeDecodeError) as exc:
            logger.warning("Season %s: %s unavailable (%s)", season, url, exc)
            last_error = exc

    if raw is None:
        raise RuntimeError(f"Could not fetch season {season} from any known source") from last_error

    available_raw_cols = [c for c in _RAW_COLUMNS if c in raw.columns]
    keep_cols = list(available_raw_cols)
    available_odds_cols = [c for c in _ODDS_COLUMNS if c in raw.columns]
    keep_cols += available_odds_cols

    df = raw[keep_cols].rename(columns={**_RAW_COLUMNS, **_ODDS_COLUMNS})
    df = df.dropna(subset=["date", "home_team", "away_team"])

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    df["season"] = season
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    return df.sort_values("date").reset_index(drop=True)


def _raw_cache_path(season: str, league: str) -> Path:
    return config.DATA_RAW / f"{league}_{season}.parquet"


# TheSportsDB's free test key ("123", documented at thesportsdb.com/free_sports_api)
# is used only for Mexico: football-data.co.uk's own Mexico endpoint has a
# broken TLS cert chain (verified independently via curl/WebFetch/requests),
# and the previously-used GitHub mirror (footballcsv/cache.footballdata)
# stopped updating in May 2024. TheSportsDB's bulk "full season" endpoint is
# capped at 5 sample events on the free tier, but eventsround.php isn't —
# looping round-by-round gets the complete regular-phase schedule. It does
# NOT include the post-season "liguilla" knockout (round numbering stops at
# the end of the regular phase), which this project accepts as a known gap.
THESPORTSDB_API_KEY = "123"
THESPORTSDB_LEAGUE_IDS = {"MEX": "4350"}
THESPORTSDB_ROUND_URL = (
    "https://www.thesportsdb.com/api/v1/json/{key}/eventsround.php?id={league_id}&r={round_num}&s={season}"
)
# Liga MX's regular phase is 17 matchdays (18 teams, single round-robin per
# tournament); the free key also rate-limits fairly aggressively, so pad a
# couple of rounds past that and retry 429s with backoff rather than hammer it.
THESPORTSDB_MAX_ROUNDS = 20


def _mx_pseudo_season(date: pd.Timestamp) -> str:
    """Liga MX plays two short tournaments a year (Apertura Jul-Dec,
    Clausura Jan-Jun) instead of one Aug-May season. Splitting by calendar
    month (rather than trusting the source's own season/round labels) keeps
    the Elo/Pi/Poisson season-regression logic, which assumes squads reset
    between "seasons", meaningful."""
    return f"{date.year}A" if date.month >= 7 else f"{date.year}C"


def _thesportsdb_round(league_id: str, season: str, round_num: int, retries: int = 6) -> list[dict]:
    url = THESPORTSDB_ROUND_URL.format(
        key=THESPORTSDB_API_KEY, league_id=league_id, round_num=round_num, season=season
    )
    for attempt in range(retries):
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and resp.text.strip():
            return resp.json().get("events") or []
        if resp.status_code != 429:
            resp.raise_for_status()
        logger.info("Rate-limited on round %s of season %s, backing off", round_num, season)
        time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"TheSportsDB kept rate-limiting round {round_num} of season {season} after {retries} retries")


def fetch_raw_mexico_season(season: str) -> pd.DataFrame:
    """Download one TheSportsDB season (e.g. "2024-2025") for Liga MX,
    round by round. A round with zero events ends the regular phase for
    that season; not-yet-played fixtures (no score yet, relevant for the
    current in-progress season) are dropped."""
    league_id = THESPORTSDB_LEAGUE_IDS["MEX"]
    rows = []
    for round_num in range(1, THESPORTSDB_MAX_ROUNDS + 1):
        events = _thesportsdb_round(league_id, season, round_num)
        if not events:
            break
        for e in events:
            if e.get("intHomeScore") is None or e.get("intAwayScore") is None:
                continue
            rows.append(e)
        time.sleep(3.0)

    home_goals = pd.Series([int(e["intHomeScore"]) for e in rows])
    away_goals = pd.Series([int(e["intAwayScore"]) for e in rows])
    result = np.where(home_goals > away_goals, "H", np.where(home_goals < away_goals, "A", "D"))

    df = pd.DataFrame(
        {
            "date": pd.to_datetime([e["dateEvent"] for e in rows]),
            "home_team": [_normalize_mexico_team(e["strHomeTeam"]) for e in rows],
            "away_team": [_normalize_mexico_team(e["strAwayTeam"]) for e in rows],
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": result,
        }
    )
    df["season"] = df["date"].map(_mx_pseudo_season)
    return df.sort_values("date").reset_index(drop=True)


def refresh(league: str) -> pd.DataFrame:
    """Fetch all configured seasons for `league`, caching completed ones and
    always re-fetching the most recent (in-progress) one. Returns the
    combined, deduplicated match table and writes it to
    config.matches_path(league).
    """
    source = config.LEAGUES[league]["source"]
    frames = []

    if source == "football-data":
        seasons = config.SEASONS
        current_season = seasons[-1]
        for season in seasons:
            cache_path = _raw_cache_path(season, league)
            if season != current_season and cache_path.exists():
                logger.info("Using cached season %s", season)
                frames.append(pd.read_parquet(cache_path))
                continue

            logger.info("Downloading season %s", season)
            df = fetch_raw_season_csv(season, league)
            df.to_parquet(cache_path, index=False)
            frames.append(df)
    elif source == "thesportsdb":
        seasons = config.MEXICO_SEASONS
        current_season = seasons[-1]
        for season in seasons:
            cache_path = _raw_cache_path(season, league)
            if season != current_season and cache_path.exists():
                logger.info("Using cached season %s", season)
                frames.append(pd.read_parquet(cache_path))
                continue

            logger.info("Downloading season %s", season)
            df = fetch_raw_mexico_season(season)
            df.to_parquet(cache_path, index=False)
            frames.append(df)
    else:
        raise ValueError(f"Unknown data source {source!r} for league {league!r}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "home_team", "away_team"])
    combined = combined.sort_values("date").reset_index(drop=True)

    matches_path = config.matches_path(league)
    combined.to_parquet(matches_path, index=False)
    logger.info("Wrote %d matches to %s", len(combined), matches_path)
    return combined


def load_matches(league: str) -> pd.DataFrame:
    matches_path = config.matches_path(league)
    if not matches_path.exists():
        return refresh(league)
    return pd.read_parquet(matches_path)
