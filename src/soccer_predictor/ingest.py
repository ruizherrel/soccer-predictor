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


# Same issue as Mexico, found the same way: diffed the ~100 distinct names
# actually returned across all fetched Champions League seasons (36 clubs/
# season with heavy year-to-year turnover from qualification, so a high
# raw count is expected — these two are genuine duplicates, not just two
# different clubs). Checked programmatically for near-duplicates too
# (accent-insensitive substring), which also flagged "Viking"/"Víkingur
# Reykjavík" — confirmed those are two different real clubs (Norway vs
# Iceland) and left unmapped; a substring match alone isn't reliable
# enough to auto-merge without checking.
UCL_TEAM_NAME_MAP: dict[str, str] = {
    "Atletico Madrid": "Atlético Madrid",
    "Paris SG": "Paris Saint-Germain",
}


def _normalize_ucl_team(name: str) -> str:
    return UCL_TEAM_NAME_MAP.get(name, name)


# Older MLS seasons in TheSportsDB drop the "FC" some clubs added later —
# found the same way (checked all distinct name pairs for near-duplicates,
# confirmed these two are the same club rather than a real coincidence).
MLS_TEAM_NAME_MAP: dict[str, str] = {
    "New York City": "New York City FC",
    "Seattle Sounders": "Seattle Sounders FC",
}


def _normalize_mls_team(name: str) -> str:
    return MLS_TEAM_NAME_MAP.get(name, name)


# Same issue again, this time Eredivisie dropping (mostly) or adding club
# prefixes/suffixes across seasons. Canonicalized to whichever variant the
# most recent (2026-2027) season actually uses, checked per pair rather
# than assumed -- note Sparta went the opposite direction from the rest
# (gained "Rotterdam" instead of losing a prefix).
N1_TEAM_NAME_MAP: dict[str, str] = {
    "SC Cambuur": "Cambuur",
    "FC Groningen": "Groningen",
    "FC Twente": "Twente",
    "FC Utrecht": "Utrecht",
    "SC Heerenveen": "Heerenveen",
    "SC Heracles Almelo": "Heracles Almelo",
    "SC Telstar": "Telstar",
    "Sparta": "Sparta Rotterdam",
}


def _normalize_n1_team(name: str) -> str:
    return N1_TEAM_NAME_MAP.get(name, name)


# Primeira Liga, same issue. "Estoril-Praia"/"Estoril Praia" (hyphen vs
# space) is a good example of why this needs checking per pair rather than
# trusting the automated near-duplicate search alone: neither is a
# substring of the other (the hyphen breaks it), so it wasn't even
# flagged as a candidate -- found by reading the actual full team list.
P1_TEAM_NAME_MAP: dict[str, str] = {
    "AVS Futebol SAD": "AVS",
    "CD Nacional de Madeira": "Nacional de Madeira",
    "FC Porto": "Porto",
    "Guimaraes": "Vitória de Guimarães",
    "Maritimo": "Marítimo",
    "Estoril-Praia": "Estoril Praia",
}


def _normalize_p1_team(name: str) -> str:
    return P1_TEAM_NAME_MAP.get(name, name)


# Conference League draws from ~180 distinct clubs across just 3 seasons
# (an even wider qualifying pyramid than Champions League), which makes an
# exhaustive manual review of every possible pair impractical the way it
# was for the smaller domestic leagues -- these five were found by
# combining the automated near-duplicate search with a manual read of the
# full team list, not an exhaustive pairwise check. "Jagiellonia
# Bialystok"/"Jagiellonia Białystok" is a good example of why the search
# alone isn't enough: Polish "ł" has no NFKD decomposition to a plain "l"
# the way accented Latin letters do, so accent-stripping just drops it
# entirely ("Białystok" -> "Biaystok"), which doesn't match "Bialystok" as
# a substring either. "TNS"/"The New Saints" is a plain abbreviation, not
# a spelling variant, so no substring relationship exists at all. If a
# rating for an unfamiliar club here looks obviously wrong, check for a
# further unmapped alias before trusting it.
UECL_TEAM_NAME_MAP: dict[str, str] = {
    "FC Astana": "Astana",
    "Jagiellonia Bialystok": "Jagiellonia Białystok",
    "Istanbul Basaksehir": "İstanbul Başakşehir",
    "Shamrock": "Shamrock Rovers",
    "TNS": "The New Saints",
}


def _normalize_uecl_team(name: str) -> str:
    return UECL_TEAM_NAME_MAP.get(name, name)


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
# Verified directly against the API (search_all_leagues.php / all_leagues.php
# / searchteams.php) rather than guessed — TheSportsDB's naming/ID scheme
# isn't predictable from the league code.
THESPORTSDB_LEAGUE_IDS = {
    "E0": "4328",
    "SP1": "4335",
    "D1": "4331",
    "I1": "4332",
    "F1": "4334",
    "N1": "4337",
    "P1": "4344",
    "MEX": "4350",
    "MLS": "4346",
    "UCL": "4480",
    "UECL": "5071",
}
THESPORTSDB_ROUND_URL = (
    "https://www.thesportsdb.com/api/v1/json/{key}/eventsround.php?id={league_id}&r={round_num}&s={season}"
)
# High enough to cover MLS (~34 matchdays across 30 teams, some bye weeks);
# harmless for shorter competitions since the fetch loop breaks on the
# first empty round regardless (Liga MX stops at ~18, Champions League's
# league phase at 9). The free key also rate-limits fairly aggressively, so
# retry 429s with backoff rather than hammer it.
THESPORTSDB_MAX_ROUNDS = 40


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
        # 429 (rate limit) and 5xx (server-side, seen live: 503) are both
        # transient — retry with backoff. Anything else (404, auth, etc.)
        # is a real failure, not worth retrying.
        if resp.status_code != 429 and resp.status_code < 500:
            resp.raise_for_status()
        logger.info(
            "TheSportsDB %s on round %s of season %s, backing off", resp.status_code, round_num, season
        )
        time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"TheSportsDB kept failing round {round_num} of season {season} after {retries} retries")


def _thesportsdb_normalize(league: str, name: str) -> str:
    if league == "MEX":
        return _normalize_mexico_team(name)
    if league == "UCL":
        return _normalize_ucl_team(name)
    if league == "MLS":
        return _normalize_mls_team(name)
    if league == "N1":
        return _normalize_n1_team(name)
    if league == "P1":
        return _normalize_p1_team(name)
    if league == "UECL":
        return _normalize_uecl_team(name)
    return name


def fetch_raw_thesportsdb_season(league: str, season: str) -> pd.DataFrame:
    """Download one TheSportsDB season for `league`, round by round. A
    round with zero events ends the competition for that season (its
    regular/league phase, for Mexico and Champions League — see their
    config.LEAGUES notes); not-yet-played fixtures (no score yet, relevant
    for the current in-progress season) are dropped."""
    league_id = THESPORTSDB_LEAGUE_IDS[league]
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
            "home_team": [_thesportsdb_normalize(league, e["strHomeTeam"]) for e in rows],
            "away_team": [_thesportsdb_normalize(league, e["strAwayTeam"]) for e in rows],
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": result,
        }
    )
    # Only Mexico plays two short tournaments within one source "season"
    # (Apertura Jul-Dec, Clausura Jan-Jun); everyone else's season slug
    # already matches one real competitive block.
    df["season"] = df["date"].map(_mx_pseudo_season) if league == "MEX" else season
    return df.sort_values("date").reset_index(drop=True)


def _current_thesportsdb_season(today: pd.Timestamp | None = None) -> str:
    today = today or pd.Timestamp.now()
    return f"{today.year}-{today.year + 1}" if today.month >= 8 else f"{today.year - 1}-{today.year}"


def _names_roughly_match(name_a: str, name_b: str) -> bool:
    """True only when one (accent-stripped, lowercased) name is a
    substring of the other. Deliberately conservative: no token splitting
    or fuzzy distance, since e.g. "Real Madrid" and "Real Sociedad" share a
    token but are obviously different clubs — a false "next fixture" match
    is worse than missing a real one, which just falls back to the
    hypothetical-matchup framing the app already uses."""
    import unicodedata

    def norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        return s.lower().strip()

    a, b = norm(name_a), norm(name_b)
    return a == b or a in b or b in a


def fetch_upcoming_fixtures(league: str, rounds_ahead: int = 2) -> pd.DataFrame:
    """Real scheduled upcoming fixtures for `league` via TheSportsDB
    (eventsnextleague.php, the obvious endpoint, is capped at 1 result on
    the free tier; eventsround.php isn't, same as the Mexico backfill).
    Scans forward from round 1 for the first round with an unplayed
    fixture, then also collects `rounds_ahead - 1` further rounds.

    Mexico's team names are normalized the same way as the historical data
    (same source), so they match exactly. Every other league's names are
    TheSportsDB's own and don't always match football-data.co.uk's
    abbreviated style (e.g. "Athletic Bilbao" vs "Ath Bilbao") — match
    against them with _names_roughly_match, not equality.
    """
    league_id = THESPORTSDB_LEAGUE_IDS.get(league)
    if league_id is None:
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    league_seasons = config.LEAGUES[league].get("seasons")
    season = league_seasons[-1] if league_seasons else _current_thesportsdb_season()

    current_round = None
    rows = []
    for round_num in range(1, THESPORTSDB_MAX_ROUNDS + 1):
        events = _thesportsdb_round(league_id, season, round_num)
        if not events:
            break

        unplayed = [e for e in events if e.get("intHomeScore") is None]
        if unplayed and current_round is None:
            current_round = round_num
        if current_round is not None:
            rows.extend(unplayed)
            if round_num >= current_round + rounds_ahead - 1:
                break
        time.sleep(1.5)

    if not rows:
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    # strTimestamp (dateEvent + strTime combined) is UTC — confirmed live
    # against strTimeLocal for a Spain fixture (19:00 vs 21:00 CEST, a
    # UTC+2 gap). Falls back to date-only (midnight) for the rare event
    # missing it rather than dropping the row.
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [e.get("strTimestamp") or e["dateEvent"] for e in rows], utc=True
            ),
            "home_team": [_thesportsdb_normalize(league, e["strHomeTeam"]) for e in rows],
            "away_team": [_thesportsdb_normalize(league, e["strAwayTeam"]) for e in rows],
        }
    )
    return df.sort_values("date").reset_index(drop=True)


def find_upcoming_fixture(fixtures: pd.DataFrame, home_team: str, away_team: str) -> pd.Timestamp | None:
    """UTC date+time of the next real fixture between these two teams (in
    either order), or None if not found in `fixtures` (either genuinely not
    scheduled soon, or a naming mismatch — see _names_roughly_match)."""
    for _, row in fixtures.iterrows():
        teams_match = _names_roughly_match(row["home_team"], home_team) and _names_roughly_match(
            row["away_team"], away_team
        )
        teams_match_swapped = _names_roughly_match(row["home_team"], away_team) and _names_roughly_match(
            row["away_team"], home_team
        )
        if teams_match or teams_match_swapped:
            return row["date"]
    return None


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
        seasons = config.LEAGUES[league]["seasons"]
        current_season = seasons[-1]
        for season in seasons:
            cache_path = _raw_cache_path(season, league)
            if season != current_season and cache_path.exists():
                logger.info("Using cached season %s", season)
                frames.append(pd.read_parquet(cache_path))
                continue

            logger.info("Downloading season %s", season)
            df = fetch_raw_thesportsdb_season(league, season)
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
