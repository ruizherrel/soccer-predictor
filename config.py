"""Central configuration for the soccer-predictor pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

for _dir in (DATA_RAW, DATA_PROCESSED, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Leagues the app supports. "source" selects the ingest.py code path:
# - "football-data": football-data.co.uk's per-season CSVs (see SEASONS).
# - "thesportsdb": TheSportsDB's free API. Used where football-data.co.uk
#   either has no coverage at all (Champions League) or its endpoint is
#   unusable (Mexico/MLS's "extra leagues" endpoint has a broken TLS cert
#   chain, verified independently; a previously-used GitHub mirror for
#   Mexico stopped updating in May 2024). Each such league carries its own
#   "seasons" list, in whatever slug format TheSportsDB expects for it
#   (confirmed live per league, not assumed) — the last entry is treated as
#   the current, still-in-progress one and always re-fetched.
LEAGUES = {
    "E0": {"name": "Premier League (Inglaterra)", "source": "football-data"},
    "SP1": {"name": "La Liga (España)", "source": "football-data"},
    "D1": {"name": "Bundesliga (Alemania)", "source": "football-data"},
    "I1": {"name": "Serie A (Italia)", "source": "football-data"},
    "F1": {"name": "Ligue 1 (Francia)", "source": "football-data"},
    "N1": {
        "name": "Eredivisie (Holanda)",
        "source": "thesportsdb",
        "seasons": ["2022-2023", "2023-2024", "2024-2025", "2025-2026", "2026-2027"],
    },
    "P1": {
        "name": "Primeira Liga (Portugal)",
        "source": "thesportsdb",
        "seasons": ["2022-2023", "2023-2024", "2024-2025", "2025-2026", "2026-2027"],
    },
    "MEX": {
        "name": "Liga MX (México)",
        "source": "thesportsdb",
        "seasons": ["2022-2023", "2023-2024", "2024-2025", "2025-2026", "2026-2027"],
    },
    "MLS": {
        "name": "MLS (Estados Unidos)",
        "source": "thesportsdb",
        "seasons": ["2022", "2023", "2024", "2025", "2026"],
    },
    "UCL": {
        "name": "UEFA Champions League",
        "source": "thesportsdb",
        # Swiss-model league phase only (8 matchdays, single table across
        # all 36 clubs) started 2024-25; the prior 32-team group-stage
        # format doesn't fit the same round-by-round fetch, so history
        # starts here rather than mixing two incompatible formats.
        "seasons": ["2024-2025", "2025-2026", "2026-2027"],
        "notice": (
            "Solo cubre la fase de liga (jornadas 1-8), no la eliminatoria "
            "posterior (octavos en adelante). Además, al ser una competencia "
            "con solo 8 partidos por equipo y un grupo de clubes que cambia "
            "cada temporada por clasificación, los ratings son más ruidosos "
            "que en una liga doméstica."
        ),
    },
    "UECL": {
        "name": "UEFA Conference League",
        "source": "thesportsdb",
        # Same Swiss-model reform as Champions League, started 2024-25, but
        # only 6 league-phase matchdays here (confirmed live: round 7
        # returns empty), not 8.
        "seasons": ["2024-2025", "2025-2026", "2026-2027"],
        "notice": (
            "Solo cubre la fase de liga (jornadas 1-6), no la eliminatoria "
            "posterior. Igual que en Champions League, el grupo de clubes "
            "cambia cada temporada por clasificación, así que los ratings "
            "son más ruidosos que en una liga doméstica."
        ),
    },
    "MX2": {
        "name": "Liga de Expansión MX (México)",
        "source": "thesportsdb",
        "seasons": ["2022-2023", "2023-2024", "2024-2025", "2025-2026", "2026-2027"],
    },
    "LIB": {
        "name": "Copa Libertadores",
        "source": "thesportsdb",
        # Only the group stage (rounds 1-6) is fetchable the same way as
        # every other league here. The knockout rounds that follow use a
        # sparse, non-sequential round numbering scheme in TheSportsDB
        # (e.g. round 16 for the round of 16, then round 125 for the
        # quarterfinals in the 2026 season, with no discoverable pattern
        # connecting them and likely different per season) that doesn't
        # fit the "scan rounds 1..N, stop at the first empty one" fetch
        # this project uses everywhere else, so it's excluded the same way
        # UCL/UECL's knockout rounds already are.
        "seasons": ["2022", "2023", "2024", "2025", "2026"],
        "notice": (
            "Solo cubre la fase de grupos (jornadas 1-6), no la eliminatoria "
            "posterior — la numeración de esas rondas en la fuente de datos "
            "no sigue un patrón simple. El grupo de clubes también cambia "
            "cada temporada por clasificación, así que los ratings son más "
            "ruidosos que en una liga doméstica."
        ),
    },
}

# Season codes as used by football-data.co.uk, e.g. "2324" = 2023-24.
# First two seasons are only used to warm up Elo/Pi ratings and rolling
# form; backtesting only scores seasons from WARMUP_SEASONS onward. Shared
# by every league with source == "football-data".
SEASONS = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
WARMUP_SEASONS = 2


def matches_path(league: str) -> Path:
    return DATA_PROCESSED / f"matches_{league}.parquet"


def features_path(league: str) -> Path:
    return DATA_PROCESSED / f"features_{league}.parquet"


def model_path(league: str) -> Path:
    return MODELS_DIR / f"xgb_model_{league}.joblib"

# --- Elo rating hyperparameters ---
ELO_INITIAL = 1500.0
ELO_PROMOTED_INITIAL = 1350.0
ELO_K = 20.0
ELO_HOME_ADV = 80.0
ELO_SEASON_REGRESSION = 0.75  # fraction of prior rating carried into new season

# --- Pi rating hyperparameters (Constantinou & Fenton, 2013) ---
PI_PROMOTED_INITIAL = -0.3
PI_LAMBDA = 0.75  # error compression constant
PI_ALPHA = 0.15   # own-venue learning rate
PI_BETA = 0.10    # cross-venue learning rate
PI_SEASON_REGRESSION = 0.85  # fraction of prior rating carried into new season (mean is 0)

# --- Rolling form windows ---
FORM_WINDOWS = (5, 10)

# --- Poisson / Dixon-Coles ---
DIXON_COLES_XI = 0.002  # time-decay rate per day (~1yr half life)

# --- XGBoost ---
# n_estimators is a ceiling, not a target: training always uses early
# stopping (see xgb_model.train_xgb_with_early_stopping) against a
# held-out season, since a few thousand noisy match rows overfit fast at
# a fixed high tree count.
XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=3,
    max_depth=3,
    n_estimators=800,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="mlogloss",
)
XGB_EARLY_STOPPING_ROUNDS = 30
