"""Central configuration for the soccer-predictor pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

for _dir in (DATA_RAW, DATA_PROCESSED, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Football-Data.co.uk league code for the English Premier League.
LEAGUE_CODE = "E0"

# Season codes as used by football-data.co.uk, e.g. "2324" = 2023-24.
# First two seasons are only used to warm up Elo/Pi ratings and rolling
# form; backtesting only scores seasons from WARMUP_SEASONS onward.
SEASONS = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
WARMUP_SEASONS = 2

MATCHES_PATH = DATA_PROCESSED / "matches.parquet"
FEATURES_PATH = DATA_PROCESSED / "features.parquet"
MODEL_PATH = MODELS_DIR / "xgb_model.joblib"

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
