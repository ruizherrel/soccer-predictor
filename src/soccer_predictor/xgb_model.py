"""XGBoost 1X2 classifier trained on Elo/Pi ratings, rolling form, and
season-boundary Poisson expected goals."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import config
from .metrics import CLASS_ORDER

FEATURE_COLUMNS = [
    "elo_home", "elo_away", "elo_diff",
    "pi_home", "pi_away", "pi_diff",
    "home_ppg_last5", "home_gf_last5", "home_ga_last5",
    "home_ppg_last10", "home_gf_last10", "home_ga_last10", "home_rest_days",
    "away_ppg_last5", "away_gf_last5", "away_ga_last5",
    "away_ppg_last10", "away_gf_last10", "away_ga_last10", "away_rest_days",
    "poisson_lambda_home", "poisson_lambda_away",
    # NaN for every team without a known home-city location (currently:
    # every non-Mexico team) — XGBoost handles missing values natively.
    "home_altitude_m", "altitude_delta_m", "away_travel_km",
]

# Bill James' Pythagorean win expectation, added on top of FEATURE_COLUMNS
# only for leagues with config.LEAGUES[league]["use_pythagorean"] set
# (currently just MLB) -- see dataset.add_pythagorean_features and
# feature_columns_for_league below. Kept out of the shared FEATURE_COLUMNS
# list so every other league's trained model stays byte-identical.
PYTHAGOREAN_FEATURE_COLUMNS = ["pyth_home_pct", "pyth_away_pct", "pyth_diff"]


def feature_columns_for_league(league: str | None) -> list[str]:
    """The actual feature set to train/predict with for a given league:
    FEATURE_COLUMNS plus any per-league additions. Callers that already have
    a trained model should prefer the feature_columns saved alongside it
    (see save_model/load_model) over calling this again, since a league's
    config could change between training and inference."""
    cols = list(FEATURE_COLUMNS)
    if league and config.LEAGUES.get(league, {}).get("use_pythagorean"):
        cols += PYTHAGOREAN_FEATURE_COLUMNS
    return cols

# CLASS_ORDER is (away, draw, home); result column is football-data.co.uk's
# native H/D/A code.
RESULT_TO_CLASS = {"A": 0, "D": 1, "H": 2}


def prepare_xy(df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[feature_columns]
    y = df["result"].map(RESULT_TO_CLASS).to_numpy()
    return X, y


def _ensure_all_classes_present(
    X: pd.DataFrame, y: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """XGBoost's sklearn XGBClassifier hard-requires every class 0..num_class-1
    to appear at least once in `y` (raises "Invalid classes inferred from
    unique values of `y`" otherwise) -- true by construction for every
    soccer league, but baseball genuinely never draws, so MLB's `y` never
    contains class 1 at all (not just rarely). Appending one synthetic row
    labeled with the missing class, with sample_weight=0, satisfies that
    check without influencing the fit at all (a zero sample weight zeroes
    out that row's gradient/hessian entirely) -- the model still learns its
    near-0 draw probability from real data elsewhere, not from this row.
    Returns sample_weight=None (rather than an all-ones array) when nothing
    was added, so callers can skip passing sample_weight entirely and keep
    training byte-identical to before this existed -- confirmed empirically
    that even a no-op all-ones sample_weight can perturb XGBoost's internal
    binning enough to change the fitted trees on knife-edge data (seen on
    Copa Libertadores, this project's sparsest league)."""
    present = set(np.unique(y))
    missing = sorted(set(RESULT_TO_CLASS.values()) - present)
    if not missing:
        return X, y, None
    filler_rows = X.iloc[[0] * len(missing)].copy()
    X = pd.concat([X, filler_rows], ignore_index=True)
    y = np.concatenate([y, missing])
    weight = np.concatenate([np.ones(len(y) - len(missing)), np.zeros(len(missing))])
    return X, y, weight


def train_xgb(
    df: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    params: dict | None = None,
    eval_df: pd.DataFrame | None = None,
    early_stopping_rounds: int = config.XGB_EARLY_STOPPING_ROUNDS,
) -> XGBClassifier:
    X, y = prepare_xy(df, feature_columns)
    X, y, sample_weight = _ensure_all_classes_present(X, y)
    fit_kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
    params = dict(params or config.XGB_PARAMS)

    eval_set = None
    if eval_df is not None and len(eval_df) > 0:
        X_val, y_val = prepare_xy(eval_df, feature_columns)
        eval_set = [(X_val, y_val)]
        params["early_stopping_rounds"] = early_stopping_rounds

    model = XGBClassifier(**params)
    if eval_set is not None:
        model.fit(X, y, eval_set=eval_set, verbose=False, **fit_kwargs)
    else:
        model.fit(X, y, **fit_kwargs)
    return model


def train_xgb_with_early_stopping(
    df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS, params: dict | None = None
) -> XGBClassifier:
    """Holds out the most recent season in `df` as an early-stopping
    validation set and trains on everything strictly before it. Falls back
    to a plain fit if `df` doesn't span at least two seasons."""
    seasons = df.sort_values("date")["season_index"].drop_duplicates().tolist()
    if len(seasons) < 2:
        return train_xgb(df, feature_columns, params)

    val_season = seasons[-1]
    fit_df = df[df["season_index"] < val_season]
    val_df = df[df["season_index"] == val_season]
    return train_xgb(fit_df, feature_columns, params, eval_df=val_df)


def predict_proba(model: XGBClassifier, df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS) -> np.ndarray:
    """Returns (n, 3) probabilities in CLASS_ORDER (away, draw, home)."""
    return model.predict_proba(df[feature_columns])


def save_model(model: XGBClassifier, league: str, feature_columns: list[str] = FEATURE_COLUMNS) -> None:
    payload = {"model": model, "feature_columns": feature_columns, "class_order": CLASS_ORDER}
    joblib.dump(payload, config.model_path(league))


def load_model(league: str) -> tuple[XGBClassifier, list[str]]:
    payload = joblib.load(config.model_path(league))
    return payload["model"], payload["feature_columns"]
