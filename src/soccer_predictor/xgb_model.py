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
]

# CLASS_ORDER is (away, draw, home); result column is football-data.co.uk's
# native H/D/A code.
RESULT_TO_CLASS = {"A": 0, "D": 1, "H": 2}


def prepare_xy(df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[feature_columns]
    y = df["result"].map(RESULT_TO_CLASS).to_numpy()
    return X, y


def train_xgb(
    df: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    params: dict | None = None,
    eval_df: pd.DataFrame | None = None,
    early_stopping_rounds: int = config.XGB_EARLY_STOPPING_ROUNDS,
) -> XGBClassifier:
    X, y = prepare_xy(df, feature_columns)
    params = dict(params or config.XGB_PARAMS)

    eval_set = None
    if eval_df is not None and len(eval_df) > 0:
        X_val, y_val = prepare_xy(eval_df, feature_columns)
        eval_set = [(X_val, y_val)]
        params["early_stopping_rounds"] = early_stopping_rounds

    model = XGBClassifier(**params)
    if eval_set is not None:
        model.fit(X, y, eval_set=eval_set, verbose=False)
    else:
        model.fit(X, y)
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


def save_model(model: XGBClassifier, league: str) -> None:
    payload = {"model": model, "feature_columns": FEATURE_COLUMNS, "class_order": CLASS_ORDER}
    joblib.dump(payload, config.model_path(league))


def load_model(league: str) -> tuple[XGBClassifier, list[str]]:
    payload = joblib.load(config.model_path(league))
    return payload["model"], payload["feature_columns"]
