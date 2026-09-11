"""Leave-One-Season-Out backtest: Poisson baseline vs XGBoost ensemble vs
(when available) the closing bookmaker odds already present in the
football-data.co.uk CSVs.

For each season S (skipping the first config.WARMUP_SEASONS, which exist
only to warm up Elo/Pi ratings and rolling form), the XGBoost model is
retrained from scratch on every match strictly before S and evaluated on S.
The Poisson probabilities are not refit here: dataset.build_poisson_features
already computed them with the same "fit on strictly-prior seasons" rule, so
the poisson_p_* columns in `features` are already valid out-of-fold values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from . import xgb_model
from .metrics import mean_rps, multiclass_log_loss
from .xgb_model import RESULT_TO_CLASS

try:
    from . import catboost_model

    _HAS_CATBOOST = True
except ImportError:
    # catboost is an optional, backtest-only dependency (see
    # catboost_model.py's module docstring) -- not installed by default.
    _HAS_CATBOOST = False


def odds_implied_probs(df: pd.DataFrame) -> np.ndarray | None:
    """Overround-normalized implied probabilities from Bet365 closing odds,
    in (away, draw, home) order. Returns None if the odds columns are
    missing (older seasons sometimes lack them for lower-profile leagues)."""
    cols = ["b365_away", "b365_draw", "b365_home"]
    if not all(c in df.columns for c in cols) or df[cols].isna().any().any():
        return None
    inv = 1.0 / df[cols].to_numpy(dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)


def _score(name: str, season: str, probs: np.ndarray, outcome_idx: np.ndarray) -> dict:
    pred_idx = probs.argmax(axis=1)
    return {
        "season": season,
        "model": name,
        "n_matches": len(outcome_idx),
        "mean_rps": mean_rps(probs, outcome_idx),
        "log_loss": multiclass_log_loss(probs, outcome_idx),
        "accuracy": float(np.mean(pred_idx == outcome_idx)),
    }


def run_backtest(
    features: pd.DataFrame,
    warmup_seasons: int = config.WARMUP_SEASONS,
    league: str | None = None,
    include_catboost: bool = True,
) -> pd.DataFrame:
    """`league` resolves the right feature set via
    xgb_model.feature_columns_for_league (e.g. MLB's extra Pythagorean
    columns) for both training and prediction -- passing the wrong league
    (or None for a league that needs it) would silently train/score on the
    wrong columns. `include_catboost` adds a CatBoost comparison row per
    season when the optional `catboost` package is installed (see
    catboost_model.py); set False to skip it even when installed."""
    feature_columns = xgb_model.feature_columns_for_league(league)
    seasons_order = (
        features.sort_values("date")["season"].drop_duplicates().tolist()
    )

    rows = []
    for i, season in enumerate(seasons_order):
        if i < warmup_seasons:
            continue

        train_df = features[features["season_index"] < i]
        test_df = features[features["season_index"] == i]
        outcome_idx = test_df["result"].map(RESULT_TO_CLASS).to_numpy()

        poisson_probs = test_df[["poisson_p_away", "poisson_p_draw", "poisson_p_home"]].to_numpy()
        rows.append(_score("poisson", season, poisson_probs, outcome_idx))

        model = xgb_model.train_xgb_with_early_stopping(train_df, feature_columns)
        xgb_probs = xgb_model.predict_proba(model, test_df, feature_columns)
        rows.append(_score("xgboost", season, xgb_probs, outcome_idx))

        if include_catboost and _HAS_CATBOOST:
            cb_model = catboost_model.train_catboost(train_df, feature_columns)
            cb_probs = catboost_model.predict_proba(cb_model, test_df, feature_columns)
            rows.append(_score("catboost", season, cb_probs, outcome_idx))

        market_probs = odds_implied_probs(test_df)
        if market_probs is not None:
            rows.append(_score("market_odds", season, market_probs, outcome_idx))

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Match-count-weighted average of each metric per model, across seasons."""
    def _weighted(g: pd.DataFrame, col: str) -> float:
        return float(np.average(g[col], weights=g["n_matches"]))

    summary = results.groupby("model").apply(
        lambda g: pd.Series(
            {
                "mean_rps": _weighted(g, "mean_rps"),
                "log_loss": _weighted(g, "log_loss"),
                "accuracy": _weighted(g, "accuracy"),
                "n_matches": g["n_matches"].sum(),
            }
        ),
        include_groups=False,
    )
    return summary.sort_values("mean_rps")
