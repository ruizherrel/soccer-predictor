"""CatBoost 1X2 classifier -- comparison baseline against XGBoost.

Razali et al. 2022 found CatBoost+pi-ratings outperformed XGBoost+pi-ratings
on the Open International Soccer Database benchmark (RPS 0.1925 vs 0.2063;
see Bunker, Yeung & Fujii's chapter in Blondin et al. 2025, "Artificial
Intelligence, Optimization, and Data Sciences in Sports"). This module lets
scripts/run_backtest.py check whether that result holds on this project's
own leagues before considering a switch away from XGBoost.

Backtest-only: the deployed Streamlit app never imports this module, and
`catboost` is intentionally not in requirements.txt (it would bloat the
Streamlit Cloud deploy for a package the live app never loads). Install it
locally with `pip install catboost` to run the comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .xgb_model import FEATURE_COLUMNS, _ensure_all_classes_present, prepare_xy

# Deliberately mirrors config.XGB_PARAMS' depth/learning_rate/subsample so
# the comparison isolates the algorithm, not the hyperparameters. No
# early-stopping/eval_set here (unlike xgb_model.train_xgb_with_early_stopping)
# to keep this a simple, symmetric comparison against a fixed iteration count.
CATBOOST_PARAMS = dict(
    loss_function="MultiClass",
    depth=3,
    iterations=800,
    learning_rate=0.05,
    # CatBoost's default bootstrap (Bayesian) doesn't support `subsample` at
    # all -- Bernoulli is the one that matches XGBoost's per-tree row
    # subsampling semantics.
    bootstrap_type="Bernoulli",
    subsample=0.9,
    rsm=0.9,  # CatBoost's name for colsample_bytree
    verbose=False,
    allow_writing_files=False,
)


def train_catboost(
    df: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    params: dict | None = None,
) -> CatBoostClassifier:
    X, y = prepare_xy(df, feature_columns)
    X, y, sample_weight = _ensure_all_classes_present(X, y)
    fit_kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}

    model = CatBoostClassifier(**dict(params or CATBOOST_PARAMS))
    model.fit(X, y, **fit_kwargs)
    return model


def predict_proba(
    model: CatBoostClassifier, df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS
) -> np.ndarray:
    """Returns (n, 3) probabilities in CLASS_ORDER (away, draw, home) -- same
    guarantee as xgb_model.predict_proba, since y's classes are the same
    0/1/2 = away/draw/home integers and CatBoost's predict_proba columns
    follow sorted class order."""
    return model.predict_proba(df[feature_columns])
