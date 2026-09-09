"""Translates model probabilities into betting value: +EV detection and
Kelly Criterion stake sizing, backtested against historical closing odds.

This deliberately does NOT call a live odds feed. The app lets you predict
any hypothetical matchup (not necessarily a real scheduled fixture), so
there's no natural "current odds" to fetch even with a live feed, and every
live-odds provider found needs its own account/API key (see project docs).
Bet365 closing odds are already parsed by ingest.py for every league except
Mexico (source has no odds), so this simulates the same +EV/Kelly
methodology retrospectively, using the same leave-one-season-out XGBoost
folds as backtest.run_backtest, as a free stand-in that needs no signup.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from . import xgb_model
from .xgb_model import RESULT_TO_CLASS

CLASS_ORDER = ("away", "draw", "home")
ODDS_COLUMNS = {"away": "b365_away", "draw": "b365_draw", "home": "b365_home"}


def kelly_fraction(model_prob: float, decimal_odds: float, cap: float = 0.05) -> float:
    """Full Kelly stake as a fraction of bankroll, capped at `cap`.

    A 1-5% cap is standard practitioner guidance for surviving model error
    and variance that full Kelly (which assumes the model's probability is
    exactly correct) doesn't account for.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    edge = model_prob * b - (1 - model_prob)
    if edge <= 0:
        return 0.0
    return float(min(edge / b, cap))


def simulate_value_betting(
    features: pd.DataFrame,
    warmup_seasons: int = config.WARMUP_SEASONS,
    kelly_cap: float = 0.05,
    min_edge: float = 0.0,
    starting_bankroll: float = 100.0,
) -> tuple[dict, pd.DataFrame]:
    """Flat-Kelly-staked simulation against Bet365 closing odds, retrained
    per season fold exactly like backtest.run_backtest (each season scored
    using a model trained only on strictly-prior seasons). Returns a
    summary dict and a per-bet DataFrame; both are empty/NaN-summary if the
    league has no odds columns (e.g. Mexico)."""
    if not all(c in features.columns for c in ODDS_COLUMNS.values()):
        return {
            "starting_bankroll": starting_bankroll,
            "final_bankroll": starting_bankroll,
            "n_bets": 0,
            "hit_rate": float("nan"),
            "roi": float("nan"),
        }, pd.DataFrame()

    seasons_order = features.sort_values("date")["season"].drop_duplicates().tolist()
    bankroll = starting_bankroll
    bets = []

    for i, season in enumerate(seasons_order):
        if i < warmup_seasons:
            continue

        train_df = features[features["season_index"] < i]
        test_df = features[features["season_index"] == i].reset_index(drop=True)

        model = xgb_model.train_xgb_with_early_stopping(train_df)
        probs = xgb_model.predict_proba(model, test_df)  # (away, draw, home)
        outcome_idx = test_df["result"].map(RESULT_TO_CLASS).to_numpy()

        for row_i in range(len(test_df)):
            for cls_i, label in enumerate(CLASS_ORDER):
                odds = test_df.at[row_i, ODDS_COLUMNS[label]]
                if pd.isna(odds) or odds <= 1.0:
                    continue

                p = float(probs[row_i, cls_i])
                edge = p * odds - 1.0
                if edge <= min_edge:
                    continue

                stake_frac = kelly_fraction(p, odds, cap=kelly_cap)
                if stake_frac <= 0:
                    continue

                stake = bankroll * stake_frac
                won = bool(outcome_idx[row_i] == cls_i)
                bankroll += (stake * odds - stake) if won else -stake

                bets.append(
                    {
                        "season": season,
                        "outcome": label,
                        "model_prob": p,
                        "decimal_odds": float(odds),
                        "edge": edge,
                        "stake": stake,
                        "won": won,
                        "bankroll_after": bankroll,
                    }
                )

    bets_df = pd.DataFrame(bets)
    summary = {
        "starting_bankroll": starting_bankroll,
        "final_bankroll": bankroll,
        "n_bets": len(bets_df),
        "hit_rate": float(bets_df["won"].mean()) if len(bets_df) else float("nan"),
        "roi": (bankroll - starting_bankroll) / starting_bankroll,
    }
    return summary, bets_df
