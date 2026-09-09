"""CLI: Leave-One-Season-Out backtest, comparing the Poisson baseline,
XGBoost ensemble, and (when available) closing market odds, for one league
or all configured leagues.

Usage: python scripts/run_backtest.py [--league E0]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from soccer_predictor import backtest, dataset  # noqa: E402

if __name__ == "__main__":
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=list(config.LEAGUES))
    args = parser.parse_args()
    leagues = [args.league] if args.league else list(config.LEAGUES)

    for league in leagues:
        print(f"=== {config.LEAGUES[league]['name']} ({league}) ===")

        features = dataset.load_features(league)
        results = backtest.run_backtest(features)

        print("\nPor temporada:")
        print(results.pivot(index="season", columns="model", values="mean_rps"))

        print("\nResumen (promedio ponderado por partidos, menor RPS = mejor):")
        print(backtest.summarize(results))
        print()
