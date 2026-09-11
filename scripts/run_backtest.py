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
from soccer_predictor import backtest, dataset, value_betting  # noqa: E402

if __name__ == "__main__":
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=list(config.LEAGUES))
    args = parser.parse_args()
    leagues = [args.league] if args.league else list(config.LEAGUES)

    for league in leagues:
        print(f"=== {config.LEAGUES[league]['name']} ({league}) ===")

        features = dataset.load_features(league)
        results = backtest.run_backtest(features, league=league)

        if results.empty:
            # Happens when a league has config.WARMUP_SEASONS or fewer total
            # seasons (currently just MLB, with 2 seasons and a warmup of
            # 2) -- every season gets skipped as Elo/Pi/form warmup, leaving
            # nothing to backtest yet. Not an error; just nothing to report
            # until more seasons of data accumulate.
            print(f"\nSin temporadas suficientes para backtest todavía (se requieren > {config.WARMUP_SEASONS}).\n")
            continue

        print("\nPor temporada:")
        print(results.pivot(index="season", columns="model", values="mean_rps"))

        print("\nResumen (promedio ponderado por partidos, menor RPS = mejor):")
        print(backtest.summarize(results))

        value_summary, _bets = value_betting.simulate_value_betting(features)
        if value_summary["n_bets"] > 0:
            print("\nApuestas de valor (+EV) con Kelly, contra cuotas de cierre Bet365:")
            print(
                f"  {value_summary['n_bets']} apuestas | tasa de acierto "
                f"{value_summary['hit_rate']:.1%} | banca {value_summary['starting_bankroll']:.0f} -> "
                f"{value_summary['final_bankroll']:.1f} | ROI {value_summary['roi']:+.1%}"
            )
        else:
            print("\nApuestas de valor (+EV): sin cuotas de cierre disponibles para esta liga.")
        print()
