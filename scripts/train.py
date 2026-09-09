"""CLI: build the feature table (if needed) and train the XGBoost model on
all available history, for one league or all configured leagues.

Usage: python scripts/train.py [--league E0]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config  # noqa: E402
from soccer_predictor import dataset, xgb_model  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=list(config.LEAGUES))
    args = parser.parse_args()
    leagues = [args.league] if args.league else list(config.LEAGUES)

    failed = []
    for league in leagues:
        print(f"=== {config.LEAGUES[league]['name']} ({league}) ===")
        try:
            print("Building feature table...")
            features = dataset.build_and_save_features(league)
            print(f"{len(features)} rows, {features['season'].nunique()} seasons")

            print("Training XGBoost on all history...")
            model = xgb_model.train_xgb_with_early_stopping(features)
            xgb_model.save_model(model, league)
            print(f"Model saved to {config.model_path(league)}\n")
        except Exception as exc:
            print(f"FAILED: {exc}\n")
            failed.append(league)

    if failed:
        print(f"Leagues that failed to train: {', '.join(failed)}")
