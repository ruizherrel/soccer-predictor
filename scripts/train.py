"""CLI: build the feature table (if needed) and train the XGBoost model on
all available history.

Usage: python scripts/train.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config  # noqa: E402
from soccer_predictor import dataset, xgb_model  # noqa: E402

if __name__ == "__main__":
    print("Building feature table...")
    features = dataset.build_and_save_features()
    print(f"{len(features)} rows, {features['season'].nunique()} seasons")

    print("Training XGBoost on all history...")
    model = xgb_model.train_xgb_with_early_stopping(features)
    xgb_model.save_model(model)
    print(f"Model saved to {config.MODEL_PATH}")
