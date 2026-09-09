"""CLI: download/refresh Premier League historical results.

Usage: python scripts/refresh_data.py
"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from soccer_predictor import ingest  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    matches = ingest.refresh()
    print(f"Total matches: {len(matches)}")
    print(matches.tail())
