"""CLI: download/refresh historical results for one league or all configured
leagues.

Usage: python scripts/refresh_data.py [--league E0]
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config  # noqa: E402
from soccer_predictor import ingest  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--league", choices=list(config.LEAGUES))
    args = parser.parse_args()
    leagues = [args.league] if args.league else list(config.LEAGUES)

    failed = []
    for league in leagues:
        print(f"=== {config.LEAGUES[league]['name']} ({league}) ===")
        try:
            matches = ingest.refresh(league)
        except Exception as exc:
            print(f"FAILED: {exc}\n")
            failed.append(league)
            continue
        print(f"Total matches: {len(matches)}")
        print(matches.tail())
        print()

    if failed:
        print(f"Leagues that failed to refresh: {', '.join(failed)}")
