#!/usr/bin/env python3
"""Daily EOD Ingestion Script.

Run daily at 18:00 ET to fetch end-of-day data for all universe symbols
via all connectors (Yahoo, FRED, CBOE, Alpha Vantage).

Usage:
    python scripts/ingest_daily.py [--date YYYY-MM-DD]
    python -m quantview.data.ingest daily [--date YYYY-MM-DD]
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantview.config import get_settings
from quantview.data.ingest import run_daily_ingest


def main() -> int:
    """Main entry point for daily ingestion script."""
    import argparse

    parser = argparse.ArgumentParser(description="QuantView Daily EOD Ingestion")
    parser.add_argument(
        "--date", type=str, help="Target date (YYYY-MM-DD), defaults to yesterday ET"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Configure logging
    settings = get_settings()
    log_level = (
        logging.DEBUG
        if args.verbose
        else getattr(logging, settings.log_level.upper(), logging.INFO)
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(settings.log_file),
        ],
    )

    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting daily EOD ingestion script")
        stats = run_daily_ingest(date=args.date)
        logger.info(f"Daily ingestion completed successfully: {stats}")
        print(f"SUCCESS: {stats}")
        return 0
    except Exception as e:
        logger.error(f"Daily ingestion failed: {e}", exc_info=True)
        print(f"FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
