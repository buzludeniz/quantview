#!/usr/bin/env python3
"""Intraday 5-min Snapshot Ingestion Script.

Run every 5 minutes during market hours (09:30-16:00 ET, Mon-Fri)
to fetch 5-minute bars for all active equity symbols.

Usage:
    python scripts/ingest_intraday.py
    python -m quantview.data.ingest intraday
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantview.config import get_settings
from quantview.data.ingest import run_intraday_ingest


def main() -> int:
    """Main entry point for intraday ingestion script."""
    import argparse

    parser = argparse.ArgumentParser(description="QuantView Intraday 5-min Snapshot Ingestion")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Configure logging
    settings = get_settings()
    settings.ensure_directories()
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
        logger.info("Starting intraday 5-min snapshot ingestion script")
        stats = run_intraday_ingest()
        logger.info(f"Intraday ingestion completed successfully: {stats}")
        print(f"SUCCESS: {stats}")
        return 0
    except Exception as e:
        logger.error(f"Intraday ingestion failed: {e}", exc_info=True)
        print(f"FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
