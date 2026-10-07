"""FRED (Federal Reserve Economic Data) Connector using fredapi."""

import logging
import os
import time

import pandas as pd
from fredapi import Fred

from quantview.config import get_settings
from quantview.data.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

# Default FRED series for treasury rates and spreads
DEFAULT_FRED_SERIES = [
    "DGS1MO",  # 1-Month Treasury Constant Maturity Rate
    "DGS3MO",  # 3-Month Treasury Constant Maturity Rate
    "DGS6MO",  # 6-Month Treasury Constant Maturity Rate
    "DGS1",  # 1-Year Treasury Constant Maturity Rate
    "DGS2",  # 2-Year Treasury Constant Maturity Rate
    "DGS5",  # 5-Year Treasury Constant Maturity Rate
    "DGS10",  # 10-Year Treasury Constant Maturity Rate
    "DGS30",  # 30-Year Treasury Constant Maturity Rate
    "DFF",  # Federal Funds Effective Rate
    "T10Y2Y",  # 10-Year Treasury Constant Maturity Minus 2-Year
    "T10Y3M",  # 10-Year Treasury Constant Maturity Minus 3-Month
]


class FredConnector(BaseConnector):
    """FRED connector for macroeconomic time series.

    Fetches data from FRED API and normalizes to raw.fred_rates schema:
    - series_id, date, value
    """

    REQUIRED_COLUMNS = [  # noqa: RUF012 - a list, because this selects DataFrame columns
        "series_id",
        "date",
        "value",
    ]

    def __init__(self, api_key: str | None = None, cache_ttl: int = 60):
        """Initialize FRED connector.

        Args:
            api_key: FRED API key. If not provided, reads from config/environment.
            cache_ttl: Cache time-to-live in seconds.
        """
        super().__init__(cache_ttl)
        settings = get_settings()
        self.api_key = api_key or settings.fred_api_key or os.getenv("FRED_API_KEY")
        self._fred: Fred | None = None

    def _get_fred(self) -> Fred:
        """Get or create FRED client."""
        if self._fred is None:
            if not self.api_key:
                raise ValueError(
                    "FRED API key required. Set QUANTVIEW_FRED_API_KEY in config.yaml, "
                    "or pass api_key to FredConnector constructor, or set FRED_API_KEY env var."
                )
            self._fred = Fred(api_key=self.api_key)
        return self._fred

    def _fetch_series(self, series_id: str, start: str, end: str) -> pd.DataFrame:
        """Fetch a single FRED series."""
        fred = self._get_fred()

        # Fetch series data
        data = fred.get_series(
            series_id,
            observation_start=start,
            observation_end=end,
        )

        if data.empty:
            logger.warning(
                "No data returned for FRED series %s from %s to %s", series_id, start, end
            )
            return pd.DataFrame()

        # Convert to DataFrame with required schema
        df = data.reset_index()
        df.columns = ["date", "value"]
        df["series_id"] = series_id
        df = df[self.REQUIRED_COLUMNS]

        return df

    def fetch(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """Fetch FRED series data for multiple series IDs.

        Args:
            symbols: List of FRED series IDs (e.g., ['DGS10', 'DGS2', 'T10Y2Y']).
                     If empty, uses DEFAULT_FRED_SERIES.
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format.

        Returns:
            DataFrame with columns matching raw.fred_rates schema.
        """
        if not symbols:
            symbols = DEFAULT_FRED_SERIES

        cache_key = (tuple(sorted(symbols)), start, end)

        if self._is_cache_valid() and self._cached_result is not None:
            cached_symbols, cached_start, cached_end = self._cached_result.get(
                "_cache_key", (None, None, None)
            )
            if (cached_symbols, cached_start, cached_end) == cache_key:
                logger.debug("Returning cached FRED data for %s", symbols)
                return self._cached_result["data"]

        logger.info("Fetching FRED data for %d series from %s to %s", len(symbols), start, end)

        all_data = []

        for series_id in symbols:
            try:
                df = self._execute_with_retry(self._fetch_series, series_id, start, end)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.error("Failed to fetch FRED series %s: %s", series_id, e)
                # Continue with other series

        if not all_data:
            logger.warning("No data fetched for any FRED series")
            result = pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        else:
            result = pd.concat(all_data, ignore_index=True)
            result = self._validate_dataframe(result, self.REQUIRED_COLUMNS)

        # Cache the result
        self._cached_result = {"_cache_key": cache_key, "data": result.copy()}
        self._cache_timestamp = time.time()

        logger.info("Fetched %d rows for %d series", len(result), len(symbols))
        return result


# Convenience function
def fetch_fred(
    symbols: list[str], start: str, end: str, api_key: str | None = None
) -> pd.DataFrame:
    """Convenience function to fetch FRED data."""
    with FredConnector(api_key=api_key) as connector:
        return connector.fetch(symbols, start, end)


# Re-export for convenience
__all__ = ["DEFAULT_FRED_SERIES", "FredConnector", "fetch_fred"]
