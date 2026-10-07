"""Yahoo Finance Data Connector using yfinance."""

import logging
import time

import pandas as pd
import yfinance as yf

from quantview.data.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class YahooConnector(BaseConnector):
    """Yahoo Finance connector for equities OHLCV + dividends/splits.

    Fetches data from Yahoo Finance and normalizes to raw.yahoo_equities schema:
    - symbol, date, open, high, low, close, volume, adj_close, dividends, splits
    """

    REQUIRED_COLUMNS = [  # noqa: RUF012  - a list, because this selects DataFrame columns and pandas reads a tuple as a single key
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
        "dividends",
        "splits",
    ]

    def __init__(self, cache_ttl: int = 60):
        super().__init__(cache_ttl)
        self._ticker_cache: dict[str, yf.Ticker] = {}

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create yfinance Ticker object with caching."""
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)
        return self._ticker_cache[symbol]

    def _fetch_single(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch data for a single symbol."""
        ticker = self._get_ticker(symbol)

        # Fetch historical data with dividends and splits
        hist = ticker.history(
            start=start,
            end=end,
            auto_adjust=False,  # Keep raw OHLCV, get adj_close separately
            actions=True,  # Include dividends and splits
        )

        if hist.empty:
            logger.warning("No data returned for %s from %s to %s", symbol, start, end)
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        # Reset index to get date as column
        hist = hist.reset_index()

        # Rename columns to match schema
        hist = hist.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Adj Close": "adj_close",
                "Dividends": "dividends",
                "Stock Splits": "splits",
            }
        )

        # Add symbol column
        hist["symbol"] = symbol

        # Ensure all required columns exist
        for col in ["dividends", "splits"]:
            if col not in hist.columns:
                hist[col] = 0.0 if col == "dividends" else 1.0

        # Select and order columns
        hist = hist[self.REQUIRED_COLUMNS]

        return hist

    def fetch(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """Fetch OHLCV + dividends/splits for multiple symbols.

        Args:
            symbols: List of ticker symbols (e.g., ['AAPL', 'SPY']).
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format.

        Returns:
            DataFrame with columns matching raw.yahoo_equities schema.
        """
        cache_key = (tuple(sorted(symbols)), start, end)

        if self._is_cache_valid() and self._cached_result is not None:
            cached_symbols, cached_start, cached_end = self._cached_result.get(
                "_cache_key", (None, None, None)
            )
            if (cached_symbols, cached_start, cached_end) == cache_key:
                logger.debug("Returning cached Yahoo data for %s", symbols)
                return self._cached_result["data"]

        logger.info(
            "Fetching Yahoo Finance data for %d symbols from %s to %s", len(symbols), start, end
        )

        all_data = []

        for symbol in symbols:
            try:
                df = self._execute_with_retry(self._fetch_single, symbol, start, end)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.error("Failed to fetch Yahoo data for %s: %s", symbol, e)
                # Continue with other symbols

        if not all_data:
            logger.warning("No data fetched for any symbols")
            result = pd.DataFrame(columns=self.REQUIRED_COLUMNS)
        else:
            result = pd.concat(all_data, ignore_index=True)
            result = self._validate_dataframe(result, self.REQUIRED_COLUMNS)

        # Cache the result
        self._cached_result = {"_cache_key": cache_key, "data": result.copy()}
        self._cache_timestamp = time.time()

        logger.info("Fetched %d rows for %d symbols", len(result), len(symbols))
        return result


# Convenience function for direct use
def fetch_yahoo(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Convenience function to fetch Yahoo data without managing connector instance."""
    with YahooConnector() as connector:
        return connector.fetch(symbols, start, end)
