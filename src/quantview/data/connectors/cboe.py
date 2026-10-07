"""CBOE Data Connector for option chains with Greeks."""

import logging
import time
from types import TracebackType
from typing import Literal

import pandas as pd
import requests

from quantview.data.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

# CBOE API endpoints
CBOE_OPTIONS_CHAIN_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
CBOE_UNDERLYING_URL = "https://cdn.cboe.com/api/global/delayed_quotes/quote/{symbol}.json"


class CboeConnector(BaseConnector):
    """CBOE connector for option chains with Greeks.

    Fetches data from CBOE JSON API and normalizes to raw.cboe_options schema:
    - underlying, expiry, strike, option_right, bid, ask, mid, iv, delta, gamma,
      vega, theta, rho, volume, open_interest
    """

    REQUIRED_COLUMNS = [  # noqa: RUF012  - a list, because this selects DataFrame columns and pandas reads a tuple as a single key
        "underlying",
        "expiry",
        "strike",
        "option_right",
        "bid",
        "ask",
        "mid",
        "iv",
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
        "volume",
        "open_interest",
    ]

    def __init__(self, cache_ttl: int = 60, timeout: int = 30):
        """Initialize CBOE connector.

        Args:
            cache_ttl: Cache time-to-live in seconds.
            timeout: Request timeout in seconds.
        """
        super().__init__(cache_ttl)
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            }
        )

    def _fetch_underlying_price(self, symbol: str) -> float | None:
        """Fetch current underlying price for moneyness calculation."""
        try:
            url = CBOE_UNDERLYING_URL.format(symbol=symbol.upper())
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return float(data.get("data", {}).get("current_price", 0))
        except Exception as e:
            logger.debug("Could not fetch underlying price for %s: %s", symbol, e)
            return None

    def _fetch_options_chain(self, symbol: str) -> dict | None:
        """Fetch raw options chain data from CBOE."""
        url = CBOE_OPTIONS_CHAIN_URL.format(symbol=symbol.upper())
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()
        data: dict = response.json()
        return data

    def _parse_options_data(self, symbol: str, data: dict) -> pd.DataFrame:
        """Parse CBOE options chain JSON into normalized DataFrame."""
        rows = []

        # Extract options data from CBOE response
        options_data = data.get("data", {}).get("options", {})

        for expiry_str, strikes_data in options_data.items():
            try:
                expiry = pd.to_datetime(expiry_str).date()
            except Exception:
                logger.warning("Invalid expiry format: %s", expiry_str)
                continue

            for strike_str, option_data in strikes_data.items():
                try:
                    strike = float(strike_str)
                except Exception:
                    logger.warning("Invalid strike format: %s", strike_str)
                    continue

                for right in ["call", "put"]:
                    right_key = "call" if right == "call" else "put"
                    option = option_data.get(right_key)

                    if not option:
                        continue

                    # Extract Greeks and pricing
                    greeks = option.get("greeks", {}) or {}
                    quote = option.get("quote", {}) or {}

                    bid = quote.get("bid")
                    ask = quote.get("ask")
                    mid = (bid + ask) / 2 if bid is not None and ask is not None else None

                    row = {
                        "underlying": symbol.upper(),
                        "expiry": expiry,
                        "strike": strike,
                        "option_right": "C" if right == "call" else "P",
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "iv": greeks.get("iv"),
                        "delta": greeks.get("delta"),
                        "gamma": greeks.get("gamma"),
                        "vega": greeks.get("vega"),
                        "theta": greeks.get("theta"),
                        "rho": greeks.get("rho"),
                        "volume": quote.get("volume"),
                        "open_interest": quote.get("open_interest"),
                    }
                    rows.append(row)

        if not rows:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        df = pd.DataFrame(rows, columns=self.REQUIRED_COLUMNS)
        return df

    def _fetch_single(self, symbol: str) -> pd.DataFrame:
        """Fetch options chain for a single underlying symbol."""
        raw_data = self._execute_with_retry(self._fetch_options_chain, symbol)
        if not raw_data:
            logger.warning("No options data returned for %s", symbol)
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        df = self._parse_options_data(symbol, raw_data)
        return df

    def fetch(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """Fetch option chains for multiple underlying symbols.

        Note: CBOE API returns current snapshot only. start/end are ignored
        but kept for interface compatibility.

        Args:
            symbols: List of underlying symbols (e.g., ['SPY', 'AAPL']).
            start: Start date (ignored, kept for interface compatibility).
            end: End date (ignored, kept for interface compatibility).

        Returns:
            DataFrame with columns matching raw.cboe_options schema.
        """
        _ = start, end  # Acknowledge unused parameters

        cache_key = tuple(sorted(symbols))

        if self._is_cache_valid() and self._cached_result is not None:
            cached_symbols = self._cached_result.get("_cache_key")
            if cached_symbols == cache_key:
                logger.debug("Returning cached CBOE data for %s", symbols)
                return self._cached_result["data"]

        logger.info("Fetching CBOE options chains for %d symbols", len(symbols))

        all_data = []

        for symbol in symbols:
            try:
                df = self._execute_with_retry(self._fetch_single, symbol)
                if not df.empty:
                    # Add snapshot timestamp
                    df["snapshot_ts"] = pd.Timestamp.now()
                    all_data.append(df)
            except Exception as e:
                logger.error("Failed to fetch CBOE data for %s: %s", symbol, e)
                # Continue with other symbols

        if not all_data:
            logger.warning("No options data fetched for any symbols")
            result = pd.DataFrame(columns=[*self.REQUIRED_COLUMNS, "snapshot_ts"])
        else:
            result = pd.concat(all_data, ignore_index=True)
            result = self._validate_dataframe(result, self.REQUIRED_COLUMNS)

        # Cache the result
        self._cached_result = {"_cache_key": cache_key, "data": result.copy()}
        self._cache_timestamp = time.time()

        logger.info("Fetched %d option contracts for %d symbols", len(result), len(symbols))
        return result

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self._session.close()
        return super().__exit__(exc_type, exc_val, exc_tb)


# Convenience function
def fetch_cboe(symbols: list[str]) -> pd.DataFrame:
    """Convenience function to fetch CBOE options data."""
    with CboeConnector() as connector:
        return connector.fetch(symbols, "", "")


__all__ = ["CboeConnector", "fetch_cboe"]
