"""Alpha Vantage Data Connector for equities and fundamentals."""

import contextlib
import logging
import os
import time
from types import TracebackType
from typing import Literal

import pandas as pd
import requests

from quantview.config import get_settings
from quantview.data.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

# Alpha Vantage API base URL
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Fundamental metrics to fetch
FUNDAMENTAL_METRICS = [
    "OVERVIEW",  # Company overview with key ratios
    "INCOME_STATEMENT",  # Annual/quarterly income statement
    "BALANCE_SHEET",  # Annual/quarterly balance sheet
    "CASH_FLOW",  # Annual/quarterly cash flow
    "EARNINGS",  # Earnings history
]


class AlphaVantageConnector(BaseConnector):
    """Alpha Vantage connector for equities and fundamentals.

    Fetches data from Alpha Vantage REST API and normalizes to:
    - raw.alphavantage_equities (OHLCV)
    - raw.fundamentals (key metrics)
    """

    EQUITY_COLUMNS = [  # noqa: RUF012  - a list, because this selects DataFrame columns and pandas reads a tuple as a single key
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    FUNDAMENTAL_COLUMNS = [  # noqa: RUF012  - a list, because this selects DataFrame columns and pandas reads a tuple as a single key
        "symbol",
        "date",
        "metric",
        "value",
        "period",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        cache_ttl: int = 60,
        timeout: int = 30,
        premium: bool = False,
    ):
        """Initialize Alpha Vantage connector.

        Args:
            api_key: Alpha Vantage API key. If not provided, reads from config/env.
            cache_ttl: Cache time-to-live in seconds.
            timeout: Request timeout in seconds.
            premium: Whether using premium tier (higher rate limits).
        """
        super().__init__(cache_ttl)
        settings = get_settings()
        self.api_key = (
            api_key
            or getattr(settings, "alphavantage_api_key", None)
            or os.getenv("ALPHAVANTAGE_API_KEY")
        )
        self.timeout = timeout
        self.premium = premium
        self._session = requests.Session()
        self._last_request_time = 0.0
        self._min_request_interval = (
            12.0 if not premium else 1.0
        )  # 5 req/min free, 75 req/min premium

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, params: dict) -> dict:
        """Make a rate-limited request to Alpha Vantage API."""
        self._rate_limit()

        params = params.copy()
        params["apikey"] = self.api_key

        response = self._session.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        # Check for API errors, most specific first. A free-tier call to a
        # premium endpoint comes back carrying both "Information" and
        # "Error Message"; checking Information first reported the subscription
        # notice and hid the reason the call actually failed.
        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
        if "Note" in data:
            raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")
        if "Information" in data:
            raise ValueError(f"Alpha Vantage info: {data['Information']}")

        result: dict = data
        return result

    def _fetch_daily_adjusted(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily adjusted time series for a symbol."""
        data = self._execute_with_retry(
            self._make_request,
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": "full",
            },
        )

        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            logger.warning("No daily data for %s", symbol)
            return pd.DataFrame(columns=self.EQUITY_COLUMNS)

        rows = []
        start_dt = pd.to_datetime(start).date()
        end_dt = pd.to_datetime(end).date()

        for date_str, values in time_series.items():
            date = pd.to_datetime(date_str).date()
            if date < start_dt or date > end_dt:
                continue

            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "volume": int(values["6. volume"]),
                }
            )

        if not rows:
            return pd.DataFrame(columns=self.EQUITY_COLUMNS)

        df = pd.DataFrame(rows, columns=self.EQUITY_COLUMNS)
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def _fetch_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Fetch fundamental data for a symbol."""
        if not self.api_key:
            logger.warning("No API key for fundamentals, skipping %s", symbol)
            return pd.DataFrame(columns=self.FUNDAMENTAL_COLUMNS)

        all_rows = []
        today = pd.Timestamp.now().date()

        # Fetch OVERVIEW for key ratios
        try:
            data = self._execute_with_retry(
                self._make_request,
                {"function": "OVERVIEW", "symbol": symbol},
            )

            # Extract key metrics from overview
            metric_mapping = {
                "MarketCapitalization": "market_cap",
                "EBITDA": "ebitda",
                "PERatio": "pe_ratio",
                "PEGRatio": "peg_ratio",
                "BookValue": "book_value_per_share",
                "DividendPerShare": "dividend_per_share",
                "DividendYield": "dividend_yield",
                "EPS": "eps",
                "RevenuePerShareTTM": "revenue_per_share_ttm",
                "ProfitMargin": "profit_margin",
                "OperatingMarginTTM": "operating_margin_ttm",
                "ReturnOnAssetsTTM": "roa_ttm",
                "ReturnOnEquityTTM": "roe_ttm",
                "Beta": "beta",
                "52WeekHigh": "week_52_high",
                "52WeekLow": "week_52_low",
                "50DayMovingAverage": "ma_50",
                "200DayMovingAverage": "ma_200",
                "SharesOutstanding": "shares_outstanding",
                "ForwardPE": "forward_pe",
                "PriceToSalesRatioTTM": "price_to_sales_ttm",
                "PriceToBookRatio": "price_to_book",
                "EVToRevenue": "ev_to_revenue",
                "EVToEBITDA": "ev_to_ebitda",
            }

            for api_key, metric_name in metric_mapping.items():
                value = data.get(api_key)
                if value and value != "None":
                    with contextlib.suppress(ValueError, TypeError):
                        all_rows.append(
                            {
                                "symbol": symbol,
                                "date": today,
                                "metric": metric_name,
                                "value": float(value),
                                "period": "ttm",
                            }
                        )

        except Exception as e:
            logger.warning("Failed to fetch overview for %s: %s", symbol, e)

        # Fetch INCOME_STATEMENT (annual)
        try:
            data = self._execute_with_retry(
                self._make_request,
                {"function": "INCOME_STATEMENT", "symbol": symbol},
            )

            for report in data.get("annualReports", []):
                fiscal_date = report.get("fiscalDateEnding")
                if not fiscal_date:
                    continue
                date = pd.to_datetime(fiscal_date).date()

                for metric in [
                    "totalRevenue",
                    "grossProfit",
                    "operatingIncome",
                    "netIncome",
                    "ebitda",
                ]:
                    value = report.get(metric)
                    if value:
                        with contextlib.suppress(ValueError, TypeError):
                            all_rows.append(
                                {
                                    "symbol": symbol,
                                    "date": date,
                                    "metric": metric.lower(),
                                    "value": float(value),
                                    "period": "annual",
                                }
                            )

        except Exception as e:
            logger.warning("Failed to fetch income statement for %s: %s", symbol, e)

        # Fetch BALANCE_SHEET (annual)
        try:
            data = self._execute_with_retry(
                self._make_request,
                {"function": "BALANCE_SHEET", "symbol": symbol},
            )

            for report in data.get("annualReports", []):
                fiscal_date = report.get("fiscalDateEnding")
                if not fiscal_date:
                    continue
                date = pd.to_datetime(fiscal_date).date()

                for metric in [
                    "totalAssets",
                    "totalLiabilities",
                    "totalShareholderEquity",
                    "cashAndCashEquivalents",
                ]:
                    value = report.get(metric)
                    if value:
                        with contextlib.suppress(ValueError, TypeError):
                            all_rows.append(
                                {
                                    "symbol": symbol,
                                    "date": date,
                                    "metric": metric.lower(),
                                    "value": float(value),
                                    "period": "annual",
                                }
                            )

        except Exception as e:
            logger.warning("Failed to fetch balance sheet for %s: %s", symbol, e)

        if not all_rows:
            return pd.DataFrame(columns=self.FUNDAMENTAL_COLUMNS)

        df = pd.DataFrame(all_rows, columns=self.FUNDAMENTAL_COLUMNS)
        return df

    def fetch(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """Fetch equity data for multiple symbols.

        For fundamentals, use fetch_fundamentals() separately.

        Args:
            symbols: List of ticker symbols (e.g., ['AAPL', 'SPY']).
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format.

        Returns:
            DataFrame with columns matching raw.alphavantage_equities schema.
        """
        cache_key = (tuple(sorted(symbols)), start, end)

        if self._is_cache_valid() and self._cached_result is not None:
            cached_symbols, cached_start, cached_end = self._cached_result.get(
                "_cache_key", (None, None, None)
            )
            if (cached_symbols, cached_start, cached_end) == cache_key:
                logger.debug("Returning cached Alpha Vantage data for %s", symbols)
                return self._cached_result["data"]

        logger.info(
            "Fetching Alpha Vantage equity data for %d symbols from %s to %s",
            len(symbols),
            start,
            end,
        )

        all_data = []

        for symbol in symbols:
            try:
                df = self._execute_with_retry(self._fetch_daily_adjusted, symbol, start, end)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.error("Failed to fetch Alpha Vantage data for %s: %s", symbol, e)
                # Continue with other symbols

        if not all_data:
            logger.warning("No equity data fetched for any symbols")
            result = pd.DataFrame(columns=self.EQUITY_COLUMNS)
        else:
            result = pd.concat(all_data, ignore_index=True)
            result = self._validate_dataframe(result, self.EQUITY_COLUMNS)

        # Cache the result
        self._cached_result = {"_cache_key": cache_key, "data": result.copy()}
        self._cache_timestamp = time.time()

        logger.info("Fetched %d equity rows for %d symbols", len(result), len(symbols))
        return result

    def fetch_fundamentals(self, symbols: list[str]) -> pd.DataFrame:
        """Fetch fundamental data for multiple symbols.

        Args:
            symbols: List of ticker symbols.

        Returns:
            DataFrame with columns matching raw.fundamentals schema.
        """
        cache_key = ("fundamentals", tuple(sorted(symbols)))

        if self._is_cache_valid() and self._cached_result is not None:
            cached_key = self._cached_result.get("_cache_key")
            if cached_key == cache_key:
                logger.debug("Returning cached Alpha Vantage fundamentals for %s", symbols)
                return self._cached_result["data"]

        logger.info("Fetching Alpha Vantage fundamentals for %d symbols", len(symbols))

        all_data = []

        for symbol in symbols:
            try:
                df = self._execute_with_retry(self._fetch_fundamentals, symbol)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.error("Failed to fetch fundamentals for %s: %s", symbol, e)

        if not all_data:
            result = pd.DataFrame(columns=self.FUNDAMENTAL_COLUMNS)
        else:
            result = pd.concat(all_data, ignore_index=True)
            result = self._validate_dataframe(result, self.FUNDAMENTAL_COLUMNS)

        # Cache the result
        self._cached_result = {"_cache_key": cache_key, "data": result.copy()}
        self._cache_timestamp = time.time()

        logger.info("Fetched %d fundamental rows for %d symbols", len(result), len(symbols))
        return result

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self._session.close()
        return super().__exit__(exc_type, exc_val, exc_tb)


# Convenience functions
def fetch_alphavantage_equities(
    symbols: list[str], start: str, end: str, api_key: str | None = None
) -> pd.DataFrame:
    """Convenience function to fetch Alpha Vantage equity data."""
    with AlphaVantageConnector(api_key=api_key) as connector:
        return connector.fetch(symbols, start, end)


def fetch_alphavantage_fundamentals(symbols: list[str], api_key: str | None = None) -> pd.DataFrame:
    """Convenience function to fetch Alpha Vantage fundamental data."""
    with AlphaVantageConnector(api_key=api_key) as connector:
        return connector.fetch_fundamentals(symbols)


__all__ = [
    "FUNDAMENTAL_METRICS",
    "AlphaVantageConnector",
    "fetch_alphavantage_equities",
    "fetch_alphavantage_fundamentals",
]
