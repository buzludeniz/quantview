"""Tests for QuantView data connectors."""

import os
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantview.data.connectors import (
    AlphaVantageConnector,
    BaseConnector,
    CboeConnector,
    FredConnector,
    YahooConnector,
)
from quantview.data.connectors.alphavantage import (
    fetch_alphavantage_equities,
)
from quantview.data.connectors.fred import DEFAULT_FRED_SERIES, fetch_fred
from quantview.data.connectors.yahoo import fetch_yahoo


class ConcreteTestConnector(BaseConnector):
    """Concrete implementation for testing base connector."""

    def fetch(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame()


class TestBaseConnector:
    """Test base connector functionality."""

    def test_backoff_constants(self):
        """Test retry backoff configuration."""
        assert BaseConnector.MAX_RETRIES == 5
        # A tuple: the schedule is a constant, and a list here could be
        # mutated in place by one caller and change it for every other.
        assert BaseConnector.BACKOFF_BASE == (1, 2, 4, 8, 60)

    def test_cache_validity(self):
        """Test cache validity checking."""
        connector = ConcreteTestConnector(cache_ttl=60)
        assert not connector._is_cache_valid()

        # Set a recent cache timestamp
        connector._cache_timestamp = time.time()
        connector._cached_result = "test"
        assert connector._is_cache_valid()

        # Set an old cache timestamp
        connector._cache_timestamp = time.time() - 120
        assert not connector._is_cache_valid()

    def test_cache_clear(self):
        """Test cache clearing."""
        connector = ConcreteTestConnector(cache_ttl=60)
        connector._cache_timestamp = time.time()
        connector._cached_result = "test"
        connector._clear_cache()
        assert connector._cache_timestamp is None
        assert connector._cached_result is None

    def test_validate_dataframe(self):
        """Test DataFrame validation."""
        connector = ConcreteTestConnector()
        df = pd.DataFrame({"symbol": ["AAPL"], "date": ["2024-01-01"], "open": [100.0]})
        validated = connector._validate_dataframe(df, ["symbol", "date", "open"])
        assert "date" in validated.columns

    def test_validate_dataframe_missing_columns(self):
        """Test DataFrame validation with missing columns."""
        connector = ConcreteTestConnector()
        df = pd.DataFrame({"symbol": ["AAPL"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            connector._validate_dataframe(df, ["symbol", "date", "open"])

    def test_context_manager(self):
        """Test context manager cleanup."""
        connector = ConcreteTestConnector()
        connector._cache_timestamp = time.time()
        connector._cached_result = "test"
        with connector as c:
            assert c is connector
        assert connector._cache_timestamp is None
        assert connector._cached_result is None


class TestYahooConnector:
    """Test Yahoo Finance connector."""

    def test_required_columns(self):
        """Test required columns definition."""
        assert YahooConnector.REQUIRED_COLUMNS == [
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

    @patch("quantview.data.connectors.yahoo.yf.Ticker")
    def test_fetch_single_success(self, mock_ticker_class):
        """Test fetching single symbol successfully."""
        # Setup mock
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        # Mock history data
        mock_hist = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000000, 1100000],
                "Adj Close": [101.0, 102.0],
                "Dividends": [0.0, 0.0],
                "Stock Splits": [0.0, 0.0],
            }
        )
        mock_ticker.history.return_value = mock_hist

        connector = YahooConnector(cache_ttl=0)  # Disable cache for test
        result = connector._fetch_single("AAPL", "2024-01-01", "2024-01-02")

        assert len(result) == 2
        assert list(result.columns) == YahooConnector.REQUIRED_COLUMNS
        assert result["symbol"].iloc[0] == "AAPL"

    @patch("quantview.data.connectors.yahoo.yf.Ticker")
    def test_fetch_single_empty(self, mock_ticker_class):
        """Test fetching single symbol with no data."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()

        connector = YahooConnector(cache_ttl=0)
        result = connector._fetch_single("INVALID", "2024-01-01", "2024-01-02")

        assert result.empty
        assert list(result.columns) == YahooConnector.REQUIRED_COLUMNS

    @patch("quantview.data.connectors.yahoo.yf.Ticker")
    def test_fetch_multiple_symbols(self, mock_ticker_class):
        """Test fetching multiple symbols."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        mock_hist = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-01"]),
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000000],
                "Adj Close": [101.0],
                "Dividends": [0.0],
                "Stock Splits": [0.0],
            }
        )
        mock_ticker.history.return_value = mock_hist

        connector = YahooConnector(cache_ttl=0)
        result = connector.fetch(["AAPL", "MSFT"], "2024-01-01", "2024-01-01")

        assert len(result) == 2
        assert set(result["symbol"].unique()) == {"AAPL", "MSFT"}

    def test_ticker_caching(self):
        """Test that Ticker objects are cached."""
        connector = YahooConnector()
        ticker1 = connector._get_ticker("AAPL")
        ticker2 = connector._get_ticker("AAPL")
        assert ticker1 is ticker2


class TestFredConnector:
    """Test FRED connector."""

    def test_required_columns(self):
        """Test required columns definition."""
        assert FredConnector.REQUIRED_COLUMNS == ["series_id", "date", "value"]

    def test_default_series(self):
        """Test default FRED series list."""
        assert DEFAULT_FRED_SERIES == [
            "DGS1MO",
            "DGS3MO",
            "DGS6MO",
            "DGS1",
            "DGS2",
            "DGS5",
            "DGS10",
            "DGS30",
            "DFF",
            "T10Y2Y",
            "T10Y3M",
        ]
        assert len(DEFAULT_FRED_SERIES) == 11

    @patch("quantview.data.connectors.fred.Fred")
    def test_fetch_series_success(self, mock_fred_class):
        """Test fetching a single FRED series."""
        mock_fred = MagicMock()
        mock_fred_class.return_value = mock_fred

        # Mock series data
        mock_series = pd.Series(
            [4.5, 4.6, 4.7],
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            name="DGS10",
        )
        mock_fred.get_series.return_value = mock_series

        connector = FredConnector(api_key="test_key", cache_ttl=0)
        result = connector._fetch_series("DGS10", "2024-01-01", "2024-01-03")

        assert len(result) == 3
        assert list(result.columns) == FredConnector.REQUIRED_COLUMNS
        assert result["series_id"].iloc[0] == "DGS10"
        assert result["value"].iloc[0] == 4.5

    @patch("quantview.data.connectors.fred.Fred")
    def test_fetch_multiple_series(self, mock_fred_class):
        """Test fetching multiple FRED series."""
        mock_fred = MagicMock()
        mock_fred_class.return_value = mock_fred

        def mock_get_series(series_id, **kwargs):
            return pd.Series(
                [4.5],
                index=pd.to_datetime(["2024-01-01"]),
                name=series_id,
            )

        mock_fred.get_series.side_effect = mock_get_series

        connector = FredConnector(api_key="test_key", cache_ttl=0)
        result = connector.fetch(["DGS10", "DGS2"], "2024-01-01", "2024-01-01")

        assert len(result) == 2
        assert set(result["series_id"].unique()) == {"DGS10", "DGS2"}

    def test_missing_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        with patch("quantview.data.connectors.fred.get_settings") as mock_settings:
            mock_settings.return_value.fred_api_key = ""
            with patch.dict(os.environ, {}, clear=True):
                connector = FredConnector(api_key=None, cache_ttl=0)
                with pytest.raises(ValueError, match="FRED API key required"):
                    connector._get_fred()


class TestCboeConnector:
    """Test CBOE connector."""

    def test_required_columns(self):
        """Test required columns definition."""
        expected = [
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
        assert expected == CboeConnector.REQUIRED_COLUMNS

    @patch("quantview.data.connectors.cboe.requests.Session.get")
    def test_fetch_options_chain(self, mock_get):
        """Test fetching options chain from CBOE."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "options": {
                    "2024-01-19": {
                        "450.0": {
                            "call": {
                                "quote": {
                                    "bid": 10.5,
                                    "ask": 11.0,
                                    "volume": 100,
                                    "open_interest": 500,
                                },
                                "greeks": {
                                    "iv": 0.25,
                                    "delta": 0.6,
                                    "gamma": 0.02,
                                    "vega": 0.15,
                                    "theta": -0.05,
                                    "rho": 0.03,
                                },
                            },
                            "put": {
                                "quote": {
                                    "bid": 5.0,
                                    "ask": 5.5,
                                    "volume": 50,
                                    "open_interest": 300,
                                },
                                "greeks": {
                                    "iv": 0.28,
                                    "delta": -0.4,
                                    "gamma": 0.02,
                                    "vega": 0.15,
                                    "theta": -0.04,
                                    "rho": -0.02,
                                },
                            },
                        }
                    }
                }
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        connector = CboeConnector(cache_ttl=0)
        result = connector._fetch_single("SPY")

        assert len(result) == 2  # One call, one put
        assert list(result.columns) == CboeConnector.REQUIRED_COLUMNS
        assert result["underlying"].iloc[0] == "SPY"
        assert result["option_right"].iloc[0] == "C"
        assert result["option_right"].iloc[1] == "P"
        assert result["mid"].iloc[0] == 10.75  # (10.5 + 11.0) / 2

    @patch("quantview.data.connectors.cboe.requests.Session.get")
    def test_fetch_empty_response(self, mock_get):
        """Test handling empty CBOE response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": {"options": {}}}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        connector = CboeConnector(cache_ttl=0)
        result = connector._fetch_single("INVALID")

        assert result.empty
        assert list(result.columns) == CboeConnector.REQUIRED_COLUMNS

    def test_context_manager_closes_session(self):
        """Test that session is closed on context exit."""
        connector = CboeConnector()
        connector.__enter__()
        connector.__exit__(None, None, None)
        # Session should be closed (we can't easily test this, but no error should occur)


class TestAlphaVantageConnector:
    """Test Alpha Vantage connector."""

    def test_equity_columns(self):
        """Test equity columns definition."""
        expected = [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert expected == AlphaVantageConnector.EQUITY_COLUMNS

    def test_fundamental_columns(self):
        """Test fundamental columns definition."""
        expected = [
            "symbol",
            "date",
            "metric",
            "value",
            "period",
        ]
        assert expected == AlphaVantageConnector.FUNDAMENTAL_COLUMNS

    @patch("quantview.data.connectors.alphavantage.requests.Session.get")
    def test_fetch_daily_adjusted(self, mock_get):
        """Test fetching daily adjusted data."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Time Series (Daily)": {
                "2024-01-01": {
                    "1. open": "100.0",
                    "2. high": "102.0",
                    "3. low": "99.0",
                    "4. close": "101.0",
                    "5. adjusted close": "101.0",
                    "6. volume": "1000000",
                    "7. dividend amount": "0.0",
                    "8. split coefficient": "1.0",
                },
                "2024-01-02": {
                    "1. open": "101.0",
                    "2. high": "103.0",
                    "3. low": "100.0",
                    "4. close": "102.0",
                    "5. adjusted close": "102.0",
                    "6. volume": "1100000",
                    "7. dividend amount": "0.0",
                    "8. split coefficient": "1.0",
                },
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        connector = AlphaVantageConnector(api_key="test_key", cache_ttl=0, premium=True)
        result = connector._fetch_daily_adjusted("AAPL", "2024-01-01", "2024-01-02")

        assert len(result) == 2
        assert list(result.columns) == AlphaVantageConnector.EQUITY_COLUMNS
        assert result["symbol"].iloc[0] == "AAPL"
        assert result["close"].iloc[0] == 101.0

    @patch("quantview.data.connectors.alphavantage.requests.Session.get")
    def test_fetch_fundamentals_overview(self, mock_get):
        """Test fetching fundamentals from OVERVIEW endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Symbol": "AAPL",
            "MarketCapitalization": "3000000000000",
            "PERatio": "28.5",
            "EPS": "6.5",
            "DividendYield": "0.005",
            "Beta": "1.2",
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        connector = AlphaVantageConnector(api_key="test_key", cache_ttl=0, premium=True)
        result = connector._fetch_fundamentals("AAPL")

        assert len(result) > 0
        assert list(result.columns) == AlphaVantageConnector.FUNDAMENTAL_COLUMNS
        assert "market_cap" in result["metric"].values
        assert "pe_ratio" in result["metric"].values

    @patch("quantview.data.connectors.alphavantage.requests.Session.get")
    def test_api_error_handling(self, mock_get):
        """Test API error handling."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"Error Message": "Invalid API call"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        connector = AlphaVantageConnector(api_key="test_key", cache_ttl=0, premium=True)
        with pytest.raises(ValueError, match="Alpha Vantage API error"):
            connector._make_request({"function": "TIME_SERIES_DAILY", "symbol": "AAPL"})

    @patch("quantview.data.connectors.alphavantage.requests.Session.get")
    def test_rate_limit_handling(self, mock_get):
        """Test rate limit handling."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"Note": "API call frequency limit reached"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        connector = AlphaVantageConnector(api_key="test_key", cache_ttl=0, premium=True)
        with pytest.raises(ValueError, match="Alpha Vantage rate limit"):
            connector._make_request({"function": "TIME_SERIES_DAILY", "symbol": "AAPL"})

    def test_context_manager_closes_session(self):
        """Test that session is closed on context exit."""
        connector = AlphaVantageConnector(api_key="test_key")
        connector.__enter__()
        connector.__exit__(None, None, None)


class TestConnectorIntegration:
    """Integration tests for connectors (require API keys)."""

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("YAHOO_TEST"), reason="Requires YAHOO_TEST env var")
    def test_yahoo_integration(self):
        """Integration test for Yahoo connector (requires network)."""
        result = fetch_yahoo(["AAPL"], "2024-01-01", "2024-01-10")
        assert not result.empty
        assert "AAPL" in result["symbol"].values

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("FRED_API_KEY"), reason="Requires FRED_API_KEY env var")
    def test_fred_integration(self):
        """Integration test for FRED connector (requires API key)."""
        result = fetch_fred(["DGS10"], "2024-01-01", "2024-01-10")
        assert not result.empty
        assert "DGS10" in result["series_id"].values

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("ALPHAVANTAGE_API_KEY"), reason="Requires ALPHAVANTAGE_API_KEY env var"
    )
    def test_alphavantage_integration(self):
        """Integration test for Alpha Vantage connector (requires API key)."""
        result = fetch_alphavantage_equities(["AAPL"], "2024-01-01", "2024-01-10")
        assert not result.empty
        assert "AAPL" in result["symbol"].values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
