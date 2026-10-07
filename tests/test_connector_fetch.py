"""Tests for the connector fetch paths.

test_connectors.py covers construction, column schemas and API-key guards. The
public ``fetch`` methods and their parsing helpers sat near half covered, which
left the parts that decide whether a caller's frame is trustworthy untested:
cache reuse, per-symbol failure isolation, and how CBOE's nested JSON becomes
rows. These tests drive the HTTP session with fakes, so nothing touches network.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantview.data.connectors.alphavantage import AlphaVantageConnector
from quantview.data.connectors.cboe import CboeConnector
from quantview.data.connectors.fred import FredConnector
from quantview.data.connectors.yahoo import YahooConnector


@pytest.fixture
def no_retry_sleep():
    """Skip the exponential backoff so failure paths run in milliseconds.

    The retry schedule waits 1, 2, 4, 8 and 60 seconds. Sleeping through it makes
    a single test take over a minute without testing anything about the retry.
    """
    with patch("quantview.data.connectors.base.time.sleep"):
        yield


def _response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _session_returning(payload: dict) -> MagicMock:
    """A stand-in for requests.Session whose get() yields that payload."""
    session = MagicMock()
    session.get.return_value = _response(payload)
    return session


# --------------------------------------------------------------------------
# CBOE
# --------------------------------------------------------------------------

CBOE_PAYLOAD = {
    "data": {
        "options": {
            "2024-01-19": {
                "450.0": {
                    "call": {
                        "quote": {"bid": 10.5, "ask": 11.0, "volume": 100, "open_interest": 500},
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
                        "quote": {"bid": 12.0, "ask": 12.5, "volume": 80, "open_interest": 400},
                        "greeks": {"iv": 0.27, "delta": -0.4, "gamma": 0.02, "vega": 0.15},
                    },
                }
            }
        }
    }
}


class TestCboeParsing:
    def test_expands_each_strike_into_a_call_and_a_put(self):
        conn = CboeConnector()

        df = conn._parse_options_data("SPY", CBOE_PAYLOAD)

        assert len(df) == 2
        assert sorted(df["option_right"]) == ["C", "P"]

    def test_derives_mid_from_the_quote(self):
        conn = CboeConnector()

        df = conn._parse_options_data("SPY", CBOE_PAYLOAD)
        call = df[df["option_right"] == "C"].iloc[0]

        assert call["mid"] == pytest.approx((10.5 + 11.0) / 2)

    def test_leaves_mid_empty_when_a_side_is_missing(self):
        """A one-sided quote has no mid; inventing one would overstate the price."""
        payload = {
            "data": {
                "options": {
                    "2024-01-19": {
                        "450.0": {
                            "call": {"quote": {"bid": 10.5}, "greeks": {}},
                            "put": {"quote": {"bid": 12.0, "ask": 12.5}, "greeks": {}},
                        }
                    }
                }
            }
        }

        df = CboeConnector()._parse_options_data("SPY", payload)

        assert pd.isna(df[df["option_right"] == "C"].iloc[0]["mid"])
        assert df[df["option_right"] == "P"].iloc[0]["mid"] == pytest.approx(12.25)

    def test_carries_the_full_underlying_and_expiry_through(self):
        conn = CboeConnector()

        df = conn._parse_options_data("spy", CBOE_PAYLOAD)

        assert set(df["underlying"]) == {"SPY"}
        assert set(df["expiry"]) == {pd.Timestamp("2024-01-19").date()}

    def test_skips_an_unparseable_expiry_without_losing_the_rest(self):
        payload = {
            "data": {
                "options": {
                    "not-a-date": {"450.0": {"call": {"quote": {}, "greeks": {}}}},
                    "2024-01-19": {
                        "450.0": {"call": {"quote": {"bid": 1.0, "ask": 2.0}, "greeks": {}}}
                    },
                }
            }
        }

        df = CboeConnector()._parse_options_data("SPY", payload)

        assert len(df) == 1
        assert df.iloc[0]["expiry"] == pd.Timestamp("2024-01-19").date()

    def test_skips_an_unparseable_strike(self):
        payload = {
            "data": {
                "options": {
                    "2024-01-19": {
                        "not-a-number": {"call": {"quote": {}, "greeks": {}}},
                        "450.0": {"call": {"quote": {"bid": 1.0, "ask": 2.0}, "greeks": {}}},
                    }
                }
            }
        }

        df = CboeConnector()._parse_options_data("SPY", payload)

        assert len(df) == 1

    def test_skips_an_empty_option_leg(self):
        payload = {
            "data": {
                "options": {
                    "2024-01-19": {
                        "450.0": {
                            "call": None,
                            "put": {"quote": {"bid": 1.0, "ask": 2.0}, "greeks": {}},
                        }
                    }
                }
            }
        }

        df = CboeConnector()._parse_options_data("SPY", payload)

        assert list(df["option_right"]) == ["P"]

    def test_returns_the_declared_columns_for_an_empty_payload(self):
        df = CboeConnector()._parse_options_data("SPY", {"data": {"options": {}}})

        assert df.empty
        assert list(df.columns) == CboeConnector.REQUIRED_COLUMNS

    def test_tolerates_null_greeks_and_quote_blocks(self):
        """CBOE sends explicit nulls; the parser must not choke on them."""
        payload = {
            "data": {
                "options": {
                    "2024-01-19": {
                        "450.0": {"call": {"quote": None, "greeks": None}},
                    }
                }
            }
        }

        df = CboeConnector()._parse_options_data("SPY", payload)

        assert len(df) == 1
        assert pd.isna(df.iloc[0]["bid"])


class TestCboeFetch:
    @patch("quantview.data.connectors.cboe.requests.Session.get")
    def test_returns_one_row_per_contract(self, mock_get):
        mock_get.return_value = _response(CBOE_PAYLOAD)

        with CboeConnector() as conn:
            df = conn.fetch(["SPY"], "2024-01-01", "2024-01-31")

        assert len(df) == 2

    @patch("quantview.data.connectors.cboe.requests.Session.get")
    def test_stamps_a_snapshot_timestamp(self, mock_get):
        mock_get.return_value = _response(CBOE_PAYLOAD)

        with CboeConnector() as conn:
            df = conn.fetch(["SPY"], "2024-01-01", "2024-01-31")

        assert "snapshot_ts" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["snapshot_ts"])

    @patch("quantview.data.connectors.cboe.requests.Session.get")
    def test_concatenates_across_symbols(self, mock_get):
        mock_get.return_value = _response(CBOE_PAYLOAD)

        with CboeConnector() as conn:
            df = conn.fetch(["SPY", "QQQ"], "2024-01-01", "2024-01-31")

        assert set(df["underlying"]) == {"SPY", "QQQ"}

    @patch("quantview.data.connectors.cboe.requests.Session.get")
    def test_one_failing_symbol_does_not_lose_the_other(self, mock_get, no_retry_sleep):
        """A single unreachable symbol must not empty the whole result."""
        good = _response(CBOE_PAYLOAD)

        def get(url, **kwargs):
            if "/BAD.json" in url:
                bad = MagicMock()
                bad.raise_for_status.side_effect = RuntimeError("503 from CBOE")
                return bad
            return good

        mock_get.side_effect = get

        with CboeConnector(cache_ttl=0) as conn:
            df = conn.fetch(["BAD", "SPY"], "2024-01-01", "2024-01-31")

        assert set(df["underlying"]) == {"SPY"}

    @patch("quantview.data.connectors.cboe.requests.Session.get")
    def test_returns_the_declared_columns_when_everything_fails(self, mock_get, no_retry_sleep):
        bad = MagicMock()
        bad.raise_for_status.side_effect = RuntimeError("down")
        mock_get.return_value = bad

        with CboeConnector(cache_ttl=0) as conn:
            df = conn.fetch(["SPY"], "2024-01-01", "2024-01-31")

        assert df.empty
        assert list(df.columns) == [*CboeConnector.REQUIRED_COLUMNS, "snapshot_ts"]

    @patch("quantview.data.connectors.cboe.requests.Session.get")
    def test_a_repeat_request_is_served_from_cache(self, mock_get):
        mock_get.return_value = _response(CBOE_PAYLOAD)

        with CboeConnector(cache_ttl=300) as conn:
            conn.fetch(["SPY"], "2024-01-01", "2024-01-31")
            calls_after_first = mock_get.call_count
            conn.fetch(["SPY"], "2024-01-01", "2024-01-31")

        assert mock_get.call_count == calls_after_first

    @patch("quantview.data.connectors.cboe.requests.Session.get")
    def test_a_different_symbol_list_bypasses_the_cache(self, mock_get):
        """The cache key is the symbol set, so new symbols must trigger a fetch."""
        mock_get.return_value = _response(CBOE_PAYLOAD)

        with CboeConnector(cache_ttl=300) as conn:
            conn.fetch(["SPY"], "2024-01-01", "2024-01-31")
            conn.fetch(["SPY", "QQQ"], "2024-01-01", "2024-01-31")
            second = mock_get.call_count

        assert second > 1

    @patch("quantview.data.connectors.cboe.requests.Session.get")
    def test_symbol_order_does_not_defeat_the_cache(self, mock_get):
        mock_get.return_value = _response(CBOE_PAYLOAD)

        with CboeConnector(cache_ttl=300) as conn:
            conn.fetch(["SPY", "QQQ"], "2024-01-01", "2024-01-31")
            after_first = mock_get.call_count
            conn.fetch(["QQQ", "SPY"], "2024-01-01", "2024-01-31")

        assert mock_get.call_count == after_first

    @patch("quantview.data.connectors.cboe.requests.Session.get")
    def test_an_empty_symbol_list_short_circuits(self, mock_get):
        with CboeConnector() as conn:
            df = conn.fetch([], "2024-01-01", "2024-01-31")

        assert df.empty
        mock_get.assert_not_called()

    @patch("quantview.data.connectors.cboe.requests.Session.get")
    def test_the_caller_cannot_corrupt_the_cache(self, mock_get):
        """fetch caches a copy; mutating the result must not poison the cache."""
        mock_get.return_value = _response(CBOE_PAYLOAD)

        with CboeConnector(cache_ttl=300) as conn:
            first = conn.fetch(["SPY"], "2024-01-01", "2024-01-31")
            first.loc[:, "underlying"] = "MUTATED"
            second = conn.fetch(["SPY"], "2024-01-01", "2024-01-31")

        assert set(second["underlying"]) == {"SPY"}


class TestCboeUnderlying:
    @patch("quantview.data.connectors.cboe.requests.Session.get")
    def test_reads_the_current_price(self, mock_get):
        mock_get.return_value = _response({"data": {"current_price": 452.25}})

        with CboeConnector() as conn:
            assert conn._fetch_underlying_price("SPY") == pytest.approx(452.25)

    @patch("quantview.data.connectors.cboe.requests.Session.get")
    def test_returns_none_instead_of_raising(self, mock_get):
        """Moneyness is an enrichment; losing it must not fail the chain fetch."""
        bad = MagicMock()
        bad.raise_for_status.side_effect = RuntimeError("404")
        mock_get.return_value = bad

        with CboeConnector() as conn:
            assert conn._fetch_underlying_price("SPY") is None

    @patch("quantview.data.connectors.cboe.requests.Session.get")
    def test_uppercases_the_symbol_in_the_url(self, mock_get):
        mock_get.return_value = _response({"data": {"current_price": 1.0}})

        with CboeConnector() as conn:
            conn._fetch_underlying_price("spy")

        assert "/SPY.json" in mock_get.call_args[0][0]


# --------------------------------------------------------------------------
# Yahoo
# --------------------------------------------------------------------------

# yfinance returns dates as the index name "Date"; the connector resets that index
# to a column before renaming, so the fixture has to carry it.
YAHOO_HISTORY = pd.DataFrame(
    {
        "Open": [150.0, 151.0],
        "High": [152.0, 153.0],
        "Low": [149.0, 150.0],
        "Close": [151.0, 152.0],
        "Volume": [50_000_000, 51_000_000],
        "Adj Close": [151.0, 152.0],
        "Dividends": [0.0, 0.0],
        "Stock Splits": [0.0, 0.0],
    },
    index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]), name="Date"),
)


class TestYahooFetch:
    @patch("quantview.data.connectors.yahoo.yf.Ticker")
    def test_normalises_yfinance_columns(self, mock_ticker):
        mock_ticker.return_value.history.return_value = YAHOO_HISTORY

        with YahooConnector(cache_ttl=0) as conn:
            df = conn.fetch(["AAPL"], "2024-01-01", "2024-01-05")

        assert list(df.columns) == YahooConnector.REQUIRED_COLUMNS
        assert set(df["symbol"]) == {"AAPL"}

    @patch("quantview.data.connectors.yahoo.yf.Ticker")
    def test_renames_the_awkward_yfinance_names(self, mock_ticker):
        """'Adj Close' and 'Stock Splits' must become schema columns."""
        mock_ticker.return_value.history.return_value = YAHOO_HISTORY

        with YahooConnector(cache_ttl=0) as conn:
            df = conn.fetch(["AAPL"], "2024-01-01", "2024-01-05")

        assert "adj_close" in df.columns
        assert "splits" in df.columns
        assert "Adj Close" not in df.columns

    @patch("quantview.data.connectors.yahoo.yf.Ticker")
    def test_includes_the_date_in_range(self, mock_ticker):
        mock_ticker.return_value.history.return_value = YAHOO_HISTORY

        with YahooConnector(cache_ttl=0) as conn:
            df = conn.fetch(["AAPL"], "2024-01-01", "2024-01-05")

        assert len(df) == 2

    @patch("quantview.data.connectors.yahoo.yf.Ticker")
    def test_concatenates_multiple_symbols(self, mock_ticker):
        mock_ticker.return_value.history.return_value = YAHOO_HISTORY

        with YahooConnector(cache_ttl=0) as conn:
            df = conn.fetch(["AAPL", "MSFT"], "2024-01-01", "2024-01-05")

        assert set(df["symbol"]) == {"AAPL", "MSFT"}

    @patch("quantview.data.connectors.yahoo.yf.Ticker")
    def test_an_empty_history_is_not_an_error(self, mock_ticker):
        mock_ticker.return_value.history.return_value = pd.DataFrame()

        with YahooConnector(cache_ttl=0) as conn:
            df = conn.fetch(["AAPL"], "2024-01-01", "2024-01-05")

        assert df.empty
        assert list(df.columns) == YahooConnector.REQUIRED_COLUMNS

    @patch("quantview.data.connectors.yahoo.yf.Ticker")
    def test_an_empty_symbol_list_short_circuits(self, mock_ticker):
        with YahooConnector() as conn:
            df = conn.fetch([], "2024-01-01", "2024-01-05")

        assert df.empty
        mock_ticker.assert_not_called()


# --------------------------------------------------------------------------
# FRED
# --------------------------------------------------------------------------


class TestFredFetch:
    def _connector(self, monkeypatch) -> FredConnector:
        monkeypatch.setattr(FredConnector, "_get_fred", lambda self: MagicMock())
        return FredConnector(api_key="test-key", cache_ttl=0)

    def test_writes_the_three_column_schema(self, monkeypatch):
        conn = self._connector(monkeypatch)
        frame = pd.DataFrame(
            {"date": ["2024-01-02"], "value": [5.25]},
        )
        conn._get_fred().get_series.return_value = frame

        df = conn.fetch(["DGS10"], "2024-01-01", "2024-01-05")

        assert list(df.columns) == FredConnector.REQUIRED_COLUMNS

    def test_reports_a_series_that_comes_back_empty(self, monkeypatch):
        """An empty frame is normal for a holiday window, not a failure."""
        conn = self._connector(monkeypatch)
        conn._get_fred().get_series.return_value = pd.DataFrame(columns=["date", "value"])

        df = conn.fetch(["DGS10"], "2024-01-01", "2024-01-05")

        assert df.empty


# --------------------------------------------------------------------------
# Alpha Vantage
# --------------------------------------------------------------------------

# Keys are Alpha Vantage's own positional names: 1 open, 2 high, 3 low,
# 4 close, 5 adjusted close, 6 volume.
AV_SERIES = {
    "Time Series (Daily)": {
        "2024-01-02": {
            "1. open": "100.0",
            "2. high": "102.0",
            "3. low": "99.0",
            "4. close": "101.0",
            "5. adjusted close": "101.0",
            "6. volume": "1000000",
        },
        "2024-01-05": {
            "1. open": "101.0",
            "2. high": "104.0",
            "3. low": "100.5",
            "4. close": "103.0",
            "5. adjusted close": "103.0",
            "6. volume": "1200000",
        },
        "2023-12-29": {
            "1. open": "98.0",
            "2. high": "99.0",
            "3. low": "97.0",
            "4. close": "98.5",
            "5. adjusted close": "98.5",
            "6. volume": "900000",
        },
    }
}

AV_OVERVIEW = {
    "Symbol": "AAPL",
    "MarketCapitalization": "3000000000000",
    "PERatio": "28.5",
    "EPS": "6.25",
    "DividendYield": "0.005",
    "Beta": "1.2",
    "52WeekHigh": "199.62",
}

AV_INCOME = {
    "annualReports": [
        {
            "fiscalDateEnding": "2023-09-30",
            "totalRevenue": "383285000000",
            "grossProfit": "169148000000",
            "netIncome": "96995000000",
            "operatingIncome": "114301000000",
            "ebitda": "123489000000",
        }
    ]
}

AV_BALANCE = {
    "annualReports": [
        {
            "fiscalDateEnding": "2023-09-30",
            "totalAssets": "352583000000",
            "totalLiabilities": "290437000000",
            "totalShareholderEquity": "62146000000",
            "cashAndCashEquivalents": "29965000000",
        }
    ]
}


def _av_connector(monkeypatch, responses: dict) -> AlphaVantageConnector:
    """An Alpha Vantage connector whose HTTP layer is a routing table."""
    conn = AlphaVantageConnector(api_key="test-key", cache_ttl=0, premium=False)

    def request(params, **kwargs):
        fn = params["function"]
        if fn not in responses:
            raise RuntimeError(f"unexpected function {fn}")
        return responses[fn]

    monkeypatch.setattr(conn, "_make_request", request)
    return conn


class TestAlphaVantageErrors:
    @pytest.mark.parametrize(
        "payload,expected",
        [
            ({"Error Message": "Invalid API call"}, "API error"),
            ({"Note": "Thank you for using Alpha Vantage"}, "rate limit"),
            ({"Information": "premium endpoint"}, "info"),
        ],
    )
    def test_api_error_responses_raise_rather_than_return_empty(
        self, payload, expected, monkeypatch
    ):
        """Alpha Vantage reports failures with HTTP 200; ignoring that yields
        a silently empty frame that looks like a symbol with no history."""
        conn = AlphaVantageConnector(api_key="test-key")
        monkeypatch.setattr(conn, "_rate_limit", lambda: None)
        conn._session = _session_returning(payload)

        with pytest.raises(ValueError, match=expected):
            conn._make_request({"function": "OVERVIEW", "symbol": "AAPL"})

    def test_the_specific_error_wins_over_the_subscription_notice(self, monkeypatch):
        """A free-tier premium call carries both keys; the reason matters more."""
        conn = AlphaVantageConnector(api_key="test-key")
        monkeypatch.setattr(conn, "_rate_limit", lambda: None)
        conn._session = _session_returning(
            {
                "Information": "This is a premium endpoint.",
                "Error Message": "Invalid API call",
            }
        )

        with pytest.raises(ValueError, match="Invalid API call"):
            conn._make_request({"function": "TIME_SERIES_INTRADAY", "symbol": "AAPL"})

    def test_a_rate_limit_surfaces_rather_than_returning_nothing(self, monkeypatch, no_retry_sleep):
        """Retrying a rate limit just burns quota; the frame comes back empty."""
        conn = AlphaVantageConnector(api_key="test-key", cache_ttl=0)

        def request(params, **kwargs):
            raise ValueError("Alpha Vantage rate limit: slow down")

        monkeypatch.setattr(conn, "_make_request", request)

        df = conn.fetch(["AAPL"], "2024-01-01", "2024-01-05")

        assert df.empty
        assert list(df.columns) == AlphaVantageConnector.EQUITY_COLUMNS


class TestAlphaVantageDaily:
    def test_maps_the_positional_keys_to_named_columns(self, monkeypatch):
        conn = _av_connector(monkeypatch, {"TIME_SERIES_DAILY_ADJUSTED": AV_SERIES})

        df = conn.fetch(["AAPL"], "2024-01-01", "2024-01-31")

        assert list(df.columns) == AlphaVantageConnector.EQUITY_COLUMNS
        assert df.iloc[0]["open"] == pytest.approx(100.0)
        assert df.iloc[0]["volume"] == 1_000_000

    def test_filters_to_the_requested_window(self, monkeypatch):
        """The API returns full history; the caller's window has to be honoured."""
        conn = _av_connector(monkeypatch, {"TIME_SERIES_DAILY_ADJUSTED": AV_SERIES})

        df = conn.fetch(["AAPL"], "2024-01-01", "2024-01-31")

        assert len(df) == 2
        assert all(d >= pd.Timestamp("2024-01-01").date() for d in df["date"])

    def test_inclusive_of_both_endpoints(self, monkeypatch):
        conn = _av_connector(monkeypatch, {"TIME_SERIES_DAILY_ADJUSTED": AV_SERIES})

        df = conn.fetch(["AAPL"], "2024-01-02", "2024-01-05")

        assert len(df) == 2

    def test_sorts_by_date(self, monkeypatch):
        conn = _av_connector(monkeypatch, {"TIME_SERIES_DAILY_ADJUSTED": AV_SERIES})

        df = conn.fetch(["AAPL"], "2024-01-01", "2024-01-31")

        assert list(df["date"]) == sorted(df["date"])

    def test_an_empty_time_series_yields_the_declared_columns(self, monkeypatch):
        conn = _av_connector(monkeypatch, {"TIME_SERIES_DAILY_ADJUSTED": {}})

        df = conn.fetch(["AAPL"], "2024-01-01", "2024-01-31")

        assert df.empty
        assert list(df.columns) == AlphaVantageConnector.EQUITY_COLUMNS

    def test_a_window_that_matches_nothing_yields_the_declared_columns(self, monkeypatch):
        conn = _av_connector(monkeypatch, {"TIME_SERIES_DAILY_ADJUSTED": AV_SERIES})

        df = conn.fetch(["AAPL"], "2020-01-01", "2020-01-31")

        assert df.empty
        assert list(df.columns) == AlphaVantageConnector.EQUITY_COLUMNS

    def test_a_repeat_request_is_served_from_cache(self, monkeypatch):
        conn = _av_connector(monkeypatch, {"TIME_SERIES_DAILY_ADJUSTED": AV_SERIES})
        conn.cache_ttl = 300
        conn.fetch(["AAPL"], "2024-01-01", "2024-01-31")

        def explode(params, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("the cache should have answered")

        monkeypatch.setattr(conn, "_make_request", explode)
        again = conn.fetch(["AAPL"], "2024-01-01", "2024-01-31")

        assert len(again) == 2


class TestAlphaVantageFundamentals:
    def test_flattens_the_overview_into_metric_rows(self, monkeypatch):
        conn = _av_connector(
            monkeypatch,
            {"OVERVIEW": AV_OVERVIEW, "INCOME_STATEMENT": AV_INCOME, "BALANCE_SHEET": AV_BALANCE},
        )

        df = conn.fetch_fundamentals(["AAPL"])

        assert list(df.columns) == AlphaVantageConnector.FUNDAMENTAL_COLUMNS
        assert set(df["period"]) == {"ttm", "annual"}

    def test_renames_the_metrics_to_snake_case(self, monkeypatch):
        conn = _av_connector(
            monkeypatch,
            {"OVERVIEW": AV_OVERVIEW, "INCOME_STATEMENT": AV_INCOME, "BALANCE_SHEET": AV_BALANCE},
        )

        df = conn.fetch_fundamentals(["AAPL"])

        assert "market_cap" in set(df["metric"])
        assert "pe_ratio" in set(df["metric"])
        assert "totalrevenue" in set(df["metric"])

    def test_skips_the_literal_string_none_the_api_sends(self, monkeypatch):
        """Alpha Vantage sends the string "None" for a missing value."""
        conn = _av_connector(
            monkeypatch,
            {"OVERVIEW": {**AV_OVERVIEW, "BookValue": "None"}},
        )

        df = conn.fetch_fundamentals(["AAPL"])

        assert "book_value_per_share" not in set(df["metric"])

    def test_one_failing_report_does_not_lose_the_others(self, monkeypatch):
        def request(params, **kwargs):
            if params["function"] == "OVERVIEW":
                raise RuntimeError("upstream 500")
            if params["function"] == "INCOME_STATEMENT":
                return AV_INCOME
            return AV_BALANCE

        conn = AlphaVantageConnector(api_key="test-key", cache_ttl=0)
        monkeypatch.setattr(conn, "_make_request", request)

        df = conn.fetch_fundamentals(["AAPL"])

        # The statements that did arrive are still reported.
        assert "totalrevenue" in set(df["metric"])
        assert "market_cap" not in set(df["metric"])

    def test_no_api_key_yields_the_declared_columns(self):
        conn = AlphaVantageConnector(api_key=None)

        df = conn.fetch_fundamentals(["AAPL"])

        assert df.empty
        assert list(df.columns) == AlphaVantageConnector.FUNDAMENTAL_COLUMNS

    def test_an_empty_symbol_list_short_circuits(self, monkeypatch):
        conn = _av_connector(monkeypatch, {})

        assert conn.fetch_fundamentals([]).empty
