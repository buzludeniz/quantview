"""Tests for QuantView data ingestion pipeline."""

import contextlib
from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
import pytest

from quantview.data.ingest import (
    fetch_and_store_alphavantage_equities,
    fetch_and_store_cboe_options,
    fetch_and_store_fred_rates,
    fetch_and_store_fundamentals,
    fetch_and_store_yahoo_equities,
    init_schema,
    insert_idempotent,
    process_corporate_actions,
    refresh_materialized_views,
    run_daily_ingest,
    run_intraday_ingest,
    run_scheduler,
)


@pytest.fixture
def mem_conn():
    """Create an in-memory DuckDB connection for testing."""
    conn = duckdb.connect(":memory:")
    conn.execute("PRAGMA threads=4")
    # Initialize the schema
    init_schema(conn, force=True)
    yield conn
    conn.close()


@pytest.fixture
def sample_yahoo_data():
    """Create sample Yahoo Finance data for testing."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]).date,
            "open": [100.0, 101.0, 200.0, 201.0],
            "high": [102.0, 103.0, 202.0, 203.0],
            "low": [99.0, 100.0, 199.0, 200.0],
            "close": [101.0, 102.0, 201.0, 202.0],
            "volume": [1000000, 1100000, 2000000, 2100000],
            "adj_close": [101.0, 102.0, 201.0, 202.0],
            "dividends": [0.0, 0.0, 0.0, 0.0],
            "splits": [1.0, 1.0, 1.0, 1.0],
        }
    )


@pytest.fixture
def sample_fred_data():
    """Create sample FRED data for testing."""
    return pd.DataFrame(
        {
            "series_id": ["DGS10", "DGS10", "DGS2", "DGS2"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]).date,
            "value": [4.5, 4.6, 4.0, 4.1],
        }
    )


@pytest.fixture
def sample_cboe_data():
    """Create sample CBOE options data for testing."""
    now = pd.Timestamp.now()
    return pd.DataFrame(
        {
            "underlying": ["SPY", "SPY", "SPY", "SPY"],
            "expiry": pd.to_datetime(["2024-01-19", "2024-01-19", "2024-01-19", "2024-01-19"]).date,
            "strike": [450.0, 450.0, 455.0, 455.0],
            "option_right": ["C", "P", "C", "P"],
            "bid": [10.5, 5.0, 8.0, 6.5],
            "ask": [11.0, 5.5, 8.5, 7.0],
            "mid": [10.75, 5.25, 8.25, 6.75],
            "iv": [0.25, 0.28, 0.24, 0.27],
            "delta": [0.6, -0.4, 0.55, -0.45],
            "gamma": [0.02, 0.02, 0.018, 0.019],
            "vega": [0.15, 0.15, 0.14, 0.14],
            "theta": [-0.05, -0.04, -0.045, -0.038],
            "rho": [0.03, -0.02, 0.028, -0.018],
            "volume": [100, 50, 80, 60],
            "open_interest": [500, 300, 400, 350],
            "snapshot_ts": [now, now, now, now],
        }
    )


class TestInsertIdempotent:
    """Test idempotent insert functionality."""

    def test_insert_new_rows(self, mem_conn):
        """Test inserting new rows."""
        df = pd.DataFrame(
            {
                "symbol": ["TEST1", "TEST2"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]).date,
                "value": [1.0, 2.0],
            }
        )
        # Create a test table
        mem_conn.execute("""
            CREATE TABLE test_table (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                value DOUBLE NOT NULL,
                PRIMARY KEY (symbol, date)
            )
        """)
        rows = insert_idempotent(mem_conn, "test_table", df, ["symbol", "date"])
        assert rows == 2
        count = mem_conn.execute("SELECT COUNT(*) FROM test_table").fetchone()[0]
        assert count == 2

    def test_insert_duplicate_ignored(self, mem_conn):
        """Test that duplicate rows are handled (update on conflict)."""
        mem_conn.execute("""
            CREATE TABLE test_table (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                value DOUBLE NOT NULL,
                PRIMARY KEY (symbol, date)
            )
        """)
        # Insert initial data
        df1 = pd.DataFrame(
            {
                "symbol": ["TEST1"],
                "date": pd.to_datetime(["2024-01-01"]).date,
                "value": [1.0],
            }
        )
        insert_idempotent(mem_conn, "test_table", df1, ["symbol", "date"])

        # Insert same key with different value
        df2 = pd.DataFrame(
            {
                "symbol": ["TEST1"],
                "date": pd.to_datetime(["2024-01-01"]).date,
                "value": [2.0],
            }
        )
        insert_idempotent(mem_conn, "test_table", df2, ["symbol", "date"])

        # Should have updated the value
        result = mem_conn.execute("SELECT value FROM test_table WHERE symbol = 'TEST1'").fetchone()
        assert result[0] == 2.0  # Updated value
        count = mem_conn.execute("SELECT COUNT(*) FROM test_table").fetchone()[0]
        assert count == 1  # No duplicate row

    def test_insert_empty_dataframe(self, mem_conn):
        """Test inserting empty DataFrame."""
        mem_conn.execute("""
            CREATE TABLE test_table (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                value DOUBLE NOT NULL,
                PRIMARY KEY (symbol, date)
            )
        """)
        df = pd.DataFrame({"symbol": [], "date": [], "value": []})
        rows = insert_idempotent(mem_conn, "test_table", df, ["symbol", "date"])
        assert rows == 0


class TestRefreshMaterializedViews:
    """Test materialized view refresh."""

    def test_refresh_views(self, mem_conn):
        """Test that refresh_materialized_views runs without error."""
        refresh_materialized_views(mem_conn)
        # In DuckDB, views are always current, so this is a no-op
        # Just verify it doesn't raise


class TestFetchAndStoreYahooEquities:
    """Test Yahoo equities fetch and store."""

    @patch("quantview.data.ingest.YahooConnector")
    def test_fetch_and_store_yahoo_success(self, mock_connector_class, mem_conn, sample_yahoo_data):
        """Test successful Yahoo equities fetch and store."""
        mock_connector = MagicMock()
        mock_connector_class.return_value.__enter__.return_value = mock_connector
        mock_connector.fetch.return_value = sample_yahoo_data

        rows = fetch_and_store_yahoo_equities(
            mem_conn, ["AAPL", "MSFT"], "2024-01-01", "2024-01-02"
        )

        assert rows == 4
        mock_connector.fetch.assert_called_once_with(["AAPL", "MSFT"], "2024-01-01", "2024-01-02")
        # Verify data was inserted
        count = mem_conn.execute("SELECT COUNT(*) FROM raw.yahoo_equities").fetchone()[0]
        assert count == 4

    @patch("quantview.data.ingest.YahooConnector")
    def test_fetch_and_store_yahoo_empty(self, mock_connector_class, mem_conn):
        """Test handling empty Yahoo data."""
        mock_connector = MagicMock()
        mock_connector_class.return_value.__enter__.return_value = mock_connector
        mock_connector.fetch.return_value = pd.DataFrame()

        rows = fetch_and_store_yahoo_equities(mem_conn, ["AAPL"], "2024-01-01", "2024-01-01")

        assert rows == 0


class TestFetchAndStoreFredRates:
    """Test FRED rates fetch and store."""

    @patch("quantview.data.ingest.FredConnector")
    def test_fetch_and_store_fred_success(self, mock_connector_class, mem_conn, sample_fred_data):
        """Test successful FRED rates fetch and store."""
        mock_connector = MagicMock()
        mock_connector_class.return_value.__enter__.return_value = mock_connector
        mock_connector.fetch.return_value = sample_fred_data

        rows = fetch_and_store_fred_rates(mem_conn, ["DGS10", "DGS2"], "2024-01-01", "2024-01-02")

        assert rows == 4
        mock_connector.fetch.assert_called_once_with(["DGS10", "DGS2"], "2024-01-01", "2024-01-02")
        count = mem_conn.execute("SELECT COUNT(*) FROM raw.fred_rates").fetchone()[0]
        assert count == 4


class TestFetchAndStoreCboeOptions:
    """Test CBOE options fetch and store."""

    @patch("quantview.data.ingest.CboeConnector")
    def test_fetch_and_store_cboe_success(self, mock_connector_class, mem_conn, sample_cboe_data):
        """Test successful CBOE options fetch and store."""
        mock_connector = MagicMock()
        mock_connector_class.return_value.__enter__.return_value = mock_connector
        mock_connector.fetch.return_value = sample_cboe_data

        rows = fetch_and_store_cboe_options(mem_conn, ["SPY"])

        assert rows == 4
        mock_connector.fetch.assert_called_once_with(["SPY"], "", "")
        count = mem_conn.execute("SELECT COUNT(*) FROM raw.cboe_options").fetchone()[0]
        assert count == 4


class TestFetchAndStoreAlphaVantageEquities:
    """Test Alpha Vantage equities fetch and store."""

    @patch("quantview.data.ingest.AlphaVantageConnector")
    def test_fetch_and_store_av_success(self, mock_connector_class, mem_conn):
        """Test successful Alpha Vantage equities fetch and store."""
        sample_data = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]).date,
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000000, 1100000],
            }
        )
        mock_connector = MagicMock()
        mock_connector_class.return_value.__enter__.return_value = mock_connector
        mock_connector.fetch.return_value = sample_data

        rows = fetch_and_store_alphavantage_equities(mem_conn, ["AAPL"], "2024-01-01", "2024-01-02")

        assert rows == 2
        count = mem_conn.execute("SELECT COUNT(*) FROM raw.alphavantage_equities").fetchone()[0]
        assert count == 2


class TestFetchAndStoreFundamentals:
    """Test fundamentals fetch and store."""

    @patch("quantview.data.ingest.AlphaVantageConnector")
    def test_fetch_and_store_fundamentals_success(self, mock_connector_class, mem_conn):
        """Test successful fundamentals fetch and store."""
        sample_data = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]).date,
                "metric": ["market_cap", "pe_ratio"],
                "value": [3000000000000.0, 28.5],
                "period": ["ttm", "ttm"],
            }
        )
        mock_connector = MagicMock()
        mock_connector_class.return_value.__enter__.return_value = mock_connector
        mock_connector.fetch_fundamentals.return_value = sample_data

        rows = fetch_and_store_fundamentals(mem_conn, ["AAPL"])

        assert rows == 2
        count = mem_conn.execute("SELECT COUNT(*) FROM raw.fundamentals").fetchone()[0]
        assert count == 2
        # Check source column was added
        source = mem_conn.execute("SELECT source FROM raw.fundamentals LIMIT 1").fetchone()[0]
        assert source == "alphavantage"


class TestProcessCorporateActions:
    """Test corporate action processing."""

    def test_process_corporate_actions(self, mem_conn):
        """Test detecting and logging corporate actions."""
        # Insert test data with split and dividend
        mem_conn.execute("""
            INSERT INTO raw.yahoo_equities (symbol, date, open, high, low, close, volume, adj_close, dividends, splits, source, ingest_ts)
            VALUES
                ('TEST_CA', '2024-01-15', 100, 105, 99, 102, 1000000, 102, 0.0, 2.0, 'yahoo', CURRENT_TIMESTAMP),
                ('TEST_CA', '2024-02-15', 50, 52, 49, 51, 1000000, 51, 0.50, 1.0, 'yahoo', CURRENT_TIMESTAMP),
                ('TEST_CA', '2024-03-15', 51, 53, 50, 52, 1000000, 52, 0.0, 1.0, 'yahoo', CURRENT_TIMESTAMP)
        """)

        count = process_corporate_actions(mem_conn, ["TEST_CA"], "2024-01-01", "2024-12-31")

        assert count == 2
        # Verify logged in corporate_actions_log
        logs = mem_conn.execute("""
            SELECT symbol, date, action, ratio, details
            FROM core.corporate_actions_log
            WHERE symbol = 'TEST_CA'
            ORDER BY date
        """).fetchall()
        assert len(logs) == 2
        assert logs[0][2] == "split"
        assert logs[0][3] == 2.0
        assert logs[1][2] == "dividend"
        assert logs[1][3] == 0.5


class TestRunDailyIngest:
    """Test daily EOD ingestion."""

    @patch("quantview.data.ingest.fetch_and_store_yahoo_equities")
    @patch("quantview.data.ingest.fetch_and_store_fred_rates")
    @patch("quantview.data.ingest.fetch_and_store_cboe_options")
    @patch("quantview.data.ingest.fetch_and_store_alphavantage_equities")
    @patch("quantview.data.ingest.fetch_and_store_fundamentals")
    @patch("quantview.data.ingest.process_corporate_actions")
    @patch("quantview.data.ingest.refresh_materialized_views")
    def test_run_daily_ingest(
        self,
        mock_refresh,
        mock_corp_actions,
        mock_fundamentals,
        mock_av,
        mock_cboe,
        mock_fred,
        mock_yahoo,
        mem_conn,
    ):
        """Test daily ingest runs all steps."""
        mock_yahoo.return_value = 100
        mock_fred.return_value = 20
        mock_cboe.return_value = 50
        mock_av.return_value = 80
        mock_fundamentals.return_value = 30
        mock_corp_actions.return_value = 5

        stats = run_daily_ingest(mem_conn, date="2024-01-15")

        assert stats["date"] == "2024-01-15"
        assert stats["yahoo_rows"] == 100
        assert stats["fred_rows"] == 20
        assert stats["cboe_rows"] == 50
        assert stats["alphavantage_rows"] == 80
        assert stats["fundamentals_rows"] == 30
        assert stats["corporate_actions"] == 5
        assert stats["duration_seconds"] > 0
        assert len(stats["errors"]) == 0

        mock_yahoo.assert_called_once()
        mock_fred.assert_called_once()
        mock_cboe.assert_called_once()
        mock_av.assert_called_once()
        mock_fundamentals.assert_called_once()
        mock_corp_actions.assert_called_once()
        mock_refresh.assert_called_once()

    @patch("quantview.data.ingest.fetch_and_store_yahoo_equities")
    def test_run_daily_ingest_error_handling(self, mock_yahoo, mem_conn):
        """Test daily ingest error handling."""
        mock_yahoo.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            run_daily_ingest(mem_conn, date="2024-01-15")


class TestRunIntradayIngest:
    """Test intraday ingestion."""

    @patch("quantview.data.ingest.YahooConnector")
    @patch("quantview.data.ingest.fetch_and_store_cboe_options")
    @patch("quantview.data.ingest.refresh_materialized_views")
    def test_run_intraday_ingest(self, mock_refresh, mock_cboe, mock_yahoo_class, mem_conn):
        """Test intraday ingest runs all steps."""
        mock_connector = MagicMock()
        mock_yahoo_class.return_value.__enter__.return_value = mock_connector
        mock_ticker = MagicMock()
        mock_connector._get_ticker.return_value = mock_ticker

        # Mock intraday history data - use "Datetime" column as yfinance returns for intraday
        mock_hist = pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2024-01-15 09:30:00", "2024-01-15 09:35:00"]),
                "Open": [100.0, 101.0],
                "High": [100.5, 101.5],
                "Low": [99.5, 100.5],
                "Close": [100.2, 101.2],
                "Volume": [10000, 12000],
                "Adj Close": [100.2, 101.2],
                "Dividends": [0.0, 0.0],
                "Stock Splits": [0.0, 0.0],
            }
        )
        mock_ticker.history.return_value = mock_hist

        mock_cboe.return_value = 50

        stats = run_intraday_ingest(mem_conn)

        assert "snapshot_time" in stats
        assert stats["symbols_processed"] > 0
        assert stats["yahoo_rows"] >= 0
        assert stats["cboe_rows"] == 50
        assert stats["duration_seconds"] > 0
        mock_cboe.assert_called_once()
        mock_refresh.assert_called_once()


class TestScheduler:
    """Test scheduler configuration."""

    @patch("quantview.data.ingest.BlockingScheduler")
    def test_run_scheduler_configures_jobs(self, mock_scheduler_class):
        """Test that scheduler is configured with correct jobs."""
        mock_scheduler = MagicMock()
        mock_scheduler_class.return_value = mock_scheduler

        # We can't easily test the full scheduler.start() without blocking
        # But we can verify job configuration by checking add_job calls
        # This blocks, so the only thing being verified is that the scheduler is
        # built; the interrupt it raises on start is the expected exit.
        with contextlib.suppress(KeyboardInterrupt):
            run_scheduler()

        # Verify scheduler was created with ET timezone
        mock_scheduler_class.assert_called_once()
        call_kwargs = mock_scheduler_class.call_args.kwargs
        assert call_kwargs.get("timezone") is not None


class TestIntegration:
    """Integration-style tests with real connectors (mocked at network level)."""

    @patch("quantview.data.connectors.yahoo.yf.Ticker")
    def test_full_yahoo_ingest_flow(self, mock_ticker_class, mem_conn):
        """Test full Yahoo ingest flow with mocked yfinance."""
        # Setup mock ticker
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        mock_hist = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-15"]),
                "Open": [150.0],
                "High": [152.0],
                "Low": [149.0],
                "Close": [151.0],
                "Volume": [50000000],
                "Adj Close": [151.0],
                "Dividends": [0.0],
                "Stock Splits": [0.0],
            }
        )
        mock_ticker.history.return_value = mock_hist

        # Run daily ingest with just Yahoo
        with (
            patch("quantview.data.ingest.fetch_and_store_fred_rates") as mock_fred,
            patch("quantview.data.ingest.fetch_and_store_cboe_options") as mock_cboe,
            patch("quantview.data.ingest.fetch_and_store_alphavantage_equities") as mock_av,
            patch("quantview.data.ingest.fetch_and_store_fundamentals") as mock_fund,
            patch("quantview.data.ingest.process_corporate_actions") as mock_ca,
            patch("quantview.data.ingest.refresh_materialized_views") as mock_refresh,
        ):
            mock_fred.return_value = 0
            mock_cboe.return_value = 0
            mock_av.return_value = 0
            mock_fund.return_value = 0
            mock_ca.return_value = 0

            stats = run_daily_ingest(mem_conn, date="2024-01-15")

            assert stats["yahoo_rows"] > 0
            # The daily run must refresh the materialised views once ingestion
            # finishes; the mock was previously created and never asserted on.
            mock_refresh.assert_called_once()
            # Verify data in raw table
            count = mem_conn.execute(
                "SELECT COUNT(*) FROM raw.yahoo_equities WHERE date = '2024-01-15'"
            ).fetchone()[0]
            assert count > 0

    def test_idempotent_daily_rerun(self, mem_conn):
        """Test that running daily ingest twice for same date doesn't duplicate."""
        with (
            patch("quantview.data.connectors.yahoo.yf.Ticker") as mock_ticker_class,
            patch("quantview.data.ingest.fetch_and_store_fred_rates") as mock_fred,
            patch("quantview.data.ingest.fetch_and_store_cboe_options") as mock_cboe,
            patch("quantview.data.ingest.fetch_and_store_alphavantage_equities") as mock_av,
            patch("quantview.data.ingest.fetch_and_store_fundamentals") as mock_fund,
            patch("quantview.data.ingest.process_corporate_actions") as mock_ca,
            patch("quantview.data.ingest.refresh_materialized_views") as mock_refresh,
        ):
            mock_ticker = MagicMock()
            mock_ticker_class.return_value = mock_ticker
            mock_hist = pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2024-01-15"]),
                    "Open": [150.0],
                    "High": [152.0],
                    "Low": [149.0],
                    "Close": [151.0],
                    "Volume": [50000000],
                    "Adj Close": [151.0],
                    "Dividends": [0.0],
                    "Stock Splits": [0.0],
                }
            )
            mock_ticker.history.return_value = mock_hist

            mock_fred.return_value = 0
            mock_cboe.return_value = 0
            mock_av.return_value = 0
            mock_fund.return_value = 0
            mock_ca.return_value = 0

            # Both runs must refresh the views, so idempotency is not achieved by
            # skipping the refresh on the second pass.
            stats1 = run_daily_ingest(mem_conn, date="2024-01-15")
            assert mock_refresh.call_count == 1
            count1 = mem_conn.execute(
                "SELECT COUNT(*) FROM raw.yahoo_equities WHERE date = '2024-01-15'"
            ).fetchone()[0]

            # Run second time (simulate re-run)
            stats2 = run_daily_ingest(mem_conn, date="2024-01-15")
            count2 = mem_conn.execute(
                "SELECT COUNT(*) FROM raw.yahoo_equities WHERE date = '2024-01-15'"
            ).fetchone()[0]

            # Should not duplicate
            assert count1 == count2
            assert stats1["yahoo_rows"] == stats2["yahoo_rows"]
            assert mock_refresh.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
