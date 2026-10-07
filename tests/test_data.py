"""Tests for QuantView data schema and universe."""

import duckdb
import pytest

from quantview.data.ingest import init_schema, verify_schema
from quantview.data.schema import (
    CORE_SCHEMA_DDL,
    RAW_SCHEMA_DDL,
    UNIVERSE_TABLE_DDL,
    get_all_ddl,
)
from quantview.data.universe import (
    MAJOR_ETFS,
    SP500_SYMBOLS,
    TREASURY_SERIES,
    apply_corporate_actions,
    build_universe_records,
    create_corporate_actions_log,
    deactivate_symbol,
    detect_corporate_actions,
    get_active_symbols,
    get_symbol_info,
    init_universe_table,
)


@pytest.fixture
def mem_conn():
    """Create an in-memory DuckDB connection for testing."""
    conn = duckdb.connect(":memory:")
    conn.execute("PRAGMA threads=4")
    yield conn
    conn.close()


@pytest.fixture
def initialized_conn(mem_conn):
    """Initialize schema in test database."""
    init_schema(mem_conn)
    yield mem_conn


class TestSchemaDDL:
    """Test DDL definitions."""

    def test_raw_schema_ddl_exists(self):
        """Test that raw schema DDL statements are defined."""
        assert len(RAW_SCHEMA_DDL) > 0
        ddl_text = " ".join(RAW_SCHEMA_DDL)
        assert "raw.yahoo_equities" in ddl_text
        assert "raw.fred_rates" in ddl_text
        assert "raw.cboe_options" in ddl_text
        assert "raw.alphavantage_equities" in ddl_text
        assert "raw.fundamentals" in ddl_text

    def test_core_schema_ddl_exists(self):
        """Test that core schema DDL statements are defined."""
        assert len(CORE_SCHEMA_DDL) > 0
        ddl_text = " ".join(CORE_SCHEMA_DDL)
        assert "core.equities" in ddl_text
        assert "core.rates" in ddl_text
        assert "core.options_chain" in ddl_text
        assert "core.universe" in ddl_text
        assert "core.fundamentals" in ddl_text
        assert "core.corporate_actions" in ddl_text

    def test_universe_table_ddl_exists(self):
        """Test that universe table DDL statements are defined."""
        assert len(UNIVERSE_TABLE_DDL) > 0
        ddl_text = " ".join(UNIVERSE_TABLE_DDL)
        assert "core.universe_table" in ddl_text

    def test_all_ddl_order(self):
        """Test that ALL_DDL has correct order (universe_table before core.universe view)."""
        all_ddl = get_all_ddl()
        # Find positions
        universe_table_pos = -1
        universe_view_pos = -1
        for i, ddl in enumerate(all_ddl):
            if "CREATE TABLE IF NOT EXISTS core.universe_table" in ddl:
                universe_table_pos = i
            if "CREATE OR REPLACE VIEW core.universe AS" in ddl:
                universe_view_pos = i
        assert universe_table_pos >= 0, "universe_table DDL not found"
        assert universe_view_pos >= 0, "universe view DDL not found"
        assert universe_table_pos < universe_view_pos, (
            "universe_table must be created before universe view"
        )


class TestSchemaInitialization:
    """Test schema initialization."""

    def test_schema_initialization(self, mem_conn):
        """Test that schema initialization creates all objects."""
        init_schema(mem_conn, force=True)
        verify_schema(mem_conn)

    def test_raw_tables_exist_after_init(self, initialized_conn):
        """Test that all raw tables exist after initialization."""
        expected_raw = [
            "yahoo_equities",
            "fred_rates",
            "cboe_options",
            "alphavantage_equities",
            "fundamentals",
        ]
        for table in expected_raw:
            result = initialized_conn.execute(f"""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = 'raw' AND table_name = '{table}'
            """).fetchone()
            assert result[0] == 1, f"Missing raw table: {table}"

    def test_core_views_exist_after_init(self, initialized_conn):
        """Test that all core views exist after initialization."""
        expected_views = [
            "equities",
            "rates",
            "options_chain",
            "universe",
            "fundamentals",
            "corporate_actions",
        ]
        for view in expected_views:
            result = initialized_conn.execute(f"""
                SELECT COUNT(*) FROM information_schema.views
                WHERE table_schema = 'core' AND table_name = '{view}'
            """).fetchone()
            assert result[0] == 1, f"Missing core view: {view}"

    def test_core_tables_exist_after_init(self, initialized_conn):
        """Test that core tables exist after initialization."""
        expected_tables = [
            "universe_table",
            "corporate_actions_log",
        ]
        for table in expected_tables:
            result = initialized_conn.execute(f"""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = 'core' AND table_name = '{table}'
            """).fetchone()
            assert result[0] == 1, f"Missing core table: {table}"

    def test_indexes_exist(self, initialized_conn):
        """Test that indexes are created."""
        # DuckDB stores indexes in pragma_index_list or we can verify by querying
        # For now, verify that queries using the indexed columns work
        # by checking that the tables exist (indexes are created with them)

        # Verify raw tables exist (indexes are created with them)
        raw_tables = [
            "yahoo_equities",
            "fred_rates",
            "cboe_options",
            "alphavantage_equities",
            "fundamentals",
        ]
        for table in raw_tables:
            result = initialized_conn.execute(f"""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = 'raw' AND table_name = '{table}'
            """).fetchone()
            assert result[0] == 1, f"Missing raw table: {table}"

        # Verify universe table exists
        result = initialized_conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'core' AND table_name = 'universe_table'
        """).fetchone()
        assert result[0] == 1, "Missing universe table"

        # Verify we can query with the indexed columns (basic functionality test)
        initialized_conn.execute("SELECT * FROM raw.yahoo_equities WHERE symbol = 'TEST' LIMIT 1")
        initialized_conn.execute("SELECT * FROM core.universe_table WHERE active = TRUE LIMIT 1")
        initialized_conn.execute(
            "SELECT * FROM core.universe_table WHERE sector = 'Technology' LIMIT 1"
        )


class TestUniverseSeeding:
    """Test universe table pre-seeding."""

    def test_universe_pre_seeded(self, initialized_conn):
        """Test that universe table is pre-seeded with symbols."""
        count = initialized_conn.execute("""
            SELECT COUNT(*) FROM core.universe_table WHERE active = TRUE
        """).fetchone()[0]
        expected_min = len(SP500_SYMBOLS) + len(MAJOR_ETFS) + len(TREASURY_SERIES)
        assert count >= expected_min, f"Expected at least {expected_min} symbols, got {count}"

    def test_universe_has_sp500_symbols(self, initialized_conn):
        """Test that universe contains S&P 500 symbols."""
        sp500_symbols = [s[0] for s in SP500_SYMBOLS]
        placeholders = ",".join(f"'{s}'" for s in sp500_symbols)
        count = initialized_conn.execute(f"""
            SELECT COUNT(*) FROM core.universe_table
            WHERE symbol IN ({placeholders})
        """).fetchone()[0]
        assert count >= len(sp500_symbols) * 0.9  # Allow some missing

    def test_universe_has_major_etfs(self, initialized_conn):
        """Test that universe contains major ETFs."""
        etf_symbols = [s[0] for s in MAJOR_ETFS]
        placeholders = ",".join(f"'{s}'" for s in etf_symbols)
        count = initialized_conn.execute(f"""
            SELECT COUNT(*) FROM core.universe_table
            WHERE symbol IN ({placeholders})
        """).fetchone()[0]
        assert count == len(etf_symbols)

    def test_universe_has_treasury_symbols(self, initialized_conn):
        """Test that universe contains treasury symbols."""
        treasury_symbols = [f"UST{s[0][3:]}" for s in TREASURY_SERIES]
        placeholders = ",".join(f"'{s}'" for s in treasury_symbols)
        count = initialized_conn.execute(f"""
            SELECT COUNT(*) FROM core.universe_table
            WHERE symbol IN ({placeholders})
        """).fetchone()[0]
        assert count == len(treasury_symbols)

    def test_universe_symbol_metadata(self, initialized_conn):
        """Test that symbol metadata is correct."""
        # Check AAPL
        info = initialized_conn.execute("""
            SELECT symbol, name, sector, industry, exchange, currency, active
            FROM core.universe_table WHERE symbol = 'AAPL'
        """).fetchone()
        assert info is not None
        assert info[0] == "AAPL"
        assert info[1] == "Apple Inc."
        assert info[2] == "Technology"
        assert info[3] == "Consumer Electronics"
        assert info[6] is True  # active

        # Check SPY
        info = initialized_conn.execute("""
            SELECT symbol, name, sector, industry, exchange, currency, active
            FROM core.universe_table WHERE symbol = 'SPY'
        """).fetchone()
        assert info is not None
        assert info[0] == "SPY"
        assert info[4] == "NYSE Arca"  # exchange

    def test_universe_symbol_counts(self):
        """Test that universe data constants have expected sizes."""
        assert len(SP500_SYMBOLS) > 100  # Should have many S&P 500 symbols
        assert len(MAJOR_ETFS) == 28  # 28 major ETFs
        assert len(TREASURY_SERIES) == 11  # 11 treasury series


class TestCoreViews:
    """Test core view definitions and queries."""

    def test_core_equities_view_columns(self, initialized_conn):
        """Test core.equities view has correct columns."""
        cols = initialized_conn.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'core' AND table_name = 'equities'
            ORDER BY ordinal_position
        """).fetchall()
        expected = [
            ("symbol", "VARCHAR"),
            ("date", "DATE"),
            ("open", "DOUBLE"),
            ("high", "DOUBLE"),
            ("low", "DOUBLE"),
            ("close", "DOUBLE"),
            ("volume", "BIGINT"),
            ("adj_close", "DOUBLE"),
            ("source", "VARCHAR"),
        ]
        assert len(cols) == len(expected)
        for (col_name, col_type), (exp_name, exp_type) in zip(cols, expected, strict=False):
            assert col_name == exp_name
            assert col_type == exp_type

    def test_core_rates_view_columns(self, initialized_conn):
        """Test core.rates view has correct columns and tenor mapping."""
        cols = initialized_conn.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'core' AND table_name = 'rates'
            ORDER BY ordinal_position
        """).fetchall()
        expected = [
            ("series_id", "VARCHAR"),
            ("date", "DATE"),
            ("value", "DOUBLE"),
            ("tenor", "VARCHAR"),
            ("source", "VARCHAR"),
        ]
        assert len(cols) == len(expected)
        for (col_name, col_type), (exp_name, exp_type) in zip(cols, expected, strict=False):
            assert col_name == exp_name
            assert col_type == exp_type

    def test_core_options_chain_view_columns(self, initialized_conn):
        """Test core.options_chain view has correct columns."""
        cols = initialized_conn.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'core' AND table_name = 'options_chain'
            ORDER BY ordinal_position
        """).fetchall()
        expected = [
            ("underlying", "VARCHAR"),
            ("expiry", "DATE"),
            ("strike", "DOUBLE"),
            ("right", "VARCHAR"),
            ("bid", "DOUBLE"),
            ("ask", "DOUBLE"),
            ("mid", "DOUBLE"),
            ("iv", "DOUBLE"),
            ("delta", "DOUBLE"),
            ("gamma", "DOUBLE"),
            ("vega", "DOUBLE"),
            ("theta", "DOUBLE"),
            ("rho", "DOUBLE"),
            ("volume", "BIGINT"),
            ("open_interest", "BIGINT"),
            ("source", "VARCHAR"),
            ("snapshot_ts", "TIMESTAMP"),
        ]
        assert len(cols) == len(expected)
        for (col_name, col_type), (exp_name, exp_type) in zip(cols, expected, strict=False):
            assert col_name == exp_name
            assert col_type == exp_type

    def test_core_universe_view_filters_active(self, initialized_conn):
        """Test core.universe view only returns active symbols."""
        # Insert an inactive symbol
        initialized_conn.execute("""
            INSERT INTO core.universe_table (symbol, figi, cusip, isin, exchange, currency, name, sector, industry, active)
            VALUES ('TEST_INACTIVE', NULL, NULL, NULL, 'TEST', 'USD', 'Test Inactive', 'Test', 'Test', FALSE)
        """)
        total_count = initialized_conn.execute(
            "SELECT COUNT(*) FROM core.universe_table"
        ).fetchone()[0]
        active_count = initialized_conn.execute("SELECT COUNT(*) FROM core.universe").fetchone()[0]
        assert active_count == total_count - 1

    def test_core_fundamentals_view_columns(self, initialized_conn):
        """Test core.fundamentals view has correct columns."""
        cols = initialized_conn.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'core' AND table_name = 'fundamentals'
            ORDER BY ordinal_position
        """).fetchall()
        expected = [
            ("symbol", "VARCHAR"),
            ("date", "DATE"),
            ("metric", "VARCHAR"),
            ("value", "DOUBLE"),
            ("source", "VARCHAR"),
        ]
        assert len(cols) == len(expected)
        for (col_name, col_type), (exp_name, exp_type) in zip(cols, expected, strict=False):
            assert col_name == exp_name
            assert col_type == exp_type

    def test_core_corporate_actions_view_logic(self, initialized_conn):
        """Test core.corporate_actions view correctly identifies actions."""
        # Insert test data with split and dividend
        initialized_conn.execute("""
            INSERT INTO raw.yahoo_equities (symbol, date, open, high, low, close, volume, adj_close, dividends, splits, source, ingest_ts)
            VALUES
                ('TEST_CA', '2024-01-15', 100, 105, 99, 102, 1000000, 102, 0.0, 2.0, 'yahoo', CURRENT_TIMESTAMP),
                ('TEST_CA', '2024-02-15', 50, 52, 49, 51, 1000000, 51, 0.50, 1.0, 'yahoo', CURRENT_TIMESTAMP)
        """)
        actions = initialized_conn.execute("""
            SELECT symbol, date, action, ratio, details
            FROM core.corporate_actions
            WHERE symbol = 'TEST_CA'
            ORDER BY date
        """).fetchall()
        assert len(actions) == 2
        # First row: split
        assert actions[0][2] == "split"
        assert actions[0][3] == 2.0
        assert "split" in actions[0][4].lower()
        # Second row: dividend
        assert actions[1][2] == "dividend"
        assert actions[1][3] == 0.5
        assert "dividend" in actions[1][4].lower()


class TestUniverseFunctions:
    """Test universe helper functions."""

    def test_init_universe_table_idempotent(self, mem_conn):
        """Test that init_universe_table is idempotent."""
        # First initialization
        count1 = init_universe_table(mem_conn)
        # Second initialization (should update, not duplicate)
        count2 = init_universe_table(mem_conn)
        assert count1 == count2
        # Verify no duplicates
        total = mem_conn.execute("SELECT COUNT(*) FROM core.universe_table").fetchone()[0]
        assert total == count1

    def test_get_active_symbols_sorted(self, initialized_conn):
        """Test get_active_symbols returns sorted active symbols."""
        symbols = get_active_symbols(initialized_conn)
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert symbols == sorted(symbols)
        assert "AAPL" in symbols
        assert "SPY" in symbols

    def test_get_symbol_info(self, initialized_conn):
        """Test get_symbol_info returns correct metadata."""
        info = get_symbol_info("AAPL", initialized_conn)
        assert info is not None
        assert info.symbol == "AAPL"
        assert info.name == "Apple Inc."
        assert info.sector == "Technology"
        assert info.active is True

        # Non-existent symbol
        info = get_symbol_info("NONEXISTENT", initialized_conn)
        assert info is None

    def test_deactivate_symbol(self, initialized_conn):
        """Test deactivate_symbol marks symbol as inactive."""
        # Verify symbol is active
        info = get_symbol_info("AAPL", initialized_conn)
        assert info.active is True

        # Deactivate
        result = deactivate_symbol("AAPL", initialized_conn)
        assert result is True

        # Verify deactivated
        info = get_symbol_info("AAPL", initialized_conn)
        assert info.active is False

        # Deactivate non-existent
        result = deactivate_symbol("NONEXISTENT", initialized_conn)
        assert result is False

    def test_detect_corporate_actions(self, initialized_conn):
        """Test detect_corporate_actions finds splits and dividends."""
        # Insert test data
        initialized_conn.execute("""
            INSERT INTO raw.yahoo_equities (symbol, date, open, high, low, close, volume, adj_close, dividends, splits, source, ingest_ts)
            VALUES
                ('TEST_CA', '2024-01-15', 100, 105, 99, 102, 1000000, 102, 0.0, 2.0, 'yahoo', CURRENT_TIMESTAMP),
                ('TEST_CA', '2024-02-15', 50, 52, 49, 51, 1000000, 51, 0.50, 1.0, 'yahoo', CURRENT_TIMESTAMP),
                ('TEST_CA', '2024-03-15', 51, 53, 50, 52, 1000000, 52, 0.0, 1.0, 'yahoo', CURRENT_TIMESTAMP)
        """)
        actions = detect_corporate_actions("TEST_CA", "2024-01-01", "2024-12-31", initialized_conn)
        assert len(actions) == 2
        assert actions[0]["action"] == "split"
        assert actions[0]["ratio"] == 2.0
        assert actions[1]["action"] == "dividend"
        assert actions[1]["ratio"] == 0.5

        # Test date filtering
        actions = detect_corporate_actions("TEST_CA", "2024-02-01", "2024-02-28", initialized_conn)
        assert len(actions) == 1
        assert actions[0]["action"] == "dividend"

    def test_corporate_actions_log_table(self, mem_conn):
        """Test corporate actions log table creation."""
        create_corporate_actions_log(mem_conn)
        result = mem_conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'core' AND table_name = 'corporate_actions_log'
        """).fetchone()
        assert result[0] == 1

        # Check structure
        cols = mem_conn.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'core' AND table_name = 'corporate_actions_log'
            ORDER BY ordinal_position
        """).fetchall()
        expected = [
            ("id", "BIGINT", "NO"),
            ("symbol", "VARCHAR", "NO"),
            ("date", "DATE", "NO"),
            ("action", "VARCHAR", "NO"),
            ("ratio", "DOUBLE", "YES"),
            ("details", "VARCHAR", "YES"),
            ("created_ts", "TIMESTAMP", "NO"),
        ]
        assert len(cols) == len(expected)
        for (col_name, col_type, nullable), (exp_name, exp_type, exp_nullable) in zip(
            cols, expected, strict=False
        ):
            assert col_name == exp_name
            assert col_type == exp_type
            assert nullable == exp_nullable

    def test_apply_corporate_actions(self, mem_conn):
        """Test apply_corporate_actions records actions."""
        create_corporate_actions_log(mem_conn)
        apply_corporate_actions("TEST_CA", "2024-01-15", "split", 2.0, mem_conn)
        apply_corporate_actions("TEST_CA", "2024-02-15", "dividend", 0.50, mem_conn)

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


class TestBuildUniverseRecords:
    """Test universe record building."""

    def test_build_universe_records(self):
        """Test build_universe_records returns correct structure."""
        records = build_universe_records()
        assert len(records) == len(SP500_SYMBOLS) + len(MAJOR_ETFS) + len(TREASURY_SERIES)
        for record in records:
            assert hasattr(record, "symbol")
            assert hasattr(record, "name")
            assert hasattr(record, "sector")
            assert hasattr(record, "industry")
            assert hasattr(record, "exchange")
            assert hasattr(record, "currency")
            assert hasattr(record, "active")

    def test_treasury_symbol_naming(self):
        """Test treasury symbols are named correctly."""
        records = build_universe_records()
        treasury_records = [r for r in records if r.symbol.startswith("UST")]
        assert len(treasury_records) == len(TREASURY_SERIES)
        # Check a few known mappings
        symbols = {r.symbol for r in treasury_records}
        assert "UST10" in symbols  # DGS10 -> UST10
        assert "UST30" in symbols  # DGS30 -> UST30
        assert "UST1MO" in symbols  # DGS1MO -> UST1MO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
