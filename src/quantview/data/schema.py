"""QuantView Database Schema Definitions.

This module contains all DDL statements for creating raw tables, canonical views,
and the symbol universe table in DuckDB. The schema follows a raw/core pattern
where raw tables store ingested data as-is, and core views provide cleaned,
standardized access patterns.
"""


# =============================================================================
# RAW SCHEMA - Ingested data as received from sources
# =============================================================================

RAW_SCHEMA_DDL: list[str] = [
    # Raw Yahoo Finance equities data
    """
    CREATE SCHEMA IF NOT EXISTS raw
    """,
    """
    CREATE TABLE IF NOT EXISTS raw.yahoo_equities (
        symbol VARCHAR NOT NULL,
        date DATE NOT NULL,
        open DOUBLE NOT NULL,
        high DOUBLE NOT NULL,
        low DOUBLE NOT NULL,
        close DOUBLE NOT NULL,
        volume BIGINT NOT NULL,
        adj_close DOUBLE NOT NULL,
        dividends DOUBLE DEFAULT 0.0,
        splits DOUBLE DEFAULT 1.0,
        source VARCHAR NOT NULL DEFAULT 'yahoo',
        ingest_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, date)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_raw_yahoo_equities_symbol_date
    ON raw.yahoo_equities(symbol, date DESC)
    """,
    # Raw FRED rates data
    """
    CREATE TABLE IF NOT EXISTS raw.fred_rates (
        series_id VARCHAR NOT NULL,
        date DATE NOT NULL,
        value DOUBLE NOT NULL,
        source VARCHAR NOT NULL DEFAULT 'fred',
        ingest_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (series_id, date)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_raw_fred_rates_series_date
    ON raw.fred_rates(series_id, date DESC)
    """,
    # Raw CBOE options chain data
    """
    CREATE TABLE IF NOT EXISTS raw.cboe_options (
        underlying VARCHAR NOT NULL,
        expiry DATE NOT NULL,
        strike DOUBLE NOT NULL,
        option_right VARCHAR NOT NULL,  -- 'C' or 'P'
        bid DOUBLE,
        ask DOUBLE,
        mid DOUBLE,
        iv DOUBLE,
        delta DOUBLE,
        gamma DOUBLE,
        vega DOUBLE,
        theta DOUBLE,
        rho DOUBLE,
        volume BIGINT,
        open_interest BIGINT,
        source VARCHAR NOT NULL DEFAULT 'cboe',
        snapshot_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (underlying, expiry, strike, option_right, snapshot_ts)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_raw_cboe_options_underlying_expiry
    ON raw.cboe_options(underlying, expiry, strike, option_right)
    """,
    # Raw Alpha Vantage equities data
    """
    CREATE TABLE IF NOT EXISTS raw.alphavantage_equities (
        symbol VARCHAR NOT NULL,
        date DATE NOT NULL,
        open DOUBLE NOT NULL,
        high DOUBLE NOT NULL,
        low DOUBLE NOT NULL,
        close DOUBLE NOT NULL,
        volume BIGINT NOT NULL,
        source VARCHAR NOT NULL DEFAULT 'alphavantage',
        ingest_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, date)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_raw_alphavantage_equities_symbol_date
    ON raw.alphavantage_equities(symbol, date DESC)
    """,
    # Raw fundamentals data
    """
    CREATE TABLE IF NOT EXISTS raw.fundamentals (
        symbol VARCHAR NOT NULL,
        date DATE NOT NULL,
        metric VARCHAR NOT NULL,
        value DOUBLE NOT NULL,
        period VARCHAR,  -- 'annual', 'quarterly', 'ttm'
        source VARCHAR NOT NULL,
        ingest_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, date, metric, period)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_raw_fundamentals_symbol_date
    ON raw.fundamentals(symbol, date DESC)
    """,
]

# =============================================================================
# CORE SCHEMA - Canonical views for application consumption
# =============================================================================

CORE_SCHEMA_DDL: list[str] = [
    # Core schema
    """
    CREATE SCHEMA IF NOT EXISTS core
    """,
    # Canonical equities view - merges yahoo and alphavantage, prefers yahoo
    """
    CREATE OR REPLACE VIEW core.equities AS
    SELECT
        symbol,
        date,
        open,
        high,
        low,
        close,
        volume,
        adj_close,
        'yahoo' AS source
    FROM raw.yahoo_equities
    UNION ALL
    SELECT
        symbol,
        date,
        open,
        high,
        low,
        close,
        volume,
        close AS adj_close,  -- Alpha Vantage doesn't provide adj_close
        'alphavantage' AS source
    FROM raw.alphavantage_equities a
    WHERE NOT EXISTS (
        SELECT 1 FROM raw.yahoo_equities y
        WHERE y.symbol = a.symbol AND y.date = a.date
    )
    """,
    # Canonical rates view with tenor mapping
    """
    CREATE OR REPLACE VIEW core.rates AS
    SELECT
        series_id,
        date,
        value,
        CASE
            WHEN series_id IN ('DGS1MO', 'DGS3MO', 'DGS6MO') THEN 'short'
            WHEN series_id IN ('DGS1', 'DGS2', 'DGS3', 'DGS5') THEN 'medium'
            WHEN series_id IN ('DGS7', 'DGS10', 'DGS20', 'DGS30') THEN 'long'
            ELSE 'unknown'
        END AS tenor,
        'fred' AS source
    FROM raw.fred_rates
    """,
    # Canonical options chain view
    """
    CREATE OR REPLACE VIEW core.options_chain AS
    SELECT
        underlying,
        expiry,
        strike,
        option_right AS right,
        bid,
        ask,
        mid,
        iv,
        delta,
        gamma,
        vega,
        theta,
        rho,
        volume,
        open_interest,
        source,
        snapshot_ts
    FROM raw.cboe_options
    """,
    # Canonical universe view (references core.universe table)
    """
    CREATE OR REPLACE VIEW core.universe AS
    SELECT
        symbol,
        figi,
        cusip,
        isin,
        exchange,
        currency,
        name,
        sector,
        industry,
        active
    FROM core.universe_table
    WHERE active = TRUE
    """,
    # Canonical fundamentals view
    """
    CREATE OR REPLACE VIEW core.fundamentals AS
    SELECT
        symbol,
        date,
        metric,
        value,
        source
    FROM raw.fundamentals
    """,
    # Canonical corporate actions view
    """
    CREATE OR REPLACE VIEW core.corporate_actions AS
    SELECT
        symbol,
        date,
        CASE
            WHEN splits != 1.0 THEN 'split'
            WHEN dividends > 0 THEN 'dividend'
            ELSE 'other'
        END AS action,
        CASE
            WHEN splits != 1.0 THEN splits
            WHEN dividends > 0 THEN dividends
            ELSE NULL
        END AS ratio,
        CASE
            WHEN splits != 1.0 THEN 'Stock split: ' || splits::VARCHAR || ':1'
            WHEN dividends > 0 THEN 'Dividend: $' || dividends::VARCHAR
            ELSE 'Corporate action'
        END AS details
    FROM raw.yahoo_equities
    WHERE splits != 1.0 OR dividends > 0
    """,
]

# =============================================================================
# UNIVERSE TABLE - Symbol reference data
# =============================================================================

UNIVERSE_TABLE_DDL: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS core.universe_table (
        symbol VARCHAR NOT NULL PRIMARY KEY,
        figi VARCHAR,
        cusip VARCHAR,
        isin VARCHAR,
        exchange VARCHAR NOT NULL,
        currency VARCHAR NOT NULL DEFAULT 'USD',
        name VARCHAR NOT NULL,
        sector VARCHAR,
        industry VARCHAR,
        active BOOLEAN NOT NULL DEFAULT TRUE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_universe_active ON core.universe_table(active)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_universe_sector ON core.universe_table(sector)
    """,
]

# =============================================================================
# ALL DDL COMBINED
# =============================================================================

# Order matters:
# 1. RAW_SCHEMA_DDL - raw tables and indexes
# 2. CREATE SCHEMA IF NOT EXISTS core (first statement of CORE_SCHEMA_DDL)
# 3. UNIVERSE_TABLE_DDL - core.universe_table (referenced by core.universe view)
# 4. Remaining CORE_SCHEMA_DDL - views (including core.universe which references universe_table)
ALL_DDL: list[str] = RAW_SCHEMA_DDL + CORE_SCHEMA_DDL[:1] + UNIVERSE_TABLE_DDL + CORE_SCHEMA_DDL[1:]


def get_all_ddl() -> list[str]:
    """Get all DDL statements in execution order.

    Returns:
        List of DDL statements to execute in sequence.
    """
    return ALL_DDL


def get_raw_ddl() -> list[str]:
    """Get only raw schema DDL statements."""
    return RAW_SCHEMA_DDL


def get_core_ddl() -> list[str]:
    """Get only core schema DDL statements."""
    return CORE_SCHEMA_DDL


def get_universe_ddl() -> list[str]:
    """Get only universe table DDL statements."""
    return UNIVERSE_TABLE_DDL
