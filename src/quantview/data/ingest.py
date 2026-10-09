"""QuantView Data Ingestion and Schema Initialization.

This module provides the main entry point for initializing the database schema
and running data ingestion pipelines. Can be invoked via CLI:
    python -m quantview.data.ingest init_schema
    python -m quantview.data.ingest daily
    python -m quantview.data.ingest intraday
    python -m quantview.data.ingest scheduler
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from types import FrameType
from typing import Any

import duckdb
import pandas as pd
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from quantview.config import get_settings
from quantview.data.connection import get_connection
from quantview.data.connectors import (
    AlphaVantageConnector,
    CboeConnector,
    FredConnector,
    YahooConnector,
)
from quantview.data.schema import get_all_ddl
from quantview.data.universe import (
    create_corporate_actions_log,
    detect_corporate_actions,
    get_active_symbols,
    init_universe_table,
)

logger = logging.getLogger(__name__)

# Eastern Time zone
ET_TZ = pytz.timezone("US/Eastern")


def _count(conn: duckdb.DuckDBPyConnection, query: str) -> int:
    """Run a single-row COUNT query and return the integer it produced.



    ``fetchone`` is typed as possibly returning None, which every call site here

    would otherwise have to re-check. These queries are aggregate-only, so a row

    is guaranteed; the check exists to convert an impossible None into a legible

    error instead of a TypeError.

    """

    row = conn.execute(query).fetchone()

    if row is None:
        raise RuntimeError(f"count query returned no row: {query.strip()[:60]}")

    return int(row[0])


def init_schema(conn: duckdb.DuckDBPyConnection | None = None, force: bool = False) -> None:
    """Initialize the complete database schema.

    Creates all raw tables, core views, universe table, and pre-seeds
    the universe with S&P 500 symbols, major ETFs, and treasury series.

    Args:
        conn: Optional database connection. If None, uses default connection.
        force: If True, drops and recreates all objects (use with caution).
    """
    if conn is None:
        conn = get_connection()

    settings = get_settings()
    print(f"Initializing QuantView schema at: {settings.data_path}")

    if force:
        print("Dropping existing schemas (force mode)...")
        conn.execute("DROP SCHEMA IF EXISTS raw CASCADE")
        conn.execute("DROP SCHEMA IF EXISTS core CASCADE")

    # Execute all DDL statements
    ddl_statements = get_all_ddl()
    print(f"Executing {len(ddl_statements)} DDL statements...")

    for i, ddl in enumerate(ddl_statements, 1):
        try:
            conn.execute(ddl)
            if i % 10 == 0 or i == len(ddl_statements):
                print(f"  Progress: {i}/{len(ddl_statements)}")
        except Exception as e:
            print(f"  ERROR at statement {i}: {e}")
            print(f"  DDL: {ddl[:200]}...")
            raise

    # Initialize universe table with pre-seeded data
    print("Pre-seeding universe table...")
    count = init_universe_table(conn)
    print(f"  Inserted/updated {count} universe records")

    # Create corporate actions log table
    print("Creating corporate actions log table...")
    create_corporate_actions_log(conn)

    # Verify schema
    verify_schema(conn)

    print("Schema initialization complete!")


def verify_schema(conn: duckdb.DuckDBPyConnection | None = None) -> bool:
    """Verify that all expected tables and views exist.

    Args:
        conn: Optional database connection.

    Returns:
        True if verification passes, raises exception otherwise.
    """
    if conn is None:
        conn = get_connection()  # Use regular connection for verification

    # Expected raw tables
    expected_raw_tables = [
        "yahoo_equities",
        "fred_rates",
        "cboe_options",
        "alphavantage_equities",
        "fundamentals",
    ]

    # Expected core views
    expected_core_views = [
        "equities",
        "rates",
        "options_chain",
        "universe",
        "fundamentals",
        "corporate_actions",
    ]

    # Expected core tables
    expected_core_tables = [
        "universe_table",
        "corporate_actions_log",
    ]

    print("\nVerifying schema...")

    # Check raw tables
    for table in expected_raw_tables:
        n = _count(
            conn,
            f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'raw' AND table_name = '{table}'
            """,
        )
        if n == 0:
            raise RuntimeError(f"Missing raw table: raw.{table}")
        print(f"  [OK] raw.{table}")

    # Check core views
    for view in expected_core_views:
        n = _count(
            conn,
            f"""
            SELECT COUNT(*) FROM information_schema.views
            WHERE table_schema = 'core' AND table_name = '{view}'
            """,
        )
        if n == 0:
            raise RuntimeError(f"Missing core view: core.{view}")
        print(f"  [OK] core.{view} (view)")

    # Check core tables
    for table in expected_core_tables:
        n = _count(
            conn,
            f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'core' AND table_name = '{table}'
            """,
        )
        if n == 0:
            raise RuntimeError(f"Missing core table: core.{table}")
        print(f"  [OK] core.{table}")

    # Check universe has data
    universe_count = _count(conn, "SELECT COUNT(*) FROM core.universe_table WHERE active = TRUE")
    if universe_count == 0:
        raise RuntimeError("Universe table is empty - no active symbols")
    print(f"  [OK] core.universe_table has {universe_count} active symbols")

    # Sample queries to verify views work
    print("\nRunning sample queries...")

    # Test equities view
    try:
        conn.execute("SELECT * FROM core.equities LIMIT 1")
        print("  [OK] core.equities queryable")
    except Exception as e:
        print(f"  [WARN] core.equities query failed: {e}")

    # Test rates view
    try:
        conn.execute("SELECT * FROM core.rates LIMIT 1")
        print("  [OK] core.rates queryable")
    except Exception as e:
        print(f"  [WARN] core.rates query failed: {e}")

    # Test options_chain view
    try:
        conn.execute("SELECT * FROM core.options_chain LIMIT 1")
        print("  [OK] core.options_chain queryable")
    except Exception as e:
        print(f"  [WARN] core.options_chain query failed: {e}")

    # Test universe view
    try:
        n = _count(conn, "SELECT COUNT(*) FROM core.universe")
        print(f"  [OK] core.universe returns {n} active symbols")
    except Exception as e:
        print(f"  [WARN] core.universe query failed: {e}")

    # Test fundamentals view
    try:
        conn.execute("SELECT * FROM core.fundamentals LIMIT 1")
        print("  [OK] core.fundamentals queryable")
    except Exception as e:
        print(f"  [WARN] core.fundamentals query failed: {e}")

    # Test corporate_actions view
    try:
        conn.execute("SELECT * FROM core.corporate_actions LIMIT 1")
        print("  [OK] core.corporate_actions queryable")
    except Exception as e:
        print(f"  [WARN] core.corporate_actions query failed: {e}")

    print("\nSchema verification complete!")
    return True


def drop_schema(conn: duckdb.DuckDBPyConnection | None = None) -> None:
    """Drop all QuantView schemas (use with caution).

    Args:
        conn: Optional database connection.
    """
    if conn is None:
        conn = get_connection()

    print("Dropping QuantView schemas...")
    conn.execute("DROP SCHEMA IF EXISTS raw CASCADE")
    conn.execute("DROP SCHEMA IF EXISTS core CASCADE")
    print("Schemas dropped.")


def refresh_materialized_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Refresh all materialized views (core views in DuckDB are regular views,
    but we can refresh any tables that act as materialized views).

    Args:
        conn: Database connection.
    """
    # In DuckDB, views are always up-to-date. However, if we had materialized
    # tables, we would refresh them here. For now, this is a no-op but kept
    # for future extensibility.
    logger.info("Refreshing core views (DuckDB views are always current)")


def insert_idempotent(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
    conflict_columns: list[str],
) -> int:
    """Insert DataFrame into table idempotently using ON CONFLICT.

    Args:
        conn: Database connection.
        table: Target table name.
        df: DataFrame to insert.
        conflict_columns: Columns that define uniqueness.

    Returns:
        Number of rows inserted/updated.
    """
    if df.empty:
        return 0

    # Register DataFrame as a temporary view
    temp_view = f"temp_{table.replace('.', '_')}_{int(time.time() * 1000)}"
    conn.register(temp_view, df)

    # Build column list
    columns = list(df.columns)
    col_str = ", ".join(columns)

    # Build ON CONFLICT clause
    conflict_cols = ", ".join(conflict_columns)
    update_cols = [c for c in columns if c not in conflict_columns]
    if update_cols:
        update_str = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])
        conflict_clause = f"ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_str}"
    else:
        conflict_clause = f"ON CONFLICT ({conflict_cols}) DO NOTHING"

    query = f"""
        INSERT INTO {table} ({col_str})
        SELECT {col_str} FROM {temp_view}
        {conflict_clause}
    """

    try:
        conn.execute(query)
        # DuckDB doesn't return rowcount directly for INSERT ... SELECT, so the
        # row count is approximated from the frame that was just inserted.
        return len(df)
    except Exception as e:
        logger.error(f"Failed to insert into {table}: {e}")
        raise
    finally:
        conn.unregister(temp_view)


def fetch_and_store_yahoo_equities(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
    start: str,
    end: str,
) -> int:
    """Fetch Yahoo Finance equities data and store in raw.yahoo_equities.

    Args:
        conn: Database connection.
        symbols: List of symbols to fetch.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        Number of rows inserted/updated.
    """
    logger.info(f"Fetching Yahoo equities for {len(symbols)} symbols from {start} to {end}")
    start_time = time.time()

    with YahooConnector() as connector:
        df = connector.fetch(symbols, start, end)

    if df.empty:
        logger.warning("No Yahoo equities data returned")
        return 0

    rows = insert_idempotent(
        conn,
        "raw.yahoo_equities",
        df,
        ["symbol", "date"],
    )

    duration = time.time() - start_time
    logger.info(f"Inserted/updated {rows} rows in raw.yahoo_equities in {duration:.2f}s")
    return rows


def fetch_and_store_fred_rates(
    conn: duckdb.DuckDBPyConnection,
    series_ids: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> int:
    """Fetch FRED rates data and store in raw.fred_rates.

    Args:
        conn: Database connection.
        series_ids: List of FRED series IDs. If None, uses defaults.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        Number of rows inserted/updated.
    """
    logger.info(f"Fetching FRED rates for {len(series_ids) if series_ids else 'default'} series")
    start_time = time.time()

    with FredConnector() as connector:
        df = connector.fetch(series_ids or [], start or "", end or "")

    if df.empty:
        logger.warning("No FRED rates data returned")
        return 0

    rows = insert_idempotent(
        conn,
        "raw.fred_rates",
        df,
        ["series_id", "date"],
    )

    duration = time.time() - start_time
    logger.info(f"Inserted/updated {rows} rows in raw.fred_rates in {duration:.2f}s")
    return rows


def fetch_and_store_cboe_options(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
) -> int:
    """Fetch CBOE options chains and store in raw.cboe_options.

    Args:
        conn: Database connection.
        symbols: List of underlying symbols.

    Returns:
        Number of rows inserted/updated.
    """
    logger.info(f"Fetching CBOE options for {len(symbols)} symbols")
    start_time = time.time()

    with CboeConnector() as connector:
        df = connector.fetch(symbols, "", "")

    if df.empty:
        logger.warning("No CBOE options data returned")
        return 0

    # Ensure snapshot_ts is present
    if "snapshot_ts" not in df.columns:
        df["snapshot_ts"] = pd.Timestamp.now()

    rows = insert_idempotent(
        conn,
        "raw.cboe_options",
        df,
        ["underlying", "expiry", "strike", "option_right", "snapshot_ts"],
    )

    duration = time.time() - start_time
    logger.info(f"Inserted/updated {rows} rows in raw.cboe_options in {duration:.2f}s")
    return rows


def fetch_and_store_alphavantage_equities(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
    start: str,
    end: str,
) -> int:
    """Fetch Alpha Vantage equities data and store in raw.alphavantage_equities.

    Args:
        conn: Database connection.
        symbols: List of symbols to fetch.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        Number of rows inserted/updated.
    """
    logger.info(f"Fetching Alpha Vantage equities for {len(symbols)} symbols from {start} to {end}")
    start_time = time.time()

    with AlphaVantageConnector() as connector:
        df = connector.fetch(symbols, start, end)

    if df.empty:
        logger.warning("No Alpha Vantage equities data returned")
        return 0

    rows = insert_idempotent(
        conn,
        "raw.alphavantage_equities",
        df,
        ["symbol", "date"],
    )

    duration = time.time() - start_time
    logger.info(f"Inserted/updated {rows} rows in raw.alphavantage_equities in {duration:.2f}s")
    return rows


def fetch_and_store_fundamentals(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
) -> int:
    """Fetch Alpha Vantage fundamentals and store in raw.fundamentals.

    Args:
        conn: Database connection.
        symbols: List of symbols to fetch.

    Returns:
        Number of rows inserted/updated.
    """
    logger.info(f"Fetching fundamentals for {len(symbols)} symbols")
    start_time = time.time()

    with AlphaVantageConnector() as connector:
        df = connector.fetch_fundamentals(symbols)

    if df.empty:
        logger.warning("No fundamentals data returned")
        return 0

    # Add source column
    df["source"] = "alphavantage"

    rows = insert_idempotent(
        conn,
        "raw.fundamentals",
        df,
        ["symbol", "date", "metric", "period"],
    )

    duration = time.time() - start_time
    logger.info(f"Inserted/updated {rows} rows in raw.fundamentals in {duration:.2f}s")
    return rows


def process_corporate_actions(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
    start: str,
    end: str,
) -> int:
    """Detect and log corporate actions for symbols.

    Args:
        conn: Database connection.
        symbols: List of symbols to check.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        Number of corporate actions recorded.
    """
    logger.info(f"Processing corporate actions for {len(symbols)} symbols")
    total_actions = 0

    for symbol in symbols:
        try:
            actions = detect_corporate_actions(symbol, start, end, conn)
            for action in actions:
                conn.execute(
                    """
                    INSERT INTO core.corporate_actions_log (symbol, date, action, ratio, details, created_ts)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    [symbol, action["date"], action["action"], action["ratio"], action["details"]],
                )
                total_actions += 1
                logger.info(
                    f"Corporate action: {symbol} {action['date']} {action['action']} {action['ratio']}"
                )
        except Exception as e:
            logger.error(f"Failed to process corporate actions for {symbol}: {e}")

    return total_actions


def run_daily_ingest(
    conn: duckdb.DuckDBPyConnection | None = None,
    date: str | None = None,
) -> dict:
    """Run daily EOD ingestion for all universe symbols.

    Fetches EOD data from all connectors for the given date (defaults to yesterday).

    Args:
        conn: Optional database connection.
        date: Target date in YYYY-MM-DD format. Defaults to yesterday (ET).

    Returns:
        Dict with ingestion statistics.
    """
    if conn is None:
        conn = get_connection()

    # Determine target date (yesterday in ET)
    if date is None:
        et_now = datetime.now(ET_TZ)
        target_date = (et_now - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        target_date = date

    logger.info(f"Starting daily EOD ingestion for {target_date}")
    start_time = time.time()

    # Get active symbols from universe
    symbols = get_active_symbols(conn)
    equity_symbols = [
        s for s in symbols if not s.startswith("UST")
    ]  # Exclude synthetic treasury symbols
    fred_series = [s for s in symbols if s.startswith("UST")]

    # Map UST* symbols back to FRED series IDs
    fred_series_ids = [f"DGS{s[3:]}" for s in fred_series]

    stats: dict[str, Any] = {
        "date": target_date,
        "symbols_processed": len(equity_symbols),
        "fred_series_processed": len(fred_series_ids),
        "yahoo_rows": 0,
        "fred_rows": 0,
        "cboe_rows": 0,
        "alphavantage_rows": 0,
        "fundamentals_rows": 0,
        "corporate_actions": 0,
        "errors": [],
        "duration_seconds": 0,
    }

    try:
        # Fetch Yahoo equities (EOD)
        stats["yahoo_rows"] = fetch_and_store_yahoo_equities(
            conn, equity_symbols, target_date, target_date
        )

        # Fetch FRED rates
        stats["fred_rows"] = fetch_and_store_fred_rates(
            conn, fred_series_ids, target_date, target_date
        )

        # Fetch CBOE options (snapshot)
        stats["cboe_rows"] = fetch_and_store_cboe_options(
            conn, equity_symbols[:50]
        )  # Limit for API

        # Fetch Alpha Vantage equities (as backup)
        stats["alphavantage_rows"] = fetch_and_store_alphavantage_equities(
            conn, equity_symbols, target_date, target_date
        )

        # Fetch fundamentals (weekly-ish, but run daily for any new)
        stats["fundamentals_rows"] = fetch_and_store_fundamentals(
            conn, equity_symbols[:100]
        )  # Limit for API

        # Process corporate actions
        stats["corporate_actions"] = process_corporate_actions(
            conn, equity_symbols, target_date, target_date
        )

        # Refresh materialized views
        refresh_materialized_views(conn)

    except Exception as e:
        error_msg = f"Daily ingest failed: {e}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        raise

    stats["duration_seconds"] = time.time() - start_time
    logger.info(f"Daily EOD ingestion completed in {stats['duration_seconds']:.2f}s: {stats}")
    return stats


def run_intraday_ingest(
    conn: duckdb.DuckDBPyConnection | None = None,
) -> dict:
    """Run intraday 5-minute snapshot ingestion.

    Fetches current 5-minute bars for all active equity symbols during market hours.

    Args:
        conn: Optional database connection.

    Returns:
        Dict with ingestion statistics.
    """
    if conn is None:
        conn = get_connection()

    et_now = datetime.now(ET_TZ)
    snapshot_time = et_now.strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"Starting intraday 5-min snapshot at {snapshot_time} ET")
    start_time = time.time()

    # Get active equity symbols
    symbols = get_active_symbols(conn)
    equity_symbols = [s for s in symbols if not s.startswith("UST")]

    stats: dict[str, Any] = {
        "snapshot_time": snapshot_time,
        "symbols_processed": len(equity_symbols),
        "yahoo_rows": 0,
        "cboe_rows": 0,
        "errors": [],
        "duration_seconds": 0,
    }

    try:
        # For intraday, we fetch the latest data (yfinance supports intraday)
        # Note: yfinance intraday requires period="1d", interval="5m"
        # We'll use a custom fetch for intraday

        # Fetch intraday Yahoo data (5-min bars for today)
        logger.info(f"Fetching intraday 5-min bars for {len(equity_symbols)} symbols")
        with YahooConnector() as connector:
            # Use period="1d", interval="5m" for intraday
            intraday_data: list[pd.DataFrame] = []
            for symbol in equity_symbols:
                try:
                    ticker = connector._get_ticker(symbol)
                    hist = ticker.history(period="1d", interval="5m", actions=True)
                    if not hist.empty:
                        hist = hist.reset_index()
                        # Handle both "Datetime" (intraday) and "Date" (daily) column names
                        date_col = "Datetime" if "Datetime" in hist.columns else "Date"
                        hist = hist.rename(
                            columns={
                                date_col: "date",
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
                        hist["symbol"] = symbol
                        # Ensure all required columns
                        for col in ["dividends", "splits"]:
                            if col not in hist.columns:
                                hist[col] = 0.0 if col == "dividends" else 1.0
                        intraday_data.append(hist[YahooConnector.REQUIRED_COLUMNS])
                except Exception as e:
                    logger.warning(f"Failed to fetch intraday for {symbol}: {e}")
                    stats["errors"].append(f"Intraday fetch failed for {symbol}: {e}")

        if intraday_data:
            df = pd.concat(intraday_data, ignore_index=True)
            # Convert datetime to date for primary key (we store date only in raw table)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date
                # Deduplicate by (symbol, date) keeping latest
                df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
                stats["yahoo_rows"] = insert_idempotent(
                    conn, "raw.yahoo_equities", df, ["symbol", "date"]
                )
            else:
                logger.warning("No 'date' column in intraday data after concat")
                stats["yahoo_rows"] = 0

        # Fetch CBOE options snapshot
        stats["cboe_rows"] = fetch_and_store_cboe_options(conn, equity_symbols[:50])

        # Refresh materialized views
        refresh_materialized_views(conn)

    except Exception as e:
        error_msg = f"Intraday ingest failed: {e}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        raise

    stats["duration_seconds"] = time.time() - start_time
    logger.info(f"Intraday ingestion completed in {stats['duration_seconds']:.2f}s: {stats}")
    return stats


def run_scheduler() -> None:
    """Run the APScheduler blocking scheduler with daily and intraday jobs."""
    logger.info("Starting QuantView ingestion scheduler")

    scheduler = BlockingScheduler(timezone=ET_TZ)

    # Daily EOD job at 18:00 ET
    scheduler.add_job(
        run_daily_ingest,
        trigger=CronTrigger(hour=18, minute=0, timezone=ET_TZ),
        id="daily_eod",
        name="Daily EOD Ingestion",
        max_instances=1,
        misfire_grace_time=3600,  # 1 hour grace time
    )

    # Intraday 5-min snapshots during market hours (09:30-16:00 ET, Mon-Fri)
    scheduler.add_job(
        run_intraday_ingest,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute="*/5",
            timezone=ET_TZ,
            start_date=datetime.now(ET_TZ).replace(hour=9, minute=30, second=0, microsecond=0),
            end_date=datetime.now(ET_TZ).replace(hour=16, minute=0, second=0, microsecond=0),
        ),
        id="intraday_5min",
        name="Intraday 5-min Snapshots",
        max_instances=1,
        misfire_grace_time=300,  # 5 min grace time
    )

    # Also run at market open (09:30) and close (16:00) for completeness
    scheduler.add_job(
        run_intraday_ingest,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=9,
            minute=30,
            timezone=ET_TZ,
        ),
        id="intraday_open",
        name="Intraday Market Open Snapshot",
        max_instances=1,
    )

    scheduler.add_job(
        run_intraday_ingest,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=16,
            minute=0,
            timezone=ET_TZ,
        ),
        id="intraday_close",
        name="Intraday Market Close Snapshot",
        max_instances=1,
    )

    logger.info("Scheduler configured with jobs:")
    for job in scheduler.get_jobs():
        logger.info(f"  - {job.name}: {job.trigger}")

    def _shutdown(signum: int, frame: FrameType | None) -> None:
        logger.info(f"Received signal {signum}, shutting down scheduler...")
        scheduler.shutdown(wait=True)
        sys.exit(0)

    # Handle both SIGTERM (container stop) and SIGINT (Ctrl+C)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        scheduler.shutdown()


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(description="QuantView Data Ingestion and Schema Management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init_schema command
    init_parser = subparsers.add_parser("init_schema", help="Initialize database schema")
    init_parser.add_argument(
        "--force", action="store_true", help="Drop and recreate all schemas (DESTRUCTIVE)"
    )

    # verify command
    subparsers.add_parser("verify", help="Verify schema integrity")

    # drop command
    drop_parser = subparsers.add_parser("drop", help="Drop all schemas (DESTRUCTIVE)")
    drop_parser.add_argument("--yes", action="store_true", help="Confirm destructive operation")

    # daily command
    daily_parser = subparsers.add_parser("daily", help="Run daily EOD ingestion")
    daily_parser.add_argument(
        "--date", type=str, help="Target date (YYYY-MM-DD), defaults to yesterday ET"
    )

    # intraday command
    subparsers.add_parser("intraday", help="Run intraday 5-min snapshot ingestion")

    # scheduler command
    subparsers.add_parser("scheduler", help="Run APScheduler blocking scheduler")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Configure logging. The directory has to exist before FileHandler opens the
    # file; Settings no longer creates it as a constructor side effect.
    settings = get_settings()
    settings.ensure_directories()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(settings.log_file),
        ],
    )

    try:
        if args.command == "init_schema":
            init_schema(force=args.force)
        elif args.command == "verify":
            verify_schema()
        elif args.command == "drop":
            if not args.yes:
                print("ERROR: Dropping schemas is destructive. Use --yes to confirm.")
                return 1
            drop_schema()
        elif args.command == "daily":
            stats = run_daily_ingest(date=args.date)
            print(f"Daily ingestion complete: {stats}")
        elif args.command == "intraday":
            stats = run_intraday_ingest()
            print(f"Intraday ingestion complete: {stats}")
        elif args.command == "scheduler":
            run_scheduler()
        else:
            parser.print_help()
            return 1
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        print(f"ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
