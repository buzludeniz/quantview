"""DuckDB Connection Management.

Provides thread-safe connection handling for the QuantView DuckDB database.
"""

import atexit
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from quantview.config import get_settings

# Thread-local storage for connections
_local = threading.local()
_connection_lock = threading.Lock()
_initialized = False


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection for the current thread.

    Each thread gets its own connection. Connections are lazily created
    and reused within the same thread.

    Args:
        read_only: Whether to open the database in read-only mode.

    Returns:
        duckdb.DuckDBPyConnection: A DuckDB connection object.
    """
    global _initialized

    # The cache is keyed on read_only as well as thread. A single cached handle
    # per thread meant a read_only=True request arriving after a read-write one
    # got the writable connection back, so the caller's intent was silently
    # dropped and code that expected a read-only view could write.
    cache_key = "connection_ro" if read_only else "connection"

    if getattr(_local, cache_key, None) is None:
        with _connection_lock:
            # Another thread may have built it while this one waited.
            if getattr(_local, cache_key, None) is None:
                settings = get_settings()
                db_path = str(settings.data_path)

                # Ensure parent directory exists
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

                if read_only and not Path(db_path).exists():
                    raise FileNotFoundError(
                        f"cannot open {db_path} read-only: the database does not "
                        "exist yet. Create it first, or request a writable "
                        "connection."
                    )

                new_conn = duckdb.connect(db_path, read_only=read_only)
                new_conn.execute("PRAGMA threads=4")

                # A read-only handle cannot create the schema, so initialisation
                # only applies to a writable connection.
                if not read_only and not _initialized:
                    init_database(new_conn)
                    _initialized = True

                setattr(_local, cache_key, new_conn)

    conn: duckdb.DuckDBPyConnection = getattr(_local, cache_key)
    return conn


def close_connection() -> None:
    """Close the current thread's database connections if any exist."""
    for key in ("connection", "connection_ro"):
        conn = getattr(_local, key, None)
        if conn is not None:
            conn.close()
            setattr(_local, key, None)


# Register atexit handler to ensure connections are closed on process exit
atexit.register(close_connection)


@contextmanager
def connection_context(read_only: bool = False) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Context manager for database connections.

    Ensures the connection is properly closed after use.

    Args:
        read_only: Whether to open the database in read-only mode.

    Yields:
        duckdb.DuckDBPyConnection: A DuckDB connection object.
    """
    conn = get_connection(read_only=read_only)
    try:
        yield conn
    finally:
        # Don't close here - connections are reused per thread
        pass


def init_database(conn: duckdb.DuckDBPyConnection | None = None) -> None:
    """Initialize the database schema.

    Creates all required tables if they don't exist.

    Args:
        conn: Optional connection to use. If not provided, gets a new connection.
    """
    if conn is None:
        conn = get_connection()

    # Core market data tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS equity_bars (
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open DOUBLE NOT NULL,
            high DOUBLE NOT NULL,
            low DOUBLE NOT NULL,
            close DOUBLE NOT NULL,
            volume BIGINT NOT NULL,
            vwap DOUBLE,
            trade_count INTEGER,
            PRIMARY KEY (symbol, timestamp)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS option_chains (
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            expiry DATE NOT NULL,
            strike DOUBLE NOT NULL,
            option_type VARCHAR NOT NULL,  -- 'C' or 'P'
            bid DOUBLE,
            ask DOUBLE,
            last DOUBLE,
            mark DOUBLE,
            bid_size INTEGER,
            ask_size INTEGER,
            volume INTEGER,
            open_interest INTEGER,
            implied_volatility DOUBLE,
            delta DOUBLE,
            gamma DOUBLE,
            theta DOUBLE,
            vega DOUBLE,
            rho DOUBLE,
            underlying_price DOUBLE,
            PRIMARY KEY (symbol, timestamp, expiry, strike, option_type)
        )
    """)

    # Greeks and IV surfaces
    conn.execute("""
        CREATE TABLE IF NOT EXISTS greeks_snapshots (
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            expiry DATE NOT NULL,
            strike DOUBLE NOT NULL,
            option_type VARCHAR NOT NULL,
            delta DOUBLE,
            gamma DOUBLE,
            theta DOUBLE,
            vega DOUBLE,
            rho DOUBLE,
            iv DOUBLE,
            PRIMARY KEY (symbol, timestamp, expiry, strike, option_type)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS iv_surface (
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            expiry DATE NOT NULL,
            strike DOUBLE NOT NULL,
            iv DOUBLE NOT NULL,
            moneyness DOUBLE NOT NULL,  -- strike / spot
            dte INTEGER NOT NULL,  -- days to expiry
            PRIMARY KEY (symbol, timestamp, expiry, strike)
        )
    """)

    # Fundamental / macro data
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals (
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            metric VARCHAR NOT NULL,
            value DOUBLE NOT NULL,
            period VARCHAR,  -- 'annual', 'quarterly', 'ttm'
            PRIMARY KEY (symbol, timestamp, metric, period)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_series (
            series_id VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            value DOUBLE NOT NULL,
            PRIMARY KEY (series_id, timestamp)
        )
    """)

    # Alerts and signals
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id BIGINT PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            symbol VARCHAR NOT NULL,
            alert_type VARCHAR NOT NULL,
            message VARCHAR NOT NULL,
            severity VARCHAR NOT NULL,  -- 'info', 'warning', 'critical'
            acknowledged BOOLEAN DEFAULT FALSE,
            metadata JSON
        )
    """)

    # Create sequence for alert IDs
    conn.execute("CREATE SEQUENCE IF NOT EXISTS alert_id_seq START 1")

    # Indexes for common query patterns
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_equity_bars_symbol_time ON equity_bars(symbol, timestamp DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_option_chains_symbol_time ON option_chains(symbol, timestamp DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_greeks_symbol_time ON greeks_snapshots(symbol, timestamp DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_iv_surface_symbol_time ON iv_surface(symbol, timestamp DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_macro_series_time ON macro_series(series_id, timestamp DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_alerts_symbol_time ON alerts(symbol, timestamp DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_alerts_unacked ON alerts(acknowledged, timestamp DESC)"
    )


def check_connection() -> bool:
    """Verify database connection works.

    Uses a writable handle, matching what the rest of the application asks for.
    A read-only probe reported False for a database that simply had not been
    created yet, which reads as "broken" when it means "not built".

    Returns:
        bool: True if connection is successful, False otherwise.
    """
    try:
        conn = get_connection()
        conn.execute("SELECT 1").fetchone()
        return True
    except Exception:
        return False
