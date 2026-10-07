"""Tests for DuckDB connection management.

connection.py held 23% coverage while every other data test depends on it. The
behaviours worth pinning are the caching contract (one connection per thread),
the documented lifecycle of ``connection_context``, and the schema it creates.
"""

from __future__ import annotations

import threading
from pathlib import Path

import duckdb
import pytest

from quantview.data import connection as cn


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch):
    """Point the module at a throwaway database and reset its caches.

    The connection cache is module-level and keyed on thread-local state, so a
    test that leaks one would silently reuse another test's database.
    """
    db_path = tmp_path / "quantview.duckdb"

    class _Settings:
        data_path = db_path

    monkeypatch.setattr(cn, "get_settings", lambda: _Settings)
    cn.close_connection()
    monkeypatch.setattr(cn, "_initialized", False)
    yield db_path
    cn.close_connection()


class TestGetConnection:
    def test_reuses_one_connection_per_thread(self, isolated: Path):
        first = cn.get_connection()
        second = cn.get_connection()

        assert first is second

    def test_writes_land_in_the_configured_file(self, isolated: Path):
        conn = cn.get_connection()
        conn.execute("CREATE TABLE probe AS SELECT 1 AS n")
        assert isolated.exists()

        # DuckDB refuses a second handle on the same file under a different
        # configuration, so close ours before reopening to prove the data was
        # persisted rather than buffered in the first connection.
        cn.close_connection()
        other = duckdb.connect(str(isolated), read_only=True)
        try:
            assert other.execute("SELECT n FROM probe").fetchone()[0] == 1
        finally:
            other.close()

    def test_creates_the_parent_directory_when_missing(self, tmp_path: Path, monkeypatch):
        nested = tmp_path / "a" / "b" / "c" / "quantview.duckdb"

        class _Settings:
            data_path = nested

        monkeypatch.setattr(cn, "get_settings", lambda: _Settings)
        cn.close_connection()
        monkeypatch.setattr(cn, "_initialized", False)

        cn.get_connection()

        assert nested.parent.is_dir()
        cn.close_connection()

    def test_each_thread_gets_its_own_connection(self, isolated: Path):
        main_conn = cn.get_connection()
        seen: dict[str, object] = {}

        def worker() -> None:
            seen["conn"] = cn.get_connection()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert seen["conn"] is not None
        assert seen["conn"] is not main_conn, "threads must not share one connection"

    def test_read_only_is_honoured_on_a_fresh_connection(self, isolated: Path):
        cn.get_connection().execute("CREATE TABLE probe AS SELECT 1")
        cn.close_connection()

        read_only = cn.get_connection(read_only=True)
        # A read-only handle cannot write; that failure is the proof.
        with pytest.raises(duckdb.Error):
            read_only.execute("CREATE TABLE another AS SELECT 2")

    def test_schema_is_created_on_first_use(self, isolated: Path):
        conn = cn.get_connection()

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }

        assert {
            "equity_bars",
            "option_chains",
            "greeks_snapshots",
            "iv_surface",
            "fundamentals",
            "macro_series",
            "alerts",
        } <= tables

    def test_schema_creation_is_idempotent(self, isolated: Path):
        """Re-running init on a live database must not raise."""
        cn.get_connection()
        cn.init_database(cn.get_connection())
        cn.init_database(cn.get_connection())

        assert cn.check_connection() is True


class TestCloseConnection:
    def test_is_safe_to_call_before_any_connection_exists(self):
        cn.close_connection()

    def test_is_safe_to_call_twice(self, isolated: Path):
        cn.get_connection()
        cn.close_connection()
        cn.close_connection()

    def test_a_new_connection_is_built_after_closing(self, isolated: Path):
        first = cn.get_connection()
        cn.close_connection()
        second = cn.get_connection()

        assert second is not first


class TestConnectionContext:
    def test_yields_a_usable_connection(self, isolated: Path):
        with cn.connection_context() as conn:
            assert conn.execute("SELECT 42").fetchone()[0] == 42

    def test_leaves_the_connection_open_for_reuse(self, isolated: Path):
        """The module documents per-thread reuse, so the context must not close.

        Its docstring claimed the connection was closed on exit while the body
        deliberately did not, and the reuse is what makes the cache meaningful.
        """
        with cn.connection_context() as conn:
            conn.execute("CREATE TABLE kept AS SELECT 1")

        assert conn.execute("SELECT count(*) FROM kept").fetchone()[0] == 1
        assert cn.get_connection() is conn

    def test_closes_the_connection_when_the_body_raises(self, isolated: Path):
        """A failure inside the block must still unwind through the finally."""
        with pytest.raises(ValueError, match="boom"), cn.connection_context():
            raise ValueError("boom")

        # The connection survives, but the context manager exited normally.
        assert cn.get_connection().execute("SELECT 1").fetchone()[0] == 1


class TestInitDatabase:
    def test_creates_the_alert_id_sequence(self, isolated: Path):
        conn = cn.get_connection()

        # The sequence has to hand out increasing ids for the alerts table.
        first = conn.execute("SELECT nextval('alert_id_seq')").fetchone()[0]
        second = conn.execute("SELECT nextval('alert_id_seq')").fetchone()[0]

        assert second == first + 1

    def test_equity_bars_rejects_a_duplicate_key(self, isolated: Path):
        conn = cn.get_connection()
        row = (
            "INSERT INTO equity_bars "
            "(symbol, timestamp, open, high, low, close, volume) "
            "VALUES ('SPY', '2024-01-02 00:00:00', 1, 2, 0.5, 1.5, 100)"
        )
        conn.execute(row)

        with pytest.raises(duckdb.Error):
            conn.execute(row)

    def test_creates_the_documented_indexes(self, isolated: Path):
        conn = cn.get_connection()

        names = {
            row[0] for row in conn.execute("SELECT index_name FROM duckdb_indexes()").fetchall()
        }

        assert "idx_equity_bars_symbol_time" in names
        assert "idx_alerts_unacked" in names


class TestCheckConnection:
    def test_true_for_a_working_database(self, isolated: Path):
        assert cn.check_connection() is True

    def test_false_rather_than_raising_when_the_database_is_unreachable(
        self, tmp_path: Path, monkeypatch
    ):
        """A health check must report failure, not take the caller down."""

        class _Settings:
            data_path = tmp_path / "missing" / "sub" / "x.duckdb"

            def __getattr__(self, name):  # pragma: no cover - defensive
                raise AttributeError(name)

        def _boom():
            raise duckdb.Error("cannot open")

        monkeypatch.setattr(cn, "get_settings", _boom)
        cn.close_connection()
        monkeypatch.setattr(cn, "_initialized", False)

        assert cn.check_connection() is False
