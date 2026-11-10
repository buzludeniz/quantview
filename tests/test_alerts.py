"""Tests for threshold alerting: evaluation, de-duplication, persistence, delivery."""

import json
import socketserver
import threading
from datetime import date, timedelta

import duckdb
import pandas as pd
import pytest

from quantview.config import AlertThresholdsConfig, EmailSMTPConfig, Settings
from quantview.data import alerts as al
from quantview.data.ingest import init_schema, insert_idempotent, run_intraday_ingest


class _Handler(socketserver.StreamRequestHandler):
    """Just enough SMTP to accept one or more messages (no STARTTLS, no AUTH)."""

    def _send(self, line: str) -> None:
        self.wfile.write(line.encode() + b"\r\n")

    def handle(self) -> None:
        server: FakeSMTP = self.server  # type: ignore[assignment]
        self._send("220 fake ESMTP")
        data_mode = False
        lines: list[str] = []
        mail_from, rcpts = "", []
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            line = raw.decode().rstrip("\r\n")
            if data_mode:
                if line == ".":
                    data_mode = False
                    server.messages.append(
                        {"from": mail_from, "to": list(rcpts), "data": "\n".join(lines)}
                    )
                    self._send("250 queued")
                else:
                    lines.append(line)
                continue
            cmd = line.upper()
            if cmd.startswith("EHLO"):
                self._send("250 fake")
            elif cmd.startswith("MAIL FROM:"):
                mail_from, rcpts = line[10:].strip("<> "), []
                self._send("250 ok")
            elif cmd.startswith("RCPT TO:"):
                rcpts.append(line[8:].strip("<> "))
                self._send("250 ok")
            elif cmd == "DATA":
                data_mode, lines = True, []
                self._send("354 go")
            elif cmd == "QUIT":
                self._send("221 bye")
                return
            else:
                self._send("250 ok")


class FakeSMTP(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.messages: list[dict] = []


@pytest.fixture
def smtp_server():
    server = FakeSMTP()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    init_schema(c, force=True)
    yield c
    c.close()


def make_settings(port: int = 0, configured: bool = True, **thr) -> Settings:
    smtp = EmailSMTPConfig(
        host="127.0.0.1",
        port=port,
        from_address="qv@example.test" if configured else "",
        to_addresses=["a@example.test", "b@example.test"] if configured else [],
        use_tls=False,
    )
    thresholds = AlertThresholdsConfig(
        **{"price_change_pct": 2.0, "volume_spike_ratio": 3.0, **thr}
    )
    return Settings(email_smtp=smtp, alert_thresholds=thresholds)


def add_bars(conn, symbol: str, closes: list[float], volumes: list[int], start=date(2024, 1, 1)):
    n = len(closes)
    df = pd.DataFrame(
        {
            "symbol": symbol,
            "date": [start + timedelta(days=i) for i in range(n)],
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes,
            "adj_close": closes,
            "dividends": 0.0,
            "splits": 1.0,
        }
    )
    insert_idempotent(conn, "raw.yahoo_equities", df, ["symbol", "date"])


def flat(n=10, price=100.0, vol=1000):
    return [price] * n, [vol] * n


class TestEvaluation:
    def test_price_move_above_threshold_fires(self, conn):
        c, v = flat()
        add_bars(conn, "AAPL", [*c, 103.0], [*v, 1000])
        found = al.evaluate_thresholds(conn, make_settings())
        assert [(a.symbol, a.alert_type) for a in found] == [("AAPL", "price_change")]
        assert found[0].metadata["value"] == pytest.approx(3.0)
        assert found[0].severity == "warning"

    def test_downward_move_and_critical_severity(self, conn):
        c, v = flat()
        add_bars(conn, "AAPL", [*c, 95.0], [*v, 1000])
        (a,) = al.evaluate_thresholds(conn, make_settings())
        assert a.metadata["value"] == pytest.approx(-5.0)
        assert a.severity == "critical"

    def test_move_below_threshold_does_not_fire(self, conn):
        c, v = flat()
        add_bars(conn, "AAPL", [*c, 101.0], [*v, 1000])
        assert al.evaluate_thresholds(conn, make_settings()) == []

    def test_volume_spike_fires(self, conn):
        c, v = flat()
        add_bars(conn, "MSFT", [*c, 100.0], [*v, 4000])
        (a,) = al.evaluate_thresholds(conn, make_settings())
        assert a.alert_type == "volume_spike"
        assert a.metadata["value"] == pytest.approx(4.0)

    def test_volume_needs_enough_history(self, conn):
        add_bars(conn, "NEW", [100.0, 100.0, 100.0], [1000, 1000, 9000])
        assert al.evaluate_thresholds(conn, make_settings()) == []

    def test_disabled_thresholds_are_not_evaluated(self, conn):
        c, v = flat()
        add_bars(conn, "AAPL", [*c, 150.0], [*v, 9000])
        assert al.evaluate_thresholds(conn, make_settings(price_change_pct=0)) != []
        only = al.evaluate_thresholds(conn, make_settings(price_change_pct=0))
        assert {a.alert_type for a in only} == {"volume_spike"}
        assert (
            al.evaluate_thresholds(conn, make_settings(price_change_pct=0, volume_spike_ratio=-1))
            == []
        )

    def test_sustained_breach_fires_once_and_rearms_after_reset(self, conn):
        c, v = flat()
        # Day 11 breaches, day 12 still breaches (no new alert), 13 resets, 14 breaches again.
        series = [*c, 103.0, 106.0, 106.0, 110.0]
        add_bars(conn, "AAPL", series[:11], [*v, 1000])
        assert len(al.evaluate_thresholds(conn, make_settings())) == 1
        add_bars(conn, "AAPL", series[:12], [*v, 1000, 1000])
        assert al.evaluate_thresholds(conn, make_settings()) == []
        add_bars(conn, "AAPL", series[:13], v + [1000] * 3)
        assert al.evaluate_thresholds(conn, make_settings()) == []
        add_bars(conn, "AAPL", series, v + [1000] * 4)
        assert len(al.evaluate_thresholds(conn, make_settings())) == 1

    def test_zero_prior_close_is_ignored(self):
        assert al._price_change_pct([0.0, 5.0], 1) is None
        assert al._volume_ratio([0.0] * 6 + [5.0], 6) is None


class TestPersistence:
    def test_store_writes_rows_and_is_idempotent_per_bar(self, conn):
        c, v = flat()
        add_bars(conn, "AAPL", [*c, 103.0], [*v, 5000])
        cands = al.evaluate_thresholds(conn, make_settings())
        assert len(cands) == 2
        stored = al.store_alerts(conn, cands)
        assert [s.id for s in stored] == [1, 2]
        rows = conn.execute(
            "SELECT symbol, alert_type, severity, acknowledged, "
            "json_extract_string(metadata,'$.bar_date') FROM alerts ORDER BY id"
        ).fetchall()
        assert rows == [
            ("AAPL", "price_change", "warning", False, "2024-01-11"),
            ("AAPL", "volume_spike", "warning", False, "2024-01-11"),
        ]
        # Evaluating and storing the same data again adds nothing.
        assert al.store_alerts(conn, al.evaluate_thresholds(conn, make_settings())) == []
        assert conn.execute("SELECT count(*) FROM alerts").fetchone()[0] == 2

    def test_alerts_table_is_created_on_demand(self):
        c = duckdb.connect(":memory:")
        al.create_alerts_table(c)
        al.create_alerts_table(c)  # idempotent
        assert c.execute("SELECT count(*) FROM alerts").fetchone()[0] == 0


class TestDelivery:
    def _fire(self, conn):
        c, v = flat()
        add_bars(conn, "AAPL", [*c, 103.0], [*v, 5000])
        return al.store_alerts(conn, al.evaluate_thresholds(conn, make_settings()))

    def _delivery(self, conn):
        return [
            json.loads(m)["delivery"]
            for (m,) in conn.execute("SELECT metadata FROM alerts ORDER BY id").fetchall()
        ]

    def test_one_email_per_batch_over_smtp(self, conn, smtp_server):
        stored = self._fire(conn)
        settings = make_settings(smtp_server.server_address[1])
        assert al.deliver_alerts(conn, stored, settings) == "sent"
        (msg,) = smtp_server.messages
        assert msg["from"] == "qv@example.test"
        assert msg["to"] == ["a@example.test", "b@example.test"]
        assert "Subject: [QuantView] 2 alerts" in msg["data"]
        assert "AAPL moved +3.00%" in msg["data"]
        assert "volume was 5.0x" in msg["data"]
        assert [d["status"] for d in self._delivery(conn)] == ["sent", "sent"]

    def test_connection_failure_is_recorded_not_raised(self, conn):
        stored = self._fire(conn)
        # Port 1 on loopback refuses connections.
        assert al.deliver_alerts(conn, stored, make_settings(1)) == "failed"
        deliveries = self._delivery(conn)
        assert all(d["status"] == "failed" and d["error"] for d in deliveries)

    def test_starttls_unsupported_by_server_is_recorded(self, conn, smtp_server):
        stored = self._fire(conn)
        settings = make_settings(smtp_server.server_address[1])
        settings.email_smtp.use_tls = True
        assert al.deliver_alerts(conn, stored, settings) == "failed"
        assert smtp_server.messages == []
        assert "SMTP" in self._delivery(conn)[0]["error"]

    def test_unconfigured_smtp_is_skipped(self, conn):
        stored = self._fire(conn)
        assert al.deliver_alerts(conn, stored, make_settings(25, configured=False)) == "skipped"
        assert self._delivery(conn)[0]["status"] == "skipped"

    def test_nothing_to_deliver(self, conn):
        assert al.deliver_alerts(conn, [], make_settings()) == "skipped"

    def test_login_failure_is_recorded(self, conn, smtp_server):
        stored = self._fire(conn)
        settings = make_settings(smtp_server.server_address[1])
        settings.email_smtp.username = "user"  # fake server has no AUTH
        assert al.deliver_alerts(conn, stored, settings) == "failed"

    def test_status_update_failure_is_logged_not_raised(self, smtp_server):
        class Broken:
            def execute(self, *a, **k):
                raise RuntimeError("db gone")

        a = al.AlertCandidate("X", "price_change", "warning", "m", {"bar_date": "d"}, id=1)
        settings = make_settings(smtp_server.server_address[1])
        assert al.deliver_alerts(Broken(), [a], settings) == "sent"  # type: ignore[arg-type]


class TestProcessAndIngest:
    def test_process_alerts_end_to_end_then_quiet(self, conn, smtp_server):
        c, v = flat()
        add_bars(conn, "AAPL", [*c, 103.0], [*v, 1000])
        settings = make_settings(smtp_server.server_address[1])
        first = al.process_alerts(conn, settings)
        assert first == {"fired": 1, "delivery": "sent", "error": None}
        second = al.process_alerts(conn, settings)
        assert second["fired"] == 0
        assert len(smtp_server.messages) == 1

    def test_process_alerts_swallows_errors(self):
        broken = duckdb.connect(":memory:")  # no raw.yahoo_equities table
        out = al.process_alerts(broken, make_settings())
        assert out["fired"] == 0 and out["error"]

    def test_ingest_reports_alert_summary(self, conn, monkeypatch):
        seen = []
        monkeypatch.setattr(
            "quantview.data.ingest.process_alerts", lambda c: seen.append(c) or {"fired": 0}
        )
        monkeypatch.setattr("quantview.data.ingest.get_active_symbols", lambda c: [])
        monkeypatch.setattr("quantview.data.ingest.fetch_and_store_cboe_options", lambda *a: 0)
        stats = run_intraday_ingest(conn)
        assert seen == [conn]
        assert stats["alerts"] == {"fired": 0}
