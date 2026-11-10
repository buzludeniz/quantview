"""Threshold alerting: evaluate after ingest, persist, and deliver by email.

Thresholds come from ``Settings.alert_thresholds``; a threshold of zero or less
is disabled. Two conditions are evaluated against ``raw.yahoo_equities``:

* ``price_change``: absolute close-to-close change, in percent, at the latest
  bar versus the bar before it.
* ``volume_spike``: latest volume divided by the mean volume of the preceding
  ``VOLUME_WINDOW`` bars (at least ``MIN_VOLUME_HISTORY`` of them).

``iv_rank_change`` is configurable but not evaluated: no IV-rank series is
stored by the ingest pipeline yet.

An alert fires when the condition holds at the latest bar and did not hold at
the bar before it, so a sustained breach alerts once and re-arms after the
condition clears. Re-running an ingest over the same data is also safe: an alert
already recorded for the same symbol, type and bar date is not repeated.

Alerts are emailed over SMTP (``Settings.email_smtp``). Delivery never raises
into the ingest; the outcome (``sent``, ``failed`` or ``skipped``) is stored in
the alert's ``metadata.delivery``.
"""

from __future__ import annotations

import json
import logging
import smtplib
import ssl
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.message import EmailMessage
from typing import Any

import duckdb
import pandas as pd

from quantview.config import EmailSMTPConfig, Settings, get_settings
from quantview.data.connection import create_alerts_table

logger = logging.getLogger(__name__)

VOLUME_WINDOW = 20
MIN_VOLUME_HISTORY = 5
SMTP_TIMEOUT_SECONDS = 10.0
# A reading at least this many times the threshold is "critical", otherwise "warning".
CRITICAL_MULTIPLE = 2.0


@dataclass
class AlertCandidate:
    """An alert produced by evaluation, before it has been stored."""

    symbol: str
    alert_type: str
    severity: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: int | None = None


def _severity(value: float, threshold: float) -> str:
    return "critical" if value >= CRITICAL_MULTIPLE * threshold else "warning"


def _price_change_pct(close: list[float], i: int) -> float | None:
    if i < 1 or close[i - 1] == 0:
        return None
    return (close[i] / close[i - 1] - 1.0) * 100.0


def _volume_ratio(volume: list[float], i: int) -> float | None:
    history = volume[max(0, i - VOLUME_WINDOW) : i]
    if len(history) < MIN_VOLUME_HISTORY:
        return None
    mean = sum(history) / len(history)
    return volume[i] / mean if mean > 0 else None


def evaluate_thresholds(
    conn: duckdb.DuckDBPyConnection, settings: Settings | None = None
) -> list[AlertCandidate]:
    """Return alerts whose condition newly holds at each symbol's latest bar."""
    thresholds = (settings or get_settings()).alert_thresholds
    price_thr = thresholds.price_change_pct
    volume_thr = thresholds.volume_spike_ratio
    if price_thr <= 0 and volume_thr <= 0:
        return []

    bars = conn.execute(
        f"""
        SELECT symbol, date, close, volume FROM (
            SELECT symbol, date, close, volume,
                   row_number() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
            FROM raw.yahoo_equities
        ) WHERE rn <= {VOLUME_WINDOW + 2}
        ORDER BY symbol, date
        """
    ).fetchdf()

    candidates: list[AlertCandidate] = []
    for symbol, group in bars.groupby("symbol", sort=True):
        close = [float(v) for v in group["close"]]
        volume = [float(v) for v in group["volume"]]
        last = len(close) - 1
        bar_date = pd.Timestamp(group["date"].iloc[-1]).strftime("%Y-%m-%d")

        if price_thr > 0:
            now, before = _price_change_pct(close, last), _price_change_pct(close, last - 1)
            if (
                now is not None
                and abs(now) >= price_thr
                and (before is None or abs(before) < price_thr)
            ):
                candidates.append(
                    AlertCandidate(
                        str(symbol),
                        "price_change",
                        _severity(abs(now), price_thr),
                        f"{symbol} moved {now:+.2f}% on {bar_date} (threshold {price_thr:g}%)",
                        {"bar_date": bar_date, "value": now, "threshold": price_thr},
                    )
                )
        if volume_thr > 0:
            now, before = _volume_ratio(volume, last), _volume_ratio(volume, last - 1)
            if now is not None and now >= volume_thr and (before is None or before < volume_thr):
                candidates.append(
                    AlertCandidate(
                        str(symbol),
                        "volume_spike",
                        _severity(now, volume_thr),
                        f"{symbol} volume was {now:.1f}x its recent average on {bar_date} "
                        f"(threshold {volume_thr:g}x)",
                        {"bar_date": bar_date, "value": now, "threshold": volume_thr},
                    )
                )
    return candidates


def store_alerts(
    conn: duckdb.DuckDBPyConnection, candidates: list[AlertCandidate]
) -> list[AlertCandidate]:
    """Write candidates to ``alerts``, skipping any already recorded for the same bar.

    Returns the candidates that were newly stored, with their ``id`` set.
    """
    create_alerts_table(conn)
    stored: list[AlertCandidate] = []
    for c in candidates:
        exists = conn.execute(
            """
            SELECT count(*) FROM alerts
            WHERE symbol = ? AND alert_type = ?
              AND json_extract_string(metadata, '$.bar_date') = ?
            """,
            [c.symbol, c.alert_type, c.metadata.get("bar_date")],
        ).fetchone()
        if exists and exists[0]:
            continue
        row = conn.execute(
            """
            INSERT INTO alerts (id, timestamp, symbol, alert_type, message, severity, metadata)
            VALUES (nextval('alert_id_seq'), ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            [
                datetime.now(UTC).replace(tzinfo=None),
                c.symbol,
                c.alert_type,
                c.message,
                c.severity,
                json.dumps(c.metadata),
            ],
        ).fetchone()
        c.id = int(row[0]) if row else None
        stored.append(c)
    return stored


def _smtp_ready(cfg: EmailSMTPConfig) -> bool:
    return bool(cfg.host and cfg.from_address and cfg.to_addresses)


def send_email(cfg: EmailSMTPConfig, subject: str, body: str) -> None:
    """Send one message through the configured SMTP server. Raises on failure."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.from_address
    msg["To"] = ", ".join(cfg.to_addresses)
    msg.set_content(body)
    with smtplib.SMTP(cfg.host, cfg.port, timeout=SMTP_TIMEOUT_SECONDS) as smtp:
        smtp.ehlo()
        if cfg.use_tls:
            smtp.starttls(context=ssl.create_default_context())
            smtp.ehlo()
        if cfg.username:
            smtp.login(cfg.username, cfg.password)
        smtp.send_message(msg)


def deliver_alerts(
    conn: duckdb.DuckDBPyConnection,
    alerts: list[AlertCandidate],
    settings: Settings | None = None,
) -> str:
    """Email ``alerts`` in one message and record the outcome on each alert.

    Returns ``"sent"``, ``"failed"`` or ``"skipped"`` (nothing to send, or SMTP
    sender/recipients not configured). Never raises.
    """
    if not alerts:
        return "skipped"
    cfg = (settings or get_settings()).email_smtp
    outcome: dict[str, Any]
    if not _smtp_ready(cfg):
        outcome = {"status": "skipped", "reason": "SMTP sender or recipients not configured"}
    else:
        subject = f"[QuantView] {len(alerts)} alert{'s' if len(alerts) != 1 else ''}"
        body = "\n".join(f"[{a.severity}] {a.message}" for a in alerts)
        try:
            send_email(cfg, subject, body)
            outcome = {"status": "sent"}
        except Exception as e:  # smtplib.SMTPException, OSError, ssl.SSLError, ...
            logger.error(f"Alert email delivery failed: {e}")
            outcome = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
    outcome["at"] = datetime.now(UTC).isoformat()
    for a in alerts:
        try:
            conn.execute(
                "UPDATE alerts SET metadata = ? WHERE id = ?",
                [json.dumps({**a.metadata, "delivery": outcome}), a.id],
            )
        except Exception as e:
            logger.error(f"Could not record delivery status for alert {a.id}: {e}")
    return str(outcome["status"])


def process_alerts(
    conn: duckdb.DuckDBPyConnection, settings: Settings | None = None
) -> dict[str, Any]:
    """Evaluate, store and deliver alerts after an ingest. Never raises."""
    summary: dict[str, Any] = {"fired": 0, "delivery": "skipped", "error": None}
    try:
        stored = store_alerts(conn, evaluate_thresholds(conn, settings))
        summary["fired"] = len(stored)
        summary["delivery"] = deliver_alerts(conn, stored, settings)
    except Exception as e:
        logger.error(f"Alert processing failed: {e}")
        summary["error"] = str(e)
    return summary
