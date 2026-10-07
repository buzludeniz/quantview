"""
Tests for the QuantView Desktop package.

Every test runs against Qt's offscreen platform plugin, so no display server is
needed:

Run with: pytest tests/test_desktop.py -v
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from quantview.desktop.charts import (
    CandlestickChart,
    MovingAverageOverlay,
    VolumeChart,
    as_epoch_ms,
    row_count,
)
from quantview.desktop.main import build_window, demo_rows
from quantview.desktop.monitor import DEFAULT_INTERVAL_MS, MIN_INTERVAL_MS, Monitor

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """
    One QApplication for the whole module.

    Qt permits only a single QApplication per process, so this is session
    scoped and every test reuses the same instance.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """Small deterministic OHLCV frame."""
    rng = np.random.default_rng(11)
    n = 60
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))
    open_ = np.concatenate(([close[0]], close[:-1]))
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * 1.004,
            "low": np.minimum(open_, close) * 0.996,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }
    )


@pytest.fixture
def empty_frame() -> pd.DataFrame:
    """Frame with the right columns but no rows."""
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])


def _spin_until(qapp: QApplication, watched: list[object], timeout_ms: int = 2_000) -> None:
    """
    Pump the event loop until ``watched`` is non-empty or the timeout expires.

    A bare processEvents() spin does not let wall-clock time advance enough for
    a short QTimer to fire, so QTest.qWait is what actually lets the timer run.
    """
    waited = 0
    while not watched and waited < timeout_ms:
        QTest.qWait(10)
        waited += 10
    qapp.processEvents()


# ============================================================================
# Helpers
# ============================================================================


def test_row_count_tolerates_junk() -> None:
    """row_count must not explode on None or on odd frames."""
    assert row_count(None) == 0
    assert row_count(pd.DataFrame()) == 0
    assert row_count(pd.DataFrame({"a": [1, 2, 3]})) == 3


def test_as_epoch_ms_handles_naive_tz_and_plain_index() -> None:
    """Every index flavour yields a monotonic, correctly scaled ms list."""
    naive = pd.bdate_range("2024-01-01", periods=5)
    assert as_epoch_ms(naive) == sorted(as_epoch_ms(naive))
    first_ms = as_epoch_ms(naive)[0]
    assert 1_700_000_000_000 < first_ms < 1_800_000_000_000

    tz_aware = naive.tz_localize("UTC")
    assert as_epoch_ms(tz_aware) == as_epoch_ms(naive)

    plain = as_epoch_ms(pd.RangeIndex(4))
    assert plain == [0.0, 86_400_000.0, 172_800_000.0, 259_200_000.0]


# ============================================================================
# CandlestickChart
# ============================================================================


def test_candlestick_chart_constructs(qapp: QApplication) -> None:
    """The widget builds with no data and no parent."""
    chart = CandlestickChart()
    assert chart.chart is not None
    assert chart.series.count() == 0
    assert chart.view is not None


def test_candlestick_chart_accepts_parent(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """A chart parented to a widget still takes data."""
    parent = CandlestickChart()
    child = CandlestickChart(parent)
    assert child.set_data(ohlcv) == len(ohlcv)


def test_candlestick_chart_empty_data_does_not_raise(
    qapp: QApplication, empty_frame: pd.DataFrame
) -> None:
    """The headline requirement: empty input is a no-op, not an exception."""
    chart = CandlestickChart()
    assert chart.set_data(empty_frame) == 0
    assert chart.series.count() == 0


def test_candlestick_chart_none_and_nan_are_safe(qapp: QApplication) -> None:
    """None and all-NaN frames clear the chart without raising."""
    chart = CandlestickChart()
    assert chart.set_data(None) == 0

    nan_frame = pd.DataFrame(
        {
            "open": [np.nan, np.nan],
            "high": [np.nan, np.nan],
            "low": [np.nan, np.nan],
            "close": [np.nan, np.nan],
        }
    )
    assert chart.set_data(nan_frame) == 0


def test_candlestick_chart_renders_data(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """Every valid row becomes one candle."""
    chart = CandlestickChart()
    assert chart.set_data(ohlcv) == len(ohlcv)
    assert chart.series.count() == len(ohlcv)


def test_candlestick_chart_accepts_datetime_index(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """Dates in the index work as well as a date column."""
    chart = CandlestickChart()
    indexed = ohlcv.set_index("date")[["open", "high", "low", "close"]]
    assert chart.set_data(indexed) == len(ohlcv)


def test_candlestick_chart_reload_replaces(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """A second set_data replaces rather than appends."""
    chart = CandlestickChart()
    chart.set_data(ohlcv)
    chart.set_data(ohlcv)
    assert chart.series.count() == len(ohlcv)

    chart.set_data(ohlcv.head(10))
    assert chart.series.count() == 10


def test_candlestick_chart_100k_rows_completes(qapp: QApplication) -> None:
    """100k bars must build in reasonable time without blowing up."""
    n = 100_000
    rng = np.random.default_rng(3)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, n)))
    open_ = np.concatenate(([close[0]], close[:-1]))
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("1990-01-01", periods=n),
            "open": open_,
            "high": np.maximum(open_, close) * 1.002,
            "low": np.minimum(open_, close) * 0.998,
            "close": close,
        }
    )

    chart = CandlestickChart()
    assert chart.set_data(frame) == n
    assert chart.series.count() == n


def test_candlestick_chart_clear_and_title(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """clear() empties the series and set_title does not raise."""
    chart = CandlestickChart()
    chart.set_data(ohlcv)
    chart.set_title("AAPL")
    chart.clear()
    assert chart.series.count() == 0


# ============================================================================
# VolumeChart
# ============================================================================


def test_volume_chart_constructs(qapp: QApplication) -> None:
    """The volume widget builds with one empty bar set attached."""
    chart = VolumeChart()
    assert chart.chart is not None
    # QBarSeries.count() reports bar sets, not bars, so check the set itself.
    assert len(chart.series.barSets()) == 1
    assert chart.series.barSets()[0].count() == 0


def test_volume_chart_empty_data_does_not_raise(
    qapp: QApplication, empty_frame: pd.DataFrame
) -> None:
    """Empty volume input is a no-op."""
    chart = VolumeChart()
    assert chart.set_data(empty_frame) == 0


def test_volume_chart_renders_data(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """One bar per row, and a reload does not stack up."""
    chart = VolumeChart()
    assert chart.set_data(ohlcv) == len(ohlcv)
    assert chart.series.count() == 1
    chart.set_data(ohlcv)
    assert chart.series.barSets()[0].count() == len(ohlcv)


def test_volume_chart_without_volume_column(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """A frame missing 'volume' yields nothing rather than an error."""
    chart = VolumeChart()
    assert chart.set_data(ohlcv[["open", "high", "low", "close"]]) == 0
    assert chart.set_data(None) == 0


def test_volume_chart_accepts_datetime_index(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """Index-dated volume frames work."""
    chart = VolumeChart()
    indexed = ohlcv.set_index("date")[["volume"]]
    assert chart.set_data(indexed) == len(indexed)


def test_volume_chart_100k_rows_completes(qapp: QApplication) -> None:
    """100k volume bars build quickly."""
    n = 100_000
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("1990-01-01", periods=n),
            "volume": np.linspace(1_000, 5_000_000, n),
        }
    )
    chart = VolumeChart()
    assert chart.set_data(frame) == n


# ============================================================================
# MovingAverageOverlay
# ============================================================================


def test_overlay_constructs_and_attaches(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """An overlay adds a line series to a price chart."""
    price = CandlestickChart()
    price.set_data(ohlcv)

    overlay = MovingAverageOverlay(price.chart, period=10)
    assert overlay.series.count() == 0
    assert overlay.period == 10
    assert overlay.kind == "sma"
    assert overlay.set_data(ohlcv) == len(ohlcv) - 9


def test_overlay_requires_a_chart(qapp: QApplication) -> None:
    """Passing no chart raises instead of silently doing nothing."""
    with pytest.raises(ValueError, match="needs a chart"):
        MovingAverageOverlay(None)


def test_overlay_finds_axis_when_omitted(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """With no axis passed, the overlay picks up the chart's horizontal axis."""
    price = CandlestickChart()
    price.set_data(ohlcv)
    assert price.chart.axes(Qt.Orientation.Horizontal)

    overlay = MovingAverageOverlay(price.chart)
    assert overlay.series.attachedAxes()
    overlay.set_data(ohlcv)


def test_overlay_sma_point_count_matches_window(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """SMA drops exactly period-1 leading points."""
    price = CandlestickChart()
    overlay = MovingAverageOverlay(price.chart, period=20, kind="sma")
    assert overlay.set_data(ohlcv) == len(ohlcv) - 19


def test_overlay_ema_keeps_every_point(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """EMA is defined from the first bar, so no points are dropped."""
    price = CandlestickChart()
    overlay = MovingAverageOverlay(price.chart, period=10, kind="ema")
    assert overlay.set_data(ohlcv) == len(ohlcv)


def test_overlay_sma_too_short_returns_zero(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """Fewer rows than the window produces no line, and raises no error."""
    price = CandlestickChart()
    overlay = MovingAverageOverlay(price.chart, period=50, kind="sma")
    assert overlay.set_data(ohlcv.head(10)) == 0
    assert overlay.point_count == 0


def test_overlay_empty_and_none_are_safe(qapp: QApplication) -> None:
    """Empty input clears the line."""
    price = CandlestickChart()
    overlay = MovingAverageOverlay(price.chart, period=5)
    assert overlay.set_data(None) == 0
    assert overlay.set_data(pd.DataFrame()) == 0


def test_overlay_rejects_bad_period(qapp: QApplication) -> None:
    """Period must be an integer >= 1."""
    price = CandlestickChart()
    for bad in (0, -1, -50):
        with pytest.raises(ValueError, match="period"):
            MovingAverageOverlay(price.chart, period=bad)
    with pytest.raises(ValueError, match="period"):
        MovingAverageOverlay(price.chart, period="abc")

    overlay = MovingAverageOverlay(price.chart, period=5)
    with pytest.raises(ValueError, match="period"):
        overlay.set_period(0)


def test_overlay_rejects_bad_kind(qapp: QApplication) -> None:
    """Kind must be sma or ema, case-insensitively."""
    price = CandlestickChart()
    with pytest.raises(ValueError, match="kind"):
        MovingAverageOverlay(price.chart, kind="wma")

    overlay = MovingAverageOverlay(price.chart, period=5, kind="EMA")
    assert overlay.kind == "ema"
    with pytest.raises(ValueError, match="kind"):
        overlay.set_kind("hma")


def test_overlay_period_and_kind_changes_redraw(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """Switching window or kind re-runs the overlay against cached data."""
    price = CandlestickChart()
    overlay = MovingAverageOverlay(price.chart, period=10)
    overlay.set_data(ohlcv)
    assert overlay.series.name() == "SMA10"

    overlay.set_period(30)
    assert overlay.period == 30
    assert overlay.series.name() == "SMA30"
    assert overlay.point_count == len(ohlcv) - 29

    overlay.set_kind("ema")
    assert overlay.series.name() == "EMA30"
    assert overlay.point_count == len(ohlcv)


def test_overlay_detach_removes_series(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """detach() takes the line off the chart."""
    price = CandlestickChart()
    price.set_data(ohlcv)
    before = len(price.chart.series())

    overlay = price.attach_moving_average(10)
    overlay.set_data(ohlcv)
    assert len(price.chart.series()) == before + 1

    overlay.detach()
    assert len(price.chart.series()) == before
    assert overlay.point_count == 0


def test_candlestick_chart_attach_moving_average(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """The convenience helper returns a usable overlay."""
    price = CandlestickChart()
    overlay = price.attach_moving_average(15, "ema")
    price.set_data(ohlcv)
    assert overlay.set_data(ohlcv) == len(ohlcv)


# ============================================================================
# Monitor
# ============================================================================


def test_monitor_default_interval_is_five_seconds(qapp: QApplication) -> None:
    """Default poll period is 5000ms."""
    monitor = Monitor(lambda: None)
    assert monitor.interval() == DEFAULT_INTERVAL_MS == 5_000
    assert monitor.timer.interval() == 5_000


def test_monitor_interval_is_settable(qapp: QApplication) -> None:
    """A valid interval round-trips through the timer."""
    monitor = Monitor(lambda: None, interval_ms=2_500)
    assert monitor.interval() == 2_500
    assert monitor.set_interval(9_000) == 9_000
    assert monitor.interval() == 9_000


def test_monitor_interval_clamps_to_one_second(qapp: QApplication) -> None:
    """Anything under 1s is raised to the 1000ms floor, not rejected."""
    monitor = Monitor(lambda: None)
    for requested in (0, 1, 500, 999):
        assert monitor.set_interval(requested) == MIN_INTERVAL_MS == 1_000
    assert monitor.interval() == 1_000


def test_monitor_interval_rejects_non_numeric(qapp: QApplication) -> None:
    """Non-numeric intervals raise ValueError."""
    monitor = Monitor(lambda: None)
    for bad in ("abc", None, object()):
        with pytest.raises(ValueError, match="interval_ms"):
            monitor.set_interval(bad)


def test_monitor_rejects_non_callable_source(qapp: QApplication) -> None:
    """A non-callable source is a TypeError at construction."""
    with pytest.raises(TypeError, match="callable"):
        Monitor("not a function")  # type: ignore[arg-type]


def test_monitor_emits_payload(qapp: QApplication) -> None:
    """poll_once() emits the source's return value on dataReady."""
    received: list[object] = []
    monitor = Monitor(lambda: {"bars": 3})
    monitor.dataReady.connect(received.append)

    assert monitor.poll_once() == {"bars": 3}
    assert received == [{"bars": 3}]
    assert monitor.poll_count == 1


def test_monitor_counts_polls_and_resets(qapp: QApplication) -> None:
    """poll_count tracks polls and reset() zeroes it."""
    counts: list[int] = []
    monitor = Monitor(lambda: 1)
    monitor.pollCountChanged.connect(counts.append)

    for _ in range(3):
        monitor.poll_once()
    assert monitor.poll_count == 3
    assert counts == [1, 2, 3]

    monitor.reset()
    assert monitor.poll_count == 0
    assert counts == [1, 2, 3, 0]


def test_monitor_swallows_source_errors(qapp: QApplication) -> None:
    """A raising source emits errorRaised and does not propagate."""

    def boom() -> None:
        raise RuntimeError("source unavailable")

    errors: list[str] = []
    monitor = Monitor(boom)
    monitor.errorRaised.connect(errors.append)

    assert monitor.poll_once() is None
    assert len(errors) == 1
    assert "source unavailable" in errors[0]
    assert monitor.poll_count == 0


def test_monitor_start_stop_toggle(qapp: QApplication) -> None:
    """start/stop/toggle move the timer through the expected states."""
    monitor = Monitor(lambda: None, interval_ms=1_000)
    assert not monitor.is_running

    monitor.start()
    assert monitor.is_running
    assert monitor.toggle() is False
    assert monitor.toggle() is True
    monitor.stop()
    assert not monitor.is_running


def test_monitor_auto_start(qapp: QApplication) -> None:
    """auto_start=True leaves the timer running."""
    monitor = Monitor(lambda: None, interval_ms=1_000, auto_start=True)
    assert monitor.is_running
    monitor.stop()


def test_monitor_fires_on_real_timer_tick(qapp: QApplication) -> None:
    """The QTimer itself drives a poll, not just poll_once()."""
    received: list[object] = []
    monitor = Monitor(lambda: "tick", interval_ms=MIN_INTERVAL_MS)
    monitor.dataReady.connect(received.append)

    monitor.timer.setInterval(1)
    monitor.start()
    try:
        _spin_until(qapp, received)
    finally:
        monitor.stop()

    assert received, "timer never fired a poll"
    assert received[0] == "tick"
    assert monitor.poll_count == len(received)


def test_monitor_timer_stops_with_the_object(qapp: QApplication) -> None:
    """A stopped monitor issues no further polls."""
    received: list[object] = []
    monitor = Monitor(lambda: 1, interval_ms=MIN_INTERVAL_MS)
    monitor.dataReady.connect(received.append)
    monitor.timer.setInterval(1)
    monitor.start()
    try:
        _spin_until(qapp, received)
    finally:
        monitor.stop()
    assert received, "timer never fired"

    settled = len(received)
    QTest.qWait(50)
    assert len(received) == settled, "timer kept firing after stop()"
    monitor.deleteLater()
    qapp.processEvents()


# ============================================================================
# MainWindow
# ============================================================================


def test_demo_rows_shape() -> None:
    """The placeholder source returns a usable OHLCV frame."""
    frame = demo_rows(30)
    assert len(frame) == 30
    for column in ("date", "open", "high", "low", "close", "volume"):
        assert column in frame.columns
    assert frame["high"].ge(frame["low"]).all()


def test_main_window_builds_and_renders(qapp: QApplication) -> None:
    """build_window() produces a populated window without an event loop."""
    window = build_window()
    assert window.price_chart.series.count() > 0
    assert window.volume_chart.series.barSets()[0].count() > 0
    assert window.ma_overlay.point_count > 0
    assert "bars" in window.status_label.text()
    window.close()


def test_main_window_accepts_custom_source(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """A supplied source drives the initial paint."""
    window = build_window(source=lambda: ohlcv, interval_ms=2_000)
    assert window.price_chart.series.count() == len(ohlcv)
    assert window.monitor.interval() == 2_000
    window.close()


def test_main_window_apply_data_handles_empty(qapp: QApplication, ohlcv: pd.DataFrame) -> None:
    """Applying an empty frame clears both charts without raising."""
    window = build_window()
    window.apply_data(pd.DataFrame())
    assert window.price_chart.series.count() == 0
    assert "0 bars" in window.status_label.text()

    assert window.apply_data(ohlcv) == len(ohlcv)
    window.close()


def test_main_window_toggle_button_drives_monitor(qapp: QApplication) -> None:
    """Checking the button starts the timer, unchecking stops it."""
    window = build_window()
    window.toggle_button.setChecked(True)
    assert window.monitor.is_running

    window.toggle_button.setChecked(True)  # idempotent
    assert window.monitor.is_running

    window.toggle_button.setChecked(False)
    assert not window.monitor.is_running
    window.close()


def test_main_window_monitor_repaint(qapp: QApplication) -> None:
    """A monitor tick repaints the charts through the signal."""
    window = build_window()
    window.apply_data(pd.DataFrame())

    window.monitor.poll_once()
    assert window.price_chart.series.count() > 0
    assert window.ma_overlay.point_count > 0
    window.close()


def test_main_window_survives_failing_source(qapp: QApplication) -> None:
    """A source that raises leaves the window alive and reports the error."""

    def boom() -> pd.DataFrame:
        raise ConnectionError("no feed")

    window = build_window(source=boom)
    window.monitor.poll_once()
    assert "Error" in window.status_label.text()
    window.close()


def test_main_window_close_stops_monitor(qapp: QApplication) -> None:
    """Closing the window stops polling."""
    window = build_window()
    window.toggle_button.setChecked(True)
    assert window.monitor.is_running
    window.close()
    assert not window.monitor.is_running
