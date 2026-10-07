"""
QuantView Desktop Entry Point.

Builds the main window (price chart, volume pane, status bar) and hands control
to the Qt event loop. Run it with::

    python -m quantview.desktop.main
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .charts import CandlestickChart, MovingAverageOverlay, VolumeChart
from .monitor import DEFAULT_INTERVAL_MS, Monitor

__all__ = ["MainWindow", "build_window", "demo_rows", "main"]

WINDOW_TITLE = "QuantView"
SYMBOL = "DEMO"


def demo_rows(count: int = 250) -> pd.DataFrame:
    """
    Build a deterministic OHLCV frame for the empty-state chart.

    A real session would read from DuckDB via quantview.data; this keeps the
    window runnable with no data present.

    Args:
        count: Number of bars to synthesise.

    Returns:
        DataFrame with date/open/high/low/close/volume.
    """
    import numpy as np

    periods = max(int(count), 1)
    dates = pd.bdate_range("2024-01-01", periods=periods)
    rng = np.random.default_rng(7)
    steps = rng.normal(0.0004, 0.014, periods)
    closes = 100.0 * np.exp(np.cumsum(steps))
    opens = np.concatenate(([closes[0]], closes[:-1]))
    spread = np.abs(rng.normal(0.0, 0.004, periods)) + 0.001
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": np.maximum(opens, closes) * (1.0 + spread),
            "low": np.minimum(opens, closes) * (1.0 - spread),
            "close": closes,
            "volume": rng.integers(1_000_000, 9_000_000, periods).astype(float),
        }
    )


class MainWindow(QMainWindow):
    """
    Application window: candlestick chart above a volume pane.

    A :class:`~quantview.desktop.monitor.Monitor` polls the data source on a
    timer and repaints both charts when a new frame arrives.

    Args:
        source: Zero-argument callable returning an OHLCV DataFrame.
        interval_ms: Poll period handed to the monitor.
        parent: Qt parent.
    """

    def __init__(
        self,
        source: Callable[[], pd.DataFrame] | None = None,
        interval_ms: int = DEFAULT_INTERVAL_MS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1180, 820)

        self._source = source if source is not None else demo_rows

        self.price_chart = CandlestickChart()
        self.volume_chart = VolumeChart()
        self.ma_overlay: MovingAverageOverlay = self.price_chart.attach_moving_average(20, "sma")

        self.status_label = QLabel("Ready")
        self.toggle_button = QPushButton("Pause")
        self.toggle_button.setCheckable(True)

        self._build_layout()

        self.monitor = Monitor(self._source, interval_ms=interval_ms, parent=self)
        self.monitor.dataReady.connect(self._on_data)
        self.monitor.errorRaised.connect(self._on_error)
        self.monitor.pollCountChanged.connect(self._on_poll)

        # 'toggled' rather than 'clicked': programmatic setChecked() in tests
        # then exercises the same path as a real click.
        self.toggle_button.toggled.connect(self._on_toggle)

        # A source that is down on startup must not stop the window opening.
        try:
            self.refresh()
        except Exception as exc:
            self._on_error(f"{type(exc).__name__}: {exc}")

    def _build_layout(self) -> None:
        """Assemble the splitter, toolbar row, and status bar."""
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.price_chart)
        splitter.addWidget(self.volume_chart)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.setContentsMargins(8, 8, 8, 0)
        row.addWidget(self.toggle_button)
        row.addStretch(1)
        buttons.setMaximumHeight(44)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(buttons)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self.statusBar().addPermanentWidget(self.status_label)

    def refresh(self) -> int:
        """
        Pull one frame from the source and repaint both charts.

        Returns:
            The number of candles drawn.
        """
        frame = self._source()
        return self.apply_data(frame)

    def apply_data(self, frame: pd.DataFrame) -> int:
        """
        Render ``frame`` into the charts and refresh the overlay.

        Args:
            frame: OHLCV DataFrame. An empty frame clears the charts.

        Returns:
            The number of candles drawn.
        """
        drawn = self.price_chart.set_data(frame)
        self.volume_chart.set_data(frame)
        if drawn:
            self.ma_overlay.set_data(frame)
        self.status_label.setText(f"{SYMBOL} - {drawn} bars")
        return drawn

    def _on_data(self, frame: object) -> None:
        """Monitor slot: a poll returned a payload."""
        if isinstance(frame, pd.DataFrame):
            self.apply_data(frame)

    def _on_error(self, message: str) -> None:
        """Monitor slot: a poll raised."""
        self.status_label.setText(f"Error: {message}")

    def _on_poll(self, count: int) -> None:
        """Monitor slot: poll counter advanced."""
        if not self.monitor.is_running:
            self.statusBar().showMessage(f"Polled {count} time(s), stopped")

    def _on_toggle(self, checked: bool) -> None:
        """
        Toolbar slot: start or stop polling.

        Returns early when the state already matches, so programmatic and
        user-driven changes stay idempotent.
        """
        if checked == self.monitor.is_running:
            return
        if checked:
            self.monitor.start()
            self.toggle_button.setText("Pause")
            self.statusBar().showMessage("Live")
        else:
            self.monitor.stop()
            self.toggle_button.setText("Resume")
            self.statusBar().showMessage("Paused")

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop the timer so a closed window leaves no poll running."""
        self.monitor.stop()
        super().closeEvent(event)


def build_window(
    source: Callable[[], pd.DataFrame] | None = None,
    interval_ms: int = DEFAULT_INTERVAL_MS,
) -> MainWindow:
    """
    Construct the application window.

    Args:
        source: Zero-argument callable returning an OHLCV DataFrame.
        interval_ms: Poll period in milliseconds.

    Returns:
        A fully populated, unpainted MainWindow.
    """
    return MainWindow(source=source, interval_ms=interval_ms)


def main(argv: list[str] | None = None) -> int:
    """
    Start the Qt application and show the main window.

    Args:
        argv: Command line arguments. Defaults to ``sys.argv``.

    Returns:
        The Qt exit code.
    """
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(argv if argv is not None else sys.argv)

    window = build_window()
    window.show()

    if not owns_app:
        # A QApplication already existed (embedded or under test), so there is
        # no event loop of ours to enter.
        return 0
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
