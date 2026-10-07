"""
QuantView Desktop Charts.

Qt-based chart widgets built on PyQt6.QtCharts. Each class wraps a
``QChartView`` so it can be dropped straight into a Qt layout.

Charts in this module accept a pandas DataFrame with the canonical OHLCV
columns (``date``/``open``/``high``/``low``/``close``/``volume``) and are
defensive about empty or malformed input: feeding them an empty frame is a
no-op rather than an exception.
"""

from __future__ import annotations

import pandas as pd
from PyQt6.QtCharts import (
    QBarSeries,
    QBarSet,
    QCandlestickSeries,
    QCandlestickSet,
    QChart,
    QChartView,
    QDateTimeAxis,
    QLineSeries,
    QValueAxis,
)
from PyQt6.QtCore import QDateTime, QObject, QPointF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QVBoxLayout, QWidget

__all__ = [
    "OHLCV_COLUMNS",
    "CandlestickChart",
    "MovingAverageOverlay",
    "VolumeChart",
    "as_epoch_ms",
    "row_count",
]

# OHLCV_COLUMNS keeps source order (date/open/...); __all__ is sorted for ruff.

# Columns a price chart needs. 'date' is optional -- it may live in the index.
OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")

UP_COLOR = QColor("#3fb950")
DOWN_COLOR = QColor("#f85149")
GRID_COLOR = QColor(255, 255, 255, 24)
BACKGROUND = QColor("#0d1117")


def row_count(rows: pd.DataFrame) -> int:
    """Number of plottable rows in ``rows``, tolerating anything."""
    if rows is None:
        return 0
    try:
        return len(rows.index)
    except Exception:
        return 0


def as_epoch_ms(index: pd.Index) -> list[float]:
    """
    Convert a pandas index into epoch milliseconds.

    Falls back to a synthetic daily sequence when the index holds no dates, so
    the x-axis still has a monotonic scale to draw against.
    """
    if isinstance(index, pd.DatetimeIndex):
        # as_unit("ms") rather than a hardcoded 10**9 divisor: pandas 3 defaults
        # DatetimeIndex to microsecond resolution, so the raw int64 is unit
        # dependent. tz-aware indexes are converted first to stay in range.
        stamps = index.tz_convert("UTC") if index.tz is not None else index
        return [float(v) for v in stamps.as_unit("ms").astype("int64")]
    return [float(i) * 86_400_000.0 for i in range(len(index))]


def _pen(colour: QColor) -> QPen:
    """Flat 1px pen in ``colour``, matching a candlestick body outline."""
    pen = QPen(colour)
    pen.setWidthF(1.0)
    return pen


class CandlestickChart(QWidget):
    """
    Price chart rendering OHLC bars as candlesticks.

    The widget owns a ``QCandlestickSeries`` on a ``QDateTimeAxis``. Prices are
    coloured per bar (green up, red down) so no legend lookups are needed.

    Args:
        parent: Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._series = QCandlestickSeries()
        self._series.setName("OHLC")

        self._chart = QChart()
        self._chart.addSeries(self._series)
        self._chart.setBackgroundBrush(BACKGROUND)
        self._chart.setPlotAreaBackgroundBrush(QColor("#010409"))
        self._chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        self._chart.legend().setVisible(True)
        self._chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._chart.legend().setBackgroundVisible(False)
        self._chart.legend().setLabelColor(QColor("#e6edf3"))

        self._axis_x = QDateTimeAxis()
        self._axis_x.setFormat("yyyy-MM-dd")
        self._axis_x.setTickCount(6)
        self._axis_x.setLabelsColor(QColor("#8b949e"))
        self._axis_x.setGridLineColor(GRID_COLOR)
        self._chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)
        self._series.attachAxis(self._axis_x)

        self._view = QChartView(self._chart)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setMinimumHeight(240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._row_count = 0
        self._overlays: list[MovingAverageOverlay] = []

    @property
    def chart(self) -> QChart:
        """The underlying QChart."""
        return self._chart

    @property
    def series(self) -> QCandlestickSeries:
        """The underlying candlestick series."""
        return self._series

    @property
    def view(self) -> QChartView:
        """The QChartView wrapping the chart."""
        return self._view

    def set_title(self, title: str) -> None:
        """Set the chart title. Pass an empty string to clear it."""
        self._chart.setTitle(title)

    def clear(self) -> None:
        """Remove every candle from the chart."""
        self._series.clear()
        self._row_count = 0

    def set_data(self, rows: pd.DataFrame) -> int:
        """
        Replace the chart contents with ``rows``.

        Args:
            rows: DataFrame carrying open/high/low/close. A ``date`` column or a
                DatetimeIndex supplies the x-axis; without one the bars are
                spaced by position. Volume, when present, is ignored here --
                VolumeChart owns it.

        Returns:
            The number of candles actually drawn.
        """
        prepared = self._prepare(rows)
        if prepared is None:
            self.clear()
            return 0

        stamps, opens, highs, lows, closes = prepared
        bars = [
            QCandlestickSet(o, h, low_, c, t)
            for t, o, h, low_, c in zip(stamps, opens, highs, lows, closes, strict=True)
        ]
        for bar in bars:
            colour = UP_COLOR if bar.close() >= bar.open() else DOWN_COLOR
            bar.setBrush(QBrush(colour))
            bar.setPen(_pen(colour))

        if self._series.count():
            self._series.clear()
        self._series.append(bars)

        first, last = stamps[0], stamps[-1]
        if first == last:
            first, last = first - 1.0, last + 1.0
        self._axis_x.setRange(
            QDateTime.fromMSecsSinceEpoch(int(first)),
            QDateTime.fromMSecsSinceEpoch(int(last)),
        )

        self._row_count = len(bars)
        return self._row_count

    def attach_moving_average(self, period: int, kind: str = "sma") -> MovingAverageOverlay:
        """
        Overlay a moving average of ``close`` on this chart.

        The overlay shares the chart and the date axis, so it stays aligned with
        the candles as either side is updated.

        Args:
            period: Lookback window in bars. Must be >= 1.
            kind: ``"sma"`` for simple, ``"ema"`` for exponential.

        Returns:
            The attached MovingAverageOverlay.
        """
        overlay = MovingAverageOverlay(self._chart, self._axis_x, period=period, kind=kind)
        self._overlays.append(overlay)
        return overlay

    @staticmethod
    def _prepare(
        rows: pd.DataFrame,
    ) -> tuple[list[float], list[float], list[float], list[float], list[float]] | None:
        """Validate and reshape ``rows`` into per-bar values, or None if empty."""
        if rows is None or row_count(rows) == 0:
            return None

        frame = rows
        if "close" not in frame.columns:
            if frame.shape[1] < 4:
                return None
            frame = frame.copy()
            frame.columns = list(OHLCV_COLUMNS[1:5]) + list(frame.columns[4:])

        frame = frame.dropna(subset=["open", "high", "low", "close"], how="any")
        if frame.empty:
            return None

        stamps = as_epoch_ms(pd.Index(frame["date"]) if "date" in frame.columns else frame.index)

        def column(name: str) -> list[float]:
            return [float(v) for v in frame[name].tolist()]

        return stamps, column("open"), column("high"), column("low"), column("close")


class VolumeChart(QWidget):
    """
    Volume histogram, usually stacked beneath a :class:`CandlestickChart`.

    Bars are coloured by the direction of their own bar (close above open is
    green) so the two panes read together.

    Args:
        parent: Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._bar_set = QBarSet("Volume")
        self._bar_set.setBrush(QBrush(QColor("#58a6ff")))
        self._bar_set.setPen(_pen(QColor("#58a6ff")))

        self._series = QBarSeries()
        self._series.append(self._bar_set)
        self._series.setName("Volume")

        self._chart = QChart()
        self._chart.addSeries(self._series)
        self._chart.setBackgroundBrush(BACKGROUND)
        self._chart.setPlotAreaBackgroundBrush(QColor("#010409"))
        self._chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        self._chart.legend().setVisible(False)

        self._axis_x = QDateTimeAxis()
        self._axis_x.setFormat("yyyy-MM-dd")
        self._axis_x.setTickCount(6)
        self._axis_x.setLabelsColor(QColor("#8b949e"))
        self._axis_x.setGridLineColor(GRID_COLOR)
        self._chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)
        self._series.attachAxis(self._axis_x)

        self._axis_y = QValueAxis()
        self._axis_y.setLabelFormat("%.0f")
        self._axis_y.setLabelsColor(QColor("#8b949e"))
        self._axis_y.setGridLineColor(GRID_COLOR)
        self._chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)
        self._series.attachAxis(self._axis_y)

        self._view = QChartView(self._chart)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setMinimumHeight(100)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

    @property
    def chart(self) -> QChart:
        """The underlying QChart."""
        return self._chart

    @property
    def series(self) -> QBarSeries:
        """The underlying bar series."""
        return self._series

    @property
    def view(self) -> QChartView:
        """The QChartView wrapping the chart."""
        return self._view

    def set_title(self, title: str) -> None:
        """Set the chart title. Pass an empty string to clear it."""
        self._chart.setTitle(title)

    def clear(self) -> None:
        """Remove every volume bar."""
        count = self._bar_set.count()
        if count:
            self._bar_set.remove(0, count)
        self._axis_y.setRange(0.0, 1.0)

    def set_data(self, rows: pd.DataFrame) -> int:
        """
        Replace the chart contents with the volume in ``rows``.

        Args:
            rows: DataFrame with a ``volume`` column. ``open`` and ``close`` are
                used only to colour the bars when both are present.

        Returns:
            The number of bars actually drawn.
        """
        if rows is None or row_count(rows) == 0:
            self.clear()
            return 0

        frame = rows.dropna(subset=["volume"], how="any") if "volume" in rows.columns else None
        if frame is None or frame.empty:
            self.clear()
            return 0

        values = [float(v) for v in frame["volume"].tolist()]
        coloured = "open" in frame.columns and "close" in frame.columns
        if coloured:
            opens = frame["open"].tolist()
            closes = frame["close"].tolist()

        self.clear()
        for i, value in enumerate(values):
            if coloured:
                colour = UP_COLOR if closes[i] >= opens[i] else DOWN_COLOR
                self._bar_set.setColor(QColor(colour.red(), colour.green(), colour.blue(), 170))
            self._bar_set.append(value)

        peak = max(values) if values else 0.0
        self._axis_y.setRange(0.0, peak * 1.1 if peak > 0 else 1.0)

        if "date" in frame.columns:
            stamps = as_epoch_ms(pd.Index(frame["date"]))
        elif isinstance(frame.index, pd.DatetimeIndex):
            stamps = as_epoch_ms(frame.index)
        else:
            stamps = [float(i) * 86_400_000.0 for i in range(len(frame))]
        self._axis_x.setRange(
            QDateTime.fromMSecsSinceEpoch(int(stamps[0])),
            QDateTime.fromMSecsSinceEpoch(int(stamps[-1])),
        )
        return len(values)


class MovingAverageOverlay(QObject):
    """
    A moving-average line drawn on top of an existing chart.

    The overlay attaches a ``QLineSeries`` to a target chart and axis, so it
    inherits their scale and needs no axes of its own.

    Args:
        chart: Chart to draw into. Must already contain a price series.
        axis: Horizontal axis to share. Defaults to the chart's first one.
        period: Lookback window in bars.
        kind: ``"sma"`` or ``"ema"``.
        color: Line colour.
        parent: Qt parent.
    """

    MA_KINDS = ("sma", "ema")

    def __init__(
        self,
        chart: QChart,
        axis: object | None = None,
        period: int = 20,
        kind: str = "sma",
        color: str = "#d29922",
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        if chart is None:
            raise ValueError("MovingAverageOverlay needs a chart to attach to")

        self._period = self._validate_period(period)
        self._kind = self._validate_kind(kind)

        self._series = QLineSeries()
        self._series.setName(f"{self._kind.upper()}{self._period}")
        self._series.setPen(_pen(QColor(color)))

        self._chart = chart
        self._axis = axis if axis is not None else self._first_horizontal_axis(chart)
        # Order matters: the axis can only attach to a series already in a chart.
        self._chart.addSeries(self._series)
        if self._axis is not None:
            self._series.attachAxis(self._axis)
        self._point_count = 0
        self._source: pd.DataFrame | None = None

    @property
    def series(self) -> QLineSeries:
        """The underlying line series."""
        return self._series

    @property
    def period(self) -> int:
        """Lookback window in bars."""
        return self._period

    @property
    def kind(self) -> str:
        """``"sma"`` or ``"ema"``."""
        return self._kind

    @staticmethod
    def _validate_period(period: int) -> int:
        """Coerce ``period`` to an int >= 1 or raise ValueError."""
        try:
            value = int(period)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"period must be an integer, got {period!r}") from exc
        if value < 1:
            raise ValueError(f"period must be >= 1, got {value}")
        return value

    @staticmethod
    def _validate_kind(kind: str) -> str:
        """Normalise ``kind`` to lowercase, or raise ValueError."""
        if not isinstance(kind, str):
            raise ValueError(f"kind must be a string, got {type(kind).__name__}")
        value = kind.strip().lower()
        if value not in MovingAverageOverlay.MA_KINDS:
            allowed = ", ".join(MovingAverageOverlay.MA_KINDS)
            raise ValueError(f"kind must be one of {allowed}, got {kind!r}")
        return value

    @staticmethod
    def _first_horizontal_axis(chart: QChart) -> object | None:
        """The chart's first horizontal axis, if it has one."""
        axes = chart.axes(Qt.Orientation.Horizontal)
        return axes[0] if axes else None

    def set_period(self, period: int) -> None:
        """Change the lookback window. Redraws if data was already supplied."""
        self._period = self._validate_period(period)
        self._series.setName(f"{self._kind.upper()}{self._period}")
        if self._source is not None:
            self.set_data(self._source)

    def set_kind(self, kind: str) -> None:
        """Change the average between ``sma`` and ``ema``. Redraws if needed."""
        self._kind = self._validate_kind(kind)
        self._series.setName(f"{self._kind.upper()}{self._period}")
        if self._source is not None:
            self.set_data(self._source)

    def set_data(self, rows: pd.DataFrame) -> int:
        """
        Recompute the average from ``rows`` and redraw the line.

        Args:
            rows: DataFrame with a ``close`` column, optionally dated.

        Returns:
            The number of plotted points, which is ``len(rows) - period + 1``
            once enough history exists, otherwise 0.
        """
        self._source = rows

        if rows is None or row_count(rows) == 0 or "close" not in rows.columns:
            self.clear()
            return 0

        frame = rows.dropna(subset=["close"], how="any")
        closes = frame["close"].astype(float)
        if self._kind == "ema":
            average = closes.ewm(span=self._period, adjust=False).mean()
        else:
            average = closes.rolling(self._period).mean()

        valid = average.dropna()
        if valid.empty:
            self.clear()
            return 0

        if "date" in frame.columns:
            stamps = as_epoch_ms(pd.Index(frame["date"]))
        else:
            stamps = as_epoch_ms(frame.index)
        points = [QPointF(stamps[i], float(average.iloc[i])) for i in valid.index]

        self._series.replace(points)
        self._point_count = len(points)
        return self._point_count

    def clear(self) -> None:
        """Remove the line from the chart."""
        self._series.replace([])
        self._point_count = 0

    def detach(self) -> None:
        """Remove the line series from its chart entirely."""
        self.clear()
        self._chart.removeSeries(self._series)

    @property
    def point_count(self) -> int:
        """Number of points currently plotted."""
        return self._point_count
