"""
QuantView Desktop Package.

PyQt6 desktop front end: candlestick/volume charts with moving-average
overlays, a timer-driven data monitor, and the application entry point.
"""

from .charts import CandlestickChart, MovingAverageOverlay, VolumeChart
from .monitor import Monitor

__all__ = [
    "CandlestickChart",
    "Monitor",
    "MovingAverageOverlay",
    "VolumeChart",
    "main",
]

__version__ = "0.1.0"


def main() -> int:
    """Launch the QuantView desktop application.

    Imported lazily so that ``import quantview.desktop`` does not require a
    QApplication to exist yet.

    Returns:
        The Qt exit code from ``app.exec()``.
    """
    from .main import main as _main

    return _main()
