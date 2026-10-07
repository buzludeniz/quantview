"""
QuantView Desktop Monitor.

A ``QObject`` that drives a user-supplied callable on a ``QTimer`` and emits
whatever the callable returns. The intended shape is polling a market data
source and repainting a chart when fresh rows arrive.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

__all__ = ["DEFAULT_INTERVAL_MS", "MIN_INTERVAL_MS", "Monitor"]

DEFAULT_INTERVAL_MS = 5_000
MIN_INTERVAL_MS = 1_000


class Monitor(QObject):
    """
    Polls a callable on a timer and republishes its results as a signal.

    The callable is invoked with no arguments and its return value is emitted
    through :attr:`dataReady`. Exceptions raised by the callable are caught and
    re-emitted on :attr:`errorRaised` so one bad poll does not kill the timer.

    Args:
        source: Zero-argument callable returning the payload to publish.
        interval_ms: Poll period in milliseconds. Clamped to
            :data:`MIN_INTERVAL_MS`; the default is :data:`DEFAULT_INTERVAL_MS`.
        parent: Qt parent.
        auto_start: Start the timer immediately. Defaults to False.
    """

    # Qt signals are camelCase by convention, which pep8-naming flags here.
    dataReady = pyqtSignal(object)
    errorRaised = pyqtSignal(str)
    pollCountChanged = pyqtSignal(int)

    def __init__(
        self,
        source: Callable[[], object],
        interval_ms: int = DEFAULT_INTERVAL_MS,
        parent: QObject | None = None,
        auto_start: bool = False,
    ) -> None:
        super().__init__(parent)

        if not callable(source):
            raise TypeError(f"source must be callable, got {type(source).__name__}")

        self._source = source
        self._poll_count = 0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timeout)

        self.set_interval(interval_ms)
        if auto_start:
            self.start()

    @property
    def timer(self) -> QTimer:
        """The underlying QTimer."""
        return self._timer

    @property
    def source(self) -> Callable[[], object]:
        """The polled callable."""
        return self._source

    @property
    def poll_count(self) -> int:
        """How many polls have completed since the last :meth:`reset`."""
        return self._poll_count

    @property
    def is_running(self) -> bool:
        """Whether the timer is active."""
        return self._timer.isActive()

    def set_interval(self, interval_ms: int) -> int:
        """
        Set the poll period, clamped to :data:`MIN_INTERVAL_MS`.

        Args:
            interval_ms: Requested period in milliseconds.

        Returns:
            The interval actually applied.
        """
        interval = self._validate_interval(interval_ms)
        self._timer.setInterval(interval)
        return interval

    def interval(self) -> int:
        """The active poll period in milliseconds."""
        return self._timer.interval()

    @staticmethod
    def _validate_interval(interval_ms: object) -> int:
        """Coerce to int milliseconds and clamp below to the 1s floor."""
        try:
            value = int(interval_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"interval_ms must be an integer number of milliseconds, got {interval_ms!r}"
            ) from exc
        if value < MIN_INTERVAL_MS:
            return MIN_INTERVAL_MS
        return value

    def start(self) -> None:
        """Start polling."""
        self._timer.start()

    def stop(self) -> None:
        """Stop polling."""
        self._timer.stop()

    def toggle(self) -> bool:
        """
        Flip the running state.

        Returns:
            True if the timer is running after the toggle.
        """
        if self._timer.isActive():
            self._timer.stop()
        else:
            self._timer.start()
        return self._timer.isActive()

    def reset(self) -> None:
        """Zero the poll counter without touching the timer."""
        self._poll_count = 0
        self.pollCountChanged.emit(0)

    def poll_once(self) -> object:
        """
        Poll immediately, outside the timer.

        Same error handling as a timer tick, so this is safe to call in tests.

        Returns:
            The callable's return value, or None if it raised.
        """
        return self._on_timeout()

    def _on_timeout(self) -> object:
        """Run one poll, emit the result, and swallow source errors."""
        try:
            payload = self._source()
        except Exception as exc:
            self.errorRaised.emit(f"{type(exc).__name__}: {exc}")
            return None
        self._poll_count += 1
        self.pollCountChanged.emit(self._poll_count)
        self.dataReady.emit(payload)
        return payload
