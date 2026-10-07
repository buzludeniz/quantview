"""Base Data Connector with retry, caching, and error logging."""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from types import TracebackType
from typing import Any, Literal, Self, TypeVar

import pandas as pd

T = TypeVar("T")

logger = logging.getLogger(__name__)


class BaseConnector(ABC):
    """Base class for all data connectors.

    Provides:
    - Exponential backoff retry (1, 2, 4, 8, 60s)
    - 60-second in-memory cache via lru_cache
    - Structured error logging
    """

    # Retry configuration
    MAX_RETRIES = 5
    BACKOFF_BASE = (1, 2, 4, 8, 60)  # seconds, a fixed schedule

    def __init__(self, cache_ttl: int = 60):
        """Initialize the connector.

        Args:
            cache_ttl: Cache time-to-live in seconds (default 60).
        """
        self.cache_ttl = cache_ttl
        self._cache_timestamp: float | None = None
        self._cached_result: Any = None

    def _is_cache_valid(self) -> bool:
        """Check if cached result is still valid."""
        if self._cache_timestamp is None:
            return False
        return (time.time() - self._cache_timestamp) < self.cache_ttl

    def _clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache_timestamp = None
        self._cached_result = None

    def _execute_with_retry(self, func: Callable[..., T], *args: object, **kwargs: object) -> T:
        """Execute a function with exponential backoff retry.

        Args:
            func: Callable to execute.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Result of func(*args, **kwargs).

        Raises:
            Exception: Re-raises the last exception if all retries exhausted.
        """
        last_exception = None

        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                wait_time = self.BACKOFF_BASE[attempt] if attempt < len(self.BACKOFF_BASE) else 60

                logger.warning(
                    "Connector %s attempt %d/%d failed: %s. Retrying in %ds...",
                    self.__class__.__name__,
                    attempt + 1,
                    self.MAX_RETRIES,
                    str(e),
                    wait_time,
                )

                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(wait_time)

        # MAX_RETRIES is at least 1, so the loop always records a failure before
        # reaching here; the fallback keeps the raise total.
        assert last_exception is not None, "retry loop completed without a failure"
        logger.error(
            "Connector %s failed after %d attempts: %s",
            self.__class__.__name__,
            self.MAX_RETRIES,
            str(last_exception),
        )
        raise last_exception

    @abstractmethod
    def fetch(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """Fetch data for the given symbols and date range.

        Args:
            symbols: List of symbols/tickers to fetch.
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format.

        Returns:
            Normalized DataFrame matching the corresponding raw table schema.
        """
        pass

    def _validate_dataframe(self, df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
        """Validate DataFrame has required columns and proper types.

        Args:
            df: DataFrame to validate.
            required_columns: List of required column names.

        Returns:
            Validated DataFrame.

        Raises:
            ValueError: If required columns are missing.
        """
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure date column is proper datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date

        return df

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self._clear_cache()
        return False
