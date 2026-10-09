"""Tests for QuantView configuration."""

from pathlib import Path

from quantview.config import Settings, get_settings, reload_settings


def test_settings_load_defaults() -> None:
    """Test that settings load with default values."""
    settings = Settings()
    assert settings.data_path == Path("./data/quantview.duckdb")
    assert settings.polling_interval_seconds == 60
    assert settings.symbols == ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
    assert settings.fred_api_key == ""
    assert settings.log_level == "INFO"


def test_settings_env_override(monkeypatch) -> None:
    """Test that environment variables override config values."""
    monkeypatch.setenv("QUANTVIEW_DATA_PATH", "/custom/path.duckdb")
    monkeypatch.setenv("QUANTVIEW_POLLING_INTERVAL_SECONDS", "30")
    monkeypatch.setenv("QUANTVIEW_FRED_API_KEY", "test_key")

    settings = Settings()
    assert settings.data_path == Path("/custom/path.duckdb")
    assert settings.polling_interval_seconds == 30
    assert settings.fred_api_key == "test_key"


def test_settings_nested_env_override(monkeypatch) -> None:
    """Test that nested environment variables work with double underscore."""
    monkeypatch.setenv("QUANTVIEW_ALERT_THRESHOLDS__PRICE_CHANGE_PCT", "5.0")
    monkeypatch.setenv("QUANTVIEW_EMAIL_SMTP__HOST", "smtp.custom.com")
    monkeypatch.setenv("QUANTVIEW_EMAIL_SMTP__PORT", "465")

    settings = Settings()
    assert settings.alert_thresholds.price_change_pct == 5.0
    assert settings.email_smtp.host == "smtp.custom.com"
    assert settings.email_smtp.port == 465


def test_get_settings_returns_global() -> None:
    """Test that get_settings returns the global instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_reload_settings_creates_new_instance() -> None:
    """Test that reload_settings creates a new instance."""
    s1 = get_settings()
    s2 = reload_settings()
    assert s1 is not s2
    assert s2 is get_settings()


def test_constructing_settings_does_not_touch_the_filesystem(tmp_path: Path, monkeypatch) -> None:
    """Building a settings object is a read, and a read must not write.

    The constructor used to mkdir the data and log directories. That made merely
    reading configuration fail wherever the configured path is not writable: on a
    Linux runner a data_path of "/custom/path.duckdb" raised PermissionError out
    of ``Settings()``, because the runner user cannot create a directory at the
    filesystem root.
    """
    unwritable = tmp_path / "not-created-yet" / "sub" / "quantview.duckdb"
    monkeypatch.setenv("QUANTVIEW_DATA_PATH", str(unwritable))
    monkeypatch.setenv("QUANTVIEW_LOG_FILE", str(tmp_path / "logs" / "qv.log"))

    settings = Settings()

    assert settings.data_path == unwritable
    assert not unwritable.parent.exists(), "the constructor created a directory"
    assert not (tmp_path / "logs").exists(), "the constructor created a log directory"


def test_ensure_directories_creates_both(tmp_path: Path, monkeypatch) -> None:
    """ensure_directories is the explicit step, called where a path is opened."""
    monkeypatch.setenv("QUANTVIEW_DATA_PATH", str(tmp_path / "data" / "qv.duckdb"))
    monkeypatch.setenv("QUANTVIEW_LOG_FILE", str(tmp_path / "logs" / "qv.log"))

    settings = Settings()
    settings.ensure_directories()

    assert settings.data_path.parent.is_dir()
    assert settings.log_file.parent.is_dir()


def test_ensure_directories_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QUANTVIEW_DATA_PATH", str(tmp_path / "data" / "qv.duckdb"))
    monkeypatch.setenv("QUANTVIEW_LOG_FILE", str(tmp_path / "logs" / "qv.log"))

    settings = Settings()
    settings.ensure_directories()
    settings.ensure_directories()

    assert settings.data_path.parent.is_dir()


def test_settings_env_override_accepts_an_unwritable_path(monkeypatch) -> None:
    """An override to a path that cannot be created must still read back.

    The env override is about precedence, not about whether the path happens to
    be creatable, so it must not depend on the filesystem.
    """
    monkeypatch.setenv("QUANTVIEW_DATA_PATH", "/custom/path.duckdb")
    monkeypatch.setenv("QUANTVIEW_LOG_FILE", "/custom/logs/qv.log")

    settings = Settings()

    assert settings.data_path == Path("/custom/path.duckdb")
    assert settings.log_file == Path("/custom/logs/qv.log")
