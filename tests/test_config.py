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


def test_data_directory_created() -> None:
    """Test that data directory is created on settings initialization."""
    settings = Settings()
    assert settings.data_path.parent.exists()


def test_log_directory_created() -> None:
    """Test that log directory is created on settings initialization."""
    settings = Settings()
    assert settings.log_file.parent.exists()
