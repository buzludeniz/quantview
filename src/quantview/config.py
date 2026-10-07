"""QuantView Configuration Management.

This module provides a Pydantic Settings-based configuration system that loads
from config.yaml and allows environment variable overrides with the prefix
QUANTVIEW_.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class EmailSMTPConfig(BaseSettings):
    """SMTP configuration for email alerts."""

    model_config = SettingsConfigDict(env_prefix="QUANTVIEW_EMAIL_SMTP_")

    host: str = "smtp.gmail.com"
    port: int = 587
    username: str = ""
    password: str = ""
    from_address: str = ""
    to_addresses: list[str] = Field(default_factory=list)
    use_tls: bool = True


class AlertThresholdsConfig(BaseSettings):
    """Alert threshold configuration."""

    model_config = SettingsConfigDict(env_prefix="QUANTVIEW_ALERT_THRESHOLDS_")

    price_change_pct: float = 2.0
    volume_spike_ratio: float = 3.0
    iv_rank_change: float = 10.0


class Settings(BaseSettings):
    """Main application settings.

    Loads from config.yaml and environment variables with prefix QUANTVIEW_.
    Environment variables take precedence over config file values.
    """

    model_config = SettingsConfigDict(
        env_prefix="QUANTVIEW_",
        env_nested_delimiter="__",
        yaml_file="config.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    # Data storage
    data_path: Path = Path("./data/quantview.duckdb")

    # Polling interval for background data ingestion (seconds)
    polling_interval_seconds: int = 60

    # Default symbols to track
    symbols: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"])

    # Alert thresholds
    alert_thresholds: AlertThresholdsConfig = Field(default_factory=AlertThresholdsConfig)

    # Email/SMTP configuration
    email_smtp: EmailSMTPConfig = Field(default_factory=EmailSMTPConfig)

    # FRED API key
    fred_api_key: str = ""

    # Alpha Vantage API key
    alphavantage_api_key: str = ""

    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("./logs/quantview.log")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings sources to include YAML config file.

        Priority order (highest to lowest):
        1. init_settings (explicit arguments)
        2. env_settings (environment variables)
        3. YAML config file
        4. dotenv_settings (.env file)
        5. file_secret_settings
        """
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )

    def model_post_init(self, __context: object) -> None:
        # Ensure data directory exists
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance.

    Returns:
        Settings: The global settings object.
    """
    return settings


def reload_settings() -> Settings:
    """Reload settings from config file and environment.

    Returns:
        Settings: A new settings instance with fresh values.
    """
    global settings
    settings = Settings()
    return settings
