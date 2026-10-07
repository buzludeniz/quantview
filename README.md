# QuantView

**Local-First Quant Research Desktop Application**

A Python-based quantitative research platform for equity and options analysis with local data persistence via DuckDB. Built for researchers who want full control over their data and analytics pipeline.

## Features

- **Local-First Architecture**: All data stored locally in DuckDB - no cloud dependencies
- **Multi-Asset Coverage**: Equities, options chains, futures, and macro/fundamental data
- **Real-Time & Historical**: Live polling with configurable intervals + historical backfill
- **Greeks & IV Analytics**: Black-Scholes Greeks, IV surface modeling, term structure
- **Desktop GUI**: PyQt6-based interface with pyqtgraph/Plotly visualizations
- **Extensible**: Plugin architecture for custom data sources and strategies
- **Alerting**: Configurable price/volume/IV alerts with email notifications

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd quantview

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Configuration

Copy the example config and adjust as needed:

```bash
# config.yaml is already included with defaults
# Override via environment variables (prefix: QUANTVIEW_)
export QUANTVIEW_DATA_PATH="./my_data/quantview.duckdb"
export QUANTVIEW_FRED_API_KEY="your_fred_api_key"
export QUANTVIEW_EMAIL_SMTP__USERNAME="your_email@gmail.com"
export QUANTVIEW_EMAIL_SMTP__PASSWORD="your_app_password"
```

### Running

**Desktop Application:**
```bash
python -m quantview.desktop
```

**Jupyter Lab (for research):**
```bash
jupyter lab
```

**Headless Data Ingestion:**
```bash
python -m quantview.data init_schema   # create the DuckDB schema
python -m quantview.data verify        # check it
python -m quantview.data daily         # end-of-day backfill
python -m quantview.data intraday      # 5-minute snapshot
python -m quantview.data scheduler     # long-running scheduler
```

## Project Structure

```
quantview/
|-- config.yaml                 # runtime configuration
|-- pyproject.toml              # packaging, lint, type-check and test config
|-- src/quantview/
|   |-- config.py               # settings, with QUANTVIEW_ env overrides
|   |-- analytics/              # curves, portfolio attribution, risk, surfaces
|   |-- data/
|   |   |-- connection.py       # DuckDB connection management
|   |   |-- schema.py           # DDL
|   |   |-- universe.py         # symbol universe
|   |   |-- ingest.py           # ingestion CLI and pipeline
|   |   `-- connectors/         # Yahoo, FRED, CBOE, Alpha Vantage
|   |-- notebook/               # Plotly figures, ipywidgets controls, export
|   `-- desktop/                # PyQt6 charts, monitor, application
|-- scripts/                    # thin wrappers around the ingestion CLI
|-- notebooks/                  # demo notebooks
`-- tests/
```

## Configuration Reference

| Setting | Description | Default |
|---------|-------------|---------|
| `data_path` | DuckDB database file path | `./data/quantview.duckdb` |
| `polling_interval_seconds` | Background polling interval | `60` |
| `symbols` | Default symbols to track | `["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]` |
| `alert_thresholds.price_change_pct` | Price change alert % | `2.0` |
| `alert_thresholds.volume_spike_ratio` | Volume spike ratio | `3.0` |
| `email_smtp.*` | SMTP configuration for alerts | See config.yaml |
| `fred_api_key` | FRED API key for macro data | `""` |

Environment variables override config.yaml values with prefix `QUANTVIEW_` and double underscore for nesting:
```bash
QUANTVIEW_DATA_PATH=/custom/path.duckdb
QUANTVIEW_ALERT_THRESHOLDS__PRICE_CHANGE_PCT=3.0
```

## Development

### Code Quality

```bash
# Linting
ruff check .

# Formatting
ruff format .

# Type checking
mypy src/quantview

# Run all checks
ruff check . && ruff format --check . && mypy src/quantview
```

### Testing

```bash
# Run tests
pytest

# With the coverage gate (the same command CI runs)
pytest --cov=quantview --cov-fail-under=85

# A single file
pytest tests/test_analytics.py

# Collect only (dry run)
pytest --collect-only
```

The three tests that skip need live API keys (FRED, Alpha Vantage); everything
else runs offline.

### Adding Dependencies

```bash
# Production dependency
uv add <package>

# Development dependency
uv add --dev <package>
```

## Data Sources

- **Yahoo Finance** (yfinance): Equity prices, options chains, fundamentals
- **FRED** (fredapi): Macroeconomic time series
- **Custom**: Extensible plugin system for additional sources

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run quality checks, which is what CI runs:
   `ruff check . && ruff format --check . && mypy src/quantview && pytest --cov=quantview --cov-fail-under=85`
5. Submit a pull request

---

**Note**: This is an early-stage project. APIs and schema may change. Use with appropriate caution for production trading decisions.