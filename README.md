# QuantView

**Local-First Quant Research Desktop Application**

A Python-based quantitative research platform for equity and options analysis with local data persistence via DuckDB. Built for researchers who want full control over their data and analytics pipeline.

## Features

- **Local-First Architecture**: All data stored locally in DuckDB - no cloud dependencies
- **Multi-Asset Coverage**: Equities, options chains, rates, and macro/fundamental data
- **Real-Time & Historical**: Live polling with configurable intervals + historical backfill
- **Greeks & IV Analytics**: Black-Scholes Greeks, IV surface modeling, term structure
- **Desktop GUI**: PyQt6-based interface with pyqtgraph/Plotly visualizations
- **Extensible**: Data connectors share a common base class (`data/connectors/base.py`) with caching and retry, so new sources slot in alongside the built-in ones
- **Alerting**: after every daily/intraday ingest, price-change and volume-spike thresholds are evaluated, fired alerts are stored in the `alerts` table and emailed over SMTP (see [Alerting](#alerting))

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

## Alerting

After each `daily` or `intraday` ingest QuantView evaluates the thresholds in `alert_thresholds` against the newest bar of every symbol in `raw.yahoo_equities`:

| Alert type | Fires when | Threshold |
|------------|------------|-----------|
| `price_change` | absolute close-to-close change of the latest bar is at least the threshold (%) | `price_change_pct` |
| `volume_spike` | latest volume is at least the threshold times the mean of the previous 20 bars (needs 5+ bars) | `volume_spike_ratio` |

A threshold of `0` (or less) disables that check. `iv_rank_change` is accepted in the configuration but not yet evaluated, because no IV-rank series is stored.

An alert fires once when the condition becomes true and not again while it stays true; it re-arms after a bar where the condition does not hold. Re-running an ingest on the same data never repeats an alert. Severity is `critical` at twice the threshold or more, otherwise `warning`. Alerts are rows in the `alerts` table (`metadata` JSON holds `bar_date`, `value`, `threshold`).

Fired alerts from one ingest are sent as a single email. Configure SMTP in `config.yaml` or the environment, and keep credentials out of the repository:

```bash
export QUANTVIEW_EMAIL_SMTP__HOST="smtp.example.com"
export QUANTVIEW_EMAIL_SMTP__PORT=587
export QUANTVIEW_EMAIL_SMTP__USE_TLS=true          # STARTTLS; set false for a plain local relay
export QUANTVIEW_EMAIL_SMTP__USERNAME="user"       # optional; login only if set
export QUANTVIEW_EMAIL_SMTP__PASSWORD="app_password"
export QUANTVIEW_EMAIL_SMTP__FROM_ADDRESS="quantview@example.com"
export QUANTVIEW_EMAIL_SMTP__TO_ADDRESSES='["me@example.com"]'
```

If the sender or recipients are not set, alerts are still stored and delivery is recorded as `skipped`. A delivery error (refused connection, TLS or login failure, ...) never fails the ingest: it is logged and stored as `metadata.delivery` (`{"status": "failed", "error": ...}`) on the affected alerts, which are not retried.

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
- **Alpha Vantage**: Equities and fundamentals
- **CBOE**: Option chains with Greeks
- **Custom**: Subclass `BaseConnector` to add another source

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