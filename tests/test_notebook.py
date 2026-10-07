"""
Tests for QuantView Notebook Charts Module.

Run with: pytest tests/test_notebook.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from quantview.notebook.charts import (
    THEME,
    ChartTheme,
    NSSFit,
    YieldCurveData,
    candlestick_chart,
    create_options_chain_grid,
    iv_smile_chart,
    options_chain_greeks,
    term_structure_chart,
    vol_surface_2d,
    vol_surface_3d,
    yield_curve_carry_roll,
    yield_curve_chart,
)
from quantview.notebook.export import (
    estimate_html_size,
    export_html,
    optimize_figure_size,
)

# ============================================================================
# Black-Scholes Availability Check
# ============================================================================

try:
    from black_scholes import OptionParams, black_scholes_price
    from black_scholes.surface import VolSurface

    HAS_BLACK_SCHOLES = True
except ImportError:
    HAS_BLACK_SCHOLES = False


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    returns = np.random.normal(0.0005, 0.015, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))

    noise = np.random.uniform(0.995, 1.005, (len(dates), 4))
    df = pd.DataFrame(
        {
            "open": prices * noise[:, 0],
            "high": prices * noise[:, 1],
            "low": prices * noise[:, 2],
            "close": prices * noise[:, 3],
            "volume": np.random.lognormal(13, 0.5, len(dates)).astype(int),
        },
        index=dates,
    )

    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


@pytest.fixture
def sample_yield_curve():
    """Generate sample yield curve data."""
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    spot = np.array([0.052, 0.050, 0.048, 0.046, 0.045, 0.044, 0.044, 0.0445, 0.0455, 0.046])
    forward = np.array([0.052, 0.049, 0.046, 0.044, 0.0435, 0.043, 0.0435, 0.0445, 0.046, 0.047])
    par = np.array([0.052, 0.050, 0.048, 0.046, 0.045, 0.044, 0.044, 0.0445, 0.0455, 0.046])
    return YieldCurveData(tenors=tenors, spot_rates=spot, forward_rates=forward, par_rates=par)


@pytest.fixture
def sample_nss_fit():
    """Generate sample NSS fit."""
    return NSSFit(beta0=0.045, beta1=-0.01, beta2=-0.015, beta3=0.005, tau1=1.5, tau2=8.0)


@pytest.fixture
def sample_vol_surface():
    """Generate sample VolSurface for testing."""
    if not HAS_BLACK_SCHOLES:
        pytest.skip("black_scholes not available")
    strikes = np.array([90, 95, 100, 105, 110])
    vols = np.array([0.25, 0.23, 0.22, 0.23, 0.25])
    spot = 100.0
    T = 0.25
    r = 0.05

    prices = [
        black_scholes_price(OptionParams(spot, k, T, r, v))
        for k, v in zip(strikes, vols, strict=False)
    ]
    surface = VolSurface.from_market_prices(spot, strikes, prices, T, r)
    return surface


@pytest.fixture
def sample_svi_fit(sample_vol_surface):
    """Generate sample SVIFit for testing."""
    if not HAS_BLACK_SCHOLES:
        pytest.skip("black_scholes not available")
    return sample_vol_surface.fit()


@pytest.fixture
def sample_surfaces_dict():
    """Generate dict of surfaces for multiple expiries."""
    if not HAS_BLACK_SCHOLES:
        pytest.skip("black_scholes not available")
    surfaces = {}
    spot = 100.0
    r = 0.05
    base_strikes = np.array([90, 95, 100, 105, 110])
    base_vols = np.array([0.25, 0.23, 0.22, 0.23, 0.25])

    for T in [0.25, 0.5, 1.0, 2.0]:
        # Add term structure: longer expiry has slightly higher ATM vol
        vol_adj = 1.0 + 0.05 * np.sqrt(T)
        vols = base_vols * vol_adj
        prices = [
            black_scholes_price(OptionParams(spot, k, T, r, v))
            for k, v in zip(base_strikes, vols, strict=False)
        ]
        surface = VolSurface.from_market_prices(spot, base_strikes, prices, T, r)
        surfaces[T] = surface
    return surfaces


# ============================================================================
# IV Smile Chart Tests
# ============================================================================


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_iv_smile_chart_basic(sample_vol_surface, sample_svi_fit):
    """Test basic IV smile chart creation."""
    fig = iv_smile_chart(sample_vol_surface, expiry=0.25, svi_fit=sample_svi_fit)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2  # Market points + SVI curve


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_iv_smile_chart_no_svi(sample_vol_surface):
    """Test IV smile chart without SVI fit."""
    fig = iv_smile_chart(sample_vol_surface, expiry=0.25, svi_fit=None)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # Only market points


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_iv_smile_chart_confidence_band(sample_vol_surface, sample_svi_fit):
    """Test IV smile chart with confidence band."""
    fig = iv_smile_chart(
        sample_vol_surface,
        expiry=0.25,
        svi_fit=sample_svi_fit,
        show_confidence_band=True,
        confidence_level=0.95,
    )
    assert isinstance(fig, go.Figure)
    # Should have market points, SVI curve, and confidence band
    assert len(fig.data) >= 3


# ============================================================================
# Term Structure Chart Tests
# ============================================================================


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_term_structure_chart_basic(sample_surfaces_dict):
    """Test basic term structure chart."""
    fig = term_structure_chart(sample_surfaces_dict)
    assert isinstance(fig, go.Figure)
    # Should have 3 subplots (ATM, Skew, Curvature)
    assert len(fig.data) >= 3


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_term_structure_chart_atm_only(sample_surfaces_dict):
    """Test term structure chart with only ATM vol."""
    fig = term_structure_chart(
        sample_surfaces_dict,
        show_atm=True,
        show_skew=False,
        show_curvature=False,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_term_structure_chart_custom_delta(sample_surfaces_dict):
    """Test term structure chart with custom delta."""
    fig = term_structure_chart(sample_surfaces_dict, delta=0.10)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3


# ============================================================================
# Options Chain Grid Tests
# ============================================================================


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_create_options_chain_grid(sample_vol_surface):
    """Test options chain grid creation."""
    df, fig = create_options_chain_grid(sample_vol_surface, expiry=0.25)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Table)
    # Check required columns
    required_cols = [
        "Strike",
        "Moneyness",
        "IV",
        "Call Delta",
        "Call Gamma",
        "Call Vega",
        "Call Theta",
    ]
    for col in required_cols:
        assert col in df.columns


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_create_options_chain_grid_with_bid_ask(sample_vol_surface):
    """Test options chain grid with bid/ask data."""
    bid_ask_data = pd.DataFrame(
        {
            "strike": [90, 95, 100, 105, 110],
            "bid": [10.5, 7.2, 4.8, 3.1, 1.9],
            "ask": [10.7, 7.4, 5.0, 3.3, 2.1],
            "volume": [100, 200, 300, 150, 80],
            "open_interest": [500, 800, 1200, 600, 300],
        }
    )
    df, fig = create_options_chain_grid(
        sample_vol_surface,
        expiry=0.25,
        include_bid_ask=True,
        bid_ask_data=bid_ask_data,
    )
    # Both halves of the return value are part of the contract.
    assert fig is not None
    assert "Bid" in df.columns
    assert "Ask" in df.columns
    assert "Volume" in df.columns
    assert "Open Interest" in df.columns


# ============================================================================
# Analytics Surface Tests
# ============================================================================


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_chain_to_volsurface():
    """Test chain_to_volsurface function."""
    from quantview.analytics.surface import build_chain_dataframe, chain_to_volsurface

    chain = build_chain_dataframe(
        strikes=[90, 95, 100, 105, 110],
        bids=[10.5, 7.2, 4.8, 3.1, 1.9],
        asks=[10.7, 7.4, 5.0, 3.3, 2.1],
        volumes=[100, 200, 300, 150, 80],
        open_interests=[500, 800, 1200, 600, 300],
    )

    surface = chain_to_volsurface(spot=100.0, chain=chain, expiry=0.25, risk_free_rate=0.05)
    assert isinstance(surface, VolSurface)
    # Some strikes may be filtered if IV inversion fails
    assert len(surface.strikes) >= 2


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_compute_chain_greeks(sample_vol_surface):
    """Test compute_chain_greeks function."""
    from quantview.analytics.surface import compute_chain_greeks

    df = compute_chain_greeks(sample_vol_surface, expiry=0.25)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_vol_surface.strikes)
    assert "Call Delta" in df.columns
    assert "Put Delta" in df.columns
    assert "Call Gamma" in df.columns


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_compute_term_structure(sample_surfaces_dict):
    """Test compute_term_structure function."""
    from quantview.analytics.surface import compute_term_structure

    df = compute_term_structure(sample_surfaces_dict)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_surfaces_dict)
    assert "Expiry" in df.columns
    assert "ATM Vol" in df.columns
    assert "25d Skew" in df.columns
    assert "Curvature" in df.columns


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_compute_svi_parameters_per_expiry(sample_surfaces_dict):
    """Test compute_svi_parameters_per_expiry function."""
    from quantview.analytics.surface import compute_svi_parameters_per_expiry

    df = compute_svi_parameters_per_expiry(sample_surfaces_dict)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_surfaces_dict)
    assert "ATM Variance" in df.columns
    assert "ATM Skew" in df.columns
    assert "Put Wing Slope" in df.columns
    assert "Call Wing Slope" in df.columns


# ============================================================================
# Export Tests
# ============================================================================


def test_chart_theme_defaults():
    """Test ChartTheme default values."""
    theme = ChartTheme()
    assert theme.template == "plotly_dark"
    assert theme.primary_color == "#58a6ff"
    assert theme.success_color == "#3fb950"
    assert theme.danger_color == "#f85149"


def test_theme_layout_kwargs():
    """Test theme layout kwargs generation."""
    kwargs = THEME.layout_kwargs()
    assert "template" in kwargs
    assert "font" in kwargs
    assert kwargs["template"] == "plotly_dark"


# ============================================================================
# Candlestick Chart Tests
# ============================================================================


def test_candlestick_chart_basic(sample_ohlcv_data):
    """Test basic candlestick chart creation."""
    fig = candlestick_chart(sample_ohlcv_data, symbol="TEST")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1  # At least candlestick trace


def test_candlestick_chart_with_indicators(sample_ohlcv_data):
    """Test candlestick chart with indicators."""
    fig = candlestick_chart(
        sample_ohlcv_data,
        symbol="TEST",
        indicators=["sma_20", "sma_50", "bbands", "rsi", "macd"],
        volume=True,
    )
    assert isinstance(fig, go.Figure)
    # Should have: candlestick + 2 SMAs + 3 BBands + volume + RSI + MACD + signal + histogram
    assert len(fig.data) >= 8


def test_candlestick_chart_no_volume(sample_ohlcv_data):
    """Test candlestick chart without volume."""
    df_no_vol = sample_ohlcv_data.drop(columns=["volume"])
    fig = candlestick_chart(df_no_vol, volume=False)
    assert isinstance(fig, go.Figure)


def test_candlestick_chart_invalid_index():
    """Test candlestick chart with invalid index raises error."""
    df = pd.DataFrame(
        {
            "open": [100, 101],
            "high": [101, 102],
            "low": [99, 100],
            "close": [100, 101],
            "volume": [1000, 1000],
        }
    )
    with pytest.raises(ValueError, match="DatetimeIndex"):
        candlestick_chart(df)


def test_candlestick_chart_missing_columns():
    """Test candlestick chart with missing columns raises error."""
    df = pd.DataFrame(
        {"open": [100], "high": [101], "low": [99]}, index=pd.date_range("2024-01-01", periods=1)
    )
    with pytest.raises(ValueError, match="Missing required column"):
        candlestick_chart(df)


# ============================================================================
# Yield Curve Tests
# ============================================================================


def test_yield_curve_data_creation(sample_yield_curve):
    """Test YieldCurveData creation."""
    assert len(sample_yield_curve.tenors) == 10
    assert len(sample_yield_curve.spot_rates) == 10
    assert sample_yield_curve.forward_rates is not None
    assert sample_yield_curve.par_rates is not None


def test_nss_fit_spot_rate(sample_nss_fit):
    """Test NSS spot rate calculation."""
    t = np.array([1.0, 2.0, 5.0, 10.0])
    rates = sample_nss_fit.spot_rate(t)
    assert len(rates) == 4
    assert all(r > 0 for r in rates)


def test_nss_fit_forward_rate(sample_nss_fit):
    """Test NSS forward rate calculation."""
    t = np.array([1.0, 2.0, 5.0, 10.0])
    rates = sample_nss_fit.forward_rate(t)
    assert len(rates) == 4
    assert all(r > 0 for r in rates)


def test_nss_fit_discount_factor(sample_nss_fit):
    """Test NSS discount factor calculation."""
    t = np.array([1.0, 2.0, 5.0, 10.0])
    df = sample_nss_fit.discount_factor(t)
    assert len(df) == 4
    assert all(0 < d < 1 for d in df)


def test_yield_curve_chart_basic(sample_yield_curve):
    """Test basic yield curve chart."""
    fig = yield_curve_chart(sample_yield_curve)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_yield_curve_chart_with_nss(sample_yield_curve, sample_nss_fit):
    """Test yield curve chart with NSS fit."""
    fig = yield_curve_chart(sample_yield_curve, nss_fit=sample_nss_fit)
    assert isinstance(fig, go.Figure)
    # Should have spot, forward, par, NSS spot, NSS forward
    assert len(fig.data) >= 5


def test_yield_curve_carry_roll(sample_yield_curve):
    """Test carry & roll chart."""
    fig = yield_curve_carry_roll(
        sample_yield_curve.tenors,
        sample_yield_curve.spot_rates,
        sample_yield_curve.forward_rates,
        horizon=1.0,
    )
    assert isinstance(fig, go.Figure)
    # Should have carry bar, roll bar, total line
    assert len(fig.data) == 3


# ============================================================================
# Volatility Surface Tests (require black_scholes)
# ============================================================================


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_vol_surface_creation():
    """Test VolSurface creation from market prices."""
    strikes = np.array([90, 95, 100, 105, 110])
    vols = np.array([0.25, 0.23, 0.22, 0.23, 0.25])
    spot = 100.0
    T = 0.25
    r = 0.05

    prices = [
        black_scholes_price(OptionParams(spot, k, T, r, v))
        for k, v in zip(strikes, vols, strict=False)
    ]
    surface = VolSurface.from_market_prices(spot, strikes, prices, T, r)

    assert len(surface.strikes) == 5
    assert abs(surface.implied_volatility(100) - 0.22) < 0.01


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_vol_surface_fit():
    """Test SVI fitting."""
    strikes = np.array([90, 95, 100, 105, 110])
    vols = np.array([0.25, 0.23, 0.22, 0.23, 0.25])
    spot = 100.0
    T = 0.25
    r = 0.05

    prices = [
        black_scholes_price(OptionParams(spot, k, T, r, v))
        for k, v in zip(strikes, vols, strict=False)
    ]
    surface = VolSurface.from_market_prices(spot, strikes, prices, T, r)

    fit = surface.fit()
    assert fit.rms_error < 0.01
    assert fit.curve.parameters_are_valid()


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_vol_surface_3d():
    """Test 3D vol surface chart."""
    strikes = np.array([90, 95, 100, 105, 110])
    vols = np.array([0.25, 0.23, 0.22, 0.23, 0.25])
    spot = 100.0
    T = 0.25
    r = 0.05

    prices = [
        black_scholes_price(OptionParams(spot, k, T, r, v))
        for k, v in zip(strikes, vols, strict=False)
    ]
    surface = VolSurface.from_market_prices(spot, strikes, prices, T, r)
    fit = surface.fit()

    fig = vol_surface_3d(surface, fit)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2  # Market points + surface


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_vol_surface_2d():
    """Test 2D vol smile chart."""
    strikes = np.array([90, 95, 100, 105, 110])
    vols = np.array([0.25, 0.23, 0.22, 0.23, 0.25])
    spot = 100.0
    T = 0.25
    r = 0.05

    prices = [
        black_scholes_price(OptionParams(spot, k, T, r, v))
        for k, v in zip(strikes, vols, strict=False)
    ]
    surface = VolSurface.from_market_prices(spot, strikes, prices, T, r)
    fit = surface.fit()

    fig = vol_surface_2d(surface, fit)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2  # Market points + SVI curve


# ============================================================================
# Options Chain Tests
# ============================================================================


@pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")
def test_options_chain_greeks():
    """Test options chain Greeks table."""
    strikes = np.arange(80, 131, 5)
    vols = 0.23 + 0.1 * (strikes - 100) ** 2 / 10000
    surface = VolSurface(
        spot=100.0,
        strikes=tuple(strikes),
        implied_vols=tuple(vols),
        time_to_maturity=0.25,
        risk_free_rate=0.05,
        dividend_yield=0.0,
    )

    fig = options_chain_greeks(surface, expiry=0.25)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Table)


# ============================================================================
# Export Tests
# ============================================================================


def test_export_html(tmp_path, sample_ohlcv_data):
    """Test HTML export."""
    fig = candlestick_chart(sample_ohlcv_data, symbol="TEST")
    path = tmp_path / "test.html"
    export_html(fig, path)
    assert path.exists()
    content = path.read_text()
    assert "plotly" in content.lower()
    assert "TEST" in content or "Candlestick" in content


def test_export_html_compressed(tmp_path, sample_ohlcv_data):
    """Test compressed HTML export."""
    fig = candlestick_chart(sample_ohlcv_data)
    path = tmp_path / "test.html.gz"
    export_html(fig, path, compress=True)
    assert path.exists()


def test_estimate_html_size(sample_ohlcv_data):
    """Test HTML size estimation."""
    fig = candlestick_chart(sample_ohlcv_data)
    size = estimate_html_size(fig)
    assert size > 1000  # At least 1KB


def test_optimize_figure_size(sample_ohlcv_data):
    """Test figure optimization."""
    fig = candlestick_chart(sample_ohlcv_data)
    # Add a large trace
    fig.add_trace(go.Scatter(x=np.arange(10000), y=np.random.randn(10000)))

    optimized = optimize_figure_size(fig, max_points=1000)

    # Find the trace that was added above and confirm it was actually reduced.
    # The previous version looped with a bare break, so if no trace matched the
    # shape the test passed without asserting anything.
    lengths = [len(tr.x) for tr in optimized.data if getattr(tr, "x", None) is not None]
    assert lengths, "the optimised figure should still carry x data"
    assert max(lengths) <= 1000, f"expected every series downsampled, got {lengths}"


# ============================================================================
# Integration Tests
# ============================================================================


def test_all_charts_render_without_error(sample_ohlcv_data, sample_yield_curve, sample_nss_fit):
    """Integration test: all chart types render without error."""
    charts = [
        candlestick_chart(sample_ohlcv_data, "TEST"),
        yield_curve_chart(sample_yield_curve, sample_nss_fit),
        yield_curve_carry_roll(
            sample_yield_curve.tenors,
            sample_yield_curve.spot_rates,
            sample_yield_curve.forward_rates,
        ),
    ]

    for fig in charts:
        assert isinstance(fig, go.Figure)
        # Verify layout has theme applied
        assert fig.layout.paper_bgcolor == THEME.paper_bgcolor
        assert fig.layout.plot_bgcolor == THEME.plot_bgcolor


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
