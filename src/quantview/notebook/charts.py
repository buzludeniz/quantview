"""
QuantView Notebook Charts Module.

Interactive Plotly figures for equity candles, volatility surfaces,
yield curves, and options analytics. All figures are exportable to
HTML, PNG, and SVG formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional Black-Scholes for SVI integration. Declared up front so the names are
# known to be optional: without the extra they bind None and the SVI paths
# Explicitly bind to None so the names exist in the module namespace when
# the optional dependency is absent. The try/except rebinds on success.
SVICurve: Any = None
SVIFit: Any = None
VolSurface: Any = None
black_scholes_price: Any = None
OptionParams: Any = None
try:
    from black_scholes import (
        OptionParams,
        black_scholes_price,
    )
    from black_scholes.surface import (  # noqa: F401 - probed for availability, used only as a guard
        SVICurve,
        SVIFit,
        VolSurface,
    )
except ImportError:
    pass


@dataclass(frozen=True, slots=True)
class ChartTheme:
    """Consistent styling across all QuantView charts."""

    template: str = "plotly_dark"
    font_family: str = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
    font_size: int = 12
    title_font_size: int = 16
    grid_color: str = "rgba(255, 255, 255, 0.08)"
    paper_bgcolor: str = "#0d1117"
    plot_bgcolor: str = "#0d1117"
    primary_color: str = "#58a6ff"
    secondary_color: str = "#a371f7"
    success_color: str = "#3fb950"
    danger_color: str = "#f85149"
    warning_color: str = "#d29922"

    def layout_kwargs(self) -> dict:
        return {
            "template": self.template,
            "font": {"family": self.font_family, "size": self.font_size, "color": "#e6edf3"},
            "title": {"font": {"size": self.title_font_size, "color": "#f0f6fc"}},
            "paper_bgcolor": self.paper_bgcolor,
            "plot_bgcolor": self.plot_bgcolor,
            "xaxis": {
                "gridcolor": self.grid_color,
                "zerolinecolor": self.grid_color,
                "showgrid": True,
                "showspikes": True,
                "spikecolor": self.primary_color,
                "spikethickness": 1,
                "spikedash": "dot",
            },
            "yaxis": {
                "gridcolor": self.grid_color,
                "zerolinecolor": self.grid_color,
                "showgrid": True,
                "showspikes": True,
                "spikecolor": self.primary_color,
                "spikethickness": 1,
                "spikedash": "dot",
            },
            "legend": {
                "bgcolor": "rgba(13, 17, 23, 0.8)",
                "bordercolor": "rgba(255, 255, 255, 0.1)",
                "borderwidth": 1,
                "font": {"size": 11},
            },
            "hovermode": "x unified",
            "spikedistance": -1,
            "hoverdistance": 100,
        }


THEME = ChartTheme()


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply QuantView theme to a figure."""
    fig.update_layout(**THEME.layout_kwargs())
    return fig


def _add_range_selector(fig: go.Figure, xaxis: str = "xaxis") -> go.Figure:
    """Add range selector buttons to x-axis."""
    fig.update_layout(
        **{
            xaxis: {
                "rangeselector": {
                    "buttons": [
                        {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                        {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                        {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                        {"count": 1, "label": "YTD", "step": "year", "stepmode": "todate"},
                        {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                        {"step": "all", "label": "All"},
                    ],
                    "bgcolor": THEME.paper_bgcolor,
                    "activecolor": THEME.primary_color,
                    "bordercolor": THEME.grid_color,
                    "font": {"color": "#e6edf3", "size": 11},
                },
                "rangeslider": {"visible": False},
                "type": "date",
            }
        }
    )
    return fig


# ============================================================================
# Candlestick Chart with Indicators
# ============================================================================


def candlestick_chart(
    df: pd.DataFrame,
    symbol: str = "",
    indicators: list[str] | None = None,
    volume: bool = True,
    height: int = 800,
) -> go.Figure:
    """
    Create an interactive candlestick chart with technical indicators.

    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            Index must be DatetimeIndex.
        symbol: Ticker symbol for title.
        indicators: List of indicators to overlay. Supported:
            - 'sma_20', 'sma_50', 'sma_200' - Simple Moving Averages
            - 'ema_12', 'ema_26', 'ema_50' - Exponential Moving Averages
            - 'bbands' - Bollinger Bands (20, 2)
            - 'rsi' - Relative Strength Index (14)
            - 'macd' - MACD (12, 26, 9)
        volume: Whether to show volume subplot.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with candlesticks, volume, and requested indicators.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    required_cols = ["open", "high", "low", "close"]
    if volume:
        required_cols.append("volume")
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    indicators = indicators or []
    n_rows = (
        1
        + (1 if volume else 0)
        + (1 if "rsi" in indicators else 0)
        + (1 if "macd" in indicators else 0)
    )
    row_heights = []

    row_heights = [0.55, 0.15] if volume else [0.7]

    if "rsi" in indicators:
        row_heights.append(0.15)
    if "macd" in indicators:
        row_heights.append(0.15)

    # Normalize row heights to sum to 1
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    subplot_titles = []
    if symbol:
        subplot_titles.append(f"{symbol} — OHLC")
    else:
        subplot_titles.append("OHLC")
    if volume:
        subplot_titles.append("Volume")
    if "rsi" in indicators:
        subplot_titles.append("RSI (14)")
    if "macd" in indicators:
        subplot_titles.append("MACD (12, 26, 9)")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Row 1: Candlesticks
    row = 1
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color=THEME.success_color,
            decreasing_line_color=THEME.danger_color,
            increasing_fillcolor=THEME.success_color,
            decreasing_fillcolor=THEME.danger_color,
            whiskerwidth=0.8,
        ),
        row=row,
        col=1,
    )

    # Add moving averages
    ma_colors = {
        "sma_20": THEME.primary_color,
        "sma_50": THEME.secondary_color,
        "sma_200": THEME.warning_color,
        "ema_12": "#ff7b72",
        "ema_26": "#7ee787",
        "ema_50": "#ffa657",
    }

    for ind in indicators:
        if ind in ma_colors:
            period = int(ind.split("_")[1])
            col_name = f"{ind.upper()}"
            if ind.startswith("sma"):
                df[col_name] = df["close"].rolling(period).mean()
            elif ind.startswith("ema"):
                df[col_name] = df["close"].ewm(span=period, adjust=False).mean()

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col_name],
                    name=col_name,
                    line={"color": ma_colors[ind], "width": 1.5},
                    hovertemplate="%{y:.2f}<extra></extra>",
                ),
                row=row,
                col=1,
            )

    # Bollinger Bands
    if "bbands" in indicators:
        period = 20
        std_mult = 2
        df["BB_MID"] = df["close"].rolling(period).mean()
        df["BB_STD"] = df["close"].rolling(period).std()
        df["BB_UPPER"] = df["BB_MID"] + std_mult * df["BB_STD"]
        df["BB_LOWER"] = df["BB_MID"] - std_mult * df["BB_STD"]

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_UPPER"],
                name="BB Upper",
                line={"color": THEME.primary_color, "width": 1, "dash": "dash"},
                hovertemplate="Upper: %{y:.2f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_LOWER"],
                name="BB Lower",
                line={"color": THEME.primary_color, "width": 1, "dash": "dash"},
                fill="tonexty",
                fillcolor="rgba(88, 166, 255, 0.1)",
                hovertemplate="Lower: %{y:.2f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_MID"],
                name="BB Mid (SMA 20)",
                line={"color": THEME.primary_color, "width": 1.5},
                hovertemplate="Mid: %{y:.2f}<extra></extra>",
            ),
            row=row,
            col=1,
        )

    # Volume subplot
    if volume:
        row += 1
        colors = np.where(df["close"] >= df["open"], THEME.success_color, THEME.danger_color)
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
                hovertemplate="Volume: %{y:,.0f}<extra></extra>",
            ),
            row=row,
            col=1,
        )

    # RSI subplot
    if "rsi" in indicators:
        row += 1
        period = 14
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                name="RSI (14)",
                line={"color": THEME.secondary_color, "width": 1.5},
                hovertemplate="RSI: %{y:.1f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        # Overbought/oversold lines
        fig.add_hline(
            y=70, line_dash="dash", line_color=THEME.danger_color, line_width=1, row=row, col=1
        )
        fig.add_hline(
            y=30, line_dash="dash", line_color=THEME.success_color, line_width=1, row=row, col=1
        )
        fig.add_hline(
            y=50, line_dash="dot", line_color=THEME.grid_color, line_width=1, row=row, col=1
        )
        fig.update_yaxes(range=[0, 100], row=row, col=1)

    # MACD subplot
    if "macd" in indicators:
        row += 1
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema_12 - ema_26
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                name="MACD",
                line={"color": THEME.primary_color, "width": 1.5},
                hovertemplate="MACD: %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_SIGNAL"],
                name="Signal",
                line={"color": THEME.danger_color, "width": 1.5},
                hovertemplate="Signal: %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        colors = np.where(df["MACD_HIST"] >= 0, THEME.success_color, THEME.danger_color)
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["MACD_HIST"],
                name="Histogram",
                marker_color=colors,
                opacity=0.7,
                hovertemplate="Hist: %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=1,
        )

    # Layout
    fig.update_layout(
        title=f"{symbol} — Technical Analysis" if symbol else "Technical Analysis",
        xaxis_rangeslider_visible=False,
        height=height,
        showlegend=True,
        dragmode="pan",
    )

    _apply_theme(fig)
    _add_range_selector(fig)

    # Hide x-axis labels on all but bottom subplot
    for r in range(1, n_rows):
        fig.update_xaxes(showticklabels=False, row=r, col=1)

    return fig


# ============================================================================
# Volatility Surface 3D
# ============================================================================


def vol_surface_3d(
    surface: VolSurface,
    svi_fit: SVIFit | None = None,
    title: str = "Volatility Surface",
    height: int = 700,
    colorscale: str = "Viridis",
    show_wireframe: bool = True,
    show_contours: bool = True,
) -> go.Figure:
    """
    Create a 3D WebGL volatility surface with optional SVI wireframe overlay.

    Args:
        surface: VolSurface instance with market quotes.
        svi_fit: Optional SVIFit for smooth wireframe overlay.
        title: Chart title.
        height: Figure height in pixels.
        colorscale: Plotly colorscale for surface.
        show_wireframe: Show SVI wireframe if fit provided.
        show_contours: Show contour lines on surface.

    Returns:
        Plotly Figure with 3D surface.
    """
    if VolSurface is None:
        raise ImportError("black_scholes package required for vol_surface_3d")

    strikes = np.array(surface.strikes)
    vols = np.array(surface.implied_vols)
    T = surface.time_to_maturity
    F = surface.forward

    # Create mesh grid for smooth surface
    k_min, k_max = np.log(strikes.min() / F), np.log(strikes.max() / F)
    k_grid = np.linspace(k_min, k_max, 80)
    strike_grid = F * np.exp(k_grid)

    # Interpolate market vols onto grid
    from scipy.interpolate import interp1d

    interp = interp1d(np.log(strikes / F), vols, kind="cubic", fill_value="extrapolate")
    vol_grid = interp(k_grid)

    # SVI wireframe
    svi_vols = None
    if svi_fit is not None and show_wireframe:
        svi_vols = np.array([svi_fit.curve.implied_volatility(k, T) for k in k_grid])

    fig = go.Figure()

    # Market surface (scatter for actual points)
    fig.add_trace(
        go.Scatter3d(
            x=strikes,
            y=[T] * len(strikes),
            z=vols,
            mode="markers",
            name="Market Quotes",
            marker={
                "size": 6,
                "color": vols,
                "colorscale": colorscale,
                "showscale": True,
                "colorbar": {"title": "IV", "thickness": 15, "len": 0.7},
                "opacity": 0.9,
            },
            hovertemplate="Strike: %{x:.2f}<br>Expiry: %{y:.3f}Y<br>IV: %{z:.2%}<extra></extra>",
        )
    )

    # Smooth surface mesh
    X, Y = np.meshgrid(strike_grid, np.linspace(max(0.01, T - 0.5), T + 0.5, 30))
    Z = np.tile(vol_grid, (30, 1)).T

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale=colorscale,
            opacity=0.7,
            showscale=False,
            contours={
                "z": {
                    "show": show_contours,
                    "usecolormap": True,
                    "highlightcolor": "white",
                    "project": {"z": True},
                }
            }
            if show_contours
            else {},
            name="Interpolated Surface",
            hovertemplate="Strike: %{x:.2f}<br>Expiry: %{y:.3f}Y<br>IV: %{z:.2%}<extra></extra>",
        )
    )

    # SVI wireframe
    if svi_vols is not None:
        X_svi, Y_svi = np.meshgrid(strike_grid, np.linspace(max(0.01, T - 0.5), T + 0.5, 30))
        Z_svi = np.tile(svi_vols, (30, 1)).T

        fig.add_trace(
            go.Surface(
                x=X_svi,
                y=Y_svi,
                z=Z_svi,
                colorscale=[[0, THEME.secondary_color], [1, THEME.secondary_color]],
                opacity=0.4,
                showscale=False,
                surfacecolor=np.ones_like(Z_svi),
                name="SVI Fit",
                hovertemplate="Strike: %{x:.2f}<br>Expiry: %{y:.3f}Y<br>SVI IV: %{z:.2%}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "Strike",
            "yaxis_title": "Time to Expiry (Years)",
            "zaxis_title": "Implied Volatility",
            "xaxis": {"gridcolor": THEME.grid_color, "backgroundcolor": THEME.plot_bgcolor},
            "yaxis": {"gridcolor": THEME.grid_color, "backgroundcolor": THEME.plot_bgcolor},
            "zaxis": {"gridcolor": THEME.grid_color, "backgroundcolor": THEME.plot_bgcolor},
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2}},
        },
        height=height,
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    _apply_theme(fig)
    return fig


# ============================================================================
# Volatility Surface 2D Heatmap + SVI
# ============================================================================


def vol_surface_2d(
    surface: VolSurface,
    svi_fit: SVIFit | None = None,
    title: str = "Volatility Smile",
    height: int = 500,
    show_residuals: bool = True,
) -> go.Figure:
    """
    Create a 2D volatility smile heatmap with SVI curve and residuals.

    Args:
        surface: VolSurface instance with market quotes.
        svi_fit: Optional SVIFit for smooth curve overlay.
        title: Chart title.
        height: Figure height in pixels.
        show_residuals: Show residuals subplot if SVI fit provided.

    Returns:
        Plotly Figure with 2D smile + SVI + residuals.
    """
    if VolSurface is None:
        raise ImportError("black_scholes package required for vol_surface_2d")

    strikes = np.array(surface.strikes)
    vols = np.array(surface.implied_vols)
    T = surface.time_to_maturity
    F = surface.forward
    moneyness = strikes / F

    n_rows = 2 if (svi_fit is not None and show_residuals) else 1
    row_heights = [0.7, 0.3] if n_rows == 2 else [1.0]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=["Implied Volatility Smile", "Residuals (Market - SVI)"]
        if n_rows == 2
        else ["Implied Volatility Smile"],
    )

    # Market points
    fig.add_trace(
        go.Scatter(
            x=moneyness,
            y=vols,
            mode="markers",
            name="Market Quotes",
            marker={
                "size": 10,
                "color": THEME.primary_color,
                "symbol": "circle",
                "line": {"width": 1, "color": "white"},
            },
            hovertemplate="Moneyness: %{x:.3f}<br>IV: %{y:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # SVI curve
    if svi_fit is not None:
        k_fine = np.linspace(np.log(moneyness.min()), np.log(moneyness.max()), 200)
        strike_fine = F * np.exp(k_fine)
        svi_vols = np.array([svi_fit.curve.implied_volatility(k, T) for k in k_fine])
        moneyness_fine = strike_fine / F

        fig.add_trace(
            go.Scatter(
                x=moneyness_fine,
                y=svi_vols,
                mode="lines",
                name="SVI Fit",
                line={"color": THEME.secondary_color, "width": 2.5},
                hovertemplate="Moneyness: %{x:.3f}<br>SVI IV: %{y:.2%}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # ATM line
        fig.add_vline(
            x=1.0, line_dash="dot", line_color=THEME.grid_color, line_width=1, row=1, col=1
        )

        # Residuals
        if show_residuals:
            model_vols = np.array(svi_fit.model_vols)
            residuals = vols - model_vols

            colors = np.where(residuals >= 0, THEME.success_color, THEME.danger_color)
            fig.add_trace(
                go.Bar(
                    x=moneyness,
                    y=residuals,
                    name="Residuals",
                    marker_color=colors,
                    opacity=0.7,
                    hovertemplate="Moneyness: %{x:.3f}<br>Residual: %{y:.4f}<extra></extra>",
                ),
                row=2,
                col=1,
            )
            fig.add_hline(y=0, line_color=THEME.grid_color, line_width=1, row=2, col=1)

    fig.update_layout(
        title=f"{title} (T={T:.3f}Y, F={F:.2f})",
        height=height,
        showlegend=True,
        hovermode="x unified",
    )

    _apply_theme(fig)
    fig.update_xaxes(title_text="Moneyness (K/F)", row=n_rows, col=1)
    fig.update_yaxes(title_text="Implied Volatility", row=1, col=1, tickformat=".1%")
    if n_rows == 2:
        fig.update_yaxes(title_text="Residual", row=2, col=1)

    return fig


# ============================================================================
# Yield Curve Charts
# ============================================================================


@dataclass(frozen=True, slots=True)
class YieldCurveData:
    """Container for yield curve data."""

    tenors: np.ndarray  # in years
    spot_rates: np.ndarray  # continuously compounded
    forward_rates: np.ndarray | None = None
    par_rates: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class NSSFit:
    """Nelson-Siegel-Svensson fit parameters."""

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float

    def spot_rate(self, t: np.ndarray) -> np.ndarray:
        """Compute spot rates from NSS parameters."""
        t = np.asarray(t, dtype=float)
        tau1, tau2 = self.tau1, self.tau2

        term1 = self.beta0
        term2 = self.beta1 * (1 - np.exp(-t / tau1)) / (t / tau1)
        term3 = self.beta2 * ((1 - np.exp(-t / tau1)) / (t / tau1) - np.exp(-t / tau1))
        term4 = self.beta3 * ((1 - np.exp(-t / tau2)) / (t / tau2) - np.exp(-t / tau2))

        return term1 + term2 + term3 + term4

    def forward_rate(self, t: np.ndarray) -> np.ndarray:
        """Compute instantaneous forward rates from NSS parameters."""
        t = np.asarray(t, dtype=float)
        tau1, tau2 = self.tau1, self.tau2

        term1 = self.beta0
        term2 = self.beta1 * np.exp(-t / tau1)
        term3 = self.beta2 * (t / tau1) * np.exp(-t / tau1)
        term4 = self.beta3 * (t / tau2) * np.exp(-t / tau2)

        return term1 + term2 + term3 + term4

    def discount_factor(self, t: np.ndarray) -> np.ndarray:
        """Compute discount factors from spot rates."""
        return np.exp(-self.spot_rate(t) * t)


def yield_curve_chart(
    curve: YieldCurveData,
    nss_fit: NSSFit | None = None,
    title: str = "Yield Curve",
    height: int = 600,
    show_forward: bool = True,
    show_par: bool = True,
) -> go.Figure:
    """
    Create yield curve chart with spot, forward, par rates and NSS fit.

    Args:
        curve: YieldCurveData with tenors and rates.
        nss_fit: Optional NSSFit for smooth curve overlay.
        title: Chart title.
        height: Figure height in pixels.
        show_forward: Show forward curve.
        show_par: Show par curve.

    Returns:
        Plotly Figure with yield curves.
    """
    tenors = curve.tenors
    spot = curve.spot_rates

    fig = go.Figure()

    # Spot curve (market)
    fig.add_trace(
        go.Scatter(
            x=tenors,
            y=spot * 100,
            mode="lines+markers",
            name="Spot (Market)",
            line={"color": THEME.primary_color, "width": 2.5},
            marker={"size": 8, "symbol": "circle"},
            hovertemplate="Tenor: %{x:.2f}Y<br>Spot: %{y:.2f}%<extra></extra>",
        )
    )

    # Forward curve (market)
    if show_forward and curve.forward_rates is not None:
        fig.add_trace(
            go.Scatter(
                x=tenors,
                y=curve.forward_rates * 100,
                mode="lines+markers",
                name="Forward (Market)",
                line={"color": THEME.success_color, "width": 2, "dash": "dash"},
                marker={"size": 6, "symbol": "diamond"},
                hovertemplate="Tenor: %{x:.2f}Y<br>Forward: %{y:.2f}%<extra></extra>",
            )
        )

    # Par curve (market)
    if show_par and curve.par_rates is not None:
        fig.add_trace(
            go.Scatter(
                x=tenors,
                y=curve.par_rates * 100,
                mode="lines+markers",
                name="Par (Market)",
                line={"color": THEME.warning_color, "width": 2, "dash": "dot"},
                marker={"size": 6, "symbol": "square"},
                hovertemplate="Tenor: %{x:.2f}Y<br>Par: %{y:.2f}%<extra></extra>",
            )
        )

    # NSS fit
    if nss_fit is not None:
        t_fine = np.linspace(tenors.min(), tenors.max(), 200)

        # NSS spot
        nss_spot = nss_fit.spot_rate(t_fine)
        fig.add_trace(
            go.Scatter(
                x=t_fine,
                y=nss_spot * 100,
                mode="lines",
                name="NSS Spot Fit",
                line={"color": THEME.secondary_color, "width": 2, "dash": "dash"},
                hovertemplate="Tenor: %{x:.2f}Y<br>NSS Spot: %{y:.2f}%<extra></extra>",
            )
        )

        # NSS forward
        if show_forward:
            nss_fwd = nss_fit.forward_rate(t_fine)
            fig.add_trace(
                go.Scatter(
                    x=t_fine,
                    y=nss_fwd * 100,
                    mode="lines",
                    name="NSS Forward Fit",
                    line={"color": THEME.secondary_color, "width": 1.5, "dash": "dot"},
                    hovertemplate="Tenor: %{x:.2f}Y<br>NSS Fwd: %{y:.2f}%<extra></extra>",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Tenor (Years)",
        yaxis_title="Rate (%)",
        height=height,
        hovermode="x unified",
        showlegend=True,
    )

    _apply_theme(fig)
    fig.update_xaxes(gridcolor=THEME.grid_color)
    fig.update_yaxes(gridcolor=THEME.grid_color, tickformat=".2f")

    return fig


def yield_curve_carry_roll(
    tenors: np.ndarray,
    spot_rates: np.ndarray,
    forward_rates: np.ndarray,
    horizon: float = 1.0,  # years
    title: str = "Carry & Roll-Down",
    height: int = 500,
) -> go.Figure:
    """
    Create carry and roll-down heatmap/bar chart.

    Args:
        tenors: Tenor points in years.
        spot_rates: Current spot rates.
        forward_rates: Implied forward rates.
        horizon: Investment horizon in years.
        title: Chart title.
        height: Figure height.

    Returns:
        Plotly Figure with carry/roll visualization.
    """
    # Carry = forward rate - spot rate (approx)
    # Roll-down = spot rate at (tenor - horizon) - spot rate at tenor
    carry = forward_rates - spot_rates

    # Roll-down: interpolate spot at tenor - horizon
    from scipy.interpolate import interp1d

    interp = interp1d(
        tenors, spot_rates, kind="cubic", fill_value="extrapolate", bounds_error=False
    )
    spot_shifted = interp(np.maximum(tenors - horizon, 0.01))
    roll_down = spot_shifted - spot_rates

    total_carry_roll = carry + roll_down

    fig = go.Figure()

    # Stacked bar chart
    fig.add_trace(
        go.Bar(
            x=tenors,
            y=carry * 100,
            name="Carry",
            marker_color=THEME.primary_color,
            hovertemplate="Tenor: %{x:.2f}Y<br>Carry: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=tenors,
            y=roll_down * 100,
            name="Roll-Down",
            marker_color=THEME.secondary_color,
            hovertemplate="Tenor: %{x:.2f}Y<br>Roll-Down: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tenors,
            y=total_carry_roll * 100,
            mode="lines+markers",
            name="Total Carry+Roll",
            line={"color": THEME.warning_color, "width": 3},
            marker={"size": 8, "symbol": "star"},
            hovertemplate="Tenor: %{x:.2f}Y<br>Total: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{title} (Horizon: {horizon}Y)",
        xaxis_title="Tenor (Years)",
        yaxis_title="Annualized Return (%)",
        barmode="relative",
        height=height,
        hovermode="x unified",
        showlegend=True,
    )

    _apply_theme(fig)
    fig.update_xaxes(gridcolor=THEME.grid_color)
    fig.update_yaxes(
        gridcolor=THEME.grid_color,
        tickformat=".2f",
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor=THEME.grid_color,
    )

    return fig


def carry_rolldown_heatmap(
    tenors: np.ndarray,
    spot_rates: np.ndarray,
    forward_rates: np.ndarray,
    horizons: np.ndarray,
    title: str = "Carry & Roll-Down Heatmap",
    height: int = 600,
    colorscale: str = "RdBu",
    show_carry: bool = True,
    show_roll: bool = True,
    show_total: bool = True,
) -> go.Figure:
    """
    Create 2D heatmap of carry and roll-down across tenors and horizons.

    Args:
        tenors: Tenor points in years.
        spot_rates: Current spot rates at each tenor.
        forward_rates: Forward rates at each tenor.
        horizons: Investment horizons in years.
        title: Chart title.
        height: Figure height in pixels.
        colorscale: Plotly colorscale for heatmaps.
        show_carry: Show carry heatmap subplot.
        show_roll: Show roll-down heatmap subplot.
        show_total: Show total carry+roll heatmap subplot.

    Returns:
        Plotly Figure with 2D heatmaps (tenor × horizon).
    """
    from quantview.analytics.curves import carry_rolldown_matrix

    carry_m, roll_m, total_m = carry_rolldown_matrix(spot_rates, forward_rates, tenors, horizons)

    n_subplots = sum([show_carry, show_roll, show_total])
    if n_subplots == 0:
        n_subplots = 1
        show_total = True

    fig = make_subplots(
        rows=1,
        cols=n_subplots,
        subplot_titles=[
            t
            for t, s in [
                ("Carry", show_carry),
                ("Roll-Down", show_roll),
                ("Total Carry+Roll", show_total),
            ]
            if s
        ],
        horizontal_spacing=0.05,
    )

    col = 0
    z_min = min(carry_m.min(), roll_m.min(), total_m.min()) * 100
    z_max = max(carry_m.max(), roll_m.max(), total_m.max()) * 100
    z_range = [z_min, z_max]

    if show_carry:
        col += 1
        fig.add_trace(
            go.Heatmap(
                z=carry_m * 100,
                x=tenors,
                y=horizons,
                colorscale=colorscale,
                zmin=z_range[0],
                zmax=z_range[1],
                colorbar={"title": "Carry (%)", "thickness": 15, "len": 0.7}
                if col == n_subplots
                else None,
                hovertemplate="Tenor: %{x:.2f}Y<br>Horizon: %{y:.2f}Y<br>Carry: %{z:.2f}%<extra></extra>",
                name="Carry",
            ),
            row=1,
            col=col,
        )

    if show_roll:
        col += 1
        fig.add_trace(
            go.Heatmap(
                z=roll_m * 100,
                x=tenors,
                y=horizons,
                colorscale=colorscale,
                zmin=z_range[0],
                zmax=z_range[1],
                colorbar={"title": "Roll-Down (%)", "thickness": 15, "len": 0.7}
                if col == n_subplots
                else None,
                hovertemplate="Tenor: %{x:.2f}Y<br>Horizon: %{y:.2f}Y<br>Roll-Down: %{z:.2f}%<extra></extra>",
                name="Roll-Down",
            ),
            row=1,
            col=col,
        )

    if show_total:
        col += 1
        fig.add_trace(
            go.Heatmap(
                z=total_m * 100,
                x=tenors,
                y=horizons,
                colorscale=colorscale,
                zmin=z_range[0],
                zmax=z_range[1],
                colorbar={"title": "Total (%)", "thickness": 15, "len": 0.7},
                hovertemplate="Tenor: %{x:.2f}Y<br>Horizon: %{y:.2f}Y<br>Total: %{z:.2f}%<extra></extra>",
                name="Total Carry+Roll",
            ),
            row=1,
            col=col,
        )

    # Update all x-axes
    for c in range(1, n_subplots + 1):
        fig.update_xaxes(title_text="Tenor (Years)", gridcolor=THEME.grid_color, row=1, col=c)

    # Update y-axis (shared)
    fig.update_yaxes(title_text="Horizon (Years)", gridcolor=THEME.grid_color, row=1, col=1)

    fig.update_layout(
        title=title,
        height=height,
        hovermode="closest",
    )

    _apply_theme(fig)
    return fig


# ============================================================================
# Options Chain Greeks Grid
# ============================================================================


def options_chain_greeks(
    surface: VolSurface,
    expiry: float,
    strikes: np.ndarray | None = None,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
    title: str = "Options Chain Greeks",
    height: int = 600,
) -> go.Figure:
    """
    Create an interactive options chain grid with Greeks.

    Args:
        surface: VolSurface for implied vols.
        expiry: Time to expiry in years.
        strikes: Optional array of strikes (uses surface strikes if None).
        risk_free_rate: Risk-free rate for Greeks calculation.
        dividend_yield: Dividend yield.
        title: Chart title.
        height: Figure height.

    Returns:
        Plotly Figure with sortable Greeks table.
    """
    from black_scholes import (
        OptionParams,
        OptionType,
        black_scholes_greeks,
        black_scholes_price,
    )

    if strikes is None:
        strikes = np.array(surface.strikes)

    F = surface.forward
    spot = surface.spot

    # Compute Greeks for each strike
    data = []
    for K in strikes:
        iv = surface.implied_volatility(K)

        # Call
        call_params = OptionParams(
            spot=spot,
            strike=K,
            time_to_maturity=expiry,
            risk_free_rate=risk_free_rate,
            volatility=iv,
            option_type=OptionType.CALL,
            dividend_yield=dividend_yield,
        )
        call_price = black_scholes_price(call_params)
        call_greeks = black_scholes_greeks(call_params)

        # Put
        put_params = OptionParams(
            spot=spot,
            strike=K,
            time_to_maturity=expiry,
            risk_free_rate=risk_free_rate,
            volatility=iv,
            option_type=OptionType.PUT,
            dividend_yield=dividend_yield,
        )
        put_price = black_scholes_price(put_params)
        put_greeks = black_scholes_greeks(put_params)

        moneyness = K / F
        data.append(
            {
                "Strike": K,
                "Moneyness": moneyness,
                "IV": iv,
                "Call Price": call_price,
                "Call Delta": call_greeks.delta,
                "Call Gamma": call_greeks.gamma,
                "Call Vega": call_greeks.vega,
                "Call Theta": call_greeks.theta,
                "Put Price": put_price,
                "Put Delta": put_greeks.delta,
                "Put Gamma": put_greeks.gamma,
                "Put Vega": put_greeks.vega,
                "Put Theta": put_greeks.theta,
            }
        )

    df = pd.DataFrame(data)

    # Create table
    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": list(df.columns),
                    "fill_color": THEME.primary_color,
                    "font": {"color": "white", "size": 11},
                    "align": "center",
                    "height": 30,
                },
                cells={
                    "values": [df[col] for col in df.columns],
                    "fill_color": [[THEME.plot_bgcolor, "#161b22"] * (len(df) // 2 + 1)],
                    "font": {"color": "#e6edf3", "size": 10},
                    "align": "center",
                    "height": 28,
                    "format": [
                        None,  # Strike
                        ".3f",  # Moneyness
                        ".2%",  # IV
                        ".2f",  # Call Price
                        ".4f",  # Call Delta
                        ".6f",  # Call Gamma
                        ".4f",  # Call Vega
                        ".4f",  # Call Theta
                        ".2f",  # Put Price
                        ".4f",  # Put Delta
                        ".6f",  # Put Gamma
                        ".4f",  # Put Vega
                        ".4f",  # Put Theta
                    ],
                },
                columnwidth=[80, 80, 70, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80],
            )
        ]
    )

    fig.update_layout(
        title=f"{title} (Expiry: {expiry:.3f}Y, Spot: {spot:.2f}, Fwd: {F:.2f})",
        height=height,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )

    _apply_theme(fig)
    return fig


# ============================================================================
# IV Smile Chart with SVI Overlay
# ============================================================================


def iv_smile_chart(
    surface: VolSurface,
    expiry: float,
    svi_fit: SVIFit | None = None,
    title: str = "IV Smile",
    height: int = 500,
    show_confidence_band: bool = True,
    confidence_level: float = 0.95,
    moneyness_range: tuple[float, float] = (0.7, 1.3),
) -> go.Figure:
    """
    Create an IV smile chart with market points, SVI curve, and confidence band.

    Args:
        surface: VolSurface instance with market quotes.
        expiry: Time to expiry in years.
        svi_fit: Optional SVIFit for smooth curve overlay.
        title: Chart title.
        height: Figure height in pixels.
        show_confidence_band: Show confidence band around SVI fit.
        confidence_level: Confidence level for band (e.g., 0.95 for 95%).
        moneyness_range: (min, max) moneyness range for x-axis.

    Returns:
        Plotly Figure with IV smile, SVI fit, and confidence band.
    """
    if VolSurface is None:
        raise ImportError("black_scholes package required for iv_smile_chart")

    strikes = np.array(surface.strikes)
    vols = np.array(surface.implied_vols)
    T = surface.time_to_maturity
    F = surface.forward
    moneyness = strikes / F

    # Filter by moneyness range
    mask = (moneyness >= moneyness_range[0]) & (moneyness <= moneyness_range[1])
    strikes = strikes[mask]
    vols = vols[mask]
    moneyness = moneyness[mask]

    fig = go.Figure()

    # Market points
    fig.add_trace(
        go.Scatter(
            x=moneyness,
            y=vols,
            mode="markers",
            name="Market Quotes",
            marker={
                "size": 10,
                "color": THEME.primary_color,
                "symbol": "circle",
                "line": {"width": 1, "color": "white"},
            },
            hovertemplate="Strike: %{customdata[0]:.2f}<br>Moneyness: %{x:.3f}<br>IV: %{y:.2%}<extra></extra>",
            customdata=np.column_stack([strikes]),
        )
    )

    # SVI curve and confidence band
    if svi_fit is not None:
        k_fine = np.linspace(np.log(moneyness_range[0]), np.log(moneyness_range[1]), 300)
        strike_fine = F * np.exp(k_fine)
        moneyness_fine = strike_fine / F

        # SVI fit curve
        svi_vols = np.array([svi_fit.curve.implied_volatility(k, T) for k in k_fine])

        fig.add_trace(
            go.Scatter(
                x=moneyness_fine,
                y=svi_vols,
                mode="lines",
                name="SVI Fit",
                line={"color": THEME.secondary_color, "width": 2.5},
                hovertemplate="Moneyness: %{x:.3f}<br>SVI IV: %{y:.2%}<extra></extra>",
            )
        )

        # Confidence band (based on RMS error)
        if show_confidence_band:
            from scipy.stats import norm

            z = norm.ppf((1 + confidence_level) / 2)
            rms = svi_fit.rms_error
            band = z * rms

            upper_band = np.minimum(svi_vols + band, 1.0)  # Cap at 100%
            lower_band = np.maximum(svi_vols - band, 0.0)  # Floor at 0%

            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([moneyness_fine, moneyness_fine[::-1]]),
                    y=np.concatenate([upper_band, lower_band[::-1]]),
                    fill="toself",
                    fillcolor="rgba(163, 113, 247, 0.15)",
                    line={"color": "rgba(255,255,255,0)"},
                    name=f"{int(confidence_level * 100)}% Confidence Band",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

        # ATM line
        fig.add_vline(x=1.0, line_dash="dot", line_color=THEME.grid_color, line_width=1)

    fig.update_layout(
        title=f"{title} (T={T:.3f}Y, F={F:.2f})",
        xaxis_title="Moneyness (K/F)",
        yaxis_title="Implied Volatility",
        height=height,
        showlegend=True,
        hovermode="x unified",
    )

    _apply_theme(fig)
    fig.update_yaxes(tickformat=".1%")
    fig.update_xaxes(range=moneyness_range)

    return fig


# ============================================================================
# Term Structure Chart
# ============================================================================


def term_structure_chart(
    surfaces: dict[float, VolSurface],
    title: str = "Volatility Term Structure",
    height: int = 600,
    show_atm: bool = True,
    show_skew: bool = True,
    show_curvature: bool = True,
    delta: float = 0.25,
) -> go.Figure:
    """
    Create a term structure chart: ATM vol, skew, curvature vs expiry.

    Args:
        surfaces: Dict mapping expiry (years) -> VolSurface.
        title: Chart title.
        height: Figure height in pixels.
        show_atm: Show ATM volatility term structure.
        show_skew: Show 25-delta skew term structure.
        show_curvature: Show curvature (butterfly) term structure.
        delta: Delta for skew calculation (default 0.25 for 25-delta).

    Returns:
        Plotly Figure with term structure subplots.
    """
    if VolSurface is None:
        raise ImportError("black_scholes package required for term_structure_chart")

    from scipy.stats import norm

    if not surfaces:
        raise ValueError("No surfaces provided")

    expiries = np.array(sorted(surfaces.keys()))

    # Compute metrics for each expiry
    atm_vols = []
    skew_25d = []
    curvature = []
    atm_var = []
    jw_skew = []
    put_wing = []
    call_wing = []

    for expiry in expiries:
        surface = surfaces[expiry]
        try:
            fit = surface.fit()
            T = expiry

            # ATM vol
            atm_iv = fit.curve.implied_volatility(0.0, T)
            atm_vols.append(atm_iv)

            # Jump-Wings parameters
            jw = fit.curve.jump_wings_parameters(T)
            atm_var.append(jw["atm_variance"])
            jw_skew.append(jw["atm_skew"])
            put_wing.append(jw["put_wing_slope"])
            call_wing.append(jw["call_wing_slope"])

            # 25-delta skew and curvature
            d1_call = norm.ppf(delta)
            d1_put = norm.ppf(1 - delta)

            sigma_atm = atm_iv
            sqrt_T = np.sqrt(T)

            k_call = d1_call * sigma_atm * sqrt_T - 0.5 * sigma_atm**2 * T
            k_put = d1_put * sigma_atm * sqrt_T - 0.5 * sigma_atm**2 * T

            iv_call_25d = fit.curve.implied_volatility(k_call, T)
            iv_put_25d = fit.curve.implied_volatility(k_put, T)

            # 25-delta skew
            skew = (iv_put_25d - iv_call_25d) / (2 * atm_iv) if atm_iv > 0 else 0
            skew_25d.append(skew)

            # Curvature (butterfly)
            curv = (iv_call_25d + iv_put_25d) / 2 - atm_iv
            curvature.append(curv)

        except Exception:
            atm_vols.append(np.nan)
            skew_25d.append(np.nan)
            curvature.append(np.nan)
            atm_var.append(np.nan)
            jw_skew.append(np.nan)
            put_wing.append(np.nan)
            call_wing.append(np.nan)

    # Build subplots
    n_rows = sum([show_atm, show_skew, show_curvature])
    if n_rows == 0:
        n_rows = 1
        show_atm = True

    row_heights = [1.0 / n_rows] * n_rows
    subplot_titles = []
    if show_atm:
        subplot_titles.append("ATM Volatility")
    if show_skew:
        subplot_titles.append(f"{int(delta * 100)}d Skew")
    if show_curvature:
        subplot_titles.append("Curvature (Butterfly)")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    row = 0
    if show_atm:
        row += 1
        fig.add_trace(
            go.Scatter(
                x=expiries,
                y=np.array(atm_vols) * 100,
                mode="lines+markers",
                name="ATM Vol",
                line={"color": THEME.primary_color, "width": 2.5},
                marker={"size": 8, "symbol": "circle"},
                hovertemplate="Expiry: %{x:.3f}Y<br>ATM Vol: %{y:.2f}%<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="Vol (%)", row=row, col=1, tickformat=".1f")

    if show_skew:
        row += 1
        fig.add_trace(
            go.Scatter(
                x=expiries,
                y=np.array(skew_25d) * 100,
                mode="lines+markers",
                name=f"{int(delta * 100)}d Skew",
                line={"color": THEME.secondary_color, "width": 2.5},
                marker={"size": 8, "symbol": "diamond"},
                hovertemplate="Expiry: %{x:.3f}Y<br>Skew: %{y:.2f}%<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.add_hline(
            y=0, line_dash="dot", line_color=THEME.grid_color, line_width=1, row=row, col=1
        )
        fig.update_yaxes(title_text="Skew (%)", row=row, col=1, tickformat=".2f")

    if show_curvature:
        row += 1
        fig.add_trace(
            go.Scatter(
                x=expiries,
                y=np.array(curvature) * 100,
                mode="lines+markers",
                name="Curvature",
                line={"color": THEME.warning_color, "width": 2.5},
                marker={"size": 8, "symbol": "square"},
                hovertemplate="Expiry: %{x:.3f}Y<br>Curvature: %{y:.4f}%<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.add_hline(
            y=0, line_dash="dot", line_color=THEME.grid_color, line_width=1, row=row, col=1
        )
        fig.update_yaxes(title_text="Curvature (%)", row=row, col=1, tickformat=".3f")

    fig.update_layout(
        title=title,
        xaxis_title="Time to Expiry (Years)",
        height=height,
        showlegend=False,
        hovermode="x unified",
    )

    _apply_theme(fig)
    fig.update_xaxes(gridcolor=THEME.grid_color, row=n_rows, col=1)

    return fig


# ============================================================================
# Interactive Options Chain DataGrid (ipywidgets)
# ============================================================================


def create_options_chain_grid(
    surface: VolSurface,
    expiry: float,
    strikes: np.ndarray | None = None,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
    include_bid_ask: bool = False,
    bid_ask_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, go.Figure]:
    """
    Create an options chain DataFrame and Plotly table with Greeks.

    Args:
        surface: VolSurface for implied vols.
        expiry: Time to expiry in years.
        strikes: Optional array of strikes (uses surface strikes if None).
        risk_free_rate: Risk-free rate for Greeks calculation.
        dividend_yield: Dividend yield.
        include_bid_ask: Whether to include bid/ask/mid, volume, OI columns.
        bid_ask_data: Optional DataFrame with bid/ask/volume/oi for each strike.

    Returns:
        Tuple of (DataFrame, Plotly Figure with table).
    """
    from black_scholes import (
        OptionParams,
        OptionType,
        black_scholes_greeks,
        black_scholes_price,
    )

    if strikes is None:
        strikes = np.array(surface.strikes)

    F = surface.forward
    spot = surface.spot

    data = []
    for K in strikes:
        iv = surface.implied_volatility(K)

        # Call
        call_params = OptionParams(
            spot=spot,
            strike=K,
            time_to_maturity=expiry,
            risk_free_rate=risk_free_rate,
            volatility=iv,
            option_type=OptionType.CALL,
            dividend_yield=dividend_yield,
        )
        call_price = black_scholes_price(call_params)
        call_greeks = black_scholes_greeks(call_params)

        # Put
        put_params = OptionParams(
            spot=spot,
            strike=K,
            time_to_maturity=expiry,
            risk_free_rate=risk_free_rate,
            volatility=iv,
            option_type=OptionType.PUT,
            dividend_yield=dividend_yield,
        )
        put_price = black_scholes_price(put_params)
        put_greeks = black_scholes_greeks(put_params)

        moneyness = K / F

        row = {
            "Strike": K,
            "Moneyness": moneyness,
            "IV": iv,
            "Call Price": call_price,
            "Call Delta": call_greeks.delta,
            "Call Gamma": call_greeks.gamma,
            "Call Vega": call_greeks.vega,
            "Call Theta": call_greeks.theta,
            "Call Rho": call_greeks.rho,
            "Put Price": put_price,
            "Put Delta": put_greeks.delta,
            "Put Gamma": put_greeks.gamma,
            "Put Vega": put_greeks.vega,
            "Put Theta": put_greeks.theta,
            "Put Rho": put_greeks.rho,
        }

        if include_bid_ask and bid_ask_data is not None:
            # Match by strike
            match = bid_ask_data[bid_ask_data["strike"] == K]
            if len(match) > 0:
                m = match.iloc[0]
                row.update(
                    {
                        "Bid": m.get("bid", np.nan),
                        "Ask": m.get("ask", np.nan),
                        "Mid": m.get("mid", (m.get("bid", 0) + m.get("ask", 0)) / 2),
                        "Volume": m.get("volume", 0),
                        "Open Interest": m.get("open_interest", 0),
                    }
                )

        data.append(row)

    df = pd.DataFrame(data)

    # Create table figure
    columns = list(df.columns)
    formats: list[str | None] = []
    for col in columns:
        if col in ["Strike"]:
            formats.append(None)
        elif col in ["Moneyness"]:
            formats.append(".3f")
        elif col in ["IV"]:
            formats.append(".2%")
        elif "Price" in col or "Bid" in col or "Ask" in col or "Mid" in col:
            formats.append(".2f")
        elif "Delta" in col or "Vega" in col or "Theta" in col or "Rho" in col:
            formats.append(".4f")
        elif "Gamma" in col:
            formats.append(".6f")
        elif col in ["Volume", "Open Interest"]:
            formats.append(",d")
        else:
            formats.append(".4f")

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": columns,
                    "fill_color": THEME.primary_color,
                    "font": {"color": "white", "size": 11},
                    "align": "center",
                    "height": 30,
                },
                cells={
                    "values": [df[col] for col in columns],
                    "fill_color": [[THEME.plot_bgcolor, "#161b22"] * (len(df) // 2 + 1)],
                    "font": {"color": "#e6edf3", "size": 10},
                    "align": "center",
                    "height": 28,
                    "format": formats,
                },
                columnwidth=[70] * len(columns),
            )
        ]
    )

    fig.update_layout(
        title=f"Options Chain Greeks (Expiry: {expiry:.3f}Y, Spot: {spot:.2f}, Fwd: {F:.2f})",
        height=600,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )

    _apply_theme(fig)

    return df, fig


# ============================================================================
# Export Utilities
# ============================================================================


# ============================================================================
# Demo / Test Functions
# ============================================================================


def _demo_candlestick() -> go.Figure:
    """Generate demo candlestick chart with synthetic data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    returns = np.random.normal(0.0005, 0.015, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
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

    # Ensure high >= max(o,c) and low <= min(o,c)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return candlestick_chart(df, "SPY", indicators=["sma_20", "sma_50", "bbands", "rsi", "macd"])


def _demo_vol_surface() -> tuple[go.Figure, go.Figure]:
    """Generate demo volatility surface charts."""
    if VolSurface is None or black_scholes_price is None or OptionParams is None:
        raise ImportError(
            "black_scholes is required for the demo; install it with "
            "'pip install quantview[surfaces]' or run from a checkout with "
            "the kernel on the path"
        )

    # Create synthetic vol surface
    strikes = np.array([70, 80, 90, 95, 100, 105, 110, 120, 130, 140, 150])
    vols = np.array([0.32, 0.28, 0.25, 0.235, 0.23, 0.235, 0.24, 0.26, 0.29, 0.31, 0.33])
    spot = 100.0
    T = 0.25
    r = 0.05

    surface = VolSurface.from_market_prices(
        spot=spot,
        strikes=strikes,
        prices=[
            black_scholes_price(OptionParams(spot, k, T, r, v))
            for k, v in zip(strikes, vols, strict=False)
        ],
        time_to_maturity=T,
        risk_free_rate=r,
    )

    svi_fit = surface.fit()

    fig_3d = vol_surface_3d(surface, svi_fit, title="SPX Vol Surface (3M)")
    fig_2d = vol_surface_2d(surface, svi_fit, title="SPX Vol Smile (3M)")

    return fig_3d, fig_2d


def _demo_yield_curve() -> tuple[go.Figure, go.Figure]:
    """Generate demo yield curve charts."""
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    spot = np.array([0.052, 0.050, 0.048, 0.046, 0.045, 0.044, 0.044, 0.0445, 0.0455, 0.046])
    forward = np.array([0.052, 0.049, 0.046, 0.044, 0.0435, 0.043, 0.0435, 0.0445, 0.046, 0.047])
    par = np.array([0.052, 0.050, 0.048, 0.046, 0.045, 0.044, 0.044, 0.0445, 0.0455, 0.046])

    curve = YieldCurveData(tenors, spot, forward, par)

    # NSS fit (rough calibration)
    nss = NSSFit(beta0=0.045, beta1=-0.01, beta2=-0.015, beta3=0.005, tau1=1.5, tau2=8.0)

    fig_curve = yield_curve_chart(curve, nss_fit=nss, title="USD Treasury Yield Curve")
    fig_carry = yield_curve_carry_roll(
        tenors, spot, forward, horizon=1.0, title="Carry & Roll-Down (1Y Horizon)"
    )

    return fig_curve, fig_carry


def _demo_options_chain() -> go.Figure:
    """Generate demo options chain Greeks table."""
    if VolSurface is None:
        raise ImportError("black_scholes package required for demo")

    strikes = np.arange(80, 131, 5)
    vols = 0.23 + 0.1 * (strikes - 100) ** 2 / 10000  # Simple smile
    surface = VolSurface(
        spot=100.0,
        strikes=tuple(strikes),
        implied_vols=tuple(vols),
        time_to_maturity=0.25,
        risk_free_rate=0.05,
        dividend_yield=0.0,
    )

    return options_chain_greeks(surface, expiry=0.25, title="SPX Options Chain (3M Expiry)")


if __name__ == "__main__":
    # Quick demo when run directly
    from .export import export_html

    print("Generating demo charts...")

    fig1 = _demo_candlestick()
    export_html(fig1, "demo_candlestick.html")
    print("Saved demo_candlestick.html")

    try:
        fig3d, fig2d = _demo_vol_surface()
        export_html(fig3d, "demo_vol_3d.html")
        export_html(fig2d, "demo_vol_2d.html")
        print("Saved demo_vol_3d.html, demo_vol_2d.html")
    except ImportError as e:
        print(f"Skipping vol surface demo: {e}")

    fig_curve, fig_carry = _demo_yield_curve()
    export_html(fig_curve, "demo_yield_curve.html")
    export_html(fig_carry, "demo_carry_roll.html")
    print("Saved demo_yield_curve.html, demo_carry_roll.html")

    try:
        fig_opt = _demo_options_chain()
        export_html(fig_opt, "demo_options_chain.html")
        print("Saved demo_options_chain.html")
    except ImportError as e:
        print(f"Skipping options chain demo: {e}")

    print("All demos complete!")
