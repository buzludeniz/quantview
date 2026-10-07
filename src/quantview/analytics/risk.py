"""
QuantView Analytics Risk Module.

Risk analytics: Greeks heatmaps, bucketing, risk limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - import for annotations only
    # matplotlib and seaborn are optional extras; the return annotations below
    # name Figure, which from __future__ import annotations leaves unevaluated.
    import matplotlib.figure

from .portfolio import Position, generate_synthetic_portfolio

GreekName = Literal["delta", "gamma", "vega", "theta", "rho", "notional", "count"]


@dataclass(frozen=True, slots=True)
class GreeksHeatmap:
    """Greeks heatmap data for visualization."""

    expiry_bins: list[str]
    moneyness_bins: list[str]
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    rho: np.ndarray
    notional: np.ndarray
    count: np.ndarray

    def to_dataframe(
        self, greek: Literal["delta", "gamma", "vega", "theta", "rho", "notional", "count"]
    ) -> pd.DataFrame:
        """Convert a specific Greek to a pivot DataFrame for heatmap plotting."""
        data = getattr(self, greek)
        return pd.DataFrame(
            data,
            index=self.expiry_bins,
            columns=self.moneyness_bins,
        )

    def get_max_abs(
        self, greek: Literal["delta", "gamma", "vega", "theta", "rho"]
    ) -> tuple[float, tuple[int, int]]:
        """Get maximum absolute value and its position."""
        data = getattr(self, greek)
        max_idx = np.unravel_index(np.abs(data).argmax(), data.shape)
        return float(data[max_idx]), max_idx


def greeks_buckets(
    positions: list[Position],
    expiry_bins: list[float] | None = None,
    moneyness_bins: list[float] | None = None,
) -> GreeksHeatmap:
    """
    Create expiry × moneyness heatmap of portfolio Greeks.

    Args:
        positions: List of Position objects.
        expiry_bins: Expiry bin edges in years.
                     Default: [0, 1/52, 1/12, 2/12, 3/12, 6/12, 1.0, 2.0, inf]
        moneyness_bins: Moneyness bin edges (K/F).
                        Default: [0, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 2.0]

    Returns:
        GreeksHeatmap with 2D arrays for each Greek.

    Example:
        >>> positions = generate_synthetic_portfolio(100)
        >>> heatmap = greeks_buckets(positions)
        >>> delta_df = heatmap.to_dataframe("delta")
        >>> print(delta_df)
    """
    if expiry_bins is None:
        expiry_bins = [0, 1 / 52, 1 / 12, 2 / 12, 3 / 12, 6 / 12, 1.0, 2.0, np.inf]
    if moneyness_bins is None:
        moneyness_bins = [0, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 2.0]

    # Create labels
    expiry_labels = []
    for i in range(len(expiry_bins) - 1):
        lo, hi = expiry_bins[i], expiry_bins[i + 1]
        if hi == np.inf:
            expiry_labels.append(f">{lo:.2f}Y")
        elif lo == 0:
            expiry_labels.append(f"<{hi:.2f}Y")
        else:
            expiry_labels.append(f"{lo:.2f}-{hi:.2f}Y")

    moneyness_labels = []
    for i in range(len(moneyness_bins) - 1):
        lo, hi = moneyness_bins[i], moneyness_bins[i + 1]
        if lo == 0:
            moneyness_labels.append(f"<{hi:.2f}")
        elif hi >= 2.0:
            moneyness_labels.append(f">{lo:.2f}")
        else:
            moneyness_labels.append(f"{lo:.2f}-{hi:.2f}")

    n_expiry = len(expiry_labels)
    n_mny = len(moneyness_labels)

    # Initialize arrays
    delta_arr = np.zeros((n_expiry, n_mny))
    gamma_arr = np.zeros((n_expiry, n_mny))
    vega_arr = np.zeros((n_expiry, n_mny))
    theta_arr = np.zeros((n_expiry, n_mny))
    rho_arr = np.zeros((n_expiry, n_mny))
    notional_arr = np.zeros((n_expiry, n_mny))
    count_arr = np.zeros((n_expiry, n_mny), dtype=int)

    try:
        from black_scholes import black_scholes_greeks
    except ImportError as exc:
        raise ImportError(
            "black_scholes package required for greeks_buckets; "
            "install it with 'pip install quantview[surfaces]'"
        ) from exc

    for pos in positions:
        params = pos.to_option_params()
        g = black_scholes_greeks(params)
        F = params.forward_price
        moneyness = pos.strike / F

        # Find expiry bin
        # searchsorted yields an np.int64; numpy rejects that as an index in
        # some versions, so clamp to a plain int.
        expiry_idx = int(np.searchsorted(expiry_bins, pos.expiry, side="right") - 1)
        expiry_idx = max(0, min(expiry_idx, n_expiry - 1))

        # Find moneyness bin
        mny_idx = int(np.searchsorted(moneyness_bins, moneyness, side="right") - 1)
        mny_idx = max(0, min(mny_idx, n_mny - 1))

        qty = pos.quantity
        notional = abs(qty) * pos.spot * 100

        delta_arr[expiry_idx, mny_idx] += g.delta * qty
        gamma_arr[expiry_idx, mny_idx] += g.gamma * qty
        vega_arr[expiry_idx, mny_idx] += g.vega * qty
        theta_arr[expiry_idx, mny_idx] += g.theta * qty
        rho_arr[expiry_idx, mny_idx] += g.rho * qty
        notional_arr[expiry_idx, mny_idx] += notional
        count_arr[expiry_idx, mny_idx] += 1

    return GreeksHeatmap(
        expiry_bins=expiry_labels,
        moneyness_bins=moneyness_labels,
        delta=delta_arr,
        gamma=gamma_arr,
        vega=vega_arr,
        theta=theta_arr,
        rho=rho_arr,
        notional=notional_arr,
        count=count_arr,
    )


def greeks_heatmap_plot(
    heatmap: GreeksHeatmap,
    greek: Literal["delta", "gamma", "vega", "theta", "rho"] = "delta",
    title: str | None = None,
    cmap: str = "RdBu_r",
    center_zero: bool = True,
) -> matplotlib.figure.Figure:
    """
    Create a matplotlib heatmap plot of portfolio Greeks.

    Args:
        heatmap: GreeksHeatmap from greeks_buckets().
        greek: Which Greek to plot.
        title: Optional plot title.
        cmap: Colormap name.
        center_zero: Whether to center colormap at zero.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = heatmap.to_dataframe(greek)

    fig, ax = plt.subplots(figsize=(10, 6))

    if center_zero:
        vmax = np.abs(data.values).max()
        vmin = -vmax
        sns.heatmap(
            data,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar_kws={"label": greek.capitalize()},
        )
    else:
        sns.heatmap(
            data,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            ax=ax,
            cbar_kws={"label": greek.capitalize()},
        )

    ax.set_xlabel("Moneyness (K/F)")
    ax.set_ylabel("Time to Expiry")
    ax.set_title(title or f"Portfolio {greek.capitalize()} Heatmap")

    plt.tight_layout()
    return fig


def plot_all_greeks_heatmaps(
    heatmap: GreeksHeatmap,
    cmap: str = "RdBu_r",
) -> matplotlib.figure.Figure:
    """
    Create a 2x3 grid of all Greeks heatmaps.

    Args:
        heatmap: GreeksHeatmap from greeks_buckets().
        cmap: Colormap name.

    Returns:
        matplotlib Figure object with 6 subplots.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    greeks = ["delta", "gamma", "vega", "theta", "rho", "notional"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, greek in enumerate(greeks):
        ax = axes[idx]
        # greeks is a fixed Literal tuple, but enumerate widens it to str for
        # mypy; to_dataframe only accepts the Literal it declares.
        data = heatmap.to_dataframe(cast("GreekName", greek))

        if greek != "notional" and greek != "count":
            vmax = np.abs(data.values).max()
            vmin = -vmax
            sns.heatmap(
                data,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                center=0,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
                cbar_kws={"label": greek.capitalize()},
            )
        else:
            sns.heatmap(
                data,
                annot=True,
                fmt=".2f" if greek == "notional" else "d",
                cmap="Blues",
                ax=ax,
                cbar_kws={"label": greek.capitalize()},
            )

        ax.set_xlabel("Moneyness (K/F)")
        ax.set_ylabel("Time to Expiry")
        ax.set_title(f"{greek.capitalize()}")

    plt.tight_layout()
    return fig


def risk_limits_check(
    positions: list[Position],
    limits: dict[str, float],
) -> pd.DataFrame:
    """
    Check portfolio Greeks against risk limits.

    Args:
        positions: List of Position objects.
        limits: Dict mapping greek -> max absolute value.
                e.g., {"delta": 1000, "gamma": 500, "vega": 2000, "theta": -10000}

    Returns:
        DataFrame with columns: greek, current, limit, utilization, breach.
    """
    from .portfolio import aggregate_greeks

    greeks = aggregate_greeks(positions)
    current = greeks.as_dict()

    rows = []
    for greek, limit in limits.items():
        if greek in current:
            val = current[greek]
            util = abs(val) / limit if limit != 0 else np.inf
            breach = abs(val) > limit
            rows.append(
                {
                    "greek": greek,
                    "current": val,
                    "limit": limit,
                    "utilization_pct": util * 100,
                    "breach": breach,
                }
            )

    return pd.DataFrame(rows)


def stress_test_greeks(
    positions: list[Position],
    spot_shocks: list[float] | None = None,
    vol_shocks: list[float] | None = None,
    rate_shocks: list[float] | None = None,
) -> pd.DataFrame:
    """
    Stress test portfolio Greeks under various scenarios.

    Args:
        positions: List of Position objects.
        spot_shocks: List of spot price shocks (e.g., [-0.1, -0.05, 0, 0.05, 0.1]).
        vol_shocks: List of vol shocks (e.g., [-0.05, 0, 0.05, 0.1]).
        rate_shocks: List of rate shocks (e.g., [-0.01, 0, 0.01]).

    Returns:
        DataFrame with P&L under each scenario.
    """
    if spot_shocks is None:
        spot_shocks = [-0.1, -0.05, 0, 0.05, 0.1]
    if vol_shocks is None:
        vol_shocks = [-0.05, 0, 0.05, 0.1]
    if rate_shocks is None:
        rate_shocks = [-0.01, 0, 0.01]

    from .portfolio import pnl_waterfall

    rows = []
    for dS_pct in spot_shocks:
        for dVol in vol_shocks:
            for dRate in rate_shocks:
                price_changes = {pos.symbol: pos.spot * dS_pct for pos in positions}
                vol_changes = {pos.symbol: dVol for pos in positions}
                rate_changes = {pos.symbol: dRate for pos in positions}

                pnl = pnl_waterfall(
                    positions,
                    price_changes,
                    vol_changes,
                    rate_changes,
                    time_decay=0,  # No time decay for stress test
                )

                total_pnl = pnl[pnl["symbol"] == "TOTAL"]["exact_pnl"].values[0]

                rows.append(
                    {
                        "spot_shock_pct": dS_pct * 100,
                        "vol_shock": dVol * 100,
                        "rate_shock_bps": dRate * 10000,
                        "total_pnl": total_pnl,
                    }
                )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Demo
    positions = generate_synthetic_portfolio(100)
    heatmap = greeks_buckets(positions)

    print("Delta heatmap:")
    print(heatmap.to_dataframe("delta"))
    print("\nGamma heatmap:")
    print(heatmap.to_dataframe("gamma"))
    print("\nVega heatmap:")
    print(heatmap.to_dataframe("vega"))
