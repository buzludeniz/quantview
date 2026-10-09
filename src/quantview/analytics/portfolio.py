"""
QuantView Analytics Portfolio Module.

Portfolio-level analytics: Greeks aggregation, P&L waterfall, factor attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import linalg

# Explicitly bind to None so the names exist in the module namespace when
# the optional dependency is absent. The try/except then rebinds on success.
OptionParams: Any = None
OptionType: Any = None
black_scholes_greeks: Any = None
try:
    from black_scholes import OptionParams, OptionType, black_scholes_greeks
except ImportError:
    pass


@dataclass(frozen=True, slots=True)
class Position:
    """Single option position in a portfolio."""

    symbol: str
    strike: float
    expiry: float  # years
    option_type: Literal["call", "put"]
    quantity: float  # positive = long, negative = short
    spot: float
    implied_vol: float
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0

    def to_option_params(self) -> OptionParams:
        """Convert to black_scholes OptionParams."""
        if OptionParams is None:
            raise ImportError("black_scholes package required")
        opt_type = OptionType.CALL if self.option_type == "call" else OptionType.PUT
        return OptionParams(
            spot=self.spot,
            strike=self.strike,
            time_to_maturity=self.expiry,
            risk_free_rate=self.risk_free_rate,
            volatility=self.implied_vol,
            option_type=opt_type,
            dividend_yield=self.dividend_yield,
        )


@dataclass(frozen=True, slots=True)
class Greeks:
    """Aggregated portfolio Greeks."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def as_dict(self) -> dict[str, float]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }


@dataclass(frozen=True, slots=True)
class GreeksBucket:
    """Greeks aggregated by expiry and moneyness bucket."""

    expiry: float
    moneyness_bin: str  # e.g., "OTM", "ATM", "ITM"
    greeks: Greeks
    notional: float
    position_count: int

    def as_dict(self) -> dict:
        # The greeks dict is str -> float; the extra keys add a date, a label
        # and two counts, so the merged value type is wider than float.
        d: dict[str, Any] = dict(self.greeks.as_dict())
        d.update(
            {
                "expiry": self.expiry,
                "moneyness_bin": self.moneyness_bin,
                "notional": self.notional,
                "position_count": self.position_count,
            }
        )
        return d


def _moneyness_bin(moneyness: float) -> str:
    """Classify moneyness into buckets."""
    if moneyness < 0.95:
        return "OTM"
    elif moneyness > 1.05:
        return "ITM"
    else:
        return "ATM"


def aggregate_greeks(positions: list[Position]) -> Greeks:
    """
    Aggregate net Greeks across all positions.

    Args:
        positions: List of Position objects.

    Returns:
        Greeks with net delta, gamma, vega, theta, rho.

    Example:
        >>> pos = Position("SPY", 450, 0.25, "call", 10, 455, 0.2)
        >>> greeks = aggregate_greeks([pos])
        >>> print(greeks.delta)
    """
    if black_scholes_greeks is None:
        raise ImportError("black_scholes package required for aggregate_greeks")

    net_delta = 0.0
    net_gamma = 0.0
    net_vega = 0.0
    net_theta = 0.0
    net_rho = 0.0

    for pos in positions:
        params = pos.to_option_params()
        g = black_scholes_greeks(params)
        qty = pos.quantity

        net_delta += g.delta * qty
        net_gamma += g.gamma * qty
        net_vega += g.vega * qty
        net_theta += g.theta * qty
        net_rho += g.rho * qty

    return Greeks(
        delta=net_delta,
        gamma=net_gamma,
        vega=net_vega,
        theta=net_theta,
        rho=net_rho,
    )


def aggregate_greeks_by_bucket(
    positions: list[Position],
    expiry_bins: list[float] | None = None,
    moneyness_bins: list[float] | None = None,
) -> pd.DataFrame:
    """
    Aggregate Greeks by expiry and moneyness buckets.

    Args:
        positions: List of Position objects.
        expiry_bins: Expiry bin edges in years. Default: [0, 0.083, 0.25, 0.5, 1, 2, inf]
        moneyness_bins: Moneyness bin edges (K/F). Default: [0, 0.9, 0.95, 1.05, 1.1, 2]

    Returns:
        DataFrame with columns: expiry_bin, moneyness_bin, delta, gamma, vega, theta, rho, notional, count
    """
    if black_scholes_greeks is None:
        raise ImportError("black_scholes package required")

    if expiry_bins is None:
        expiry_bins = [0, 1 / 12, 0.25, 0.5, 1.0, 2.0, np.inf]
    if moneyness_bins is None:
        moneyness_bins = [0, 0.9, 0.95, 1.05, 1.1, 2.0]

    expiry_labels = [
        f"{expiry_bins[i]:.3f}-{expiry_bins[i + 1]:.3f}" for i in range(len(expiry_bins) - 1)
    ]
    moneyness_labels = ["Deep OTM", "OTM", "ATM", "ITM", "Deep ITM"]

    rows = []
    for pos in positions:
        params = pos.to_option_params()
        g = black_scholes_greeks(params)
        F = params.forward_price
        moneyness = pos.strike / F

        # Find expiry bin
        # int() because searchsorted yields an np.int64, which numpy rejects
        # as an array index in some versions.
        expiry_idx = int(np.searchsorted(expiry_bins, pos.expiry, side="right") - 1)
        expiry_idx = max(0, min(expiry_idx, len(expiry_labels) - 1))

        # Find moneyness bin
        mny_idx = int(np.searchsorted(moneyness_bins, moneyness, side="right") - 1)
        mny_idx = max(0, min(mny_idx, len(moneyness_labels) - 1))

        notional = abs(pos.quantity) * pos.spot * 100  # Assuming 100 multiplier

        rows.append(
            {
                "expiry_bin": expiry_labels[expiry_idx],
                "moneyness_bin": moneyness_labels[mny_idx],
                "delta": g.delta * pos.quantity,
                "gamma": g.gamma * pos.quantity,
                "vega": g.vega * pos.quantity,
                "theta": g.theta * pos.quantity,
                "rho": g.rho * pos.quantity,
                "notional": notional,
                "count": 1,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Aggregate
    agg = (
        df.groupby(["expiry_bin", "moneyness_bin"])
        .agg(
            {
                "delta": "sum",
                "gamma": "sum",
                "vega": "sum",
                "theta": "sum",
                "rho": "sum",
                "notional": "sum",
                "count": "sum",
            }
        )
        .reset_index()
    )

    return agg


def pnl_waterfall(
    positions: list[Position],
    price_changes: dict[str, float],
    vol_changes: dict[str, float] | None = None,
    rate_changes: dict[str, float] | None = None,
    time_decay: float = 1 / 365,
) -> pd.DataFrame:
    """
    Daily P&L decomposition for a portfolio.

    Decomposes P&L into:
    - Delta P&L (spot price change)
    - Gamma P&L (convexity from spot change)
    - Vega P&L (volatility change)
    - Theta P&L (time decay)
    - Rho P&L (rate change)
    - Residual (higher order / cross effects)

    Args:
        positions: List of Position objects.
        price_changes: Dict mapping symbol -> spot price change (absolute).
        vol_changes: Optional dict mapping symbol -> vol change (absolute, e.g., 0.01 for +1%).
        rate_changes: Optional dict mapping symbol -> rate change (absolute, e.g., 0.0001 for +1bp).
        time_decay: Time decay in years (default 1 day = 1/365).

    Returns:
        DataFrame with P&L components per position and totals.
    """
    if black_scholes_greeks is None:
        raise ImportError("black_scholes package required")

    rows = []
    for pos in positions:
        params = pos.to_option_params()
        g = black_scholes_greeks(params)

        # Spot change for this symbol
        dS = price_changes.get(pos.symbol, 0.0)

        # Vol change for this symbol
        dVol = vol_changes.get(pos.symbol, 0.0) if vol_changes else 0.0

        # Rate change for this symbol
        dRate = rate_changes.get(pos.symbol, 0.0) if rate_changes else 0.0

        qty = pos.quantity

        # First-order P&L components
        delta_pnl = g.delta * dS * qty * 100
        gamma_pnl = 0.5 * g.gamma * (dS**2) * qty * 100
        vega_pnl = g.vega * dVol * qty * 100
        theta_pnl = g.theta * time_decay * qty * 100
        rho_pnl = g.rho * dRate * qty * 100

        # Total first-order approximation
        total_first_order = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl

        # Reprice for exact P&L (for residual calculation)
        # New params after changes
        new_spot = pos.spot + dS
        new_vol = max(pos.implied_vol + dVol, 0.001)
        new_rate = pos.risk_free_rate + dRate
        new_expiry = max(pos.expiry - time_decay, 1e-6)

        new_params = pos.to_option_params()
        # We need to manually construct new params since to_option_params uses fixed values
        from black_scholes import OptionParams, OptionType

        opt_type = OptionType.CALL if pos.option_type == "call" else OptionType.PUT
        new_params = OptionParams(
            spot=new_spot,
            strike=pos.strike,
            time_to_maturity=new_expiry,
            risk_free_rate=new_rate,
            volatility=new_vol,
            option_type=opt_type,
            dividend_yield=pos.dividend_yield,
        )

        from black_scholes import black_scholes_price

        old_price = black_scholes_price(params)
        new_price = black_scholes_price(new_params)
        exact_pnl = (new_price - old_price) * qty * 100

        residual = exact_pnl - total_first_order

        rows.append(
            {
                "symbol": pos.symbol,
                "strike": pos.strike,
                "expiry": pos.expiry,
                "type": pos.option_type,
                "quantity": qty,
                "delta_pnl": delta_pnl,
                "gamma_pnl": gamma_pnl,
                "vega_pnl": vega_pnl,
                "theta_pnl": theta_pnl,
                "rho_pnl": rho_pnl,
                "first_order_pnl": total_first_order,
                "exact_pnl": exact_pnl,
                "residual": residual,
            }
        )

    df = pd.DataFrame(rows)

    # Add total row
    if not df.empty:
        total_row = {
            "symbol": "TOTAL",
            "strike": np.nan,
            "expiry": np.nan,
            "type": "",
            "quantity": df["quantity"].sum(),
            "delta_pnl": df["delta_pnl"].sum(),
            "gamma_pnl": df["gamma_pnl"].sum(),
            "vega_pnl": df["vega_pnl"].sum(),
            "theta_pnl": df["theta_pnl"].sum(),
            "rho_pnl": df["rho_pnl"].sum(),
            "first_order_pnl": df["first_order_pnl"].sum(),
            "exact_pnl": df["exact_pnl"].sum(),
            "residual": df["residual"].sum(),
        }
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    return df


@dataclass(frozen=True, slots=True)
class FactorAttributionResult:
    """Results from factor attribution analysis."""

    factor_loadings: pd.DataFrame  # positions x factors
    factor_returns: pd.Series  # factor returns
    idiosyncratic: pd.Series  # idiosyncratic returns per position
    r_squared: pd.Series  # R² per position
    total_variance_explained: float

    def summary(self) -> pd.DataFrame:
        """Return summary statistics."""
        return pd.DataFrame(
            {
                "factor": self.factor_returns.index,
                "return": self.factor_returns.values,
                "loading_mean": self.factor_loadings.mean(axis=0).values,
                "loading_std": self.factor_loadings.std(axis=0).values,
            }
        )


def factor_attribution(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    method: Literal["pca", "style"] = "style",
    n_pca_factors: int = 3,
) -> FactorAttributionResult:
    """
    Factor attribution using PCA or style factors.

    Args:
        returns: DataFrame (dates x positions) of position returns.
        factors: DataFrame (dates x factors) of factor returns.
                 If method="pca", this is ignored and PCA is run on returns.
        method: "pca" for statistical factors, "style" for fundamental factors.
        n_pca_factors: Number of PCA factors to extract (if method="pca").

    Returns:
        FactorAttributionResult with loadings, factor returns, idiosyncratic, R².
    """
    returns = returns.dropna(axis=1, how="all")
    if returns.empty:
        raise ValueError("No valid return data after dropping NaN columns")

    if method == "pca":
        # Run PCA on returns covariance matrix
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "factor_attribution(method='pca') requires scikit-learn. "
                "Install it with 'pip install scikit-learn', or use "
                "method='style', which needs only numpy and scipy."
            ) from exc

        # Standardize returns
        returns_std = (returns - returns.mean()) / returns.std().replace(0, 1)

        pca = PCA(n_components=min(n_pca_factors, returns.shape[1]))
        factor_rets = pca.fit_transform(returns_std)
        factor_rets = pd.DataFrame(
            factor_rets,
            index=returns.index,
            columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
        )

        # Factor loadings = components * std(returns).
        # components_ is (n_components, n_features), so after the transpose the
        # per-asset volatility scales rows, not columns. Broadcasting against a
        # bare (n_features,) vector instead fails unless n_components happens to
        # equal n_features.
        loadings = pd.DataFrame(
            pca.components_.T * returns.std().values[:, None],
            index=returns.columns,
            columns=factor_rets.columns,
        )

        factor_returns = factor_rets.mean()

        # The result object carries idiosyncratic return and R-squared per
        # position, so both have to exist on this path too. Reconstructing each
        # position from its loadings on the principal components leaves a
        # residual that is, by construction, orthogonal to those components.
        explained = factor_rets.to_numpy() @ loadings.to_numpy().T
        residual = returns.to_numpy() - explained
        idiosyncratic = pd.Series(residual.mean(axis=0), index=returns.columns)

        total_variances = returns.var().replace(0, np.nan)
        # columns must be named: without them the frame defaults to integer
        # labels and the division against total_variances aligns to the union of
        # both indexes, yielding all-NaN R-squared.
        explained_variances = pd.DataFrame(
            explained, index=returns.index, columns=returns.columns
        ).var()
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = (explained_variances / total_variances).replace([np.inf, -np.nan], np.nan)
        r_squared = r2.fillna(0.0)

    else:  # style factors
        # Align factors and returns
        common_dates = returns.index.intersection(factors.index)
        if len(common_dates) < 10:
            raise ValueError("Insufficient overlapping dates between returns and factors")

        returns_aligned = returns.loc[common_dates]
        factors_aligned = factors.loc[common_dates]

        # OLS regression for each position
        loadings_list = []
        idio_list = []
        r2_list = []

        for col in returns_aligned.columns:
            y = returns_aligned[col].values
            X = factors_aligned.values
            X = np.column_stack([np.ones(len(X)), X])  # Add intercept

            # OLS: beta = (X'X)^-1 X'y
            try:
                # No conditioning argument: SciPy removed the `rcond` keyword in
                # 1.14 in favour of `cond`, so passing it by name broke this
                # path outright on current SciPy. The default is the same
                # machine-precision cutoff the caller intended, and omitting it
                # keeps the call working on both sides of the rename.
                beta = linalg.lstsq(X, y)[0]
                alpha = beta[0]
                betas = beta[1:]

                y_pred = X @ beta
                residuals = y - y_pred
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                loadings_list.append(betas)
                idio_list.append(alpha)  # alpha as idiosyncratic
                r2_list.append(max(0, r2))
            except linalg.LinAlgError:
                loadings_list.append(np.zeros(len(factors_aligned.columns)))
                idio_list.append(0.0)
                r2_list.append(0.0)

        loadings = pd.DataFrame(
            loadings_list,
            index=returns_aligned.columns,
            columns=factors_aligned.columns,
        )
        factor_returns = factors_aligned.mean()
        idiosyncratic = pd.Series(idio_list, index=returns_aligned.columns)
        r_squared = pd.Series(r2_list, index=returns_aligned.columns)

    # Total variance explained
    if method == "pca":
        total_var_explained = np.sum(pca.explained_variance_ratio_)
    else:
        # For style factors, use average R² weighted by variance
        var_weights = returns.var()
        total_var_explained = np.average(r_squared, weights=var_weights)

    return FactorAttributionResult(
        factor_loadings=loadings,
        factor_returns=factor_returns,
        idiosyncratic=idiosyncratic,
        r_squared=r_squared,
        total_variance_explained=total_var_explained,
    )


def generate_synthetic_portfolio(
    n_positions: int = 100,
    symbols: list[str] | None = None,
    seed: int = 42,
) -> list[Position]:
    """
    Generate a synthetic portfolio for testing.

    Args:
        n_positions: Number of positions to generate.
        symbols: List of symbols to use. Default: ["SPY", "QQQ", "IWM", "AAPL", "MSFT"].
        seed: Random seed for reproducibility.

    Returns:
        List of Position objects.
    """
    np.random.seed(seed)

    if symbols is None:
        symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

    positions = []
    for _ in range(n_positions):
        symbol = np.random.choice(symbols)
        spot = np.random.uniform(50, 500)
        strike = spot * np.random.uniform(0.8, 1.2)
        expiry = np.random.choice([1 / 52, 1 / 12, 2 / 12, 3 / 12, 6 / 12, 1.0])
        option_type = np.random.choice(["call", "put"])
        quantity = np.random.choice([-10, -5, -2, -1, 1, 2, 5, 10])
        implied_vol = np.random.uniform(0.15, 0.5)
        risk_free_rate = 0.05
        dividend_yield = np.random.uniform(0, 0.03)

        positions.append(
            Position(
                symbol=symbol,
                strike=strike,
                expiry=expiry,
                option_type=option_type,
                quantity=quantity,
                spot=spot,
                implied_vol=implied_vol,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            )
        )

    return positions
