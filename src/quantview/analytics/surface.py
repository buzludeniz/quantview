"""
QuantView Analytics Surface Module.

Utilities for building VolSurface objects from options chain data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

# Explicitly bind to None so the names exist in the module namespace when
# the optional dependency is absent. The try/except rebinds on success.
SVICurve: Any = None
SVIFit: Any = None
VolSurface: Any = None
OptionParams: Any = None
OptionType: Any = None
black_scholes_price: Any = None
implied_volatility: Any = None
try:
    from black_scholes import (  # noqa: F401 - probed for availability, used only as a guard
        OptionParams,
        OptionType,
        black_scholes_price,
        implied_volatility,
    )
    from black_scholes.surface import (  # noqa: F401 - probed for availability, used only as a guard
        SVICurve,
        SVIFit,
        VolSurface,
    )
except ImportError:
    pass


@dataclass(frozen=True, slots=True)
class ChainQuote:
    """Single option quote from an options chain."""

    strike: float
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int
    implied_vol: float | None = None
    option_type: Literal["call", "put"] = "call"


def chain_to_volsurface(
    spot: float,
    chain: pd.DataFrame,
    expiry: float,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
    option_type: Literal["call", "put"] = "call",
    price_column: Literal["mid", "bid", "ask"] = "mid",
    min_volume: int = 0,
    min_oi: int = 0,
) -> VolSurface:
    """
    Build a VolSurface from an options chain DataFrame.

    Args:
        spot: Current spot price of the underlying.
        chain: DataFrame with columns ['strike', 'bid', 'ask', 'volume', 'open_interest']
               Optionally 'implied_vol' if pre-computed.
        expiry: Time to expiry in years.
        risk_free_rate: Risk-free rate.
        dividend_yield: Dividend yield.
        option_type: 'call' or 'put'.
        price_column: Which price to use for IV inversion ('mid', 'bid', 'ask').
        min_volume: Minimum volume filter.
        min_oi: Minimum open interest filter.

    Returns:
        VolSurface instance.

    Raises:
        ValueError: If required columns missing or black_scholes unavailable.
    """
    if VolSurface is None:
        raise ImportError("black_scholes package required for chain_to_volsurface")

    required_cols = ["strike", "bid", "ask", "volume", "open_interest"]
    for col in required_cols:
        if col not in chain.columns:
            raise ValueError(f"Missing required column: {col}")

    # Filter by volume and OI
    df = chain.copy()
    df = df[(df["volume"] >= min_volume) & (df["open_interest"] >= min_oi)]
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]

    if len(df) < 2:
        raise ValueError("Insufficient quotes after filtering (need at least 2)")

    # Compute mid price if not present
    if price_column == "mid":
        if "mid" not in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2
        prices = df["mid"].values
    elif price_column == "bid":
        prices = df["bid"].values
    elif price_column == "ask":
        prices = df["ask"].values
    else:
        raise ValueError(f"Unknown price_column: {price_column}")

    strikes = df["strike"].values

    # Use pre-computed IV if available
    if "implied_vol" in df.columns and df["implied_vol"].notna().any():
        # copy=True is required, not incidental. pandas 3.0 enables copy-on-write
        # by default, where `.values` can hand back a read-only view, so filling
        # the gaps below raised "assignment destination is read-only".
        vols = df["implied_vol"].to_numpy(dtype=float, copy=True)
        # Fill any NaN with inversion
        mask = ~np.isfinite(vols)
        if mask.any():
            opt_type = OptionType.CALL if option_type == "call" else OptionType.PUT
            for i in np.where(mask)[0]:
                try:
                    params = OptionParams(
                        spot=spot,
                        strike=strikes[i],
                        time_to_maturity=expiry,
                        risk_free_rate=risk_free_rate,
                        volatility=0.2,
                        option_type=opt_type,
                        dividend_yield=dividend_yield,
                    )
                    vols[i] = implied_volatility(prices[i], params)
                except ValueError:
                    vols[i] = np.nan
        # Drop any remaining NaN
        valid = np.isfinite(vols)
        strikes = strikes[valid]
        vols = vols[valid]
    else:
        # Invert all prices to implied vols
        opt_type = OptionType.CALL if option_type == "call" else OptionType.PUT
        vols = []
        valid_strikes = []
        for K, price in zip(strikes, prices, strict=False):
            params = OptionParams(
                spot=spot,
                strike=K,
                time_to_maturity=expiry,
                risk_free_rate=risk_free_rate,
                volatility=0.2,
                option_type=opt_type,
                dividend_yield=dividend_yield,
            )
            try:
                iv = implied_volatility(price, params)
                vols.append(iv)
                valid_strikes.append(K)
            except ValueError:
                continue
        strikes = np.array(valid_strikes)
        vols = np.array(vols)

    if len(strikes) < 2:
        raise ValueError("Could not compute valid implied volatilities for enough strikes")

    return VolSurface(
        spot=spot,
        strikes=tuple(strikes),
        implied_vols=tuple(vols),
        time_to_maturity=expiry,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        option_type=OptionType.CALL if option_type == "call" else OptionType.PUT,
    )


def chain_to_volsurface_multi(
    spot: float,
    chains: dict[float, pd.DataFrame],
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
    option_type: Literal["call", "put"] = "call",
    price_column: Literal["mid", "bid", "ask"] = "mid",
    min_volume: int = 0,
    min_oi: int = 0,
) -> dict[float, VolSurface]:
    """
    Build VolSurface objects for multiple expiries from a dict of chains.

    Args:
        spot: Current spot price.
        chains: Dict mapping expiry (years) -> chain DataFrame.
        risk_free_rate: Risk-free rate.
        dividend_yield: Dividend yield.
        option_type: 'call' or 'put'.
        price_column: Price column for IV inversion.
        min_volume: Minimum volume filter.
        min_oi: Minimum open interest filter.

    Returns:
        Dict mapping expiry -> VolSurface.
    """
    surfaces = {}
    for expiry, chain in chains.items():
        try:
            surfaces[expiry] = chain_to_volsurface(
                spot=spot,
                chain=chain,
                expiry=expiry,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                option_type=option_type,
                price_column=price_column,
                min_volume=min_volume,
                min_oi=min_oi,
            )
        except (ValueError, ImportError):
            # Skip expiries that fail
            continue
    return surfaces


def build_chain_dataframe(
    strikes: list[float] | np.ndarray,
    bids: list[float] | np.ndarray,
    asks: list[float] | np.ndarray,
    volumes: list[int] | np.ndarray,
    open_interests: list[int] | np.ndarray,
    implied_vols: list[float] | np.ndarray | None = None,
    option_type: Literal["call", "put"] = "call",
) -> pd.DataFrame:
    """
    Build a standard options chain DataFrame from raw arrays.

    Args:
        strikes: Strike prices.
        bids: Bid prices.
        asks: Ask prices.
        volumes: Volume.
        open_interests: Open interest.
        implied_vols: Optional pre-computed implied volatilities.
        option_type: 'call' or 'put'.

    Returns:
        DataFrame with standard chain columns.
    """
    df = pd.DataFrame(
        {
            "strike": np.asarray(strikes),
            "bid": np.asarray(bids),
            "ask": np.asarray(asks),
            "volume": np.asarray(volumes, dtype=int),
            "open_interest": np.asarray(open_interests, dtype=int),
        }
    )
    df["mid"] = (df["bid"] + df["ask"]) / 2
    if implied_vols is not None:
        df["implied_vol"] = np.asarray(implied_vols)
    df["option_type"] = option_type
    return df


def compute_chain_greeks(
    surface: VolSurface,
    expiry: float,
    strikes: np.ndarray | None = None,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """
    Compute Greeks for all strikes in a chain at a given expiry.

    Args:
        surface: VolSurface for implied vols.
        expiry: Time to expiry in years.
        strikes: Optional array of strikes (uses surface strikes if None).
        risk_free_rate: Risk-free rate.
        dividend_yield: Dividend yield.

    Returns:
        DataFrame with columns: Strike, Moneyness, IV, Call/Put Price, Delta, Gamma, Vega, Theta, Rho.
    """
    if VolSurface is None:
        raise ImportError("black_scholes package required for compute_chain_greeks")

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
        moneyness = K / F

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
                "Call Rho": call_greeks.rho,
                "Put Price": put_price,
                "Put Delta": put_greeks.delta,
                "Put Gamma": put_greeks.gamma,
                "Put Vega": put_greeks.vega,
                "Put Theta": put_greeks.theta,
                "Put Rho": put_greeks.rho,
            }
        )

    return pd.DataFrame(data)


def compute_svi_parameters_per_expiry(
    surfaces: dict[float, VolSurface],
) -> pd.DataFrame:
    """
    Compute SVI Jump-Wings parameters for each expiry surface.

    Args:
        surfaces: Dict mapping expiry -> VolSurface.

    Returns:
        DataFrame with columns: Expiry, ATM Variance, ATM Skew, Put Wing Slope,
        Call Wing Slope, Min Variance.
    """
    if VolSurface is None:
        raise ImportError("black_scholes package required")

    rows = []
    for expiry, surface in sorted(surfaces.items()):
        try:
            fit = surface.fit()
            jw = fit.curve.jump_wings_parameters(expiry)
            rows.append(
                {
                    "Expiry": expiry,
                    "ATM Variance": jw["atm_variance"],
                    "ATM Skew": jw["atm_skew"],
                    "Put Wing Slope": jw["put_wing_slope"],
                    "Call Wing Slope": jw["call_wing_slope"],
                    "Min Variance": jw["min_variance"],
                    "RMS Error": fit.rms_error,
                    "Max Error": fit.max_error,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def compute_term_structure(
    surfaces: dict[float, VolSurface],
    delta: float = 0.25,
) -> pd.DataFrame:
    """
    Compute term structure of ATM vol, skew, and curvature.

    Args:
        surfaces: Dict mapping expiry -> VolSurface.
        delta: Delta for skew calculation (e.g., 0.25 for 25-delta).

    Returns:
        DataFrame with columns: Expiry, ATM Vol, 25d Skew, Curvature.
    """
    if VolSurface is None:
        raise ImportError("black_scholes package required")

    rows = []
    for expiry, surface in sorted(surfaces.items()):
        try:
            fit = surface.fit()
            T = expiry

            # ATM vol
            atm_iv = fit.curve.implied_volatility(0.0, T)

            # 25-delta skew: find strikes for 25-delta call and put
            # For call: delta = N(d1) = 0.25 => d1 = N^-1(0.25) ≈ -0.674
            # For put: delta = N(d1) - 1 = -0.25 => d1 = N^-1(0.75) ≈ 0.674
            from scipy.stats import norm

            d1_call = norm.ppf(delta)
            d1_put = norm.ppf(1 - delta)

            # d1 = (ln(F/K) + 0.5*σ²T) / (σ√T)
            # Solve for K: ln(F/K) = d1*σ√T - 0.5*σ²T
            # This is implicit since σ depends on K. Use ATM vol as approximation.
            sigma_atm = atm_iv
            sqrt_T = np.sqrt(T)

            # Iterate to find consistent strike for 25-delta
            k_call = d1_call * sigma_atm * sqrt_T - 0.5 * sigma_atm**2 * T
            k_put = d1_put * sigma_atm * sqrt_T - 0.5 * sigma_atm**2 * T

            # Get IV at these log-moneyness
            iv_call_25d = fit.curve.implied_volatility(k_call, T)
            iv_put_25d = fit.curve.implied_volatility(k_put, T)

            # 25-delta skew = (IV_put_25d - IV_call_25d) / (2 * ATM_IV)
            skew_25d = (iv_put_25d - iv_call_25d) / (2 * atm_iv) if atm_iv > 0 else 0

            # Curvature (butterfly): 25d fly = IV_call_25d + IV_put_25d - 2*ATM_IV
            curvature = (iv_call_25d + iv_put_25d) / 2 - atm_iv

            jw = fit.curve.jump_wings_parameters(T)

            rows.append(
                {
                    "Expiry": expiry,
                    "ATM Vol": atm_iv,
                    f"{int(delta * 100)}d Skew": skew_25d,
                    "Curvature": curvature,
                    "ATM Variance": jw["atm_variance"],
                    "ATM Skew (JW)": jw["atm_skew"],
                    "Put Wing": jw["put_wing_slope"],
                    "Call Wing": jw["call_wing_slope"],
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Demo
    print("QuantView Analytics Surface Module")
    print("Functions:")
    print("  - chain_to_volsurface()")
    print("  - chain_to_volsurface_multi()")
    print("  - build_chain_dataframe()")
    print("  - compute_chain_greeks()")
    print("  - compute_svi_parameters_per_expiry()")
    print("  - compute_term_structure()")
