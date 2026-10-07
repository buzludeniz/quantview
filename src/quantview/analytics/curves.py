"""
QuantView Analytics Curves Module.

Yield curve analytics: Nelson-Siegel-Svensson fitting, forward curves,
carry and roll-down calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True, slots=True)
class NSSParameters:
    """Nelson-Siegel-Svensson model parameters.

    The NSS model extends the Nelson-Siegel model with a second hump factor:
    r(t) = β0 + β1 * (1 - exp(-t/τ1)) / (t/τ1)
           + β2 * ((1 - exp(-t/τ1)) / (t/τ1) - exp(-t/τ1))
           + β3 * ((1 - exp(-t/τ2)) / (t/τ2) - exp(-t/τ2))

    Where:
    - β0: Long-term rate level
    - β1: Short-term rate component (slope)
    - β2: Medium-term curvature (first hump)
    - β3: Long-term curvature (second hump)
    - τ1: First decay parameter (short/medium term)
    - τ2: Second decay parameter (long term)
    """

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float

    def spot_rate(self, t: np.ndarray) -> np.ndarray:
        """Compute spot rates from NSS parameters.

        Args:
            t: Tenors in years.

        Returns:
            Continuously compounded spot rates.
        """
        t = np.asarray(t, dtype=float)
        tau1, tau2 = self.tau1, self.tau2

        # Handle t=0 using limiting values:
        # lim_{t->0} (1 - exp(-t/tau)) / (t/tau) = 1
        # lim_{t->0} ((1 - exp(-t/tau)) / (t/tau) - exp(-t/tau)) = 0
        ratio1 = t / tau1
        ratio2 = t / tau2

        # Both (1 - exp(-x)) / x and its derivative approach 1 at x = 0, so the
        # divide is masked rather than done on the whole array: np.where still
        # evaluates the untaken branch, which produced a divide-by-zero warning
        # for every tenor even though the result was correct.
        safe1 = np.where(ratio1 == 0, 1.0, ratio1)
        safe2 = np.where(ratio2 == 0, 1.0, ratio2)

        term1 = self.beta0
        term2 = self.beta1 * (1 - np.exp(-safe1)) / safe1
        term3 = self.beta2 * ((1 - np.exp(-safe1)) / safe1 - np.exp(-safe1))
        term4 = self.beta3 * ((1 - np.exp(-safe2)) / safe2 - np.exp(-safe2))

        return term1 + term2 + term3 + term4

    def forward_rate(self, t: np.ndarray) -> np.ndarray:
        """Compute instantaneous forward rates from NSS parameters.

        The instantaneous forward rate is:
        f(t) = β0 + β1 * exp(-t/τ1)
               + β2 * (t/τ1) * exp(-t/τ1)
               + β3 * (t/τ2) * exp(-t/τ2)

        Args:
            t: Tenors in years.

        Returns:
            Instantaneous forward rates.
        """
        t = np.asarray(t, dtype=float)
        tau1, tau2 = self.tau1, self.tau2

        term1 = self.beta0
        term2 = self.beta1 * np.exp(-t / tau1)
        term3 = self.beta2 * (t / tau1) * np.exp(-t / tau1)
        term4 = self.beta3 * (t / tau2) * np.exp(-t / tau2)

        return term1 + term2 + term3 + term4

    def discount_factor(self, t: np.ndarray) -> np.ndarray:
        """Compute discount factors from spot rates.

        Args:
            t: Tenors in years.

        Returns:
            Discount factors (ZCB prices).
        """
        return np.exp(-self.spot_rate(t) * t)

    def par_rate(self, t: np.ndarray | float) -> np.ndarray | float:
        """Compute par rates from NSS parameters.

        The par rate is the coupon rate that makes a bond price equal to par.
        For continuous compounding: par = (1 - D(t)) / ∫₀ᵗ D(s) ds

        Args:
            t: One tenor in years, or an array of tenors.

        Returns:
            A float for a scalar tenor, otherwise the annualized par rates
            (continuous compounding) as an array.
        """
        t_arr = np.asarray(t, dtype=float)
        # A scalar tenor produced a 0-d array, and the loop below then raised
        # "iteration over a 0-d array". at least_1d keeps a scalar working.
        scalar = t_arr.ndim == 0
        t_arr = np.atleast_1d(t_arr)
        df = self.discount_factor(t_arr)

        # Numerical integration of discount factors for each tenor
        # Using trapezoidal rule on a fine grid per tenor
        par_rates = np.zeros_like(t_arr)
        for i, ti in enumerate(t_arr):
            if ti <= 0:
                par_rates[i] = self.spot_rate(np.array([1e-6]))[0]  # Approximate
                continue
            n_points = 200
            t_grid = np.linspace(0, ti, n_points)
            df_grid = self.discount_factor(t_grid)
            integral = np.trapezoid(df_grid, t_grid)
            par_rates[i] = (1 - df[i]) / integral if integral > 0 else 0.0

        return float(par_rates[0]) if scalar else par_rates

    def to_array(self) -> np.ndarray:
        """Convert to parameter array for optimization."""
        return np.array([self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> NSSParameters:
        """Create from parameter array."""
        return cls(beta0=arr[0], beta1=arr[1], beta2=arr[2], beta3=arr[3], tau1=arr[4], tau2=arr[5])


def nss_spot_rate(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute NSS spot rates from parameter array.

    Args:
        params: [β0, β1, β2, β3, τ1, τ2]
        t: Tenors in years.

    Returns:
        Spot rates.
    """
    beta0, beta1, beta2, beta3, tau1, tau2 = params
    t = np.asarray(t, dtype=float)

    ratio1 = t / tau1
    ratio2 = t / tau2

    term1 = beta0
    term2 = np.where(ratio1 == 0, beta1, beta1 * (1 - np.exp(-ratio1)) / ratio1)
    term3 = np.where(ratio1 == 0, 0.0, beta2 * ((1 - np.exp(-ratio1)) / ratio1 - np.exp(-ratio1)))
    term4 = np.where(ratio2 == 0, 0.0, beta3 * ((1 - np.exp(-ratio2)) / ratio2 - np.exp(-ratio2)))

    return term1 + term2 + term3 + term4


def nss_forward_rate(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute NSS forward rates from parameter array.

    Args:
        params: [β0, β1, β2, β3, τ1, τ2]
        t: Tenors in years.

    Returns:
        Forward rates.
    """
    beta0, beta1, beta2, beta3, tau1, tau2 = params
    t = np.asarray(t, dtype=float)

    term1 = beta0
    term2 = beta1 * np.exp(-t / tau1)
    term3 = beta2 * (t / tau1) * np.exp(-t / tau1)
    term4 = beta3 * (t / tau2) * np.exp(-t / tau2)

    return term1 + term2 + term3 + term4


def nss_residuals(params: np.ndarray, tenors: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Compute residuals between NSS model and observed rates.

    Args:
        params: [β0, β1, β2, β3, τ1, τ2]
        tenors: Observed tenors in years.
        rates: Observed spot rates.

    Returns:
        Residuals (model - observed).
    """
    model_rates = nss_spot_rate(params, tenors)
    return model_rates - rates


def fit_nss(
    tenors: np.ndarray,
    rates: np.ndarray,
    initial_guess: np.ndarray | None = None,
    bounds: list[tuple[float | None, float | None]] | None = None,
    method: str = "L-BFGS-B",
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> NSSParameters:
    """Fit Nelson-Siegel-Svensson model to observed yield curve.

    Minimizes sum of squared residuals between model and observed spot rates.

    Args:
        tenors: Tenors in years (e.g., [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]).
        rates: Observed spot rates (continuously compounded, e.g., [0.052, 0.050, ...]).
        initial_guess: Initial parameter guess [β0, β1, β2, β3, τ1, τ2].
        bounds: Parameter bounds. Default: β unrestricted, τ ∈ (0.1, 30).
        method: Optimization method (default: L-BFGS-B).
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        NSSParameters with fitted parameters.

    Raises:
        ValueError: If optimization fails or inputs invalid.

    Example:
        >>> tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
        >>> rates = np.array([0.052, 0.050, 0.048, 0.046, 0.045, 0.044, 0.044, 0.0445, 0.0455, 0.046])
        >>> nss = fit_nss(tenors, rates)
        >>> print(nss.beta0, nss.tau1, nss.tau2)
    """
    tenors = np.asarray(tenors, dtype=float)
    rates = np.asarray(rates, dtype=float)

    if len(tenors) != len(rates):
        raise ValueError("tenors and rates must have same length")
    if len(tenors) < 4:
        raise ValueError("Need at least 4 tenors for NSS fit (6 parameters)")

    # Default initial guess based on typical yield curve shapes
    if initial_guess is None:
        # β0 ≈ long rate (30Y)
        beta0 = rates[-1] if len(rates) > 0 else 0.04
        # β1 ≈ short rate - long rate (slope)
        beta1 = rates[0] - beta0 if len(rates) > 0 else -0.01
        # β2, β3: small curvature terms
        beta2 = -0.01
        beta3 = 0.005
        # τ1: short/medium decay (1-3 years)
        tau1 = 2.0
        # τ2: long decay (5-15 years)
        tau2 = 10.0
        initial_guess = np.array([beta0, beta1, beta2, beta3, tau1, tau2])

    # Default bounds: β unrestricted, τ > 0.1. SciPy takes None for an open
    # bound, so the pairs are deliberately nullable.
    if bounds is None:
        bounds = [
            (None, None),  # β0
            (None, None),  # β1
            (None, None),  # β2
            (None, None),  # β3
            (0.1, 30.0),  # τ1
            (0.1, 30.0),  # τ2
        ]

    def objective(params: np.ndarray) -> float:
        residuals = nss_residuals(params, tenors, rates)
        return np.sum(residuals**2)

    result = minimize(
        objective,
        initial_guess,
        method=method,
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )

    if not result.success:
        raise ValueError(f"NSS optimization failed: {result.message}")

    return NSSParameters.from_array(result.x)


def forward_curve(nss_params: NSSParameters, tenors: np.ndarray) -> np.ndarray:
    """Compute forward curve from NSS parameters.

    Args:
        nss_params: Fitted NSS parameters.
        tenors: Tenors in years for forward rate output.

    Returns:
        Instantaneous forward rates at each tenor.
    """
    return nss_params.forward_rate(tenors)


def par_curve(nss_params: NSSParameters, tenors: np.ndarray) -> np.ndarray:
    """Compute par curve from NSS parameters.

    Args:
        nss_params: Fitted NSS parameters.
        tenors: Tenors in years for par rate output.

    Returns:
        Par rates at each tenor. Always an array, even for a one-element input,
        so the shape matches the tenor grid that was passed in.
    """
    # par_rate returns a float for a scalar, but this function is vector-shaped
    # by contract: its result is indexed positionally alongside the tenors.
    result = nss_params.par_rate(np.atleast_1d(tenors))
    return np.asarray(result, dtype=float)


def carry_rolldown(
    spot_curve: np.ndarray,
    forward_curve: np.ndarray,
    horizon: float,
    tenors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate carry, roll-down, and total carry+roll for a yield curve.

    Carry = Forward rate - Spot rate (approximation for carry over horizon)
    Roll-down = Spot rate at (tenor - horizon) - Spot rate at tenor
    Total = Carry + Roll-down

    Args:
        spot_curve: Current spot rates at each tenor.
        forward_curve: Forward rates at each tenor.
        horizon: Investment horizon in years.
        tenors: Tenor points in years. Required if spot_curve not evenly spaced.

    Returns:
        Tuple of (carry, roll_down, total_carry_roll) arrays.

    Example:
        >>> tenors = np.array([1, 2, 3, 5, 7, 10])
        >>> spot = np.array([0.04, 0.042, 0.043, 0.044, 0.0445, 0.045])
        >>> fwd = np.array([0.044, 0.044, 0.044, 0.044, 0.045, 0.046])
        >>> carry, roll, total = carry_rolldown(spot, fwd, horizon=1.0, tenors=tenors)
    """
    spot_curve = np.asarray(spot_curve, dtype=float)
    forward_curve = np.asarray(forward_curve, dtype=float)

    if len(spot_curve) != len(forward_curve):
        raise ValueError("spot_curve and forward_curve must have same length")

    # Carry = forward - spot (approximate carry over horizon)
    carry = forward_curve - spot_curve

    # Roll-down: need spot at (tenor - horizon)
    if tenors is None:
        # Assume unit spacing
        tenors = np.arange(1, len(spot_curve) + 1, dtype=float)

    tenors = np.asarray(tenors, dtype=float)

    # Interpolate spot curve at shifted tenors
    from scipy.interpolate import interp1d

    # Extend interpolation to handle extrapolation
    interp = interp1d(
        tenors,
        spot_curve,
        kind="cubic",
        fill_value="extrapolate",
        bounds_error=False,
    )

    shifted_tenors = np.maximum(tenors - horizon, tenors.min())
    spot_shifted = interp(shifted_tenors)

    # Roll-down = spot(tenor - horizon) - spot(tenor)
    # Positive roll-down means rates fall (profit for long position)
    roll_down = spot_shifted - spot_curve

    total = carry + roll_down

    return carry, roll_down, total


def carry_rolldown_matrix(
    spot_curve: np.ndarray,
    forward_curve: np.ndarray,
    tenors: np.ndarray,
    horizons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate carry/roll-down matrix for multiple horizons.

    Args:
        spot_curve: Current spot rates at each tenor.
        forward_curve: Forward rates at each tenor.
        tenors: Tenor points in years.
        horizons: Investment horizons in years.

    Returns:
        Tuple of (carry_matrix, roll_matrix, total_matrix) where each
        matrix has shape (n_horizons, n_tenors).
    """
    n_horizons = len(horizons)
    n_tenors = len(tenors)

    carry_matrix = np.zeros((n_horizons, n_tenors))
    roll_matrix = np.zeros((n_horizons, n_tenors))
    total_matrix = np.zeros((n_horizons, n_tenors))

    for i, h in enumerate(horizons):
        carry, roll, total = carry_rolldown(spot_curve, forward_curve, h, tenors)
        carry_matrix[i, :] = carry
        roll_matrix[i, :] = roll
        total_matrix[i, :] = total

    return carry_matrix, roll_matrix, total_matrix


# Convenience function to match the NSSFit interface from charts.py
class NSSFitLike(Protocol):
    """The shape nss_fit_to_params needs: the six NSS coefficients.

    Declared here rather than importing NSSFit from the notebook module, so the
    analytics layer does not depend on the notebook layer.
    """

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float


def nss_fit_to_params(nss_fit: NSSFitLike) -> NSSParameters:
    """Convert a fitted NSS parameter set into an NSSParameters curve.

    Args:
        nss_fit: Any object exposing the six NSS coefficients, including the
            NSSFit dataclass from quantview.notebook.charts.

    Returns:
        NSSParameters instance.
    """
    return NSSParameters(
        beta0=nss_fit.beta0,
        beta1=nss_fit.beta1,
        beta2=nss_fit.beta2,
        beta3=nss_fit.beta3,
        tau1=nss_fit.tau1,
        tau2=nss_fit.tau2,
    )


if __name__ == "__main__":
    # Demo with synthetic Treasury-like curve
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    rates = np.array([0.052, 0.050, 0.048, 0.046, 0.045, 0.044, 0.044, 0.0445, 0.0455, 0.046])

    print("Fitting NSS to synthetic Treasury curve...")
    nss = fit_nss(tenors, rates)
    print(f"β0={nss.beta0:.6f}, β1={nss.beta1:.6f}, β2={nss.beta2:.6f}, β3={nss.beta3:.6f}")
    print(f"τ1={nss.tau1:.6f}, τ2={nss.tau2:.6f}")

    # Check fit
    fitted_rates = nss.spot_rate(tenors)
    rmse = np.sqrt(np.mean((fitted_rates - rates) ** 2))
    print(f"RMSE: {rmse:.6f}")

    # Forward curve
    fwd = nss.forward_rate(tenors)
    print("\nForward rates:")
    for t, s, f in zip(tenors, rates, fwd, strict=False):
        print(f"  {t:>4.1f}Y: spot={s:.4%}, fwd={f:.4%}")

    # Carry/roll-down
    carry, roll, total = carry_rolldown(rates, fwd, horizon=1.0, tenors=tenors)
    print("\nCarry/Roll-down (1Y horizon):")
    for t, c, r, tot in zip(tenors, carry, roll, total, strict=False):
        print(f"  {t:>4.1f}Y: carry={c:.4%}, roll={r:.4%}, total={tot:.4%}")

    # Multi-horizon matrix
    horizons = np.array([0.25, 0.5, 1.0, 2.0])
    carry_m, roll_m, total_m = carry_rolldown_matrix(rates, fwd, tenors, horizons)
    print(f"\nCarry matrix shape: {carry_m.shape}")
    print(f"Roll matrix shape: {roll_m.shape}")
    print(f"Total matrix shape: {total_m.shape}")
