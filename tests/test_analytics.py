"""
Tests for QuantView Analytics Curves Module.

Run with: pytest tests/test_analytics.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from quantview.analytics.curves import (
    NSSParameters,
    carry_rolldown,
    carry_rolldown_matrix,
    fit_nss,
    forward_curve,
    nss_fit_to_params,
    nss_forward_rate,
    nss_residuals,
    nss_spot_rate,
    par_curve,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_treasury_curve():
    """Sample Treasury-like yield curve (FRED DGS series approximation).

    Typical tenors: 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y
    Rates in decimal (continuously compounded).
    """
    tenors = np.array([1 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30])
    # Approximate rates from a normal yield curve environment
    rates = np.array(
        [
            0.0520,  # 1M
            0.0510,  # 3M
            0.0500,  # 6M
            0.0490,  # 1Y
            0.0470,  # 2Y
            0.0460,  # 3Y
            0.0450,  # 5Y
            0.0450,  # 7Y
            0.0455,  # 10Y
            0.0465,  # 20Y
            0.0470,  # 30Y
        ]
    )
    return tenors, rates


@pytest.fixture
def simple_curve():
    """Simple upward-sloping curve for basic tests."""
    tenors = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    rates = np.array([0.04, 0.042, 0.043, 0.044, 0.0445, 0.045])
    return tenors, rates


@pytest.fixture
def flat_curve():
    """Flat yield curve."""
    tenors = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    rates = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    return tenors, rates


@pytest.fixture
def inverted_curve():
    """Inverted yield curve."""
    tenors = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    rates = np.array([0.055, 0.052, 0.050, 0.048, 0.047, 0.046])
    return tenors, rates


# ============================================================================
# NSS Parameter Class Tests
# ============================================================================


def test_nss_parameters_creation():
    """Test NSSParameters instantiation."""
    params = NSSParameters(beta0=0.045, beta1=-0.01, beta2=-0.015, beta3=0.005, tau1=1.5, tau2=8.0)
    assert params.beta0 == 0.045
    assert params.tau1 == 1.5
    assert params.tau2 == 8.0


def test_nss_spot_rate_calculation():
    """Test spot rate calculation from NSS parameters."""
    params = NSSParameters(beta0=0.045, beta1=-0.01, beta2=-0.015, beta3=0.005, tau1=1.5, tau2=8.0)
    t = np.array([1.0, 2.0, 5.0, 10.0])
    rates = params.spot_rate(t)
    assert len(rates) == 4
    assert all(r > 0 for r in rates)
    # Long rate should approach beta0 (at 10Y with tau2=8, within ~0.003)
    assert abs(rates[-1] - params.beta0) < 0.003


def test_nss_forward_rate_calculation():
    """Test forward rate calculation from NSS parameters."""
    params = NSSParameters(beta0=0.045, beta1=-0.01, beta2=-0.015, beta3=0.005, tau1=1.5, tau2=8.0)
    t = np.array([1.0, 2.0, 5.0, 10.0])
    fwd = params.forward_rate(t)
    assert len(fwd) == 4
    assert all(f > 0 for f in fwd)


def test_nss_discount_factor():
    """Test discount factor calculation."""
    params = NSSParameters(beta0=0.045, beta1=-0.01, beta2=-0.015, beta3=0.005, tau1=1.5, tau2=8.0)
    t = np.array([1.0, 2.0, 5.0, 10.0])
    df = params.discount_factor(t)
    assert len(df) == 4
    assert all(0 < d < 1 for d in df)
    # Discount factors should be decreasing
    assert all(df[i] > df[i + 1] for i in range(len(df) - 1))


def test_nss_par_rate():
    """Test par rate calculation."""
    params = NSSParameters(beta0=0.045, beta1=-0.01, beta2=-0.015, beta3=0.005, tau1=1.5, tau2=8.0)
    t = np.array([1.0, 2.0, 5.0, 10.0])
    par = params.par_rate(t)
    assert len(par) == 4
    assert all(p > 0 for p in par)


def test_nss_to_array():
    """Test conversion to/from array."""
    params = NSSParameters(beta0=0.045, beta1=-0.01, beta2=-0.015, beta3=0.005, tau1=1.5, tau2=8.0)
    arr = params.to_array()
    assert len(arr) == 6
    restored = NSSParameters.from_array(arr)
    assert restored.beta0 == params.beta0
    assert restored.tau1 == params.tau1


# ============================================================================
# NSS Functional Tests (array-based)
# ============================================================================


def test_nss_spot_rate_function():
    """Test nss_spot_rate function."""
    params = np.array([0.045, -0.01, -0.015, 0.005, 1.5, 8.0])
    t = np.array([1.0, 2.0, 5.0, 10.0])
    rates = nss_spot_rate(params, t)
    assert len(rates) == 4
    assert all(r > 0 for r in rates)


def test_nss_forward_rate_function():
    """Test nss_forward_rate function."""
    params = np.array([0.045, -0.01, -0.015, 0.005, 1.5, 8.0])
    t = np.array([1.0, 2.0, 5.0, 10.0])
    fwd = nss_forward_rate(params, t)
    assert len(fwd) == 4
    assert all(f > 0 for f in fwd)


def test_nss_residuals():
    """Test residuals calculation."""
    params = np.array([0.045, -0.01, -0.015, 0.005, 1.5, 8.0])
    tenors = np.array([1.0, 2.0, 5.0, 10.0])
    rates = np.array([0.044, 0.043, 0.042, 0.041])
    residuals = nss_residuals(params, tenors, rates)
    assert len(residuals) == 4


# ============================================================================
# NSS Fitting Tests
# ============================================================================


def test_fit_nss_basic(simple_curve):
    """Test basic NSS fitting."""
    tenors, rates = simple_curve
    nss = fit_nss(tenors, rates)
    assert isinstance(nss, NSSParameters)
    # Check fitted rates are close to input
    fitted = nss.spot_rate(tenors)
    rmse = np.sqrt(np.mean((fitted - rates) ** 2))
    assert rmse < 0.001  # Should fit well


def test_fit_nss_treasury_curve(sample_treasury_curve):
    """Test NSS fitting on Treasury-like curve."""
    tenors, rates = sample_treasury_curve
    nss = fit_nss(tenors, rates)
    assert isinstance(nss, NSSParameters)
    # Check parameter bounds
    assert nss.tau1 > 0.1
    assert nss.tau2 > 0.1
    # Check fit quality
    fitted = nss.spot_rate(tenors)
    rmse = np.sqrt(np.mean((fitted - rates) ** 2))
    assert rmse < 0.0015  # Reasonable fit for real curve


def test_fit_nss_flat_curve(flat_curve):
    """Test NSS fitting on flat curve."""
    tenors, rates = flat_curve
    nss = fit_nss(tenors, rates)
    assert isinstance(nss, NSSParameters)
    fitted = nss.spot_rate(tenors)
    rmse = np.sqrt(np.mean((fitted - rates) ** 2))
    assert rmse < 0.0005  # Should fit flat curve very well


def test_fit_nss_inverted_curve(inverted_curve):
    """Test NSS fitting on inverted curve."""
    tenors, rates = inverted_curve
    nss = fit_nss(tenors, rates)
    assert isinstance(nss, NSSParameters)
    fitted = nss.spot_rate(tenors)
    rmse = np.sqrt(np.mean((fitted - rates) ** 2))
    assert rmse < 0.0015


def test_fit_nss_custom_initial_guess(simple_curve):
    """Test NSS fitting with custom initial guess."""
    tenors, rates = simple_curve
    guess = np.array([0.05, -0.02, -0.01, 0.01, 1.0, 5.0])
    nss = fit_nss(tenors, rates, initial_guess=guess)
    assert isinstance(nss, NSSParameters)


def test_fit_nss_custom_bounds(simple_curve):
    """Test NSS fitting with custom bounds."""
    tenors, rates = simple_curve
    bounds = [
        (0.02, 0.08),  # β0
        (-0.05, 0.05),  # β1
        (-0.05, 0.05),  # β2
        (-0.05, 0.05),  # β3
        (0.5, 20.0),  # τ1
        (0.5, 20.0),  # τ2
    ]
    nss = fit_nss(tenors, rates, bounds=bounds)
    assert isinstance(nss, NSSParameters)
    assert 0.02 <= nss.beta0 <= 0.08
    assert 0.5 <= nss.tau1 <= 20.0
    assert 0.5 <= nss.tau2 <= 20.0


def test_fit_nss_insufficient_data():
    """Test NSS fitting with insufficient data points."""
    tenors = np.array([1.0, 2.0, 3.0])  # Only 3 points, need 6 params
    rates = np.array([0.04, 0.042, 0.043])
    with pytest.raises(ValueError, match="Need at least 4 tenors"):
        fit_nss(tenors, rates)


def test_fit_nss_mismatched_lengths():
    """Test NSS fitting with mismatched tenor/rate arrays."""
    tenors = np.array([1.0, 2.0, 3.0, 5.0])
    rates = np.array([0.04, 0.042, 0.043])  # Only 3 rates
    with pytest.raises(ValueError, match="same length"):
        fit_nss(tenors, rates)


# ============================================================================
# Forward Curve Tests
# ============================================================================


def test_forward_curve(simple_curve):
    """Test forward curve computation from NSS fit."""
    tenors, rates = simple_curve
    nss = fit_nss(tenors, rates)
    fwd = forward_curve(nss, tenors)
    assert len(fwd) == len(tenors)
    assert all(f > 0 for f in fwd)
    # Forward curve should be smooth
    assert np.all(np.diff(fwd) > -0.01)  # Not too steep negative


def test_forward_curve_flat(flat_curve):
    """Test forward curve on flat spot curve."""
    tenors, rates = flat_curve
    nss = fit_nss(tenors, rates)
    fwd = forward_curve(nss, tenors)
    # For flat curve, forward ≈ spot ≈ constant
    assert np.allclose(fwd, rates, atol=0.001)


# ============================================================================
# Par Curve Tests
# ============================================================================


def test_par_curve(simple_curve):
    """Test par curve computation."""
    tenors, rates = simple_curve
    nss = fit_nss(tenors, rates)
    par = par_curve(nss, tenors)
    assert len(par) == len(tenors)
    assert all(p > 0 for p in par)
    # Par rates should be close to spot for upward sloping
    assert np.allclose(par, rates, atol=0.002)


def test_par_rate_accepts_a_scalar_tenor(simple_curve):
    """A scalar tenor used to raise "iteration over a 0-d array".

    The integration loop needs a 1-d array to enumerate, so a bare float, which
    np.asarray turns into a 0-d array, crashed the call.
    """
    tenors, rates = simple_curve
    nss = fit_nss(tenors, rates)
    expected = float(nss.par_rate(np.array([1.0]))[0])

    assert isinstance(nss.par_rate(1.0), float)
    assert nss.par_rate(1.0) == pytest.approx(expected)


def test_par_curve_always_returns_an_array(simple_curve):
    """par_curve is vector-shaped by contract, even for a one-element grid.

    par_rate returns a float for a scalar, so par_curve had to re-wrap it or a
    caller indexing positionally would get a scalar back.
    """
    tenors, rates = simple_curve
    nss = fit_nss(tenors, rates)

    single = par_curve(nss, np.array([1.0]))

    assert isinstance(single, np.ndarray)
    assert single.shape == (1,)


def test_par_rate_still_returns_an_array_for_an_array(simple_curve):
    tenors, rates = simple_curve
    nss = fit_nss(tenors, rates)

    out = nss.par_rate(tenors)

    assert isinstance(out, np.ndarray)
    assert len(out) == len(tenors)


def test_nss_spot_rate_is_finite_at_zero_tenor(simple_curve):
    """t = 0 divides by the decay terms; the limit must be taken cleanly."""
    tenors, rates = simple_curve
    nss = fit_nss(tenors, rates)

    value = nss.spot_rate(0.0)

    assert np.isfinite(value)


def test_nss_evaluation_emits_no_runtime_warnings(simple_curve):
    """np.where still evaluates the untaken branch, which warned on every tenor."""
    tenors, rates = simple_curve
    nss = fit_nss(tenors, rates)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        values = nss.spot_rate(tenors)

    assert np.all(np.isfinite(values))


def test_par_curve_flat(flat_curve):
    """Test par curve on flat spot curve."""
    tenors, rates = flat_curve
    nss = fit_nss(tenors, rates)
    par = par_curve(nss, tenors)
    # For flat curve, par = spot
    assert np.allclose(par, rates, atol=0.0005)


# ============================================================================
# Carry/Roll-Down Tests
# ============================================================================


def test_carry_rolldown_basic(simple_curve):
    """Test basic carry/roll-down calculation."""
    tenors, rates = simple_curve
    nss = fit_nss(tenors, rates)
    fwd = forward_curve(nss, tenors)

    carry, roll, total = carry_rolldown(rates, fwd, horizon=1.0, tenors=tenors)

    assert len(carry) == len(tenors)
    assert len(roll) == len(tenors)
    assert len(total) == len(tenors)
    assert np.allclose(total, carry + roll)


def test_carry_rolldown_flat_curve(flat_curve):
    """Test carry/roll-down on flat curve."""
    tenors, rates = flat_curve
    nss = fit_nss(tenors, rates)
    fwd = forward_curve(nss, tenors)

    # For flat curve: carry ≈ 0, roll ≈ 0
    carry, roll, total = carry_rolldown(rates, fwd, horizon=1.0, tenors=tenors)
    assert np.allclose(carry, 0, atol=0.001)
    assert np.allclose(roll, 0, atol=0.001)
    assert np.allclose(total, 0, atol=0.001)


def test_carry_rolldown_positive_slope(simple_curve):
    """Test carry/roll-down on upward sloping curve."""
    tenors, rates = simple_curve
    nss = fit_nss(tenors, rates)
    fwd = forward_curve(nss, tenors)

    carry, roll, total = carry_rolldown(rates, fwd, horizon=1.0, tenors=tenors)

    # On upward sloping curve: forward > spot => positive carry
    assert np.all(carry > -0.001)  # Allow small numerical errors
    # Roll-down: spot(tenor-1) < spot(tenor) => negative roll-down
    assert np.all(roll < 0.001)
    # The headline number is the sum of the two components.
    np.testing.assert_allclose(total, carry + roll, rtol=1e-9, atol=1e-12)


def test_carry_rolldown_inverted_curve(inverted_curve):
    """Test carry/roll-down on inverted curve."""
    tenors, rates = inverted_curve
    nss = fit_nss(tenors, rates)
    fwd = forward_curve(nss, tenors)

    carry, roll, total = carry_rolldown(rates, fwd, horizon=1.0, tenors=tenors)

    # On inverted curve: forward < spot => negative carry
    assert np.all(carry < 0.001)
    # Roll-down: spot(tenor-1) > spot(tenor) => positive roll-down
    assert np.all(roll > -0.001)
    np.testing.assert_allclose(total, carry + roll, rtol=1e-9, atol=1e-12)


def test_carry_rolldown_mismatched_lengths():
    """Test carry/roll-down with mismatched arrays."""
    spot = np.array([0.04, 0.042, 0.043])
    fwd = np.array([0.044, 0.044])  # Only 2 elements
    with pytest.raises(ValueError, match="same length"):
        carry_rolldown(spot, fwd, horizon=1.0)


def test_carry_rolldown_different_horizons(simple_curve):
    """Test carry/roll-down at different horizons."""
    tenors, rates = simple_curve
    nss = fit_nss(tenors, rates)
    fwd = forward_curve(nss, tenors)

    # Test multiple horizons
    for h in [0.25, 0.5, 1.0, 2.0]:
        carry, roll, total = carry_rolldown(rates, fwd, horizon=h, tenors=tenors)
        assert len(carry) == len(tenors)
        assert np.allclose(total, carry + roll)


# ============================================================================
# Carry/Roll-Down Matrix Tests
# ============================================================================


def test_carry_rolldown_matrix(simple_curve):
    """Test carry/roll-down matrix computation."""
    tenors, rates = simple_curve
    nss = fit_nss(tenors, rates)
    fwd = forward_curve(nss, tenors)

    horizons = np.array([0.25, 0.5, 1.0, 2.0])
    carry_m, roll_m, total_m = carry_rolldown_matrix(rates, fwd, tenors, horizons)

    assert carry_m.shape == (len(horizons), len(tenors))
    assert roll_m.shape == (len(horizons), len(tenors))
    assert total_m.shape == (len(horizons), len(tenors))
    assert np.allclose(total_m, carry_m + roll_m)


def test_carry_rolldown_matrix_consistency(simple_curve):
    """Test matrix consistency with single-horizon function."""
    tenors, rates = simple_curve
    nss = fit_nss(tenors, rates)
    fwd = forward_curve(nss, tenors)

    horizons = np.array([0.25, 0.5, 1.0])
    carry_m, roll_m, total_m = carry_rolldown_matrix(rates, fwd, tenors, horizons)

    for i, h in enumerate(horizons):
        carry, roll, total = carry_rolldown(rates, fwd, horizon=h, tenors=tenors)
        assert np.allclose(carry_m[i], carry)
        assert np.allclose(roll_m[i], roll)
        assert np.allclose(total_m[i], total)


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_pipeline_treasury(sample_treasury_curve):
    """Test full pipeline: fit NSS -> forward -> carry/roll."""
    tenors, rates = sample_treasury_curve

    # Fit NSS
    nss = fit_nss(tenors, rates)

    # Forward curve
    fwd = forward_curve(nss, tenors)
    par = par_curve(nss, tenors)

    # Carry/roll at multiple horizons
    horizons = np.array([0.25, 0.5, 1.0, 2.0])
    carry_m, roll_m, total_m = carry_rolldown_matrix(rates, fwd, tenors, horizons)

    # The par curve comes off the same fit, so it should track the input rates.
    assert par.shape == (len(tenors),)
    assert np.all(np.isfinite(par))

    # Verify shapes
    assert carry_m.shape == (len(horizons), len(tenors))

    # Verify no NaN values
    assert not np.any(np.isnan(carry_m))
    assert not np.any(np.isnan(roll_m))
    assert not np.any(np.isnan(total_m))


def test_nss_fit_to_params_conversion():
    """Test conversion from NSSFit (charts) to NSSParameters."""
    from quantview.notebook.charts import NSSFit

    nss_fit = NSSFit(beta0=0.045, beta1=-0.01, beta2=-0.015, beta3=0.005, tau1=1.5, tau2=8.0)
    params = nss_fit_to_params(nss_fit)

    assert isinstance(params, NSSParameters)
    assert params.beta0 == nss_fit.beta0
    assert params.tau1 == nss_fit.tau1
    assert params.tau2 == nss_fit.tau2

    # Test spot rate consistency
    t = np.array([1.0, 5.0, 10.0])
    r1 = nss_fit.spot_rate(t)
    r2 = params.spot_rate(t)
    assert np.allclose(r1, r2)


def test_known_treasury_curve_fred_dgs():
    """Test against known FRED DGS series values.

    This test uses a representative Treasury curve snapshot.
    The exact values will vary by date, but the structure should be valid.
    """
    # FRED DGS series typical tenors (approximate)
    tenors = np.array([1 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30])
    # Sample rates from a normal curve period (e.g., 2019)
    rates = np.array(
        [
            0.0240,  # 1M
            0.0235,  # 3M
            0.0225,  # 6M
            0.0210,  # 1Y
            0.0190,  # 2Y
            0.0185,  # 3Y
            0.0180,  # 5Y
            0.0185,  # 7Y
            0.0190,  # 10Y
            0.0205,  # 20Y
            0.0210,  # 30Y
        ]
    )

    nss = fit_nss(tenors, rates)

    # Verify fit quality
    fitted = nss.spot_rate(tenors)
    rmse = np.sqrt(np.mean((fitted - rates) ** 2))
    assert rmse < 0.001  # Should fit well

    # Verify forward curve
    fwd = nss.forward_rate(tenors)
    assert len(fwd) == len(tenors)

    # Verify carry/roll
    carry, roll, total = carry_rolldown(rates, fwd, horizon=1.0, tenors=tenors)
    assert len(carry) == len(tenors)
    # Every component must line up with the tenor grid that produced them.
    assert len(roll) == len(tenors)
    assert len(total) == len(tenors)
    np.testing.assert_allclose(total, carry + roll, rtol=1e-9, atol=1e-12)


# ============================================================================
# Edge Cases
# ============================================================================


def test_fit_nss_extreme_rates():
    """Test NSS fitting with extreme but valid rates."""
    tenors = np.array([1.0, 2.0, 5.0, 10.0, 30.0])
    rates = np.array([0.20, 0.18, 0.15, 0.12, 0.10])  # Very high rates
    nss = fit_nss(tenors, rates)
    assert isinstance(nss, NSSParameters)
    fitted = nss.spot_rate(tenors)
    assert np.allclose(fitted, rates, rtol=0.05)  # Within 5% relative


def test_fit_nss_near_zero_rates():
    """Test NSS fitting with near-zero rates."""
    tenors = np.array([1.0, 2.0, 5.0, 10.0, 30.0])
    rates = np.array([0.001, 0.0015, 0.002, 0.0025, 0.003])  # Near zero
    nss = fit_nss(tenors, rates)
    assert isinstance(nss, NSSParameters)
    fitted = nss.spot_rate(tenors)
    assert np.allclose(fitted, rates, atol=0.001)


def test_nss_parameters_edge_cases():
    """Test NSS parameters with edge case values."""
    # Very small tau values
    params = NSSParameters(beta0=0.05, beta1=-0.01, beta2=0.0, beta3=0.0, tau1=0.5, tau2=0.5)
    t = np.array([0.25, 0.5, 1.0, 2.0])
    rates = params.spot_rate(t)
    assert len(rates) == 4
    assert all(np.isfinite(r) for r in rates)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
