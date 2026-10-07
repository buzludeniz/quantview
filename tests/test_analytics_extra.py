"""
Tests for QuantView Analytics: portfolio, risk and surface modules.

Companion to tests/test_analytics.py, which only covers analytics/curves.py.
This file covers the three remaining analytics modules:

- analytics/portfolio.py  Greeks aggregation, P&L waterfall, factor attribution
- analytics/risk.py      Greeks heatmaps, risk limits, stress testing
- analytics/surface.py   Option-chain -> volatility-surface utilities

Every test runs on deterministic synthetic data. No network access, no API
keys, no live market data.

Run with: pytest tests/test_analytics_extra.py -v

Known defects characterised (not endorsed) by tests in this file
----------------------------------------------------------------
Three source defects make some branches unreachable in a passing test run.
Each is pinned so the behaviour is documented rather than discovered later:

1. ``factor_attribution(method="style")`` raises ``TypeError``: SciPy removed
   the ``rcond`` keyword from ``scipy.linalg.lstsq`` in 1.14, and the module
   still passes it.
2. ``factor_attribution(method="pca")`` needs scikit-learn, which is not a
   declared dependency. Even with it installed the branch is broken: it never
   binds the ``idiosyncratic`` and ``r_squared`` locals that the final
   ``FactorAttributionResult`` construction reads.
3. ``chain_to_volsurface`` writes recovered implied vols into
   ``df["implied_vol"].values``. Under pandas copy-on-write that array is
   read-only, so a partially-NaN ``implied_vol`` column raises ``ValueError``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from quantview.analytics import portfolio as pf
from quantview.analytics import risk as rk
from quantview.analytics import surface as sf

# ============================================================================
# Black-Scholes Availability Check
# ============================================================================

try:
    from black_scholes import OptionParams, OptionType, black_scholes_price
    from black_scholes.surface import VolSurface

    HAS_BLACK_SCHOLES = True
except ImportError:  # pragma: no cover - exercised only on a bare install
    HAS_BLACK_SCHOLES = False

needs_bs = pytest.mark.skipif(not HAS_BLACK_SCHOLES, reason="black_scholes not available")


def _plotting_available() -> bool:
    """True when the optional matplotlib/seaborn stack is installed."""
    return all(importlib.util.find_spec(name) is not None for name in ("matplotlib", "seaborn"))


# ============================================================================
# Independent Black-Scholes Reference
# ============================================================================
#
# analytics/portfolio.py delegates every Greek to black_scholes. Comparing its
# output against the same library would only prove it calls the library, so the
# expectations below are rebuilt here from the published formulas. Any drift in
# the units the portfolio module assumes (vega per 1%, theta per day, rho per
# 1%) shows up as a failure here.

DAYS_PER_YEAR = 365.0
PERCENT = 0.01


def reference_greeks(
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    vol: float,
    dividend: float = 0.0,
    option_type: str = "call",
) -> dict[str, float]:
    """Black-Scholes price and Greeks from first principles (BSM with dividend)."""
    sqrt_t = math.sqrt(expiry)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    disc_q = math.exp(-dividend * expiry)
    disc_r = math.exp(-rate * expiry)
    pdf_d1 = float(norm.pdf(d1))

    gamma = disc_q * pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * disc_q * pdf_d1 * sqrt_t * PERCENT

    if option_type == "call":
        price = spot * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2)
        delta = disc_q * norm.cdf(d1)
        theta = (
            -spot * disc_q * pdf_d1 * vol / (2 * sqrt_t)
            - rate * strike * disc_r * norm.cdf(d2)
            + dividend * spot * disc_q * norm.cdf(d1)
        ) / DAYS_PER_YEAR
        rho = strike * expiry * disc_r * norm.cdf(d2) * PERCENT
    else:
        price = strike * disc_r * norm.cdf(-d2) - spot * disc_q * norm.cdf(-d1)
        delta = disc_q * (norm.cdf(d1) - 1.0)
        theta = (
            -spot * disc_q * pdf_d1 * vol / (2 * sqrt_t)
            + rate * strike * disc_r * norm.cdf(-d2)
            - dividend * spot * disc_q * norm.cdf(-d1)
        ) / DAYS_PER_YEAR
        rho = -strike * expiry * disc_r * norm.cdf(-d2) * PERCENT

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _preserve_global_numpy_rng():
    """generate_synthetic_portfolio reseeds the global numpy RNG; keep it local."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.fixture
def call_position():
    """One long 10-lot SPY call, 3 months out, 20% implied vol."""
    return pf.Position("SPY", 450.0, 0.25, "call", 10.0, 455.0, 0.20)


@pytest.fixture
def two_position_book():
    """A call and a put on different underlyings, deliberately mixed signs."""
    return [
        pf.Position("SPY", 450.0, 0.25, "call", 10.0, 455.0, 0.20),
        pf.Position("QQQ", 380.0, 0.75, "put", -4.0, 375.0, 0.30, 0.04, 0.015),
    ]


@pytest.fixture
def synthetic_book():
    """60-position seeded book; every aggregation invariant below is exact."""
    return pf.generate_synthetic_portfolio(60, seed=11)


@pytest.fixture
def smile_strikes():
    return np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0])


@pytest.fixture
def smile_vols():
    return np.array([0.280, 0.245, 0.230, 0.220, 0.228, 0.240, 0.270])


@pytest.fixture
def option_chain(smile_strikes, smile_vols):
    """Synthetic call chain priced off a known smile, quoted 10bp wide.

    Deliberately carries no ``implied_vol`` column so that
    ``chain_to_volsurface`` has to invert prices, and the round trip back to
    the input volatilities is a real assertion.
    """
    spot, expiry, rate = 100.0, 0.5, 0.05
    prices = np.array(
        [
            black_scholes_price(OptionParams(spot, k, expiry, rate, v))
            for k, v in zip(smile_strikes, smile_vols, strict=True)
        ]
    )
    return sf.build_chain_dataframe(
        strikes=smile_strikes,
        bids=prices * 0.999,
        asks=prices * 1.001,
        volumes=np.full(len(smile_strikes), 100),
        open_interests=np.full(len(smile_strikes), 500),
    )


@pytest.fixture
def single_surface(option_chain):
    """VolSurface recovered from the 6-month chain."""
    return sf.chain_to_volsurface(100.0, option_chain, expiry=0.5, risk_free_rate=0.05)


@pytest.fixture
def surfaces_by_expiry(smile_strikes, smile_vols):
    """Four expiries of the same smile shape with a mild term structure."""
    spot, rate = 100.0, 0.05
    out: dict[float, VolSurface] = {}
    for expiry in (0.25, 0.5, 1.0, 2.0):
        vols = smile_vols * (1.0 + 0.03 * math.sqrt(expiry))
        out[expiry] = VolSurface(
            spot=spot,
            strikes=tuple(float(k) for k in smile_strikes),
            implied_vols=tuple(float(v) for v in vols),
            time_to_maturity=expiry,
            risk_free_rate=rate,
            dividend_yield=0.0,
        )
    return out


@pytest.fixture
def flat_surfaces(smile_strikes):
    """Two expiries of a perfectly flat 22% smile."""
    return {
        expiry: VolSurface(
            spot=100.0,
            strikes=tuple(float(k) for k in smile_strikes),
            implied_vols=tuple([0.22] * len(smile_strikes)),
            time_to_maturity=expiry,
            risk_free_rate=0.05,
            dividend_yield=0.0,
        )
        for expiry in (0.25, 1.0)
    }


# ============================================================================
# portfolio.py :: dataclasses
# ============================================================================


@needs_bs
def test_position_to_option_params_call(call_position):
    """Position.to_option_params maps every field onto OptionParams."""
    params = call_position.to_option_params()
    assert params.spot == pytest.approx(455.0)
    assert params.strike == pytest.approx(450.0)
    assert params.time_to_maturity == pytest.approx(0.25)
    assert params.risk_free_rate == pytest.approx(0.05)
    assert params.volatility == pytest.approx(0.20)
    assert params.option_type is OptionType.CALL
    assert params.dividend_yield == pytest.approx(0.0)


@needs_bs
def test_position_to_option_params_put_and_dividend():
    """A put position produces a put OptionParams carrying the dividend yield."""
    pos = pf.Position("QQQ", 380.0, 0.75, "put", -4.0, 375.0, 0.30, 0.04, 0.015)
    params = pos.to_option_params()
    assert params.option_type is OptionType.PUT
    assert params.risk_free_rate == pytest.approx(0.04)
    assert params.dividend_yield == pytest.approx(0.015)
    assert params.forward_price == pytest.approx(375.0 * math.exp(0.025 * 0.75))


def test_greeks_as_dict_exposes_all_five():
    """Greeks.as_dict returns the five Greek names and nothing else."""
    greeks = pf.Greeks(delta=0.5, gamma=0.02, vega=0.3, theta=-0.05, rho=0.4)
    assert greeks.as_dict() == {
        "delta": 0.5,
        "gamma": 0.02,
        "vega": 0.3,
        "theta": -0.05,
        "rho": 0.4,
    }


def test_greeks_bucket_as_dict_merges_bucket_fields():
    """GreeksBucket.as_dict flattens the Greeks plus the bucket metadata."""
    bucket = pf.GreeksBucket(
        expiry=0.5,
        moneyness_bin="ATM",
        greeks=pf.Greeks(1.0, 0.5, 0.25, -0.125, 0.0625),
        notional=1000.0,
        position_count=3,
    )
    assert bucket.as_dict() == {
        "delta": 1.0,
        "gamma": 0.5,
        "vega": 0.25,
        "theta": -0.125,
        "rho": 0.0625,
        "expiry": 0.5,
        "moneyness_bin": "ATM",
        "notional": 1000.0,
        "position_count": 3,
    }


@pytest.mark.parametrize(
    ("moneyness", "expected"),
    [
        (0.50, "OTM"),
        (0.9499, "OTM"),
        (0.95, "ATM"),
        (1.00, "ATM"),
        (1.05, "ATM"),
        (1.0501, "ITM"),
        (3.00, "ITM"),
    ],
)
def test_moneyness_bin_boundaries(moneyness, expected):
    """The 0.95/1.05 boundaries are inclusive on the ATM side."""
    assert pf._moneyness_bin(moneyness) == expected


# ============================================================================
# portfolio.py :: aggregate_greeks
# ============================================================================


@needs_bs
def test_aggregate_greeks_matches_independent_black_scholes(call_position):
    """Single long call: every Greek equals the first-principles value x 10 lots."""
    ref = reference_greeks(455.0, 450.0, 0.25, 0.05, 0.20, 0.0, "call")
    greeks = pf.aggregate_greeks([call_position])
    assert greeks.delta == pytest.approx(ref["delta"] * 10.0, rel=1e-12)
    assert greeks.gamma == pytest.approx(ref["gamma"] * 10.0, rel=1e-12)
    assert greeks.vega == pytest.approx(ref["vega"] * 10.0, rel=1e-12)
    assert greeks.theta == pytest.approx(ref["theta"] * 10.0, rel=1e-12)
    assert greeks.rho == pytest.approx(ref["rho"] * 10.0, rel=1e-12)


@needs_bs
def test_aggregate_greeks_weighted_sum_of_two_positions(two_position_book):
    """Hand-summed two-leg book, including the short leg flipping the sign."""
    spy_ref = reference_greeks(455.0, 450.0, 0.25, 0.05, 0.20, 0.0, "call")
    qqq_ref = reference_greeks(375.0, 380.0, 0.75, 0.04, 0.30, 0.015, "put")

    greeks_names = ("delta", "gamma", "vega", "theta", "rho")
    expected = {n: spy_ref[n] * 10.0 + qqq_ref[n] * -4.0 for n in greeks_names}
    greeks = pf.aggregate_greeks(two_position_book)
    for name, value in expected.items():
        assert getattr(greeks, name) == pytest.approx(value, rel=1e-12), name


@needs_bs
def test_aggregate_greeks_long_short_offset_is_exactly_zero(call_position):
    """Identical long and short legs net to zero in every Greek."""
    short = pf.Position("SPY", 450.0, 0.25, "call", -10.0, 455.0, 0.20)
    greeks = pf.aggregate_greeks([call_position, short])
    assert greeks.as_dict() == pytest.approx(
        {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}, abs=1e-12
    )


@needs_bs
def test_aggregate_greeks_scales_linearly_with_quantity(call_position):
    """Tripling the position triples every Greek."""
    single = pf.aggregate_greeks([call_position])
    triple = pf.aggregate_greeks([call_position, call_position, call_position])
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert getattr(triple, name) == pytest.approx(3.0 * getattr(single, name), rel=1e-12)


@needs_bs
def test_aggregate_greeks_empty_portfolio_is_zero():
    """No positions means no exposure, not an error."""
    assert pf.aggregate_greeks([]).as_dict() == {
        "delta": 0.0,
        "gamma": 0.0,
        "vega": 0.0,
        "theta": 0.0,
        "rho": 0.0,
    }


@needs_bs
def test_aggregate_greeks_call_put_delta_parity_holds():
    """Delta_call - Delta_put = exp(-qT) survives the aggregation."""
    call = pf.Position("X", 100.0, 0.5, "call", 1.0, 100.0, 0.20, 0.05, 0.02)
    put = pf.Position("X", 100.0, 0.5, "put", 1.0, 100.0, 0.20, 0.05, 0.02)
    long_call = pf.aggregate_greeks([call])
    long_put = pf.aggregate_greeks([put])
    combined = pf.aggregate_greeks([call, put])

    assert long_call.delta - long_put.delta == pytest.approx(math.exp(-0.02 * 0.5), abs=1e-12)
    # Gamma and vega are option-type independent, so they cancel in the sum.
    assert combined.gamma == pytest.approx(2.0 * long_call.gamma, rel=1e-12)
    assert combined.vega == pytest.approx(2.0 * long_call.vega, rel=1e-12)


@needs_bs
def test_aggregate_greeks_sign_conventions():
    """A single long option is convex, long delta and decaying."""
    single_lot = pf.Position("SPY", 450.0, 0.25, "call", 1.0, 455.0, 0.20)
    greeks = pf.aggregate_greeks([single_lot])
    assert greeks.gamma > 0
    assert greeks.theta < 0
    assert greeks.vega > 0
    assert greeks.rho > 0
    assert 0 < greeks.delta < 1


# ============================================================================
# portfolio.py :: aggregate_greeks_by_bucket
# ============================================================================

# Default bin edges, and the labels the module derives from them:
#   expiry   [0, 1/12, 0.25, 0.5, 1, 2, inf]
#   moneyness [0, 0.9, 0.95, 1.05, 1.1, 2] labelled Deep OTM/OTM/ATM/ITM/Deep ITM


@needs_bs
@pytest.mark.parametrize(
    ("strike", "expiry", "expiry_bin", "moneyness_bin"),
    [
        # K=100 on a 100 spot 3M call: F=101.25, so K/F=0.9877 lands in [0.95, 1.05)
        (100.0, 0.25, "0.250-0.500", "ATM"),
        # K=80: K/F=0.790, below the 0.90 edge -> Deep OTM
        (80.0, 0.25, "0.250-0.500", "Deep OTM"),
        # K=107: K/F=1.0567, inside [1.05, 1.10) -> ITM
        (107.0, 0.25, "0.250-0.500", "ITM"),
        # K=115: K/F=1.1358, inside [1.10, 2] -> Deep ITM
        (115.0, 0.25, "0.250-0.500", "Deep ITM"),
        # 1-week expiry is inside the first (0, 1/12) edge pair
        (100.0, 0.05, "0.000-0.083", "ATM"),
        # an expiry sitting exactly on the 0.25 edge falls in the upper bin
        (100.0, 0.25, "0.250-0.500", "ATM"),
        # an expiry sitting exactly on the 0.5 edge also falls in the upper bin
        (100.0, 0.5, "0.500-1.000", "ATM"),
    ],
)
def test_aggregate_greeks_by_bucket_placement(strike, expiry, expiry_bin, moneyness_bin):
    """Each position lands in the bin its hand-computed K/F ratio selects."""
    pos = pf.Position("SPY", strike, expiry, "call", 1.0, 100.0, 0.20)
    df = pf.aggregate_greeks_by_bucket([pos])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["expiry_bin"] == expiry_bin
    assert row["moneyness_bin"] == moneyness_bin
    assert row["count"] == 1


@needs_bs
def test_aggregate_greeks_by_bucket_sums_two_positions_in_one_bucket():
    """Two ATM positions collapse to one row with count 2.

    Both legs sit at K=100/102 on a 100 spot, so K/F = 0.988 and 1.007, each
    inside the ATM edges [0.95, 1.05).
    """
    positions = [
        pf.Position("SPY", 100.0, 0.25, "call", 2.0, 100.0, 0.20),
        pf.Position("QQQ", 102.0, 0.25, "put", 3.0, 100.0, 0.25),
    ]
    df = pf.aggregate_greeks_by_bucket(positions)
    assert len(df) == 1
    row = df.iloc[0]

    ref_call = reference_greeks(100.0, 100.0, 0.25, 0.05, 0.20, 0.0, "call")
    ref_put = reference_greeks(100.0, 102.0, 0.25, 0.05, 0.25, 0.0, "put")
    assert row["expiry_bin"] == "0.250-0.500"
    assert row["moneyness_bin"] == "ATM"
    assert row["count"] == 2
    assert row["delta"] == pytest.approx(ref_call["delta"] * 2 + ref_put["delta"] * 3, rel=1e-12)
    assert row["gamma"] == pytest.approx(ref_call["gamma"] * 2 + ref_put["gamma"] * 3, rel=1e-12)
    # notional = |qty| * spot * 100, summed over the bucket
    assert row["notional"] == pytest.approx((2 + 3) * 100.0 * 100)


@needs_bs
def test_aggregate_greeks_by_bucket_separates_different_buckets():
    """Different expiries and moneyness produce separate rows."""
    positions = [
        pf.Position("SPY", 100.0, 0.25, "call", 1.0, 100.0, 0.20),
        pf.Position("SPY", 100.0, 1.0, "call", 1.0, 100.0, 0.20),
        pf.Position("SPY", 80.0, 0.25, "call", 1.0, 100.0, 0.20),
    ]
    df = pf.aggregate_greeks_by_bucket(positions)
    pairs = set(zip(df["expiry_bin"], df["moneyness_bin"], strict=True))
    assert pairs == {
        ("0.250-0.500", "ATM"),
        ("1.000-2.000", "ATM"),
        ("0.250-0.500", "Deep OTM"),
    }
    assert df["count"].sum() == 3


@needs_bs
def test_aggregate_greeks_by_bucket_notional_uses_absolute_quantity():
    """A short position contributes positive notional but negative Greeks."""
    long_leg = pf.Position("SPY", 100.0, 0.25, "call", 5.0, 100.0, 0.20)
    short_leg = pf.Position("SPY", 100.0, 0.25, "call", -5.0, 100.0, 0.20)
    row = pf.aggregate_greeks_by_bucket([long_leg, short_leg]).iloc[0]
    assert row["notional"] == pytest.approx(10 * 100.0 * 100)
    assert row["count"] == 2
    assert row["delta"] == pytest.approx(0.0, abs=1e-12)
    assert row["gamma"] == pytest.approx(0.0, abs=1e-12)


@needs_bs
def test_aggregate_greeks_by_bucket_custom_edges_change_labels():
    """Caller-supplied expiry edges are reflected in the labels.

    The moneyness labels are a fixed five-element list, independent of the
    edges passed in, so custom edges only renumber the buckets: with edges
    [0, 1.0, 2.0] an ATM-ish K/F of 0.988 falls in the first bucket, which
    carries the first label in that fixed list.
    """
    pos = pf.Position("SPY", 100.0, 0.5, "call", 1.0, 100.0, 0.20)
    df = pf.aggregate_greeks_by_bucket(
        [pos],
        expiry_bins=[0, 0.5, 1.0],
        moneyness_bins=[0, 1.0, 2.0],
    )
    assert len(df) == 1
    # expiry 0.5 sits on the lower edge of [0.5, 1.0) -> "0.500-1.000"
    assert df.iloc[0]["expiry_bin"] == "0.500-1.000"
    assert df.iloc[0]["moneyness_bin"] == "Deep OTM"


@needs_bs
def test_aggregate_greeks_by_bucket_empty_positions():
    """No positions yields an empty frame rather than an exception."""
    df = pf.aggregate_greeks_by_bucket([])
    assert df.empty
    assert len(df.columns) == 0


@needs_bs
def test_aggregate_greeks_by_bucket_totals_match_aggregate_greeks(synthetic_book):
    """Bucketing only regroups: the totals must reproduce the flat aggregate."""
    df = pf.aggregate_greeks_by_bucket(synthetic_book)
    flat = pf.aggregate_greeks(synthetic_book)
    assert df["count"].sum() == len(synthetic_book)
    assert df["delta"].sum() == pytest.approx(flat.delta, abs=1e-9)
    assert df["gamma"].sum() == pytest.approx(flat.gamma, abs=1e-9)
    assert df["vega"].sum() == pytest.approx(flat.vega, abs=1e-9)
    assert df["theta"].sum() == pytest.approx(flat.theta, abs=1e-9)
    assert df["rho"].sum() == pytest.approx(flat.rho, abs=1e-9)
    expected_notional = sum(abs(p.quantity) * p.spot * 100 for p in synthetic_book)
    assert df["notional"].sum() == pytest.approx(expected_notional)


# ============================================================================
# portfolio.py :: pnl_waterfall
# ============================================================================

PNL_COLUMNS = [
    "symbol",
    "strike",
    "expiry",
    "type",
    "quantity",
    "delta_pnl",
    "gamma_pnl",
    "vega_pnl",
    "theta_pnl",
    "rho_pnl",
    "first_order_pnl",
    "exact_pnl",
    "residual",
]


@needs_bs
def test_pnl_waterfall_columns_and_total_row(two_position_book):
    """One row per position plus a trailing TOTAL row."""
    df = pf.pnl_waterfall(two_position_book, {"SPY": 1.0, "QQQ": -0.5})
    assert list(df.columns) == PNL_COLUMNS
    assert len(df) == 3
    assert df["symbol"].iloc[-1] == "TOTAL"
    assert df["symbol"].iloc[0] == "SPY"
    assert df["symbol"].iloc[1] == "QQQ"
    assert df["type"].iloc[-1] == ""
    assert df["quantity"].iloc[-1] == pytest.approx(6.0)


@needs_bs
def test_pnl_waterfall_components_hand_computed(call_position):
    """Every first-order component equals Greek x shock x quantity x 100."""
    d_spot, d_vol, d_rate, decay = 1.5, 0.02, 0.0025, 1 / 365
    df = pf.pnl_waterfall(
        [call_position],
        {"SPY": d_spot},
        {"SPY": d_vol},
        {"SPY": d_rate},
        time_decay=decay,
    )
    ref = reference_greeks(455.0, 450.0, 0.25, 0.05, 0.20, 0.0, "call")
    qty = 10.0
    row = df.iloc[0]

    assert row["delta_pnl"] == pytest.approx(ref["delta"] * d_spot * qty * 100, rel=1e-12)
    assert row["gamma_pnl"] == pytest.approx(0.5 * ref["gamma"] * d_spot**2 * qty * 100, rel=1e-12)
    assert row["vega_pnl"] == pytest.approx(ref["vega"] * d_vol * qty * 100, rel=1e-12)
    assert row["theta_pnl"] == pytest.approx(ref["theta"] * decay * qty * 100, rel=1e-12)
    assert row["rho_pnl"] == pytest.approx(ref["rho"] * d_rate * qty * 100, rel=1e-12)


@needs_bs
def test_pnl_waterfall_exact_pnl_matches_full_repricing(call_position):
    """exact_pnl is the repriced difference, including the gamma term."""
    d_spot, d_vol, d_rate, decay = -2.0, 0.01, -0.001, 0.5 / 365
    df = pf.pnl_waterfall(
        [call_position],
        {"SPY": d_spot},
        {"SPY": d_vol},
        {"SPY": d_rate},
        time_decay=decay,
    )
    old = reference_greeks(455.0, 450.0, 0.25, 0.05, 0.20, 0.0, "call")["price"]
    new = reference_greeks(
        455.0 + d_spot,
        450.0,
        0.25 - decay,
        0.05 + d_rate,
        0.20 + d_vol,
        0.0,
        "call",
    )["price"]
    assert df.iloc[0]["exact_pnl"] == pytest.approx((new - old) * 10.0 * 100, rel=1e-10)


@needs_bs
def test_pnl_waterfall_first_order_and_residual_identities(two_position_book):
    """first_order is the component sum; residual is exact minus first order."""
    df = pf.pnl_waterfall(two_position_book, {"SPY": 1.0, "QQQ": -1.0}, {"SPY": 0.01, "QQQ": 0.02})
    for _, row in df.iterrows():
        components = (
            row["delta_pnl"]
            + row["gamma_pnl"]
            + row["vega_pnl"]
            + row["theta_pnl"]
            + row["rho_pnl"]
        )
        assert row["first_order_pnl"] == pytest.approx(components, abs=1e-9)
        assert row["residual"] == pytest.approx(row["exact_pnl"] - row["first_order_pnl"], abs=1e-9)


@needs_bs
def test_pnl_waterfall_total_row_equals_column_sums(two_position_book):
    """The TOTAL row is a straight column sum of the legs above it."""
    df = pf.pnl_waterfall(
        two_position_book, {"SPY": 0.75, "QQQ": -0.25}, {"SPY": 0.01, "QQQ": 0.01}
    )
    legs, total = df.iloc[:-1], df.iloc[-1]
    for column in (
        "delta_pnl",
        "gamma_pnl",
        "vega_pnl",
        "theta_pnl",
        "rho_pnl",
        "first_order_pnl",
        "exact_pnl",
        "residual",
    ):
        assert total[column] == pytest.approx(legs[column].sum(), abs=1e-9)


@needs_bs
def test_pnl_waterfall_no_shocks_gives_exactly_zero(call_position):
    """Zero shock and zero decay must reprice to the same option, so P&L is 0."""
    df = pf.pnl_waterfall([call_position], {}, None, None, time_decay=0.0)
    row = df.iloc[0]
    assert row["exact_pnl"] == pytest.approx(0.0, abs=1e-12)
    assert row["first_order_pnl"] == pytest.approx(0.0, abs=1e-12)
    assert row["residual"] == pytest.approx(0.0, abs=1e-12)
    assert df.iloc[-1]["exact_pnl"] == pytest.approx(0.0, abs=1e-12)


@needs_bs
def test_pnl_waterfall_ignores_shocks_for_other_symbols(call_position):
    """A shock keyed to a symbol the book does not hold is dropped."""
    df = pf.pnl_waterfall([call_position], {"QQQ": 5.0}, {"QQQ": 0.05}, {"QQQ": 0.01})
    assert df.iloc[0]["delta_pnl"] == pytest.approx(0.0)
    assert df.iloc[0]["vega_pnl"] == pytest.approx(0.0)


@needs_bs
def test_pnl_waterfall_volatility_change_is_floored_at_ten_bp():
    """A -200 vol shock reprices at vol 0.001, the hard floor in the module."""
    pos = pf.Position("SPY", 450.0, 0.25, "call", 1.0, 455.0, 0.15)
    df = pf.pnl_waterfall([pos], {"SPY": 0.0}, {"SPY": -0.2}, {"SPY": 0.0}, time_decay=0.0)
    old = reference_greeks(455.0, 450.0, 0.25, 0.05, 0.15, 0.0, "call")["price"]
    floored = reference_greeks(455.0, 450.0, 0.25, 0.05, 0.001, 0.0, "call")["price"]
    assert df.iloc[0]["exact_pnl"] == pytest.approx((floored - old) * 100, rel=1e-10)


@needs_bs
def test_pnl_waterfall_short_position_mirrors_long_position():
    """P&L is linear in quantity, so a short book is the exact negative."""
    long_leg = pf.Position("SPY", 450.0, 0.25, "call", 10.0, 455.0, 0.20)
    short_leg = pf.Position("SPY", 450.0, 0.25, "call", -10.0, 455.0, 0.20)
    shocks = {"SPY": 1.25}
    long_pnl = pf.pnl_waterfall([long_leg], shocks, {"SPY": 0.01}, time_decay=1 / 365)
    short_pnl = pf.pnl_waterfall([short_leg], shocks, {"SPY": 0.01}, time_decay=1 / 365)
    assert short_pnl.iloc[0]["exact_pnl"] == pytest.approx(
        -long_pnl.iloc[0]["exact_pnl"], rel=1e-12
    )


@needs_bs
def test_pnl_waterfall_signs_follow_the_shock_direction(call_position):
    """A long call gains when spot rises and when vol rises; theta bleeds daily."""
    up = pf.pnl_waterfall([call_position], {"SPY": 2.0}, time_decay=0.0).iloc[0]
    down = pf.pnl_waterfall([call_position], {"SPY": -2.0}, time_decay=0.0).iloc[0]
    vol_up = pf.pnl_waterfall([call_position], {}, {"SPY": 0.05}, time_decay=0.0).iloc[0]
    vol_down = pf.pnl_waterfall([call_position], {}, {"SPY": -0.05}, time_decay=0.0).iloc[0]

    assert up["delta_pnl"] > 0
    assert down["delta_pnl"] < 0
    assert up["exact_pnl"] > down["exact_pnl"]
    assert vol_up["vega_pnl"] > 0
    assert vol_down["vega_pnl"] < 0
    # Gamma is a long option's convexity bonus: it always adds for either sign.
    assert up["gamma_pnl"] > 0
    assert down["gamma_pnl"] > 0


@needs_bs
def test_pnl_waterfall_empty_positions_returns_empty_frame():
    """An empty book produces an empty frame and no TOTAL row."""
    df = pf.pnl_waterfall([], {})
    assert df.empty
    assert "symbol" not in df.columns


@needs_bs
def test_pnl_waterfall_gamma_pnl_is_second_order_in_the_move():
    """Halving the spot move quarters the gamma contribution."""
    pos = pf.Position("SPY", 450.0, 0.25, "call", 1.0, 455.0, 0.20)
    full = pf.pnl_waterfall([pos], {"SPY": 4.0}, time_decay=0.0).iloc[0]["gamma_pnl"]
    half = pf.pnl_waterfall([pos], {"SPY": 2.0}, time_decay=0.0).iloc[0]["gamma_pnl"]
    assert full == pytest.approx(4.0 * half, rel=1e-12)


# ============================================================================
# portfolio.py :: factor_attribution
# ============================================================================


def test_factor_attribution_summary_reports_loading_moments():
    """FactorAttributionResult.summary pairs each factor return with its loadings."""
    loadings = pd.DataFrame(
        np.array([[1.0, 2.0], [3.0, 6.0]]), index=["p", "q"], columns=["f1", "f2"]
    )
    result = pf.FactorAttributionResult(
        factor_loadings=loadings,
        factor_returns=pd.Series([0.01, 0.02], index=["f1", "f2"]),
        idiosyncratic=pd.Series([0.1, 0.2], index=["p", "q"]),
        r_squared=pd.Series([0.5, 0.7], index=["p", "q"]),
        total_variance_explained=0.6,
    )
    summary = result.summary()
    assert list(summary.columns) == ["factor", "return", "loading_mean", "loading_std"]
    assert list(summary["factor"]) == ["f1", "f2"]
    assert summary["return"].tolist() == pytest.approx([0.01, 0.02])
    assert summary["loading_mean"].tolist() == pytest.approx([2.0, 4.0])
    assert summary["loading_std"].tolist() == pytest.approx([math.sqrt(2.0), 2 * math.sqrt(2.0)])


def test_factor_attribution_rejects_returns_that_are_all_nan():
    """All-NaN columns are dropped, and an empty result is an error."""
    dates = pd.date_range("2023-01-02", periods=40, freq="B")
    returns = pd.DataFrame({"a": [np.nan] * 40, "b": [np.nan] * 40}, index=dates)
    factors = pd.DataFrame({"f1": np.linspace(0, 1, 40)}, index=dates)
    with pytest.raises(ValueError, match="No valid return data"):
        pf.factor_attribution(returns, factors)


def test_factor_attribution_rejects_empty_returns():
    """An empty frame trips the same guard."""
    with pytest.raises(ValueError, match="No valid return data"):
        pf.factor_attribution(pd.DataFrame(), pd.DataFrame())


def test_factor_attribution_rejects_short_overlap():
    """Fewer than 10 shared dates is rejected before any regression runs."""
    dates = pd.date_range("2023-01-02", periods=40, freq="B")
    returns = pd.DataFrame({"a": np.linspace(0, 1, 40)}, index=dates)
    factors = pd.DataFrame({"f1": np.linspace(0, 1, 40)}, index=dates)
    with pytest.raises(ValueError, match="Insufficient overlapping dates"):
        pf.factor_attribution(returns.iloc[:5], factors)


def test_factor_attribution_style_path_recovers_exact_loadings():
    """The style branch must recover the loadings that generated the returns.

    This path was dead: it called ``scipy.linalg.lstsq(..., rcond=None)`` and
    SciPy removed that keyword in 1.14 in favour of ``cond``, so every call
    raised ``TypeError``. The earlier version of this test pinned that failure;
    it now checks the analysis actually produces the right answer.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-02", periods=120, freq="B")
    factors = pd.DataFrame(
        {
            "f1": rng.normal(0.0005, 0.01, 120),
            "f2": rng.normal(0.0002, 0.008, 120),
        },
        index=dates,
    )
    # Returns are an exact linear combination, so the regression must recover it.
    # Two assets built separately: passing one Series with a two-name column list
    # makes pandas infer a (n, 1) frame and then reject the index/columns shape.
    returns = pd.DataFrame(
        {
            "a": 1.5 * factors["f1"] - 0.75 * factors["f2"],
            "b": 0.5 * factors["f1"] + 2.0 * factors["f2"],
        },
        index=dates,
    )

    result = pf.factor_attribution(returns, factors, method="style")

    assert set(result.factor_loadings.index) == {"a", "b"}
    assert set(result.factor_loadings.columns) == {"f1", "f2"}
    np.testing.assert_allclose(
        result.factor_loadings.loc["a"].to_numpy(), [1.5, -0.75], rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        result.factor_loadings.loc["b"].to_numpy(), [0.5, 2.0], rtol=1e-6, atol=1e-9
    )
    # An exact fit leaves nothing idiosyncratic and explains all the variance.
    np.testing.assert_allclose(result.idiosyncratic.to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(result.r_squared.to_numpy(), 1.0, atol=1e-8)
    assert float(result.total_variance_explained) == pytest.approx(1.0, abs=1e-8)


def test_factor_attribution_pca_path_is_reachable_and_consistent():
    """The PCA branch must run end to end and populate every result field.

    Three faults were stacked on this path: it needed scikit-learn, which is not
    a dependency; it never bound the ``idiosyncratic``/``r_squared`` locals its
    result object reads, so it would raise ``UnboundLocalError`` even with
    scikit-learn present; and it broadcast ``components_.T * std`` against the
    wrong axis. The branch is exercised with a stub following scikit-learn's real
    component convention, ``components_`` of shape ``(n_components,
    n_features)``, so a heavyweight optional dependency stays out of the suite.
    """
    _install_pca_stub()

    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-02", periods=200, freq="B")
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, (200, 4)), index=dates, columns=["a", "b", "c", "d"]
    )

    result = pf.factor_attribution(returns, None, method="pca", n_pca_factors=2)

    assert len(result.idiosyncratic) == 4
    assert len(result.r_squared) == 4
    assert set(result.r_squared.index) == {"a", "b", "c", "d"}
    assert result.factor_loadings.shape == (4, 2)
    assert np.all(result.r_squared.to_numpy() >= 0.0)
    assert np.all(result.r_squared.to_numpy() <= 1.0 + 1e-9)
    assert np.all(np.isfinite(result.idiosyncratic.to_numpy()))


def test_factor_attribution_pca_without_sklearn_names_the_dependency(monkeypatch):
    """Without scikit-learn the guard must name it and offer the alternative.

    scikit-learn is hidden explicitly: the companion test installs a stub into
    ``sys.modules``, and without hiding it this would depend on test order and
    silently stop raising.
    """
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.decomposition", None)
    with pytest.raises(ImportError, match="scikit-learn"):
        pf.factor_attribution(_random_returns(60, 3), None, method="pca")


def _random_returns(rows: int, cols: int) -> pd.DataFrame:
    """A returns frame with business-day dates and letter tickers."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-02", periods=rows, freq="B")
    names = [chr(ord("a") + i) for i in range(cols)]
    return pd.DataFrame(rng.normal(0.0, 0.01, (rows, cols)), index=dates, columns=names)


def _install_pca_stub() -> None:
    """Put a scikit-learn-compatible ``PCA`` on ``sys.modules`` if absent.

    Mirrors the one attribute the caller depends on: ``components_`` shaped
    ``(n_components, n_features)``, taken from the SVD's right singular vectors,
    which is what scikit-learn documents.
    """
    if "sklearn" in sys.modules and hasattr(sys.modules["sklearn"], "decomposition"):
        return

    sklearn = types.ModuleType("sklearn")
    decomposition = types.ModuleType("sklearn.decomposition")

    class _PCA:
        def __init__(self, n_components: int = 3) -> None:
            self.n_components = n_components
            self.n_components_ = n_components
            self.components_ = None
            self.explained_variance_ratio_ = None

        def fit_transform(self, x):
            a = np.asarray(x, dtype=float)
            k = min(self.n_components, a.shape[1])
            _, s, vh = np.linalg.svd(a, full_matrices=False)
            self.components_ = vh[:k, :]
            self.n_components_ = k
            self.explained_variance_ratio_ = (s[:k] ** 2) / max(float((s**2).sum()), 1e-300)
            return a @ self.components_.T

    decomposition.PCA = _PCA
    sklearn.decomposition = decomposition
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.decomposition"] = decomposition


def test_chain_to_volsurface_fills_missing_implied_vols_under_pandas3():
    """Filling NaN implied vols must work under pandas copy-on-write.

    ``df["implied_vol"].values`` can hand back a read-only view under pandas 3.0,
    so writing the inverted values back raised "assignment destination is
    read-only". The frame is read with ``copy=True`` so the fill has its own
    buffer.
    """
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]
    bids = [8.0, 9.0, 10.0, 9.0, 8.0, 7.0, 6.0]
    asks = [8.4, 9.4, 10.4, 9.4, 8.4, 7.4, 6.4]
    chain = sf.build_chain_dataframe(
        strikes=strikes,
        bids=bids,
        asks=asks,
        volumes=[100] * 7,
        open_interests=[500] * 7,
        implied_vols=[float("nan")] * 7,
    )
    assert chain["implied_vol"].isna().all(), "precondition: nothing pre-computed"

    surface = sf.chain_to_volsurface(spot=100.0, chain=chain, expiry=1.0, risk_free_rate=0.05)
    assert len(surface.strikes) >= 4
    assert all(v > 0 for v in surface.implied_vols)


# ============================================================================
# portfolio.py :: generate_synthetic_portfolio
# ============================================================================


def test_generate_synthetic_portfolio_is_reproducible():
    """Same seed, same book."""
    first = pf.generate_synthetic_portfolio(8, seed=42)
    second = pf.generate_synthetic_portfolio(8, seed=42)
    assert first == second


def test_generate_synthetic_portfolio_seed_changes_the_draw():
    """A different seed gives a materially different book."""
    first = pf.generate_synthetic_portfolio(20, seed=42)
    second = pf.generate_synthetic_portfolio(20, seed=7)
    assert [p.spot for p in first] != [p.spot for p in second]


def test_generate_synthetic_portfolio_honours_custom_symbols():
    """The symbol list is a closed set."""
    book = pf.generate_synthetic_portfolio(50, ["AAA", "BBB"], seed=1)
    assert {p.symbol for p in book} <= {"AAA", "BBB"}


def test_generate_synthetic_portfolio_default_symbol_pool():
    """The default pool is the ten large caps."""
    book = pf.generate_synthetic_portfolio(300, seed=5)
    assert {str(p.symbol) for p in book} <= {
        "SPY",
        "QQQ",
        "IWM",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
    }


def test_generate_synthetic_portfolio_parameter_ranges():
    """Every drawn field stays inside the documented generation ranges."""
    book = pf.generate_synthetic_portfolio(300, seed=5)
    assert len(book) == 300
    for pos in book:
        assert 50.0 <= pos.spot <= 500.0
        assert 0.8 <= pos.strike / pos.spot <= 1.2
        assert 0.15 <= pos.implied_vol <= 0.5
        assert 0.0 <= pos.dividend_yield <= 0.03
        assert pos.risk_free_rate == pytest.approx(0.05)
        assert pos.option_type in ("call", "put")
        assert pos.quantity in (-10, -5, -2, -1, 1, 2, 5, 10)
        assert round(pos.expiry, 6) in (
            round(x, 6) for x in (1 / 52, 1 / 12, 2 / 12, 3 / 12, 6 / 12, 1.0)
        )


def test_generate_synthetic_portfolio_zero_positions():
    """Zero positions is an empty book."""
    assert pf.generate_synthetic_portfolio(0, seed=1) == []


def test_generate_synthetic_portfolio_output_is_pricable():
    """The generated book is valid input to the rest of the module."""
    book = pf.generate_synthetic_portfolio(25, seed=9)
    greeks = pf.aggregate_greeks(book)
    assert all(np.isfinite(value) for value in greeks.as_dict().values())
    df = pf.aggregate_greeks_by_bucket(book)
    assert df["count"].sum() == 25


# ============================================================================
# risk.py :: GreeksHeatmap
# ============================================================================


@needs_bs
def test_greeks_heatmap_to_dataframe_labels_and_values():
    """to_dataframe pivots the array with expiry on rows and moneyness on columns."""
    book = pf.generate_synthetic_portfolio(40, seed=2)
    heatmap = rk.greeks_buckets(book)
    df = heatmap.to_dataframe("delta")
    assert df.shape == (len(heatmap.expiry_bins), len(heatmap.moneyness_bins))
    assert list(df.index) == heatmap.expiry_bins
    assert list(df.columns) == heatmap.moneyness_bins
    assert df.values == pytest.approx(heatmap.delta)


@needs_bs
def test_greeks_heatmap_to_dataframe_notional_and_count(synthetic_book):
    """Notional and count are exposed through the same accessor."""
    heatmap = rk.greeks_buckets(synthetic_book)
    assert heatmap.to_dataframe("notional").values == pytest.approx(heatmap.notional)
    assert heatmap.to_dataframe("count").values == pytest.approx(heatmap.count)
    assert heatmap.to_dataframe("count").values.sum() == len(synthetic_book)


@needs_bs
@pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "theta", "rho"])
def test_greeks_heatmap_get_max_abs_locates_largest(synthetic_book, greek):
    """get_max_abs returns the signed value at the argmax of its absolute."""
    heatmap = rk.greeks_buckets(synthetic_book)
    data = getattr(heatmap, greek)
    value, index = heatmap.get_max_abs(greek)
    assert abs(value) == pytest.approx(np.abs(data).max())
    assert data[index] == pytest.approx(value)
    assert index == np.unravel_index(np.abs(data).argmax(), data.shape)


@needs_bs
def test_greeks_heatmap_get_max_abs_on_empty_book():
    """An empty book has no exposure; the accessor still returns a position."""
    heatmap = rk.greeks_buckets([])
    assert heatmap.get_max_abs("delta") == (0.0, (0, 0))


# ============================================================================
# risk.py :: greeks_buckets
# ============================================================================


@needs_bs
def test_greeks_buckets_default_labels():
    """Default edges produce the documented 8x8 grid of labels."""
    heatmap = rk.greeks_buckets([pf.Position("SPY", 100.0, 0.25, "call", 1.0, 100.0, 0.2)])
    assert heatmap.expiry_bins == [
        "<0.02Y",
        "0.02-0.08Y",
        "0.08-0.17Y",
        "0.17-0.25Y",
        "0.25-0.50Y",
        "0.50-1.00Y",
        "1.00-2.00Y",
        ">2.00Y",
    ]
    assert heatmap.moneyness_bins == [
        "<0.80",
        "0.80-0.90",
        "0.90-0.95",
        "0.95-1.00",
        "1.00-1.05",
        "1.05-1.10",
        "1.10-1.20",
        ">1.20",
    ]
    assert heatmap.delta.shape == (8, 8)
    assert heatmap.count.shape == (8, 8)


@needs_bs
@pytest.mark.parametrize(
    ("strike", "expiry", "expiry_label", "moneyness_label", "why"),
    [
        # spot 400, T=1.0, r=5% -> F = 420.61
        (200.0, 1.0, "1.00-2.00Y", "<0.80", "K/F=0.476, deep below the 0.80 edge"),
        # K/F=0.963 sits in the 0.95-1.00 band
        (404.88, 1.0, "1.00-2.00Y", "0.95-1.00", "forward-ATM, just under 1.0"),
        # K/F=2.140 is past the top 1.20 edge
        (900.0, 1.0, "1.00-2.00Y", ">1.20", "clamped into the last bin"),
        # K/F=3.805 is far past the top edge
        (1600.0, 1.0, "1.00-2.00Y", ">1.20", "clamped into the last bin"),
        # T=5.0 is past the 2.0 expiry edge -> last row
        (404.88, 5.0, ">2.00Y", "<0.80", "F grows to 547.06, so K/F=0.740"),
        # T=0.01 is inside the first expiry band
        (400.0, 0.01, "<0.02Y", "0.95-1.00", "one-week expiry, F almost equals spot"),
    ],
)
def test_greeks_buckets_placement(strike, expiry, expiry_label, moneyness_label, why):
    """Each position lands in the cell its hand-computed K/F and T select."""
    pos = pf.Position("SPY", strike, expiry, "call", 1.0, 400.0, 0.25)
    heatmap = rk.greeks_buckets([pos])
    occupied = np.argwhere(heatmap.count > 0)
    assert len(occupied) == 1, why
    row, col = occupied[0]
    assert heatmap.expiry_bins[row] == expiry_label, why
    assert heatmap.moneyness_bins[col] == moneyness_label, why


@needs_bs
def test_greeks_buckets_cell_value_matches_single_position_greeks():
    """A lone position's cell holds exactly its own Greeks times quantity."""
    pos = pf.Position("SPY", 404.88, 1.0, "call", 3.0, 400.0, 0.25)
    heatmap = rk.greeks_buckets([pos])
    row, col = np.argwhere(heatmap.count > 0)[0]
    ref = reference_greeks(400.0, 404.88, 1.0, 0.05, 0.25, 0.0, "call")
    assert heatmap.delta[row, col] == pytest.approx(ref["delta"] * 3.0, rel=1e-12)
    assert heatmap.gamma[row, col] == pytest.approx(ref["gamma"] * 3.0, rel=1e-12)
    assert heatmap.vega[row, col] == pytest.approx(ref["vega"] * 3.0, rel=1e-12)
    assert heatmap.theta[row, col] == pytest.approx(ref["theta"] * 3.0, rel=1e-12)
    assert heatmap.rho[row, col] == pytest.approx(ref["rho"] * 3.0, rel=1e-12)
    assert heatmap.notional[row, col] == pytest.approx(3.0 * 400.0 * 100)
    assert heatmap.count[row, col] == 1
    assert heatmap.gamma[row, col] > 0


@needs_bs
def test_greeks_buckets_counts_and_notional_conserve(synthetic_book):
    """Every position is counted exactly once and its notional fully attributed."""
    heatmap = rk.greeks_buckets(synthetic_book)
    assert heatmap.count.sum() == len(synthetic_book)
    assert heatmap.notional.sum() == pytest.approx(
        sum(abs(p.quantity) * p.spot * 100 for p in synthetic_book)
    )


@needs_bs
def test_greeks_buckets_greek_totals_match_aggregate_greeks(synthetic_book):
    """The heatmap is a regrouping, so summing it must reproduce the flat book."""
    heatmap = rk.greeks_buckets(synthetic_book)
    flat = pf.aggregate_greeks(synthetic_book)
    for greek in ("delta", "gamma", "vega", "theta", "rho"):
        assert getattr(heatmap, greek).sum() == pytest.approx(getattr(flat, greek), abs=1e-9), greek


@needs_bs
def test_greeks_buckets_custom_bins():
    """Custom edges shorten the labels and the arrays together."""
    pos = pf.Position("SPY", 100.0, 0.5, "call", 1.0, 100.0, 0.20)
    heatmap = rk.greeks_buckets([pos], expiry_bins=[0, 0.5, 1.0], moneyness_bins=[0, 1.0, 2.0])
    assert heatmap.expiry_bins == ["<0.50Y", "0.50-1.00Y"]
    assert heatmap.moneyness_bins == ["<1.00", ">1.00"]
    assert heatmap.delta.shape == (2, 2)
    assert heatmap.count.sum() == 1


@needs_bs
def test_greeks_buckets_empty_portfolio_is_all_zero():
    """No positions gives an all-zero grid of the full default shape."""
    heatmap = rk.greeks_buckets([])
    assert heatmap.delta.shape == (8, 8)
    for greek in ("delta", "gamma", "vega", "theta", "rho", "notional", "count"):
        assert np.array_equal(getattr(heatmap, greek), np.zeros((8, 8)))


@needs_bs
def test_greeks_buckets_short_offsets_long_in_the_same_cell():
    """A hedged pair nets to zero Greeks but keeps its gross notional."""
    long_leg = pf.Position("SPY", 404.88, 1.0, "call", 4.0, 400.0, 0.25)
    short_leg = pf.Position("SPY", 404.88, 1.0, "call", -4.0, 400.0, 0.25)
    heatmap = rk.greeks_buckets([long_leg, short_leg])
    row, col = np.argwhere(heatmap.count > 0)[0]
    assert heatmap.count[row, col] == 2
    assert heatmap.delta[row, col] == pytest.approx(0.0, abs=1e-12)
    assert heatmap.notional[row, col] == pytest.approx(8.0 * 400.0 * 100)


# ============================================================================
# risk.py :: risk_limits_check
# ============================================================================

LIMITS_COLUMNS = ["greek", "current", "limit", "utilization_pct", "breach"]


@needs_bs
def test_risk_limits_check_columns_and_utilization(call_position):
    """Utilization is |current| / limit expressed in percent."""
    df = rk.risk_limits_check([call_position], {"delta": 1_000_000.0, "gamma": 1.0})
    assert list(df.columns) == LIMITS_COLUMNS
    assert list(df["greek"]) == ["delta", "gamma"]

    greeks = pf.aggregate_greeks([call_position])
    assert df.iloc[0]["current"] == pytest.approx(greeks.delta, rel=1e-12)
    assert df.iloc[0]["utilization_pct"] == pytest.approx(
        abs(greeks.delta) / 1_000_000.0 * 100, rel=1e-12
    )
    assert df.iloc[0]["limit"] == pytest.approx(1_000_000.0)


@needs_bs
def test_risk_limits_check_flags_breach_on_signed_limits():
    """A large net short delta breaches a tight limit and the utilisation is >100%."""
    short_call = pf.Position("SPY", 450.0, 0.25, "call", -20.0, 455.0, 0.20)
    df = rk.risk_limits_check([short_call], {"delta": 1.0})
    assert bool(df.iloc[0]["breach"]) is True
    assert df.iloc[0]["current"] < 0
    assert df.iloc[0]["utilization_pct"] > 100.0


@needs_bs
def test_risk_limits_check_reports_no_breach_when_inside_limits(call_position):
    """Comfortably inside every limit, no breach flags."""
    df = rk.risk_limits_check(
        [call_position], {"delta": 1e9, "gamma": 1e9, "vega": 1e9, "theta": 1e9, "rho": 1e9}
    )
    assert len(df) == 5
    assert not df["breach"].any()
    assert (df["utilization_pct"] < 100.0).all()


@needs_bs
def test_risk_limits_check_ignores_unknown_greeks(call_position):
    """Limit keys that are not Greeks are dropped from the report."""
    df = rk.risk_limits_check([call_position], {"delta": 1e9, "vanna": 1.0, "sigma": 1.0})
    assert list(df["greek"]) == ["delta"]


@needs_bs
def test_risk_limits_check_zero_limit_is_infinite_utilization(call_position):
    """A zero limit divides by zero on purpose: infinite utilisation, breach."""
    df = rk.risk_limits_check([call_position], {"gamma": 0.0})
    assert df.iloc[0]["utilization_pct"] == float("inf")
    assert bool(df.iloc[0]["breach"]) is True


@needs_bs
def test_risk_limits_check_empty_limits(call_position):
    """No limits means nothing to check."""
    df = rk.risk_limits_check([call_position], {})
    assert df.empty
    assert list(df.columns) == []


@needs_bs
def test_risk_limits_check_current_matches_aggregate_greeks(synthetic_book):
    """Reported exposure is the net book exposure, not a bucket total."""
    greeks = pf.aggregate_greeks(synthetic_book)
    limits = dict.fromkeys(greeks.as_dict(), 1e9)
    df = rk.risk_limits_check(synthetic_book, limits)
    for name in greeks.as_dict():
        row = df[df["greek"] == name].iloc[0]
        assert row["current"] == pytest.approx(getattr(greeks, name), rel=1e-12)


# ============================================================================
# risk.py :: stress_test_greeks
# ============================================================================

STRESS_COLUMNS = ["spot_shock_pct", "vol_shock", "rate_shock_bps", "total_pnl"]


@needs_bs
def test_stress_test_greeks_default_grid(call_position):
    """Defaults are 5 spot x 4 vol x 3 rate shocks = 60 scenarios."""
    df = rk.stress_test_greeks([call_position])
    assert list(df.columns) == STRESS_COLUMNS
    assert len(df) == 60
    assert sorted(df["spot_shock_pct"].unique()) == pytest.approx([-10.0, -5.0, 0.0, 5.0, 10.0])
    assert sorted(df["vol_shock"].unique()) == pytest.approx([-5.0, 0.0, 5.0, 10.0])
    assert sorted(df["rate_shock_bps"].unique()) == pytest.approx([-100.0, 0.0, 100.0])


@needs_bs
def test_stress_test_greeks_zero_shock_is_zero_pnl(call_position):
    """The identity scenario reprices to itself, so P&L is exactly zero."""
    df = rk.stress_test_greeks([call_position], [0.0], [0.0], [0.0])
    assert len(df) == 1
    assert df.iloc[0]["total_pnl"] == pytest.approx(0.0, abs=1e-12)


@needs_bs
def test_stress_test_greeks_matches_pnl_waterfall(call_position):
    """One grid cell is the TOTAL row of the equivalent pnl_waterfall run."""
    spot_shock, vol_shock, rate_shock = 0.05, 0.05, 0.01
    df = rk.stress_test_greeks([call_position], [spot_shock], [vol_shock], [rate_shock])
    pnl = pf.pnl_waterfall(
        [call_position],
        {"SPY": call_position.spot * spot_shock},
        {"SPY": vol_shock},
        {"SPY": rate_shock},
        time_decay=0.0,
    )
    expected = pnl[pnl["symbol"] == "TOTAL"].iloc[0]["exact_pnl"]
    assert df.iloc[0]["total_pnl"] == pytest.approx(expected, rel=1e-12)


@needs_bs
def test_stress_test_greeks_shock_columns_are_scaled_correctly():
    """Shocks are stored in percent and basis points, not decimals."""
    pos = pf.Position("SPY", 450.0, 0.25, "call", 1.0, 455.0, 0.20)
    df = rk.stress_test_greeks(pos and [pos], [0.03], [-0.02], [0.005])
    assert df.iloc[0]["spot_shock_pct"] == pytest.approx(3.0)
    assert df.iloc[0]["vol_shock"] == pytest.approx(-2.0)
    assert df.iloc[0]["rate_shock_bps"] == pytest.approx(50.0)


@needs_bs
def test_stress_test_greeks_pnl_is_monotone_in_spot_for_a_long_call(call_position):
    """More spot appreciation means more P&L for a long call."""
    df = rk.stress_test_greeks([call_position], [-0.10, -0.05, 0.0, 0.05, 0.10], [0.0], [0.0])
    pnl = df["total_pnl"].tolist()
    assert all(pnl[i] < pnl[i + 1] for i in range(len(pnl) - 1))


@needs_bs
def test_stress_test_greeks_pnl_is_monotone_in_vol_for_a_long_call(call_position):
    """A long call gains from a vol rally."""
    df = rk.stress_test_greeks([call_position], [0.0], [-0.05, 0.0, 0.05], [0.0])
    pnl = df["total_pnl"].tolist()
    assert all(pnl[i] < pnl[i + 1] for i in range(len(pnl) - 1))


@needs_bs
def test_stress_test_greeks_short_book_mirrors_long_book(call_position):
    """Doubling the short is the negative of doubling the long."""
    long_book = [call_position, call_position]
    short_book = [
        pf.Position("SPY", 450.0, 0.25, "call", -10.0, 455.0, 0.20),
        pf.Position("SPY", 450.0, 0.25, "call", -10.0, 455.0, 0.20),
    ]
    shocks = ([0.0, 0.10], [0.0, 0.05], [0.0])
    long_pnl = rk.stress_test_greeks(long_book, *shocks)["total_pnl"].tolist()
    short_pnl = rk.stress_test_greeks(short_book, *shocks)["total_pnl"].tolist()
    assert short_pnl == pytest.approx([-x for x in long_pnl], rel=1e-12)


@needs_bs
def test_stress_test_greeks_custom_grid_size():
    """The row count is the product of the three shock lists."""
    df = rk.stress_test_greeks(
        [pf.Position("SPY", 100.0, 0.5, "call", 1.0, 100.0, 0.2)],
        [-0.05, 0.0, 0.05],
        [0.0, 0.02],
        [0.0, 0.005, -0.005],
    )
    assert len(df) == 3 * 2 * 3
    assert df["total_pnl"].notna().all()


# ============================================================================
# risk.py :: plotting helpers
# ============================================================================


@needs_bs
def test_greeks_heatmap_plot_single_panel(synthetic_book):
    """One Greek renders as a single heatmap panel.

    matplotlib and seaborn are not declared dependencies, so the only reachable
    outcome on a bare install is the import failure.
    """
    heatmap = rk.greeks_buckets(synthetic_book)
    if not _plotting_available():
        with pytest.raises(ImportError):
            rk.greeks_heatmap_plot(heatmap, greek="delta")
        return
    import matplotlib

    matplotlib.use("Agg")
    fig = rk.greeks_heatmap_plot(heatmap, greek="vega", title="Vega")
    assert len(fig.axes) >= 1
    assert fig.axes[0].get_title() == "Vega"


@needs_bs
def test_greeks_heatmap_plot_without_centre_zero(synthetic_book):
    """The centre_zero=False branch drops the symmetric vmin/vmax."""
    heatmap = rk.greeks_buckets(synthetic_book)
    if not _plotting_available():
        with pytest.raises(ImportError):
            rk.greeks_heatmap_plot(heatmap, greek="gamma", center_zero=False)
        return
    import matplotlib

    matplotlib.use("Agg")
    fig = rk.greeks_heatmap_plot(heatmap, greek="gamma", center_zero=False)
    assert len(fig.axes) >= 1


@needs_bs
def test_plot_all_greeks_heatmaps_grid(synthetic_book):
    """The overview renders six panels in a 2x3 grid."""
    heatmap = rk.greeks_buckets(synthetic_book)
    if not _plotting_available():
        with pytest.raises(ImportError):
            rk.plot_all_greeks_heatmaps(heatmap)
        return
    import matplotlib

    matplotlib.use("Agg")
    fig = rk.plot_all_greeks_heatmaps(heatmap)
    assert len(fig.axes) == 6


# ============================================================================
# surface.py :: ChainQuote / build_chain_dataframe
# ============================================================================


def test_chain_quote_defaults():
    """ChainQuote defaults to no implied vol and a call."""
    quote = sf.ChainQuote(strike=100.0, bid=1.0, ask=1.2, mid=1.1, volume=10, open_interest=20)
    assert quote.strike == 100.0
    assert quote.implied_vol is None
    assert quote.option_type == "call"


def test_chain_quote_full_fields():
    """Every field round-trips, including the optional ones."""
    quote = sf.ChainQuote(
        strike=105.0,
        bid=2.0,
        ask=2.4,
        mid=2.2,
        volume=3,
        open_interest=40,
        implied_vol=0.31,
        option_type="put",
    )
    assert quote.implied_vol == pytest.approx(0.31)
    assert quote.option_type == "put"
    assert quote.mid == pytest.approx((quote.bid + quote.ask) / 2)


def test_build_chain_dataframe_computes_mid_and_int_types():
    """Mid is the bid/ask midpoint; volume and open interest stay integral."""
    df = sf.build_chain_dataframe(
        strikes=np.array([90.0, 100.0]),
        bids=np.array([1.0, 2.0]),
        asks=np.array([1.2, 2.2]),
        volumes=np.array([5, 6]),
        open_interests=np.array([7, 8]),
    )
    assert list(df.columns) == [
        "strike",
        "bid",
        "ask",
        "volume",
        "open_interest",
        "mid",
        "option_type",
    ]
    assert df["mid"].tolist() == pytest.approx([1.1, 2.1])
    assert df["volume"].tolist() == [5, 6]
    assert pd.api.types.is_integer_dtype(df["volume"])
    assert pd.api.types.is_integer_dtype(df["open_interest"])
    assert set(df["option_type"]) == {"call"}


def test_build_chain_dataframe_adds_implied_vol_only_when_given():
    """The implied_vol column is optional."""
    without = sf.build_chain_dataframe([90.0], [1.0], [1.2], [5], [7])
    assert "implied_vol" not in without.columns
    with_vols = sf.build_chain_dataframe(
        [90.0], [1.0], [1.2], [5], [7], implied_vols=[0.25], option_type="put"
    )
    assert with_vols["implied_vol"].tolist() == pytest.approx([0.25])
    assert set(with_vols["option_type"]) == {"put"}


# ============================================================================
# surface.py :: chain_to_volsurface
# ============================================================================


@needs_bs
def test_chain_to_volsurface_inverts_prices_back_to_the_input_smile(option_chain, smile_vols):
    """Mid-price inversion recovers the generating volatilities."""
    surface = sf.chain_to_volsurface(100.0, option_chain, expiry=0.5, risk_free_rate=0.05)
    assert np.allclose(surface.implied_vols, smile_vols, atol=1e-8)
    assert surface.spot == pytest.approx(100.0)
    assert surface.time_to_maturity == pytest.approx(0.5)
    assert surface.risk_free_rate == pytest.approx(0.05)
    assert surface.option_type is OptionType.CALL


@needs_bs
def test_chain_to_volsurface_interpolates_quoted_strikes(option_chain, smile_strikes):
    """Each quoted strike keeps its own implied vol on the built surface."""
    surface = sf.chain_to_volsurface(100.0, option_chain, expiry=0.5, risk_free_rate=0.05)
    for strike, vol in zip(surface.strikes, surface.implied_vols, strict=True):
        assert surface.implied_volatility(strike) == pytest.approx(vol, abs=1e-10)
    assert surface.strikes == pytest.approx(tuple(smile_strikes))


@needs_bs
def test_chain_to_volsurface_forward_price(option_chain):
    """The surface forward is spot grown at r-q for the expiry."""
    surface = sf.chain_to_volsurface(100.0, option_chain, expiry=0.5, risk_free_rate=0.05)
    assert surface.forward == pytest.approx(100.0 * math.exp(0.05 * 0.5))


@needs_bs
def test_chain_to_volsurface_prefers_supplied_implied_vol(smile_strikes, smile_vols):
    """A pre-computed implied_vol column short-circuits the price inversion."""
    spot, expiry, rate = 100.0, 0.5, 0.05
    prices = np.array(
        [
            black_scholes_price(OptionParams(spot, k, expiry, rate, v))
            for k, v in zip(smile_strikes, smile_vols, strict=True)
        ]
    )
    chain = sf.build_chain_dataframe(
        smile_strikes,
        prices * 0.5,
        prices * 0.5,
        np.full(7, 10),
        np.full(7, 10),
        implied_vols=smile_vols,
    )
    surface = sf.chain_to_volsurface(spot, chain, expiry=expiry, risk_free_rate=rate)
    assert surface.implied_vols == pytest.approx(tuple(smile_vols))


@needs_bs
def test_chain_to_volsurface_price_column_brackets_the_truth(option_chain, smile_vols):
    """Bid inversion lands below mid, ask above, and mid on the input smile."""
    args = {"expiry": 0.5, "risk_free_rate": 0.05}
    bid = np.array(
        sf.chain_to_volsurface(100.0, option_chain, price_column="bid", **args).implied_vols
    )
    mid = np.array(
        sf.chain_to_volsurface(100.0, option_chain, price_column="mid", **args).implied_vols
    )
    ask = np.array(
        sf.chain_to_volsurface(100.0, option_chain, price_column="ask", **args).implied_vols
    )

    assert np.all(bid < mid)
    assert np.all(mid < ask)
    assert np.allclose(mid, smile_vols, atol=1e-8)
    # The 10bp-wide quote implies roughly a 0.002 vol gap end to end.
    assert np.all((ask - bid) < 0.01)


@needs_bs
def test_chain_to_volsurface_put_chain_round_trip(smile_strikes, smile_vols):
    """Put prices invert to the same volatilities and tag the surface as PUT."""
    spot, expiry, rate = 100.0, 0.5, 0.05
    prices = np.array(
        [
            black_scholes_price(OptionParams(spot, k, expiry, rate, v, OptionType.PUT))
            for k, v in zip(smile_strikes, smile_vols, strict=True)
        ]
    )
    chain = sf.build_chain_dataframe(
        smile_strikes, prices * 0.999, prices * 1.001, np.full(7, 10), np.full(7, 10)
    )
    surface = sf.chain_to_volsurface(
        spot, chain, expiry=expiry, risk_free_rate=rate, option_type="put"
    )
    assert surface.option_type is OptionType.PUT
    assert np.allclose(surface.implied_vols, smile_vols, atol=1e-8)


@needs_bs
def test_chain_to_volsurface_with_dividend_yield(smile_strikes, smile_vols):
    """The dividend yield must be passed through or the smile shifts."""
    spot, expiry, rate, dividend = 100.0, 0.5, 0.05, 0.03
    prices = np.array(
        [
            black_scholes_price(OptionParams(spot, k, expiry, rate, v, OptionType.CALL, dividend))
            for k, v in zip(smile_strikes, smile_vols, strict=True)
        ]
    )
    chain = sf.build_chain_dataframe(
        smile_strikes, prices * 0.999, prices * 1.001, np.full(7, 10), np.full(7, 10)
    )
    surface = sf.chain_to_volsurface(
        spot, chain, expiry=expiry, risk_free_rate=rate, dividend_yield=dividend
    )
    assert np.allclose(surface.implied_vols, smile_vols, atol=1e-8)
    assert surface.forward == pytest.approx(spot * math.exp((rate - dividend) * expiry))


@needs_bs
def test_chain_to_volsurface_filters_volume_and_open_interest(option_chain):
    """Liquidity floors drop the quotes before any inversion happens."""
    thinned = option_chain.copy()
    thinned.loc[0, "volume"] = 0
    thinned.loc[1, "open_interest"] = 0
    surface = sf.chain_to_volsurface(
        100.0, thinned, expiry=0.5, risk_free_rate=0.05, min_volume=1, min_oi=1
    )
    assert surface.strikes == pytest.approx((95.0, 100.0, 105.0, 110.0, 120.0))


@needs_bs
def test_chain_to_volsurface_drops_zero_quotes(option_chain):
    """A zero bid or ask is not a quote and is discarded."""
    chain = option_chain.copy()
    chain.loc[2, "bid"] = 0.0
    surface = sf.chain_to_volsurface(100.0, chain, expiry=0.5, risk_free_rate=0.05)
    assert 95.0 not in surface.strikes
    assert len(surface.strikes) == 6


@needs_bs
def test_chain_to_volsurface_missing_column_raises(option_chain):
    """Required columns are validated up front."""
    with pytest.raises(ValueError, match="Missing required column: ask"):
        sf.chain_to_volsurface(
            100.0, option_chain.drop(columns=["ask"]), expiry=0.5, risk_free_rate=0.05
        )


@needs_bs
def test_chain_to_volsurface_unknown_price_column_raises(option_chain):
    """An unrecognised price selector is rejected."""
    with pytest.raises(ValueError, match="Unknown price_column: last"):
        sf.chain_to_volsurface(
            100.0, option_chain, expiry=0.5, risk_free_rate=0.05, price_column="last"
        )


@needs_bs
def test_chain_to_volsurface_insufficient_quotes_raises(option_chain):
    """Two quotes are the minimum; the default filters keep everything."""
    with pytest.raises(ValueError, match="Insufficient quotes after filtering"):
        sf.chain_to_volsurface(
            100.0, option_chain, expiry=0.5, risk_free_rate=0.05, min_volume=10_000
        )


@needs_bs
def test_chain_to_volsurface_uninvertible_prices_raise():
    """Prices below the no-arbitrage floor leave too few strikes to fit."""
    chain = pd.DataFrame(
        {
            "strike": [10.0, 20.0, 30.0],
            "bid": [0.001, 0.001, 0.001],
            "ask": [0.001, 0.001, 0.001],
            "volume": [1, 1, 1],
            "open_interest": [1, 1, 1],
        }
    )
    with pytest.raises(ValueError, match="Could not compute valid implied volatilities"):
        sf.chain_to_volsurface(100.0, chain, expiry=0.5, risk_free_rate=0.05)


def test_chain_to_volsurface_partial_nan_implied_vol_is_read_only():
    """Documents a real defect: NaN-filling writes into a read-only array.

    The NaN branch does ``vols = df["implied_vol"].values`` and then assigns into
    ``vols``. Under pandas copy-on-write that array is read-only. When the source
    is fixed, replace this with a round-trip assertion on the recovered vol.
    """
    if not HAS_BLACK_SCHOLES:
        pytest.skip("black_scholes not available")
    spot, expiry, rate = 100.0, 0.5, 0.05
    strikes = np.array([90.0, 100.0, 110.0])
    vols = np.array([0.24, 0.22, 0.24])
    prices = np.array(
        [
            black_scholes_price(OptionParams(spot, k, expiry, rate, v))
            for k, v in zip(strikes, vols, strict=True)
        ]
    )
    chain = sf.build_chain_dataframe(
        strikes, prices * 0.999, prices * 1.001, np.full(3, 10), np.full(3, 10)
    )
    chain.loc[1, "implied_vol"] = np.nan
    try:
        surface = sf.chain_to_volsurface(
            spot, chain, expiry=expiry, risk_free_rate=rate, price_column="bid"
        )
    except ValueError as exc:  # pandas >= 3 copy-on-write
        assert "read-only" in str(exc)
        return
    # Older pandas hands out a writable view; the NaN is then recovered from price.
    assert surface.strikes == pytest.approx((90.0, 100.0, 110.0))
    assert surface.implied_vols == pytest.approx(tuple(vols), abs=1e-3)


# ============================================================================
# surface.py :: chain_to_volsurface_multi
# ============================================================================


@needs_bs
def test_chain_to_volsurface_multi_builds_one_surface_per_expiry(
    option_chain, smile_strikes, smile_vols
):
    """Every expiry in the input dict gets its own surface."""
    chains = {}
    for expiry in (0.25, 0.5, 1.0):
        prices = np.array(
            [
                black_scholes_price(OptionParams(100.0, k, expiry, 0.05, v))
                for k, v in zip(smile_strikes, smile_vols, strict=True)
            ]
        )
        chains[expiry] = sf.build_chain_dataframe(
            smile_strikes, prices * 0.999, prices * 1.001, np.full(7, 10), np.full(7, 10)
        )
    surfaces = sf.chain_to_volsurface_multi(100.0, chains, risk_free_rate=0.05)
    assert sorted(surfaces) == [0.25, 0.5, 1.0]
    for expiry, surface in surfaces.items():
        assert surface.time_to_maturity == pytest.approx(expiry)
        assert np.allclose(surface.implied_vols, smile_vols, atol=1e-8)
        assert surface.spot == pytest.approx(100.0)


@needs_bs
def test_chain_to_volsurface_multi_skips_unusable_expiry(option_chain):
    """A failing expiry is dropped instead of aborting the whole build."""
    unusable = pd.DataFrame(
        {
            "strike": [10.0, 20.0],
            "bid": [0.001, 0.001],
            "ask": [0.001, 0.001],
            "volume": [1, 1],
            "open_interest": [1, 1],
        }
    )
    surfaces = sf.chain_to_volsurface_multi(
        100.0, {0.25: option_chain, 0.5: unusable, 1.0: option_chain}
    )
    assert sorted(surfaces) == [0.25, 1.0]


@needs_bs
def test_chain_to_volsurface_multi_empty_input():
    """No chains means no surfaces."""
    assert sf.chain_to_volsurface_multi(100.0, {}) == {}


# ============================================================================
# surface.py :: compute_chain_greeks
# ============================================================================

CHAIN_GREEKS_COLUMNS = [
    "Strike",
    "Moneyness",
    "IV",
    "Call Price",
    "Call Delta",
    "Call Gamma",
    "Call Vega",
    "Call Theta",
    "Call Rho",
    "Put Price",
    "Put Delta",
    "Put Gamma",
    "Put Vega",
    "Put Theta",
    "Put Rho",
]


@needs_bs
def test_compute_chain_greeks_columns_and_row_count(single_surface, smile_strikes):
    """One row per surface strike, with both call and put Greeks."""
    df = sf.compute_chain_greeks(single_surface, expiry=0.5, risk_free_rate=0.05)
    assert list(df.columns) == CHAIN_GREEKS_COLUMNS
    assert len(df) == len(smile_strikes)
    assert df["Strike"].tolist() == pytest.approx(tuple(smile_strikes))


@needs_bs
def test_compute_chain_greeks_iv_matches_the_surface(single_surface, smile_vols):
    """The IV column is read straight off the surface, not re-inverted."""
    df = sf.compute_chain_greeks(single_surface, expiry=0.5, risk_free_rate=0.05)
    assert df["IV"].tolist() == pytest.approx(list(smile_vols), abs=1e-8)


@needs_bs
def test_compute_chain_greeks_moneyness_is_against_the_forward(single_surface):
    """Moneyness is K/F, so the forward-ATM strike is just under 1."""
    df = sf.compute_chain_greeks(single_surface, expiry=0.5, risk_free_rate=0.05)
    forward = single_surface.forward
    assert df["Moneyness"].tolist() == pytest.approx([k / forward for k in single_surface.strikes])
    assert 0.95 < df.loc[df["Strike"] == 100.0, "Moneyness"].iloc[0] < 1.0


@needs_bs
def test_compute_chain_greeks_custom_strike_subset(single_surface):
    """An explicit strike array overrides the surface's own strikes."""
    df = sf.compute_chain_greeks(
        single_surface, expiry=0.5, strikes=np.array([100.0, 120.0]), risk_free_rate=0.05
    )
    assert len(df) == 2
    assert df["Strike"].tolist() == pytest.approx([100.0, 120.0])
    assert df["IV"].tolist() == pytest.approx(
        [single_surface.implied_volatility(100.0), single_surface.implied_volatility(120.0)],
        abs=1e-10,
    )


@needs_bs
def test_compute_chain_greeks_put_call_price_parity(single_surface):
    """C - P equals the discounted forward spread at the requested expiry."""
    expiry, rate, dividend = 0.5, 0.05, 0.0
    df = sf.compute_chain_greeks(
        single_surface, expiry=expiry, risk_free_rate=rate, dividend_yield=dividend
    )
    parity = single_surface.spot * math.exp(-dividend * expiry) - df["Strike"].values * math.exp(
        -rate * expiry
    )
    assert (df["Call Price"] - df["Put Price"]).values == pytest.approx(parity, abs=1e-10)


@needs_bs
def test_compute_chain_greeks_put_call_delta_parity(single_surface):
    """Delta_call - Delta_put equals the discount on the dividend."""
    df = sf.compute_chain_greeks(
        single_surface, expiry=0.5, risk_free_rate=0.05, dividend_yield=0.02
    )
    assert (df["Call Delta"] - df["Put Delta"]).values == pytest.approx(
        math.exp(-0.02 * 0.5), abs=1e-12
    )


@needs_bs
def test_compute_chain_greeks_gamma_and_vega_are_type_independent(single_surface):
    """Gamma and vega do not depend on the option type; delta does."""
    df = sf.compute_chain_greeks(single_surface, expiry=0.5, risk_free_rate=0.05)
    assert df["Call Gamma"].values == pytest.approx(df["Put Gamma"].values, abs=1e-12)
    assert df["Call Vega"].values == pytest.approx(df["Put Vega"].values, abs=1e-12)
    assert (df["Call Delta"].values > df["Put Delta"].values).all()


@needs_bs
def test_compute_chain_greeks_signs_follow_moneyness(single_surface):
    """Delta falls with strike; call rho is positive, put rho negative."""
    df = sf.compute_chain_greeks(single_surface, expiry=0.5, risk_free_rate=0.05)
    assert np.all(np.diff(df["Call Delta"].values) < 0)
    assert (df["Call Delta"].values > 0).all()
    assert (df["Put Delta"].values < 0).all()
    assert (df["Call Rho"].values > 0).all()
    assert (df["Put Rho"].values < 0).all()
    assert (df["Call Gamma"].values > 0).all()
    assert (df["Call Vega"].values > 0).all()


@needs_bs
def test_compute_chain_greeks_prices_match_independent_pricing(
    single_surface, smile_strikes, smile_vols
):
    """Each row is a full Black-Scholes reprice at the surface implied vol."""
    expiry, rate = 0.5, 0.05
    df = sf.compute_chain_greeks(single_surface, expiry=expiry, risk_free_rate=rate)
    for strike, vol in zip(smile_strikes, smile_vols, strict=True):
        row = df[df["Strike"] == strike].iloc[0]
        call_ref = reference_greeks(100.0, strike, expiry, rate, vol, 0.0, "call")
        put_ref = reference_greeks(100.0, strike, expiry, rate, vol, 0.0, "put")
        assert row["Call Price"] == pytest.approx(call_ref["price"], rel=1e-10)
        assert row["Put Price"] == pytest.approx(put_ref["price"], rel=1e-10)
        assert row["Call Delta"] == pytest.approx(call_ref["delta"], rel=1e-10)
        assert row["Put Gamma"] == pytest.approx(put_ref["gamma"], rel=1e-10)


# ============================================================================
# surface.py :: compute_svi_parameters_per_expiry
# ============================================================================

SVI_COLUMNS = [
    "Expiry",
    "ATM Variance",
    "ATM Skew",
    "Put Wing Slope",
    "Call Wing Slope",
    "Min Variance",
    "RMS Error",
    "Max Error",
]


@needs_bs
def test_compute_svi_parameters_columns(surfaces_by_expiry):
    """One row per expiry with the full SVI-Jump-Wings report."""
    df = sf.compute_svi_parameters_per_expiry(surfaces_by_expiry)
    assert list(df.columns) == SVI_COLUMNS
    assert len(df) == len(surfaces_by_expiry)


@needs_bs
def test_compute_svi_parameters_sorted_by_expiry(surfaces_by_expiry):
    """Rows come back in ascending expiry order regardless of dict order."""
    shuffled = dict(reversed(list(surfaces_by_expiry.items())))
    df = sf.compute_svi_parameters_per_expiry(shuffled)
    assert df["Expiry"].tolist() == sorted(surfaces_by_expiry)


@needs_bs
def test_compute_svi_parameters_fit_the_input_smile(surfaces_by_expiry):
    """A smooth synthetic smile is fitted tightly, so the errors are small."""
    df = sf.compute_svi_parameters_per_expiry(surfaces_by_expiry)
    assert (df["RMS Error"] < 0.005).all()
    assert (df["Max Error"] < 0.01).all()
    assert (df["Max Error"] >= df["RMS Error"]).all()


@needs_bs
def test_compute_svi_parameters_atm_variance_matches_term_structure(surfaces_by_expiry):
    """ATM variance should approximate the square of the quoted ATM level."""
    df = sf.compute_svi_parameters_per_expiry(surfaces_by_expiry).set_index("Expiry")
    for expiry, surface in surfaces_by_expiry.items():
        atm_vol = surface.implied_volatility(surface.forward)
        assert df.loc[expiry, "ATM Variance"] == pytest.approx(atm_vol**2, abs=0.005)


@needs_bs
def test_compute_svi_parameters_match_the_jump_wings_formula(surfaces_by_expiry):
    """Recompute the JW parameters from the raw SVI parameters independently."""
    df = sf.compute_svi_parameters_per_expiry(surfaces_by_expiry).set_index("Expiry")
    for expiry, surface in surfaces_by_expiry.items():
        curve = surface.fit().curve
        a, b, rho, m, sigma = curve.a, curve.b, curve.rho, curve.m, curve.sigma
        root_m = math.sqrt(m * m + sigma * sigma)
        atm_variance = (a + b * (-rho * m + root_m)) / expiry
        sqrt_w = math.sqrt(atm_variance * expiry)

        assert df.loc[expiry, "ATM Variance"] == pytest.approx(atm_variance, rel=1e-12)
        assert df.loc[expiry, "ATM Skew"] == pytest.approx(
            (b / 2.0) * (-m / root_m + rho) / sqrt_w, rel=1e-12
        )
        assert df.loc[expiry, "Put Wing Slope"] == pytest.approx(
            b * (1.0 - rho) / sqrt_w, rel=1e-12
        )
        assert df.loc[expiry, "Call Wing Slope"] == pytest.approx(
            b * (1.0 + rho) / sqrt_w, rel=1e-12
        )
        assert df.loc[expiry, "Min Variance"] == pytest.approx(
            (a + b * sigma * math.sqrt(1.0 - rho * rho)) / expiry, rel=1e-12
        )


@needs_bs
def test_compute_svi_parameters_min_variance_below_atm(surfaces_by_expiry):
    """The variance minimum sits at or below the ATM level for these smiles."""
    df = sf.compute_svi_parameters_per_expiry(surfaces_by_expiry)
    assert (df["Min Variance"] <= df["ATM Variance"]).all()
    assert (df["ATM Variance"] > 0).all()


@needs_bs
def test_compute_svi_parameters_term_structure_rises_with_expiry(surfaces_by_expiry):
    """The fixture has ATM vol rising with expiry, so ATM variance must too."""
    df = sf.compute_svi_parameters_per_expiry(surfaces_by_expiry)
    assert list(df["ATM Variance"]) == sorted(df["ATM Variance"])


@needs_bs
def test_compute_svi_parameters_skips_surface_with_too_few_strikes(surfaces_by_expiry):
    """SVI needs four strikes; a three-strike slice is skipped, not fatal."""
    tiny = VolSurface(
        spot=100.0,
        strikes=(90.0, 100.0, 110.0),
        implied_vols=(0.23, 0.22, 0.23),
        time_to_maturity=0.5,
        risk_free_rate=0.05,
        dividend_yield=0.0,
    )
    df = sf.compute_svi_parameters_per_expiry({0.5: tiny})
    assert df.empty
    assert len(df.columns) == 0


@needs_bs
def test_compute_svi_parameters_empty_input():
    """No surfaces means no rows."""
    df = sf.compute_svi_parameters_per_expiry({})
    assert df.empty


# ============================================================================
# surface.py :: compute_term_structure
# ============================================================================


@needs_bs
def test_compute_term_structure_columns(flat_surfaces):
    """Default delta of 0.25 names the skew column '25d Skew'."""
    df = sf.compute_term_structure(flat_surfaces)
    assert list(df.columns) == [
        "Expiry",
        "ATM Vol",
        "25d Skew",
        "Curvature",
        "ATM Variance",
        "ATM Skew (JW)",
        "Put Wing",
        "Call Wing",
    ]
    assert len(df) == len(flat_surfaces)


@needs_bs
def test_compute_term_structure_flat_smile_is_flat(flat_surfaces):
    """A flat 22% smile has no skew and no curvature, exactly."""
    df = sf.compute_term_structure(flat_surfaces).set_index("Expiry")
    for expiry in flat_surfaces:
        assert df.loc[expiry, "ATM Vol"] == pytest.approx(0.22, abs=1e-6)
        assert df.loc[expiry, "25d Skew"] == pytest.approx(0.0, abs=1e-9)
        assert df.loc[expiry, "Curvature"] == pytest.approx(0.0, abs=1e-9)
        assert df.loc[expiry, "ATM Variance"] == pytest.approx(0.22**2, rel=1e-6)
        assert df.loc[expiry, "ATM Skew (JW)"] == pytest.approx(0.0, abs=1e-9)


@needs_bs
def test_compute_term_structure_sorted_by_expiry(surfaces_by_expiry):
    """Rows are ordered by expiry, not by dict insertion order."""
    shuffled = dict(reversed(list(surfaces_by_expiry.items())))
    df = sf.compute_term_structure(shuffled)
    assert df["Expiry"].tolist() == sorted(surfaces_by_expiry)


@needs_bs
def test_compute_term_structure_detects_a_negative_skew(smile_strikes):
    """Vol falling as strike rises (put skew) shows up as negative 25d skew."""
    forward = 100.0 * math.exp(0.05 * 1.0)
    log_moneyness = np.log(smile_strikes / forward)
    vols = 0.22 - 0.15 * log_moneyness + 0.15 * np.abs(log_moneyness)
    surface = VolSurface(
        spot=100.0,
        strikes=tuple(float(k) for k in smile_strikes),
        implied_vols=tuple(float(v) for v in vols),
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        dividend_yield=0.0,
    )
    row = sf.compute_term_structure({1.0: surface}).iloc[0]
    assert row["25d Skew"] < 0
    assert row["ATM Skew (JW)"] < 0
    # The put wing is loaded, the call wing is nearly flat.
    assert row["Put Wing"] > row["Call Wing"]


@needs_bs
def test_compute_term_structure_detects_a_positive_skew(smile_strikes):
    """The mirror-image smile flips the sign of every skew measure."""
    forward = 100.0 * math.exp(0.05 * 1.0)
    log_moneyness = np.log(smile_strikes / forward)
    vols = 0.22 + 0.15 * log_moneyness + 0.15 * np.abs(log_moneyness)
    surface = VolSurface(
        spot=100.0,
        strikes=tuple(float(k) for k in smile_strikes),
        implied_vols=tuple(float(v) for v in vols),
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        dividend_yield=0.0,
    )
    row = sf.compute_term_structure({1.0: surface}).iloc[0]
    assert row["25d Skew"] > 0
    assert row["ATM Skew (JW)"] > 0
    assert row["Call Wing"] > row["Put Wing"]


@needs_bs
def test_compute_term_structure_curvature_is_positive_for_a_butterfly(smile_strikes):
    """A convex smile means the wings sit above the ATM level."""
    forward = 100.0 * math.exp(0.05 * 1.0)
    log_moneyness = np.log(smile_strikes / forward)
    vols = 0.22 - 0.05 * log_moneyness + 0.15 * np.abs(log_moneyness)
    surface = VolSurface(
        spot=100.0,
        strikes=tuple(float(k) for k in smile_strikes),
        implied_vols=tuple(float(v) for v in vols),
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        dividend_yield=0.0,
    )
    row = sf.compute_term_structure({1.0: surface}).iloc[0]
    assert row["Curvature"] > 0


@needs_bs
def test_compute_term_structure_delta_parameter_renames_the_column(flat_surfaces):
    """A different delta changes the label, not the calculation."""
    df = sf.compute_term_structure(flat_surfaces, delta=0.10)
    assert "10d Skew" in df.columns
    assert "25d Skew" not in df.columns
    assert df["10d Skew"].tolist() == pytest.approx([0.0, 0.0], abs=1e-9)


@needs_bs
def test_compute_term_structure_atm_variance_matches_atm_vol_squared(surfaces_by_expiry):
    """The two ATM columns are consistent with each other."""
    df = sf.compute_term_structure(surfaces_by_expiry)
    assert df["ATM Variance"].tolist() == pytest.approx([v**2 for v in df["ATM Vol"]], rel=1e-6)


@needs_bs
def test_compute_term_structure_skips_unfittable_surface(flat_surfaces):
    """A slice SVI cannot fit is dropped, leaving the rest of the curve."""
    tiny = VolSurface(
        spot=100.0,
        strikes=(90.0, 100.0, 110.0),
        implied_vols=(0.23, 0.22, 0.23),
        time_to_maturity=0.75,
        risk_free_rate=0.05,
        dividend_yield=0.0,
    )
    df = sf.compute_term_structure({0.75: tiny, 0.25: flat_surfaces[0.25]})
    assert df["Expiry"].tolist() == [0.25]


@needs_bs
def test_compute_term_structure_empty_input():
    """No surfaces means no rows."""
    assert sf.compute_term_structure({}).empty


# ============================================================================
# Cross-module integration
# ============================================================================


@needs_bs
def test_portfolio_and_risk_agree_on_every_greek(synthetic_book):
    """The heatmap and the flat aggregate are two views of the same numbers."""
    flat = pf.aggregate_greeks(synthetic_book).as_dict()
    heatmap = rk.greeks_buckets(synthetic_book)
    for greek, value in flat.items():
        assert getattr(heatmap, greek).sum() == pytest.approx(value, abs=1e-9), greek


@needs_bs
def test_portfolio_bucket_and_heatmap_bucket_agree(synthetic_book):
    """aggregate_greeks_by_bucket and greeks_buckets bucket identically."""
    by_bucket = pf.aggregate_greeks_by_bucket(synthetic_book)
    heatmap = rk.greeks_buckets(synthetic_book)
    assert by_bucket["count"].sum() == heatmap.count.sum()
    assert by_bucket["notional"].sum() == pytest.approx(heatmap.notional.sum())
    assert by_bucket["delta"].sum() == pytest.approx(heatmap.delta.sum(), abs=1e-9)


@needs_bs
def test_synthetic_portfolio_drives_the_whole_risk_stack(synthetic_book):
    """A generated book flows through aggregation, heatmap, limits and stress."""
    limits = rk.risk_limits_check(synthetic_book, {"delta": 1e9, "vega": 1e9})
    assert not limits["breach"].any()

    stress = rk.stress_test_greeks(synthetic_book, [-0.10, -0.05, 0.0, 0.05, 0.10], [0.0], [0.0])
    assert len(stress) == 5
    assert stress["total_pnl"].notna().all()
    assert stress["total_pnl"].iloc[2] == pytest.approx(0.0, abs=1e-9)

    # This generated book is net short gamma, so convexity works against it in
    # both directions and the flat spot is the best of the five scenarios.
    assert pf.aggregate_greeks(synthetic_book).gamma < 0
    assert stress["total_pnl"].iloc[2] == stress["total_pnl"].max()
    assert (stress["total_pnl"] <= 0).all()


@needs_bs
def test_chain_surface_feeds_the_pnl_waterfall(option_chain, single_surface):
    """A surface built from a chain prices a portfolio P&L consistently."""
    spots = np.array(single_surface.strikes)
    positions = [
        pf.Position("SYN", float(strike), 0.5, "call", 1.0, 100.0, float(vol))
        for strike, vol in zip(spots, single_surface.implied_vols, strict=True)
    ]
    reference = pf.aggregate_greeks(positions)
    stress = rk.stress_test_greeks(positions, [0.05], [0.0], [0.0])
    zero = rk.stress_test_greeks(positions, [0.0], [0.0], [0.0])
    assert zero["total_pnl"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    # Buying the whole smile at zero cost would pay off on an up move.
    assert stress["total_pnl"].iloc[0] > 0.0
    assert reference.gamma > 0


# ============================================================================
# Optional-dependency guards
# ============================================================================
#
# black_scholes, matplotlib and seaborn are all optional extras. Every entry
# point below promises a clear ImportError when its dependency is absent,
# rather than a NameError or AttributeError from a module-level None.


@needs_bs
def test_position_to_option_params_requires_black_scholes(call_position, monkeypatch):
    """Position.to_option_params fails loudly when the kernel is missing."""
    monkeypatch.setattr(pf, "OptionParams", None)
    with pytest.raises(ImportError, match="black_scholes package required"):
        call_position.to_option_params()


@needs_bs
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda positions: pf.aggregate_greeks(positions), id="aggregate_greeks"),
        pytest.param(
            lambda positions: pf.aggregate_greeks_by_bucket(positions),
            id="aggregate_greeks_by_bucket",
        ),
        pytest.param(lambda positions: pf.pnl_waterfall(positions, {}), id="pnl_waterfall"),
    ],
)
def test_portfolio_entry_points_require_black_scholes(call_position, monkeypatch, call):
    """Each portfolio entry point raises ImportError without the kernel."""
    monkeypatch.setattr(pf, "black_scholes_greeks", None)
    with pytest.raises(ImportError, match="black_scholes package required"):
        call([call_position])


@needs_bs
def test_chain_to_volsurface_requires_black_scholes(option_chain, monkeypatch):
    """The chain builder refuses to run without the kernel."""
    monkeypatch.setattr(sf, "VolSurface", None)
    with pytest.raises(ImportError, match="black_scholes package required"):
        sf.chain_to_volsurface(100.0, option_chain, expiry=0.5)


@needs_bs
def test_compute_chain_greeks_requires_black_scholes(single_surface, monkeypatch):
    """Greeks for a chain refuse to run without the kernel."""
    monkeypatch.setattr(sf, "VolSurface", None)
    with pytest.raises(ImportError, match="black_scholes package required"):
        sf.compute_chain_greeks(single_surface, expiry=0.5)


@needs_bs
def test_compute_svi_parameters_per_expiry_requires_black_scholes(surfaces_by_expiry, monkeypatch):
    """The SVI-JW report refuses to run without the kernel."""
    monkeypatch.setattr(sf, "VolSurface", None)
    with pytest.raises(ImportError, match="black_scholes package required"):
        sf.compute_svi_parameters_per_expiry(surfaces_by_expiry)


@needs_bs
def test_compute_term_structure_requires_black_scholes(flat_surfaces, monkeypatch):
    """The term structure refuses to run without the kernel."""
    monkeypatch.setattr(sf, "VolSurface", None)
    with pytest.raises(ImportError, match="black_scholes package required"):
        sf.compute_term_structure(flat_surfaces)


@needs_bs
def test_greeks_buckets_requires_black_scholes(call_position, monkeypatch):
    """greeks_buckets re-imports the kernel inside the call, so hide it there."""
    monkeypatch.setitem(sys.modules, "black_scholes", None)
    with pytest.raises(ImportError, match="black_scholes package required"):
        rk.greeks_buckets([call_position])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
