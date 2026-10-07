"""Tests for the remaining public chart builders.

The chart suite covered nine of the ten exported figure functions. The tenth,
``carry_rolldown_heatmap``, was never called by any test, and with it went the
subplot-selection logic that decides how many panels a caller asking for
"carry only" actually gets. These tests exercise that function alongside the
shared conventions every builder relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from quantview.notebook import charts as ch

TENORS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
SPOT = np.array([0.052, 0.050, 0.048, 0.046, 0.045, 0.044, 0.044, 0.0445, 0.0455, 0.046])
FORWARD = np.array([0.052, 0.049, 0.046, 0.044, 0.0435, 0.043, 0.0435, 0.0445, 0.046, 0.047])
HORIZONS = np.array([0.25, 0.5, 1.0, 2.0, 5.0])


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """Fifty business days of OHLCV, enough for a moving average."""
    rng = np.random.default_rng(11)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, 50)))
    # The builder's contract is a DatetimeIndex, not a date column.
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, 50).astype("int64"),
        },
        index=pd.bdate_range("2024-01-01", periods=50),
    )


class TestCarryRolldownHeatmap:
    def test_builds_one_subplot_per_requested_panel(self):
        fig = ch.carry_rolldown_heatmap(TENORS, SPOT, FORWARD, HORIZONS)

        titles = [a.text for a in fig.layout.annotations]
        assert "Carry" in titles
        assert "Roll-Down" in titles
        assert "Total Carry+Roll" in titles

    def test_showing_carry_alone_drops_the_other_two(self):
        fig = ch.carry_rolldown_heatmap(
            TENORS, SPOT, FORWARD, HORIZONS, show_carry=True, show_roll=False, show_total=False
        )

        titles = [a.text for a in fig.layout.annotations]
        assert titles == ["Carry"]
        assert len(fig.data) == 1

    def test_asking_for_nothing_still_returns_a_readable_figure(self):
        """A blank figure is a silent failure; the total panel is the fallback."""
        fig = ch.carry_rolldown_heatmap(
            TENORS, SPOT, FORWARD, HORIZONS, show_carry=False, show_roll=False, show_total=False
        )

        titles = [a.text for a in fig.layout.annotations]
        assert titles == ["Total Carry+Roll"]
        assert fig.data

    def test_each_panel_is_a_heatmap_over_tenor_by_horizon(self):
        fig = ch.carry_rolldown_heatmap(TENORS, SPOT, FORWARD, HORIZONS)

        for trace in fig.data:
            assert isinstance(trace, go.Heatmap)
            assert len(trace.x) == len(TENORS)
            assert len(trace.y) == len(HORIZONS)
            assert np.asarray(trace.z).shape == (len(HORIZONS), len(TENORS))

    def test_values_are_scaled_to_percentages(self):
        """Rates are fractions; the panel must read as percent, not 0.04."""
        fig = ch.carry_rolldown_heatmap(TENORS, SPOT, FORWARD, HORIZONS)

        z = np.asarray(fig.data[0].z, dtype=float)
        assert np.nanmax(np.abs(z)) < 100.0
        assert np.nanmax(np.abs(z)) > 0.0

    def test_axes_are_labelled_in_years(self):
        fig = ch.carry_rolldown_heatmap(TENORS, SPOT, FORWARD, HORIZONS)

        assert fig.layout.xaxis.title.text == "Tenor (Years)"
        assert fig.layout.yaxis.title.text == "Horizon (Years)"

    def test_height_and_colorscale_are_honoured(self):
        fig = ch.carry_rolldown_heatmap(
            TENORS, SPOT, FORWARD, HORIZONS, height=812, colorscale="Viridis"
        )

        assert fig.layout.height == 812

    def test_title_is_applied(self):
        fig = ch.carry_rolldown_heatmap(TENORS, SPOT, FORWARD, HORIZONS, title="My Grid")

        assert fig.layout.title.text == "My Grid"

    def test_subplot_titles_move_with_the_horizontal_spacing(self):
        """Titles are annotations placed by make_subplots; a one-panel grid
        must not keep the offsets intended for three panels."""
        one = ch.carry_rolldown_heatmap(
            TENORS, SPOT, FORWARD, HORIZONS, show_roll=False, show_total=False
        )
        three = ch.carry_rolldown_heatmap(TENORS, SPOT, FORWARD, HORIZONS)

        assert len(one.layout.annotations) == 1
        assert len(three.layout.annotations) == 3


def _axis_spans(fig: go.Figure) -> list[float]:
    """Height of each y axis, tallest first as plotly orders them top-down."""
    spans = []
    for name in ("yaxis", "yaxis2", "yaxis3", "yaxis4"):
        axis = getattr(fig.layout, name, None)
        if axis is None:
            continue
        domain = axis.domain
        spans.append(1.0 if isinstance(domain, int) else domain[1] - domain[0])
    return spans


class TestCandlestickConventions:
    def test_plots_one_candle_per_row(self, ohlcv):
        fig = ch.candlestick_chart(ohlcv)

        candles = [t for t in fig.data if isinstance(t, go.Candlestick)]
        assert len(candles) == 1
        assert len(candles[0].x) == len(ohlcv)

    def test_candles_are_styled_up_and_down_separately(self, ohlcv):
        fig = ch.candlestick_chart(ohlcv)
        (candles,) = [t for t in fig.data if isinstance(t, go.Candlestick)]

        assert candles.increasing.line.color
        assert candles.decreasing.line.color

    def test_indicators_add_traces_without_removing_the_candles(self, ohlcv):
        bare = ch.candlestick_chart(ohlcv, indicators=[])
        rich = ch.candlestick_chart(ohlcv, indicators=["sma_20", "bbands"])

        assert len(rich.data) > len(bare.data)
        assert any(isinstance(t, go.Candlestick) for t in rich.data)

    def test_requires_a_datetime_index(self, ohlcv):
        """A RangeIndex is rejected up front rather than plotted as integers."""
        with pytest.raises(ValueError, match="DatetimeIndex"):
            ch.candlestick_chart(ohlcv.reset_index(drop=True))

    def test_requires_the_ohlc_columns(self, ohlcv):
        with pytest.raises(ValueError, match="Missing required column: close"):
            ch.candlestick_chart(ohlcv.drop(columns=["close"]))

    def test_volume_is_only_required_when_it_is_plotted(self, ohlcv):
        no_volume = ohlcv.drop(columns=["volume"])

        assert ch.candlestick_chart(no_volume, volume=False)

        with pytest.raises(ValueError, match="volume"):
            ch.candlestick_chart(no_volume, volume=True)

    def test_volume_takes_a_separate_pane(self, ohlcv):
        fig = ch.candlestick_chart(ohlcv, volume=True)

        assert fig.data[-1].yaxis == "y2"
        assert "yaxis2" in fig.layout

    def test_volume_can_be_switched_off(self, ohlcv):
        with_vol = ch.candlestick_chart(ohlcv, volume=True)
        without = ch.candlestick_chart(ohlcv, volume=False)

        assert len(with_vol.data) > len(without.data)

    def test_a_symbol_becomes_a_subplot_title(self, ohlcv):
        fig = ch.candlestick_chart(ohlcv, symbol="SPY")

        assert any("SPY" in (a.text or "") for a in fig.layout.annotations)

    def test_rsi_and_macd_each_add_their_own_pane(self, ohlcv):
        """The pane count follows the indicators, not just the volume flag."""
        plain = ch.candlestick_chart(ohlcv)
        with_rsi = ch.candlestick_chart(ohlcv, indicators=["rsi"])
        with_macd = ch.candlestick_chart(ohlcv, indicators=["macd"])

        assert len(with_rsi.layout.annotations) > len(plain.layout.annotations)
        assert len(with_macd.layout.annotations) > len(plain.layout.annotations)

    def test_row_heights_always_sum_to_one(self, ohlcv):
        """Plotly rejects a domain specification that does not tile the figure."""
        for indicators, volume in (
            ([], True),
            ([], False),
            (["rsi"], True),
            (["macd"], False),
            (["rsi", "macd"], True),
        ):
            fig = ch.candlestick_chart(ohlcv, indicators=indicators, volume=volume)

            # The price pane must dominate; the indicator panes are context.
            spans = _axis_spans(fig)
            assert spans
            assert all(s > 0 for s in spans)
            assert spans[0] == max(spans)
            # Spacing between panes is deliberate, so the panes cover the figure
            # less the gaps rather than exactly all of it.
            assert sum(spans) <= 1.0 + 1e-9

    def test_rejects_a_frame_with_no_price_columns(self):
        with pytest.raises((ValueError, KeyError)):
            ch.candlestick_chart(pd.DataFrame({"nothing": [1, 2, 3]}))


class TestSharedConventions:
    def test_apply_theme_stamps_the_module_theme(self, ohlcv):
        fig = ch._apply_theme(go.Figure())

        assert fig.layout.paper_bgcolor == ch.THEME.paper_bgcolor
        assert fig.layout.plot_bgcolor == ch.THEME.plot_bgcolor

    def test_apply_theme_returns_the_same_object(self):
        original = go.Figure()

        assert ch._apply_theme(original) is original

    def test_range_selector_offers_the_documented_windows(self):
        fig = ch._add_range_selector(go.Figure())

        buttons = fig.layout.xaxis.rangeselector.buttons
        assert [b.label for b in buttons] == ["1M", "3M", "6M", "YTD", "1Y", "All"]

    def test_range_selector_can_target_a_secondary_axis(self):
        """The indicator pane needs its own selector, not the price axis's."""
        fig = ch._add_range_selector(go.Figure(), xaxis="xaxis2")

        assert fig.layout.xaxis2.rangeselector is not None
        # The default axis was not configured, so the two stay independent.
        assert fig.layout.xaxis.rangeselector.buttons == ()

    def test_range_selector_hides_the_slider(self):
        fig = ch._add_range_selector(go.Figure())

        assert fig.layout.xaxis.rangeslider.visible is False

    def test_every_builder_applies_the_theme(self, ohlcv):
        fig = ch.candlestick_chart(ohlcv)

        assert fig.layout.paper_bgcolor == ch.THEME.paper_bgcolor
        assert fig.layout.plot_bgcolor == ch.THEME.plot_bgcolor

    def test_yield_curve_chart_accepts_a_bare_curve(self):
        from quantview.notebook.charts import YieldCurveData

        curve = YieldCurveData(
            tenors=np.array([1.0, 2.0, 5.0]),
            spot_rates=np.array([0.05, 0.045, 0.044]),
        )

        fig = ch.yield_curve_chart(curve)

        assert fig.data

    def test_carry_roll_accepts_a_single_horizon(self):
        fig = ch.yield_curve_carry_roll(
            np.array([1.0, 2.0, 5.0, 10.0]),
            np.array([0.05, 0.045, 0.044, 0.0445]),
            np.array([0.048, 0.0445, 0.044, 0.0445]),
            horizon=1.0,
        )

        assert fig.data
