"""Tests for the notebook widget factories.

widgets.py sat at 11% coverage while being the entire interactive surface of the
notebook side. ipywidgets construct without a kernel, so these factories can be
built and driven directly: the meaningful assertions are about defaults, ranges,
and what the callbacks actually receive when a value changes.
"""

from __future__ import annotations

from datetime import date, timedelta

import ipywidgets as widgets
import pytest
from traitlets import TraitError

from quantview.notebook import widgets as wg

SYMBOLS = ["AAPL", "MSFT", "SPY", "QQQ"]


def _set(widget: widgets.Widget, value) -> None:
    """Drive an observed widget the way a user interaction would."""
    widget.value = value


class TestSymbolDropdown:
    def test_defaults_to_the_first_symbol(self):
        w = wg.create_symbol_dropdown(SYMBOLS)

        assert isinstance(w, widgets.Dropdown)
        assert w.value == "AAPL"

    def test_honours_an_explicit_initial_value(self):
        assert wg.create_symbol_dropdown(SYMBOLS, value="SPY").value == "SPY"

    def test_carries_the_description_through(self):
        assert wg.create_symbol_dropdown(SYMBOLS, description="Ticker:").description == "Ticker:"

    def test_fires_the_callback_on_change(self):
        seen: list = []
        w = wg.create_symbol_dropdown(SYMBOLS, on_change=seen.append)

        _set(w, "QQQ")

        assert len(seen) == 1
        assert seen[0]["new"] == "QQQ"

    def test_no_callback_is_safe(self):
        w = wg.create_symbol_dropdown(SYMBOLS)

        _set(w, "MSFT")

        assert w.value == "MSFT"


class TestDatePicker:
    def test_defaults_to_today(self):
        assert wg.create_date_picker().value == date.today()

    def test_honours_an_explicit_value_and_bounds(self):
        w = wg.create_date_picker(
            value=date(2024, 6, 1), min_date=date(2020, 1, 1), max_date=date(2025, 1, 1)
        )

        assert w.value == date(2024, 6, 1)
        assert w.min == date(2020, 1, 1)
        assert w.max == date(2025, 1, 1)

    def test_fires_the_callback_on_change(self):
        seen: list = []
        w = wg.create_date_picker(on_change=seen.append)

        _set(w, date(2024, 3, 15))

        assert seen[-1]["new"] == date(2024, 3, 15)


class TestDateRangePicker:
    def test_defaults_to_the_last_year(self):
        w = wg.create_date_range_picker()

        assert isinstance(w, widgets.HBox)
        start, end = _range_pickers(w)
        assert end.value == date.today()
        assert start.value == date.today() - timedelta(days=365)

    def test_start_can_never_exceed_end(self):
        """The bound coupling is the whole point of the pair."""
        w = wg.create_date_range_picker(start=date(2024, 1, 1), end=date(2024, 6, 1))
        start, end = _range_pickers(w)

        assert end.min == date(2024, 1, 1)
        assert start.max == date(2024, 6, 1)

    def test_moving_start_pushes_the_end_minimum_forward(self):
        w = wg.create_date_range_picker(start=date(2024, 1, 1), end=date(2024, 6, 1))
        start, end = _range_pickers(w)

        _set(start, date(2024, 3, 1))

        assert end.min == date(2024, 3, 1)

    def test_callback_receives_a_range_dict_not_a_change_event(self):
        seen: list = []
        w = wg.create_date_range_picker(
            start=date(2024, 1, 1), end=date(2024, 6, 1), on_change=seen.append
        )
        start, _ = _range_pickers(w)

        _set(start, date(2024, 2, 1))

        assert seen[-1] == {"start": date(2024, 2, 1), "end": date(2024, 6, 1)}


class TestIndicatorChecklist:
    def test_offers_the_documented_indicator_set(self):
        w = wg.create_indicator_checklist()
        values = {v for _label, v in w.options}

        assert {"sma_20", "sma_50", "sma_200", "ema_12", "rsi", "macd"} <= values

    def test_uses_internal_keys_not_labels(self):
        """Callers switch on 'sma_20', never on 'SMA 20'."""
        w = wg.create_indicator_checklist(value=["sma_20", "rsi"])

        assert list(w.value) == ["sma_20", "rsi"]

    def test_rejects_an_unknown_indicator(self):
        """An indicator key that is not on the list must fail loudly, not silently."""
        with pytest.raises(TraitError):
            wg.create_indicator_checklist(value=["not_an_indicator"])

    def test_fires_the_callback_on_change(self):
        seen: list = []
        w = wg.create_indicator_checklist(value=["sma_20"], on_change=seen.append)

        _set(w, ("sma_20", "rsi"))

        assert list(seen[-1]["new"]) == ["sma_20", "rsi"]


class TestSviSliders:
    def test_uses_sensible_defaults_without_a_curve(self):
        box = wg.create_svi_sliders()
        sliders = _svi_sliders(box)

        assert set(sliders) == {"a", "b", "\u03c1", "m", "\u03c3"}
        assert sliders["b"].value == pytest.approx(0.10)
        assert sliders["\u03c1"].value == pytest.approx(-0.30)

    def test_ranges_admit_the_parameter_domain(self):
        sliders = _svi_sliders(wg.create_svi_sliders())
        rho = sliders["\u03c1"]
        sigma = sliders["\u03c3"]

        # SVI needs rho strictly inside (-1, 1) and sigma strictly positive.
        assert rho.min <= -0.999 and rho.max >= 0.999
        assert rho.min < 0.0 < rho.max
        assert sigma.min > 0.0
        assert sliders["b"].min >= 0.0

    def test_seeds_from_a_supplied_curve(self):
        class _Curve:
            a, b, rho, m, sigma = 0.11, 0.22, -0.44, 0.05, 0.33

        sliders = _svi_sliders(wg.create_svi_sliders(initial=_Curve()))

        assert sliders["a"].value == pytest.approx(0.11)
        assert sliders["b"].value == pytest.approx(0.22)
        assert sliders["\u03c1"].value == pytest.approx(-0.44)
        assert sliders["\u03c3"].value == pytest.approx(0.33)

    def test_callback_receives_every_parameter_not_just_the_one_moved(self):
        seen: list = []
        box = wg.create_svi_sliders(on_change=seen.append)
        sliders = _svi_sliders(box)

        _set(sliders["\u03c1"], -0.5)

        assert set(seen[-1]) == {"a", "b", "rho", "m", "sigma"}
        assert seen[-1]["rho"] == pytest.approx(-0.5)

    def test_greek_letters_survive_in_the_slider_labels(self):
        """The labels name the maths: rho and sigma must not render as '?'."""
        sliders = _svi_sliders(wg.create_svi_sliders())

        assert "\u03c1" in sliders["\u03c1"].description
        assert "\u03c3" in sliders["\u03c3"].description


class TestSviJwSliders:
    def test_covers_the_five_trader_parameters(self):
        sliders = _jw_sliders(wg.create_svi_jw_sliders())

        assert set(sliders) == {
            "atm_variance",
            "atm_skew",
            "put_wing_slope",
            "call_wing_slope",
            "min_variance",
        }

    def test_seeds_from_a_dict(self):
        initial = {
            "atm_variance": 0.04,
            "atm_skew": -0.1,
            "put_wing_slope": 0.3,
            "call_wing_slope": 0.25,
            "min_variance": 0.01,
        }
        sliders = _jw_sliders(wg.create_svi_jw_sliders(initial=initial))

        assert sliders["atm_variance"].value == pytest.approx(0.04)
        assert sliders["put_wing_slope"].value == pytest.approx(0.3)

    def test_min_variance_cannot_be_negative(self):
        sliders = _jw_sliders(wg.create_svi_jw_sliders())

        assert sliders["min_variance"].min >= 0.0

    def test_callback_reports_all_parameters(self):
        seen: list = []
        sliders = _jw_sliders(wg.create_svi_jw_sliders(on_change=seen.append))

        _set(sliders["atm_skew"], -0.2)

        assert len(seen[-1]) == 5
        assert seen[-1]["atm_skew"] == pytest.approx(-0.2)


class TestWeightsToggle:
    def test_offers_vega_and_uniform_and_defaults_to_vega(self):
        w = wg.create_weights_toggle()

        assert isinstance(w, widgets.ToggleButtons)
        assert [value for _label, value in w.options] == ["vega", "uniform"]
        assert w.value == "vega"
        assert {label for label, _value in w.options} == {"Vega-Weighted", "Uniform"}

    def test_fires_the_callback_on_change(self):
        seen: list = []
        w = wg.create_weights_toggle(on_change=seen.append)

        _set(w, "uniform")

        assert seen[-1]["new"] == "uniform"


class TestChartTypeSelector:
    def test_offers_the_supported_chart_types(self):
        w = wg.create_chart_type_selector()
        values = {v for _label, v in w.options}

        assert {
            "candlestick",
            "vol_3d",
            "vol_2d",
            "yield_curve",
            "carry_roll",
            "options_chain",
        } == values

    def test_defaults_to_candlestick(self):
        assert wg.create_chart_type_selector().value == "candlestick"


class TestSurfaceAndCurveControls:
    def test_vol_surface_controls_expose_the_documented_fields(self):
        box = wg.create_vol_surface_controls()
        sliders = _descendants(box, widgets.FloatSlider)
        ranges = _descendants(box, widgets.FloatRangeSlider)
        checkboxes = {c.description: c for c in _descendants(box, widgets.Checkbox)}

        assert any("Expiry" in s.description for s in sliders)
        assert any("Moneyness" in r.description for r in ranges)
        assert {"Show SVI Fit", "Show Contours (3D)"} <= set(checkboxes)
        # The fit-weight toggle is part of the panel, not a separate widget.
        assert _descendants(box, widgets.ToggleButtons)

    def test_vol_surface_fits_are_shown_by_default(self):
        box = wg.create_vol_surface_controls()
        checkboxes = {c.description: c for c in _descendants(box, widgets.Checkbox)}

        assert checkboxes["Show SVI Fit"].value is True
        assert checkboxes["Show Contours (3D)"].value is True

    def test_moneyness_range_spans_the_standard_band(self):
        box = wg.create_vol_surface_controls()
        (moneyness,) = _descendants(box, widgets.FloatRangeSlider)

        assert tuple(moneyness.value) == (0.7, 1.3)
        assert moneyness.min < 0.7 and moneyness.max > 1.3

    def test_yield_curve_controls_expose_type_and_fit_toggle(self):
        box = wg.create_yield_curve_controls()
        dropdowns = {d.description: d for d in _descendants(box, widgets.Dropdown)}
        checkboxes = {c.description: c for c in _descendants(box, widgets.Checkbox)}

        assert "Curve Type:" in dropdowns
        assert {v for _label, v in dropdowns["Curve Type:"].options} == {
            "spot",
            "forward",
            "par",
            "all",
        }
        assert "Show NSS Fit" in checkboxes

    def test_yield_curve_carries_a_horizon_slider(self):
        box = wg.create_yield_curve_controls()
        sliders = _descendants(box, widgets.FloatSlider)

        assert any("Carry Horizon" in s.description for s in sliders)

    def test_control_callbacks_fire(self):
        seen: list = []
        box = wg.create_vol_surface_controls(on_change=seen.append)
        slider = next(
            s for s in _descendants(box, widgets.FloatSlider) if "Expiry" in s.description
        )

        _set(slider, 0.5)

        assert seen


class TestPanelCallbacks:
    """Every panel callback has to report what the user can see on screen."""

    def test_vol_surface_callback_names_every_control(self):
        seen: list = []
        box = wg.create_vol_surface_controls(on_change=seen.append)
        expiry = next(
            s for s in _descendants(box, widgets.FloatSlider) if "Expiry" in s.description
        )

        _set(expiry, 0.75)

        payload = seen[-1]
        assert payload["expiry"] == pytest.approx(0.75)
        assert payload["moneyness_range"] == (0.7, 1.3)
        assert payload["show_svi"] is True
        assert payload["weights"] == "vega"

    def test_vol_surface_callback_reports_a_weight_change(self):
        seen: list = []
        box = wg.create_vol_surface_controls(on_change=seen.append)
        (toggle,) = _descendants(box, widgets.ToggleButtons)

        _set(toggle, "uniform")

        assert seen[-1]["weights"] == "uniform"

    def test_yield_curve_callback_names_every_control(self):
        seen: list = []
        box = wg.create_yield_curve_controls(on_change=seen.append)
        horizon = next(
            s for s in _descendants(box, widgets.FloatSlider) if "Horizon" in s.description
        )

        _set(horizon, 2.5)

        payload = seen[-1]
        assert payload["horizon"] == pytest.approx(2.5)
        assert payload["curve_type"] == "all"
        assert payload["show_nss"] is True

    def test_equity_callback_reports_a_date_range(self):
        """The date range is an HBox, so its start and end arrive separately."""
        seen: list = []
        box = wg.create_equity_dashboard_controls(SYMBOLS, on_change=seen.append)
        # The range picker is a direct child of the panel; nothing nests an
        # HBox inside it, so the first descendant is also the only one.
        date_range = next(c for c in box.children if isinstance(c, widgets.HBox))

        start = next(c for c in date_range.children if isinstance(c, widgets.DatePicker))
        _set(start, date(2024, 5, 1))

        payload = seen[-1]
        assert payload["start"] == date(2024, 5, 1)
        assert payload["end"] == date.today()
        # The panel's own keys come through alongside the flattened date range.
        assert payload["symbol"] == SYMBOLS[0]

    def test_equity_callback_reports_a_symbol_change(self):
        seen: list = []
        box = wg.create_equity_dashboard_controls(SYMBOLS, on_change=seen.append)
        # The symbol picker and the chart-type picker are both dropdowns.
        dropdown = next(
            d for d in _descendants(box, widgets.Dropdown) if d.description == "Symbol:"
        )

        _set(dropdown, "QQQ")

        assert seen[-1]["symbol"] == "QQQ"
        assert seen[-1]["chart_type"] == "candlestick"


class TestDashboardLayout:
    def test_equity_controls_include_a_symbol_dropdown(self):
        box = wg.create_equity_dashboard_controls(SYMBOLS)
        dropdown = _descendants(box, widgets.Dropdown)

        assert dropdown
        assert list(dropdown[0].options) == SYMBOLS

    def test_equity_controls_include_indicators_and_chart_type(self):
        box = wg.create_equity_dashboard_controls(SYMBOLS)

        assert _descendants(box, widgets.SelectMultiple)
        assert _descendants(box, widgets.Dropdown)

    def test_vol_dashboard_controls_are_built(self):
        box = wg.create_vol_dashboard_controls()

        assert isinstance(box, widgets.VBox)
        assert _descendants(box, widgets.FloatSlider)

    def test_vol_dashboard_callback_reports_the_current_settings(self):
        """The callback used to fire with an empty dict, so nothing could redraw."""
        seen: list = []
        box = wg.create_vol_dashboard_controls(on_change=seen.append)
        sliders = _descendants(box, widgets.FloatSlider)

        _set(next(s for s in sliders if s.description.startswith("a ")), 0.09)

        assert seen, "moving a slider must reach the callback"
        payload = seen[-1]
        assert payload, "the callback must not receive an empty dict"
        # The panel's own settings come through under their internal keys.
        assert payload["expiry"] == pytest.approx(0.25)
        assert payload["weights"] == "vega"
        # And so do the SVI sliders, keyed by parameter symbol.
        assert payload["a"] == pytest.approx(0.09)

    def test_vol_dashboard_reports_the_svi_skew(self):
        seen: list = []
        box = wg.create_vol_dashboard_controls(on_change=seen.append)
        slider = next(
            s for s in _descendants(box, widgets.FloatSlider) if s.description.startswith("\u03c1")
        )

        _set(slider, -0.6)

        assert seen[-1]["\u03c1"] == pytest.approx(-0.6)

    def test_vol_dashboard_callback_survives_a_panel_change(self):
        seen: list = []
        box = wg.create_vol_dashboard_controls(on_change=seen.append)
        checkboxes = {c.description: c for c in _descendants(box, widgets.Checkbox)}

        _set(checkboxes["Show SVI Fit"], False)

        assert seen[-1]["show_svi"] is False

    def test_output_area_is_an_output_widget(self):
        assert isinstance(wg.create_output_area(), widgets.Output)

    def test_dashboard_places_the_panels_in_a_sidebar_and_the_output_beside_them(self):
        dashboard = wg.create_chart_dashboard(
            wg.create_equity_dashboard_controls(SYMBOLS),
            wg.create_vol_dashboard_controls(),
            wg.create_yield_curve_controls(),
            wg.create_output_area(),
        )

        assert isinstance(dashboard, widgets.HBox)
        sidebar, main = dashboard.children

        # The three control panels live in the sidebar's own accordion, which is
        # a direct child; the panels contain nested accordions of their own.
        accordion = next(c for c in sidebar.children if isinstance(c, widgets.Accordion))
        assert len(accordion.children) == 3

        # The output area is the main panel, not buried in the sidebar.
        assert _descendants(main, widgets.Output)
        assert not _descendants(sidebar, widgets.Output)

    def test_dashboard_titles_each_accordion_section(self):
        dashboard = wg.create_chart_dashboard(
            wg.create_equity_dashboard_controls(SYMBOLS),
            wg.create_vol_dashboard_controls(),
            wg.create_yield_curve_controls(),
            wg.create_output_area(),
        )

        sidebar = dashboard.children[0]
        accordion = next(c for c in sidebar.children if isinstance(c, widgets.Accordion))
        assert accordion.get_title(0) == "Equity"
        assert accordion.get_title(1) == "Volatility"
        assert accordion.get_title(2) == "Yield Curves"


def _descendants(box, kind: type) -> list:
    """Every descendant of a container, depth-first, of the requested type."""
    found = []
    for child in getattr(box, "children", ()):
        if isinstance(child, kind):
            found.append(child)
        found.extend(_descendants(child, kind))
    return found


def _svi_sliders(box) -> dict:
    """Map the SVI sliders by their parameter letter, whatever script it uses.

    The labels name the maths, so rho and sigma arrive as Greek letters.
    """
    return {s.description[0]: s for s in _descendants(box, widgets.FloatSlider)}


def _jw_sliders(box) -> dict:
    keys = {
        "ATM Var:": "atm_variance",
        "ATM Skew:": "atm_skew",
        "Put Wing:": "put_wing_slope",
        "Call Wing:": "call_wing_slope",
        "Min Var:": "min_variance",
    }
    return {
        keys[s.description]: s
        for s in _descendants(box, widgets.FloatSlider)
        if s.description in keys
    }


def _range_pickers(box) -> tuple:
    pickers = [c for c in box.children if isinstance(c, widgets.DatePicker)]
    assert len(pickers) == 2, "expected a start and an end picker"
    return pickers[0], pickers[1]
