"""
QuantView Notebook Widgets Module.

Interactive ipywidgets for chart controls: symbol dropdown,
date picker, SVI parameter sliders, weights toggle, etc.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import ipywidgets as widgets
from IPython.display import display

# Explicitly bind to None so the name exists in the module namespace when
# the optional dependency is absent. The try/except rebinds on success.
SVICurve: Any = None
try:
    from black_scholes.surface import SVICurve
except ImportError:
    pass


@dataclass
class WidgetConfig:
    """Configuration for widget layout and styling."""

    layout_width: str = "300px"
    continuous_update: bool = False
    style: dict = field(
        default_factory=lambda: {
            "description_width": "120px",
        }
    )


CONFIG = WidgetConfig()


def _panel(box: widgets.VBox, controls: dict) -> widgets.VBox:
    """Tag a control panel with the key names it reports to its callback.

    The builders key their callbacks by internal names ("expiry", "symbol")
    rather than by the labels on screen, which are presentation. A panel that
    embeds another panel needs those names, and they would otherwise be out of
    scope, so they are attached to the returned widget.
    """
    box._quantview_controls = controls
    return box


def control_values(container: widgets.Widget) -> dict:
    """Read a panel's current values under the keys its callback uses.

    Walks nested panels so a composite control reports the same flat mapping a
    caller would have received from the leaf panel directly.
    """
    controls = getattr(container, "_quantview_controls", None)
    if controls is None:
        return {}

    values: dict = {}
    for key, widget in controls.items():
        if hasattr(widget, "value"):
            values[key] = widget.value
        elif hasattr(widget, "children"):  # the date-range HBox
            pickers = [c for c in widget.children if isinstance(c, widgets.DatePicker)]
            if len(pickers) == 2:
                values["start"] = pickers[0].value
                values["end"] = pickers[1].value
        values.update(control_values(widget))
    return values


def create_symbol_dropdown(
    symbols: list[str],
    value: str | None = None,
    description: str = "Symbol:",
    on_change: Callable | None = None,
) -> widgets.Dropdown:
    """
    Create a symbol selection dropdown.

    Args:
        symbols: List of ticker symbols.
        value: Initial selected symbol.
        description: Widget label.
        on_change: Callback function(change) on value change.

    Returns:
        ipywidgets.Dropdown instance.
    """
    w = widgets.Dropdown(
        options=symbols,
        value=value or (symbols[0] if symbols else None),
        description=description,
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
    )
    if on_change:
        w.observe(on_change, names="value")
    return w


def create_date_picker(
    value: date | None = None,
    description: str = "Date:",
    min_date: date | None = None,
    max_date: date | None = None,
    on_change: Callable | None = None,
) -> widgets.DatePicker:
    """
    Create a date picker widget.

    Args:
        value: Initial date.
        description: Widget label.
        min_date: Minimum selectable date.
        max_date: Maximum selectable date.
        on_change: Callback function(change) on value change.

    Returns:
        ipywidgets.DatePicker instance.
    """
    today = date.today()
    w = widgets.DatePicker(
        value=value or today,
        description=description,
        disabled=False,
        min=min_date or date(2000, 1, 1),
        max=max_date or today,
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
    )
    if on_change:
        w.observe(on_change, names="value")
    return w


def create_date_range_picker(
    start: date | None = None,
    end: date | None = None,
    description: str = "Date Range:",
    on_change: Callable | None = None,
) -> widgets.HBox:
    """
    Create a start/end date range picker.

    Args:
        start: Initial start date.
        end: Initial end date.
        description: Label for the group.
        on_change: Callback function(change) on either value change.

    Returns:
        ipywidgets.HBox containing two DatePickers.
    """
    today = date.today()
    default_start = start or (today - timedelta(days=365))
    default_end = end or today

    start_picker = widgets.DatePicker(
        value=default_start,
        description="Start:",
        min=date(2000, 1, 1),
        max=default_end,
        layout=widgets.Layout(width="160px"),
        style=CONFIG.style,
    )
    end_picker = widgets.DatePicker(
        value=default_end,
        description="End:",
        min=default_start,
        max=today,
        layout=widgets.Layout(width="160px"),
        style=CONFIG.style,
    )

    def _update_bounds(change: dict) -> None:
        if change["owner"] is start_picker:
            end_picker.min = change["new"]
        elif change["owner"] is end_picker:
            start_picker.max = change["new"]
        if on_change:
            on_change({"start": start_picker.value, "end": end_picker.value})

    start_picker.observe(_update_bounds, names="value")
    end_picker.observe(_update_bounds, names="value")

    label = widgets.Label(value=description, layout=widgets.Layout(width="100px"))
    return widgets.HBox([label, start_picker, end_picker])


def create_indicator_checklist(
    value: list[str] | None = None,
    description: str = "Indicators:",
    on_change: Callable | None = None,
) -> widgets.SelectMultiple:
    """
    Create a multi-select for technical indicators.

    Args:
        value: Initially selected indicators.
        description: Widget label.
        on_change: Callback function(change) on value change.

    Returns:
        ipywidgets.SelectMultiple instance.
    """
    options = [
        ("SMA 20", "sma_20"),
        ("SMA 50", "sma_50"),
        ("SMA 200", "sma_200"),
        ("EMA 12", "ema_12"),
        ("EMA 26", "ema_26"),
        ("EMA 50", "ema_50"),
        ("Bollinger Bands", "bbands"),
        ("RSI (14)", "rsi"),
        ("MACD", "macd"),
    ]

    w = widgets.SelectMultiple(
        options=options,
        value=value or ["sma_20", "sma_50", "bbands"],
        description=description,
        layout=widgets.Layout(width=CONFIG.layout_width, height="200px"),
        style=CONFIG.style,
    )
    if on_change:
        w.observe(on_change, names="value")
    return w


# ============================================================================
# SVI Parameter Sliders
# ============================================================================


def create_svi_sliders(
    initial: SVICurve | None = None,
    on_change: Callable | None = None,
) -> widgets.VBox:
    """
    Create sliders for SVI raw parameters (a, b, rho, m, sigma).

    Args:
        initial: Initial SVICurve for default values.
        on_change: Callback function(params_dict) on any slider change.

    Returns:
        ipywidgets.VBox containing all sliders.
    """
    defaults = {
        "a": 0.04,
        "b": 0.10,
        "rho": -0.30,
        "m": 0.0,
        "sigma": 0.15,
    }
    if initial is not None:
        defaults = {
            "a": initial.a,
            "b": initial.b,
            "rho": initial.rho,
            "m": initial.m,
            "sigma": initial.sigma,
        }

    sliders = {}

    # a: level of total variance
    sliders["a"] = widgets.FloatSlider(
        value=defaults["a"],
        min=-0.5,
        max=2.0,
        step=0.001,
        description="a (level):",
        continuous_update=CONFIG.continuous_update,
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
        readout_format=".4f",
    )

    # b: slope/smile
    sliders["b"] = widgets.FloatSlider(
        value=defaults["b"],
        min=0.0,
        max=2.0,
        step=0.001,
        description="b (slope):",
        continuous_update=CONFIG.continuous_update,
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
        readout_format=".4f",
    )

    # rho: skew correlation
    sliders["rho"] = widgets.FloatSlider(
        value=defaults["rho"],
        min=-0.999,
        max=0.999,
        step=0.001,
        description="ρ (skew):",
        continuous_update=CONFIG.continuous_update,
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
        readout_format=".4f",
    )

    # m: horizontal shift
    sliders["m"] = widgets.FloatSlider(
        value=defaults["m"],
        min=-1.0,
        max=1.0,
        step=0.001,
        description="m (shift):",
        continuous_update=CONFIG.continuous_update,
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
        readout_format=".4f",
    )

    # sigma: wing curvature
    sliders["sigma"] = widgets.FloatSlider(
        value=defaults["sigma"],
        min=0.001,
        max=1.0,
        step=0.001,
        description="σ (wing):",
        continuous_update=CONFIG.continuous_update,
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
        readout_format=".4f",
    )

    def _on_change(change: dict) -> None:
        if on_change:
            params = {k: v.value for k, v in sliders.items()}
            on_change(params)

    for s in sliders.values():
        s.observe(_on_change, names="value")

    title = widgets.HTML(
        value="<b>SVI Raw Parameters</b><br><i>w(k) = a + b[ρ(k-m) + √((k-m)²+σ²)]</i>",
        layout=widgets.Layout(margin="0 0 10px 0"),
    )

    return widgets.VBox([title, *list(sliders.values())])


def create_svi_jw_sliders(
    initial: dict | None = None,
    on_change: Callable | None = None,
) -> widgets.VBox:
    """
    Create sliders for SVI-Jump-Wings parameters (trader-friendly).

    Args:
        initial: Dict with keys atm_variance, atm_skew, put_wing_slope,
                 call_wing_slope, min_variance.
        on_change: Callback function(params_dict) on any slider change.

    Returns:
        ipywidgets.VBox containing all sliders.
    """
    defaults = {
        "atm_variance": 0.055,
        "atm_skew": -0.06,
        "put_wing_slope": 0.15,
        "call_wing_slope": 0.05,
        "min_variance": 0.05,
    }
    if initial is not None:
        defaults.update(initial)

    sliders = {}

    sliders["atm_variance"] = widgets.FloatSlider(
        value=defaults["atm_variance"],
        min=0.001,
        max=1.0,
        step=0.001,
        description="ATM Var:",
        continuous_update=CONFIG.continuous_update,
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
        readout_format=".4f",
    )

    sliders["atm_skew"] = widgets.FloatSlider(
        value=defaults["atm_skew"],
        min=-1.0,
        max=1.0,
        step=0.001,
        description="ATM Skew:",
        continuous_update=CONFIG.continuous_update,
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
        readout_format=".4f",
    )

    sliders["put_wing_slope"] = widgets.FloatSlider(
        value=defaults["put_wing_slope"],
        min=0.0,
        max=2.0,
        step=0.001,
        description="Put Wing:",
        continuous_update=CONFIG.continuous_update,
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
        readout_format=".4f",
    )

    sliders["call_wing_slope"] = widgets.FloatSlider(
        value=defaults["call_wing_slope"],
        min=0.0,
        max=2.0,
        step=0.001,
        description="Call Wing:",
        continuous_update=CONFIG.continuous_update,
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
        readout_format=".4f",
    )

    sliders["min_variance"] = widgets.FloatSlider(
        value=defaults["min_variance"],
        min=0.0,
        max=1.0,
        step=0.001,
        description="Min Var:",
        continuous_update=CONFIG.continuous_update,
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
        readout_format=".4f",
    )

    def _on_change(change: dict) -> None:
        if on_change:
            params = {k: v.value for k, v in sliders.items()}
            on_change(params)

    for s in sliders.values():
        s.observe(_on_change, names="value")

    title = widgets.HTML(
        value="<b>SVI Jump-Wings Parameters</b><br><i>Trader-readable parameterization</i>",
        layout=widgets.Layout(margin="0 0 10px 0"),
    )

    return widgets.VBox([title, *list(sliders.values())])


# ============================================================================
# Weights Toggle
# ============================================================================


def create_weights_toggle(
    value: str = "vega",
    description: str = "Fit Weights:",
    on_change: Callable | None = None,
) -> widgets.ToggleButtons:
    """
    Create a toggle between uniform and vega weighting for SVI fit.

    Args:
        value: Initial value ('uniform' or 'vega').
        description: Widget label.
        on_change: Callback function(change) on value change.

    Returns:
        ipywidgets.ToggleButtons instance.
    """
    w = widgets.ToggleButtons(
        options=[
            ("Vega-Weighted", "vega"),
            ("Uniform", "uniform"),
        ],
        value=value,
        description=description,
        button_style="",
        tooltips=[
            "Weight by Black-Scholes vega (ATM strikes dominate)",
            "Equal weight for all strikes",
        ],
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
    )
    if on_change:
        w.observe(on_change, names="value")
    return w


# ============================================================================
# Chart Type Selectors
# ============================================================================


def create_chart_type_selector(
    value: str = "candlestick",
    description: str = "Chart Type:",
    on_change: Callable | None = None,
) -> widgets.Dropdown:
    """
    Create a chart type selector dropdown.

    Args:
        value: Initial chart type.
        description: Widget label.
        on_change: Callback function(change) on value change.

    Returns:
        ipywidgets.Dropdown instance.
    """
    options = [
        ("Candlestick + Indicators", "candlestick"),
        ("Vol Surface 3D", "vol_3d"),
        ("Vol Smile 2D", "vol_2d"),
        ("Yield Curve", "yield_curve"),
        ("Carry & Roll", "carry_roll"),
        ("Options Chain Greeks", "options_chain"),
    ]

    w = widgets.Dropdown(
        options=options,
        value=value,
        description=description,
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
    )
    if on_change:
        w.observe(on_change, names="value")
    return w


def create_vol_surface_controls(
    on_change: Callable | None = None,
) -> widgets.VBox:
    """
    Create a complete control panel for volatility surface charts.

    Args:
        on_change: Callback function(control_dict) on any control change.

    Returns:
        ipywidgets.VBox with all vol surface controls.
    """
    controls = {}

    controls["expiry"] = widgets.FloatSlider(
        value=0.25,
        min=0.01,
        max=2.0,
        step=0.01,
        description="Expiry (Y):",
        continuous_update=False,
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
        readout_format=".2f",
    )

    controls["moneyness_range"] = widgets.FloatRangeSlider(
        value=[0.7, 1.3],
        min=0.5,
        max=2.0,
        step=0.01,
        description="Moneyness:",
        continuous_update=False,
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
        readout_format=".2f",
    )

    controls["show_svi"] = widgets.Checkbox(
        value=True,
        description="Show SVI Fit",
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
    )

    controls["show_contours"] = widgets.Checkbox(
        value=True,
        description="Show Contours (3D)",
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
    )

    controls["weights"] = create_weights_toggle()

    title = widgets.HTML(
        value="<b>Vol Surface Controls</b>",
        layout=widgets.Layout(margin="0 0 10px 0"),
    )
    box = _panel(widgets.VBox([title, *list(controls.values())]), controls)

    def _on_change(change: dict) -> None:
        if on_change:
            on_change(control_values(box))

    for c in controls.values():
        if hasattr(c, "observe"):
            c.observe(_on_change, names="value")

    return box


def create_yield_curve_controls(
    on_change: Callable | None = None,
) -> widgets.VBox:
    """
    Create a complete control panel for yield curve charts.

    Args:
        on_change: Callback function(control_dict) on any control change.

    Returns:
        ipywidgets.VBox with all yield curve controls.
    """
    controls = {}

    controls["curve_type"] = widgets.Dropdown(
        options=[
            ("Spot", "spot"),
            ("Forward", "forward"),
            ("Par", "par"),
            ("All Three", "all"),
        ],
        value="all",
        description="Curve Type:",
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
    )

    controls["show_nss"] = widgets.Checkbox(
        value=True,
        description="Show NSS Fit",
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
    )

    controls["horizon"] = widgets.FloatSlider(
        value=1.0,
        min=0.1,
        max=5.0,
        step=0.1,
        description="Carry Horizon (Y):",
        continuous_update=False,
        layout=widgets.Layout(width=CONFIG.layout_width),
        style=CONFIG.style,
        readout_format=".1f",
    )

    title = widgets.HTML(
        value="<b>Yield Curve Controls</b>",
        layout=widgets.Layout(margin="0 0 10px 0"),
    )
    box = _panel(widgets.VBox([title, *list(controls.values())]), controls)

    def _on_change(change: dict) -> None:
        if on_change:
            on_change(control_values(box))

    for c in controls.values():
        if hasattr(c, "observe"):
            c.observe(_on_change, names="value")

    return box


# ============================================================================
# Complete Dashboard Builders
# ============================================================================


def create_equity_dashboard_controls(
    symbols: list[str],
    on_change: Callable | None = None,
) -> widgets.VBox:
    """
    Create a complete control panel for equity charts.

    Args:
        symbols: List of available symbols.
        on_change: Callback function(control_dict) on any control change.

    Returns:
        ipywidgets.VBox with all equity controls.
    """
    controls = {}

    controls["symbol"] = create_symbol_dropdown(symbols)
    controls["date_range"] = create_date_range_picker()
    controls["indicators"] = create_indicator_checklist()
    controls["chart_type"] = create_chart_type_selector(value="candlestick")

    title = widgets.HTML(
        value="<b>Equity Chart Controls</b>",
        layout=widgets.Layout(margin="0 0 10px 0"),
    )
    box = _panel(widgets.VBox([title, *list(controls.values())]), controls)

    def _on_change(change: dict) -> None:
        if on_change:
            on_change(control_values(box))

    for c in controls.values():
        if hasattr(c, "value"):
            c.observe(_on_change, names="value")
        elif hasattr(c, "children"):
            # Test for .value, not .observe: every ipywidget exposes observe,
            # containers included, so the previous check sent the date-range HBox
            # down the first branch and left its pickers unwired. Changing the
            # start date then never reached the caller.
            for child in c.children:
                if hasattr(child, "value"):
                    child.observe(_on_change, names="value")

    return box


def create_vol_dashboard_controls(
    on_change: Callable | None = None,
) -> widgets.VBox:
    """
    Create a complete control panel for volatility charts.

    Args:
        on_change: Callback function(control_dict) on any control change.

    Returns:
        ipywidgets.VBox with all vol controls including SVI sliders.
    """
    # Main controls
    main_controls = create_vol_surface_controls()

    # SVI parameter sliders (collapsible)
    svi_sliders = create_svi_sliders()

    accordion = widgets.Accordion(children=[svi_sliders])
    accordion.set_title(0, "SVI Raw Parameters")

    controls = {
        "main": main_controls,
        "svi": accordion,
    }

    def _on_change(change: dict) -> None:
        if not on_change:
            return
        # Report every setting currently on screen. The previous version looped
        # over the children and dropped them on the floor, so a caller's callback
        # always received an empty dict and could never redraw anything.
        params = dict(control_values(main_controls))
        params.update(_svi_values(svi_sliders))
        on_change(params)

    # Wire both halves: the panel's own controls and the SVI sliders.
    for child in main_controls.children:
        if hasattr(child, "observe"):
            child.observe(_on_change, names="value")
    for child in svi_sliders.children:
        if hasattr(child, "observe"):
            child.observe(_on_change, names="value")

    return _panel(widgets.VBox(list(controls.values())), controls)


def _svi_values(box: widgets.Widget) -> dict:
    """Map the SVI sliders back to their parameter names.

    The SVI builders report a, b, rho, m, sigma by name; the sliders on screen are
    labelled in the same symbols, so the leading character is the key.
    """
    values = {}
    for child in getattr(box, "children", ()):
        description = getattr(child, "description", "")
        if isinstance(child, widgets.FloatSlider) and description:
            values[description[0]] = child.value
    return values


# ============================================================================
# Output/Display Helpers
# ============================================================================


def create_output_area() -> widgets.Output:
    """Create an output widget for displaying charts."""
    return widgets.Output(
        layout=widgets.Layout(
            width="100%",
            min_height="500px",
            border="1px solid #30363d",
            border_radius="4px",
        )
    )


def create_chart_dashboard(
    equity_controls: widgets.VBox,
    vol_controls: widgets.VBox,
    yield_controls: widgets.VBox,
    output: widgets.Output,
) -> widgets.HBox:
    """
    Assemble a complete chart dashboard layout.

    Args:
        equity_controls: Equity control panel.
        vol_controls: Volatility control panel.
        yield_controls: Yield curve control panel.
        output: Output widget for chart display.

    Returns:
        ipywidgets.HBox with sidebar and main area.
    """
    sidebar = widgets.VBox(
        [
            widgets.HTML(value="<h3>QuantView Charts</h3>"),
            widgets.Accordion(children=[equity_controls, vol_controls, yield_controls]),
        ],
        layout=widgets.Layout(width="350px", padding="10px", border_right="1px solid #30363d"),
    )

    # Set accordion titles
    accordion = sidebar.children[1]
    accordion.set_title(0, "Equity")
    accordion.set_title(1, "Volatility")
    accordion.set_title(2, "Yield Curves")

    main = widgets.VBox([output], layout=widgets.Layout(flex="1", padding="10px"))

    return widgets.HBox([sidebar, main], layout=widgets.Layout(width="100%", height="800px"))


# ============================================================================
# Demo / Test
# ============================================================================


def demo_widgets() -> None:
    """Display all widgets for testing."""
    print("=== Symbol Dropdown ===")
    display(create_symbol_dropdown(["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]))

    print("\n=== Date Range Picker ===")
    display(create_date_range_picker())

    print("\n=== Indicator Checklist ===")
    display(create_indicator_checklist())

    print("\n=== SVI Raw Sliders ===")
    display(create_svi_sliders())

    print("\n=== SVI Jump-Wings Sliders ===")
    display(create_svi_jw_sliders())

    print("\n=== Weights Toggle ===")
    display(create_weights_toggle())

    print("\n=== Vol Surface Controls ===")
    display(create_vol_surface_controls())

    print("\n=== Yield Curve Controls ===")
    display(create_yield_curve_controls())

    print("\n=== Equity Dashboard Controls ===")
    display(create_equity_dashboard_controls(["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]))


if __name__ == "__main__":
    demo_widgets()
