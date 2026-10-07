"""
QuantView Notebook Package.

Interactive Plotly charts and widgets for Jupyter notebooks.
"""

from .charts import (
    THEME,
    ChartTheme,
    NSSFit,
    YieldCurveData,
    candlestick_chart,
    options_chain_greeks,
    vol_surface_2d,
    vol_surface_3d,
    yield_curve_carry_roll,
    yield_curve_chart,
)
from .export import (
    export_dashboard,
    export_html,
    export_image,
    export_json,
    export_notebook_report,
    optimize_figure_size,
    quick_export,
)
from .widgets import (
    create_chart_type_selector,
    create_date_picker,
    create_date_range_picker,
    create_equity_dashboard_controls,
    create_indicator_checklist,
    create_output_area,
    create_svi_jw_sliders,
    create_svi_sliders,
    create_symbol_dropdown,
    create_vol_dashboard_controls,
    create_vol_surface_controls,
    create_weights_toggle,
    create_yield_curve_controls,
)

__all__ = [
    # Charts
    "THEME",
    "ChartTheme",
    "NSSFit",
    "YieldCurveData",
    "candlestick_chart",
    "create_chart_dashboard",
    "create_chart_type_selector",
    "create_date_picker",
    "create_date_range_picker",
    "create_equity_dashboard_controls",
    "create_indicator_checklist",
    "create_output_area",
    "create_svi_jw_sliders",
    "create_svi_sliders",
    # Widgets
    "create_symbol_dropdown",
    "create_vol_dashboard_controls",
    "create_vol_surface_controls",
    "create_weights_toggle",
    "create_yield_curve_controls",
    # Export
    "export_dashboard",
    "export_html",
    "export_image",
    "export_json",
    "export_notebook_report",
    "optimize_figure_size",
    "options_chain_greeks",
    "quick_export",
    "vol_surface_2d",
    "vol_surface_3d",
    "yield_curve_carry_roll",
    "yield_curve_chart",
]

__version__ = "0.1.0"
