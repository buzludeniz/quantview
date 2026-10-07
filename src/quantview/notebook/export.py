"""
QuantView Notebook Export Module.

Utilities for exporting Plotly figures to HTML, PNG, SVG, PDF
with size optimization and batch export capabilities.
"""

from __future__ import annotations

import base64
import gzip
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from .charts import ChartTheme

THEME = ChartTheme()


# Configure kaleido for static export
try:
    import kaleido  # noqa: F401 - probed for availability, never referenced

    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False


def export_html(
    fig: go.Figure,
    path: str | Path,
    *,
    include_plotlyjs: str | bool = "cdn",
    full_html: bool = True,
    auto_play: bool = False,
    config: dict | None = None,
    compress: bool = False,
) -> None:
    """
    Export figure to standalone HTML file.

    Args:
        fig: Plotly figure.
        path: Output file path.
        include_plotlyjs: 'cdn', 'directory', True, False, or path to plotly.js.
        full_html: Include full HTML wrapper (head, body).
        auto_play: Auto-play animations on load.
        config: Plotly config dict for mode bar, toImage, etc.
        compress: Gzip compress the HTML (saves ~70% size).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    default_config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "lasso2d",
            "select2d",
            "autoScale2d",
            "toggleSpikelines",
            "hoverCompareCartesian",
        ],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "quantview_chart",
            "height": 800,
            "width": 1200,
            "scale": 2,
        },
        "responsive": True,
    }
    if config:
        default_config.update(config)

    html_str = pio.to_html(
        fig,
        include_plotlyjs=include_plotlyjs,
        full_html=full_html,
        auto_play=auto_play,
        config=default_config,
    )

    if compress:
        html_bytes = html_str.encode("utf-8")
        compressed = gzip.compress(html_bytes)
        # Write as .html.gz with a small decompressor script
        wrapper = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>QuantView Chart</title></head>
<body><div id="chart"></div>
<script>
const compressed = "{base64.b64encode(compressed).decode()}";
const binary = atob(compressed);
const bytes = new Uint8Array(binary.length);
for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
const decompressed = new DecompressionStream("gzip");
const blob = new Blob([bytes], {{type: "application/gzip"}});
blob.stream().pipeThrough(decompressed).pipeTo(new WritableStream({{
    write(chunk) {{
        document.getElementById("chart").innerHTML += new TextDecoder().decode(chunk);
    }}
}}));
</script></body></html>"""
        path.write_text(wrapper)
    else:
        path.write_text(html_str)


def export_image(
    fig: go.Figure,
    path: str | Path,
    *,
    format: Literal["png", "svg", "pdf", "jpg", "webp"] = "png",
    scale: float = 2.0,
    width: int | None = None,
    height: int | None = None,
    engine: str = "kaleido",
) -> None:
    """
    Export figure to static image file.

    Args:
        fig: Plotly figure.
        path: Output file path.
        format: Image format (requires kaleido for png/jpg/webp/pdf).
        scale: Resolution scale factor (for raster formats).
        width: Output width in pixels (overrides figure layout).
        height: Output height in pixels (overrides figure layout).
        engine: 'kaleido' or 'orca' (orca deprecated).
    """
    if not KALEIDO_AVAILABLE and engine == "kaleido":
        raise RuntimeError("kaleido not installed. Run: pip install kaleido")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pio.write_image(
        fig,
        path,
        format=format,
        scale=scale,
        width=width,
        height=height,
        engine=engine,
    )


def export_json(fig: go.Figure, path: str | Path, *, pretty: bool = True) -> None:
    """
    Export figure to Plotly JSON format.

    Args:
        fig: Plotly figure.
        path: Output file path.
        pretty: Pretty-print JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig_json = pio.to_json(fig, pretty=pretty)
    path.write_text(fig_json)


def export_dashboard(
    figures: dict[str, go.Figure],
    output_dir: str | Path,
    *,
    prefix: str = "chart",
    formats: Sequence[str] = ("html", "png"),
    create_index: bool = True,
) -> None:
    """
    Batch export multiple figures to a directory.

    Args:
        figures: Dict of {name: figure}.
        output_dir: Output directory.
        prefix: Filename prefix.
        formats: List of formats ('html', 'png', 'svg', 'pdf', 'json').
        create_index: Create index.html with all charts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in figures.items():
        safe_name = name.replace(" ", "_").replace("/", "_")
        base = f"{prefix}_{safe_name}"

        if "html" in formats:
            export_html(fig, output_dir / f"{base}.html")
        if "png" in formats:
            export_image(fig, output_dir / f"{base}.png", format="png")
        if "svg" in formats:
            export_image(fig, output_dir / f"{base}.svg", format="svg")
        if "pdf" in formats:
            export_image(fig, output_dir / f"{base}.pdf", format="pdf")
        if "json" in formats:
            export_json(fig, output_dir / f"{base}.json")

    if create_index:
        _create_index_html(figures, output_dir, prefix)


def _create_index_html(
    figures: dict[str, go.Figure],
    output_dir: Path,
    prefix: str,
) -> None:
    """Create an index.html with embedded iframes for all charts."""
    cards = []
    for name in figures:
        safe_name = name.replace(" ", "_").replace("/", "_")
        cards.append(f"""
        <div class="card">
            <h3>{name}</h3>
            <iframe src="{prefix}_{safe_name}.html" frameborder="0"></iframe>
        </div>""")

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantView Dashboard</title>
    <style>
        body {{
            font-family: {THEME.font_family};
            background: {THEME.paper_bgcolor};
            color: #e6edf3;
            margin: 0;
            padding: 20px;
        }}
        h1 {{ color: #f0f6fc; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .card {{
            background: {THEME.plot_bgcolor};
            border: 1px solid {THEME.grid_color};
            border-radius: 8px;
            overflow: hidden;
        }}
        .card h3 {{
            margin: 0;
            padding: 12px 16px;
            background: rgba(88, 166, 255, 0.1);
            border-bottom: 1px solid {THEME.grid_color};
            font-size: 14px;
        }}
        .card iframe {{
            width: 100%;
            height: 500px;
            border: none;
            background: {THEME.plot_bgcolor};
        }}
        @media (max-width: 800px) {{
            .grid {{ grid-template-columns: 1fr; }}
            .card iframe {{ height: 400px; }}
        }}
    </style>
</head>
<body>
    <h1 style="text-align:center; max-width:1400px; margin:0 auto 30px;">QuantView Chart Dashboard</h1>
    <div class="grid">
        {"".join(cards)}
    </div>
</body>
</html>"""

    (output_dir / "index.html").write_text(index_html)


def optimize_figure_size(fig: go.Figure, *, max_points: int = 5000) -> go.Figure:
    """
    Optimize figure for smaller file size by reducing trace point counts.

    Args:
        fig: Plotly figure.
        max_points: Maximum points per trace.

    Returns:
        Optimized figure copy.
    """
    fig = go.Figure(fig)

    for trace in fig.data:
        x = getattr(trace, "x", None)
        if x is None or not hasattr(x, "__len__") or len(x) <= max_points:
            continue

        # Only thin traces whose x axis is a continuous quantity. Striding a
        # categorical axis drops whole categories: a three-bar chart with
        # max_points=1 lost two of its bars, which is data loss presented as an
        # optimisation. Date and number axes stay in order under a stride, so
        # they remain safe to sample.
        if not _is_continuous_axis(x):
            continue

        stride = len(x) // max_points + 1
        for attr in ("x", "y", "z", "text"):
            val = getattr(trace, attr, None)
            if val is None or not hasattr(val, "__len__"):
                continue
            try:
                if len(val) == len(x):
                    setattr(trace, attr, val[::stride])
            except (TypeError, ValueError):
                pass

    return fig


def _is_continuous_axis(values: Sequence) -> bool:
    """True when an x series is numeric or dates rather than category labels."""
    if isinstance(values, str):
        return False
    if np.issubdtype(np.asarray(values).dtype, np.number):
        return True
    return hasattr(values, "__iter__") and not all(isinstance(v, str) for v in values)


def estimate_html_size(fig: go.Figure) -> int:
    """Estimate HTML file size in bytes."""
    html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
    return len(html.encode("utf-8"))


def export_notebook_report(
    figures: dict[str, go.Figure],
    output_path: str | Path,
    *,
    title: str = "QuantView Notebook Report",
    description: str = "",
    include_toc: bool = True,
) -> None:
    """
    Export a complete notebook-style HTML report with all figures.

    Args:
        figures: Dict of {section_name: figure or list of figures}.
        output_path: Output HTML file path.
        title: Report title.
        description: Report description.
        include_toc: Include table of contents.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections_html = []
    toc_items = []

    for section_name, fig_or_list in figures.items():
        safe_id = section_name.lower().replace(" ", "-")
        toc_items.append(f'<li><a href="#{safe_id}">{section_name}</a></li>')

        figs = fig_or_list if isinstance(fig_or_list, list) else [fig_or_list]

        fig_htmls = []
        for i, fig in enumerate(figs):
            fig_id = f"{safe_id}-{i}"
            html = pio.to_html(
                fig,
                include_plotlyjs="cdn" if i == 0 else False,
                full_html=False,
                config={"displayModeBar": True, "displaylogo": False},
            )
            fig_htmls.append(f'<div id="{fig_id}" class="figure">{html}</div>')

        sections_html.append(f"""
        <section id="{safe_id}">
            <h2>{section_name}</h2>
            {"".join(fig_htmls)}
        </section>""")

    toc_html = f"<nav><h3>Contents</h3><ul>{''.join(toc_items)}</ul></nav>" if include_toc else ""

    report = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: {THEME.font_family};
            background: {THEME.paper_bgcolor};
            color: #e6edf3;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        h1 {{ color: #f0f6fc; border-bottom: 2px solid {THEME.primary_color}; padding-bottom: 10px; }}
        h2 {{ color: #f0f6fc; border-bottom: 1px solid {THEME.grid_color}; padding-bottom: 5px; margin-top: 40px; }}
        h3 {{ color: {THEME.primary_color}; }}
        nav {{ background: {THEME.plot_bgcolor}; border: 1px solid {THEME.grid_color}; border-radius: 8px; padding: 20px; margin-bottom: 30px; }}
        nav ul {{ list-style: none; padding-left: 0; }}
        nav li {{ margin: 8px 0; }}
        nav a {{ color: {THEME.primary_color}; text-decoration: none; }}
        nav a:hover {{ text-decoration: underline; }}
        .figure {{ margin: 20px 0; }}
        .description {{ color: #8b949e; font-style: italic; margin-bottom: 30px; }}
        hr {{ border-color: {THEME.grid_color}; margin: 40px 0; }}
        .footer {{ text-align: center; color: #8b949e; font-size: 12px; margin-top: 60px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {f'<p class="description">{description}</p>' if description else ""}
    {toc_html}
    {"".join(sections_html)}
    <hr>
    <div class="footer">Generated by QuantView Notebook Export</div>
</body>
</html>"""

    output_path.write_text(report)


# Convenience function for quick exports
def quick_export(fig: go.Figure, name: str, output_dir: str | Path = "exports") -> dict[str, Path]:
    """
    Quick export to multiple formats.

    Args:
        fig: Plotly figure.
        name: Base filename.
        output_dir: Output directory.

    Returns:
        Dict of format -> Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    safe_name = name.replace(" ", "_").replace("/", "_")

    html_path = output_dir / f"{safe_name}.html"
    export_html(fig, html_path)
    results["html"] = html_path

    if KALEIDO_AVAILABLE:
        png_path = output_dir / f"{safe_name}.png"
        export_image(fig, png_path, format="png", scale=2)
        results["png"] = png_path

        svg_path = output_dir / f"{safe_name}.svg"
        export_image(fig, svg_path, format="svg")
        results["svg"] = svg_path

    return results


if __name__ == "__main__":
    # Test exports

    from .charts import _demo_candlestick

    fig = _demo_candlestick()

    print("Testing exports...")
    results = quick_export(fig, "test_chart", "test_exports")
    for fmt, path in results.items():
        size = path.stat().st_size
        print(f"  {fmt}: {path} ({size:,} bytes)")

    print("Export test complete!")
