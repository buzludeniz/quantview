"""Tests for the notebook export module.

export.py held a 41% coverage share of the package while being on the path of
every "share this chart" action, so its failure modes are worth pinning: the
on-disk layout it produces, and the guards it is supposed to raise on.
"""

from __future__ import annotations

import base64
import gzip
import json
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from quantview.notebook import export as ex


@pytest.fixture
def fig() -> go.Figure:
    """A small line figure with named axes, so traces carry x and y."""
    return go.Figure(
        data=[go.Scatter(x=[1, 2, 3, 4], y=[10, 20, 30, 40], name="series")],
        layout=go.Layout(title="Test Figure", width=640, height=480),
    )


class TestExportHtml:
    def test_writes_a_file_that_contains_the_figure(self, fig, tmp_path: Path):
        out = tmp_path / "chart.html"
        ex.export_html(fig, out)

        assert out.exists()
        body = out.read_text(encoding="utf-8")
        assert "Test Figure" in body
        assert "<html" in body.lower()

    def test_creates_missing_parent_directories(self, fig, tmp_path: Path):
        # The notebook usage writes into a folder that does not exist yet.
        out = tmp_path / "nested" / "deeper" / "chart.html"
        ex.export_html(fig, out)

        assert out.exists()

    def test_compress_embeds_gzip_that_decodes_to_the_figure(self, fig, tmp_path: Path):
        """compress=True wraps base64 gzip inside a page the browser inflates.

        The output is still HTML, not a .gz file, so the payload is checked by
        extracting it and decompressing rather than by looking at magic bytes.
        """
        out = tmp_path / "chart.html"
        ex.export_html(fig, out, compress=True)

        body = out.read_text(encoding="utf-8")
        payload = re.search(r'const compressed = "([A-Za-z0-9+/=]+)"', body)
        assert payload, "no embedded payload found in the compressed output"
        assert b"Test Figure" in gzip.decompress(base64.b64decode(payload.group(1)))

    def test_compress_is_smaller_than_plain(self, fig, tmp_path: Path):
        plain = tmp_path / "plain.html"
        packed = tmp_path / "packed.html"
        ex.export_html(fig, plain)
        ex.export_html(fig, packed, compress=True)

        assert packed.stat().st_size < plain.stat().st_size

    def test_full_html_false_omits_the_document_shell(self, fig, tmp_path: Path):
        out = tmp_path / "fragment.html"
        ex.export_html(fig, out, full_html=False)

        body = out.read_text(encoding="utf-8").lower()
        assert "<!doctype" not in body

    def test_accepts_a_string_path(self, fig, tmp_path: Path):
        out = tmp_path / "as_string.html"
        ex.export_html(fig, str(out))

        assert out.exists()


class TestExportJson:
    def test_round_trips_through_json(self, fig, tmp_path: Path):
        out = tmp_path / "chart.json"
        ex.export_json(fig, out)

        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["data"][0]["name"] == "series"
        assert list(doc["data"][0]["x"]) == [1, 2, 3, 4]

    def test_pretty_false_is_still_valid_json(self, fig, tmp_path: Path):
        out = tmp_path / "compact.json"
        ex.export_json(fig, out, pretty=False)

        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["layout"]["title"]["text"] == "Test Figure"

    def test_creates_missing_parent_directories(self, fig, tmp_path: Path):
        out = tmp_path / "a" / "b" / "chart.json"
        ex.export_json(fig, out)

        assert out.exists()


class TestExportImage:
    def test_names_kaleido_in_the_error_when_absent(self, fig, tmp_path: Path, monkeypatch):
        """Without kaleido the message must say what to install, not just fail."""
        monkeypatch.setattr(ex, "KALEIDO_AVAILABLE", False)

        with pytest.raises(RuntimeError, match="kaleido"):
            ex.export_image(fig, tmp_path / "chart.png")

    def test_forwards_dimensions_to_plotly(self, fig, tmp_path: Path, monkeypatch):
        """width/height must reach write_image rather than being swallowed."""
        seen: dict[str, object] = {}

        def fake_write_image(_fig, path, **kwargs):
            seen.update(kwargs)
            Path(path).write_bytes(b"not really a png")

        monkeypatch.setattr(ex, "KALEIDO_AVAILABLE", True)
        monkeypatch.setattr(ex.pio, "write_image", fake_write_image)

        ex.export_image(fig, tmp_path / "chart.png", width=800, height=600, scale=3)

        assert seen["width"] == 800
        assert seen["height"] == 600
        assert seen["scale"] == 3
        assert seen["format"] == "png"


class TestExportDashboard:
    def test_writes_one_file_per_requested_format(self, fig, tmp_path: Path):
        out = tmp_path / "dash"
        ex.export_dashboard({"My Chart": fig}, out, prefix="chart", formats=["html", "json"])

        assert (out / "chart_My_Chart.html").exists()
        assert (out / "chart_My_Chart.json").exists()

    def test_sanitises_names_that_are_not_pathsafe(self, fig, tmp_path: Path):
        """A name with a slash must not create a directory or escape the output dir."""
        out = tmp_path / "dash"
        ex.export_dashboard({"AAPL / SPY": fig}, out, formats=["html"])

        assert (out / "chart_AAPL___SPY.html").exists()
        assert not (out / "AAPL").exists()

    def test_creates_the_index_unless_asked_not_to(self, fig, tmp_path: Path):
        with_index = tmp_path / "with"
        without = tmp_path / "without"

        ex.export_dashboard({"Chart": fig}, with_index, formats=["html"])
        ex.export_dashboard({"Chart": fig}, without, formats=["html"], create_index=False)

        assert (with_index / "index.html").exists()
        assert not (without / "index.html").exists()

    def test_index_embeds_one_iframe_per_figure(self, tmp_path: Path):
        a = go.Figure(go.Scatter(x=[1], y=[1], name="a"))
        b = go.Figure(go.Scatter(x=[1], y=[2], name="b"))
        out = tmp_path / "dash"

        ex.export_dashboard({"First": a, "Second": b}, out, formats=["html"])

        index = (out / "index.html").read_text(encoding="utf-8")
        assert index.count("<iframe") == 2
        assert "chart_First.html" in index
        assert "chart_Second.html" in index

    def test_empty_figures_still_produce_a_valid_index(self, tmp_path: Path):
        """An empty dashboard is a misuse but should not crash."""
        out = tmp_path / "dash"
        ex.export_dashboard({}, out, formats=["html"])

        assert (out / "index.html").exists()


class TestOptimizeFigureSize:
    def test_downsamples_only_the_oversized_trace(self, tmp_path: Path):
        big = go.Figure(
            data=[
                go.Scatter(x=np.arange(20_000), y=np.arange(20_000), name="big"),
                go.Scatter(x=[1, 2, 3], y=[1, 2, 3], name="small"),
            ]
        )

        out = ex.optimize_figure_size(big, max_points=1_000)

        assert len(out.data[0].x) <= 1_000
        assert len(out.data[1].x) == 3, "a trace under the limit must be untouched"

    def test_keeps_x_and_y_the_same_length(self):
        """Downsampling one axis and not the other would corrupt the series."""
        fig = go.Figure(go.Scatter(x=np.arange(10_000), y=np.arange(10_000)))

        out = ex.optimize_figure_size(fig, max_points=500)

        assert len(out.data[0].x) == len(out.data[0].y)

    def test_does_not_mutate_the_input_figure(self):
        fig = go.Figure(go.Scatter(x=np.arange(5_000), y=np.arange(5_000)))
        original = len(fig.data[0].x)

        ex.optimize_figure_size(fig, max_points=100)

        assert len(fig.data[0].x) == original, "the caller's figure was modified"

    def test_leaves_traces_without_x_alone(self):
        """A bar chart has categorical x; y is the value axis and must not be cut."""
        fig = go.Figure(go.Bar(x=["a", "b", "c"], y=[1.0, 2.0, 3.0]))

        out = ex.optimize_figure_size(fig, max_points=1)

        assert list(out.data[0].y) == [1.0, 2.0, 3.0]

    def test_estimate_html_size_grows_with_more_points(self):
        small = go.Figure(go.Scatter(x=[1, 2], y=[1, 2]))
        large = go.Figure(go.Scatter(x=np.arange(5_000), y=np.arange(5_000)))

        assert ex.estimate_html_size(large) > ex.estimate_html_size(small)

    def test_estimate_html_size_is_positive(self, fig):
        assert ex.estimate_html_size(fig) > 0


class TestNotebookReport:
    def test_writes_a_section_and_toc_entry_per_group(self, tmp_path: Path):
        fig = go.Figure(go.Scatter(x=[1], y=[1], name="s"))
        out = tmp_path / "report.html"

        ex.export_notebook_report(
            {"Returns": fig, "Vol Surface": [fig, fig]},
            out,
            title="Nightly Report",
        )

        body = out.read_text(encoding="utf-8")
        assert "Nightly Report" in body
        assert '<section id="returns">' in body
        assert '<section id="vol-surface">' in body
        # Two figures in the list-valued section.
        assert body.count('class="figure"') == 3

    def test_toc_can_be_omitted(self, tmp_path: Path):
        fig = go.Figure(go.Scatter(x=[1], y=[1]))
        out = tmp_path / "report.html"

        ex.export_notebook_report({"A": fig}, out, include_toc=False)

        assert "<nav>" not in out.read_text(encoding="utf-8")

    def test_description_is_rendered_only_when_given(self, tmp_path: Path):
        fig = go.Figure(go.Scatter(x=[1], y=[1]))

        with_desc = tmp_path / "with.html"
        without = tmp_path / "without.html"
        ex.export_notebook_report({"A": fig}, with_desc, description="hello there")
        ex.export_notebook_report({"A": fig}, without)

        assert "hello there" in with_desc.read_text(encoding="utf-8")
        assert "hello there" not in without.read_text(encoding="utf-8")

    def test_creates_missing_parent_directories(self, tmp_path: Path):
        fig = go.Figure(go.Scatter(x=[1], y=[1]))
        out = tmp_path / "reports" / "nightly" / "r.html"

        ex.export_notebook_report({"A": fig}, out)

        assert out.exists()


class TestQuickExport:
    def test_returns_html_and_only_raster_formats_when_kaleido_is_present(
        self, fig, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(ex, "KALEIDO_AVAILABLE", True)
        monkeypatch.setattr(
            ex.pio, "write_image", lambda _f, path, **k: Path(path).write_bytes(b"x")
        )

        results = ex.quick_export(fig, "my chart", output_dir=tmp_path)

        assert set(results) == {"html", "png", "svg"}
        assert all(p.exists() for p in results.values())

    def test_skips_raster_formats_when_kaleido_is_absent(self, fig, tmp_path: Path, monkeypatch):
        """A caller must not receive a path to a file that was never written."""
        monkeypatch.setattr(ex, "KALEIDO_AVAILABLE", False)

        results = ex.quick_export(fig, "chart", output_dir=tmp_path)

        assert set(results) == {"html"}
        assert all(p.exists() for p in results.values())

    def test_sanitises_the_name(self, fig, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(ex, "KALEIDO_AVAILABLE", False)

        results = ex.quick_export(fig, "a/b c", output_dir=tmp_path)

        assert results["html"].name == "a_b_c.html"
