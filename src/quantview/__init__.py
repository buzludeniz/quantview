"""QuantView: Local-First Quant Research Desktop Application.

A Python-based quantitative research platform for equity and options analysis
with local data persistence via DuckDB.
"""

from quantview.config import Settings, get_settings, reload_settings, settings

__version__ = "0.1.0"
__all__ = ["Settings", "get_settings", "reload_settings", "settings"]
