"""Analytics layer: curves, portfolio attribution, risk, and volatility surfaces.

Kept deliberately free of imports from the notebook and desktop layers. Those
present what this layer computes; nothing here should need Qt or ipywidgets
installed to fit a curve.
"""

from __future__ import annotations

__all__ = ["curves", "portfolio", "risk", "surface"]
