"""Entry point for ``python -m quantview.desktop``.

The README documents this command, so it needs to actually work; without this
module ``python -m quantview.desktop`` failed with "No module named
quantview.desktop.__main__".
"""

from __future__ import annotations

from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
