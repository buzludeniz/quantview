"""Entry point for ``python -m quantview.data``.

Runs the ingestion pipeline from the command line. The README previously
documented ``python -m quantview.ingest``, which was a module path that does not
exist; the ingest code lives in quantview.data.ingest.
"""

from __future__ import annotations

from .ingest import main

if __name__ == "__main__":
    raise SystemExit(main())
