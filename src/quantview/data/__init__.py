"""QuantView Data Layer.

This package provides database connection management, schema definitions,
and data ingestion utilities for the QuantView application.
"""

from quantview.data.connection import get_connection, init_database
from quantview.data.ingest import drop_schema, init_schema, verify_schema
from quantview.data.schema import get_all_ddl, get_core_ddl, get_raw_ddl, get_universe_ddl
from quantview.data.universe import (
    UniverseSymbol,
    build_universe_records,
    deactivate_symbol,
    detect_corporate_actions,
    get_active_symbols,
    get_symbol_info,
    init_universe_table,
)

__all__ = [
    "UniverseSymbol",
    "build_universe_records",
    "deactivate_symbol",
    "detect_corporate_actions",
    "drop_schema",
    "get_active_symbols",
    "get_all_ddl",
    "get_connection",
    "get_core_ddl",
    "get_raw_ddl",
    "get_symbol_info",
    "get_universe_ddl",
    "init_database",
    "init_schema",
    "init_universe_table",
    "verify_schema",
]
