"""QuantView Data Connectors.

This package provides data connectors for various financial data sources.
Each connector implements a fetch() method returning normalized DataFrames
matching the raw schema tables.
"""

from quantview.data.connectors.alphavantage import AlphaVantageConnector
from quantview.data.connectors.base import BaseConnector
from quantview.data.connectors.cboe import CboeConnector
from quantview.data.connectors.fred import FredConnector
from quantview.data.connectors.yahoo import YahooConnector

__all__ = [
    "AlphaVantageConnector",
    "BaseConnector",
    "CboeConnector",
    "FredConnector",
    "YahooConnector",
]
