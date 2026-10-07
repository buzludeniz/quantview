"""QuantView Symbol Universe and Corporate Actions.

This module provides symbol reference data management, including the universe table
pre-seeded with S&P 500 symbols, major ETFs, and treasury series. It also handles
corporate action detection and processing.
"""

from dataclasses import dataclass

import duckdb

from quantview.data.connection import get_connection
from quantview.data.schema import get_universe_ddl

# =============================================================================
# SP500 Symbols (as of 2024) - Major components for pre-seeding
# =============================================================================

# Note: This is a representative subset of S&P 500 for development.
# Production should use a complete, updated list from a reliable source.
SP500_SYMBOLS = [
    # Technology
    ("AAPL", "Apple Inc.", "Technology", "Consumer Electronics"),
    ("MSFT", "Microsoft Corporation", "Technology", "Software"),
    ("NVDA", "NVIDIA Corporation", "Technology", "Semiconductors"),
    ("GOOGL", "Alphabet Inc. Class A", "Technology", "Internet Services"),
    ("GOOG", "Alphabet Inc. Class C", "Technology", "Internet Services"),
    ("META", "Meta Platforms Inc.", "Technology", "Internet Services"),
    ("AVGO", "Broadcom Inc.", "Technology", "Semiconductors"),
    ("TSLA", "Tesla Inc.", "Consumer Cyclical", "Auto Manufacturers"),
    ("ORCL", "Oracle Corporation", "Technology", "Software"),
    ("CRM", "Salesforce Inc.", "Technology", "Software"),
    ("ADBE", "Adobe Inc.", "Technology", "Software"),
    ("AMD", "Advanced Micro Devices Inc.", "Technology", "Semiconductors"),
    ("INTC", "Intel Corporation", "Technology", "Semiconductors"),
    ("CSCO", "Cisco Systems Inc.", "Technology", "Communication Equipment"),
    ("QCOM", "QUALCOMM Inc.", "Technology", "Semiconductors"),
    ("TXN", "Texas Instruments Inc.", "Technology", "Semiconductors"),
    ("AMAT", "Applied Materials Inc.", "Technology", "Semiconductor Equipment"),
    ("INTU", "Intuit Inc.", "Technology", "Software"),
    ("IBM", "International Business Machines", "Technology", "IT Services"),
    ("NOW", "ServiceNow Inc.", "Technology", "Software"),
    ("AMZN", "Amazon.com Inc.", "Consumer Cyclical", "Internet Retail"),
    # Financials
    ("JPM", "JPMorgan Chase & Co.", "Financial Services", "Banks"),
    ("BAC", "Bank of America Corp.", "Financial Services", "Banks"),
    ("WFC", "Wells Fargo & Co.", "Financial Services", "Banks"),
    ("GS", "Goldman Sachs Group Inc.", "Financial Services", "Capital Markets"),
    ("MS", "Morgan Stanley", "Financial Services", "Capital Markets"),
    ("C", "Citigroup Inc.", "Financial Services", "Banks"),
    ("AXP", "American Express Co.", "Financial Services", "Credit Services"),
    ("BLK", "BlackRock Inc.", "Financial Services", "Asset Management"),
    ("SCHW", "Charles Schwab Corp.", "Financial Services", "Capital Markets"),
    ("USB", "U.S. Bancorp", "Financial Services", "Banks"),
    ("PNC", "PNC Financial Services", "Financial Services", "Banks"),
    ("TFC", "Truist Financial Corp.", "Financial Services", "Banks"),
    ("COF", "Capital One Financial", "Financial Services", "Credit Services"),
    ("SPGI", "S&P Global Inc.", "Financial Services", "Financial Data"),
    ("MCO", "Moody's Corporation", "Financial Services", "Financial Data"),
    ("ICE", "Intercontinental Exchange", "Financial Services", "Capital Markets"),
    ("CME", "CME Group Inc.", "Financial Services", "Capital Markets"),
    # Healthcare
    ("UNH", "UnitedHealth Group Inc.", "Healthcare", "Healthcare Plans"),
    ("JNJ", "Johnson & Johnson", "Healthcare", "Drug Manufacturers"),
    ("LLY", "Eli Lilly and Company", "Healthcare", "Drug Manufacturers"),
    ("PFE", "Pfizer Inc.", "Healthcare", "Drug Manufacturers"),
    ("ABBV", "AbbVie Inc.", "Healthcare", "Drug Manufacturers"),
    ("MRK", "Merck & Co. Inc.", "Healthcare", "Drug Manufacturers"),
    ("TMO", "Thermo Fisher Scientific", "Healthcare", "Medical Devices"),
    ("ABT", "Abbott Laboratories", "Healthcare", "Medical Devices"),
    ("DHR", "Danaher Corporation", "Healthcare", "Medical Devices"),
    ("BMY", "Bristol-Myers Squibb", "Healthcare", "Drug Manufacturers"),
    ("AMGN", "Amgen Inc.", "Healthcare", "Biotechnology"),
    ("GILD", "Gilead Sciences Inc.", "Healthcare", "Biotechnology"),
    ("MDT", "Medtronic plc", "Healthcare", "Medical Devices"),
    ("VRTX", "Vertex Pharmaceuticals", "Healthcare", "Biotechnology"),
    ("REGN", "Regeneron Pharmaceuticals", "Healthcare", "Biotechnology"),
    ("ZTS", "Zoetis Inc.", "Healthcare", "Drug Manufacturers"),
    ("ISRG", "Intuitive Surgical Inc.", "Healthcare", "Medical Devices"),
    ("CVS", "CVS Health Corp.", "Healthcare", "Healthcare Plans"),
    ("CI", "Cigna Group", "Healthcare", "Healthcare Plans"),
    ("HUM", "Humana Inc.", "Healthcare", "Healthcare Plans"),
    # Consumer Staples
    ("PG", "Procter & Gamble Co.", "Consumer Defensive", "Household Products"),
    ("KO", "Coca-Cola Company", "Consumer Defensive", "Beverages"),
    ("PEP", "PepsiCo Inc.", "Consumer Defensive", "Beverages"),
    ("COST", "Costco Wholesale Corp.", "Consumer Defensive", "Discount Stores"),
    ("WMT", "Walmart Inc.", "Consumer Defensive", "Discount Stores"),
    ("PM", "Philip Morris International", "Consumer Defensive", "Tobacco"),
    ("MO", "Altria Group Inc.", "Consumer Defensive", "Tobacco"),
    ("MDLZ", "Mondelez International", "Consumer Defensive", "Packaged Foods"),
    ("CL", "Colgate-Palmolive Co.", "Consumer Defensive", "Household Products"),
    ("KMB", "Kimberly-Clark Corp.", "Consumer Defensive", "Household Products"),
    ("GIS", "General Mills Inc.", "Consumer Defensive", "Packaged Foods"),
    ("K", "Kellogg Company", "Consumer Defensive", "Packaged Foods"),
    ("HSY", "Hershey Company", "Consumer Defensive", "Packaged Foods"),
    ("KHC", "Kraft Heinz Company", "Consumer Defensive", "Packaged Foods"),
    ("SJM", "J.M. Smucker Company", "Consumer Defensive", "Packaged Foods"),
    # Consumer Discretionary
    ("HD", "Home Depot Inc.", "Consumer Cyclical", "Home Improvement"),
    ("LOW", "Lowe's Companies Inc.", "Consumer Cyclical", "Home Improvement"),
    ("MCD", "McDonald's Corporation", "Consumer Cyclical", "Restaurants"),
    ("SBUX", "Starbucks Corporation", "Consumer Cyclical", "Restaurants"),
    ("NKE", "Nike Inc.", "Consumer Cyclical", "Apparel"),
    ("TJX", "TJX Companies Inc.", "Consumer Cyclical", "Apparel Retail"),
    ("BKNG", "Booking Holdings Inc.", "Consumer Cyclical", "Travel Services"),
    ("MAR", "Marriott International", "Consumer Cyclical", "Lodging"),
    ("HLT", "Hilton Worldwide Holdings", "Consumer Cyclical", "Lodging"),
    ("GM", "General Motors Company", "Consumer Cyclical", "Auto Manufacturers"),
    ("F", "Ford Motor Company", "Consumer Cyclical", "Auto Manufacturers"),
    # Industrials
    ("RTX", "RTX Corporation", "Industrials", "Aerospace & Defense"),
    ("HON", "Honeywell International", "Industrials", "Conglomerates"),
    ("UPS", "United Parcel Service", "Industrials", "Integrated Freight"),
    ("CAT", "Caterpillar Inc.", "Industrials", "Construction Equipment"),
    ("DE", "Deere & Company", "Industrials", "Farm & Heavy Machinery"),
    ("BA", "Boeing Company", "Industrials", "Aerospace & Defense"),
    ("LMT", "Lockheed Martin Corp.", "Industrials", "Aerospace & Defense"),
    ("NOC", "Northrop Grumman Corp.", "Industrials", "Aerospace & Defense"),
    ("GE", "General Electric Company", "Industrials", "Conglomerates"),
    ("MMM", "3M Company", "Industrials", "Conglomerates"),
    ("ITW", "Illinois Tool Works Inc.", "Industrials", "Specialty Industrial Machinery"),
    ("EMR", "Emerson Electric Co.", "Industrials", "Specialty Industrial Machinery"),
    ("ETN", "Eaton Corporation plc", "Industrials", "Specialty Industrial Machinery"),
    ("PH", "Parker-Hannifin Corp.", "Industrials", "Specialty Industrial Machinery"),
    ("ROK", "Rockwell Automation Inc.", "Industrials", "Specialty Industrial Machinery"),
    ("DOV", "Dover Corporation", "Industrials", "Specialty Industrial Machinery"),
    ("XYL", "Xylem Inc.", "Industrials", "Specialty Industrial Machinery"),
    ("IEX", "IDEX Corporation", "Industrials", "Specialty Industrial Machinery"),
    ("NDSN", "Nordson Corporation", "Industrials", "Specialty Industrial Machinery"),
    ("FLS", "Flowserve Corporation", "Industrials", "Specialty Industrial Machinery"),
    # Energy
    ("XOM", "Exxon Mobil Corporation", "Energy", "Oil & Gas Integrated"),
    ("CVX", "Chevron Corporation", "Energy", "Oil & Gas Integrated"),
    ("COP", "ConocoPhillips", "Energy", "Oil & Gas E&P"),
    ("EOG", "EOG Resources Inc.", "Energy", "Oil & Gas E&P"),
    ("SLB", "Schlumberger Limited", "Energy", "Oil & Gas Equipment & Services"),
    ("MPC", "Marathon Petroleum Corp.", "Energy", "Oil & Gas Refining & Marketing"),
    ("VLO", "Valero Energy Corporation", "Energy", "Oil & Gas Refining & Marketing"),
    ("PSX", "Phillips 66", "Energy", "Oil & Gas Refining & Marketing"),
    ("OXY", "Occidental Petroleum Corp.", "Energy", "Oil & Gas E&P"),
    ("HAL", "Halliburton Company", "Energy", "Oil & Gas Equipment & Services"),
    ("BKR", "Baker Hughes Company", "Energy", "Oil & Gas Equipment & Services"),
    # Utilities
    ("NEE", "NextEra Energy Inc.", "Utilities", "Electric Utilities"),
    ("DUK", "Duke Energy Corporation", "Utilities", "Electric Utilities"),
    ("SO", "Southern Company", "Utilities", "Electric Utilities"),
    ("D", "Dominion Energy Inc.", "Utilities", "Electric Utilities"),
    ("AEP", "American Electric Power", "Utilities", "Electric Utilities"),
    ("EXC", "Exelon Corporation", "Utilities", "Electric Utilities"),
    ("SRE", "Sempra Energy", "Utilities", "Electric Utilities"),
    ("PEG", "Public Service Enterprise Group", "Utilities", "Electric Utilities"),
    ("ED", "Consolidated Edison Inc.", "Utilities", "Electric Utilities"),
    ("PPL", "PPL Corporation", "Utilities", "Electric Utilities"),
    # Real Estate
    ("AMT", "American Tower Corporation", "Real Estate", "REIT"),
    ("PLD", "Prologis Inc.", "Real Estate", "REIT"),
    ("CCI", "Crown Castle Inc.", "Real Estate", "REIT"),
    ("EQIX", "Equinix Inc.", "Real Estate", "REIT"),
    ("PSA", "Public Storage", "Real Estate", "REIT"),
    ("SPG", "Simon Property Group", "Real Estate", "REIT"),
    ("O", "Realty Income Corporation", "Real Estate", "REIT"),
    ("WELL", "Welltower Inc.", "Real Estate", "REIT"),
    ("DLR", "Digital Realty Trust", "Real Estate", "REIT"),
    ("VICI", "VICI Properties Inc.", "Real Estate", "REIT"),
    # Materials
    ("LIN", "Linde plc", "Basic Materials", "Specialty Chemicals"),
    ("APD", "Air Products & Chemicals", "Basic Materials", "Specialty Chemicals"),
    ("SHW", "Sherwin-Williams Company", "Basic Materials", "Specialty Chemicals"),
    ("FCX", "Freeport-McMoRan Inc.", "Basic Materials", "Copper"),
    ("NEM", "Newmont Corporation", "Basic Materials", "Gold"),
    ("ECL", "Ecolab Inc.", "Basic Materials", "Specialty Chemicals"),
    ("DD", "DuPont de Nemours", "Basic Materials", "Specialty Chemicals"),
    ("DOW", "Dow Inc.", "Basic Materials", "Chemicals"),
    ("PPG", "PPG Industries Inc.", "Basic Materials", "Specialty Chemicals"),
    ("IFF", "International Flavors & Fragrances", "Basic Materials", "Specialty Chemicals"),
    # Communication Services
    ("NFLX", "Netflix Inc.", "Communication Services", "Entertainment"),
    ("DIS", "Walt Disney Company", "Communication Services", "Entertainment"),
    ("CMCSA", "Comcast Corporation", "Communication Services", "Telecom Services"),
    ("VZ", "Verizon Communications", "Communication Services", "Telecom Services"),
    ("T", "AT&T Inc.", "Communication Services", "Telecom Services"),
    ("TMUS", "T-Mobile US Inc.", "Communication Services", "Telecom Services"),
    ("CHTR", "Charter Communications", "Communication Services", "Telecom Services"),
]

# Major ETFs
MAJOR_ETFS = [
    ("SPY", "SPDR S&P 500 ETF Trust", "Financial Services", "ETF", "NYSE Arca"),
    ("QQQ", "Invesco QQQ Trust", "Financial Services", "ETF", "NASDAQ"),
    ("IWM", "iShares Russell 2000 ETF", "Financial Services", "ETF", "NYSE Arca"),
    ("DIA", "SPDR Dow Jones Industrial Average ETF", "Financial Services", "ETF", "NYSE Arca"),
    ("VTI", "Vanguard Total Stock Market ETF", "Financial Services", "ETF", "NYSE Arca"),
    ("VOO", "Vanguard S&P 500 ETF", "Financial Services", "ETF", "NYSE Arca"),
    ("VEA", "Vanguard FTSE Developed Markets ETF", "Financial Services", "ETF", "NYSE Arca"),
    ("VWO", "Vanguard FTSE Emerging Markets ETF", "Financial Services", "ETF", "NYSE Arca"),
    ("AGG", "iShares Core U.S. Aggregate Bond ETF", "Financial Services", "ETF", "NYSE Arca"),
    ("TLT", "iShares 20+ Year Treasury Bond ETF", "Financial Services", "ETF", "NASDAQ"),
    ("IEF", "iShares 7-10 Year Treasury Bond ETF", "Financial Services", "ETF", "NASDAQ"),
    ("SHY", "iShares 1-3 Year Treasury Bond ETF", "Financial Services", "ETF", "NASDAQ"),
    (
        "LQD",
        "iShares iBoxx $ Investment Grade Corporate Bond ETF",
        "Financial Services",
        "ETF",
        "NYSE Arca",
    ),
    (
        "HYG",
        "iShares iBoxx $ High Yield Corporate Bond ETF",
        "Financial Services",
        "ETF",
        "NYSE Arca",
    ),
    ("GLD", "SPDR Gold Shares", "Financial Services", "ETF", "NYSE Arca"),
    ("SLV", "iShares Silver Trust", "Financial Services", "ETF", "NYSE Arca"),
    ("USO", "United States Oil Fund", "Financial Services", "ETF", "NYSE Arca"),
    ("XLF", "Financial Select Sector SPDR Fund", "Financial Services", "ETF", "NYSE Arca"),
    ("XLK", "Technology Select Sector SPDR Fund", "Financial Services", "ETF", "NYSE Arca"),
    ("XLE", "Energy Select Sector SPDR Fund", "Financial Services", "ETF", "NYSE Arca"),
    ("XLV", "Health Care Select Sector SPDR Fund", "Financial Services", "ETF", "NYSE Arca"),
    ("XLI", "Industrial Select Sector SPDR Fund", "Financial Services", "ETF", "NYSE Arca"),
    ("XLP", "Consumer Staples Select Sector SPDR Fund", "Financial Services", "ETF", "NYSE Arca"),
    (
        "XLY",
        "Consumer Discretionary Select Sector SPDR Fund",
        "Financial Services",
        "ETF",
        "NYSE Arca",
    ),
    ("XLU", "Utilities Select Sector SPDR Fund", "Financial Services", "ETF", "NYSE Arca"),
    ("XLB", "Materials Select Sector SPDR Fund", "Financial Services", "ETF", "NYSE Arca"),
    ("XLRE", "Real Estate Select Sector SPDR Fund", "Financial Services", "ETF", "NYSE Arca"),
    (
        "XLC",
        "Communication Services Select Sector SPDR Fund",
        "Financial Services",
        "ETF",
        "NYSE Arca",
    ),
]

# Treasury series (FRED series IDs)
TREASURY_SERIES = [
    ("DGS1MO", "1-Month Treasury Constant Maturity Rate", "short"),
    ("DGS3MO", "3-Month Treasury Constant Maturity Rate", "short"),
    ("DGS6MO", "6-Month Treasury Constant Maturity Rate", "short"),
    ("DGS1", "1-Year Treasury Constant Maturity Rate", "short"),
    ("DGS2", "2-Year Treasury Constant Maturity Rate", "short"),
    ("DGS3", "3-Year Treasury Constant Maturity Rate", "medium"),
    ("DGS5", "5-Year Treasury Constant Maturity Rate", "medium"),
    ("DGS7", "7-Year Treasury Constant Maturity Rate", "medium"),
    ("DGS10", "10-Year Treasury Constant Maturity Rate", "long"),
    ("DGS20", "20-Year Treasury Constant Maturity Rate", "long"),
    ("DGS30", "30-Year Treasury Constant Maturity Rate", "long"),
]


@dataclass
class UniverseSymbol:
    """Symbol metadata for the universe table."""

    symbol: str
    figi: str | None = None
    cusip: str | None = None
    isin: str | None = None
    exchange: str = "NASDAQ"
    currency: str = "USD"
    name: str = ""
    sector: str | None = None
    industry: str | None = None
    active: bool = True


def build_universe_records() -> list[UniverseSymbol]:
    """Build complete universe records from all sources.

    Returns:
        List of UniverseSymbol records for insertion.
    """
    records = []

    # Add S&P 500 symbols
    for symbol, name, sector, industry in SP500_SYMBOLS:
        records.append(
            UniverseSymbol(
                symbol=symbol,
                name=name,
                sector=sector,
                industry=industry,
                exchange="NASDAQ" if symbol not in ["BRK.B", "BF.B"] else "NYSE",
            )
        )

    # Add major ETFs
    for symbol, name, sector, industry, exchange in MAJOR_ETFS:
        records.append(
            UniverseSymbol(
                symbol=symbol,
                name=name,
                sector=sector,
                industry=industry,
                exchange=exchange,
            )
        )

    # Add treasury series as synthetic symbols
    for series_id, name, tenor in TREASURY_SERIES:
        records.append(
            UniverseSymbol(
                symbol=f"UST{series_id[3:]}",  # e.g., UST10 for DGS10
                name=f"US Treasury {name}",
                sector="Government",
                industry=f"Treasury {tenor.capitalize()}",
                exchange="FRED",
            )
        )

    return records


def init_universe_table(conn: duckdb.DuckDBPyConnection | None = None) -> int:
    """Initialize the universe table with pre-seeded data.

    Args:
        conn: Optional database connection. If None, creates a new one.

    Returns:
        Number of records inserted/updated.
    """
    if conn is None:
        conn = get_connection()

    # Ensure core schema exists
    conn.execute("CREATE SCHEMA IF NOT EXISTS core")

    # Ensure universe table exists
    for ddl in get_universe_ddl():
        conn.execute(ddl)

    # Build records
    records = build_universe_records()

    # Insert or update records
    inserted = 0
    for record in records:
        conn.execute(
            """
            INSERT INTO core.universe_table (symbol, figi, cusip, isin, exchange, currency, name, sector, industry, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (symbol) DO UPDATE SET
                figi = EXCLUDED.figi,
                cusip = EXCLUDED.cusip,
                isin = EXCLUDED.isin,
                exchange = EXCLUDED.exchange,
                currency = EXCLUDED.currency,
                name = EXCLUDED.name,
                sector = EXCLUDED.sector,
                industry = EXCLUDED.industry,
                active = EXCLUDED.active
        """,
            [
                record.symbol,
                record.figi,
                record.cusip,
                record.isin,
                record.exchange,
                record.currency,
                record.name,
                record.sector,
                record.industry,
                record.active,
            ],
        )
        inserted += 1

    return inserted


def get_active_symbols(conn: duckdb.DuckDBPyConnection | None = None) -> list[str]:
    """Get list of active symbols from universe.

    Args:
        conn: Optional database connection.

    Returns:
        List of active symbol strings.
    """
    if conn is None:
        conn = get_connection(read_only=True)

    result = conn.execute("""
        SELECT symbol FROM core.universe_table WHERE active = TRUE ORDER BY symbol
    """).fetchall()

    return [row[0] for row in result]


def get_symbol_info(
    symbol: str, conn: duckdb.DuckDBPyConnection | None = None
) -> UniverseSymbol | None:
    """Get metadata for a specific symbol.

    Args:
        symbol: The symbol to look up.
        conn: Optional database connection.

    Returns:
        UniverseSymbol if found, None otherwise.
    """
    if conn is None:
        conn = get_connection(read_only=True)

    row = conn.execute(
        """
        SELECT symbol, figi, cusip, isin, exchange, currency, name, sector, industry, active
        FROM core.universe_table WHERE symbol = ?
    """,
        [symbol],
    ).fetchone()

    if row is None:
        return None

    return UniverseSymbol(
        symbol=row[0],
        figi=row[1],
        cusip=row[2],
        isin=row[3],
        exchange=row[4],
        currency=row[5],
        name=row[6],
        sector=row[7],
        industry=row[8],
        active=row[9],
    )


def deactivate_symbol(symbol: str, conn: duckdb.DuckDBPyConnection | None = None) -> bool:
    """Mark a symbol as inactive in the universe.

    Args:
        symbol: The symbol to deactivate.
        conn: Optional database connection.

    Returns:
        True if symbol was found and deactivated, False otherwise.
    """
    if conn is None:
        conn = get_connection()

    # Check if symbol exists and is active
    existing = conn.execute(
        "SELECT active FROM core.universe_table WHERE symbol = ?",
        [symbol],
    ).fetchone()
    if existing is None or existing[0] is False:
        return False

    # Perform update
    conn.execute(
        """
        UPDATE core.universe_table SET active = FALSE WHERE symbol = ?
    """,
        [symbol],
    )

    # Verify the update worked
    updated = conn.execute(
        "SELECT active FROM core.universe_table WHERE symbol = ?",
        [symbol],
    ).fetchone()
    return updated is not None and updated[0] is False


def detect_corporate_actions(
    symbol: str, start_date: str, end_date: str, conn: duckdb.DuckDBPyConnection | None = None
) -> list[dict]:
    """Detect corporate actions from raw Yahoo data.

    Args:
        symbol: The symbol to check.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        conn: Optional database connection.

    Returns:
        List of corporate action dicts with keys: date, action, ratio, details.
    """
    if conn is None:
        conn = get_connection(read_only=True)

    rows = conn.execute(
        """
        SELECT date, splits, dividends
        FROM raw.yahoo_equities
        WHERE symbol = ? AND date BETWEEN ? AND ?
        AND (splits != 1.0 OR dividends > 0)
        ORDER BY date
    """,
        [symbol, start_date, end_date],
    ).fetchall()

    actions = []
    for row in rows:
        date, splits, dividends = row
        if splits != 1.0:
            actions.append(
                {
                    "date": date,
                    "action": "split",
                    "ratio": float(splits),
                    "details": f"Stock split: {splits}:1",
                }
            )
        if dividends > 0:
            actions.append(
                {
                    "date": date,
                    "action": "dividend",
                    "ratio": float(dividends),
                    "details": f"Dividend: ${dividends}",
                }
            )

    return actions


def apply_corporate_actions(
    symbol: str,
    action_date: str,
    action_type: str,
    ratio: float,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> None:
    """Record a corporate action for a symbol.

    Args:
        symbol: The affected symbol.
        action_date: Date of action (YYYY-MM-DD).
        action_type: Type of action ('split', 'dividend', 'merger', 'spinoff', 'other').
        ratio: Split ratio or dividend amount.
        conn: Optional database connection.
    """
    if conn is None:
        conn = get_connection()

    # This is a placeholder for a more sophisticated corporate actions table
    # In production, this would insert into a dedicated corporate_actions table
    details = {
        "split": f"Stock split: {ratio}:1",
        "dividend": f"Dividend: ${ratio}",
        "merger": f"Merger with ratio {ratio}",
        "spinoff": f"Spinoff with ratio {ratio}",
    }.get(action_type, f"Corporate action: {action_type}")

    conn.execute(
        """
        INSERT INTO core.corporate_actions_log (symbol, date, action, ratio, details, created_ts)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """,
        [symbol, action_date, action_type, ratio, details],
    )


def create_corporate_actions_log(conn: duckdb.DuckDBPyConnection | None = None) -> None:
    """Create the corporate actions log table if it doesn't exist."""
    if conn is None:
        conn = get_connection()

    # Ensure core schema exists
    conn.execute("CREATE SCHEMA IF NOT EXISTS core")

    conn.execute("CREATE SEQUENCE IF NOT EXISTS corporate_action_id_seq START 1")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS core.corporate_actions_log (
            id BIGINT PRIMARY KEY DEFAULT nextval('corporate_action_id_seq'),
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            action VARCHAR NOT NULL,
            ratio DOUBLE,
            details VARCHAR,
            created_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_corp_actions_symbol_date ON core.corporate_actions_log(symbol, date DESC)"
    )
