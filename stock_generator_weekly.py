"""
Stock Report Generator using Groq API

This script fetches stocks that dropped more than 10% in the last 24 hours
with a market cap greater than $1 billion, and generates an analysis report
using the Groq API.
"""

import os
import json
import logging
from typing import Optional
from datetime import datetime
from pathlib import Path
import requests
from groq import Groq


# Load GROQ_API_KEY from groq.env file
def load_groq_env():
    """Load GROQ_API_KEY from groq.env file"""
    env_file_paths = [
        'groq.env',
        '.env',
        Path.cwd() / 'groq.env',
        Path.home() / 'groq.env',
    ]

    for env_path in env_file_paths:
        env_path_str = str(env_path) if isinstance(env_path, Path) else env_path
        if os.path.exists(env_path_str):
            try:
                with open(env_path_str, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            os.environ[key] = value
                return True
            except Exception as e:
                pass
    return False


# Load env before using
load_groq_env()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockReportGenerator:
    """Generate stock reports using market data and Groq API."""

    def __init__(self, groq_api_key: Optional[str] = None):
        """
        Initialize the stock report generator.

        Args:
            groq_api_key: Groq API key. If None, tries to load from:
                1. groq.env file in same directory
                2. GROQ_API_KEY environment variable
        """
        self.groq_api_key = groq_api_key or self._load_groq_key()
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not provided. Please either:\n"
                "1. Create a 'groq.env' file in the same directory with: GROQ_API_KEY=your_key_here\n"
                "2. Set the GROQ_API_KEY environment variable"
            )

        self.client = Groq(api_key=self.groq_api_key)
        self.stocks_data = []

    def _load_groq_key(self) -> Optional[str]:
        """
        Load Groq API key from groq.env file or environment variable.

        Returns:
            The API key if found, None otherwise.
        """
        # Try loading from groq.env in current directory
        groq_env_path = Path(__file__).parent / "groq.env"
        if groq_env_path.exists():
            try:
                with open(groq_env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            if key.strip() == "GROQ_API_KEY":
                                api_key = value.strip().strip('"').strip("'")
                                logger.info("Loaded GROQ_API_KEY from groq.env")
                                return api_key
            except Exception as e:
                logger.warning(f"Error reading groq.env: {e}")

        # Fall back to environment variable
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            logger.info("Loaded GROQ_API_KEY from environment variable")
            return api_key

        return None

    def fetch_stocks_data(self) -> list:
        """
        Fetch stocks that dropped more than 10% in last 5 days (weekly timeframe).

        Returns:
            List of stock dictionaries with symbol, price, change, market_cap, etc.
        """
        logger.info("Fetching stocks data (5-day/weekly timeframe)...")

        try:
            import yfinance as yf

            symbols = self._get_all_large_cap_stocks()

            stocks_with_drops = []
            total_checked = 0
            skipped_stocks = {}  # Track why stocks are skipped

            for symbol in symbols:
                try:
                    total_checked += 1
                    if total_checked % 50 == 0:
                        logger.info(f"Progress: Checked {total_checked}/{len(symbols)} stocks...")

                    ticker = yf.Ticker(symbol)
                    info = ticker.info

                    # Check if ticker exists
                    if not info or 'longName' not in info or info.get('marketCap') is None:
                        if symbol == 'CRWD':
                            logger.info(
                                f"CRWD SKIP: No market cap data - info keys: {list(info.keys()) if info else 'None'}")
                            skipped_stocks['CRWD'] = 'No market cap data'
                        logger.debug(f"Skipping {symbol}: No market cap data available")
                        continue

                    # Get historical data for 5-day change (fetch 10 days to ensure we have 5+ trading days)
                    hist = ticker.history(period="10d")
                    if hist is None or len(hist) < 6:
                        if symbol == 'CRWD':
                            logger.info(
                                f"CRWD SKIP: Insufficient data - got {len(hist) if hist is not None else 0} days (need 6+)")
                            skipped_stocks[
                                'CRWD'] = f'Only {len(hist) if hist is not None else 0} trading days (need 6+)'
                        logger.debug(
                            f"Skipping {symbol}: Insufficient historical data (need 6+ trading days, have {len(hist) if hist is not None else 0})")
                        continue

                    closing_prices = hist['Close']

                    # Get price from exactly 5 trading days ago
                    # If we don't have 6 days of data, skip this stock
                    if len(closing_prices) < 6:
                        logger.debug(
                            f"Skipping {symbol}: Not enough trading days (need 6+, have {len(closing_prices)})")
                        continue

                    # Price from 5 trading days ago (index -6, since -1 is today)
                    price_5days_ago = closing_prices.iloc[-6]
                    # Current price (most recent)
                    current_price = closing_prices.iloc[-1]

                    # Skip if prices are invalid
                    if price_5days_ago <= 0 or current_price <= 0:
                        logger.debug(f"Skipping {symbol}: Invalid price data")
                        continue

                    # Check for NaN values
                    if price_5days_ago != price_5days_ago or current_price != current_price:
                        logger.debug(f"Skipping {symbol}: Invalid price data (NaN)")
                        continue

                    # Calculate 5-day percentage change
                    change_percent = ((current_price - price_5days_ago) / price_5days_ago) * 100

                    # Check criteria: >10% drop AND market cap >$500M
                    market_cap = info.get('marketCap', 0)

                    if not market_cap or market_cap == 0:
                        logger.debug(f"Skipping {symbol}: No market cap data")
                        continue

                    if change_percent < -10 and market_cap > 500_000_000:  # Back to >10% drop
                        stocks_with_drops.append({
                            'symbol': symbol,
                            'price': current_price,
                            'price_5days_ago': price_5days_ago,
                            'change_percent': round(change_percent, 2),
                            'market_cap': market_cap,
                            'company_name': info.get('longName', symbol),
                            'sector': info.get('sector', 'N/A'),
                            'industry': info.get('industry', 'N/A'),
                        })
                    else:
                        if symbol == 'CRWD':
                            logger.info(
                                f"CRWD SKIP: Drop {change_percent:.2f}% (need >10%) OR Market Cap ${market_cap / 1e9:.2f}B (need >$0.5B)")
                            skipped_stocks['CRWD'] = f'Drop {change_percent:.2f}% | Market Cap ${market_cap / 1e9:.2f}B'
                except Exception as e:
                    logger.debug(f"Skipping {symbol}: {type(e).__name__}")
                    continue

            # Sort by percentage drop (worst first)
            stocks_with_drops.sort(key=lambda x: x['change_percent'])

            self.stocks_data = stocks_with_drops
            logger.info(f"Found {len(self.stocks_data)} stocks with >10% drop in last 5 days and >$500M market cap")

            # Log why CRWD was skipped if it was
            if 'CRWD' in skipped_stocks:
                logger.warning(f"⚠️  CRWD SKIPPED: {skipped_stocks['CRWD']}")
            elif any(s['symbol'] == 'CRWD' for s in stocks_with_drops):
                logger.info("✅ CRWD INCLUDED in report")

            return self.stocks_data

        except ImportError:
            logger.error("yfinance not installed. Install with: pip install yfinance")
            raise

    def _get_all_large_cap_stocks(self) -> list:
        """
        Get top large-cap stocks dynamically based on market trends and volume.
        Uses multi-strategy approach to fetch S&P 500 + sector stocks.

        Returns:
            List of stock symbols to analyze (dynamically fetched from market data)
        """
        logger.info("Fetching latest trending stocks dynamically for weekly analysis...")

        try:
            import yfinance as yf

            symbols = []

            # Strategy 1: Get S&P 500 stocks (most liquid)
            try:
                logger.info("Fetching S&P 500 stocks...")
                sp500_symbols = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK.B',
                    'JNJ', 'V', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'INTC', 'PYPL',
                    'CRM', 'NFLX', 'PEP', 'CSCO', 'AMD', 'LLY', 'ABBV', 'IBM',
                    'ORCL', 'COST', 'MCD', 'ADBE', 'ACN', 'NKE', 'QCOM', 'KO',
                    'AVGO', 'CCI', 'TXN', 'TMUS', 'HON', 'MU', 'INTU', 'UPS',
                    'BA', 'CAT', 'RTX', 'LMT', 'GE', 'MMM', 'XOM', 'CVX',
                    'SLB', 'COP', 'EOG', 'MPC', 'PSX', 'NEE', 'DUK', 'SO',
                    'AEP', 'EXC', 'UNH', 'PFE', 'AMGN', 'TMO', 'MRK', 'ABT',
                    'BMY', 'GILD', 'BIIB', 'REGN', 'CRWD', 'ZM', 'DDOG', 'NET',
                    'SNOW', 'OKTA', 'TWLO', 'SHOP', 'RBLX', 'DASH', 'UBER', 'COIN',
                ]
                symbols.extend(sp500_symbols)
                logger.info(f"Added {len(sp500_symbols)} S&P 500 stocks")
            except Exception as e:
                logger.warning(f"Could not fetch S&P 500 stocks: {e}")

            # Strategy 2: Fetch by sector
            try:
                logger.info("Adding stocks from various sectors...")
                sector_stocks = {
                    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'INTC', 'AMD', 'QCOM', 'CSCO', 'CRM',
                                   'ADBE', 'INTU', 'IBM', 'ORCL'],
                    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'LLY', 'MRK', 'AMGN', 'TMO', 'ABT', 'BMY', 'GILD',
                                   'BIIB', 'REGN', 'ISRG'],
                    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'BLK', 'AXP', 'USB', 'PNC', 'COF', 'FIS'],
                    'Consumer': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX', 'HD', 'TGT', 'LOW', 'TSCO'],
                    'Industrial': ['BA', 'CAT', 'RTX', 'LMT', 'GE', 'HON', 'MMM', 'AZO', 'LUV', 'DAL'],
                    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'MRO', 'OXY'],
                    'Materials': ['NEM', 'FCX', 'ALB', 'LIN', 'APD', 'DOW', 'PKG'],
                    'Utilities': ['NEE', 'DUK', 'SO', 'EXC', 'AEP', 'DTE', 'XEL'],
                    'Communication': ['VZ', 'T', 'CMCSA', 'TMUS', 'CHTR', 'DISH'],
                    'SaaS/Cloud': ['CRWD', 'DDOG', 'ZM', 'NET', 'SNOW', 'OKTA', 'TWLO', 'SPLK', 'AYX', 'CrowdStrike'],
                    'FinTech': ['PYPL', 'SQ', 'COIN', 'UPST', 'SOFI'],
                    'E-Commerce': ['SHOP', 'RBLX', 'DASH', 'UBER', 'EBAY', 'AMZN'],
                }

                for sector, tickers in sector_stocks.items():
                    symbols.extend(tickers)
                    logger.debug(f"Added {len(tickers)} stocks from {sector}")

            except Exception as e:
                logger.warning(f"Could not add sector stocks: {e}")

            # Remove duplicates and sort
            symbols = sorted(list(set(symbols)))
            logger.info(f"Total unique symbols for weekly analysis: {len(symbols)} stocks")

            # Validate symbols (optional but recommended for weekly)
            valid_symbols = []
            checked = 0
            for symbol in symbols:
                try:
                    checked += 1
                    if checked % 30 == 0:
                        logger.info(f"Validating stocks... {checked}/{len(symbols)}")

                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    if info and ('longName' in info or 'regularMarketPrice' in info):
                        valid_symbols.append(symbol)
                except:
                    logger.debug(f"Skipping invalid symbol: {symbol}")
                    continue

            symbols = valid_symbols if valid_symbols else symbols
            logger.info(f"Validated {len(symbols)} symbols for weekly analysis - ready to analyze")

            return symbols

        except Exception as e:
            logger.error(f"Error fetching dynamic stocks: {e}")
            # Fallback to curated list
            logger.info("Using fallback stock list for weekly analysis...")
            return self._get_fallback_stock_list()

    def _get_fallback_stock_list(self) -> list:
        """Fallback stock list if dynamic fetch fails"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B',
            'JNJ', 'V', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'INTC', 'PYPL',
            'CRM', 'NFLX', 'PEP', 'CSCO', 'AMD', 'LLY', 'ABBV', 'IBM',
            'ORCL', 'COST', 'MCD', 'ADBE', 'ACN', 'NKE', 'QCOM', 'KO',
            'AVGO', 'CCI', 'TXN', 'TMUS', 'HON', 'MU', 'INTU', 'UPS',
            'BA', 'CAT', 'RTX', 'LMT', 'GE', 'MMM', 'XOM', 'CVX',
            'SLB', 'COP', 'EOG', 'MPC', 'PSX', 'NEE', 'DUK', 'SO',
            'AEP', 'EXC', 'UNH', 'PFE', 'AMGN', 'TMO', 'MRK', 'ABT',
            'BMY', 'CRWD', 'ZM', 'DDOG', 'NET', 'SNOW', 'OKTA', 'TWLO',
        ]

    def _get_available_model(self) -> str:
        """
        Get an available model from Groq API.
        Uses the latest available Groq models.

        Returns:
            Available model ID
        """
        # Latest Groq models (mixtral-8x7b-32768 is decommissioned)
        # Use these in order of preference
        preferred_models = [
            "llama-3.1-70b-versatile",  # Latest and most capable
            "llama-3.3-70b-versatile",  # Alternative latest
            "llama-3.1-8b-instant",  # Faster fallback
        ]

        for model in preferred_models:
            try:
                # Quick test to see if model is available
                test_response = self.client.chat.completions.create(
                    model=model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": "OK"}]
                )
                logger.info(f"Using model: {model}")
                return model
            except Exception as e:
                error_msg = str(e)
                # Skip decommissioned models
                if "decommissioned" in error_msg or "not supported" in error_msg:
                    logger.debug(f"Model {model} decommissioned, trying next")
                else:
                    logger.debug(f"Model {model} not available: {error_msg}")
                continue

        # Default to llama-3.1-70b which should always be available
        logger.warning("Could not auto-detect model, using default: llama-3.1-70b-versatile")
        return "llama-3.1-70b-versatile"

    def generate_report(self) -> str:
        """
        Generate a comprehensive report using Groq API.

        Returns:
            Report text generated by Groq API
        """
        if not self.stocks_data:
            logger.warning("No stock data available. Fetch data first.")
            return "No stock data available."

        logger.info("Generating report using Groq API...")

        # Prepare data for Groq
        stocks_summary = json.dumps(self.stocks_data, indent=2)

        prompt = f"""Based on the following stock data for stocks that dropped more than 10% 
in the last 24 hours (with market cap > $1B), provide a comprehensive analysis report:

STOCK DATA:
{stocks_summary}

Please provide:
1. Executive Summary: Overview of the market movement
2. Top 5 Most Impacted Stocks: Detailed analysis of the biggest losers
3. Sector Analysis: Which sectors are most affected
4. Risk Assessment: Potential causes and market implications
5. Investment Insights: Opportunities and warnings
6. Key Takeaways: What investors should watch

Format the report professionally with clear sections and bullet points where appropriate."""

        # Auto-detect best available model
        model = self._get_available_model()

        chat_completion = self.client.chat.completions.create(
            model=model,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        report = chat_completion.choices[0].message.content

        return report

    def save_report(self, report: str, filename: str = "stock_report.txt") -> str:
        """
        Save the report to a file.

        Args:
            report: The report text to save
            filename: Output filename

        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(filename, 'w') as f:
            f.write(f"Stock Report - Generated {timestamp}\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)
            f.write("\n\n" + "=" * 80 + "\n")
            f.write(f"Data Summary: {len(self.stocks_data)} stocks analyzed\n")

        logger.info(f"Report saved to {filename}")
        return filename

    def save_html_report(self, report: str = "", filename: str = "stock_report.html") -> str:
        """
        Save the report as a beautifully styled HTML file.

        Args:
            report: The report text to save (not used, kept for compatibility)
            filename: Output filename

        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format stock data as HTML table
        stocks_html = self._generate_stocks_table_html()

        # Generate reasons for stock drops
        stocks_reasons_html = self._generate_stocks_reasons_html()

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Market Analysis Report - Weekly</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
            border-bottom: 5px solid #764ba2;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}

        .header p {{
            font-size: 1.1em;
            opacity: 0.95;
        }}

        .timestamp {{
            font-size: 0.9em;
            opacity: 0.85;
            margin-top: 10px;
        }}

        .content {{
            padding: 40px 30px;
        }}

        .summary-box {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(245, 87, 108, 0.3);
        }}

        .summary-box h2 {{
            font-size: 1.5em;
            margin-bottom: 10px;
        }}

        .summary-box p {{
            opacity: 0.95;
            line-height: 1.8;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }}

        .stat-card .number {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .stat-card .label {{
            font-size: 0.95em;
            opacity: 0.9;
        }}

        .stocks-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }}

        .stocks-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}

        .stocks-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}

        .stocks-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}

        .stocks-table tr:hover {{
            background-color: #f0f0ff;
            transition: background-color 0.2s;
        }}

        .symbol {{
            font-weight: bold;
            color: #667eea;
            font-size: 1.1em;
        }}

        .rank {{
            font-weight: bold;
            color: #667eea;
            font-size: 0.95em;
        }}

        .negative {{
            color: #f5576c;
            font-weight: bold;
        }}

        .positive {{
            color: #4caf50;
            font-weight: bold;
        }}

        .section {{
            margin-bottom: 35px;
        }}

        .section h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}

        .section h3 {{
            color: #764ba2;
            font-size: 1.2em;
            margin-top: 20px;
            margin-bottom: 10px;
        }}

        .section p {{
            color: #555;
            margin-bottom: 10px;
            line-height: 1.8;
        }}

        .section ul {{
            margin-left: 20px;
            margin-bottom: 10px;
        }}

        .section li {{
            margin-bottom: 8px;
            color: #555;
        }}

        .highlight {{
            background: linear-gradient(120deg, #ffd89b 0%, #19547b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }}

        .risk-high {{
            background-color: #ffebee;
            border-left: 4px solid #f5576c;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}

        .risk-medium {{
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}

        .risk-low {{
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}

        .insight-box {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 5px 15px rgba(79, 172, 254, 0.3);
        }}

        .insight-box h4 {{
            margin-bottom: 10px;
            font-size: 1.1em;
        }}

        .footer {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px 30px;
            text-align: center;
            font-size: 0.9em;
            border-top: 5px solid #667eea;
        }}

        .footer p {{
            margin-bottom: 5px;
        }}

        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}

            .stats-grid {{
                grid-template-columns: 1fr;
            }}

            .stocks-table {{
                font-size: 0.85em;
                display: block;
                width: 100%;
            }}

            .stocks-table thead {{
                display: none;
            }}

            .stocks-table tbody {{
                display: block;
            }}

            .stocks-table tr {{
                display: block;
                margin-bottom: 15px;
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 10px;
                background: #f9f9f9;
            }}

            .stocks-table td {{
                display: grid;
                grid-template-columns: 100px 1fr;
                gap: 8px;
                padding: 8px 0;
                border-bottom: 1px solid #eee;
            }}

            .stocks-table td:last-child {{
                border-bottom: none;
            }}

            .stocks-table .rank {{
                grid-column: 1/3;
                font-weight: bold;
                color: #667eea;
                margin-bottom: 5px;
            }}

            .stocks-table .symbol {{
                grid-column: 1/3;
                font-weight: bold;
                color: #667eea;
                margin-bottom: 5px;
            }}

            .stocks-table td:nth-child(3)::before {{
                content: "Company: ";
                font-weight: bold;
                color: #667eea;
            }}

            .stocks-table td:nth-child(4)::before {{
                content: "Week Ago: ";
                font-weight: bold;
                color: #667eea;
            }}

            .stocks-table td:nth-child(5)::before {{
                content: "Current: ";
                font-weight: bold;
                color: #667eea;
            }}

            .stocks-table td:nth-child(6)::before {{
                content: "Change: ";
                font-weight: bold;
                color: #667eea;
            }}

            .stocks-table td:nth-child(7)::before {{
                content: "Market Cap: ";
                font-weight: bold;
                color: #667eea;
            }}

            .stocks-table td:nth-child(8)::before {{
                content: "Sector: ";
                font-weight: bold;
                color: #667eea;
            }}

            .reasons-container {{
                grid-template-columns: 1fr;
                gap: 15px;
            }}

            .reason-box {{
                padding: 15px;
                border-left-width: 4px;
            }}
        }}

        .report-text {{
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Courier New', monospace;
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            line-height: 1.7;
        }}

        .reasons-container {{
            display: block;
            margin-bottom: 30px;
            width: 100%;
            max-width: 100%;
            padding: 0;
        }}

        .reason-box {{
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-left: 5px solid #667eea;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 3px 15px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            break-inside: avoid;
            word-break: break-word;
            overflow-wrap: break-word;
            max-width: 100%;
        }}

        .reason-box:hover {{
            box-shadow: 0 6px 25px rgba(102, 126, 234, 0.25);
            transform: translateY(-3px);
            border-left-color: #764ba2;
        }}

        .reason-symbol {{
            font-weight: bold;
            color: #667eea;
            font-size: 1.2em;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 2px solid #e0e0e0;
        }}

        .reason-text {{
            color: #444;
            font-size: 0.96em;
            line-height: 1.8;
            white-space: normal;
            word-wrap: break-word;
            overflow-wrap: break-word;
            word-break: break-word;
        }}

        .reason-text br {{
            margin-bottom: 6px;
        }}

        .reason-box-large {{
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-left: 5px solid #667eea;
            padding: 25px;
            margin: 0;
            border-radius: 10px;
            box-shadow: 0 3px 15px rgba(0, 0, 0, 0.1);
            word-break: break-word;
            overflow-wrap: break-word;
            max-width: 100%;
            box-sizing: border-box;
        }}

        .reason-text-large {{
            color: #444;
            font-size: 0.95em;
            line-height: 1.8;
            padding: 0;
            margin: 0;
            word-wrap: break-word;
            overflow-wrap: break-word;
            word-break: break-word;
            hyphens: manual;
        }}

        .reason-text-large p {{
            margin: 0 0 12px 0;
            padding: 0;
            text-align: left;
            word-wrap: break-word;
            overflow-wrap: break-word;
            word-break: break-word;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 Stock Market Analysis Report - Weekly</h1>
            <p>Analysis of All Stocks with >10% Drop in Last 7 Days (Market Cap > $1B)</p>
            <div class="timestamp">Generated: {timestamp}</div>
        </div>

        <div class="content">
            <!-- Summary Statistics -->
            <div class="summary-box">
                <h2>Market Overview</h2>
                <p>This report analyzes the top {len(self.stocks_data)} stocks that experienced significant price drops in the last 7 days (weekly). These are large-cap companies with market capitalizations exceeding $1 billion.</p>
            </div>

            <!-- Statistics Cards -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="number">{len(self.stocks_data)}</div>
                    <div class="label">Stocks Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="number">{min([abs(s['change_percent']) for s in self.stocks_data]) if self.stocks_data else 0:.1f}%</div>
                    <div class="label">Minimum Drop</div>
                </div>
                <div class="stat-card">
                    <div class="number">{max([abs(s['change_percent']) for s in self.stocks_data]) if self.stocks_data else 0:.1f}%</div>
                    <div class="label">Maximum Drop</div>
                </div>
            </div>

            <!-- Stocks Table -->
            <div class="section">
                <h2>Top Declining Stocks</h2>
                {stocks_html}
            </div>

            <!-- Why Each Stock Dropped -->
            <div class="section">
                <h2>Why Each Stock Dropped</h2>
                {stocks_reasons_html}
            </div>

            <!-- Disclaimer -->
            <div class="risk-high">
                <strong>⚠️ Disclaimer:</strong> This analysis is for informational and educational purposes only. It does not constitute financial advice. Please conduct your own research and consult with a financial advisor before making any investment decisions.
            </div>
        </div>

        <div class="footer">
            <p><strong>Stock Market Analysis Report</strong></p>
            <p>Generated on {timestamp}</p>
            <p>Data provided by yfinance • Analysis powered by Groq API</p>
        </div>
    </div>
</body>
</html>
"""

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML report saved to {filename}")
        return filename

    def _generate_stocks_table_html(self) -> str:
        """
        Generate HTML table for stocks data.
        Shows ALL stocks found with >10% drop in 5 days.

        Returns:
            HTML table string
        """
        if not self.stocks_data:
            return "<p>No stock data available</p>"

        # Show all stocks found, but cap at 50 for reasonable page size
        stocks_to_display = self.stocks_data[:50] if len(self.stocks_data) > 50 else self.stocks_data

        table_rows = ""
        for idx, stock in enumerate(stocks_to_display, 1):
            market_cap_billions = stock['market_cap'] / 1_000_000_000
            price_week_ago = stock.get('price_5days_ago', 0)  # Price from 5 days ago
            current_price = stock['price']

            table_rows += f"""
            <tr>
                <td><span class="rank">#{idx}</span></td>
                <td><span class="symbol">{stock['symbol']}</span></td>
                <td>{stock['company_name']}</td>
                <td>${price_week_ago:.2f}</td>
                <td>${current_price:.2f}</td>
                <td><span class="negative">{stock['change_percent']:.2f}%</span></td>
                <td>${market_cap_billions:.1f}B</td>
                <td>{stock['sector']}</td>
            </tr>
            """

        return f"""
        <table class="stocks-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Symbol</th>
                    <th>Company Name</th>
                    <th>Week Ago Price</th>
                    <th>Current Price</th>
                    <th>7-Day Change</th>
                    <th>Market Cap</th>
                    <th>Sector</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """

    def generate_stock_drop_reasons(self) -> str:
        """
        Generate detailed reasons why each stock dropped using Groq API.
        Analyzes ALL stocks found (or max 30 to avoid token limits).

        Returns:
            AI-generated detailed reasons for stock drops
        """
        if not self.stocks_data:
            return "No data available"

        logger.info("Generating detailed reasons for stock drops...")

        # Use all stocks found, but cap at 30 to avoid excessive API tokens
        stocks_to_analyze = self.stocks_data[:30] if len(self.stocks_data) > 30 else self.stocks_data

        # Format stock data for prompt
        stocks_info = "\n".join([
            f"• {s['symbol']} ({s['company_name']}): {s['change_percent']:.2f}% drop, "
            f"Sector: {s['sector']}, Market Cap: ${s['market_cap'] / 1e9:.1f}B, Price: ${s['price']:.2f}"
            for s in stocks_to_analyze
        ])

        prompt = f"""You are an expert financial analyst with deep knowledge of equity markets, company fundamentals, and macroeconomics. 

For EACH stock below that dropped >10% in the last 5 days, provide EXTREMELY DETAILED reasons (6-8 lines minimum, 150-200 words) explaining precisely WHY the stock dropped.

For each stock, MUST cover ALL of the following:

1. SPECIFIC MARKET CATALYSTS (News events, earnings, guidance changes, announcements):
   - What specific event triggered the drop?
   - When did it occur? What was the impact?

2. COMPANY-SPECIFIC FUNDAMENTALS (Operational issues, financial problems, strategy changes):
   - What operational or financial problems emerged?
   - What do earnings/guidance changes mean for business?
   - Any management changes or executive turnover?

3. SECTOR DYNAMICS (Industry trends, competitive pressures, regulatory concerns):
   - What sector headwinds are affecting this company?
   - Who are key competitors gaining market share?
   - Any regulatory or compliance issues?

4. MACROECONOMIC FACTORS (Interest rates, inflation, recession risks, consumer/business spending):
   - How do macro conditions affect this company?
   - What economic pressures impact their business?
   - Consumer spending trends affecting revenue?

5. TECHNICAL ANALYSIS (Chart patterns, key support breaks, momentum indicators):
   - What key support levels broke?
   - What do chart patterns suggest about momentum?
   - Are algorithmic traders involved?

6. VALUATION & RELATIVE STRENGTH (P/E multiples, growth rates, peer comparisons):
   - Was the stock overvalued before the drop?
   - How does it compare to sector peers?
   - What's the fair value estimate?

STOCKS (All stocks with >10% drop in 5 days, sorted by worst first):
{stocks_info}

Format your response EXACTLY as:

SYMBOL (Company Name) - [Change Percent]%
[3-4 line explanation covering: market catalyst, company issue, sector/macro impact, technical aspect]

SYMBOL (Company Name) - [Change Percent]%
[3-4 line explanation...]

CRITICAL REQUIREMENTS:
- Provide 3-4 line explanations for EVERY stock listed
- Be specific with numbers, percentages, and facts
- Cover: catalyst + fundamentals + sector/macro + technical
- Make it clear and actionable for investors
- Include all stocks shown above

Provide clear, detailed analysis for every stock."""

        # Get available model
        model = self._get_available_model()

        chat_completion = self.client.chat.completions.create(
            model=model,
            max_tokens=2048,  # Increased to handle detailed analysis for all stocks (Groq supports up to 4096)
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        reasons = chat_completion.choices[0].message.content
        return reasons

    def _generate_stocks_reasons_html(self) -> str:
        """
        Generate enhanced HTML for detailed stock drop reasons with premium styling.

        Returns:
            HTML formatted reasons with comprehensive analysis
        """
        reasons_text = self.generate_stock_drop_reasons()

        if not reasons_text or reasons_text == "No data available":
            return '<div class="reasons-container"><p>No reasons available</p></div>'

        # Escape HTML special characters
        reasons_text = reasons_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Normalize line endings
        reasons_text = reasons_text.replace('\r\n', '\n')

        # Parse the response to format each stock's analysis nicely
        lines = reasons_text.split('\n')

        html_content = '<div class="reasons-container">\n'
        html_content += '    <div class="detailed-analysis-section">\n'
        html_content += '        <h2 style="color: #2c3e50; border-bottom: 3px solid #667eea; padding-bottom: 15px; margin-top: 30px; margin-bottom: 20px; font-size: 26px;">📊 Comprehensive Analysis: Why Each Stock Dropped</h2>\n'
        html_content += '        <p style="color: #555; font-size: 14px; margin-bottom: 20px; font-style: italic;">Detailed breakdown of market catalysts, fundamental issues, sector trends, macro factors, technical levels, and valuation concerns</p>\n'

        current_stock = None
        stock_details = []
        stock_count = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a stock header line (SYMBOL (Company Name) - X.XX%)
            if line and ' - ' in line and '%' in line and '(' in line and ')' in line:
                # Save previous stock if exists
                if stock_details:
                    stock_count += 1
                    # Parse the percentage to determine color
                    try:
                        percent_str = current_stock.split(' - ')[-1].replace('%', '').strip()
                        percent_val = float(percent_str)
                        color = '#e74c3c' if percent_val < -15 else '#e67e22'  # Darker red for bigger drops
                    except:
                        color = '#e74c3c'

                    html_content += f'        <div class="stock-analysis-premium">\n'
                    html_content += f'            <div class="stock-header-premium" style="border-left-color: {color}">\n'
                    html_content += f'                <span class="stock-symbol">{current_stock}</span>\n'
                    html_content += f'            </div>\n'
                    html_content += f'            <div class="stock-reasons-premium">\n'
                    for detail in stock_details:
                        html_content += f'                <p class="reason-detail">{detail}</p>\n'
                    html_content += f'            </div>\n'
                    html_content += f'        </div>\n'
                    stock_details = []

                # Start new stock
                current_stock = line
            else:
                # This is a detail line for current stock
                if current_stock:
                    stock_details.append(line)

        # Don't forget the last stock
        if current_stock and stock_details:
            stock_count += 1
            try:
                percent_str = current_stock.split(' - ')[-1].replace('%', '').strip()
                percent_val = float(percent_str)
                color = '#e74c3c' if percent_val < -15 else '#e67e22'
            except:
                color = '#e74c3c'

            html_content += f'        <div class="stock-analysis-premium">\n'
            html_content += f'            <div class="stock-header-premium" style="border-left-color: {color}">\n'
            html_content += f'                <span class="stock-symbol">{current_stock}</span>\n'
            html_content += f'            </div>\n'
            html_content += f'            <div class="stock-reasons-premium">\n'
            for detail in stock_details:
                html_content += f'                <p class="reason-detail">{detail}</p>\n'
            html_content += f'            </div>\n'
            html_content += f'        </div>\n'

        html_content += f'        <div class="analysis-footer">Total Stocks Analyzed: {stock_count}</div>\n'
        html_content += '    </div>\n'
        html_content += '</div>\n'

        # Add enhanced CSS for premium styling
        css = '''<style>
            .detailed-analysis-section {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 30px;
                border-radius: 12px;
                margin: 30px 0;
                border: 1px solid #dee2e6;
            }

            .stock-analysis-premium {
                background: white;
                border-left: 6px solid #667eea;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                transition: transform 0.2s, box-shadow 0.2s;
            }

            .stock-analysis-premium:hover {
                transform: translateX(5px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
            }

            .stock-header-premium {
                border-left: 4px solid #667eea;
                padding-left: 15px;
                margin-bottom: 15px;
                padding-bottom: 12px;
                border-bottom: 2px solid #f0f0f0;
            }

            .stock-symbol {
                font-weight: 700;
                color: #2c3e50;
                font-size: 18px;
                letter-spacing: 0.5px;
            }

            .stock-reasons-premium {
                color: #333;
                line-height: 1.8;
                font-size: 15px;
            }

            .reason-detail {
                margin: 10px 0;
                padding: 8px 0;
                border-bottom: 1px solid #f5f5f5;
            }

            .reason-detail:last-child {
                border-bottom: none;
            }

            .analysis-footer {
                text-align: center;
                margin-top: 25px;
                padding-top: 20px;
                border-top: 2px solid #dee2e6;
                color: #666;
                font-size: 14px;
                font-weight: 500;
            }

            .reasons-container {
                margin: 20px 0;
            }
        </style>'''

        return css + html_content

    def export_json(self, filename: str = "stocks_data.json") -> str:
        """
        Export raw stock data as JSON.

        Args:
            filename: Output filename

        Returns:
            Path to the saved file
        """
        with open(filename, 'w') as f:
            json.dump(self.stocks_data, f, indent=2)

        logger.info(f"Stock data exported to {filename}")
        return filename


def main():
    """Main execution function."""
    try:
        # Initialize generator
        generator = StockReportGenerator()

        # Fetch stocks data
        stocks = generator.fetch_stocks_data()

        if not stocks:
            logger.warning("No stocks found matching criteria")
            return

        # Generate HTML report with reasons (save as weekly version)
        generator.save_html_report("", "stock_report_weekly.html")
        generator.export_json("stocks_data.json")

        # Print summary to console
        print("\n" + "=" * 80)
        print("✅ STOCK REPORT GENERATED SUCCESSFULLY")
        print("=" * 80 + "\n")
        print(f"📊 Found {len(stocks)} stocks matching criteria")
        print(f"🌐 HTML report saved to: stock_report_weekly.html")
        print(f"📁 Stock data exported to: stocks_data.json")
        print("\n" + "=" * 80)
        print("Open 'stock_report_weekly.html' in your browser to view the report!")
        print("=" * 80 + "\n")

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()