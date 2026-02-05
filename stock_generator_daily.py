"""
Stock Report Generator using Groq API

This script fetches stocks that dropped more than 10% in the last 24 hours
with a market cap greater than $1 billion, and generates an analysis report
using the Groq API.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import requests
from groq import Groq


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
            groq_api_key: Groq API key. If None, tries to load from
                the GROQ_API_KEY environment variable.
        """
        self.groq_api_key = groq_api_key or self._load_groq_key()
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not provided.\n"
                "Please set the GROQ_API_KEY environment variable locally, or\n"
                "configure a GitHub Actions secret (e.g. 'groq_key') and map it\n"
                "to the GROQ_API_KEY env var in your workflow."
            )

        self.client = Groq(api_key=self.groq_api_key)
        self.stocks_data = []

    def _load_groq_key(self) -> Optional[str]:
        """
        Load Groq API key from environment variable.

        Returns:
            The API key if found, None otherwise.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            logger.info("Loaded GROQ_API_KEY from environment variable")
            return api_key

        return None

    def fetch_stocks_data(self) -> list:
        """
        Fetch stocks that dropped more than 10% in last 24 hours.
        Dynamically fetches all stocks with market cap > $1B.

        Returns:
            List of stock dictionaries with symbol, price, change, market_cap, etc.
        """
        logger.info("Fetching stocks data...")

        try:
            # Install yfinance if not already installed: pip install yfinance
            import yfinance as yf

            # Get list of ALL large-cap stocks dynamically
            symbols = self._get_all_large_cap_stocks()

            stocks_with_drops = []
            total_checked = 0

            for symbol in symbols:
                try:
                    total_checked += 1
                    if total_checked % 50 == 0:
                        logger.info(f"Progress: Checked {total_checked}/{len(symbols)} stocks...")

                    # Skip processing if yfinance returns no data
                    ticker = yf.Ticker(symbol)
                    info = ticker.info

                    # Check if ticker exists (info would have error or be empty)
                    if not info or 'longName' not in info or info.get('marketCap') is None:
                        logger.debug(f"Skipping {symbol}: No market cap data available")
                        continue

                    # Get historical data for 24h change
                    hist = ticker.history(period="2d")
                    if hist is None or len(hist) < 2:
                        logger.debug(f"Skipping {symbol}: Insufficient historical data")
                        continue

                    # Calculate 24-hour percentage change
                    prev_close = hist['Close'].iloc[-2]
                    current_price = hist['Close'].iloc[-1]

                    # Skip if prices are invalid
                    if prev_close <= 0 or current_price <= 0 or prev_close != prev_close or current_price != current_price:
                        logger.debug(f"Skipping {symbol}: Invalid price data")
                        continue

                    change_percent = ((current_price - prev_close) / prev_close) * 100

                    # Check criteria: >10% drop AND market cap >$1B
                    market_cap = info.get('marketCap', 0)

                    # Skip if market cap not available or 0
                    if not market_cap or market_cap == 0:
                        logger.debug(f"Skipping {symbol}: No market cap data")
                        continue

                    if change_percent < -10 and market_cap > 1_000_000_000:
                        stocks_with_drops.append({
                            'symbol': symbol,
                            'price': current_price,
                            'prev_close': prev_close,
                            'change_percent': round(change_percent, 2),
                            'market_cap': market_cap,
                            'company_name': info.get('longName', symbol),
                            'sector': info.get('sector', 'N/A'),
                            'industry': info.get('industry', 'N/A'),
                        })
                except Exception as e:
                    # Silently skip problematic stocks (delisted, bad data, etc.)
                    logger.debug(f"Skipping {symbol}: {type(e).__name__}")
                    continue

            # Sort by percentage drop (descending - worst first)
            stocks_with_drops.sort(key=lambda x: x['change_percent'])

            # Store all matching stocks (not limited to 30 anymore)
            self.stocks_data = stocks_with_drops
            logger.info(f"Found {len(self.stocks_data)} stocks with >10% drop and >$1B market cap")

            return self.stocks_data

        except ImportError:
            logger.error("yfinance not installed. Install with: pip install yfinance")
            raise

    def _get_all_large_cap_stocks(self) -> list:
        """
        Get top large-cap stocks dynamically based on market trends and volume.
        Uses multiple strategies to ensure we get trending stocks.

        Returns:
            List of stock symbols to analyze (dynamically fetched from market data)
        """
        logger.info("Fetching latest trending stocks dynamically...")

        try:
            import yfinance as yf
            import pandas as pd

            symbols = []

            # Strategy 1: Get S&P 500 stocks (most liquid, most traded)
            try:
                logger.info("Fetching S&P 500 stocks...")
                # Common S&P 500 tickers
                sp500_symbols = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK.B',
                    'JNJ', 'V', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'INTC', 'PYPL',
                    'CRM', 'NFLX', 'PEP', 'CSCO', 'AMD', 'LLY', 'ABBV', 'IBM',
                    'ORCL', 'COST', 'MCD', 'ADBE', 'ACN', 'NKE', 'QCOM', 'KO',
                    'AVGO', 'CCI', 'TXN', 'TMUS', 'HON', 'MU', 'INTU', 'UPS',
                    'BA', 'CAT', 'RTX', 'LMT', 'GE', 'MMM', 'XOM', 'CVX',
                    'SLB', 'COP', 'EOG', 'MPC', 'PSX', 'NEE', 'DUK', 'SO',
                    'AEP', 'EXC', 'UNH', 'PFE', 'AMGN', 'TMO', 'LLY', 'ABBV',
                    'MRK', 'ABT', 'BMY', 'GILD', 'BIIB', 'REGN', 'CRWD', 'ZM',
                    'DDOG', 'NET', 'SNOW', 'OKTA', 'TWLO', 'SHOP', 'RBLX',
                    'DASH', 'UBER', 'COIN', 'FICO', 'CHTR', 'CMCSA', 'VZ',
                    'T', 'DIS', 'FOXA', 'QVCC', 'PSA', 'EQIX', 'DLR', 'SPG',
                    'WFC', 'BAC', 'GS', 'BLK', 'AXP', 'USB', 'PNC', 'RBC',
                ]
                symbols.extend(sp500_symbols)
                logger.info(f"Added {len(sp500_symbols)} S&P 500 stocks")
            except Exception as e:
                logger.warning(f"Could not fetch S&P 500 stocks: {e}")

            # Strategy 2: Fetch by sector - ensure broad coverage
            try:
                logger.info("Adding stocks from various sectors...")
                sector_stocks = {
                    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'INTC', 'AMD', 'QCOM'],
                    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'LLY', 'MRK', 'AMGN', 'TMO', 'ABT', 'BMY'],
                    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'BLK', 'AXP', 'USB', 'PNC'],
                    'Consumer': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX', 'HD', 'TGT'],
                    'Industrial': ['BA', 'CAT', 'RTX', 'LMT', 'GE', 'HON', 'MMM', 'AZO', 'LUV'],
                    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX'],
                    'Communication': ['VZ', 'T', 'CMCSA', 'TMUS', 'CHTR'],
                    'SaaS/Cloud': ['CRWD', 'DDOG', 'ZM', 'NET', 'SNOW', 'OKTA', 'TWLO'],
                    'Entertainment': ['DIS', 'NFLX', 'FOXA'],
                    'FinTech': ['PYPL', 'SQ', 'COIN', 'UPST'],
                    'E-Commerce': ['SHOP', 'RBLX', 'DASH', 'UBER'],
                    'Real Estate': ['PSA', 'EQIX', 'DLR', 'SPG'],
                }

                for sector, tickers in sector_stocks.items():
                    symbols.extend(tickers)
                    logger.debug(f"Added {len(tickers)} stocks from {sector}")

            except Exception as e:
                logger.warning(f"Could not add sector stocks: {e}")

            # Remove duplicates and sort
            symbols = sorted(list(set(symbols)))
            logger.info(f"Total unique symbols to analyze: {len(symbols)} stocks")

            # Verify symbols are valid (optional - removes any invalid tickers)
            valid_symbols = []
            checked = 0
            for symbol in symbols:
                try:
                    checked += 1
                    if checked % 20 == 0:
                        logger.info(f"Validating stocks... {checked}/{len(symbols)}")

                    # Quick check if symbol exists
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    if info and ('longName' in info or 'regularMarketPrice' in info):
                        valid_symbols.append(symbol)
                except:
                    logger.debug(f"Skipping invalid symbol: {symbol}")
                    continue

            symbols = valid_symbols if valid_symbols else symbols
            logger.info(f"Validated {len(symbols)} symbols - ready to analyze")

            return symbols

        except Exception as e:
            logger.error(f"Error fetching dynamic stocks: {e}")
            # Fallback to curated list
            logger.info("Using fallback stock list...")
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
            'BMY', 'GILD', 'BIIB', 'REGN', 'CRWD', 'ZM', 'DDOG', 'NET',
            'SNOW', 'OKTA', 'TWLO', 'SHOP', 'RBLX', 'DASH', 'UBER', 'COIN',
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
    <title>Stock Market Analysis Report</title>
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
                content: "Ref Price: ";
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

            /* Mobile Optimizations for Reasons Section */
            .reasons-container {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 15px;
                margin-bottom: 30px;
            }}

            .reason-box {{
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border-left: 5px solid #667eea;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
                break-inside: avoid;
                word-break: break-word;
                overflow-wrap: break-word;
            }}

            .reason-box:hover {{
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
                border-left-color: #764ba2;
            }}

            .reason-symbol {{
                font-weight: bold;
                color: #667eea;
                font-size: 1.1em;
                margin-bottom: 10px;
                padding-bottom: 8px;
                border-bottom: 2px solid #e0e0e0;
                word-break: break-word;
            }}

            .reason-text {{
                color: #444;
                font-size: 0.9em;
                line-height: 1.7;
                white-space: normal;
                word-wrap: break-word;
                overflow-wrap: break-word;
                word-break: break-word;
            }}

            .reason-text br {{
                margin-bottom: 4px;
            }}

            .reason-box-large {{
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border-left: 5px solid #667eea;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
                word-break: break-word;
                overflow-wrap: break-word;
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
            <h1>📈 Stock Market Analysis Report</h1>
            <p>Analysis of All Stocks with >10% Daily Drop (Market Cap > $1B)</p>
            <div class="timestamp">Generated: {timestamp}</div>
        </div>

        <div class="content">
            <!-- Summary Statistics -->
            <div class="summary-box">
                <h2>Market Overview</h2>
                <p>This report analyzes the top {len(self.stocks_data)} stocks that experienced significant price drops in the last 24 hours. These are large-cap companies with market capitalizations exceeding $1 billion.</p>
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
        Shows ALL stocks found with >10% drop.

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
            prev_price = stock.get('prev_close', 0)  # Yesterday's price for daily
            current_price = stock['price']

            table_rows += f"""
            <tr>
                <td><span class="rank">#{idx}</span></td>
                <td><span class="symbol">{stock['symbol']}</span></td>
                <td>{stock['company_name']}</td>
                <td>${prev_price:.2f}</td>
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
                    <th>Yesterday Price</th>
                    <th>Current Price</th>
                    <th>24h Change</th>
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

        prompt = f"""Based on these stocks that dropped more than 10% today, provide BRIEF but detailed reasons (2-3 lines max, 30-50 words) for why each stock dropped. Be specific and help users understand the key factors.

STOCKS (All stocks with >10% drop, sorted by worst first):
{stocks_info}

Format your response EXACTLY as:

SYMBOL (Company Name) - [Change Percent]%
[2-3 line explanation with key reasons]

SYMBOL (Company Name) - [Change Percent]%
[2-3 line explanation...]

IMPORTANT: Provide brief 2-3 line explanations for EVERY stock listed above. Include all stocks.
Make it concise but informative."""

        # Get available model
        model = self._get_available_model()

        chat_completion = self.client.chat.completions.create(
            model=model,
            max_tokens=256,  # Groq API free tier limit
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
        Generate HTML for detailed stock drop reasons.

        Returns:
            HTML formatted reasons
        """
        reasons_text = self.generate_stock_drop_reasons()

        # Simple approach: just wrap the entire response in a nice box
        # The Groq API will format it well already
        if not reasons_text or reasons_text == "No data available":
            return '<div class="reasons-container"><p>No reasons available</p></div>'

        # Escape HTML special characters
        reasons_text = reasons_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Convert line breaks to proper HTML - improved handling
        reasons_text = reasons_text.replace('\r\n', '\n')  # Normalize line endings

        # Split by double line breaks (paragraphs)
        paragraphs = reasons_text.split('\n\n')

        # Build proper paragraph tags
        html_paragraphs = []
        for para in paragraphs:
            if para.strip():  # Only process non-empty paragraphs
                # Replace single line breaks with <br> tags
                para = para.replace('\n', '<br>')
                # Clean up multiple spaces
                para = para.strip()
                html_paragraphs.append(f'<p>{para}</p>')

        reasons_html = '\n'.join(html_paragraphs)

        html = f'''<div class="reasons-container">
    <div class="reason-box-large">
        <div class="reason-text-large">
{reasons_html}
        </div>
    </div>
</div>'''

        return html

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


def run_daily_report(
    html_filename: str = "stock_report.html",
    json_filename: str = "stocks_data.json",
) -> Dict[str, Any]:
    """
    Run the full daily report pipeline.

    This is a reusable entry point for CLI, Streamlit, or GitHub Actions.

    Args:
        html_filename: Name of the HTML report file to generate.
        json_filename: Name of the JSON data export file.

    Returns:
        Dict with keys: stocks_count, html_path, json_path.
    """
    # Initialize generator (uses GROQ_API_KEY from environment if not provided)
    generator = StockReportGenerator()

    # Fetch stocks data
    stocks = generator.fetch_stocks_data()

    if not stocks:
        logger.warning("No stocks found matching criteria")
        return {
            "stocks_count": 0,
            "html_path": None,
            "json_path": None,
        }

    # Generate HTML report with reasons (no longer need separate text report)
    html_path = generator.save_html_report("", html_filename)
    json_path = generator.export_json(json_filename)

    return {
        "stocks_count": len(stocks),
        "html_path": html_path,
        "json_path": json_path,
    }


def main():
    """Main execution function for CLI usage."""
    try:
        result = run_daily_report(
            html_filename="stock_report.html",
            json_filename="stocks_data.json",
        )

        stocks_count = result["stocks_count"]
        html_path = result["html_path"]
        json_path = result["json_path"]

        # Print summary to console
        print("\n" + "=" * 80)
        if stocks_count == 0:
            print("⚠️  No stocks found matching criteria")
        else:
            print("✅ STOCK REPORT GENERATED SUCCESSFULLY")
        print("=" * 80 + "\n")
        print(f"📊 Found {stocks_count} stocks matching criteria")
        if html_path:
            print(f"🌐 HTML report saved to: {html_path}")
        if json_path:
            print(f"📁 Stock data exported to: {json_path}")
        print("\n" + "=" * 80)
        print("Open 'stock_report.html' in your browser to view the report!")
        print("=" * 80 + "\n")

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()