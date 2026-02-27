"""
Monthly Stock Report Generator using Groq API

This module reuses the weekly report pipeline but changes the data
collection window to approximately one month (~22 trading days).
It looks for large-cap stocks that have dropped more than 10% over
that period and generates an analysis report using the Groq API.
"""

import logging
from typing import Dict, Any

from stock_generator_weekly import StockReportGenerator as WeeklyStockReportGenerator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MonthlyStockReportGenerator(WeeklyStockReportGenerator):
    """Generate monthly stock reports using market data and Groq API."""

    def fetch_stocks_data(self) -> list:
        """
        Fetch stocks that dropped more than 10% over roughly the last month.

        Uses the same large-cap universe as the weekly generator but looks
        at a longer window (~22 trading days).

        Returns:
            List of stock dictionaries with symbol, price, change, market_cap, etc.
        """
        logger.info("Fetching stocks data (1-month timeframe)...")

        try:
            import yfinance as yf

            symbols = self._get_all_large_cap_stocks()

            stocks_with_drops = []
            total_checked = 0

            for symbol in symbols:
                try:
                    total_checked += 1
                    if total_checked % 50 == 0:
                        logger.info(
                            f"Progress: Checked {total_checked}/{len(symbols)} stocks..."
                        )

                    ticker = yf.Ticker(symbol)
                    info = ticker.info

                    # Check if ticker exists
                    if (
                        not info
                        or "longName" not in info
                        or info.get("marketCap") is None
                    ):
                        logger.debug(
                            f"Skipping {symbol}: No market cap data available"
                        )
                        continue

                    # Get historical data for ~1 month change
                    # Fetch 40 days to ensure we have at least ~22 trading days
                    hist = ticker.history(period="40d")
                    if hist is None or len(hist) < 22:
                        logger.debug(
                            f"Skipping {symbol}: Insufficient historical data "
                            f"(need 22+ trading days, have {len(hist) if hist is not None else 0})"
                        )
                        continue

                    closing_prices = hist["Close"]

                    # Ensure we have enough daily closes
                    if len(closing_prices) < 22:
                        logger.debug(
                            f"Skipping {symbol}: Not enough trading days "
                            f"(need 22+, have {len(closing_prices)})"
                        )
                        continue

                    # Price from ~1 month (22 trading days) ago
                    price_1mo_ago = closing_prices.iloc[-22]
                    # Current price (most recent)
                    current_price = closing_prices.iloc[-1]

                    # Skip if prices are invalid
                    if price_1mo_ago <= 0 or current_price <= 0:
                        logger.debug(f"Skipping {symbol}: Invalid price data")
                        continue

                    # NaN checks
                    if (
                        price_1mo_ago != price_1mo_ago
                        or current_price != current_price
                    ):
                        logger.debug(f"Skipping {symbol}: Invalid price data (NaN)")
                        continue

                    # Calculate percentage change over the month window
                    change_percent = (
                        (current_price - price_1mo_ago) / price_1mo_ago
                    ) * 100

                    # Check criteria: >10% drop AND market cap >$500M
                    market_cap = info.get("marketCap", 0)

                    if not market_cap or market_cap == 0:
                        logger.debug(f"Skipping {symbol}: No market cap data")
                        continue

                    if change_percent < -10 and market_cap > 500_000_000:
                        stocks_with_drops.append(
                            {
                                "symbol": symbol,
                                "price": current_price,
                                "price_1mo_ago": price_1mo_ago,
                                "change_percent": round(change_percent, 2),
                                "market_cap": market_cap,
                                "company_name": info.get("longName", symbol),
                                "sector": info.get("sector", "N/A"),
                                "industry": info.get("industry", "N/A"),
                            }
                        )
                except Exception as e:
                    # Silently skip problematic stocks (delisted, bad data, etc.)
                    logger.debug(f"Skipping {symbol}: {type(e).__name__}")
                    continue

            # Sort by percentage drop (worst first)
            stocks_with_drops.sort(key=lambda x: x["change_percent"])

            self.stocks_data = stocks_with_drops
            logger.info(
                "Found %d stocks with >10%% drop in last ~1 month and >$500M market cap",
                len(self.stocks_data),
            )

            return self.stocks_data

        except ImportError:
            logger.error("yfinance not installed. Install with: pip install yfinance")
            raise

    def _generate_stocks_table_html(self) -> str:
        """
        Generate the HTML table for monthly stocks.

        This overrides the weekly version so we can show a
        month-ago price instead of "Week Ago Price".
        """
        if not self.stocks_data:
            return """
            <tr>
                <td colspan="8" style="text-align: center; padding: 20px;">
                    No stocks matched the monthly criteria.
                </td>
            </tr>
            """

        # Show all stocks found, but cap at 50 for reasonable page size
        stocks_to_display = (
            self.stocks_data[:50]
            if len(self.stocks_data) > 50
            else self.stocks_data
        )

        table_rows = ""
        for idx, stock in enumerate(stocks_to_display, 1):
            market_cap_billions = stock["market_cap"] / 1_000_000_000
            # Prefer monthly price key, but fall back gracefully
            price_month_ago = stock.get(
                "price_1mo_ago", stock.get("price_5days_ago", 0)
            )
            current_price = stock["price"]

            table_rows += f"""
            <tr>
                <td><span class="rank">#{idx}</span></td>
                <td><span class="symbol">{stock['symbol']}</span></td>
                <td>{stock['company_name']}</td>
                <td>${price_month_ago:.2f}</td>
                <td>${current_price:.2f}</td>
                <td><span class="negative">{stock['change_percent']:.2f}%</span></td>
                <td>${market_cap_billions:.1f}B</td>
                <td>{stock['sector']}</td>
            </tr>
            """

        return """
        <table class="stocks-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Symbol</th>
                    <th>Company Name</th>
                    <th>Month Ago Price</th>
                    <th>Current Price</th>
                    <th>Monthly Change</th>
                    <th>Market Cap</th>
                    <th>Sector</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """.format(
            rows=table_rows
        )


def run_monthly_report(
    html_filename: str = "stock_report_monthly.html",
    json_filename: str = "stocks_data_monthly.json",
) -> Dict[str, Any]:
    """
    Run the full monthly report pipeline.

    This is a reusable entry point for CLI, Streamlit, or GitHub Actions.

    Args:
        html_filename: Name of the HTML report file to generate.
        json_filename: Name of the JSON data export file.

    Returns:
        Dict with keys: stocks_count, html_path, json_path.
    """
    generator = MonthlyStockReportGenerator()

    # Fetch stocks data
    stocks = generator.fetch_stocks_data()
    if not stocks:
        logger.warning("No monthly stocks found matching criteria")

    # Always generate HTML and JSON so downstream consumers have files
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
        result = run_monthly_report(
            html_filename="stock_report_monthly.html",
            json_filename="stocks_data_monthly.json",
        )

        stocks_count = result["stocks_count"]
        html_path = result["html_path"]
        json_path = result["json_path"]

        print("\n" + "=" * 80)
        if stocks_count == 0:
            print("⚠️  No monthly stocks found matching criteria")
        else:
            print("✅ MONTHLY STOCK REPORT GENERATED SUCCESSFULLY")
        print("=" * 80 + "\n")
        print(f"📊 Found {stocks_count} monthly stocks matching criteria")
        if html_path:
            print(f"🌐 Monthly HTML report saved to: {html_path}")
        if json_path:
            print(f"📁 Monthly stock data exported to: {json_path}")
        print("\n" + "=" * 80)
        print("Open 'stock_report_monthly.html' in your browser to view the report!")
        print("=" * 80 + "\n")

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
    except Exception as e:
        logger.error(
            f"Error generating monthly report: {str(e)}",
            exc_info=True,
        )


if __name__ == "__main__":
    main()

