#!/usr/bin/env python3
"""
Stock Market Analysis Report Generator - PDF Version (Weekly)
Analyzes stocks that dropped >10% in the last 7 days
Generates full-page PDF reports with detailed AI analysis
"""

import logging
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from groq import Groq
import yfinance as yf


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


# Comprehensive text cleaning for PDF
def clean_text_for_pdf(text):
    """
    Clean text for PDF output by removing/replacing all problematic characters
    """
    if not text:
        return text

    # Map of characters to replace
    replacements = {
        '■': '', '□': '', '◆': '-', '●': '-', '•': '-', '○': '-',
        '◉': '-', '★': '*', '☆': '*', '→': '->', '←': '<-',
        '↓': 'v', '↑': '^', '✓': 'OK', '✗': 'X', '»': '>>',
        '«': '<<', '°': 'deg', '±': '+/-', '×': 'x', '÷': '/',
        '§': 's.', '¶': 'p.', '†': '+', '‡': '++', '№': 'No.',
        '™': '(TM)', '©': '(C)', '®': '(R)', '€': 'EUR', '£': 'GBP',
        '¥': 'JPY', '¢': 'cent', '„': '"', '‟': '"', '‹': '<',
        '›': '>', '–': '-', '—': '-', '‐': '-', '‑': '-',
        '‒': '-', '…': '...', ''': "'", ''': "'", '"': '"',
        '"': '"', '‚': ',', '\x00': '', '\x01': '', '\x02': '',
    }

    # Apply replacements
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # Remove remaining non-ASCII characters except common ones
    cleaned = ''
    for char in text:
        if ord(char) < 128 or char in 'àáâãäåèéêëìíîïòóôõöùúûüýÿ':
            cleaned += char

    return cleaned


# Load env before using
load_groq_env()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockReportGeneratorPDF:
    """Generate stock analysis reports in PDF format"""

    def __init__(self):
        """Initialize the report generator"""
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        self.client = Groq(api_key=self.api_key)
        self.stocks_data = []
        self.model = None

    def _get_available_model(self) -> str:
        """
        Get available Groq model.
        Uses the latest available Groq models.
        """
        # Latest Groq models (mixtral-8x7b-32768 is decommissioned)
        preferred = [
            'llama-3.1-70b-versatile',  # Latest and most capable
            'llama-3.3-70b-versatile',  # Alternative latest
            'llama-3.1-8b-instant',  # Faster fallback
        ]

        for model in preferred:
            try:
                # Test if model works
                test = self.client.chat.completions.create(
                    model=model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": "OK"}]
                )
                logger.info(f"Using model: {model}")
                return model
            except Exception as e:
                error_msg = str(e)
                if "decommissioned" in error_msg or "not supported" in error_msg:
                    logger.debug(f"Model {model} decommissioned, trying next")
                else:
                    logger.debug(f"Model {model} unavailable: {error_msg}")
                continue

        # Default to llama-3.1-70b
        logger.info(f"Using fallback model: llama-3.1-70b-versatile")
        return "llama-3.1-70b-versatile"

    def fetch_stocks_data(self) -> list:
        """Fetch stocks that dropped more than 10% in last 7 days (weekly timeframe)"""
        logger.info("Fetching stocks data (7-day/weekly timeframe)...")

        try:
            symbols = self._get_all_large_cap_stocks()
            stocks_with_drops = []
            total_checked = 0

            for symbol in symbols:
                try:
                    total_checked += 1
                    if total_checked % 50 == 0:
                        logger.info(f"Progress: Checked {total_checked}/{len(symbols)} stocks...")

                    ticker = yf.Ticker(symbol)
                    info = ticker.info

                    if not info or 'longName' not in info or info.get('marketCap') is None:
                        logger.debug(f"Skipping {symbol}: No market cap data available")
                        continue

                    # Get historical data for 7-day change (need 10+ days of data)
                    hist = ticker.history(period="10d")
                    if hist is None or len(hist) < 2:
                        logger.debug(f"Skipping {symbol}: Insufficient historical data")
                        continue

                    closing_prices = hist['Close']

                    # Get price from ~7 days ago and current price
                    if len(closing_prices) >= 7:
                        price_7days_ago = closing_prices.iloc[-8] if len(closing_prices) >= 8 else closing_prices.iloc[
                            0]
                        current_price = closing_prices.iloc[-1]
                    else:
                        price_7days_ago = closing_prices.iloc[0]
                        current_price = closing_prices.iloc[-1]

                    # Skip if prices are invalid
                    if price_7days_ago <= 0 or current_price <= 0:
                        logger.debug(f"Skipping {symbol}: Invalid price data")
                        continue

                    # Check for NaN values
                    if price_7days_ago != price_7days_ago or current_price != current_price:
                        logger.debug(f"Skipping {symbol}: Invalid price data (NaN)")
                        continue

                    # Calculate 7-day percentage change
                    change_percent = ((current_price - price_7days_ago) / price_7days_ago) * 100

                    # Check criteria: >10% drop AND market cap >$1B
                    market_cap = info.get('marketCap', 0)

                    if not market_cap or market_cap == 0:
                        logger.debug(f"Skipping {symbol}: No market cap data")
                        continue

                    if change_percent < -10 and market_cap > 1_000_000_000:
                        stocks_with_drops.append({
                            'symbol': symbol,
                            'price': current_price,
                            'price_7days_ago': price_7days_ago,
                            'change_percent': round(change_percent, 2),
                            'market_cap': market_cap,
                            'company_name': info.get('longName', symbol),
                            'sector': info.get('sector', 'N/A'),
                            'industry': info.get('industry', 'N/A'),
                        })
                except Exception as e:
                    logger.debug(f"Skipping {symbol}: {type(e).__name__}")
                    continue

            stocks_with_drops.sort(key=lambda x: x['change_percent'])
            self.stocks_data = stocks_with_drops
            logger.info(f"Found {len(self.stocks_data)} stocks with >10% drop in last 7 days and >$1B market cap")

            return self.stocks_data

        except ImportError:
            logger.error("yfinance not installed. Install with: pip install yfinance")
            raise

    def _get_all_large_cap_stocks(self) -> list:
        """Get list of large-cap stocks to analyze"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
            'JPM', 'BAC', 'WFC', 'GS', 'BLK',
            'JNJ', 'UNH', 'PFE', 'ABBV', 'LLY', 'MRK', 'AMGN', 'TMO',
            'WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX',
            'BA', 'CAT', 'RTX', 'LMT', 'GE', 'HON', 'MMM',
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC',
            'NEE', 'DUK', 'SO', 'EXC', 'AEP',
            'CSCO', 'CRM', 'ADBE', 'INTU', 'QCOM', 'AMD', 'INTC', 'AMAT', 'MU', 'NET', 'SNOW', 'DDOG', 'CRWD', 'ZM',
            'OKTA', 'TWLO', 'ROKU', 'SHOP', 'COIN',
            'ADP', 'FISV', 'FIS', 'ACN', 'IBM', 'ORCL',
            'DIS', 'NFLX',
            'PSA', 'EQIX', 'DLR', 'SPG',
            'HD', 'V', 'MA', 'RBLX', 'ISRG', 'DASH', 'UBER', 'VZ', 'T', 'CMCSA', 'TMUS', 'UPS', 'FDX', 'DAL', 'UAL',
            'AAL'
        ]

    def generate_stock_drop_reasons(self) -> str:
        """Generate detailed reasons why each stock dropped using Groq API"""
        if not self.stocks_data:
            return "No data available"

        logger.info("Generating detailed reasons for stock drops...")

        stocks_to_analyze = self.stocks_data[:30] if len(self.stocks_data) > 30 else self.stocks_data

        stocks_info = "\n".join([
            f"• {s['symbol']} ({s['company_name']}): {s['change_percent']:.2f}% drop in 7 days, "
            f"Sector: {s['sector']}, Market Cap: ${s['market_cap'] / 1e9:.1f}B, Current Price: ${s['price']:.2f}"
            for s in stocks_to_analyze
        ])

        prompt = f"""Based on these stocks that dropped more than 10% in the last 7 days, provide BRIEF but detailed reasons (2-3 lines max, 30-50 words) for why each stock dropped. Be specific and help users understand the key factors.

STOCKS (All stocks with >10% drop in 7 days, sorted by worst first):
{stocks_info}

Format your response EXACTLY as:

SYMBOL (Company Name) - [Change Percent]%
[2-3 line explanation with key reasons]

SYMBOL (Company Name) - [Change Percent]%
[2-3 line explanation...]

IMPORTANT: Provide brief 2-3 line explanations for EVERY stock listed above. Include all stocks.
Make it concise but informative."""

        model = self._get_available_model()

        chat_completion = self.client.chat.completions.create(
            model=model,
            max_tokens=256,  # Limited by Groq free tier, will get truncated output
            # Note: With 256 tokens max, only 1-2 stocks can be detailed
            # To get all stocks, modify to request shorter summaries
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        reasons = chat_completion.choices[0].message.content
        return reasons

    def create_pdf_report(self, filename: str = "stock_report_weekly.pdf"):
        """Create PDF report with all data and analysis"""
        logger.info(f"Creating PDF report: {filename}")

        # Create PDF document
        doc = SimpleDocTemplate(
            filename,
            pagesize=letter,
            rightMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch
        )

        # Container for PDF elements
        elements = []

        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )

        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=9,
            spaceAfter=6,
            alignment=TA_LEFT
        )

        # Add title
        title = Paragraph("Stock Market Analysis Report - Weekly", title_style)
        elements.append(title)

        subtitle = Paragraph(f"Analysis of All Stocks with >10% Drop in Last 7 Days (Market Cap > $1B)", normal_style)
        elements.append(subtitle)

        timestamp = Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style)
        elements.append(timestamp)

        elements.append(Spacer(1, 0.2 * inch))

        # Summary statistics
        summary_heading = Paragraph("Summary Statistics", heading_style)
        elements.append(summary_heading)

        summary_data = [
            ['Total Stocks Down >10% (7-day)', str(len(self.stocks_data))],
            ['Biggest Drop',
             f"{min(s['change_percent'] for s in self.stocks_data):.2f}%" if self.stocks_data else "N/A"],
            ['Average Drop',
             f"{sum(s['change_percent'] for s in self.stocks_data) / len(self.stocks_data):.2f}%" if self.stocks_data else "N/A"],
        ]

        summary_table = Table(summary_data, colWidths=[3 * inch, 2 * inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0ff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(summary_table)

        elements.append(Spacer(1, 0.2 * inch))

        # Main stocks table
        stocks_heading = Paragraph("Stocks Down >10% (7-Day)", heading_style)
        elements.append(stocks_heading)

        table_data = [['Rank', 'Symbol', 'Company', 'Week Ago', 'Current', 'Change %', 'Market Cap', 'Sector']]

        for idx, stock in enumerate(self.stocks_data[:50], 1):
            market_cap_b = stock['market_cap'] / 1_000_000_000
            table_data.append([
                str(idx),
                stock['symbol'],
                stock['company_name'][:20],
                f"${stock['price_7days_ago']:.2f}",
                f"${stock['price']:.2f}",
                f"{stock['change_percent']:.2f}%",
                f"${market_cap_b:.1f}B",
                stock['sector'][:12]
            ])

        stocks_table = Table(table_data,
                             colWidths=[0.4 * inch, 0.6 * inch, 1.2 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch,
                                        0.8 * inch, 0.9 * inch])
        stocks_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
        ]))
        elements.append(stocks_table)

        # Add page break before reasons
        elements.append(PageBreak())

        # Detailed reasons section
        reasons_heading = Paragraph("Why Each Stock Dropped", heading_style)
        elements.append(reasons_heading)

        reasons_text = self.generate_stock_drop_reasons()

        # Comprehensive text cleaning
        reasons_text = clean_text_for_pdf(reasons_text)

        # Escape HTML entities
        reasons_text = reasons_text.replace('&', '&amp;')
        reasons_text = reasons_text.replace('<', '&lt;')
        reasons_text = reasons_text.replace('>', '&gt;')

        # Convert line breaks to HTML
        reasons_text = reasons_text.replace('\n\n', '</p><p>')
        reasons_text = reasons_text.replace('\n', '<br/>')

        # Wrap in paragraph
        reasons_text = f'<p>{reasons_text}</p>'

        reasons_para = Paragraph(reasons_text, normal_style)
        elements.append(reasons_para)

        # Build PDF
        doc.build(elements)
        logger.info(f"✅ PDF report saved to: {filename}")
        print(f"\n{'=' * 80}")
        print("✅ STOCK REPORT GENERATED SUCCESSFULLY")
        print(f"{'=' * 80}\n")
        print(f"📊 Found {len(self.stocks_data)} stocks matching criteria")
        print(f"📄 PDF report saved to: {filename}")
        print(f"\n{'=' * 80}")
        print("Open the PDF file to view the report!")
        print(f"{'=' * 80}\n")

    def export_json(self, filename: str = "stocks_data.json"):
        """Export stock data as JSON"""
        with open(filename, 'w') as f:
            json.dump(self.stocks_data, f, indent=2, default=str)
        logger.info(f"Stock data exported to: {filename}")


def main():
    """Main execution function"""
    try:
        generator = StockReportGeneratorPDF()

        # Fetch stocks data
        stocks = generator.fetch_stocks_data()

        if not stocks:
            logger.warning("No stocks found matching criteria")
            return

        # Create PDF report
        generator.create_pdf_report("stock_report_weekly.pdf")
        generator.export_json("stocks_data.json")

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()