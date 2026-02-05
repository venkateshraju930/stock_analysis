import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from stock_generator_daily import run_daily_report


def ensure_groq_key_warning():
    """Show a warning in the UI if GROQ_API_KEY is not set."""
    if not os.getenv("GROQ_API_KEY"):
        st.warning(
            "Environment variable `GROQ_API_KEY` is not set. "
            "Report generation will fail until it is configured.\n\n"
            "- **Locally**: set `GROQ_API_KEY` in your shell\n"
            "- **GitHub Actions**: create a secret (e.g. `groq_key`) and map it\n"
            "  to `GROQ_API_KEY` in your workflow `env` section."
        )


def load_report_html(html_path: Path) -> str:
    """Load the HTML report content if it exists."""
    if not html_path.exists():
        return ""
    try:
        return html_path.read_text(encoding="utf-8")
    except Exception:
        return ""


def main():
    st.set_page_config(
        page_title="Daily Stock Report",
        layout="wide",
        page_icon="📈",
    )

    st.title("📈 Daily Stock Market Drop Report")
    st.write(
        "This app generates a daily AI-powered report for large-cap stocks that "
        "dropped more than **10%** in the last 24 hours and displays the "
        "resulting `stock_report.html`."
    )

    ensure_groq_key_warning()

    html_path = Path("stock_report.html")

    # Sidebar controls
    with st.sidebar:
        st.header("Report Controls")
        generate_clicked = st.button("🔄 Generate / Refresh Report", type="primary")
        st.markdown("---")
        st.markdown(
            "**Tip:** In GitHub Actions, the report can be generated on a schedule "
            "and uploaded as an artifact. You can then view the latest committed "
            "`stock_report.html` here."
        )

    # Generate / refresh report
    if generate_clicked:
        with st.spinner("Generating report with Groq and yfinance..."):
            try:
                result = run_daily_report(
                    html_filename=str(html_path),
                    json_filename="stocks_data.json",
                )
                stocks_count = result.get("stocks_count", 0)

                if stocks_count == 0:
                    st.warning(
                        "No stocks matched the criteria today (drop > 10% and "
                        "market cap > $1B)."
                    )
                else:
                    st.success(
                        f"Report generated successfully for {stocks_count} stocks."
                    )
            except Exception as e:
                st.error(
                    "Failed to generate report. "
                    "Check that `GROQ_API_KEY` is configured and that the Groq API "
                    "is reachable."
                )
                st.exception(e)

    # Display current report (if available)
    html_content = load_report_html(html_path)

    if not html_content:
        st.info(
            "No `stock_report.html` found yet. Click **Generate / Refresh Report** "
            "to create one, or run `python stock_generator_daily.py` separately."
        )
        return

    st.subheader("Latest Generated Report")
    components.v1.html(html_content, height=900, scrolling=True)


if __name__ == "__main__":
    main()

