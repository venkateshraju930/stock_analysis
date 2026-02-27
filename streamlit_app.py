import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from stock_generator_daily import run_daily_report
from stock_generator_weekly import run_weekly_report
from stock_generator_monthly import run_monthly_report


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
        initial_sidebar_state="collapsed",
    )

    st.title("📈 Stock Market Drop Reports")
    st.write(
        "This app generates AI-powered reports for large-cap stocks that dropped "
        "more than **10%** over different timeframes and displays the resulting "
        "HTML reports."
    )

    ensure_groq_key_warning()

    daily_html_path = Path("stock_report.html")
    weekly_html_path = Path("stock_report_weekly.html")
    monthly_html_path = Path("stock_report_monthly.html")

    # Sidebar controls
    with st.sidebar:
        st.header("Report Controls")
        generate_clicked = st.button("🔄 Generate / Refresh Daily, Weekly & Monthly Reports", type="primary")
        st.markdown("---")
        st.markdown(
            "**Tip:** In GitHub Actions, the daily report can be generated on a "
            "schedule and committed to the repo. You can then view the latest "
            "committed `stock_report.html` and `stock_report_weekly.html` here."
        )

    # Generate / refresh report
    if generate_clicked:
        with st.spinner("Generating report with Groq and yfinance..."):
            try:
                # Daily report
                daily_result = run_daily_report(
                    html_filename=str(daily_html_path),
                    json_filename="stocks_data.json",
                )
                daily_count = daily_result.get("stocks_count", 0)

                # Weekly report
                weekly_result = run_weekly_report(
                    html_filename=str(weekly_html_path),
                    json_filename="stocks_data_weekly.json",
                )
                weekly_count = weekly_result.get("stocks_count", 0)

                # Monthly report
                monthly_result = run_monthly_report(
                    html_filename=str(monthly_html_path),
                    json_filename="stocks_data_monthly.json",
                )
                monthly_count = monthly_result.get("stocks_count", 0)

                if daily_count == 0 and weekly_count == 0 and monthly_count == 0:
                    st.warning(
                        "No stocks matched the criteria for the daily, weekly, or "
                        "monthly reports."
                    )
                else:
                    st.success(
                        f"Reports generated successfully "
                        f"(daily: {daily_count} stocks, "
                        f"weekly: {weekly_count} stocks, "
                        f"monthly: {monthly_count} stocks)."
                    )
            except Exception as e:
                st.error(
                    "Failed to generate report. "
                    "Check that `GROQ_API_KEY` is configured and that the Groq API "
                    "is reachable."
                )
                st.exception(e)

    # Display current reports (if available)
    daily_html_content = load_report_html(daily_html_path)
    weekly_html_content = load_report_html(weekly_html_path)
    monthly_html_content = load_report_html(monthly_html_path)

    if not daily_html_content and not weekly_html_content and not monthly_html_content:
        st.info(
            "No reports found yet. Click **Generate / Refresh Daily, Weekly & "
            "Monthly Reports** to create them, or run the generator scripts "
            "separately."
        )
        return

    tab_daily, tab_weekly, tab_monthly = st.tabs(
        ["📆 Daily Report", "📊 Weekly Report", "📅 Monthly Report"]
    )

    with tab_daily:
        if not daily_html_content:
            st.info(
                "No `stock_report.html` found yet. Generate reports to see the "
                "daily view."
            )
        else:
            st.subheader("Latest Daily Report")
            components.html(daily_html_content, height=900, scrolling=True)

    with tab_weekly:
        if not weekly_html_content:
            st.info(
                "No `stock_report_weekly.html` found yet. Generate reports to see "
                "the weekly view."
            )
        else:
            st.subheader("Latest Weekly Report")
            components.html(weekly_html_content, height=900, scrolling=True)

    with tab_monthly:
        if not monthly_html_content:
            st.info(
                "No `stock_report_monthly.html` found yet. Generate reports to see "
                "the monthly view."
            )
        else:
            st.subheader("Latest Monthly Report")
            components.html(monthly_html_content, height=900, scrolling=True)


if __name__ == "__main__":
    main()

