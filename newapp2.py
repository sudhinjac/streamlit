import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="Best Stock Picker", layout="wide")

st.title("📈 Smart Stock Picker – Buy the Dip, Not the Hype!")

uploaded_file = st.file_uploader("Upload your stock metrics CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Extract tickers
    tickers = df['Ticker'].dropna().unique().tolist()

    # Create columns to store Yahoo Finance values
    df["Current Price"] = None
    df["1Y High"] = None
    df["Price Drop %"] = None

    st.info("🔄 Fetching data from Yahoo Finance...")

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")

            if hist.empty:
                continue

            current_price = hist["Close"].iloc[-1]
            high_1y = hist["High"].max()
            drop_pct = ((high_1y - current_price) / high_1y) * 100

            df.loc[df["Ticker"] == ticker, "Current Price"] = current_price
            df.loc[df["Ticker"] == ticker, "1Y High"] = high_1y
            df.loc[df["Ticker"] == ticker, "Price Drop %"] = drop_pct

        except Exception as e:
            st.warning(f"⚠️ Failed for {ticker}: {e}")

    # Drop rows with missing current price
    df.dropna(subset=["Current Price"], inplace=True)

    # Filter: current price is 35% below 1Y high
    df_filtered = df[df["Price Drop %"] >= 20]

    # Filter for Bullish RSI & MACD
    df_filtered = df_filtered[
        (df_filtered["RSI Trend"] == "Bullish") & (df_filtered["MACD Trend"] == "Bullish")
    ]

    st.success(f"✅ Found {len(df_filtered)} potential undervalued stocks with bullish indicators.")

    # Compute a score
    df_filtered["Score"] = (
        df_filtered["CAGR"].rank(ascending=False) * 0.4 +
        df_filtered["Sharp Ratio"].rank(ascending=False) * 0.3 +
        df_filtered["Return%"].rank(ascending=False) * 0.2 +
        df_filtered["Odds_Winning"].rank(ascending=False) * 0.1
    )

    df_filtered.sort_values("Score", ascending=False, inplace=True)

    st.subheader("📊 Top Stocks to Buy Based on Composite Score")
    st.dataframe(df_filtered[[
        "Ticker", "Current Price", "1Y High", "Price Drop %", "RSI Trend", "MACD Trend",
        "CAGR", "Sharp Ratio", "Return%", "Odds_Winning", "Score"
    ]].reset_index(drop=True), use_container_width=True)

    st.subheader("📈 Score Chart")
    st.bar_chart(df_filtered.set_index("Ticker")["Score"])

    # Export
    output = io.BytesIO()
    df_filtered.to_excel(output, index=False)
    st.download_button(
        label="📥 Download Results as Excel",
        data=output.getvalue(),
        file_name="top_stocks_to_buy.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")