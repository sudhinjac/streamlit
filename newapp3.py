import streamlit as st
import pandas as pd
import yfinance as yf
from io import BytesIO
import base64

# Title
st.set_page_config(page_title="Stock Ranker", layout="wide")
st.title("📈 Smart Stock Ranker & Filter App")

# Load CSV
uploaded_file = st.file_uploader("Upload CSV with Stock Metrics", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV Loaded!")

    # Clean ticker column
    df['Ticker'] = df['Ticker'].str.strip()

    # Sidebar filters
    st.sidebar.header("Filter Settings")
    apply_rsi = st.sidebar.checkbox("✅ Filter for RSI Trend = Bullish", value=True)
    apply_macd = st.sidebar.checkbox("✅ Filter for MACD Trend = Bullish", value=True)
    show_all_columns = st.sidebar.checkbox("📊 Show all columns", value=False)

    # Get current price and 1Y high
    st.info("🔄 Fetching current price and 1-year high from Yahoo Finance...")

    price_data = []
    for ticker in df['Ticker']:
        try:
            tkr = yf.Ticker(ticker)
            hist = tkr.history(period="1y")
            if not hist.empty:
                current = tkr.history(period="1d")['Close'].iloc[-1]
                high_1y = hist['High'].max()
                price_data.append((ticker, current, high_1y))
            else:
                price_data.append((ticker, None, None))
        except:
            price_data.append((ticker, None, None))

    price_df = pd.DataFrame(price_data, columns=["Ticker", "CurrentPrice", "High_1Y"])
    df = df.merge(price_df, on="Ticker")

    # Filter for current price < 65% of 1Y high
    df = df[df["CurrentPrice"] < 0.80 * df["High_1Y"]]

    if apply_rsi:
        df = df[df["RSI Trend"].str.lower() == "bullish"]
    if apply_macd:
        df = df[df["MACD Trend"].str.lower() == "bullish"]

    # Compute composite score
    st.markdown("### 📊 Composite Score = (CAGR * Sharpe Ratio * Return% * Odds_Winning)")
    df["CompositeScore"] = (
        df["CAGR"] * df["Sharp Ratio"] * df["Return%"] * df["Odds_Winning"]
    )

    # Rank stocks
    df = df.sort_values(by="CompositeScore", ascending=False).reset_index(drop=True)

    # Show table
    st.subheader("📋 Top Ranked Stocks (Price 20% below 1Y High)")
    if not df.empty:
        st.dataframe(df if show_all_columns else df[["Ticker", "CurrentPrice", "High_1Y", "CAGR", "Sharp Ratio", "Return%", "Odds_Winning", "CompositeScore"]])
    else:
        st.warning("⚠️ No stocks matched the filter criteria.")

    # Plot
    st.subheader("📈 Chart: Composite Score vs Ticker")
    st.bar_chart(df.set_index("Ticker")["CompositeScore"])

    # Download
    def to_csv_download_link(dataframe):
        csv = dataframe.to_csv(index=False).encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="filtered_stocks.csv">📥 Download Result as CSV</a>'
        return href

    if not df.empty:
        st.markdown(to_csv_download_link(df), unsafe_allow_html=True)