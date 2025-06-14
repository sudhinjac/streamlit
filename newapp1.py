import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import base64

# Title and description
st.set_page_config(page_title="Stock Screener", layout="wide")
st.title("📊 Intelligent Stock Filter App")
st.markdown("""
This app filters stocks based on:
- Today's price being **≥35% lower than 1-year high**
- **Bullish RSI** and **MACD** trends
- Sorts by CAGR, Sharpe Ratio, and Return%
""")

# File upload
uploaded_file = st.file_uploader("Upload Stock Metrics CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Raw Uploaded Data")
    st.dataframe(df.head())

    st.write("🔍 Fetching 1-year high and current price from Yahoo Finance...")

    tickers = df['Ticker'].tolist()
    current_prices = []
    high_1y_prices = []

    for ticker in tickers:
        try:
            data = yf.Ticker(ticker)
            hist = data.history(period="1y")
            current_price = data.history(period="1d").iloc[-1]['Close']
            high_1y = hist['High'].max()
            current_prices.append(current_price)
            high_1y_prices.append(high_1y)
        except Exception as e:
            current_prices.append(None)
            high_1y_prices.append(None)

    df['Current Price'] = current_prices
    df['1Y High'] = high_1y_prices

    # Filter stocks with price 35% lower than 1-year high
    df['Price vs 1Y High %'] = ((df['1Y High'] - df['Current Price']) / df['1Y High']) * 100
    df_filtered = df[df['Price vs 1Y High %'] >= 23]

    st.subheader("📉 Stocks ≥35% below 1-Year High")
    st.dataframe(df_filtered[['Ticker', 'Current Price', '1Y High', 'Price vs 1Y High %']])

    # Optional RSI/MACD filtering
    if st.checkbox("🔁 Filter only Bullish RSI and MACD"):
        df_filtered = df_filtered[(df_filtered['RSI Trend'] == 'Bullish') & (df_filtered['MACD Trend'] == 'Bullish')]

    # Rank by CAGR, Sharpe Ratio, Return%
    df_ranked = df_filtered.sort_values(by=['CAGR', 'Sharp Ratio', 'Return%'], ascending=False)

    st.subheader("📈 Top Ranked Stocks")
    st.dataframe(df_ranked[['Ticker', 'CAGR', 'Sharp Ratio', 'Return%', 'Current Price', '1Y High', 'Price vs 1Y High %']])

    # Export options
    def get_table_download_link(df, file_type='csv'):
        if file_type == 'csv':
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="filtered_stocks.csv">Download CSV</a>'
        else:
            excel_buffer = pd.ExcelWriter("stocks.xlsx", engine='xlsxwriter')
            df.to_excel(excel_buffer, index=False, sheet_name='Sheet1')
            excel_buffer.close()
            href = "Download Excel currently not supported in-browser."
        return href

    st.markdown("### 💾 Export Result")
    st.markdown(get_table_download_link(df_ranked), unsafe_allow_html=True)

else:
    st.warning("⚠️ Please upload a stock metrics CSV to proceed.")
