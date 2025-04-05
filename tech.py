import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --------------------- Technical Indicator Functions ---------------------

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20):
    sma = data['Close'].rolling(window).mean()
    std = data['Close'].rolling(window).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    return upper, lower

# --------------------- Streamlit UI ---------------------

st.set_page_config(page_title="📊 Technical Stock Analyzer", layout="wide")
st.title("📈 Technical Indicator Dashboard")

ticker = st.text_input("Enter Stock Ticker:", "AAPL")

if st.button("Analyze"):
    stock_data = yf.download(ticker, period="2y")
    df =stock_data
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = df.columns.droplevel(1)
    df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    if df.empty:
        st.error("No data found. Please check the ticker symbol.")
    else:
        # Create separate DataFrames for each use case
        df_price = df[['Open', 'High', 'Low', 'Close']].copy()
        df_indicators = df[['Close']].copy()

        # Technical indicators
        df_indicators['50_SMA'] = df_indicators['Close'].rolling(50).mean()
        df_indicators['100_EWMA'] = df_indicators['Close'].ewm(span=100, adjust=False).mean()
        df_indicators['100_SMA'] = df_indicators['Close'].rolling(100).mean()
        df_indicators['200_SMA'] = df_indicators['Close'].rolling(200).mean()
        df_indicators['RSI'] = calculate_rsi(df_indicators)
        df_indicators['MACD'], df_indicators['Signal'] = calculate_macd(df_indicators)
        df_indicators['Upper'], df_indicators['Lower'] = calculate_bollinger_bands(df_indicators)

        # Merge with price data for charting
        df_candle = df_price.join(df_indicators[['50_SMA', '100_EWMA']], how='left')

        st.subheader("📉 Candlestick Chart with 50 SMA & 100 EWMA")

        required_cols = ['Open', 'High', 'Low', 'Close', '50_SMA', '100_EWMA']
        missing_cols = [col for col in required_cols if col not in df_candle.columns]

        if missing_cols:
            st.warning(f"Missing columns for candlestick chart: {missing_cols}")
        else:
            df_candle = df_candle.dropna(subset=['Open', 'High', 'Low', 'Close'])

            fig_candle = go.Figure()

            fig_candle.add_trace(go.Candlestick(
                x=df_candle.index,
                open=df_candle['Open'],
                high=df_candle['High'],
                low=df_candle['Low'],
                close=df_candle['Close'],
                name='Candlestick'
            ))

            fig_candle.add_trace(go.Scatter(
                x=df_candle.index,
                y=df_candle['50_SMA'],
                mode='lines',
                name='50 SMA',
                line=dict(color='blue')
            ))

            fig_candle.add_trace(go.Scatter(
                x=df_candle.index,
                y=df_candle['100_EWMA'],
                mode='lines',
                name='100 EWMA',
                line=dict(color='orange')
            ))

            fig_candle.update_layout(
                title=f"{ticker} Candlestick Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                height=600,
                template="plotly_white"
            )

            st.plotly_chart(fig_candle, use_container_width=True)

        # Plot other indicators
        st.subheader("📊 Technical Indicators")
        df_indicators.dropna(inplace=True)

        fig, axs = plt.subplots(4, 1, figsize=(14, 18), sharex=True)

        df_indicators[['Close', '100_SMA', '200_SMA']].plot(ax=axs[0], title="Close with 100 & 200 SMA")
        df_indicators['RSI'].plot(ax=axs[1], title="RSI", color='purple')
        df_indicators[['MACD', 'Signal']].plot(ax=axs[2], title="MACD", color=['green', 'red'])
        df_indicators[['Close', 'Upper', 'Lower']].plot(ax=axs[3], title="Bollinger Bands")

        plt.tight_layout()
        st.pyplot(fig)