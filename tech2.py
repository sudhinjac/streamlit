import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from io import BytesIO

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
        df['50_SMA'] = df['Close'].rolling(50).mean()
        df['100_EWMA'] = df['Close'].ewm(span=100, adjust=False).mean()
        df['100_SMA'] = df['Close'].rolling(100).mean()
        df['200_SMA'] = df['Close'].rolling(200).mean()
        df['RSI'] = calculate_rsi(df)
        df['MACD'], df['Signal'] = calculate_macd(df)
        df['Upper'], df['Lower'] = calculate_bollinger_bands(df)

        df.dropna(inplace=True)

        st.subheader("📉 Candlestick Chart with Indicators")

        fig_candle = go.Figure()

        fig_candle.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlestick'))

        fig_candle.add_trace(go.Scatter(x=df.index, y=df['50_SMA'], mode='lines', name='50 SMA', line=dict(color='blue')))
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['100_EWMA'], mode='lines', name='100 EWMA', line=dict(color='orange')))
        fig_candle.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2', marker=dict(color='lightgrey')))

        fig_candle.update_layout(
            title=f"{ticker} Candlestick Chart with Indicators",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Volume'),
            height=700,
            template="plotly_white")

        st.plotly_chart(fig_candle, use_container_width=True)

        # RSI, MACD, Bollinger Band
        fig_indicators = go.Figure()
        fig_indicators.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
        fig_indicators.update_layout(title='RSI Indicator')
        st.plotly_chart(fig_indicators, use_container_width=True)

        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='green')))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal Line', line=dict(color='red')))
        fig_macd.update_layout(title='MACD Indicator')
        st.plotly_chart(fig_macd, use_container_width=True)

        fig_boll = go.Figure()
        fig_boll.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='black')))
        fig_boll.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='Upper Band', line=dict(color='green')))
        fig_boll.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='Lower Band', line=dict(color='red')))
        fig_boll.update_layout(title='Bollinger Bands')
        st.plotly_chart(fig_boll, use_container_width=True)

        # Trade Signal Table
        latest = df.iloc[-1]
        signal_data = {
            'Indicator': ['RSI', 'MACD', 'Bollinger'],
            'Signal': [
                'Bullish' if latest['RSI'] < 30 else 'Bearish' if latest['RSI'] > 70 else 'Neutral',
                'Bullish' if latest['MACD'] > latest['Signal'] else 'Bearish',
                'Bullish' if latest['Close'] < latest['Lower'] else 'Bearish' if latest['Close'] > latest['Upper'] else 'Neutral']
        }
        signal_df = pd.DataFrame(signal_data)
        st.subheader("🔔 Trade Signal Summary (Latest Day)")
        st.table(signal_df)

        # Forecasting using Exponential Smoothing
        st.subheader("📈 Forecasting (Next 30 Days)")

        try:
            model = ExponentialSmoothing(df['Close'], trend='add', seasonal=None, initialization_method='estimated')
            model_fit = model.fit()
            forecast = model_fit.forecast(30)

            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Actual'))
            forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
            fig_forecast.add_trace(go.Scatter(x=forecast_index, y=forecast, name='Forecast', line=dict(color='blue')))
            fig_forecast.update_layout(title='30-Day Forecast using Exponential Smoothing')
            st.plotly_chart(fig_forecast, use_container_width=True)

        except Exception as e:
            st.error(f"Forecasting error: {e}")

        # PNG Export
        buffer = BytesIO()
        fig_candle.write_image(buffer, format='png')
        st.download_button(label="📷 Download Candlestick Chart as PNG", data=buffer.getvalue(), file_name=f"{ticker}_candlestick.png", mime="image/png")