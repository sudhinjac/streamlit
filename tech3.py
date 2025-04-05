import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
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

def calculate_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_ichimoku(data):
    high_9 = data['High'].rolling(window=9).max()
    low_9 = data['Low'].rolling(window=9).min()
    tenkan_sen = (high_9 + low_9) / 2

    high_26 = data['High'].rolling(window=26).max()
    low_26 = data['Low'].rolling(window=26).min()
    kijun_sen = (high_26 + low_26) / 2

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    high_52 = data['High'].rolling(window=52).max()
    low_52 = data['Low'].rolling(window=52).min()
    senkou_span_b = ((high_52 + low_52) / 2).shift(26)

    chikou_span = data['Close'].shift(-26)

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

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
    st.write(df.tail(10))
    


    if df.empty:
        st.error("No data found. Please check the ticker symbol.")
    else:
        df['50_SMA'] = df['Close'].rolling(50).mean()
        df['100_EWMA'] = df['Close'].ewm(span=100, adjust=False).mean()
        df['RSI'] = calculate_rsi(df)
        df['MACD'], df['Signal'] = calculate_macd(df)
        df['Upper'], df['Lower'] = calculate_bollinger_bands(df)
        df['ATR'] = calculate_atr(df)
        tenkan, kijun, span_a, span_b, chikou = calculate_ichimoku(df)

        df['Tenkan'], df['Kijun'], df['SpanA'], df['SpanB'], df['Chikou'] = tenkan, kijun, span_a, span_b, chikou

        df.dropna(subset=['RSI', 'MACD', 'Signal', 'Upper', 'Lower', 'ATR', 'Tenkan', 'Kijun', 'SpanA', 'SpanB'], inplace=True)

        st.subheader("📉 Candlestick Chart with Indicators")
        fig_candle = go.Figure()

        fig_candle.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))

        fig_candle.add_trace(go.Scatter(x=df.index, y=df['50_SMA'], mode='lines', name='50 SMA', line=dict(color='blue')))
        fig_candle.add_trace(go.Scatter(x=df.index, y=df['100_EWMA'], mode='lines', name='100 EWMA', line=dict(color='orange')))
        fig_candle.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2', marker=dict(color='lightgrey')))

        fig_candle.update_layout(
            title=f"{ticker} Candlestick Chart with Indicators",
            xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False,
            yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Volume'), height=700,
            template="plotly_white")

        st.plotly_chart(fig_candle, use_container_width=True)

        st.subheader("Indicators: RSI, MACD, Bollinger Bands")
        st.plotly_chart(go.Figure([go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple'))])
                        .update_layout(title='RSI Indicator'), use_container_width=True)

        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='green')))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='red')))
        fig_macd.update_layout(title='MACD Indicator')
        st.plotly_chart(fig_macd, use_container_width=True)

        fig_boll = go.Figure()
        fig_boll.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
        fig_boll.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='Upper Band'))
        fig_boll.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='Lower Band'))
        fig_boll.update_layout(title='Bollinger Bands')
        st.plotly_chart(fig_boll, use_container_width=True)

        fig_atr = go.Figure([go.Scatter(x=df.index, y=df['ATR'], name='ATR', line=dict(color='brown'))])
        fig_atr.update_layout(title='ATR (Average True Range)')
        st.plotly_chart(fig_atr, use_container_width=True)

        fig_ichi = go.Figure()
        fig_ichi.add_trace(go.Scatter(x=df.index, y=df['Tenkan'], name='Tenkan Sen'))
        fig_ichi.add_trace(go.Scatter(x=df.index, y=df['Kijun'], name='Kijun Sen'))
        fig_ichi.add_trace(go.Scatter(x=df.index, y=df['SpanA'], name='Senkou Span A'))
        fig_ichi.add_trace(go.Scatter(x=df.index, y=df['SpanB'], name='Senkou Span B'))
        fig_ichi.update_layout(title='Ichimoku Cloud Components')
        st.plotly_chart(fig_ichi, use_container_width=True)

        latest = df.iloc[-1]
        signal_data = {
            'Indicator': ['RSI', 'MACD', 'Bollinger'],
            'Signal': [
                'Bullish' if latest['RSI'] < 30 else 'Bearish' if latest['RSI'] > 70 else 'Neutral',
                'Bullish' if latest['MACD'] > latest['Signal'] else 'Bearish',
                'Bullish' if latest['Close'] < latest['Lower'] else 'Bearish' if latest['Close'] > latest['Upper'] else 'Neutral'
            ]
        }
        st.subheader("🔔 Trade Signal Summary (Latest Day)")
        st.table(pd.DataFrame(signal_data))

        st.subheader("🔮 Forecasting (ARIMA & Holt-Winters)")
        forecast_df = df[['Close']].copy().iloc[-180:]
        forecast_df.index = pd.to_datetime(forecast_df.index)

        hw_model = ExponentialSmoothing(forecast_df['Close'], trend='add', seasonal=None).fit()
        hw_forecast = hw_model.forecast(30)

        arima_model = ARIMA(forecast_df['Close'], order=(5, 1, 0)).fit()
        arima_forecast = arima_model.forecast(30)

        forecast_dates = pd.date_range(start=forecast_df.index[-1] + pd.Timedelta(days=1), periods=30)

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Close'], name="Actual"))
        fig_forecast.add_trace(go.Scatter(x=forecast_dates, y=hw_forecast, name="Holt-Winters Forecast"))
        fig_forecast.add_trace(go.Scatter(x=forecast_dates, y=arima_forecast, name="ARIMA Forecast"))
        fig_forecast.update_layout(title="30-Day Forecast", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.dataframe(pd.DataFrame({
            'Date': forecast_dates,
            'ARIMA Forecast': arima_forecast,
            'Holt-Winters Forecast': hw_forecast
        }).set_index('Date').round(2))

        buffer = BytesIO()
        fig_candle.write_image(buffer, format='png')
        st.download_button(label="📷 Download Candlestick Chart as PNG", data=buffer.getvalue(),
                           file_name=f"{ticker}_candlestick.png", mime="image/png")
