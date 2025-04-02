import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as si
from datetime import datetime
import yfinance as yf

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    
    return price

def calculate_implied_volatility(market_price, S, K, T, r, option_type, tol=1e-5, max_iter=100):
    sigma = 0.2  # Initial guess
    for _ in range(max_iter):
        price = black_scholes(S, K, T, r, sigma, option_type)
        diff = price - market_price
        
        if np.abs(diff) < tol:
            return sigma
        
        vega = S * si.norm.pdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)
        sigma -= diff / vega  # Newton-Raphson update
        
        if sigma <= 0:
            return None
    return sigma

def main():
    st.set_page_config(page_title="Options Pricing App", page_icon="📈", layout="wide")
    st.title("Options Pricing & Stock Trend Prediction")
    
    ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()
    expiration_date = st.date_input("Enter Option Expiration Date:")
    strike_price = st.number_input("Enter Strike Price:", min_value=1.0, value=150.0)
    risk_free_rate = st.number_input("Enter Risk-Free Interest Rate (as decimal):", min_value=0.0, value=0.05)
    market_price = st.number_input("Enter Market Option Price:", min_value=0.01, value=10.0)
    option_type = st.selectbox("Select Option Type:", ["call", "put"])
    
    if st.button("Calculate IV, Option Price & Trend Prediction"):
        stock_data = yf.Ticker(ticker)
        df = stock_data.history(period="200d")
        
        if df.empty:
            st.error("Could not retrieve stock data!")
            return
        
        current_price = df['Close'].iloc[-1]
        today = datetime.today()
        T = (expiration_date - today.date()).days / 365
        
        if T > 0:
            iv = calculate_implied_volatility(market_price, current_price, strike_price, T, risk_free_rate, option_type)
            if iv is None:
                st.error("Could not determine implied volatility.")
                return
            
            option_price = black_scholes(current_price, strike_price, T, risk_free_rate, iv, option_type)
            st.write(f"### {option_type.capitalize()} Option Pricing")
            st.write(f"**Implied Volatility:** {iv:.2%}")
            st.write(f"**Option Price:** ${option_price:.2f}")
            
            trend = predict_trend(df)
            st.write(f"### Stock Trend Prediction: {trend}")
            
            if trend == "Bullish":
                st.success("Consider buying a Call option!")
            elif trend == "Bearish":
                st.warning("Consider buying a Put option!")
            else:
                st.info("Market is neutral. Wait for a better entry point.")
        else:
            st.error("Expiration date must be in the future!")

def predict_trend(df):
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['Signal'] = compute_macd(df['Close'])
    df['MA_100'] = df['Close'].rolling(window=100).mean()
    df['Ichimoku'] = compute_ichimoku(df)
    
    bullish = (df['MACD'].iloc[-1] > df['Signal'].iloc[-1]) and (df['RSI'].iloc[-1] > 50) and (df['Close'].iloc[-1] > df['MA_100'].iloc[-1])
    bearish = (df['MACD'].iloc[-1] < df['Signal'].iloc[-1]) and (df['RSI'].iloc[-1] < 50) and (df['Close'].iloc[-1] < df['MA_100'].iloc[-1])
    
    return "Bullish" if bullish else "Bearish" if bearish else "Neutral"

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(prices):
    short_ema = prices.ewm(span=12, adjust=False).mean()
    long_ema = prices.ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def compute_ichimoku(df):
    nine_period_high = df['High'].rolling(window=9).max()
    nine_period_low = df['Low'].rolling(window=9).min()
    tenkan_sen = (nine_period_high + nine_period_low) / 2
    
    twenty_six_period_high = df['High'].rolling(window=26).max()
    twenty_six_period_low = df['Low'].rolling(window=26).min()
    kijun_sen = (twenty_six_period_high + twenty_six_period_low) / 2
    
    return tenkan_sen.iloc[-1] > kijun_sen.iloc[-1]

if __name__ == "__main__":
    main()
