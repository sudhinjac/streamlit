import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as si
from datetime import datetime, timedelta

def calculate_macd(data):
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_rsi(data, period=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    
    return price

def main():
    st.set_page_config(page_title="Options Pricing App", page_icon="📈", layout="wide")
    st.title("Options Trading Recommendation using MACD, RSI, and Black-Scholes Model")
    
    ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()
    expiration_date = st.date_input("Enter Option Expiration Date:")
    strike_price = st.number_input("Enter Strike Price:", min_value=1.0, value=150.0)
    risk_free_rate = st.number_input("Enter Risk-Free Interest Rate (as decimal):", min_value=0.0, value=0.05)
    option_type = st.selectbox("Select Option Type:", ["call", "put"])
    
    if st.button("Analyze Stock and Predict Options"):
        stock_data = yf.Ticker(ticker)
        df = stock_data.history(period="6mo")
        current_price = df['Close'].iloc[-1]
        
        macd, signal = calculate_macd(df)
        rsi = calculate_rsi(df)
        
        trend = "Neutral"
        if macd.iloc[-1] > signal.iloc[-1] and rsi.iloc[-1] < 70:
            trend = "Bullish - Consider Buying a Call Option"
        elif macd.iloc[-1] < signal.iloc[-1] and rsi.iloc[-1] > 30:
            trend = "Bearish - Consider Buying a Put Option"
        
        today = datetime.today()
        T = (expiration_date - today.date()).days / 365
        
        if T > 0:
            sigma = np.std(df['Close'].pct_change()) * np.sqrt(252)
            option_price = black_scholes(current_price, strike_price, T, risk_free_rate, sigma, option_type)
            
            st.write(f"### {option_type.capitalize()} Option Recommendation")
            st.write(f"**Current Stock Price:** ${current_price:.2f}")
            st.write(f"**Trend Analysis:** {trend}")
            st.write(f"**Predicted Option Price:** ${option_price:.2f}")
        else:
            st.error("Expiration date must be in the future!")

if __name__ == "__main__":
    main()
