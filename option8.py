import streamlit as st
import numpy as np
import scipy.stats as si
import yfinance as yf
from datetime import datetime, timedelta

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    
    return price

def implied_volatility(market_price, S, K, T, r, option_type):
    sigma = 0.2  # Initial guess
    for _ in range(100):
        price = black_scholes(S, K, T, r, sigma, option_type)
        vega = S * si.norm.pdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)
        
        if vega == 0:
            break
        
        sigma -= (price - market_price) / vega
        if abs(price - market_price) < 1e-5:
            break
    
    return sigma

def get_trend_recommendation(df):
    df['100_MA'] = df['Close'].rolling(window=100).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(window=14).mean()))
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] and df['RSI'].iloc[-1] < 70:
        return "Call Option (Stock is Bullish)"
    elif df['MACD'].iloc[-1] < df['Signal'].iloc[-1] and df['RSI'].iloc[-1] > 30:
        return "Put Option (Stock is Bearish)"
    else:
        return "Wait (Unclear Trend)"

def main():
    st.set_page_config(page_title="Options Trading Assistant", page_icon="📈", layout="wide")
    st.title("Options Trading Prediction Tool")
    
    ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()
    expiration_date = st.date_input("Enter Option Expiration Date:")
    strike_price = st.number_input("Enter Strike Price:", min_value=1.0, value=150.0)
    risk_free_rate = st.number_input("Enter Risk-Free Interest Rate (as decimal):", min_value=0.0, value=0.05)
    market_price = st.number_input("Enter Market Option Price:", min_value=0.01, value=10.0)
    option_type = st.selectbox("Select Option Type:", ["call", "put"])
    
    if st.button("Calculate Implied Volatility and Recommendation"):
        stock_data = yf.Ticker(ticker)
        current_price = stock_data.history(period="1d")['Close'].iloc[-1]
        today = datetime.today()
        T = (expiration_date - today.date()).days / 365
        
        if T > 0:
            IV = implied_volatility(market_price, current_price, strike_price, T, risk_free_rate, option_type)
            premium_price = black_scholes(current_price, strike_price, T, risk_free_rate, IV, option_type)
            
            hist_data = stock_data.history(period="6mo")
            trend_recommendation = get_trend_recommendation(hist_data)
            
            st.write(f"### {option_type.capitalize()} Option Analysis")
            st.write(f"**Implied Volatility:** {IV:.2%}")
            st.write(f"**Estimated Option Price:** ${premium_price:.2f}")
            st.write(f"**Trading Recommendation:** {trend_recommendation}")
        else:
            st.error("Expiration date must be in the future!")

if __name__ == "__main__":
    main()
