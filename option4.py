import streamlit as st
import numpy as np
import scipy.stats as si
import yfinance as yf
from datetime import datetime

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
    st.title("Options Pricing & Recommendation")
    
    ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()
    expiration_date = st.date_input("Enter Option Expiration Date:")
    strike_price = st.number_input("Enter Strike Price:", min_value=1.0, value=150.0)
    risk_free_rate = st.number_input("Enter Risk-Free Interest Rate (as decimal):", min_value=0.0, value=0.05)
    volatility = st.number_input("Enter Implied Volatility (as decimal):", min_value=0.01, value=0.2)
    
    if st.button("Calculate Option Prices & Recommendation"):
        stock_data = yf.Ticker(ticker)
        current_price = stock_data.history(period="1d")['Close'].iloc[-1]
        
        today = datetime.today()
        T = (expiration_date - today.date()).days / 365  # Time in years
        
        if T > 0:
            call_price = black_scholes(current_price, strike_price, T, risk_free_rate, volatility, "call")
            put_price = black_scholes(current_price, strike_price, T, risk_free_rate, volatility, "put")
            
            st.write(f"### Call Option Price INR: {call_price:.2f}")
            st.write(f"### Put Option Price INR: {put_price:.2f}")
            
            if current_price > strike_price:
                st.success("**Recommendation: Buy Call Option 📈** (Stock price is above strike price)")
            else:
                st.warning("**Recommendation: Buy Put Option 📉** (Stock price is below strike price)")
        else:
            st.error("Expiration date must be in the future!")

if __name__ == "__main__":
    main()
