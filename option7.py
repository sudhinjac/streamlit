import streamlit as st
import numpy as np
import scipy.stats as si
from datetime import datetime

def black_scholes(S, K, T, r, sigma, option_type):
    """Calculate Black-Scholes option price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    
    return price

def implied_volatility(S, K, T, r, market_price, option_type, tol=1e-5, max_iter=100):
    """Use Newton-Raphson method to estimate implied volatility."""
    sigma = 0.2  # Initial guess
    for i in range(max_iter):
        price = black_scholes(S, K, T, r, sigma, option_type)
        vega = S * si.norm.pdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)
        diff = price - market_price
        
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
    
    return None  # Return None if IV not found within max_iter

def main():
    st.set_page_config(page_title="Option Pricing & IV Calculator", layout="wide")
    st.title("Options Pricing & Implied Volatility Calculator")
    
    ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()
    current_price = st.number_input("Enter Current Stock Price:", min_value=1.0, value=150.0)
    strike_price = st.number_input("Enter Strike Price:", min_value=1.0, value=150.0)
    expiration_date = st.date_input("Enter Expiration Date:")
    risk_free_rate = st.number_input("Enter Risk-Free Rate (Decimal):", min_value=0.0, value=0.05)
    option_price = st.number_input("Enter Market Option Price:", min_value=0.01, value=5.0)
    option_type = st.selectbox("Select Option Type:", ["call", "put"])
    
    if st.button("Calculate Implied Volatility & Recommendation"):
        today = datetime.today()
        T = (expiration_date - today.date()).days / 365
        
        if T <= 0:
            st.error("Expiration date must be in the future!")
        else:
            iv = implied_volatility(current_price, strike_price, T, risk_free_rate, option_price, option_type)
            
            if iv is not None:
                st.write(f"### Implied Volatility: {iv:.4f} or {iv*100:.2f}%")
                
                # Decision Recommendation based on IV
                if iv > 0.5:
                    st.warning("High IV: Consider selling options (Premium is high, risk is high)")
                else:
                    st.success("Low IV: Consider buying options (Premium is low, risk is lower)")
                
                if option_type == "call":
                    st.write("✅ Suggested Strategy: Buy Call if bullish on stock movement")
                else:
                    st.write("✅ Suggested Strategy: Buy Put if bearish on stock movement")
            else:
                st.error("Could not calculate Implied Volatility. Try adjusting inputs!")
                
if __name__ == "__main__":
    main()