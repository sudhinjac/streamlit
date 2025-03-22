import streamlit as st
import numpy as np
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
    
    delta = si.norm.cdf(d1) if option_type == "call" else si.norm.cdf(d1) - 1
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (- (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r * T) * si.norm.cdf(d2 if option_type == "call" else -d2))
    vega = S * si.norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * si.norm.cdf(d2 if option_type == "call" else -d2)
    
    return price, delta, gamma, theta, vega, rho

def main():
    st.set_page_config(page_title="Options Pricing App", page_icon="📈", layout="wide")
    st.title("Options Pricing Using Black-Scholes Model")
    
    ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()
    expiration_date = st.date_input("Enter Option Expiration Date:")
    strike_price = st.number_input("Enter Strike Price:", min_value=1.0, value=150.0)
    risk_free_rate = st.number_input("Enter Risk-Free Interest Rate (as decimal):", min_value=0.0, value=0.05)
    volatility = st.number_input("Enter Implied Volatility (as decimal):", min_value=0.01, value=0.2)
    option_type = st.selectbox("Select Option Type:", ["call", "put"])
    
    if st.button("Calculate Price and Greeks"):
        stock_data = yf.Ticker(ticker)
        current_price = stock_data.history(period="1d")['Close'].iloc[-1]
        
        today = datetime.today()
        T = (expiration_date - today.date()).days / 365
        
        if T > 0:
            price, delta, gamma, theta, vega, rho = black_scholes(current_price, strike_price, T, risk_free_rate, volatility, option_type)
            
            st.write(f"### {option_type.capitalize()} Option Pricing and Greeks")
            st.write(f"**Option Price:** ${price:.2f}")
            st.write(f"**Delta:** {delta:.4f}")
            st.write(f"**Gamma:** {gamma:.4f}")
            st.write(f"**Theta:** {theta:.4f}")
            st.write(f"**Vega:** {vega:.4f}")
            st.write(f"**Rho:** {rho:.4f}")
        else:
            st.error("Expiration date must be in the future!")

if __name__ == "__main__":
    main()
