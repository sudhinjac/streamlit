import streamlit as st
import numpy as np
import scipy.stats as si
from scipy.optimize import brentq
import yfinance as yf
from datetime import datetime

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

def implied_volatility(market_price, S, K, T, r, option_type):
    objective_function = lambda sigma: black_scholes(S, K, T, r, sigma, option_type)[0] - market_price
    try:
        return brentq(objective_function, 0.01, 3.0)
    except ValueError:
        return np.nan

def predict_stock_price(S, IV, T):
    drift = 0  # Assuming no drift for simplicity
    randomness = IV * np.sqrt(T) * np.random.randn()
    return S * np.exp(drift + randomness)

def main():
    st.set_page_config(page_title="Options Pricing and Prediction", page_icon="📈", layout="wide")
    st.title("Options Pricing, Greeks, and Stock Price Prediction")
    
    ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()
    expiration_date = st.date_input("Enter Option Expiration Date:")
    strike_price = st.number_input("Enter Strike Price:", min_value=1.0, value=150.0)
    risk_free_rate = st.number_input("Enter Risk-Free Interest Rate (as decimal):", min_value=0.0, value=0.05)
    market_price = st.number_input("Enter Market Option Price:", min_value=0.1, value=10.0)
    option_type = st.selectbox("Select Option Type:", ["call", "put"])
    
    if st.button("Calculate IV, Greeks, and Predict Stock Price"):
        stock_data = yf.Ticker(ticker)
        current_price = stock_data.history(period="1d")['Close'].iloc[-1]
        today = datetime.today()
        T = (expiration_date - today.date()).days / 365
        
        if T > 0:
            iv = implied_volatility(market_price, current_price, strike_price, T, risk_free_rate, option_type)
            
            if np.isnan(iv):
                st.error("Implied Volatility could not be determined!")
                return
            
            price, delta, gamma, theta, vega, rho = black_scholes(current_price, strike_price, T, risk_free_rate, iv, option_type)
            future_price = predict_stock_price(current_price, iv, T)
            
            st.write(f"### {option_type.capitalize()} Option Pricing and Greeks")
            st.write(f"**Implied Volatility:** {iv:.4f}")
            st.write(f"**Option Price:** ${price:.2f}")
            st.write(f"**Delta:** {delta:.4f}")
            st.write(f"**Gamma:** {gamma:.4f}")
            st.write(f"**Theta:** {theta:.4f}")
            st.write(f"**Vega:** {vega:.4f}")
            st.write(f"**Rho:** {rho:.4f}")
            st.write(f"**Predicted Future Stock Price:** ${future_price:.2f}")
            
            recommendation = "Buy a Call Option" if future_price > strike_price else "Buy a Put Option"
            st.success(f"**Recommendation:** {recommendation}")
        else:
            st.error("Expiration date must be in the future!")

if __name__ == "__main__":
    main()
