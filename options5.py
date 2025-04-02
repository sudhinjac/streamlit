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

def monte_carlo_simulation(S, r, sigma, T, num_simulations=10000):
    dt = T / 252  # Convert time to trading days
    simulated_prices = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn(num_simulations))
    return simulated_prices

def implied_volatility(S, K, T, r, market_price, option_type="call", tol=1e-5, max_iter=100):
    low, high = 0.01, 5.0  # Bounds for volatility
    for _ in range(max_iter):
        mid = (low + high) / 2
        price = black_scholes(S, K, T, r, mid, option_type)
        if abs(price - market_price) < tol:
            return mid
        if price > market_price:
            high = mid
        else:
            low = mid
    return mid  # Return best estimate

def main():
    st.set_page_config(page_title="Options Strategy & Monte Carlo", page_icon="📈", layout="wide")
    st.title("Options Trading Strategy & Monte Carlo Simulation")
    
    ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()
    strike_price = st.number_input("Enter Strike Price:", min_value=1.0, value=150.0)
    expiration_date = st.date_input("Enter Option Expiration Date:")
    market_price = st.number_input("Enter Market Option Price:", min_value=0.1, value=10.0)
    risk_free_rate = st.number_input("Enter Risk-Free Interest Rate (as decimal):", min_value=0.0, value=0.05)
    option_type = st.selectbox("Select Option Type:", ["call", "put"])
    
    if st.button("Analyze Option"):
        stock_data = yf.Ticker(ticker)
        current_price = stock_data.history(period="1d")['Close'].iloc[-1]
        today = datetime.today()
        T = (expiration_date - today.date()).days / 365  # Time to expiration in years
        
        if T > 0:
            implied_vol = implied_volatility(current_price, strike_price, T, risk_free_rate, market_price, option_type)
            simulated_prices = monte_carlo_simulation(current_price, risk_free_rate, implied_vol, T)
            expected_future_price = np.mean(simulated_prices)
            
            st.write(f"**Implied Volatility:** {implied_vol:.2%}")
            st.write(f"**Expected Future Stock Price:** ${expected_future_price:.2f}")
            
            if expected_future_price > strike_price and option_type == "call":
                st.success("Recommendation: BUY CALL - Expected price increase")
            elif expected_future_price < strike_price and option_type == "put":
                st.success("Recommendation: BUY PUT - Expected price decrease")
            else:
                st.warning("Market uncertain - Consider waiting for better entry")
        else:
            st.error("Expiration date must be in the future!")

if __name__ == "__main__":
    main()
