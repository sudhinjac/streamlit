import streamlit as st
import numpy as np
import scipy.stats as si

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Calculate Black-Scholes option price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    
    return price

def implied_volatility(S, K, T, r, market_price, option_type="call", tol=1e-6, max_iter=100):
    """Compute implied volatility using Newton-Raphson method."""
    sigma = 0.2  # Initial guess
    for i in range(max_iter):
        price = black_scholes(S, K, T, r, sigma, option_type)
        vega = S * si.norm.pdf((np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)
        
        price_diff = price - market_price
        if abs(price_diff) < tol:
            return sigma  # Converged solution
        
        sigma -= price_diff / vega  # Newton-Raphson update
    
    return None  # If no solution found

def main():
    st.set_page_config(page_title="Implied Volatility Calculator", page_icon="📈", layout="wide")
    st.title("Options Implied Volatility & Trading Recommendation")
    
    S = st.number_input("Enter Current Stock Price ($)", min_value=1.0, value=150.0)
    K = st.number_input("Enter Strike Price ($)", min_value=1.0, value=155.0)
    T = st.number_input("Enter Time to Expiration (Years)", min_value=0.01, value=0.5)
    r = st.number_input("Enter Risk-Free Rate (as decimal, e.g., 0.05 for 5%)", min_value=0.0, value=0.05)
    market_price = st.number_input("Enter Market Option Price ($)", min_value=0.01, value=10.0)
    option_type = st.selectbox("Select Option Type", ["call", "put"])
    
    if st.button("Calculate Implied Volatility & Recommendation"):
        iv = implied_volatility(S, K, T, r, market_price, option_type)
        
        if iv:
            st.write(f"**Implied Volatility:** {iv:.4f} ({iv*100:.2f}%)")
            
            # Trading Recommendation
            if iv > 0.3:
                st.write("🔴 **High IV - Consider Selling Options (Premiums are Expensive!)**")
            elif iv < 0.15:
                st.write("🟢 **Low IV - Consider Buying Options (Premiums are Cheap!)**")
            else:
                st.write("🟡 **Moderate IV - Trade Based on Market Sentiment**")
            
            # Call vs Put Recommendation
            if option_type == "call" and S > K:
                st.write("✅ **Call Option Looks Profitable (Stock is Above Strike Price)**")
            elif option_type == "put" and S < K:
                st.write("✅ **Put Option Looks Profitable (Stock is Below Strike Price)**")
            else:
                st.write("⚠️ **Reconsider Your Option Type Based on Market Trends!**")
        else:
            st.error("Implied Volatility Calculation Did Not Converge!")

if __name__ == "__main__":
    main()