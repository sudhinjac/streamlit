import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import norm
import cufflinks as cf

cf.go_offline()

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.set_page_config(page_title="Stock Analysis & Monte Carlo Simulation", layout="wide")
st.title("📈 Stock Analysis & Monte Carlo Simulation")
def CAGR(df):
    df["daily_ret"] = df["Close"].pct_change()
    df["cum_return"] = (1 + df["daily_ret"]).cumprod()
    n = len(df) / 252
    return (df["cum_return"].iloc[-1]) ** (1/n) - 1

def volatility(df):
    df["daily_ret"] = df["Close"].pct_change()
    return df["daily_ret"].std() * np.sqrt(252)

def sharpe(df, rf):
    return (CAGR(df) - rf) / volatility(df)

def sortino(df, rf):
    df["daily_ret"] = df["Close"].pct_change()
    neg_vol = df[df["daily_ret"] < 0]["daily_ret"].std() * np.sqrt(252)
    return (CAGR(df) - rf) / neg_vol

def max_dd(df):
    df["cum_return"] = (1 + df["Close"].pct_change()).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown_pct"] = (df["cum_roll_max"] - df["cum_return"]) / df["cum_roll_max"]
    return df["drawdown_pct"].max()

def calmar(df):
    return CAGR(df) / max_dd(df)

# Input field for stock ticker
ticker_input = st.text_input("Enter Stock Ticker:", "TANLA.NS")

if ticker_input:
    try:
        # Download historical stock data
        stock_data = yf.download(ticker_input, period="3y")
        df =stock_data
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = df.columns.droplevel(1)
        df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        
        cg, vola, shar, sor, md, calm = CAGR(df), volatility(df), sharpe(df, 0.06), sortino(df, 0.06), max_dd(df), calmar(df)
         # Display Metrics
        st.write("### Stock Analysis Metrics")
        metrics = {
                "CAGR (%)": cg * 100,
                "Volatility (%)": vola * 100,
                "Sharpe Ratio": shar,
                "Sortino Ratio": sor,
                "Max Drawdown (%)": md * 100,
                "Calmar Ratio": calm
            }
        st.write(metrics)

        if stock_data.empty:
            st.error(f"No data found for ticker: {ticker_input}")
        else:
            

            # Calculate Daily Log Returns
            stock_data["Returns"] = np.log(stock_data["Close"] / stock_data["Close"].shift(1)).dropna()

            # Monte Carlo Simulation
            S0 = stock_data["Close"].iloc[-1]  # Last known stock price
            mu = stock_data["Returns"].mean()  # Mean return
            sigma = stock_data["Returns"].std()  # Standard deviation

            t_intervals = 250  # Number of future days to simulate
            iterations = 1000  # Number of simulation runs

            daily_returns = np.exp(
                mu - (0.5 * sigma**2) + sigma * norm.ppf(np.random.rand(t_intervals, iterations))
            )

            price_list = np.zeros_like(daily_returns)
            price_list[0] = S0  # Start with last known price

            for t in range(1, t_intervals):
                price_list[t] = price_list[t - 1] * daily_returns[t]

            # Monte Carlo Results
            st.write("### Monte Carlo Simulation Results")
            st.write(f"📈 **Max Price:** {price_list.max():.2f}")
            st.write(f"📉 **Min Price:** {price_list.min():.2f}")
            st.write(f"📊 **Average Price:** {price_list.mean():.2f}")

            # Plot Monte Carlo Simulation
            st.write("### Monte Carlo Simulation Chart")
            fig = go.Figure()
            for i in range(iterations):
                fig.add_trace(go.Scatter(y=price_list[:, i], mode="lines", line=dict(width=0.5), opacity=0.1, showlegend=False))
            fig.update_layout(title="Monte Carlo Simulation", xaxis_title="Days", yaxis_title="Stock Price")
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error fetching data: {e}")