import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Stress Testing App", layout="wide")
st.title("📉 Portfolio Stress Testing & Scenario Analysis")

st.sidebar.header("📌 Select Parameters")
tickers = st.sidebar.text_input("Enter comma-separated tickers (e.g., AAPL, MSFT, GOOGL)", "AAPL, MSFT, GOOGL")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*3))
end_date = st.sidebar.date_input("End Date", datetime.now())

scenarios = {
    "Mild Drop (-5%)": -0.05,
    "Moderate Crash (-15%)": -0.15,
    "Severe Crash (-30%)": -0.30
}

selected_scenario = st.sidebar.selectbox("Select Stress Scenario", list(scenarios.keys()))

run_test = st.sidebar.button("Run Stress Test")

if run_test:
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip() != ""]
    
    st.subheader("📊 Historical Prices")
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    data.dropna(axis=1, inplace=True)
    st.line_chart(data)

    st.subheader("📈 Portfolio Simulation")
    weights = np.array([1/len(data.columns)] * len(data.columns))
    latest_prices = data.iloc[-1].values
    portfolio_value = np.dot(weights, latest_prices)

    st.write(f"Initial Portfolio Value (Equal Weights): **${portfolio_value:,.2f}**")

    scenario_return = scenarios[selected_scenario]
    stressed_prices = latest_prices * (1 + scenario_return)
    stressed_value = np.dot(weights, stressed_prices)
    loss = portfolio_value - stressed_value
    loss_pct = loss / portfolio_value * 100

    st.metric("Stressed Portfolio Value", f"${stressed_value:,.2f}", delta=f"-{loss_pct:.2f}%")

    # Visualize impact
    df_vis = pd.DataFrame({
        'Before Stress': latest_prices,
        'After Stress': stressed_prices
    }, index=data.columns)
    df_vis.plot(kind='bar', figsize=(10, 4))
    st.pyplot(plt.gcf())

    st.subheader("💡 Interpretation")
    st.markdown(f"This scenario simulates a **{int(abs(scenario_return)*100)}% market drop**. The portfolio is expected to lose approximately **{loss_pct:.2f}%** of its value under this condition.")
