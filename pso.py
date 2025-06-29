import streamlit as st
import pandas as pd
import yfinance as yf
import torch
import numpy as np
from datetime import datetime
from scipy.stats import linregress
from pyswarm import pso

st.set_page_config(page_title="Stock Optimizer", layout="wide")
st.title("📈 Portfolio Optimizer using PSO & Alpha")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.success(f"Running on device: {device}")

uploaded_file = st.file_uploader("Upload CSV with 'Ticker' column", type=['csv'])

if uploaded_file:
    dfticker = pd.read_csv(uploaded_file)
    tickers = dfticker['Ticker'].dropna().astype(str).str.strip().tolist()
    tickers = [t if t.endswith('.NS') or t.endswith('.BO') else t + '.NS' for t in tickers]

    st.info("📥 Downloading price data...")
    data_list = []
    for t in tickers:
        try:
            df = yf.download(t, period="3y")[['Close']].rename(columns={'Close': t})
            data_list.append(df)
        except Exception as e:
            st.warning(f"Error downloading {t}: {e}")

    price_df = pd.concat(data_list, axis=1)
    price_df = price_df.dropna(axis=1, thresh=int(0.9 * len(price_df))).dropna()

    if price_df.empty:
        st.error("No valid price data found. Try with different tickers.")
        st.stop()

    market = yf.download("^NSEI", start=price_df.index.min(), end=price_df.index.max())['Close'].ffill().bfill()
    market_returns = market.pct_change().dropna()

    common_index = price_df.index.intersection(market_returns.index)
    price_df = price_df.loc[common_index]
    market_returns = market_returns.loc[common_index]

    price_tensor = torch.tensor(price_df.values, dtype=torch.float32, device=device)
    market_tensor = torch.tensor(market_returns.values, dtype=torch.float32, device=device)

    def portfolio_metrics_score(weights):
        weights = np.array(weights)
        weights /= np.sum(weights)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

        if weights_tensor.shape[0] != price_tensor.shape[1]:
            return 1e6

        portfolio_prices = torch.matmul(price_tensor, weights_tensor)
        if len(portfolio_prices) < 2:
            return 1e6

        portfolio_returns = portfolio_prices[1:] / portfolio_prices[:-1] - 1

        if torch.isnan(portfolio_returns).any() or torch.isinf(portfolio_returns).any():
            return 1e6

        cagr = (portfolio_prices[-1] / portfolio_prices[0]).item() ** (252.0 / len(portfolio_prices)) - 1
        volatility = portfolio_returns.std().item() * np.sqrt(252)
        sharpe = (portfolio_returns.mean() / portfolio_returns.std()).item() * np.sqrt(252)
        cumulative_return = (portfolio_prices[-1] / portfolio_prices[0]).item() - 1

        portfolio_np = portfolio_returns.cpu().numpy()
        market_np = market_tensor[:len(portfolio_returns)].cpu().numpy()

        slope, intercept, *_ = linregress(market_np, portfolio_np)
        beta = slope
        alpha = intercept * 252

        score = (
            -1.0 * cagr
            -1.0 * sharpe
            -1.0 * alpha
            -1.0 * cumulative_return
            +1.0 * volatility
            +1.0 * abs(beta - 1)
        )
        return score

    st.info("⚙️ Running Particle Swarm Optimization...")
    num_assets = price_df.shape[1]
    lb = [0] * num_assets
    ub = [1] * num_assets

    best_weights, best_score = pso(portfolio_metrics_score, lb, ub, swarmsize=100, maxiter=100, debug=False)
    best_weights /= np.sum(best_weights)

    # Final results
    final_prices = torch.matmul(price_tensor, torch.tensor(best_weights, dtype=torch.float32, device=device))
    final_returns = final_prices[1:] / final_prices[:-1] - 1

    cagr = (final_prices[-1] / final_prices[0]).item() ** (252.0 / len(final_prices)) - 1
    volatility = final_returns.std().item() * np.sqrt(252)
    sharpe = (final_returns.mean() / final_returns.std()).item() * np.sqrt(252)
    cumulative_return = (final_prices[-1] / final_prices[0]).item() - 1

    aligned_portfolio = final_returns.cpu().numpy()
    aligned_market = market_tensor[:len(final_returns)].cpu().numpy()
    slope, intercept, *_ = linregress(aligned_market, aligned_portfolio)
    beta = slope
    alpha = intercept * 252

    st.subheader("📊 Portfolio Metrics")
    st.metric("CAGR", f"{cagr:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    st.metric("Alpha", f"{alpha:.2f}")
    st.metric("Beta", f"{beta:.2f}")
    st.metric("Volatility", f"{volatility:.2%}")
    st.metric("Cumulative Return", f"{cumulative_return:.2%}")

    result_df = pd.DataFrame({
        'Ticker': price_df.columns,
        'Weight': best_weights
    }).sort_values(by='Weight', ascending=False)

    st.subheader("📌 Optimized Portfolio Weights")
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Weights as CSV", csv, "optimized_portfolio.csv", "text/csv")
