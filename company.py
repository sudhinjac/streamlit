import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as stats
import plotly.graph_objects as go

# -------------------------------
# Value at Risk Monte Carlo Class
# -------------------------------
class ValueAtRiskMonteCarlo:
    def __init__(self, S, mu, sigma, c, n, iterations):
        self.S = S            # Investment amount
        self.mu = mu          # Mean daily return
        self.sigma = sigma    # Daily volatility (std dev)
        self.c = c            # Confidence level (e.g., 0.95)
        self.n = n            # Time horizon in days
        self.iterations = iterations  # Number of simulation iterations

    def simulation(self):
        # Generate random normal values: shape (1, iterations)
        rand = np.random.normal(0, 1, [1, self.iterations])
        # Simulate stock price using geometric Brownian motion formula
        stock_price = self.S * np.exp(self.n * (self.mu - 0.5 * self.sigma**2) + self.sigma * np.sqrt(self.n) * rand)
        stock_price = np.sort(stock_price)
        # Compute the percentile value at the (1 - c)*100 level
        percentile = np.percentile(stock_price, (1 - self.c) * 100)
        return self.S - percentile

# -------------------------------
# Technical Indicator Functions
# -------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # Classify RSI Trend: Overbought (>60) -> Bullish, Oversold (<40) -> Bearish, otherwise Neutral
    if rsi.iloc[-1] > 60:
        return "Bullish"
    elif rsi.iloc[-1] < 40:
        return "Bearish"
    else:
        return "Neutral"

def calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    macd_hist = macd - signal
    # Classify MACD Trend: positive histogram -> Bullish, negative -> Bearish, near zero -> Neutral
    if macd_hist.iloc[-1] > 0:
        return "Bullish"
    elif macd_hist.iloc[-1] < 0:
        return "Bearish"
    else:
        return "Neutral"

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.title("📈 Stock Analysis App")
st.write("Enter a stock ticker (e.g., RELIANCE.NS) to see its financial metrics, technical indicators, and candlestick chart.")

ticker_input = st.text_input("Enter Stock Ticker:", "RELIANCE.NS")

if ticker_input:
    try:
        # Download historical data for the given ticker (3 years)
        stock_data = yf.download(ticker_input, period="3y")
        if stock_data.empty:
            st.error(f"No data found for ticker: {ticker_input}")
        else:
            # Calculate daily log returns for the stock
            stock_data["Returns"] = np.log(stock_data["Close"] / stock_data["Close"].shift(1))
            
            # Download benchmark market data (^NSEI) and compute its returns
            market_data = yf.download("^NSEI", period="3y")["Close"]
            market_returns = np.log(market_data / market_data.shift(1)).dropna()
            market_var = market_returns.var() * 252
            market_annual_return = market_returns.mean() * 252 * 100
            
            # Align stock returns with market returns
            stock_returns = stock_data["Returns"].dropna()
            common_index = stock_returns.index.intersection(market_returns.index)
            stock_returns = stock_returns.loc[common_index]
            market_returns_aligned = market_returns.loc[common_index]
            
            # Beta Calculation
            if len(stock_returns) > 0:
                stock_cov = np.cov(stock_returns, market_returns_aligned)[0, 1] * 252
                beta = stock_cov / market_var
            else:
                beta = np.nan
            
            # Annual Return and Volatility
            annual_return = stock_returns.mean() * 252 * 100
            volatility = stock_returns.std() * np.sqrt(252) * 100
            
            # Sharpe Ratio (using 7% as risk-free rate as per your code)
            sharpe_ratio = (annual_return - 7) / volatility if volatility != 0 else np.nan
            
            # Alpha and Treynor Ratio
            alpha = annual_return - (7 + beta * (market_annual_return - 7)) if beta is not None else np.nan
            trynor_ratio = (annual_return - 7) / beta if beta and beta != 0 else np.nan
            
            # CAGR Calculation over 3 years
            CAGR = ((stock_data["Close"].iloc[-1] / stock_data["Close"].iloc[0]) ** (1 / 3) - 1) * 100
            
            # Maximum Drawdown Calculation
            roll_max = stock_data["Close"].cummax()
            drawdown = roll_max - stock_data["Close"]
            max_dd = (drawdown / roll_max).max() * 100
            
            # Monte Carlo Value at Risk (VaR) Calculation
            S = 100000     # Investment amount
            conf = 0.95    # Confidence level
            horizon = 730  # 1 day horizon (approximation using trading days)
            iterations = 100000  # Number of simulation iterations
            mu = stock_returns.mean()
            sigma = stock_returns.std()
            var_model = ValueAtRiskMonteCarlo(S, mu, sigma, conf, horizon, iterations)
            var_value = var_model.simulation()
            
            # Probability of Loss and Odds Calculation (using annual return and volatility as proxies)
            z_score = (0 - annual_return) / volatility if volatility != 0 else 0
            prob_loss = stats.norm.cdf(z_score) * 100
            prob_win = (1 - stats.norm.cdf(z_score)) * 100
            odds_win = (prob_win / prob_loss) if prob_loss > 0 else np.inf
            
            # Technical Indicators: RSI, MACD, Moving Averages
            rsi_trend = calculate_rsi(stock_data["Close"])
            macd_trend = calculate_macd(stock_data["Close"])
            dma_50 = stock_data["Close"].rolling(window=50).mean().iloc[-1]
            dma_100 = stock_data["Close"].rolling(window=100).mean().iloc[-1]
            if dma_50 > dma_100:
                ma_trend = "Bullish"
            elif dma_50 < dma_100:
                ma_trend = "Bearish"
            else:
                ma_trend = "Neutral"
            
            # Build a results DataFrame
            metrics = {
                "Metric": ["Beta", "Volatility%", "Return%", "Sharpe Ratio", "CAGR", "MAXDD%", "Value at Risk", 
                           "Alpha", "Trynor", "Prob. Loss", "Prob. Win", "Odds Winning", "RSI Trend", "MACD Trend", "MA Trend"],
                "Value": [beta, volatility, annual_return, sharpe_ratio, CAGR, max_dd, var_value, alpha, trynor_ratio,
                          prob_loss, prob_win, odds_win, rsi_trend, macd_trend, ma_trend]
            }
            results_df = pd.DataFrame(metrics)
            
            # Display the results DataFrame in Streamlit
            st.write("### Stock Analysis Results")
            st.dataframe(results_df)
            
            # Create and display a candlestick chart using Plotly
            st.write("### Candlestick Chart")
            fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                                 open=stock_data["Open"],
                                                 high=stock_data["High"],
                                                 low=stock_data["Low"],
                                                 close=stock_data["Close"])])
            fig.update_layout(title=f"{ticker_input} Candlestick Chart",
                              xaxis_title="Date",
                              yaxis_title="Price",
                              xaxis_rangeslider_visible=False)
            st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error fetching data: {e}")