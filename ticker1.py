import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import norm
import cufflinks as cf
from scipy import stats
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import norm
import cufflinks as cf
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import feedparser

cf.go_offline()
st.set_page_config(page_title="Stock Analysis & Monte Carlo Simulation", layout="wide")

# Set Light Yellow Background Color
st.markdown("""
    <style>
        .stApp {
            background-color: #FFFFCC;  /* Light Yellow */
        }
    </style>
""", unsafe_allow_html=True)
# -------------------------------
# Streamlit App Layout
# -------------------------------
#st.set_page_config(page_title="Stock Analysis & Monte Carlo Simulation", layout="wide")
st.title("📈 Stock Analysis & Monte Carlo Simulation")
def ichimoku(df):
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2  # Conversion Line

    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2  # Base Line

    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)  # Leading Span A

    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)  # Leading Span B

    df['Chikou_Span'] = df['Close'].shift(-26)  # Lagging Span
    return df

def generate_signal(df):
    signals = []
    for i in range(1, len(df)):
        if df['Tenkan_sen'][i] > df['Kijun_sen'][i] and df['Close'][i] > df['Senkou_Span_A'][i]:
            signals.append("BUY")
        elif df['Tenkan_sen'][i] < df['Kijun_sen'][i] and df['Close'][i] < df['Senkou_Span_A'][i]:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    signals.insert(0, "HOLD")
    df['Signal'] = signals
    return df

def get_google_news(company):
    base_url = "https://news.google.com/rss/search?q="
    query = company.replace(" ", "+") + "+stock"
    url = base_url + query
    feed = feedparser.parse(url)

    headlines = [entry.title for entry in feed.entries[:10]]
    return headlines

def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}
    
    for headline in headlines:
        score = analyzer.polarity_scores(headline)
        if score['compound'] >= 0.05:
            sentiments["Positive"] += 1
        elif score['compound'] <= -0.05:
            sentiments["Negative"] += 1
        else:
            sentiments["Neutral"] += 1
    
    return sentiments

def get_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    cashflow = stock.cashflow
    income_statement = stock.financials
    
    return balance_sheet, cashflow, income_statement
def bollinger_bands(df, window=20, std_dev=2):
    df['MA'] = df['Close'].rolling(window=window).mean()
    df['Upper_BB'] = df['MA'] + (df['Close'].rolling(window=window).std() * std_dev)
    df['Lower_BB'] = df['MA'] - (df['Close'].rolling(window=window).std() * std_dev)

# Function to calculate RSI
def rsi(df, window=14):
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

# Function to calculate MACD
def macd(df, short_window=12, long_window=26, signal_window=9):
    df['Short_EMA'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['Long_EMA'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['Short_EMA'] - df['Long_EMA']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()

# Function to calculate moving averages
def moving_averages(df):
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df['100_MA'] = df['Close'].rolling(window=100).mean()

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

def get_market_data():
    market_data = yf.download('^NSEI', period="3y")
    market_returns = np.log(market_data["Close"] / market_data["Close"].shift(1))
    return market_data, market_returns

def get_stock_metrics(t):
    data = pd.DataFrame()
    data['^NSEI'] = yf.download('^NSEI', period="3y").Close
    market_returns = np.log(data['^NSEI'] / data['^NSEI'].shift(1))
    mr = market_returns.mean() * 252 * 100
    market_var = market_returns.var() * 252
    data[t] = yf.download(t, period="3y").Close
    sec_returns = np.log(data / data.shift(1))
    cov = sec_returns.cov() * 252
    cov_with_market = cov[t][0]
    beta = cov_with_market / market_var
    expected_return = (sec_returns[t].mean() * 252) * 100
    vols = sec_returns[t].std() * np.sqrt(252) * 100
    sharpe_ratio = (expected_return - 7) / vols
    treynor_ratio = (expected_return - 7) / beta
    jensen_alpha = expected_return - (7 + beta * (mr - 7))
    cv = vols / expected_return
    z_score = (0 - expected_return) / vols
    pl = stats.norm.cdf(z_score) * 100
    pw = (1 - stats.norm.cdf(z_score)) * 100
    ow = pw / pl if pl > 0 else np.inf
    return beta, sharpe_ratio, treynor_ratio, jensen_alpha, cv, pl, pw, ow,vols

def compute_rsi(data, period=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_stochastic_oscillator(data, period=14):
    low_min = data['Low'].rolling(window=period).min()
    high_max = data['High'].rolling(window=period).max()
    stoch_k = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    return stoch_k

def compute_bollinger_bands(data, period=20):
    sma = data['Close'].rolling(window=period).mean()
    std_dev = data['Close'].rolling(window=period).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band

def compute_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = np.max([high_low, high_close, low_close], axis=0)
    atr = pd.Series(true_range).rolling(window=period).mean()
    return atr

def compute_macd(data, short_period=12, long_period=26, signal_period=9):
    short_ema = data['Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def make_decision(data):
    rsi = compute_rsi(data)
    stoch_k = compute_stochastic_oscillator(data)
    upper_band, lower_band = compute_bollinger_bands(data)
    atr = compute_atr(data)
    macd, signal = compute_macd(data)
    
    last_rsi = rsi.iloc[-1]
    last_stoch = stoch_k.iloc[-1]
    last_close = data['Close'].iloc[-1]
    last_macd = macd.iloc[-1]
    last_signal = signal.iloc[-1]
    
    buy_signal = (last_rsi < 30 and last_stoch < 20 and last_macd > last_signal)
    sell_signal = (last_rsi > 70 and last_stoch > 80 and last_macd < last_signal)
    
    if buy_signal:
        return "Buy now"
    elif sell_signal:
        return "Wait, price may go down"
    else:
        return "Hold"

ticker_input = st.text_input("Enter Stock Ticker:", "TANLA.NS")

if ticker_input:
    stock_data = yf.download(ticker_input, period="3y")
    df = stock_data
    df.columns = df.columns.droplevel(1)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    market_data, market_returns = get_market_data()
    beta, sharpe, treynor, jensen_alpha, cv, loss_prob, profit_prob, ow,vols = get_stock_metrics(ticker_input)
    st.write(df.tail(5))
    bollinger_bands(df)
    rsi(df)
    macd(df)
    moving_averages(df)
    df.reset_index(inplace=True)
    df =ichimoku(df)
    df = generate_signal(df)
    metrics = {
        "CAGR (%)": CAGR(df) * 100,
        "Volatility (%)": vols,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino(df, 0.06),
        "Max Drawdown (%)": max_dd(df) * 100,
        "Calmar Ratio": calmar(df),
        "Jensen's Alpha": jensen_alpha,
        "Treynor Ratio": treynor,
        "Coefficient of Variation (CV)": cv*100,
        "Loss Probability (%)": loss_prob,
        "Profit Probability (%)": profit_prob,
        "Odds of Profit": ow
    }
    st.write("### Stock Analysis Metrics")
    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])

# Apply styling for bold headers
    styled_metrics_df = metrics_df.style.set_properties(**{'font-weight': 'bold'}, subset=['Metric'])

    # Display the DataFrame in Streamlit
    st.write("### 📊 Stock Analysis Metrics")
    st.dataframe(styled_metrics_df)

    # Monte Carlo Simulation
    stock_data["Returns"] = np.log(stock_data["Close"] / stock_data["Close"].shift(1))
    stock_data = stock_data.dropna()  # Drop NaN values
    
    S0 = stock_data["Close"].iloc[-1]  # Last known stock price
    mu = stock_data["Returns"].mean()  # Mean return
    sigma = stock_data["Returns"].std()  # Standard deviation

    t_intervals = 250  # Number of future days to simulate
    iterations = 1000  # Number of simulation runs

    # Generate random daily returns
    daily_returns = np.exp((mu - 0.5 * sigma**2) + sigma * norm.ppf(np.random.rand(t_intervals, iterations)))

    # Initialize price matrix
    price_list = np.zeros((t_intervals, iterations))
    price_list[0] = S0  # Set first row as the last known price

    # Simulate future stock prices
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
    fig = go.Figure()

    # Add Candlestick trace
    fig = go.Figure()

# Add Candlestick trace
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name="Candlestick"))

    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_BB'], line=dict(color='blue', width=1), name="Upper BB"))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_BB'], line=dict(color='blue', width=1), name="Lower BB"))

    # Add Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['50_MA'], line=dict(color='red', width=1.5), name="50-day MA"))
    fig.add_trace(go.Scatter(x=df.index, y=df['100_MA'], line=dict(color='green', width=1.5), name="100-day MA"))

    # Add Volume Bars (Color-coded)
    colors = ['green' if df['Close'][i] > df['Open'][i] else 'red' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Volume", opacity=0.5, yaxis="y2"))

    # Update layout with secondary y-axis for volume
    fig.update_layout(
        title=f"{ticker_input} - Candlestick Chart with Indicators & Volume",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig)

    # Plot RSI Indicator
    st.write("### Relative Strength Index (RSI)")
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1.5), name="RSI"))
    rsi_fig.add_hline(y=70, line=dict(color='red', dash='dash'), name="Overbought")
    rsi_fig.add_hline(y=30, line=dict(color='green', dash='dash'), name="Oversold")
    rsi_fig.update_layout(yaxis_title="RSI", xaxis_title="Date")
    st.plotly_chart(rsi_fig)

    # Plot MACD Indicator
    st.write("### Moving Average Convergence Divergence (MACD)")
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue', width=1.5), name="MACD"))
    macd_fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], line=dict(color='orange', width=1.5), name="Signal Line"))
    macd_fig.update_layout(yaxis_title="MACD", xaxis_title="Date")
    st.plotly_chart(macd_fig)
    st.write("## 🏦 Fundamental Analysis")
    
    balance_sheet, cashflow, income_statement = get_fundamentals(ticker_input)

    st.write("### 📊 Profit & Loss Statement")
    st.dataframe(income_statement)

    st.write("### 💰 Cash Flow Statement")
    st.dataframe(cashflow)
    st.write("#### Balance Sheet")
    st.dataframe(balance_sheet)
    data = df

# Step 2: Feature Engineering
data['Return'] = data['Close'].pct_change()
data['50_MA'] = data['Close'].rolling(window=50).mean()
data['200_MA'] = data['Close'].rolling(window=200).mean()
data['RSI'] = 100 - (100 / (1 + data['Close'].diff(1).apply(lambda x: max(x, 0)).rolling(14).mean() / 
              data['Close'].diff(1).apply(lambda x: abs(x)).rolling(14).mean()))
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - \
               data['Close'].ewm(span=26, adjust=False).mean()

# Filling NaN values
data.fillna(0, inplace=True)

# Step 3: Labeling
# Define a threshold for labeling (for example, a return above 1% is bullish, below -1% is bearish)
threshold = 0.01
data['Label'] = np.where(data['Return'] > threshold, 1, 0)  # 1 for bullish, 0 for bearish

# Step 4: Model Training
features = data[['50_MA', '200_MA', 'RSI', 'MACD']].values
labels = data['Label'].values

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Prediction and Evaluation
y_pred = model.predict(X_test)

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
#st.write("Confusion Matrix:")
#st.write(conf_matrix)

# Print classification report
class_report = classification_report(y_test, y_pred)
#st.write("Classification Report:")
#st.write(class_report)

# Backtest: Predicting the next day's movement using the model
next_day_data = data[['50_MA', '200_MA', 'RSI', 'MACD']].iloc[-1].values.reshape(1, -1)
next_day_prediction = model.predict_proba(next_day_data)
st.write(f"Probability of being Bullish: {next_day_prediction[0][1]:.2f}")
st.write(f"Probability of being Bearish: {1 - next_day_prediction[0][1]:.2f}")
decision = make_decision(df)
st.write(f"Decision: {decision}")

#st.set_page_config(page_title="Ichimoku Cloud Analysis", page_icon="📈", layout="wide")

# Now proceed with the rest of your code
st.title("Ichimoku Cloud Stock Analysis")


    
st.write("### Stock Data with Ichimoku Indicators")
st.dataframe(df[['Date', 'Close', 'Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A', 'Senkou_Span_B', 'Chikou_Span', 'Signal']].tail(50))
    
st.write("### Ichimoku Cloud Chart")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
fig.add_trace(go.Scatter(x=df['Date'], y=df['Tenkan_sen'], mode='lines', name='Tenkan-sen', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df['Date'], y=df['Kijun_sen'], mode='lines', name='Kijun-sen', line=dict(color='red')))
fig.add_trace(go.Scatter(x=df['Date'], y=df['Senkou_Span_A'], mode='lines', name='Senkou Span A', fill='tonexty', line=dict(color='green')))
fig.add_trace(go.Scatter(x=df['Date'], y=df['Senkou_Span_B'], mode='lines', name='Senkou Span B', fill='tonexty', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=df['Date'], y=df['Chikou_Span'], mode='lines', name='Chikou Span', line=dict(color='purple')))
    
fig.update_layout(title="Ichimoku Cloud Analysis", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False)
st.plotly_chart(fig)
    
last_signal = df['Signal'].iloc[-1]
if last_signal == "BUY":
    st.success("📈 Strong Buy Signal Detected! Consider Buying the Stock.")
elif last_signal == "SELL":
    st.error("📉 Sell Signal Detected! Consider Selling the Stock.")
else:
    st.warning("⚠ No Clear Signal. Hold Your Position.")

st.write("### 📢 Sentiment Analysis Based on Google News")
company_name = st.text_input("Enter Company Name for Sentiment Analysis:", "Tanla platforms")

company_name = st.text_input("Enter Company Name for Sentiment Analysis:", "Tanla Platforms")

if company_name:
    news_headlines = get_google_news(company_name)
    
    if news_headlines:
        sentiments = analyze_sentiment(news_headlines)
        total = sum(sentiments.values())

        if total > 0:
            fig, ax = plt.subplots()
            ax.pie(
                sentiments.values(), 
                labels=sentiments.keys(), 
                autopct='%1.1f%%', 
                colors=['green', 'grey', 'red'], 
                startangle=90
            )
            ax.axis('equal')
            ax.set_title(f"Sentiment Analysis for {company_name}")
            st.pyplot(fig)
        else:
            st.warning("Sentiment analysis returned all zero values. Possibly insufficient data.")
        
        st.markdown("### 📰 Latest News Headlines")
        for headline in news_headlines:
            st.markdown(f"- {headline}")
    else:
        st.error("No news headlines found. Try a different company name or check your connection.")
