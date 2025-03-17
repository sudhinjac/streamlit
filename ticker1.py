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
def get_google_news(company):
    search_url = f"https://www.google.com/search?q={company}+stock+news&hl=en&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    headlines = []
    for item in soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd"):
        headlines.append(item.get_text())
    
    return headlines[:30]  # Return top 10 news headlines

def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    
    for headline in headlines:
        score = analyzer.polarity_scores(headline)
        if score['compound'] >= 0.05:
            sentiments["positive"] += 1
        elif score['compound'] <= -0.05:
            sentiments["negative"] += 1
        else:
            sentiments["neutral"] += 1
    
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

ticker_input = st.text_input("Enter Stock Ticker:", "TANLA.NS")

if ticker_input:
    stock_data = yf.download(ticker_input, period="3y")
    df = stock_data
    df.columns = df.columns.droplevel(1)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    market_data, market_returns = get_market_data()
    beta, sharpe, treynor, jensen_alpha, cv, loss_prob, profit_prob, ow,vols = get_stock_metrics(ticker_input)
    bollinger_bands(df)
    rsi(df)
    macd(df)
    moving_averages(df)
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
st.write("### 📢 Sentiment Analysis Based on Google News")
company_name = st.text_input("Enter Company Name for Sentiment Analysis:", "Tanla platforms")

if company_name:
    news_headlines = get_google_news(company_name)
    sentiments = analyze_sentiment(news_headlines)
    
    fig, ax = plt.subplots()
    ax.pie(sentiments.values(), labels=sentiments.keys(), autopct='%1.1f%%', colors=['green', 'grey', 'red'])
    ax.set_title(f"Sentiment Analysis for {company_name}")
    
    st.pyplot(fig)
    
    st.write("### Latest News Headlines")
    for headline in news_headlines:
        st.write(f"- {headline}")
