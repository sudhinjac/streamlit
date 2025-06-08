import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
st.set_page_config(page_title="Stock ML Dashboard", layout="wide")

# Sidebar Inputs
st.sidebar.header("Stock Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Load Stock Data
@st.cache_data
def load_data(ticker, start, end):
    stock_data = yf.download(ticker, start,end)
    df = stock_data
    df.columns = df.columns.droplevel(1)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df

data = load_data(ticker, start_date, end_date)
st.title(f"📊 Stock ML Dashboard – {ticker}")

# Add Technical Indicators
data['RSI'] = RSIIndicator(data['Close']).rsi()
data['MACD'] = MACD(data['Close']).macd_diff()
data['SMA'] = SMAIndicator(data['Close'], window=14).sma_indicator()

bb = BollingerBands(close=data['Close'])
data['BB_High'] = bb.bollinger_hband()
data['BB_Low'] = bb.bollinger_lband()

atr = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'])
data['ATR'] = atr.average_true_range()

adx = ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'])
data['ADX'] = adx.adx()
data.dropna(inplace=True)

# Price Chart
st.subheader("Price Chart with SMA and Bollinger Bands")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data['Close'], label='Close Price')
ax.plot(data['SMA'], label='SMA 14', linestyle='--')
ax.plot(data['BB_High'], label='BB High', linestyle=':')
ax.plot(data['BB_Low'], label='BB Low', linestyle=':')
ax.legend()
st.pyplot(fig)

# ML Prediction Target
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
features = ['RSI', 'MACD', 'SMA', 'BB_High', 'BB_Low', 'ATR', 'ADX']
X = data[features]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Model Training with XGBoost
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# Show Accuracy
st.subheader("📈 ML Model Accuracy")
st.metric(label="Prediction Accuracy", value=f"{acc * 100:.2f}%")

# Predictions vs Actual
st.subheader("📌 Predictions vs Actual")
pred_df = data.iloc[-len(y_test):].copy()
pred_df['Predicted'] = preds
pred_df['Actual'] = y_test.values
st.dataframe(pred_df[['Close', 'RSI', 'MACD', 'SMA', 'ATR', 'ADX', 'Actual', 'Predicted']].tail(10))

# Confusion Matrix
st.subheader("📊 Confusion Matrix")
cm = pd.crosstab(pred_df['Actual'], pred_df['Predicted'], rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
st.pyplot(plt.gcf())

# News Sentiment Analysis
st.subheader("📰 Google News Sentiment")

def get_sentiment(ticker):
    try:
        # Use Google News RSS for reliability
        url = f"https://news.google.com/rss/search?q={ticker}+stock"
        feed = feedparser.parse(url)

        headlines = [entry.title for entry in feed.entries[:10]]
        if not headlines:
            return None, []

        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]

        average_score = sum(scores) / len(scores)
        return average_score, headlines
    except Exception as e:
        return None, []

# Usage in Streamlit
sentiment_score, headlines = get_sentiment(ticker)

if sentiment_score is not None:
    st.metric("🧠 Sentiment Score", f"{sentiment_score:.2f}")
    
    st.subheader("📰 Latest News Headlines")
    for h in headlines:
        st.write(f"- {h}")
else:
    st.warning("No sentiment data found. Try a different ticker or check your connection.")


# Forecasting with Prophet
st.subheader("📊 Forecasting (Prophet)")
df_prophet = data[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
model = Prophet()
model.fit(df_prophet)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
fig1 = model.plot(forecast)
st.pyplot(fig1)
