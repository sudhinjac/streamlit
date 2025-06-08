import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_data_with_indicators(ticker):
    df = yf.download(ticker, period="2y")
    if df.empty:
        return None
    df["Return"] = df["Close"].pct_change()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Close"].rolling(10).std()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["Close"].ewm(span=12).mean()
    exp2 = df["Close"].ewm(span=26).mean()
    df["MACD"] = exp1 - exp2

    df["Target"] = np.where(df["Return"].shift(-1) > 0, 1, 0)
    return df.dropna()

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)