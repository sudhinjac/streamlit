import streamlit as st
from utils import get_data_with_indicators, scale_features
from sentiment import fetch_sentiment_score
from lstm_model import predict_lstm
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
st.title("🧠 Smart Stock Movement Predictor")

ticker = st.text_input("Enter Stock Ticker", "INFY.NS")

if ticker:
    df = get_data_with_indicators(ticker)
    if df is not None and len(df) > 100:
        X = df.drop(columns=["Target"])
        y = df["Target"]

        X_scaled = scale_features(X)

        # Time-aware split
        X_train, X_test = X_scaled[:-60], X_scaled[-60:]
        y_train, y_test = y[:-60], y[-60:]

        # XGBoost
        model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)

        st.metric("XGBoost Accuracy", f"{accuracy:.2%}")
        st.write("🔮 Prediction for Tomorrow:", "📈 Up" if model.predict([X_scaled[-1]])[0] else "📉 Down")

        # LSTM prediction
        lstm_direction = predict_lstm(ticker)
        st.write("📊 LSTM Model:", lstm_direction)

        # Sentiment
        sentiment = fetch_sentiment_score(ticker.split('.')[0])
        st.write("📰 Sentiment Score:", sentiment)

        st.line_chart(df["Close"])
    else:
        st.warning("Insufficient or invalid stock data.")