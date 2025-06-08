import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Technical Indicators
def add_indicators(df):
    df['Return'] = df['Close'].pct_change()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['Volatility'] = df['Close'].rolling(10).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2

    df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)
    return df.dropna()

# Streamlit App
st.title("📈 Advanced Stock Movement Predictor (XGBoost)")

ticker = st.text_input("Enter Stock Ticker:", value="AAPL")

if ticker:
   
    stock_data = yf.download(ticker, period="2y")
    df = stock_data
    df.columns = df.columns.droplevel(1)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    if df.empty:
        st.error("Invalid ticker or no data.")
    else:
        df = add_indicators(df)
        features = ['MA10', 'MA50', 'Volatility', 'RSI', 'MACD']
        X = df[features]
        y = df['Target']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split (time-aware)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

        # XGBoost Model
        model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.subheader("📊 Model Accuracy")
        st.write(f"✅ Accuracy: {acc:.2%}")

        st.subheader("📌 Prediction for Next Day")
        latest = scaler.transform([X.iloc[-1]])
        prediction = model.predict(latest)[0]
        st.write("📈 Up" if prediction == 1 else "📉 Down")

        st.line_chart(df[['Close']])