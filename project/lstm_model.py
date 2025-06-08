import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def predict_lstm(ticker):
    df = yf.download(ticker, period="1y")['Close'].pct_change().dropna().values
    window = 10
    X, y = [], []
    for i in range(len(df) - window - 1):
        X.append(df[i:i + window])
        y.append(1 if df[i + window] > 0 else 0)
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(32, input_shape=(window, 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X[..., np.newaxis], y, epochs=10, verbose=0)

    pred = model.predict(X[-1][np.newaxis, ..., np.newaxis])[0][0]
    return "📈 Up" if pred > 0.5 else "📉 Down"