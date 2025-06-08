import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# ----------------------
# Helper: LSTM Model
# ----------------------
class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, output_dim=2):
        super(StockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        out, _ = self.lstm(x, (h0,c0))
        out = self.fc(out[:, -1, :])
        return out

# ----------------------
# Fetch stock data
# ----------------------
@st.cache_data
def load_stock_data(ticker, ):
    stock_data = yf.download(ticker, period="3y")
    df = stock_data
    df.columns = df.columns.droplevel(1)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    return df

# ----------------------
# Feature engineering for ML models
# ----------------------
def compute_technical_indicators(df):
    df['Return'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI_14'] = compute_RSI(df['Close'], 14)
    df['MACD'] = compute_MACD(df['Close'])
    df = df.dropna()
    return df

def compute_RSI(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_gain = up.rolling(window=period).mean()
    avg_loss = down.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

# ----------------------
# Prepare features & labels for XGBoost
# ----------------------
def prepare_features(df):
    df_feat = df[['Return', 'SMA_10', 'SMA_20', 'RSI_14', 'MACD']].copy()
    df_feat = df_feat.dropna()
    X = df_feat.values[:-1]
    y = (df['Close'].shift(-1) > df['Close']).astype(int).iloc[:-1].values
    return X, y, df_feat.index[:-1]

# ----------------------
# Train or load XGBoost model
# ----------------------
def train_xgb(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

# ----------------------
# Predict with LSTM model
# ----------------------
def prepare_lstm_input(df_feat):
    data = df_feat.values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    seq_len = 10
    X_lstm = []
    for i in range(len(data_scaled)-seq_len):
        X_lstm.append(data_scaled[i:i+seq_len])
    X_lstm = np.array(X_lstm)
    return torch.tensor(X_lstm, dtype=torch.float32)

def train_lstm(X_train, y_train):
    input_dim = X_train.shape[2]
    model = StockLSTM(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

# ----------------------
# Sentiment analysis on recent news headlines (dummy example)
# ----------------------
def sentiment_score(ticker):
    analyzer = SentimentIntensityAnalyzer()
    # Normally, you would scrape or fetch recent headlines for ticker.
    # For demo, let's just simulate neutral sentiment:
    return 0.0  # placeholder for sentiment compound score

# ----------------------
# Streamlit UI & Logic
# ----------------------
def main():
    st.title("Stock Movement Prediction with Ensemble Voting")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TCS.NS):", value="AAPL")
    
    if ticker:
        with st.spinner("Fetching data and running models..."):
            df = load_stock_data(ticker)
            if df.empty:
                st.error("Could not fetch data for ticker.")
                return
            
            df = compute_technical_indicators(df)
            X, y, indices = prepare_features(df)
            if len(X) < 50:
                st.warning("Not enough data to train models reliably.")
                return
            
            # Train XGBoost
            xgb_model = train_xgb(X, y)
            xgb_pred = xgb_model.predict(X)
            
            # Prepare LSTM inputs and train
            seq_len = 10
            X_lstm = prepare_lstm_input(df[['Return', 'SMA_10', 'SMA_20', 'RSI_14', 'MACD']].iloc[:-1])
            y_lstm = torch.tensor(y[seq_len:], dtype=torch.long)
            lstm_model = train_lstm(X_lstm, y_lstm)
            lstm_model.eval()
            with torch.no_grad():
                lstm_logits = lstm_model(X_lstm)
                lstm_pred = torch.argmax(lstm_logits, axis=1).numpy()
            
            # Sentiment score (simplified)
            senti = sentiment_score(ticker)
            senti_pred = 1 if senti > 0 else 0  # positive or negative sentiment
            
            # Ensemble Voting (majority vote)
            combined_preds = []
            for i in range(len(lstm_pred)):
                votes = [xgb_pred[i+seq_len], lstm_pred[i], senti_pred]
                combined = 1 if sum(votes) >= 2 else 0
                combined_preds.append(combined)
            
            combined_preds = np.array(combined_preds)
            actuals = y[seq_len:]
            
            accuracy = (combined_preds == actuals).mean()
            
            st.write(f"Prediction accuracy on historical data (backtest): **{accuracy*100:.2f}%**")
            
            # Show predictions table
            results_df = pd.DataFrame({
                'Date': indices[seq_len:],
                'Actual': actuals,
                'XGBoost_Pred': xgb_pred[seq_len:],
                'LSTM_Pred': lstm_pred,
                'Sentiment_Pred': senti_pred,
                'Ensemble_Pred': combined_preds
            })
            st.dataframe(results_df.set_index('Date'))
            
            # Export CSV
            if st.button("Export Predictions to CSV"):
                results_df.to_csv(f"{ticker}_ensemble_predictions.csv", index=False)
                st.success("CSV exported!")
    
if __name__ == "__main__":
    main()