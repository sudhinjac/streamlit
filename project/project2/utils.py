import pandas as pd
import numpy as np
import torch
import joblib

def load_models():
    xgb_model = joblib.load("xgboost_model.pkl")
    lstm_model = torch.load("lstm_model.pt", map_location=torch.device('cpu'))
    sentiment_model = joblib.load("sentiment_model.pkl")
    return xgb_model, lstm_model, sentiment_model

def load_data(csv_path):
    return pd.read_csv(csv_path, parse_dates=["Date"])

def predict_all(df, xgb_model, lstm_model, sentiment_model):
    features = df.drop(columns=["Date", "Label"], errors="ignore")

    # XGBoost prediction
    xgb_pred = xgb_model.predict(features)

    # LSTM prediction
    lstm_input = torch.tensor(features.values, dtype=torch.float32).unsqueeze(1)
    lstm_pred = lstm_model(lstm_input).detach().numpy().flatten().round()

    # Sentiment prediction
    sentiment_pred = sentiment_model.predict(features)

    # Voting
    votes = xgb_pred + lstm_pred + sentiment_pred
    final_pred = (votes >= 2).astype(int)

    result = df[["Date"]].copy()
    result["XGBoost"] = xgb_pred
    result["LSTM"] = lstm_pred
    result["Sentiment"] = sentiment_pred
    result["Final Prediction"] = final_pred

    return result

def backtest_metrics(pred_df):
    actuals = pred_df["Label"]
    preds = pred_df["Final Prediction"]

    accuracy = (actuals == preds).mean()

    returns = np.where(preds == 1, 0.01, 0)
    cum_return = np.prod(1 + returns) - 1
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)

    return accuracy, sharpe, cum_return