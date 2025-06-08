import streamlit as st
import pandas as pd
from utils import load_data, load_models, predict_all, backtest_metrics

st.title("📈 Stock Movement Predictor (XGBoost + LSTM + Sentiment)")

uploaded_file = st.file_uploader("Upload stock data CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file, parse_dates=["Date"])
    st.write("Preview of uploaded data:", data.head())

    xgb_model, lstm_model, sentiment_model = load_models()

    predictions_df = predict_all(data, xgb_model, lstm_model, sentiment_model)
    st.write("🔮 Combined Predictions:", predictions_df)

    # Export predictions
    csv = predictions_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions CSV", data=csv, file_name="predictions.csv")

    # Backtest
    if st.button("Run Backtest"):
        acc, sharpe, cum_return = backtest_metrics(predictions_df)
        st.success(f"✅ Accuracy: {acc:.2%}")
        st.info(f"📊 Sharpe Ratio: {sharpe:.2f}")
        st.warning(f"💰 Cumulative Return: {cum_return:.2%}")