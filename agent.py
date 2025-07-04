import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from GoogleNews import GoogleNews
import ollama

# Function to fetch current price and 1-year high from Yahoo Finance
def fetch_price_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        current_price = hist['Close'][-1]
        high_1y = hist['High'].max()
        return current_price, high_1y
    except:
        return None, None

# Function to fetch recent news from Google News
def fetch_news(ticker):
    googlenews = GoogleNews(period='7d')
    googlenews.search(ticker)
    news_items = googlenews.result(sort=True)
    return news_items[:5]  # Return top 5 news articles

# Function to generate analysis using Ollama
def generate_ollama_analysis(ticker, company_name, news, financials):
    context = f"""
    Analyze whether {company_name} ({ticker}) is a good long-term stock to invest in based on the following financial information and news articles.

    Financials:
    {financials}

    News:
    {news}

    Provide a verdict if this stock is fundamentally strong and has a good future based on its products and market position.
    """
    response = ollama.chat(model='deepseek-r1:32b', messages=[{"role": "user", "content": context}])
    return response['message']['content']

st.set_page_config(page_title="Agentic AI Stock Picker", layout="wide")
st.title("📈 Agentic AI Stock Selector")

uploaded_file = st.file_uploader("Upload CSV file with stock metrics", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Original Data:", df.head())

    results = []

    for _, row in df.iterrows():
        ticker = row['Ticker']
        rsi = str(row['RSI Trend']).lower()
        macd = str(row['MACD Trend']).lower()

        if 'bullish' in rsi and 'bullish' in macd:
            current_price, high_1y = fetch_price_info(ticker)
            if current_price is None or high_1y is None:
                continue

            drop_percent = ((high_1y - current_price) / high_1y) * 100
            if drop_percent >= 10:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    long_name = info.get('longName', 'Unknown')
                    financial_summary = {
                        'sector': info.get('sector'),
                        'industry': info.get('industry'),
                        'marketCap': info.get('marketCap'),
                        'profitMargins': info.get('profitMargins'),
                        'pegRatio': info.get('pegRatio'),
                        'returnOnEquity': info.get('returnOnEquity'),
                        'grossMargins': info.get('grossMargins'),
                        'ebitdaMargins': info.get('ebitdaMargins'),
                    }

                    news_articles = fetch_news(ticker)
                    news_text = "\n".join([f"{item['title']} - {item['desc']}" for item in news_articles])

                    analysis = generate_ollama_analysis(ticker, long_name, news_text, financial_summary)

                    results.append({
                        'Ticker': ticker,
                        'Drop % from 1Y High': round(drop_percent, 2),
                        'Current Price': current_price,
                        '1Y High': high_1y,
                        'RSI': rsi,
                        'MACD': macd,
                        'Analysis': analysis
                    })
                except Exception as e:
                    st.warning(f"Error processing {ticker}: {e}")

    if results:
        result_df = pd.DataFrame(results)
        st.subheader("📊 Filtered Stocks with Analysis")
        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Result CSV", csv, "filtered_stocks.csv", "text/csv")
    else:
        st.warning("No stock matched the criteria.")
else:
    st.info("Please upload a CSV file to begin.")
