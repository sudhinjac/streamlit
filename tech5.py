import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
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

st.set_page_config(layout='wide', page_title='Advanced Stock Analyzer')

# 🎨 Optional: Custom Background Color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5dc;
    }
    </style>
    """,
    unsafe_allow_html=True
)
#st.set_page_config(layout='wide', page_title='Advanced Stock Analyzer')

st.title("📈 Enhanced Technical Analysis Dashboard")

# --- Sidebar Inputs ---
ticker = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

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

def compute_rsi(data, period=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_stochastic_oscillator(data, period=14):
    low_min = data['Low'].rolling(window=period).min()
    high_max = data['High'].rolling(window=period).max()
    stoch_k = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    return stoch_k

def compute_bollinger_bands(data, period=20):
    sma = data['Close'].rolling(window=period).mean()
    std_dev = data['Close'].rolling(window=period).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band

def compute_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = np.max([high_low, high_close, low_close], axis=0)
    atr = pd.Series(true_range).rolling(window=period).mean()
    return atr

def compute_macd(data, short_period=12, long_period=26, signal_period=9):
    short_ema = data['Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def make_decision(data):
    rsi = compute_rsi(data)
    stoch_k = compute_stochastic_oscillator(data)
    upper_band, lower_band = compute_bollinger_bands(data)
    atr = compute_atr(data)
    macd, signal = compute_macd(data)
    
    last_rsi = rsi.iloc[-1]
    last_stoch = stoch_k.iloc[-1]
    last_close = data['Close'].iloc[-1]
    last_macd = macd.iloc[-1]
    last_signal = signal.iloc[-1]
    
    buy_signal = (last_rsi < 30 and last_stoch < 20 and last_macd > last_signal)
    sell_signal = (last_rsi > 70 and last_stoch > 80 and last_macd < last_signal)
    
    if buy_signal:
        return "Buy now"
    elif sell_signal:
        return "Wait, price may go down"
    else:
        return "Hold"
def get_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    cashflow = stock.cashflow
    income_statement = stock.financials
    
    return balance_sheet, cashflow, income_statement
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

if ticker:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    df =stock_data
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = df.columns.droplevel(1)
    df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    


    if df.empty:
        st.error("No data found. Please enter a valid stock ticker.")
    else:
        df.dropna(inplace=True)
        beta, sharpe, treynor, jensen_alpha, cv, loss_prob, profit_prob, ow,vols = get_stock_metrics(ticker)
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
        styled_metrics_df = metrics_df.style.set_properties(**{'font-weight': 'bold'}, subset=['Metric'])

    # Display the DataFrame in Streamlit
      #  st.write("### 📊 Stock Analysis Metrics")
        st.dataframe(styled_metrics_df)
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
        
        st.write("## 🏦 Fundamental Analysis")
    
        balance_sheet, cashflow, income_statement = get_fundamentals(ticker)

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
        next_day_data = data[['50_MA', '200_MA', 'RSI', 'MACD']].iloc[-1].values.reshape(1, -1)
        next_day_prediction = model.predict_proba(next_day_data)
        st.write(f"Probability of being Bullish: {next_day_prediction[0][1]:.2f}")
        st.write(f"Probability of being Bearish: {1 - next_day_prediction[0][1]:.2f}")
        decision = make_decision(df)
        st.write(f"Decision: {decision}")


        # Indicators
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        df['20sma'] = df['Close'].rolling(window=20).mean()
        df['stddev'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['20sma'] + (2 * df['stddev'])
        df['Lower'] = df['20sma'] - (2 * df['stddev'])

        df['50SMA'] = df['Close'].rolling(window=50).mean()
        df['100EWMA'] = df['Close'].ewm(span=100, adjust=False).mean()

        delta = df['High'] - df['Low']
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        tr1 = delta
        tr2 = abs(df['High'] - df['Close'].shift(1))
        tr3 = abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        df['ATR'] = atr
        df['+DI'] = 100 * (plus_dm.rolling(14).mean() / df['ATR'])
        df['-DI'] = 100 * (minus_dm.rolling(14).mean() / df['ATR'])
        df['ADX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])

        # RSI Calculation (fixed with Series)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Ichimoku
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        df['Tenkan'] = (high_9 + low_9) / 2

        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        df['Kijun'] = (high_26 + low_26) / 2

        df['SpanA'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()
        df['SpanB'] = ((high_52 + low_52) / 2).shift(26)
        df['Chikou'] = df['Close'].shift(-26)

        # Signals
        df['entry'] = 0
        buy_entry = (df['Close'] > df['Lower']) & (df['Open'] < df['Lower']) & (df['ADX'] > 20)
        sell_entry = (df['Close'] < df['Upper']) & (df['Open'] > df['Upper']) & (df['ADX'] > 20)
        df.loc[buy_entry, 'entry'] = 2
        df.loc[sell_entry, 'entry'] = 1

        df['RSI Signal'] = 0
        df.loc[(df['Close'] < df['Lower']) & (df['RSI'] < 55), 'RSI Signal'] = 2
        df.loc[(df['Close'] > df['Upper']) & (df['RSI'] > 45), 'RSI Signal'] = 1

        def detect_shooting_star(row):
            body = abs(row['Open'] - row['Close'])
            upper_shadow = row['High'] - max(row['Open'], row['Close'])
            lower_shadow = min(row['Open'], row['Close']) - row['Low']
            if body < row['Open'] * 0.01:
                return 0
            if lower_shadow > 1.5 * body and upper_shadow < 0.8 * body:
                return 2
            elif upper_shadow > 1.5 * body and lower_shadow < 0.8 * body:
                return 1
            return 0

        df['shooting_star'] = df.apply(detect_shooting_star, axis=1)

        latest = df.iloc[-1]
        ichimoku_signal = "Bullish" if latest['Close'] > max(latest['SpanA'], latest['SpanB']) else "Bearish" if latest['Close'] < min(latest['SpanA'], latest['SpanB']) else "Neutral"
        adx_signal = "Strong Trend" if latest['ADX'] > 25 else "Weak/No Trend"
        entry_text = "Buy" if latest['entry'] == 2 else "Sell" if latest['entry'] == 1 else "No Signal"
        rsi_boll_signal = "Buy Signal" if latest['RSI Signal'] == 2 else "Sell Signal" if latest['RSI Signal'] == 1 else "Neutral"
        shooting_star = "Bullish Reversal" if latest['shooting_star'] == 2 else "Bearish Reversal" if latest['shooting_star'] == 1 else "No Pattern"

        signal_data = {
            'Indicator': ['RSI', 'MACD', 'Bollinger', 'ADX', 'Ichimoku', 'Entry Signal', 'RSI + Bollinger', 'Shooting Star'],
            'Signal': [
                'Bullish' if latest['RSI'] < 30 else 'Bearish' if latest['RSI'] > 70 else 'Neutral',
                'Bullish' if latest['MACD'] > latest['Signal'] else 'Bearish',
                'Bullish' if latest['Close'] < latest['Lower'] else 'Bearish' if latest['Close'] > latest['Upper'] else 'Neutral',
                adx_signal,
                ichimoku_signal,
                entry_text,
                rsi_boll_signal,
                shooting_star
            ]
        }

        st.subheader("📊 Trade Signal Summary")
        signal_df = pd.DataFrame(signal_data)
        st.dataframe(signal_df)

        st.subheader("📉 Price Chart with Bollinger Bands, 50 SMA & 100 EWMA + Volume")
        mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        mf_multiplier.replace([np.inf, -np.inf], 0, inplace=True)
        mf_multiplier.fillna(0, inplace=True)

        # Money Flow Volume
        mf_volume = mf_multiplier * df['Volume']

        # Accumulation Distribution Line (ADL)
        df['ADL'] = mf_volume.cumsum()

# Chaikin Oscillator = 3-day EMA of ADL - 10-day EMA of ADL
        df['Chaikin'] = df['ADL'].ewm(span=3, adjust=False).mean() - df['ADL'].ewm(span=10, adjust=False).mean()
        
        mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        mf_multiplier.replace([np.inf, -np.inf], 0, inplace=True)
        mf_multiplier.fillna(0, inplace=True)
        mf_volume = mf_multiplier * df['Volume']
        df['CMF'] = mf_volume.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

        # --- Chaikin Oscillator ---
        adl = mf_volume.cumsum()
        ema3 = adl.ewm(span=3, adjust=False).mean()
        ema10 = adl.ewm(span=10, adjust=False).mean()
        df['Chaikin_Oscillator'] = ema3 - ema10
        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlesticks'
        ))

        # Bollinger Bands and Moving Averages
        fig.add_trace(go.Scatter(x=df.index, y=df['20sma'], name='20 SMA', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='Upper Band', line=dict(color='green', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='Lower Band', line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['50SMA'], name='50 SMA', line=dict(color='purple', width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df['100EWMA'], name='100 EWMA', line=dict(color='orange', width=1.5)))

        # Volume Bars (on secondary y-axis)
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='lightgrey',
            yaxis='y2',
            opacity=0.5
        ))

        # Layout with dual y-axes
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=700,
            yaxis=dict(title='Price'),
            yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 RSI")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='orange')))
        fig_rsi.add_hline(y=70, line=dict(dash='dot', color='red'))
        fig_rsi.add_hline(y=30, line=dict(dash='dot', color='green'))
        fig_rsi.update_layout(height=300)
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        st.subheader("📊 Chaikin Oscillator")
        fig_chaikin = go.Figure()
        fig_chaikin.add_trace(go.Scatter(x=df.index, y=df['Chaikin_Oscillator'], name='Chaikin Oscillator', line=dict(color='darkcyan')))
        fig_chaikin.add_hline(y=0, line=dict(dash='dot', color='gray'))
        fig_chaikin.update_layout(height=300)
        st.plotly_chart(fig_chaikin, use_container_width=True)
        
        st.subheader("📊 Chaikin Money Flow (CMF)")
        fig_cmf = go.Figure()
        fig_cmf.add_trace(go.Scatter(x=df.index, y=df['CMF'], name='CMF', line=dict(color='darkgreen')))
        fig_cmf.add_hline(y=0, line=dict(dash='dot', color='gray'))
        fig_cmf.update_layout(height=300)
        st.plotly_chart(fig_cmf, use_container_width=True)
                
        st.subheader("📈 Average True Range (ATR)")

        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(x=df.index, y=df['ATR'], name='ATR', line=dict(color='brown')))
        fig_atr.update_layout(height=300, yaxis_title="ATR")
        st.plotly_chart(fig_atr, use_container_width=True)

        st.subheader("📊 MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='purple')))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal Line', line=dict(color='gray')))
        fig_macd.update_layout(height=300)
        st.plotly_chart(fig_macd, use_container_width=True)

        st.subheader("📊 Ichimoku Cloud")
        fig_ichimoku = go.Figure()
        fig_ichimoku.add_trace(go.Scatter(x=df.index, y=df['Tenkan'], name='Tenkan', line=dict(color='blue')))
        fig_ichimoku.add_trace(go.Scatter(x=df.index, y=df['Kijun'], name='Kijun', line=dict(color='red')))
        fig_ichimoku.add_trace(go.Scatter(x=df.index, y=df['SpanA'], name='Span A', line=dict(color='green')))
        fig_ichimoku.add_trace(go.Scatter(x=df.index, y=df['SpanB'], name='Span B', line=dict(color='orange')))
        fig_ichimoku.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='black')))
        fig_ichimoku.update_layout(height=400)
        st.plotly_chart(fig_ichimoku, use_container_width=True)
                
        st.subheader("🔮 Price Forecast: ARIMA vs Holt-Winters")

        # --- Prepare Data ---
        forecast_df = df[['Close']].copy()
        forecast_df.dropna(inplace=True)

        # Forecast next 30 business days
        forecast_period = 30
        future_dates = pd.date_range(start=forecast_df.index[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='B')

        # --- ARIMA Forecast ---
        try:
            arima_model = ARIMA(forecast_df['Close'], order=(5, 1, 0))
            arima_result = arima_model.fit()
            arima_pred = arima_result.forecast(steps=forecast_period)
            arima_pred.index = future_dates
            arima_df = pd.DataFrame({'ARIMA Forecast': arima_pred})
        except Exception as e:
            st.error(f"ARIMA forecast failed: {e}")
            arima_df = pd.DataFrame()

        # --- Holt-Winters Forecast ---
        try:
            hw_model = ExponentialSmoothing(forecast_df['Close'], trend='add', seasonal=None)
            hw_fit = hw_model.fit()
            hw_pred = hw_fit.forecast(steps=forecast_period)
            hw_pred.index = future_dates
            hw_df = pd.DataFrame({'HW Forecast': hw_pred})
        except Exception as e:
            st.error(f"Holt-Winters forecast failed: {e}")
            hw_df = pd.DataFrame()

        # --- Plotting ---
        fig = go.Figure()

        # Historical Close
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Close'],
            mode='lines',
            name='Historical Close',
            line=dict(color='blue')
        ))

        # ARIMA Forecast
        if not arima_df.empty:
            fig.add_trace(go.Scatter(
                x=arima_df.index,
                y=arima_df['ARIMA Forecast'],
                mode='lines',
                name='ARIMA Forecast',
                line=dict(color='green', dash='dot')
            ))

        # Holt-Winters Forecast
        if not hw_df.empty:
            fig.add_trace(go.Scatter(
                x=hw_df.index,
                y=hw_df['HW Forecast'],
                mode='lines',
                name='Holt-Winters Forecast',
                line=dict(color='orange', dash='dot')
            ))

        fig.update_layout(
            title='📉 Forecasted vs Actual Price (Next 30 Days)',
            xaxis_title='Date',
            yaxis_title='Price',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)
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
