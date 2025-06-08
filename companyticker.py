import pandas as pd
import requests

def yahoo_finance_search(company_name):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if "quotes" in data and len(data["quotes"]) > 0:
            top = data["quotes"][0]
            return top.get("symbol")
    except Exception as e:
        print(f"Error searching {company_name}: {e}")
    return None

def get_tickers_from_companies(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    tickers = []

    for company in df['Company']:
        ticker = yahoo_finance_search(company)
        print(f"{company} → {ticker}")
        tickers.append(ticker if ticker else "None")

    df['Ticker'] = tickers
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved tickers to {output_csv}")

if __name__ == "__main__":
    get_tickers_from_companies("Company2.csv", "tickers2.csv")