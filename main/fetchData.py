import requests
import yfinance as yf

def fetch_alpha_vantage_data(symbol, api_key):
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching data from Alpha Vantage: {response.status_code}")

def fetch_yfinance_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d")
    return data.to_dict()

def fetch_data(symbol, api_key):
    data_sources = {
        "Alpha Vantage": fetch_alpha_vantage_data,
        "yfinance": fetch_yfinance_data
    }
    
    results = {}
    for source, fetch_function in data_sources.items():
        try:
            if source == "Alpha Vantage":
                results[source] = fetch_function(symbol, api_key)
            else:
                results[source] = fetch_function(symbol)
        except Exception as e:
            results[source] = str(e)
    
    return results

def main():
    symbol = "AAPL"
    api_key = "your_alpha_vantage_api_key"
    data = fetch_data(symbol, api_key)
    print(data)

if __name__ == "__main__":
    main()