import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start='2015-01-01', end='2024-01-01'):
    data = yf.download(ticker, start=start, end=end)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

data = fetch_stock_data('AAPL')
data.to_csv('data\\AAPL.csv')