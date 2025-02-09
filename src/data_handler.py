import yfinance as yf
import pandas as pd
import numpy as np


def fetch_stock_data(ticker, start='2015-01-01', end='2024-01-01'):
    data = yf.download(ticker, start=start, end=end)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]


def compute_technical_indicators(df, short_window=12, long_window=26, signal_window=9, rsi_period=14,
                                 volatility_window=20):
    """
    Compute SMA, EMA, RSI, MACD, and Volatility for stock price data.

    Parameters:
        df (pd.DataFrame): Stock price data with 'Close' column.
        short_window (int): Short window for MACD calculation.
        long_window (int): Long window for MACD calculation.
        signal_window (int): Signal line window for MACD.
        rsi_period (int): Lookback period for RSI calculation.
        volatility_window (int): Rolling window for volatility calculation.

    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """

    df = df.copy()

    # Simple Moving Average (SMA)
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # Exponential Moving Average (EMA)
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD (Moving Average Convergence Divergence)
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df["Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Rolling Volatility (Standard Deviation of Log Returns)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = df["Log_Return"].rolling(volatility_window).std()

    return df.dropna()  # Remove NaN values caused by rolling calculations


data = fetch_stock_data('AAPL')
features = compute_technical_indicators(data)

data.to_csv('data\\AAPL.csv')
features.to_csv('data\\AAPL_features.csv')