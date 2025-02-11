import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



def normalize_data(df):
    """
    Normalize stock price data using MinMaxScaler.

    Parameters:
        df (pd.DataFrame): Stock data with technical indicators.

    Returns:
        pd.DataFrame, MinMaxScaler: Normalized data and the scaler for inverse transformation.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns), scaler


def create_sequences(data, labels, sequence_length=30):
    """
    Convert normalized data into sequences for Transformer input.

    Parameters:
        data (pd.DataFrame): Normalized stock data.
        labels (pd.Series): Market regime labels.
        sequence_length (int): Length of each sequence.

    Returns:
        np.array: X (features) of shape (num_samples, sequence_length, num_features)
        np.array: y (labels) of shape (num_samples,)
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i + sequence_length].values)  # Extract sequence of features
        y.append(labels.iloc[i + sequence_length])  # Predict next time step label
    return np.array(X), np.array(y)

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


def label_volatility_clusters(data, window=20):
    """
    Labels market regimes based on volatility clustering.
    High volatility -> Bearish, Low volatility -> Bullish, Medium -> Sideways.
    """
    data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
    data["Volatility"] = data["Log_Return"].rolling(window).std()

    threshold_high = data["Volatility"].quantile(0.75)
    threshold_low = data["Volatility"].quantile(0.25)

    data["Regime_Volatility"] = np.where(data["Volatility"] > threshold_high, 1,  # Bearish
                                         np.where(data["Volatility"] < threshold_low, 0,  # Bullish
                                                  2))  # Sideways
    return data


def label_kmeans_clusters(data, n_clusters=3):
    """
    Labels market regimes using K-means clustering on price returns.
    """
    log_returns = np.log(data["Close"] / data["Close"].shift(1)).fillna(0).values.reshape(-1, 1)

    scaler = StandardScaler()
    log_returns_scaled = scaler.fit_transform(log_returns)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data["Regime"] = kmeans.fit_predict(log_returns_scaled)

    return data


def label_threshold_based(data):
    """
    Labels regimes based on moving average trends.
    Bullish: Short EMA above long EMA, Bearish: Short EMA below long EMA.
    """
    data["EMA_Short"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_Long"] = data["Close"].ewm(span=26, adjust=False).mean()

    data["Regime_Threshold"] = np.where(data["EMA_Short"] > data["EMA_Long"], 0,  # Bullish
                                        np.where(data["EMA_Short"] < data["EMA_Long"], 1,  # Bearish
                                                 2))  # Sideways
    return data


def process(st):

    data = fetch_stock_data(st)
    features = compute_technical_indicators(data)

    data.to_csv(f'../data/{st}.csv')
    features.to_csv(f'../data/{st}_features.csv')

    # Apply regime labeling methods
    # stock_data = label_volatility_clusters(features)
    stock_data = label_kmeans_clusters(features)
    # stock_data = label_threshold_based(features)

    return stock_data.iloc[:, -9:-1], stock_data.iloc[:, -1]
