# Stock Market Regime Detection using Transformers

## 📌 Project Overview
This project implements a **market regime detection system** using a **Transformer-based model**. Unlike traditional models (HMMs, LSTMs), Transformers capture **long-term dependencies** in stock price movements and classify the market into three regimes:

- **Bullish (Uptrend)** 📈
- **Bearish (Downtrend)** 📉
- **Sideways (Range-bound)** ➖

The model is trained on historical stock prices and technical indicators, learning to predict future regimes.

---

## 🏗 Project Structure
```
📂 Stock_Market_Regime_Detection
│── data/                # Folder for stock price data
│── models/              # Saved trained models
│── src/
│   │── data_loader.py   # Fetch and preprocess data
│   │── transformer.py   # Transformer model architecture
│   │── train.py         # Model training script
│   │── predict.py       # Regime prediction script
│── requirements.txt     # Python dependencies
│── README.md            # Project documentation
```

---

## ⚙️ Installation & Setup
```bash
# Clone the repository
git clone https://github.com/Livnatc/Stock_Market_Regime_Detection.git
cd Stock_Market_Regime_Detection

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Data Collection
We use **Yahoo Finance** to fetch historical stock data (OHLC + volume) and compute **technical indicators**:

- **Moving Averages** (SMA, EMA)
- **Relative Strength Index (RSI)**
- **MACD (Momentum Indicator)**
- **Rolling Volatility**

```python
import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start='2015-01-01', end='2024-01-01'):
    data = yf.download(ticker, start=start, end=end)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

data = fetch_stock_data('AAPL')
data.to_csv('data/AAPL.csv')
```

---

## 🏗 Feature Engineering
We extract **log returns**, rolling volatility, and other features. We also label market regimes based on **volatility clustering** or **K-means clustering**.

```python
import numpy as np

def compute_technical_features(data):
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['volatility'] = data['log_return'].rolling(20).std()
    return data.dropna()

data = compute_technical_features(data)
```

---

## 🧠 Transformer Model
We implement a **self-attention-based Transformer encoder** to classify market regimes. The model learns to detect shifts between **bullish, bearish, and sideways** markets.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, LayerNormalization, Dropout
from tensorflow.keras.models import Model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs  # Residual connection

    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    return x + res
```

We stack multiple Transformer blocks and train the model:
```python
input_layer = Input(shape=(30, 10))  # 30-day window, 10 features
x = transformer_encoder(input_layer, head_size=64, num_heads=4, ff_dim=128)
x = Dense(64, activation="relu")(x)
x = Dropout(0.1)(x)
output_layer = Dense(3, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
```

---

## 📈 Predictions & Backtesting
Once trained, we predict market regimes and visualize them alongside stock prices:
```python
import matplotlib.pyplot as plt

def plot_regimes(price_data, regimes):
    plt.figure(figsize=(12,6))
    plt.plot(price_data, label='Stock Price', alpha=0.7)
    plt.scatter(range(len(regimes)), price_data, c=regimes, cmap='coolwarm', marker='o')
    plt.legend()
    plt.show()
```

We also backtest a simple **long-short strategy** based on detected regimes:
- **Long positions in Bull markets** 🟢
- **Short positions in Bear markets** 🔴
- **No trades in Sideways markets** ⚪

Example for prediction results above certain stock:
![predicted_results](https://github.com/user-attachments/assets/32428d33-c9cb-4f87-892a-88b182b8a2f3)


---

## 📌 Next Steps
✅ Improve feature engineering (news sentiment, macro indicators)  
✅ Test different Transformer architectures (Time-Series Transformers)  
✅ Deploy as a real-time trading signal generator  


---

## 📩 Contact
For questions or contributions, reach out via **GitHub Issues** or [LinkedIn](https://www.linkedin.com/in/livnat-cohen/). 🚀

