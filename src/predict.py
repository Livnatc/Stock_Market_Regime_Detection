import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import data_handler

def user_predict(user_model, stock):

    # Load new stock data (replace with actual file or API)
    df_features, labels = data_handler.process(stock)

    # take only 3 last weeks:
    df_features = df_features.iloc[-120:, :]
    # df = df.iloc[-120:, :]

    # Normalize using the same scaler from training
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(df_features)

    # Convert into sequence format (e.g., last 30 days of data)
    sequence_length = 30
    X_new = []
    for i in range(len(normalized_features) - sequence_length):
        X_new.append(normalized_features[i: i + sequence_length])

    X_new = np.array(X_new)  # Shape: (num_samples, 30, num_features)

    # Predict (output shape: (num_samples, 3) for 3 classes)
    predictions = user_model.predict(X_new)

    # Convert softmax probabilities to class labels (Bullish, Bearish, Sideways)
    class_labels = ["Bullish", "Bearish", "Sideways"]
    predicted_classes = np.argmax(predictions, axis=1)

    # Map to class names
    predicted_trends = [class_labels[i] for i in predicted_classes]

    df_results = pd.DataFrame({
        "Predicted Trend": predicted_trends
    })

    # Save predictions to CSV
    # df_results.to_csv("predictions.csv", index=False)

    # Print some predictions
    # print(df_results.head())

    return df_results.iloc[-1][0], df_features


def predict():

    # Load the trained model
    model = tf.keras.models.load_model("../models/model.h5")

    # Load new stock data (replace with actual file or API)
    df_features = pd.read_csv("../data/AAPL_features.csv")
    df_features = df_features.iloc[:, -9:-1]
    df = pd.read_csv("../data/AAPL.csv")

    # take only 3 last weeks:
    df_features = df_features.iloc[-120:, :]
    df = df.iloc[-120:, :]

    # Normalize using the same scaler from training
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(df_features)

    # Convert into sequence format (e.g., last 30 days of data)
    sequence_length = 30
    X_new = []
    for i in range(len(normalized_features) - sequence_length):
        X_new.append(normalized_features[i : i + sequence_length])

    X_new = np.array(X_new)  # Shape: (num_samples, 30, num_features)

    # Predict (output shape: (num_samples, 3) for 3 classes)
    predictions = model.predict(X_new)

    # Convert softmax probabilities to class labels (Bullish, Bearish, Sideways)
    class_labels = ["Bullish", "Bearish", "Sideways"]
    predicted_classes = np.argmax(predictions, axis=1)

    # Map to class names
    predicted_trends = [class_labels[i] for i in predicted_classes]

    df_results = pd.DataFrame({
        "Date": df["Price"].iloc[-len(predicted_trends):],  # Align with original data
        "Predicted Trend": predicted_trends
    })

    # Save predictions to CSV
    df_results.to_csv("predictions.csv", index=False)

    # Print some predictions
    print(df_results.head())

    prices = np.array(df["Close"].iloc[-len(predicted_trends):]).astype(float)
    color_map = {"Bullish": "green", "Bearish": "red", "Sideways": "blue"}
    colors = [color_map[trend] for trend in predicted_trends]
    names = ['Bullish', 'Bearish', 'Sideways']


    plt.figure(figsize=(12, 6))
    plt.plot(prices)
    plt.scatter(x=range(len(prices)), y=prices,
                c=colors)

    import matplotlib.patches as mpatches
    # Creating legend with color box
    pop_a = mpatches.Patch(color='red', label='Bearish', linewidth=20)
    pop_b = mpatches.Patch(color='green', label='Bullish')
    pop_c = mpatches.Patch(color='blue', label='Sideways')
    plt.legend(handles=[pop_a, pop_b, pop_c])

    plt.xticks(rotation=45)
    plt.title("Stock Price with Predicted Trends")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig("../figures/predicted_results.png")
    plt.show()