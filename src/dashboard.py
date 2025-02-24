import streamlit as st
import tensorflow as tf
import predict
import llm as llm_model


# Dummy function to simulate a stock trend classifier (replace with your actual model)
# def get_stock_trend(stock_symbol):
#     """Mock function to return a stock trend prediction."""
#     trends = {"AAPL": "Bullish", "TSLA": "Bearish", "GOOGL": "Sideways"}
#     return trends.get(stock_symbol.upper(), "Unknown")


# Function to generate an explanation using LLaMA
# def explain_stock_trend(stock_symbol, predicted_trend):
#     """Uses LLaMA to generate an explanation for a given stock's trend prediction."""
#     if predicted_trend == "Unknown":
#         return "Sorry, I don't have trend data for this stock."
#
#     prompt = f"""
#     You are a financial analyst. Given the stock symbol {stock_symbol} and its predicted trend ({predicted_trend}),
#     provide a concise explanation based on technical indicators, market sentiment, and historical data.
#
#     **Stock Symbol**: {stock_symbol}
#     **Predicted Trend**: {predicted_trend}
#
#     Explain why this trend might be occurring and what traders should consider.
#     """
#
#     response = llm(prompt, max_tokens=300, stop=["\n\n"])
#     return response["choices"][0]["text"].strip()

# Model
model = tf.keras.models.load_model("../models/model.h5")

# Streamlit UI
st.title("📈 AI-Powered Stock Trend Chatbot")

# User input field
stock_symbol = st.text_input("Enter a stock symbol (e.g., AAPL, TSLA, GOOGL):", "")

if st.button("Analyze Stock"):
    if stock_symbol:
        trend, technicals = predict.user_predict(model, stock_symbol)
        explanation = llm_model.explain_outcome(stock_symbol, technicals, trend)
        st.subheader(f"📊 Predicted Trend: {trend}")
        st.write("📝 LLaMA's Explanation:")
        st.write(explanation)
    else:
        st.warning("Please enter a valid stock symbol.")
