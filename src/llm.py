# from llama_cpp import Llama
#
# # Load the LLaMA model (make sure you have the model file downloaded)
# llm = Llama(model_path="../llama-2-7b.Q3_K_S.gguf", n_ctx=4096)
#
#
# def explain_stock_trend(stock_symbol, predicted_trend, technical_data):
#     """
#     Uses LLaMA to generate an explanation for a given stock's predicted trend.
#
#     Args:
#         stock_symbol (str): The stock ticker (e.g., "AAPL").
#         predicted_trend (str): Model's prediction ("Bullish", "Bearish", "Sideways").
#         technical_data (dict): Relevant indicators (e.g., RSI, MACD, Volatility).
#
#     Returns:
#         str: LLaMA-generated explanation.
#     """
#
#     # Construct a prompt using the trend prediction and technical indicators
#     prompt = f"""
#     You are a financial analyst. Given the following technical indicators and model predictions, explain why {stock_symbol}
#     is currently in a {predicted_trend} trend.
#
#     **Stock Symbol**: {stock_symbol}
#     **Predicted Trend**: {predicted_trend}
#     **Technical Indicators**:
#     - RSI: {technical_data['RSI']}
#     - MACD: {technical_data['MACD']}
#     - Volatility: {technical_data['Volatility']}
#     - Moving Averages: {technical_data['Moving_Avg']}
#
#     Provide a review about the {stock_symbol}.
#     """
#
#     # Generate response
#     # response = llm(prompt, max_tokens=300, stop=["\n\n"])
#     response = llm(f"Q: based on the RSI = {technical_data['RSI']}, MACD = {technical_data['MACD']} "
#                    f"and {technical_data['Volatility']} - why {stock_symbol} is in {predicted_trend}? A: ", # Prompt
#       max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
#       stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
#       echo=True # Echo the prompt back in the output
# ) # Generate a completion,
#
#     response = llm(f"Q: what do you know about the company that has the stock name APPL? A: ", # Prompt
#       max_tokens=100, # Generate up to 32 tokens, set to None to generate up to the end of the context window
#       stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
#       echo=True # Echo the prompt back in the output
# ) # Generate a completion,
#
#     return response["choices"][0]["text"].strip()
#     # return response
#
#
# # Example Usage
# technical_example = {
#     "RSI": 68.2,
#     "MACD": 1.5,
#     "Volatility": 2.1,
#     "Moving_Avg": "Above 50-day SMA"
# }
#
# # Assume our model predicted 'Bullish' for AAPL
# explanation = explain_stock_trend("AAPL", "Bullish", technical_example)
# print("\n🔍 LLaMA's Explanation:\n", explanation)
# print('Done')


import transformers
import torch


def explain_outcome(stock, technical_data, prediction):

    model_id = "WiroAI/WiroAI-Finance-Llama-8B"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    pipeline.model.eval()

    messages = [
        {"role": "system", "content": "You are a finance chatbot developed by Wiro AI"},
        {"role": "user", "content":
            f"If {stock} stock has {technical_data['RSI'][-1]} and {technical_data['MACD'][-1]} , do you think  it is a {prediction} position? "
      },
    ]


    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.9,
    )

    return outputs[0]["generated_text"][-1]['content']
