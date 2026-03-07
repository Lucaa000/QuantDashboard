# Quant Trading Dashboard

This is an interactive dashboard for analyzing and backtesting trading strategies using historical market data. It is built in **Python** with **Streamlit** and leverages libraries for technical analysis and machine learning.

## Key Features

- Automatic download of historical data from Yahoo Finance using `yfinance`.
- Calculation of technical indicators such as **RSI** and **EMA**.
- Market predictions using a **Random Forest model**.
- Trading strategy based on ML predictions.
- **Equity curve** comparison between ML strategy and buy & hold.
- Interactive charts and key metrics displayed in real-time.
- Debugging with the latest predictions and equity values.

## Requirements

- Python 3.10+ (a virtual environment is recommended)
- Python libraries:
  ```bash
  pip install streamlit yfinance pandas numpy ta scikit-learn