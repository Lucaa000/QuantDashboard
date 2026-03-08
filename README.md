# 📊 Quant Trading Dashboard

An interactive **quantitative trading dashboard** built with **Python and Streamlit** to explore the integration of financial data, technical indicators, and machine learning in a simple trading strategy.

The application downloads historical market data, computes technical indicators, trains a machine learning model to predict price direction, and compares the resulting strategy with a **Buy & Hold benchmark**.

# 🧠 Project Overview

This project demonstrates a simple **quantitative research workflow**, including:

1. Market data acquisition  
2. Feature engineering with technical indicators  
3. Machine learning model training  
4. Trading signal generation  
5. Strategy performance evaluation  

The goal is to provide an **interactive environment to experiment with ML-based trading strategies**.

---

# ⚙️ Features

- Interactive **Streamlit dashboard**
- Historical market data from **Yahoo Finance**
- Technical indicators:
  - **RSI (Relative Strength Index)**
  - **EMA (Exponential Moving Average)**
- **Random Forest classifier** predicting next-day price direction
- Strategy comparison:
  - ML-based strategy
  - Buy & Hold benchmark
- Performance evaluation using **cumulative log returns**
- Real-time display of:
  - price
  - indicators
  - model predictions

---

# 📈 Strategy Logic

The machine learning model predicts whether the **next-day price will increase or decrease**.

The strategy then follows these rules:

- **Prediction = 1 → Long position**
- **Prediction = 0 → No position**

Strategy returns are computed as:

strategy_return = prediction(t-1) × asset_return(t)


Performance is evaluated using **cumulative log returns**:

log_return = log(1 + return)


Log returns are commonly used in quantitative finance because they are **additive over time** and allow consistent strategy comparisons.

---

# 🛠 Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **yfinance**
- **ta (Technical Analysis library)**