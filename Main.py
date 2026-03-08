import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from sklearn.ensemble import RandomForestClassifier

# --- Page configuration ---
st.set_page_config(layout="wide")
st.title("📊 Quant Trading Dashboard")

# --- Sidebar: asset selection ---
asset = st.sidebar.selectbox(
    "Select Asset",
    ["AAPL", "MSFT", "TSLA", "EURUSD=X", "GBPUSD=X"]
)

# --- Download market data ---
data = yf.download(asset, start="2020-01-01")

# Fix potential multi-index columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# --- Technical indicators ---
data["rsi"] = RSIIndicator(data["Close"]).rsi()
data["ema"] = EMAIndicator(data["Close"], window=20).ema_indicator()

# --- ML target: predict next-day direction ---
data["target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)

data = data.dropna()

X = data[["rsi", "ema"]]
y = data["target"]

split = int(len(data) * 0.8)

# --- Train Random Forest model ---
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X.iloc[:split], y.iloc[:split])

# --- Model predictions ---
data["prediction"] = model.predict(X)

# --- Compute returns ---
data["returns"] = data["Close"].pct_change().fillna(0)

# --- Log returns ---
data["log_returns"] = np.log(1 + data["returns"])

# --- Buy & Hold strategy ---
data["bh_log_equity"] = data["log_returns"].cumsum()

# --- ML trading strategy ---
data["strategy_returns"] = data["prediction"].shift(1) * data["returns"]
data["strategy_returns"] = data["strategy_returns"].fillna(0)

data["strategy_log_returns"] = np.log(1 + data["strategy_returns"])
data["ml_log_equity"] = data["strategy_log_returns"].cumsum()

# --- Key metrics ---
col1, col2, col3 = st.columns(3)

col1.metric("Price", f"{data['Close'].iloc[-1]:.2f}")
col2.metric("RSI", f"{data['rsi'].iloc[-1]:.2f}")
col3.metric("EMA (20)", f"{data['ema'].iloc[-1]:.2f}")

# --- Price chart ---
st.subheader("Price and EMA")
st.line_chart(data[["Close", "ema"]])

# --- Strategy comparison ---
st.subheader("ML Strategy vs Buy & Hold (Log Returns)")
st.line_chart(data[["ml_log_equity", "bh_log_equity"]])

# --- Latest predictions ---
st.subheader("Latest Model Predictions")
st.dataframe(
    data.tail(10)[
        ["Close", "rsi", "ema", "prediction", "strategy_returns"]
    ]
)