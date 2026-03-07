import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from sklearn.ensemble import RandomForestClassifier

# --- Configurazione pagina ---
st.set_page_config(layout="wide")
st.title("📊 Quant Trading Dashboard")

# --- Sidebar asset ---
asset = st.sidebar.selectbox("Select Asset", ["AAPL", "MSFT", "TSLA", "EURUSD=X", "GBPUSD=X"])

# --- 1. Download dati ---

data = yf.download(asset, start="2020-01-01")
st.write("Prime righe dei dati:", data.head())

# --- 2. FIX MULTI-INDEX ---
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# --- 3. Serie dei prezzi di chiusura ---
close = data["Close"].squeeze()

# --- 4. Indicatori tecnici ---
data["rsi"] = RSIIndicator(close).rsi()
data["ema"] = EMAIndicator(close, window=20).ema_indicator()

# --- 5. Target ML ---
data["target"] = np.where(close.shift(-1) > close, 1, 0)
data = data.dropna()

X = data[["rsi", "ema"]]
y = data["target"]
split = int(len(data) * 0.8)

# --- 6. Modello Random Forest ---
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X.iloc[:split], y.iloc[:split])

# --- 7. Predizioni ---
data["prediction"] = model.predict(X)

# --- 8. Strategia ML e equity curve ---
data["strategy_returns"] = data["prediction"].shift(1) * close.pct_change()
data["strategy_returns"].fillna(0, inplace=True)
data["equity"] = (1 + data["strategy_returns"]).cumprod()

# --- 9. Benchmark buy & hold ---
data["buy_hold_returns"] = close.pct_change().fillna(0)
data["buy_hold_equity"] = (1 + data["buy_hold_returns"]).cumprod()

# --- 10. Metriche principali ---
col1, col2, col3 = st.columns(3)
col1.metric("Prezzo", f"{close.iloc[-1]:.2f}")
col2.metric("RSI", f"{data['rsi'].iloc[-1]:.2f}")
col3.metric("Equity ML", f"{data['equity'].iloc[-1]:.2f}")

# --- 11. Grafici ---
st.subheader("Prezzo e EMA")
st.line_chart(data[["Close", "ema"]])

st.subheader("Curva dei profitti (ML Strategy vs Buy & Hold)")
st.line_chart(data[["equity", "buy_hold_equity"]])
st.write("Ultimi valori equity ML e Buy&Hold:", data[["equity", "buy_hold_equity"]].tail())

st.subheader("Ultime predizioni")
st.dataframe(data.tail(10)[["Close", "rsi", "ema", "prediction", "strategy_returns", "equity"]])