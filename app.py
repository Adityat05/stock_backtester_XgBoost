import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from xgboost import XGBClassifier

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Stock Backtesting Engine",
    page_icon="📈",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────────
st.title("📈 ML-Based Stock Market Backtesting Engine")
st.markdown("*Predicting NSE stock direction using XGBoost and backtesting signals against buy-and-hold*")
st.divider()

# ── Load data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df       = pd.read_csv("data/RELIANCE_features_v2.csv", 
                            index_col=0, parse_dates=True)
    X_test   = pd.read_csv("data/X_test.csv", 
                            index_col=0, parse_dates=True)
    y_test   = pd.read_csv("data/y_test.csv", 
                            index_col=0, parse_dates=True).squeeze()
    results  = pd.read_csv("data/results.csv", 
                            index_col=0, parse_dates=True)
    return df, X_test, y_test, results

@st.cache_resource
def load_model():
    with open("data/model.pkl", "rb") as f:
        return pickle.load(f)

df, X_test, y_test, results = load_data()
model= load_model()


# ── Run backtest ──────────────────────────────────────────
def backtest_strategy(results, initial_capital=100000,
                      transaction_cost=0.001, threshold=0.5):
    cash      = initial_capital
    position  = 0
    portfolio = []
    cash_hist = []
    stock_hist= []

    for i in range(len(results)):
        price       = results["Close"].iloc[i]
        probability = results["Probability"].iloc[i]

        if probability > threshold and position == 0:
            shares   = cash / (price * (1 + transaction_cost))
            position = shares
            cash     = 0

        elif probability <= threshold and position > 0:
            cash     = position * price * (1 - transaction_cost)
            position = 0

        stock_val = position * price
        cash_hist.append(cash)
        stock_hist.append(stock_val)
        portfolio.append(cash + stock_val)

    results = results.copy()
    results["Cash"]           = cash_hist
    results["Stock_Value"]    = stock_hist
    results["Strategy_Value"] = portfolio
    return results

def buy_and_hold(results, initial_capital=100000, 
                 transaction_cost=0.001):
    first_price = results["Close"].iloc[0]
    shares      = initial_capital / (first_price * (1 + transaction_cost))
    results     = results.copy()
    results["BuyHold_Value"] = shares * results["Close"]
    final_val   = shares * results["Close"].iloc[-1] * (1 - transaction_cost)
    results.loc[results.index[-1], "BuyHold_Value"] = final_val
    return results

def calculate_sharpe(portfolio_values):
    returns = pd.Series(portfolio_values).pct_change().dropna()
    return (returns.mean() / returns.std()) * np.sqrt(252)

def calculate_maxdd(portfolio_values):
    values      = pd.Series(portfolio_values)
    rolling_max = values.cummax()
    drawdown    = (values - rolling_max) / rolling_max
    return drawdown.min()

INITIAL_CAPITAL = 100000
results = backtest_strategy(results, INITIAL_CAPITAL)
results = buy_and_hold(results, INITIAL_CAPITAL)

# ── Metrics ───────────────────────────────────────────────
st.subheader("Strategy Performance")

strat_return = (results["Strategy_Value"].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
bh_return    = (results["BuyHold_Value"].iloc[-1]  - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
strat_sharpe = calculate_sharpe(results["Strategy_Value"])
bh_sharpe    = calculate_sharpe(results["BuyHold_Value"])
strat_dd     = calculate_maxdd(results["Strategy_Value"]) * 100
bh_dd        = calculate_maxdd(results["BuyHold_Value"])  * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("ML Strategy Return",  f"{strat_return:.2f}%", 
              delta=f"{strat_return - bh_return:.2f}% vs Buy & Hold")
with col2:
    st.metric("Sharpe Ratio",        f"{strat_sharpe:.2f}", 
              delta=f"{strat_sharpe - bh_sharpe:.2f} vs Buy & Hold")
with col3:
    st.metric("Max Drawdown",        f"{strat_dd:.2f}%", 
              delta=f"{strat_dd - bh_dd:.2f}% vs Buy & Hold")

st.divider()

# ── Chart 1: Price + Indicators ───────────────────────────
st.subheader("Technical Indicators — Full Period")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(df["Close"],   label="Close",    color="black",  linewidth=1)
axes[0].plot(df["SMA_20"],  label="SMA 20",   color="blue",   linewidth=1)
axes[0].plot(df["SMA_50"],  label="SMA 50",   color="orange", linewidth=1)
axes[0].fill_between(df.index, df["BB_Upper"], df["BB_Lower"],
                      alpha=0.15, color="grey", label="Bollinger Bands")
axes[0].set_ylabel("Price (INR)")
axes[0].legend(fontsize=8)

axes[1].plot(df["RSI_14"], color="purple", linewidth=1)
axes[1].axhline(70, color="red",   linestyle="--", linewidth=0.8)
axes[1].axhline(30, color="green", linestyle="--", linewidth=0.8)
axes[1].set_ylabel("RSI")
axes[1].set_ylim(0, 100)

axes[2].bar(df.index, df["Volume"], color="steelblue", alpha=0.5)
axes[2].plot(df["Volume_MA_20"], color="red", linewidth=1)
axes[2].set_ylabel("Volume")

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# ── Chart 2: Backtest Results ─────────────────────────────
st.subheader("Backtest Results — Test Period (2023)")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(results.index, results["Strategy_Value"],
             label="ML Strategy", color="blue",   linewidth=1.5)
axes[0].plot(results.index, results["BuyHold_Value"],
             label="Buy & Hold",  color="orange", linewidth=1.5)
axes[0].axhline(INITIAL_CAPITAL, color="red", 
                linestyle="--", linewidth=0.8, label="Initial Capital")
axes[0].set_ylabel("Portfolio Value (INR)")
axes[0].legend()

def compute_drawdown(series):
    rolling_max = series.cummax()
    return (series - rolling_max) / rolling_max * 100

axes[1].fill_between(results.index,
                      compute_drawdown(results["Strategy_Value"]),
                      0, alpha=0.4, color="blue",   label="ML Strategy")
axes[1].fill_between(results.index,
                      compute_drawdown(results["BuyHold_Value"]),
                      0, alpha=0.4, color="orange", label="Buy & Hold")
axes[1].set_ylabel("Drawdown (%)")
axes[1].legend()

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# ── Chart 3: Feature Importance ───────────────────────────
st.subheader("Feature Importance")

importance = pd.Series(
    model.feature_importances_,
    index=X_test.columns
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
importance.plot(kind="barh", color="steelblue", ax=ax)
ax.set_xlabel("Importance Score")
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown("Built by Aditya | BITS PILANI Student | XGBoost + Custom Backtesting Engine")

