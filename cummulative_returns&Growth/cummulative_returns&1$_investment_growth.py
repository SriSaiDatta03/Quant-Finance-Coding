import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

tickers=["AAPL","NVDA","TSLA"]
data=yf.download(tickers,start='2020-01-01',end='2025-12-31')["Close"]
returns=data.pct_change().dropna()

cumulative_returns=(1+returns).cumprod()

plt.figure(figsize=(10,6))

for cum in cumulative_returns.columns:
    plt.plot(cumulative_returns[cum],label=cum)

plt.title("Growth of $1 Investment")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")

plt.legend()
plt.show()