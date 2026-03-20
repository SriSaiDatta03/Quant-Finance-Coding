# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Download Stock data
ticker='AAPL'
data=yf.download(ticker,start='2010-01-01',end='2025-12-31')
prices=data['Close']

# Simple Returns
simple_returns=prices.pct_change().dropna()
# Log Returns
log_returns=np.log(prices/prices.shift(1)).dropna()
# Cumulative Returns
cumulative_simple_returns=(1+simple_returns).cumprod()
cumulative_log_returns=log_returns.cumsum()

# Plot
plt.figure(figsize=(10,6))
plt.plot(cumulative_simple_returns,label="Cumulative Simple Returns",linewidth=2)
plt.plot(np.exp(cumulative_log_returns),label="Cumulative Log Returns (converted)",linewidth=2,alpha=0.6)

plt.legend()
plt.title("Simple vs Log Returns")
plt.show()