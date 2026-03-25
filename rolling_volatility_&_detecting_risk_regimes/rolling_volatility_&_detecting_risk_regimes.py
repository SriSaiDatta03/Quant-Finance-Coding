import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

ticker="TSLA"
prices=yf.download(ticker,start='2020-01-01',end='2025-12-31')['Close']
prices=prices.squeeze()

log_returns=np.log(prices/prices.shift(1)).dropna()
rolling_vol_20=log_returns.rolling(20).std()*np.sqrt(252)
rolling_vol_50=log_returns.rolling(50).std()*np.sqrt(252)

high_risk=rolling_vol_20>0.40

plt.figure(figsize=(14,8))

# Price plot
plt.subplot(2,1,1)
plt.plot(prices, label="Price")
plt.title("Stock Price")
plt.legend()

# Volatility plot
plt.subplot(2,1,2)
plt.plot(rolling_vol_20, label="20-Day Volatility",color="red")
plt.plot(rolling_vol_50, label="50-Day Volatility",color="blue")

# Highlight high-risk zones
plt.fill_between(
    rolling_vol_20.index,
    rolling_vol_20,
    0,
    where=high_risk.values,
    color='red',
    alpha=0.3,
    label="High Risk Zone"
)

plt.title("Rolling Volatility (Annualized)")
plt.legend()

plt.tight_layout()
plt.show()