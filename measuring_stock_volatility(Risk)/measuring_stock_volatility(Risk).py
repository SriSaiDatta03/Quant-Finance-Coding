import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


tickers=["AAPL","NVDA","TSLA"]
data=yf.download(tickers,start="2020-01-01",end="2025-12-31")["Close"]

log_returns=np.log(data/data.shift(1)).dropna()
annual_vol=log_returns.std()*np.sqrt(252)
annual_vol_pct=annual_vol*100

print("Annual Volatility (%):")
for stock in annual_vol_pct.index:
    print(f"{stock}:{annual_vol_pct[stock]:.2f}%")

plt.figure(figsize=(10,6))
plt.bar(annual_vol_pct.index,annual_vol_pct.values)
plt.title("Annual Volatility Comparison")
plt.xlabel("Stocks")
plt.ylabel("Volatility (%)")
plt.show()