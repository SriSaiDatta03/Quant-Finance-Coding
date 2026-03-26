import numpy as np
import pandas as pd
import yfinance as yf

tickers=["AAPL","NVDA","TSLA"]
data=yf.download(tickers,start="2020-01-01",end="2025-12-31")["Close"]
returns=data.pct_change().dropna()
annual_returns=returns.mean()*252

annual_volatility=returns.std()*np.sqrt(252)

risk_free_rate=0.02

sharpe_ratio=(annual_returns-risk_free_rate)/annual_volatility

print(f"Sharpe Ratio:{sharpe_ratio}")