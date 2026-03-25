import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

tickers=["AAPL","NVDA","TSLA"]
prices=yf.download(tickers,start='2020-01-01',end='2025-12-31')['Close'].squeeze()
returns=prices.pct_change().dropna()

equity_curve=(1+returns).cumprod()
running_peak=equity_curve.cummax()
drawdown=(equity_curve-running_peak)/running_peak

max_drawdown = drawdown.min()

# 🔹 Print clean table
table_df = pd.DataFrame({
    "Max Drawdown": max_drawdown
})

table_df["Max Drawdown"] = table_df["Max Drawdown"].map(lambda x: f"{x:.2%}")

print("\n Max Drawdown Summary: \n")
print(table_df)

# 🔹 Plot ONLY equity curves
plt.figure(figsize=(12,6))

for col in equity_curve.columns:
    plt.plot(equity_curve[col], label=col)

plt.title("Equity Curve Comparison (Growth of $1)")
plt.legend()
plt.grid(True)

plt.show()