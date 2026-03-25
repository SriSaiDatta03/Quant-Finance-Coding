import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

tickers=["AAPL","NVDA","TSLA","MSFT","GLD","TLT","SLV"]
data=yf.download(tickers,start="2020-01-01",end="2025-12-31")["Close"]
returns=data.pct_change().dropna()

corr_matrix=returns.corr()

# Correlation Heatmap
plt.figure(figsize=(8,6))

plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()

plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Lowest Correlation Pair
corr_pairs = corr_matrix.unstack()

# Remove self-correlation
corr_pairs = corr_pairs[corr_pairs < 0.999]

# Remove duplicate pairs (AAPL-TSLA and TSLA-AAPL)
corr_pairs = corr_pairs.sort_index()
corr_pairs = corr_pairs[~corr_pairs.index.duplicated()]

# Sort ascending → lowest correlation
lowest_pair = corr_pairs.sort_values().head(1)

print("\nLowest Correlation Pair:\n")
print(lowest_pair)