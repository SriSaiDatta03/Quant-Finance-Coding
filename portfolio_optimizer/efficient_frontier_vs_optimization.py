import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

tickers=["AAPL","NVDA","TSLA","MSFT","GOOGL"]
data=yf.download(tickers,start="2020-01-01",end="2025-12-31")["Close"]
returns=data.pct_change().dropna()

mean_returns=returns.mean()*252
cov_matrix=returns.cov()*252

num_assets=len(tickers)

# Portfolio Functions
def portfolio_performance(weights):
    ret=np.dot(weights,mean_returns)
    vol=np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights)))
    return ret,vol

def negative_sharpe(weights,risk_free_rate=0.02):
    ret,vol=portfolio_performance(weights)
    return -(ret-risk_free_rate)/vol


# MONTE CARLO SIMULATION

num_portfolios=5000

mc_returns=[]
mc_vols=[]
mc_sharpes=[]

for _ in range(num_portfolios):
    weights=np.random.random(num_assets)
    weights/=np.sum(weights)

    ret,vol=portfolio_performance(weights)
    sharpe=(ret-0.02)/vol

    mc_returns.append(ret)
    mc_vols.append(vol)
    mc_sharpes.append(sharpe)

# Constrains
constraints=({"type":"eq","fun":lambda x:np.sum(x)-1})
bounds=tuple((0,0.4) for _ in range(num_assets))

init_guess=num_assets*[1./num_assets]

# Max Sharpe Portfolio
max_sharpe=minimize(negative_sharpe,
                    init_guess,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints)

max_sharpe_weights=max_sharpe.x
max_ret,max_vol=portfolio_performance(max_sharpe_weights)

# Min Volatility Portfolio
def portfolio_volatility(weights):
    return portfolio_performance(weights)[1]

min_vol=minimize(portfolio_volatility,
                    init_guess,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints)

min_vol_weights=min_vol.x
min_ret,min_vol_val=portfolio_performance(min_vol_weights)

# Efficient Frontier
max_possible_return=np.max(mean_returns)
target_returns=np.linspace(min_ret,max_possible_return,100)
efficient_vols=[]

for target in target_returns:
    constraints_ef=(
        {"type":"eq","fun":lambda x: np.sum(x)-1},
        {"type":"eq","fun":lambda x: np.dot(x,mean_returns)-target}
    )

    ef=minimize(portfolio_volatility,
                init_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints_ef)
    
    if ef.success:
        efficient_vols.append(ef.fun)
    else:
        efficient_vols.append(np.nan)

# Capital Market Line (CML)
risk_free_rate=0.02

cml_x=np.linspace(0,max_vol,1000)
cml_y=risk_free_rate+(max_ret-risk_free_rate)/max_vol*cml_x

# Plot
plt.figure(figsize=(10,6))

#  Monte Carlo cloud
plt.scatter(mc_vols,mc_returns,c=mc_sharpes,cmap="viridis",alpha=0.5)
# Efficient Frontier (optimized)
plt.plot(efficient_vols,target_returns,linewidth=3,label='Efficient Frontier')
# Min Vol
plt.scatter(min_vol_val, min_ret,s=150,label='Min Vol')
#  Max Sharpe
plt.scatter(max_vol, max_ret, s=150,label='Max Sharpe')
# CML
plt.plot(cml_x, cml_y,linestyle='--',label='Capital Market Line')

plt.colorbar(label="Sharpe Ratio")

plt.xlabel("Volatility")
plt.ylabel("Return")
plt.title("Efficient Frontier: Monte Carlo vs Optimization")

plt.legend()
plt.show()

# Results
print("\nMax Sharpe Portfolio:")
for t, w in zip(tickers,max_sharpe_weights):
    print(f"{t}: {w:.4f}")

print("\nMin Volatility Portfolio:")
for t, w in zip(tickers,min_vol_weights):
    print(f"{t}: {w:.4f}")