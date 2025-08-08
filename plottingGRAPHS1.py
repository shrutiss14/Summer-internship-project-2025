import matplotlib.pyplot as plt
import numpy as np

# Data
'''labels = ["Max Drawdown", "Sharpe Ratio", "Sortino Ratio", "Treynor Ratio"]
stats_max_sharpe = [2.48, 1.26, 1.09, 0.24]
stats_GMV = [2.16, 1.33, 1.22, 0.34]
stats_equally_weighted = [2.59, 1.35, 1.28, 0.33]
stats_nifty500 = [2.12, 0.95, 0.86, 0.20]
stats_MutualFund = [1.65, 0.74, 0.75, 0.32]

x = np.arange(len(labels))  # label locations
bar_width = 0.15

fig, ax = plt.subplots(figsize=(12, 6))


bars1 = ax.bar(x - 2*bar_width, stats_max_sharpe, width=bar_width, label='Max Sharpe', color='#a6cee3')
bars2 = ax.bar(x - bar_width, stats_GMV, width=bar_width, label='Global Min Var', color='#b2df8a')
bars3 = ax.bar(x, stats_equally_weighted, width=bar_width, label='Equally Weighted', color="#fb9a99")
bars4 = ax.bar(x + bar_width, stats_nifty500, width=bar_width, label='Nifty 500', color='#fdbf6f')
bars5 = ax.bar(x + 2*bar_width, stats_MutualFund, width=bar_width, label='Mutual Fund', color='#cab2d6')

# Add value labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.03, f"{height:.2f}", 
                ha='center', va='bottom', fontsize=9)

for bars in [bars1, bars2, bars3, bars4, bars5]:
    add_labels(bars)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Value", fontsize=12)
ax.set_title("Performance Comparison: Risk-Adjusted Metrics", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()'''

'''labels = ["Max Drawdown", "Sharpe Ratio", "Sortino Ratio", "Treynor Ratio"]
stats_Mean_historical_return = [2.53, 1.13, 0.99, 0.22]
stats_EWMA = [2.38, 0.97, 0.82, 0.20]
stats_CAPM = [2.48, 1.26, 1.09, 0.24]

x = np.arange(len(labels))  # label locations
bar_width = 0.20

fig, ax = plt.subplots(figsize=(12, 6))


bars1 = ax.bar(x - 2*bar_width, stats_Mean_historical_return, width=bar_width, label='Mean Historical Return', color='#a6cee3')
bars2 = ax.bar(x - bar_width, stats_EWMA, width=bar_width, label='EW Mean Historical Return', color='#b2df8a')
bars3 = ax.bar(x, stats_CAPM, width=bar_width, label='CAPM Return', color="#fb9a99")

# Add value labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.03, f"{height:.2f}", 
                ha='center', va='bottom', fontsize=9)

for bars in [bars1, bars2, bars3]:
    add_labels(bars)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Value", fontsize=12)
ax.set_title("Performance Comparison: Risk-Adjusted Metrics", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()'''

'''labels = ["Max Drawdown", "Sharpe Ratio", "Sortino Ratio", "Treynor Ratio"]
stats_sample_cov = [2.36, 0.92, 0.83, 0.29]
stats_EWcov = [2.43, 0.96, 0.88, 0.30]
stats_shrinkage = [2.16, 1.33, 1.22, 0.34]

x = np.arange(len(labels))  # label locations
bar_width = 0.20

fig, ax = plt.subplots(figsize=(12, 6))


bars1 = ax.bar(x - 2*bar_width, stats_sample_cov, width=bar_width, label='Sample Covariance', color='#a6cee3')
bars2 = ax.bar(x - bar_width, stats_EWcov, width=bar_width, label='EW sample covariance', color='#b2df8a')
bars3 = ax.bar(x, stats_shrinkage, width=bar_width, label='Shrinkage covariance', color="#fb9a99")

# Add value labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.03, f"{height:.2f}", 
                ha='center', va='bottom', fontsize=9)

for bars in [bars1, bars2, bars3]:
    add_labels(bars)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Value", fontsize=12)
ax.set_title("Performance Comparison: Risk-Adjusted Metrics", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()'''

import numpy as np
import matplotlib.pyplot as plt

labels = ["Max Drawdown", "Sharpe Ratio", "Sortino Ratio", "Treynor Ratio"]
stats_max_sharpe = [2.48, 1.26, 1.09, 0.24]
stats_GMV = [2.16, 1.33, 1.22, 0.34]

x = np.arange(len(labels))  # Group centers
bar_width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(x - bar_width/2, stats_max_sharpe, width=bar_width, label='Max Sharpe', color='#a6cee3')
bars2 = ax.bar(x + bar_width/2, stats_GMV, width=bar_width, label='Global Min Variance', color='#b2df8a')

# Add value labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.03, f"{height:.2f}", 
                ha='center', va='bottom', fontsize=9)

add_labels(bars1)
add_labels(bars2)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Value", fontsize=12)
ax.set_title("Performance Comparison: Risk-Adjusted Metrics", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
