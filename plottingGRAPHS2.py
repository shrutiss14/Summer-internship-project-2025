import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Prepare the data
'''data = {
    "Metric": ["CAGR", "Volatility"],
    "Max Sharpe": [23.94, 18.37],
    "Global min Var": [15.84, 11.50],
    "Equally weighted portfolio": [35.34, 24.63],
    "Nifty 500 index": [14.68, 15.70],
    "Mutual Fund": [24, 47.94]
}
df = pd.DataFrame(data)

# Plot settings
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.14
x = np.arange(len(df))

# Define colors
colors = ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
portfolios = list(df.columns[1:])  

bars = []
for i, portfolio in enumerate(portfolios):
    bars_i = ax.bar(x + (i - len(portfolios)/2) * bar_width + bar_width/2,
                    df[portfolio],
                    width=bar_width,
                    label=portfolio,
                    color=colors[i])
    bars.append(bars_i)


# Add value labels
def add_labels(bars_group):
    for bars in bars_group:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=9)

add_labels(bars)


ax.set_xticks(x)
ax.set_xticklabels(df["Metric"], fontsize=12)
ax.set_ylabel("Percentage", fontsize=12)
ax.set_title("Performance Comparison: CAGR and Volatility", fontsize=14)
ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
ax.set_ylim(0, df.drop("Metric", axis=1).values.max() * 1.15)

plt.tight_layout()
plt.show()'''

'''data = {
    "Metric": ["CAGR", "Volatility"],
    "Mean Historical Return": [20.17,17.65],
    "EW Historical Return": [15.30, 16.16],
    "CAPM": [23.94, 18.37]
}
df = pd.DataFrame(data)

# Plot settings
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.20
x = np.arange(len(df))

# Define colors
colors = ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
portfolios = list(df.columns[1:])  

bars = []
for i, portfolio in enumerate(portfolios):
    bars_i = ax.bar(x + (i - len(portfolios)/2) * bar_width + bar_width/2,
                    df[portfolio],
                    width=bar_width,
                    label=portfolio,
                    color=colors[i])
    bars.append(bars_i)


# Add value labels
def add_labels(bars_group):
    for bars in bars_group:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=9)

add_labels(bars)


ax.set_xticks(x)
ax.set_xticklabels(df["Metric"], fontsize=12)
ax.set_ylabel("Percentage", fontsize=12)
ax.set_title("Performance Comparison: CAGR and Volatility", fontsize=14)
ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
ax.set_ylim(0, df.drop("Metric", axis=1).values.max() * 1.15)

plt.tight_layout()
plt.show()'''

'''data = {
    "Metric": ["CAGR", "Volatility"],
    "Sample Covariance": [11.64,12.75],
    "EW sample covariance": [11.98, 12.57],
    "Shrinkage covariance": [15.84, 11.50]
}
df = pd.DataFrame(data)

# Plot settings
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.20
x = np.arange(len(df))

# Define colors
colors = ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
portfolios = list(df.columns[1:])  

bars = []
for i, portfolio in enumerate(portfolios):
    bars_i = ax.bar(x + (i - len(portfolios)/2) * bar_width + bar_width/2,
                    df[portfolio],
                    width=bar_width,
                    label=portfolio,
                    color=colors[i])
    bars.append(bars_i)


# Add value labels
def add_labels(bars_group):
    for bars in bars_group:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=9)

add_labels(bars)


ax.set_xticks(x)
ax.set_xticklabels(df["Metric"], fontsize=12)
ax.set_ylabel("Percentage", fontsize=12)
ax.set_title("Performance Comparison: CAGR and Volatility", fontsize=14)
ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
ax.set_ylim(0, df.drop("Metric", axis=1).values.max() * 1.15)

plt.tight_layout()
plt.show()'''

data = {
    "Metric": ["CAGR", "Volatility"],    
    "Max Sharpe": [23.94, 18.37],
    "Global min Var": [15.84, 11.50]
}
df = pd.DataFrame(data)

# Plot settings
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.25
x = np.arange(len(df))

# Define colors
colors = ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
portfolios = list(df.columns[1:])  

bars = []
for i, portfolio in enumerate(portfolios):
    bars_i = ax.bar(x + (i - len(portfolios)/2) * bar_width + bar_width/2,
                    df[portfolio],
                    width=bar_width,
                    label=portfolio,
                    color=colors[i])
    bars.append(bars_i)


# Add value labels
def add_labels(bars_group):
    for bars in bars_group:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=9)

add_labels(bars)


ax.set_xticks(x)
ax.set_xticklabels(df["Metric"], fontsize=12)
ax.set_ylabel("Percentage", fontsize=12)
ax.set_title("Performance Comparison: CAGR and Volatility", fontsize=14)
ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
ax.set_ylim(0, df.drop("Metric", axis=1).values.max() * 1.15)

plt.tight_layout()
plt.show()
