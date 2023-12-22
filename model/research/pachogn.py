import numpy as np
import pandas as pd

def reward_sortino_function(history):
    returns = pd.Series(history).pct_change().dropna()
    downside_returns = returns.copy()
    downside_returns[returns < 0] = returns ** 2
    expected_return = returns.mean()
    downside_std = np.sqrt(np.std(downside_returns))
    if downside_std == 0 :
        return 0
    return (expected_return + 1E-9) / (downside_std + 1E-9)

def reward_sortino_functionv2(history):
    returns = pd.Series(history).pct_change().dropna()
    downside_returns = returns.copy()
    downside_returns = downside_returns.apply(lambda x: 0 if x > 0 else x)
    downside_returns = downside_returns ** 2
    expected_return = returns.mean()
    downside_std = np.sqrt(np.std(downside_returns))
    if downside_std == 0 :
        return 0
    return (expected_return + 1E-9) / (downside_std + 1E-9)


history1 = [1.0, 1.2, 1.3, 1.4, 1.5, 1.3, 1.7, 1.9]
history2 = [1.0, 1.2, 1.3, 1.4, 1.5, 1.4, 1.7, 1.9]
history3 = [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 0.8]
history4 = [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 0.8]


print(reward_sortino_function(history1))
print(reward_sortino_functionv2(history1))
print("="*20)
print(reward_sortino_function(history2))
print(reward_sortino_functionv2(history2))
print("="*20)
print(reward_sortino_function(history3))
print(reward_sortino_functionv2(history3))
print("="*20)
print(reward_sortino_function(history4))
print(reward_sortino_functionv2(history4))

from empyrical import sortino_ratio