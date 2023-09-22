import os
import pandas as pd
import numpy as np
import gymnasium as gym
from gym_trading_env.environments import TradingEnv
import src.strategies.legendary_ta as lta
import pandas_ta as ta
from pandas_ta.statistics import zscore
from model.load_train_data import load_test_data

def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2]) #log (p_t / p_t-1 )

def reward_sortino_function(history):
    returns = pd.Series(history["portfolio_valuation"][-(15+1):]).pct_change().dropna()
    downside_returns = returns.copy()
    downside_returns[returns < 0] = returns ** 2
    expected_return = returns.mean()
    downside_std = np.sqrt(np.std(downside_returns))
    if downside_std == 0 :
        return 0
    return (expected_return + 1E-9) / (downside_std + 1E-9)
def max_drawdown(history):
    networth_array = history['portfolio_valuation']
    _max_networth = networth_array[0]
    _max_drawdown = 0
    for networth in networth_array:
        if networth > _max_networth:
            _max_networth = networth
        drawdown = ( networth - _max_networth ) / _max_networth
        if drawdown < _max_drawdown:
            _max_drawdown = drawdown
    return f"{_max_drawdown*100:5.2f}%"


def create_env(config):
    df = load_test_data()
    env = TradingEnv(
        name="BTCUSD",
        df=df,
        windows=1,
        # positions=[-1, -0.5, 0, 0.5, 1],  # From -1 (=SHORT), to +1 (=LONG)
        # positions=[0, 0.5, 1],  # From -1 (=SHORT), to +1 (=LONG)
        positions=[0,  1],  # From -1 (=SHORT), to +1 (=LONG)
        # initial_position = 'random', #Initial position
        dynamic_feature_functions=[],
        initial_position=0,  # Initial position
        trading_fees=0.1 / 100,  # 0.01% per stock buy / sell
        borrow_interest_rate=0,  # per timestep (= 1h here)
        reward_function=reward_sortino_function,
        portfolio_initial_value=10000,  # in FIAT (here, USD)
        # max_episode_duration = 2400,
        # max_episode_duration=500,
    )
    env.add_metric('Position Changes', lambda history : f"{ 100*np.sum(np.diff(history['position']) != 0)/len(history['position']):5.2f}%" )
    env.add_metric('Max Drawdown', max_drawdown)
    env = gym.wrappers.NormalizeObservation(env)
    return env

from ray.tune.registry import register_env
register_env("TradingEnv2", create_env)



import ray
from ray import tune, air
from model.ray_callback import TradeMetricsCallbacks
from ray.rllib.algorithms.algorithm import Algorithm
LSTM_CELL_SIZE = 256
def test():
    # checkpoint_path = r"D:\rl\backtrader\example\gym\ray_results\PPO\PPO_TradingEnv2_1ab4e_00000_0_2023-09-13_18-34-23\checkpoint_000612"
    checkpoint_path = r"D:\rl\alpha-rptr\model\ray_results\PPO\PPO_TradingEnv2_3d9d9_00000_0_2023-09-22_16-08-40\checkpoint_000320"

    algo = Algorithm.from_checkpoint(checkpoint_path)
    env = create_env(0)

    done, truncated = False, False
    obs, info = env.reset()
    lstm_states = None
    init_state = state = [
     np.zeros([LSTM_CELL_SIZE], np.float32) for _ in range(2)
    ]
    prev_a = 0
    prev_r = 0.0
    while not done and not truncated:
        a, state_out, _ = algo.compute_single_action(
            observation=obs, state=state, prev_action=prev_a, prev_reward=prev_r)
        obs, reward, done, truncated, _ = env.step(a)
        if done:
            obs, info = env.reset()
            state = init_state
            prev_a = 0
            prev_r = 0.0
        else:
            pass
            state = state_out
            prev_a = a
            # prev_r = reward
    env.save_for_render(dir="./render_logs")

# tensorboard.exe  --logdir model/ray_results/PPO/
#pip install  ray[rllib]==2.4.0
if __name__ == '__main__':
    test()