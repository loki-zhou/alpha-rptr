import os
import pandas as pd
import numpy as np
import gymnasium as gym
from gym_trading_env.environments import TradingEnv
import src.strategies.legendary_ta as lta
import pandas_ta as ta
from pandas_ta.statistics import zscore
from model.load_train_data import load_data

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
    df = load_data()
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
    # env = gym.wrappers.NormalizeObservation(env)
    return env

from ray.tune.registry import register_env
register_env("TradingEnv2", create_env)



import ray
from ray import tune, air
from model.ray_callback import TradeMetricsCallbacks

LSTM_CELL_SIZE = 256
def train():
    ray.init(num_cpus=8)

    configs = {
        "PPO": {
            "num_sgd_iter": 16,
            "model": {
                "vf_share_layers": True,
            },
            "vf_loss_coeff": 0.0001,
            "lambda": 0.95,
            "gamma": 0.99,
        },
        "IMPALA": {
            "num_workers": 2,
            "num_gpus": 0,
            "vf_loss_coeff": 0.01,
        },

    }

    config = dict(
        configs["PPO"],
        **{
            "env": "TradingEnv2",
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                "use_lstm": True,
                "lstm_cell_size": LSTM_CELL_SIZE,
                "lstm_use_prev_action": True,
                "lstm_use_prev_reward": True,
            },
            "framework": "torch",
            "_enable_learner_api": False,
            "_enable_rl_module_api": False,
            # "observation_filter": "MeanStdFilter",
            "lr": 8e-6,
            "lr_schedule": [
                [0, 1e-1],
                [int(1e3), 1e-2],
                [int(1e4), 1e-3],
                [int(1e5), 1e-4],
                [int(1e6), 1e-5],
                [int(1e7), 1e-6],
                [int(1e8), 1e-7]
            ],
            "callbacks": TradeMetricsCallbacks,
            # "observation_filter": "MeanStdFilter",  # ConcurrentMeanStdFilter, NoFilter
        }
    )

    stop = {
        # "training_iteration": args.stop_iters,
        "timesteps_total": 10000_0000,
        "episode_reward_mean": 1000,
    }

    tuner = tune.Tuner(
        "PPO", param_space=config, run_config=air.RunConfig(stop=stop,
                                                            checkpoint_config=air.CheckpointConfig(
                                                                num_to_keep= 2,
                                                                checkpoint_frequency = 10,
                                                                checkpoint_at_end=True),
                                                            verbose=2,
                                                            local_dir = "./ray_results")
    )
    #tuner = tuner.restore(r"D:\rl\alpha-rptr\model\ray_results\PPO\")
    # tuner.fit()

    results = tuner.fit()

    ckpt = results.get_best_result(metric="episode_reward_mean", mode="max").checkpoint

    print(ckpt)


# tensorboard.exe  --logdir model/ray_results/PPO/
#pip install  ray[rllib]==2.4.0
if __name__ == '__main__':
    train()