import os
import pandas as pd
import numpy as np
import gymnasium as gym
from gym_trading_env.environments import TradingEnv
import src.strategies.legendary_ta as lta
import pandas_ta as ta
from pandas_ta.statistics import zscore
from custom_indicators import normalization
from ray.rllib.models.utils import get_activation_fn, get_filter_config

windows_size = 50

CustomStrategy = ta.Strategy(
    name="Momo and Volatility",
    description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
    ta=[
        # {"kind": "sma", "length": 30,"prefix": "feature"},
        # {"kind": "sma", "length": 50,"prefix": "feature"},
        # {"kind": "sma", "length": 200, "prefix": "feature"},
        # {"kind": "bbands", "length": 20, "prefix": "feature"},
        # {"kind": "rsi", "prefix": "feature"},
        # {"kind": "macd", "fast": 8, "slow": 21, "prefix": "feature"},
        # {"kind": "sma", "close": "volume", "length": 20, "prefix": "feature_VOLUME"},
        # {"kind": "mfi", "prefix": "feature"},
        # {"kind": "tsi", "prefix": "feature"},
        # {"kind": "uo", "prefix": "feature"},
        # {"kind": "ao", "prefix": "feature"},
        # {"kind": "vortex", "prefix": "feature"},
        # {"kind": "trix", "prefix": "feature"},
        # {"kind": "massi", "prefix": "feature"},
        # {"kind": "cci", "prefix": "feature"},
        {"kind": "dpo", "prefix": "feature"},
        # {"kind": "kst", "prefix": "feature"},
        # {"kind": "aroon", "prefix": "feature"},
        # {"kind": "kc", "prefix": "feature"},
        # {"kind": "donchian", "prefix": "feature"},
        # {"kind": "cmf", "prefix": "feature"},
        # {"kind": "efi", "prefix": "feature"},
        # {"kind": "pvt", "prefix": "feature"},
        # {"kind": "nvi", "prefix": "feature"},
    ]
)

def load_data():
    # df = ak.stock_zh_a_daily("sz000625", start_date="20200101")
    # df = ak.stock_zh_a_daily("sh601318", start_date="20200101")
    # df.set_index("date")
    # df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
    df = pd.read_csv("./data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
    # df.sort_index(inplace=True)
    # df.dropna(inplace=True)
    # df.drop_duplicates(inplace=True)
    # df["feature_return_close"] = df["close"].pct_change()
    # df["feature_diff_open"] = df["open"] / df["close"]
    # df["feature_diff_high"] = df["high"] / df["close"]
    # df["feature_diff_low"] = df["low"] / df["close"]
    # df["feature_diff_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
    # cta.NormalizedScore(df, 30*2)
    # df = lta.smi_momentum(df)
    # lta.pinbar(df, df["feature_smi"])
    # df["feature_smi"] = df["feature_smi"] / 100

    # df.ta.cores = 0
    # df.ta.strategy(CustomStrategy)
    # df['feature_z_close'] = zscore(df['close'], length=windows_size )
    # df['feature_z_open'] = zscore(df['open'], length=windows_size )
    # df['feature_z_high'] = zscore(df['high'], length=windows_size )
    # df['feature_z_low'] = zscore(df['low'], length=windows_size )
    # df['feature_z_volume'] = zscore(df['volume'], length=windows_size )


    df = normalization.highlow_winodws_ochlv(df, windows_size = windows_size)
    # df = normalization.logged_diff(df)
    df.dropna(inplace=True)
    return df

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
    env = gym.wrappers.NormalizeObservation(env)
    return env

from ray.tune.registry import register_env
register_env("TradingEnv2", create_env)

# stop = {
#     "training_iteration": args.stop_iters,
#     "timesteps_total": args.stop_timesteps,
#     "episode_reward_mean": args.stop_reward,
# }


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
        #         "conv_filters": [ [16, [8, 8], 4],
        # [32, [4, 4], 2],
        # [256, [11, 11], 1],],
                "use_lstm": True,
                "max_seq_len": 20,
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
            "observation_filter": "MeanStdFilter",  # ConcurrentMeanStdFilter, NoFilter
        }
    )

    stop = {
        # "training_iteration": args.stop_iters,
        "timesteps_total": 2000_0000,
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
    # tuner = tuner.restore(r"D:\rl\backtrader\example\gym\ray_results\PPO")
    # tuner = tuner.restore(r"D:\rl\alpha-rptr\model\ray_results\PPO")

    # tuner.fit()

    results = tuner.fit()

    ckpt = results.get_best_result(metric="episode_reward_mean", mode="max").checkpoint

    print(ckpt)


def train2():
    pass

from ray.rllib.algorithms.algorithm import Algorithm

def test():
    # checkpoint_path = r"D:\rl\backtrader\example\gym\ray_results\PPO\PPO_TradingEnv2_1ab4e_00000_0_2023-09-13_18-34-23\checkpoint_000612"
    checkpoint_path = r"D:\rl\backtrader\example\gym\ray_results\PPO\PPO_TradingEnv2_ca408_00000_0_2023-09-15_14-47-51\checkpoint_000400"

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
            state = state_out
            prev_a = a
            prev_r = reward
    env.save_for_render(dir="./render_logs")

# tensorboard.exe  --logdir model/ray_results/PPO/
#pip install  ray[rllib]==2.4.0
if __name__ == '__main__':
    # load_data()
    train()
    # test()
    # env = create_env(0)
    # print(env.observation_space.shape)
