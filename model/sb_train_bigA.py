import os
import pandas as pd
import numpy as np
import gymnasium as gym
from gym_trading_env.environments import TradingEnv
# import src.strategies.legendary_ta as lta
# import pandas_ta as ta
# from pandas_ta.statistics import zscore
from custom_indicators import normalization
from ray.rllib.models.utils import get_activation_fn, get_filter_config

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib import QRDQN

windows_size = 50

# CustomStrategy = ta.Strategy(
#     name="Momo and Volatility",
#     description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
#     ta=[
#         # {"kind": "sma", "length": 30,"prefix": "feature"},
#         # {"kind": "sma", "length": 50,"prefix": "feature"},
#         # {"kind": "sma", "length": 200, "prefix": "feature"},
#         # {"kind": "bbands", "length": 20, "prefix": "feature"},
#         # {"kind": "rsi", "prefix": "feature"},
#         # {"kind": "macd", "fast": 8, "slow": 21, "prefix": "feature"},
#         # {"kind": "sma", "close": "volume", "length": 20, "prefix": "feature_VOLUME"},
#         # {"kind": "mfi", "prefix": "feature"},
#         # {"kind": "tsi", "prefix": "feature"},
#         # {"kind": "uo", "prefix": "feature"},
#         # {"kind": "ao", "prefix": "feature"},
#         # {"kind": "vortex", "prefix": "feature"},
#         # {"kind": "trix", "prefix": "feature"},
#         # {"kind": "massi", "prefix": "feature"},
#         # {"kind": "cci", "prefix": "feature"},
#         {"kind": "dpo", "prefix": "feature"},
#         # {"kind": "kst", "prefix": "feature"},
#         # {"kind": "aroon", "prefix": "feature"},
#         # {"kind": "kc", "prefix": "feature"},
#         # {"kind": "donchian", "prefix": "feature"},
#         # {"kind": "cmf", "prefix": "feature"},
#         # {"kind": "efi", "prefix": "feature"},
#         # {"kind": "pvt", "prefix": "feature"},
#         # {"kind": "nvi", "prefix": "feature"},
#     ]
# )

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


    #df = normalization.normal_deal(df, windows_size = windows_size)
    # df = normalization.logged_diff(df)
    df = normalization.create_alphas(df)
    df.dropna(inplace=True)
    return df

def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2]) #log (p_t / p_t-1 )

def reward_sortino_function(history):
    returns = pd.Series(history["portfolio_valuation"][-(windows_size+1):]).pct_change().dropna()
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



def create_env():
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
        trading_fees= 1 / 100,  # 0.01% per stock buy / sell
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

def create_test_env():
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
        trading_fees= 0.01 / 100,  # 0.01% per stock buy / sell
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

monitor_dir = r'./monitor_log/ppo/'
os.makedirs(monitor_dir, exist_ok=True)

def train():
    # env = create_env(3)
    # training_envs = gym.vector.SyncVectorEnv(
    #     [lambda: create_env() for _ in range(5)])
    training_envs = create_env()
    model = PPO("MlpPolicy", training_envs, tensorboard_log="./tlog/ppo/", verbose=1, batch_size= 512)
    # model = PPO("MlpPolicy", training_envs, verbose=1, batch_size=512)
    #model = QRDQN("MlpPolicy", env, verbose=1)
    # model = RecurrentPPO("MlpLstmPolicy", env,
    #                      batch_size=512,
    #                      # n_steps=128,
    #                      # n_epochs=10,
    #                      # policy_kwargs={'enable_critic_lstm': False, 'lstm_hidden_size': 128},
    #                      tensorboard_log="./tlog/ppo/",
    #                      verbose=1)
    #callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=monitor_dir)
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=monitor_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    # model.set_parameters(monitor_dir + "rl_model_20000000_steps.zip")
    model.learn(total_timesteps=200_0000, callback=checkpoint_callback)
    # model.learn(total_timesteps=200_0000)


def test():
    #model = QRDQN.load(monitor_dir + "rl_model_500000_steps.zip")
    model = PPO.load(monitor_dir + "rl_model_2000000_steps.zip")
    # model = RecurrentPPO.load(monitor_dir + "rl_model_2000000_steps.zip")
    env = create_test_env()
    model.set_env(env)
    done, truncated = False, False
    observation, info = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    while not done and not truncated:
        action, _states = model.predict(observation)
        #action = env.action_space.sample()
        # action, lstm_states = model.predict(observation, state=lstm_states, episode_start=episode_starts, deterministic=True)
        observation, reward, done, truncated, info = env.step(action)
    env.save_for_render(dir="./render_logs")


# tensorboard.exe  --logdir model/ray_results/PPO/
#tensorboard.exe --logdir model/tlog/ppo
#pip install  ray[rllib]==2.4.0
if __name__ == '__main__':
    # load_data()
    # train()
    test()
    # env = create_env(0)
    # print(env.observation_space.shape)
