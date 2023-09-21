# coding: UTF-8
import os
import random

import math
import re
import time

import numpy
from hyperopt import hp
import pandas as pd
from pandas_ta.statistics import zscore
from ray.rllib.algorithms.algorithm import Algorithm
from src import logger, notify
from src.indicators import (highest, lowest, med_price, avg_price, typ_price, 
                            atr, MAX, sma, bbands, macd, adx, sar, sarext, 
                            cci, rsi, crossover, crossunder, last, rci, 
                            double_ema, ema, triple_ema, wma, ewma, ssma, hull, 
                            supertrend, Supertrend, rsx, donchian, hurst_exponent,
                            lyapunov_exponent)
from src.exchange.bitmex.bitmex import BitMex
from src.exchange.binance_futures.binance_futures import BinanceFutures
from src.exchange.bitmex.bitmex_stub import BitMexStub
from src.exchange.binance_futures.binance_futures_stub import BinanceFuturesStub
from src.bot import Bot
from src.gmail_sub import GmailSub
import src.strategies.legendary_ta as lta
import pandas_ta as ta
import numpy as np

windows_size = 50

CustomStrategy = ta.Strategy(
    name="Momo and Volatility",
    description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
    ta=[
        {"kind": "sma", "length": 30,"prefix": "feature"},
        {"kind": "sma", "length": 50,"prefix": "feature"},
        {"kind": "sma", "length": 200, "prefix": "feature"},
        {"kind": "bbands", "length": 20, "prefix": "feature"},
        {"kind": "rsi", "prefix": "feature"},
        {"kind": "macd", "fast": 8, "slow": 21, "prefix": "feature"},
        {"kind": "sma", "close": "volume", "length": 20, "prefix": "feature_VOLUME"},
        {"kind": "mfi", "prefix": "feature"},
        {"kind": "tsi", "prefix": "feature"},
        {"kind": "uo", "prefix": "feature"},
        {"kind": "ao", "prefix": "feature"},
        {"kind": "vortex", "prefix": "feature"},
        {"kind": "trix", "prefix": "feature"},
        {"kind": "massi", "prefix": "feature"},
        {"kind": "cci", "prefix": "feature"},
        # {"kind": "dpo", "prefix": "feature"},
        {"kind": "kst", "prefix": "feature"},
        {"kind": "aroon", "prefix": "feature"},
        {"kind": "kc", "prefix": "feature"},
        {"kind": "donchian", "prefix": "feature"},
        {"kind": "cmf", "prefix": "feature"},
        {"kind": "efi", "prefix": "feature"},
        {"kind": "pvt", "prefix": "feature"},
        {"kind": "nvi", "prefix": "feature"},
    ]
)

checkpoint_path = r"D:\rl\backtrader\example\gym\ray_results\PPO\PPO_TradingEnv2_0516a_00000_0_2023-09-21_09-41-44\checkpoint_000230"
algo = Algorithm.from_checkpoint(checkpoint_path)

# Candle tester
class RLTrader(Bot):
    def __init__(self):
        Bot.__init__(self, ['15m'])

        self.lstmstate = [np.zeros([256], np.float32) for _ in range(2)]

    # this is for parameter optimization in hyperopt mode
    def options(self):
        return {}

    def ohlcv_len(self):
        return 200

    def strategy(self, action, open, close, high, low, volume):
        # logger.info(f"open: {open[-1]}")
        # logger.info(f"high: {high[-1]}")
        # logger.info(f"low: {low[-1]}")
        # logger.info(f"close: {close[-1]}")
        # logger.info(f"volume: {volume[-1]}")
        obs = self.create_feature(open, close, high, low, volume)
        action = self.predict(obs)
        logger.info(f"action: {action}")

    def create_feature(self, open, close, high, low, volume):
        df = pd.DataFrame(
            {"open": open,
             "high": high,
             "low": low,
             "close": close,
             "volume": volume,
             }
        )
        df["feature_return_close"] = df["close"].pct_change()
        df["feature_diff_open"] = df["open"] / df["close"]
        df["feature_diff_high"] = df["high"] / df["close"]
        df["feature_diff_low"] = df["low"] / df["close"]
        df["feature_diff_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
        df = lta.smi_momentum(df)
        df.ta.cores = 0
        df.ta.strategy(CustomStrategy)
        df['feature_z_close'] = zscore(df['close'], length=windows_size )
        df['feature_z_open'] = zscore(df['open'], length=windows_size )
        df['feature_z_high'] = zscore(df['high'], length=windows_size )
        df['feature_z_low'] = zscore(df['low'], length=windows_size )
        df['feature_z_volume'] = zscore(df['volume'], length=windows_size )
        _features_columns = [col for col in df.columns if "feature" in col]
        _obs_array = np.array(df[_features_columns], dtype=np.float32)
        return _obs_array[[-1]]

    def predict(self, obs):
        action, self.lstmstate, _ = algo.compute_single_action(
            observation=obs, state=self.lstmstate)
        return action
