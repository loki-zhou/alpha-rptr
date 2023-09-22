import os
import pandas as pd
import numpy as np

from pandas_ta.statistics import zscore
from custom_indicators import pa, pullback, smi, sr


windows_size = 50

def load_data():
    # df = ak.stock_zh_a_daily("sz000625", start_date="20200101")
    # df = ak.stock_zh_a_daily("sh601318", start_date="20200101")
    # df.set_index("date")
    # df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
    df = pd.read_csv("./data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # df["feature_return_close"] = df["close"].pct_change()
    # df["feature_diff_open"] = df["open"] / df["close"]
    # df["feature_diff_high"] = df["high"] / df["close"]
    # df["feature_diff_low"] = df["low"] / df["close"]
    # df["feature_diff_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
    # df['feature_z_close'] = zscore(df['close'], length=windows_size )
    # df['feature_z_open'] = zscore(df['open'], length=windows_size )
    # df['feature_z_high'] = zscore(df['high'], length=windows_size )
    # df['feature_z_low'] = zscore(df['low'], length=windows_size )
    # df['feature_z_volume'] = zscore(df['volume'], length=windows_size )
    df = pa.pinbar(df)
    pullback.detect_pullback(df)
    smi.smi_trend(df)
    sr.support_resistance(df)

    df.dropna(inplace=True)
    return df


def load_test_data():
    # df = ak.stock_zh_a_daily("sz000625", start_date="20200101")
    # df = ak.stock_zh_a_daily("sh601318", start_date="20200101")
    # df.set_index("date")
    df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
    # df = pd.read_csv("./data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df["feature_return_close"] = df["close"].pct_change()
    df["feature_diff_open"] = df["open"] / df["close"]
    df["feature_diff_high"] = df["high"] / df["close"]
    df["feature_diff_low"] = df["low"] / df["close"]
    df["feature_diff_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
    df['feature_z_close'] = zscore(df['close'], length=windows_size )
    df['feature_z_open'] = zscore(df['open'], length=windows_size )
    df['feature_z_high'] = zscore(df['high'], length=windows_size )
    df['feature_z_low'] = zscore(df['low'], length=windows_size )
    df['feature_z_volume'] = zscore(df['volume'], length=windows_size )
    df = pa.pinbar(df)
    pullback.detect_pullback(df)
    smi.smi_trend(df)
    sr.support_resistance(df)

    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.tail())