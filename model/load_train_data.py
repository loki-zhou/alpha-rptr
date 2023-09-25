import os
import pandas as pd
import numpy as np

from pandas_ta.statistics import zscore
from custom_indicators import pa, pullback, smi, sr


windows_size = 100

def load_data():
    # df = ak.stock_zh_a_daily("sz000625", start_date="20200101")
    # df = ak.stock_zh_a_daily("sh601318", start_date="20200101")
    # df.set_index("date")
    # df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
    df = pd.read_csv("./data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    return feature_deal(df)




def load_test_data():
    # df = ak.stock_zh_a_daily("sz000625", start_date="20200101")
    # df = ak.stock_zh_a_daily("sh601318", start_date="20200101")
    # df.set_index("date")
    # df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
    df = pd.read_csv("./data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    return feature_deal(df)


def normal_ochlv(df):
    df['normal_close'] = zscore(df['close'], length=windows_size )
    df['normal_open'] = zscore(df['open'], length=windows_size )
    df['normal_high'] = zscore(df['high'], length=windows_size )
    df['normal_low'] = zscore(df['low'], length=windows_size )
    df['normal_volume'] = zscore(df['volume'], length=windows_size )
    return df

def feature_deal(df):
    df = normal_ochlv(df)
    df = pa.pinbar(df)
    pullback.detect_pullback(df)
    smi.smi_trend(df)
    sr.support_resistance(df)
    df.dropna(inplace=True)
    df.drop()

    return df

if __name__ == "__main__":
    df = load_data()
    print(df.tail())