from pandas import DataFrame
import pandas as pd
import numpy as np

from src.indicators import sma, ema, wma, double_ema, triple_ema


def support_resistance(df: DataFrame, periods=30):
    high = df['high']
    low = df['low']
    close = df['close']
    pl = low.rolling(window=periods * 2 + 1).min()
    ph = high.rolling(window=periods * 2 + 1).max()
    df["feature_support"] = close/pl
    df["feature_resistance"] = close/ph
    return df

if __name__ == "__main__":
    from loaddata import test_data

    df = test_data()
    df = support_resistance(df)
    print(df.tail())