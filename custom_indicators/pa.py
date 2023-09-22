from pandas import DataFrame
import pandas as pd
import numpy as np
from src.indicators import sma, ema

def true_range(dataframe):
    prev_close = dataframe['close'].shift()
    tr = pd.concat([dataframe['high'] - dataframe['low'], abs(dataframe['high'] - prev_close), abs(dataframe['low'] - prev_close)], axis=1).max(axis=1)
    return tr


def pinbar(df: DataFrame):
    low = df['low']
    high = df['high']
    close = df['close']
    tr = true_range(df)

    df['feature_pinbar'] = 0
    df.loc[(
       (high < high.shift(1)) & (close < high - (tr * 2 / 3))
    ), 'feature_pinbar'] = -1

    df.loc[(
       (low > low.shift(1)) & (close > low + (tr * 2 / 3))
    ), 'feature_pinbar'] = 1

    df["feature_pinbar_ma30"] = close/sma(close, 30)
    df["feature_pinbar_ma15"] = close / sma(close, 15)
    df["feature_pinbar_ma7"] = close / sma(close, 7)
    return df

if __name__ == "__main__":
    from loaddata import test_data

    df = test_data()
    df = pinbar(df)
    print(df.tail())