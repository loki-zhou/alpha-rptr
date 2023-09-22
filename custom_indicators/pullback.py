from pandas import DataFrame
import pandas as pd
import numpy as np

def numpy_rolling_window(data, window):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def numpy_rolling_series(func):
    def func_wrapper(data, window, as_source=False):
        series = data.values if isinstance(data, pd.Series) else data

        new_series = np.empty(len(series)) * np.nan
        calculated = func(series, window)
        new_series[-len(calculated):] = calculated

        if as_source and isinstance(data, pd.Series):
            return pd.Series(index=data.index, data=new_series)

        return new_series

    return func_wrapper

@numpy_rolling_series
def numpy_rolling_mean(data, window, as_source=False):
    return np.mean(numpy_rolling_window(data, window), axis=-1)


@numpy_rolling_series
def numpy_rolling_std(data, window, as_source=False):
    return np.std(numpy_rolling_window(data, window), axis=-1, ddof=1)

def zscore(bars, window=20, stds=1, col='close'):
    """ get zscore of price """
    std = numpy_rolling_std(bars[col], window)
    mean = numpy_rolling_mean(bars[col], window)
    return (bars[col] - mean) / (std * stds)

def detect_pullback(df: DataFrame, periods=30, method='pct_outlier'):
    """
    Pullback & Outlier Detection
    Know when a sudden move and possible reversal is coming

    Method 1: StDev Outlier (z-score)
    Method 2: Percent-Change Outlier (z-score)
    Method 3: Candle Open-Close %-Change

    outlier_threshold - Recommended: 2.0 - 3.0

    df['pullback_flag']: 1 (Outlier Up) / -1 (Outlier Down)
    """


    pb_dif = df['close'] - df['close'].shift(1)
    pb_dif_squared_sum = (pb_dif ** 2).rolling(window=periods + 1).sum()
    pb_std = np.sqrt((pb_dif_squared_sum- pb_dif.shift(0) ** 2) / (periods - 1))
    df['feature_pb_stdev_outlier'] = pb_dif / pb_std
    df["feature_pb_pct_change"] = df["close"].pct_change()
    df['feature_pb_pct_outlier'] = zscore(df, window=periods, col='feature_pb_pct_change')
    df['feature_pb_candle_body'] = (df['close'] - df['open']) / (df['high']-df['low'])

    return df


if __name__ == "__main__":
    from loaddata import test_data

    df = test_data()
    df = detect_pullback(df)
    print(df.tail())