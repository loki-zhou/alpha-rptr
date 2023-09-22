from pandas import DataFrame
import pandas as pd
import numpy as np

from src.indicators import sma, ema, wma, double_ema, triple_ema

def smi_trend(df: DataFrame, k_length=9, d_length=3, smoothing_type='EMA', smoothing=10):
    """
    Stochastic Momentum Index (SMI) Trend Indicator

    SMI > 0 and SMI > MA: (2) Bull
    SMI < 0 and SMI > MA: (1) Possible Bullish Reversal

    SMI > 0 and SMI < MA: (-1) Possible Bearish Reversal
    SMI < 0 and SMI < MA: (-2) Bear

    Returns:
        pandas.Series: New feature generated
    """

    ll = df['low'].rolling(window=k_length).min()
    hh = df['high'].rolling(window=k_length).max()

    diff = hh - ll
    rdiff = df['close'] - (hh + ll) / 2

    avgrel = rdiff.ewm(span=d_length).mean().ewm(span=d_length).mean()
    avgdiff = diff.ewm(span=d_length).mean().ewm(span=d_length).mean()

    smi = np.where(avgdiff != 0, (avgrel / (avgdiff / 2) * 100), 0)

    if smoothing_type == 'SMA':
        smi_ma = sma(smi, smoothing)
    elif smoothing_type == 'EMA':
        smi_ma = ema(smi, smoothing)
    elif smoothing_type == 'WMA':
        smi_ma = wma(smi, smoothing)
    elif smoothing_type == 'DEMA':
        smi_ma = double_ema(smi, smoothing)
    elif smoothing_type == 'TEMA':
        smi_ma = triple_ema(smi, smoothing)
    else:
        raise ValueError("Choose an MA Type: 'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA'")

    df["feature_smi"] = smi/smi_ma

    return df


if __name__ == "__main__":
    from loaddata import test_data

    df = test_data()
    df = smi_trend(df)
    print(df.tail())