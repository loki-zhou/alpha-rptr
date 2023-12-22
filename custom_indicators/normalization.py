import numpy as np
from pandas_ta.statistics import zscore




def zscore_ochlv(df, windows_size = 100):
    df['normal_close'] = zscore(df['close'], length=windows_size )
    df['normal_open'] = zscore(df['open'], length=windows_size )
    df['normal_high'] = zscore(df['high'], length=windows_size )
    df['normal_low'] = zscore(df['low'], length=windows_size )
    df['normal_volume'] = zscore(df['volume'], length=windows_size )
    return df


def highlow_winodws_ochlv(df, windows_size = 100):
    ll = df['low'].rolling(window=windows_size).min()
    hh = df['high'].rolling(window=windows_size).max()
    vll = df['volume'].rolling(window=windows_size).min()
    vhh = df['volume'].rolling(window=windows_size).max()
    scale = hh - ll
    vscale = vhh - vll

    df['feature_highlow_open'] = (df['open'] - ll)/scale
    df['feature_highlow_low'] = (df['low'] - ll) / scale
    df['feature_highlow_high'] = (df['high'] - ll) / scale
    df['feature_highlow_close'] = (df['close'] - ll) / scale
    df['feature_highlow_volume'] = (df['volume']-vll)/vscale
    columns_to_round = {'feature_highlow_open': 2, 'feature_highlow_low': 2, "feature_highlow_high": 2, "feature_highlow_close": 2, "feature_highlow_volume":2}
    return df.round(columns_to_round)

def logged_diff(df):
    for column in ["open","close", "high", "low", "volume"]:
        ld =  np.log(df[column]) - np.log(df[column].shift(1))
        cum_max = ld.cummax()
        cum_min = ld.cummin()
        df["feature_logged_diff_" + column] = (ld - cum_min) / (cum_max - cum_min)
    return df

def normal_deal(df, windows_size = 50):

    df['feature_open'] = df['open']/100000
    df['feature_low'] = df['low']/100000
    df['feature_high'] = df['high']/100000
    df['feature_close'] =  df['close']/100000
    df['feature_volume'] = df['volume']/100000

    return df