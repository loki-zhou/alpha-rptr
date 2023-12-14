from pandas import DataFrame
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from src.indicators import sma, ema, wma, double_ema, triple_ema
from technical import qtpylib

def support_resistance(df: DataFrame, periods=30):
    high = df['high']
    low = df['low']
    close = df['close']
    pl = low.rolling(window=periods * 2 + 1).min()
    ph = high.rolling(window=periods * 2 + 1).max()
    df["feature_support"] = close/pl
    df["feature_resistance"] = close/ph
    return df


def exhaustion_bars(dataframe, maj_qual=6, maj_len=12, min_qual=6, min_len=12, core_length=4):
    """
    Leledc Exhaustion Bars - Extended
    Infamous S/R Reversal Indicator

    leledc_major (Trend):
     1 Up
    -1 Down

    leledc_minor:
    1 Sellers exhausted
    0 Neutral / Hold
    -1 Buyers exhausted

    Original (MT4) https://www.abundancetradinggroup.com/leledc-exhaustion-bar-mt4-indicator/

    :return: DataFrame with columns populated
    """

    bindex_maj, sindex_maj, trend_maj = 0, 0, 0
    bindex_min, sindex_min = 0, 0
    leledc_major = np.full((len(dataframe),),np.nan)
    leledc_minor = np.full((len(dataframe),),np.nan)
    for i in range(len(dataframe)):
        close = dataframe['close'][i]

        if i < 1 or i - core_length < 0:
            leledc_major[i] = np.nan
            leledc_minor[i] = 0
            continue

        bindex_maj, sindex_maj = np.nan_to_num(bindex_maj), np.nan_to_num(sindex_maj)
        bindex_min, sindex_min = np.nan_to_num(bindex_min), np.nan_to_num(sindex_min)

        if close > dataframe['close'][i - core_length]:
            bindex_maj += 1
            bindex_min += 1
        elif close < dataframe['close'][i - core_length]:
            sindex_maj += 1
            sindex_min += 1

        update_major = False
        if bindex_maj > maj_qual and close < dataframe['open'][i] and dataframe['high'][i] >= dataframe['high'][
                                                                                              i - maj_len:i].max():
            bindex_maj, trend_maj, update_major = 0, 1, True
        elif sindex_maj > maj_qual and close > dataframe['open'][i] and dataframe['low'][i] <= dataframe['low'][
                                                                                               i - maj_len:i].min():
            sindex_maj, trend_maj, update_major = 0, -1, True

        leledc_major[i] = trend_maj if update_major else np.nan if trend_maj == 0 else trend_maj

        if bindex_min > min_qual and close < dataframe['open'][i] and dataframe['high'][i] >= dataframe['high'][
                                                                                              i - min_len:i].max():
            bindex_min = 0
            leledc_minor[i] = -1
        elif sindex_min > min_qual and close > dataframe['open'][i] and dataframe['low'][i] <= dataframe['low'][
                                                                                               i - min_len:i].min():
            sindex_min = 0
            leledc_minor[i] = 1
        else:
            leledc_minor[i] = 0
    dataframe['leledc_major'] = leledc_major
    dataframe['leledc_minor'] = leledc_minor
    return dataframe


def exhaustion_barsV2(dataframe, maj_qual=10, maj_len=40, core_length=4):
    """
    Leledc Exhaustion Bars - Extended
    Infamous S/R Reversal Indicator

    leledc_major (Trend):
     1 Up
    -1 Down

    leledc_minor:
    1 Sellers exhausted
    0 Neutral / Hold
    -1 Buyers exhausted

    Original (MT4) https://www.abundancetradinggroup.com/leledc-exhaustion-bar-mt4-indicator/

    :return: DataFrame with columns populated
    """

    bindex_maj, sindex_maj, trend_maj = 0, 0, 0
    bindex_min, sindex_min = 0, 0
    leledc_major = np.full((len(dataframe),),np.nan)
    leledc_minor = np.full((len(dataframe),),np.nan)
    for i in range(len(dataframe)):
        close = dataframe['close'][i]

        if i < 1 or i - core_length < 0:
            leledc_major[i] = np.nan
            leledc_minor[i] = 0
            continue

        bindex_maj, sindex_maj = np.nan_to_num(bindex_maj), np.nan_to_num(sindex_maj)


        if close > dataframe['close'][i - core_length]:
            bindex_maj += 1

        elif close < dataframe['close'][i - core_length]:
            sindex_maj += 1

        update_major = False
        if bindex_maj > maj_qual and close < dataframe['open'][i] and dataframe['high'][i] >= dataframe['high'][
                                                                                              i - maj_len:i].max():
            leledc_major[i] = -1
            bindex_maj = 0
        elif sindex_maj > maj_qual and close > dataframe['open'][i] and dataframe['low'][i] <= dataframe['low'][
                                                                                               i - maj_len:i].min():
            leledc_major[i] = 1
            sindex_maj = 0


    dataframe['leledc_major'] = leledc_major

    return dataframe


def dynamic_exhaustion_bars(dataframe, window=500):
    """
    Dynamic Leledc Exhaustion Bars -  By nilux
    The lookback length and exhaustion bars adjust dynamically to the market.

    leledc_major (Trend):
     1 Up
    -1 Down

    leledc_minor:
    1 Sellers exhausted
    0 Neutral / Hold
    -1 Buyers exhausted

    :return: DataFrame with columns populated
    """

    dataframe['close_pct_change'] = dataframe['close'].pct_change()
    dataframe['pct_change_zscore'] = qtpylib.zscore(dataframe, col='close_pct_change')
    dataframe['pct_change_zscore_smoothed'] = dataframe['pct_change_zscore'].rolling(window=3).mean()
    dataframe['pct_change_zscore_smoothed'].fillna(1.0, inplace=True)

    # To Do: Improve outlier detection

    zscore = dataframe['pct_change_zscore_smoothed'].to_numpy()
    zscore_multi = np.maximum(np.minimum(5.0 - zscore * 2, 5.0), 1.5)

    maj_qual, min_qual = calculate_exhaustion_candles(dataframe, window, zscore_multi)

    dataframe['maj_qual'] = maj_qual
    dataframe['min_qual'] = min_qual

    maj_len, min_len = calculate_exhaustion_lengths(dataframe)

    dataframe['maj_len'] = maj_len
    dataframe['min_len'] = min_len

    dataframe = populate_leledc_major_minor(dataframe, maj_qual, min_qual, maj_len, min_len)

    return dataframe


def populate_leledc_major_minor(dataframe, maj_qual, min_qual, maj_len, min_len):
    bindex_maj, sindex_maj, trend_maj = 0, 0, 0
    bindex_min, sindex_min = 0, 0

    # dataframe['leledc_major'] = np.nan
    # dataframe['leledc_minor'] = 0

    leledc_major = np.full((len(dataframe),),np.nan)
    leledc_minor = np.full((len(dataframe),),np.nan)

    for i in range(1, len(dataframe)):
        close = dataframe['close'][i]
        short_length = i if i < 4 else 4

        if close > dataframe['close'][i - short_length]:
            bindex_maj += 1
            bindex_min += 1
        elif close < dataframe['close'][i - short_length]:
            sindex_maj += 1
            sindex_min += 1

        update_major = False
        if bindex_maj > maj_qual[i] and close < dataframe['open'][i] and dataframe['high'][i] >= dataframe['high'][
                                                                                                 i - maj_len:i].max():
            bindex_maj, trend_maj, update_major = 0, 1, True
        elif sindex_maj > maj_qual[i] and close > dataframe['open'][i] and dataframe['low'][i] <= dataframe['low'][
                                                                                                  i - maj_len:i].min():
            sindex_maj, trend_maj, update_major = 0, -1, True

        leledc_major[i] = trend_maj if update_major else np.nan if trend_maj == 0 else trend_maj
        if bindex_min > min_qual[i] and close < dataframe['open'][i] and dataframe['high'][i] >= dataframe['high'][
                                                                                                 i - min_len:i].max():
            bindex_min = 0
            leledc_minor[i] = -1
        elif sindex_min > min_qual[i] and close > dataframe['open'][i] and dataframe['low'][i] <= dataframe['low'][
                                                                                                  i - min_len:i].min():
            sindex_min = 0
            leledc_minor[i] = 1
        else:
            leledc_minor[i] = 0
    dataframe['leledc_major'] = leledc_major
    dataframe['leledc_minor'] = leledc_minor
    return dataframe

def consecutive_count(consecutive_diff):
    return np.mean(np.abs(np.diff(np.where(consecutive_diff != 0))))
def calculate_exhaustion_candles(dataframe, window, multiplier):
    """
    Calculate the average consecutive length of ups and downs to adjust the exhaustion bands dynamically
    To Do: Apply ML (FreqAI) to make prediction
    """
    consecutive_diff = np.sign(dataframe['close'].diff())
    maj_qual = np.zeros(len(dataframe))
    min_qual = np.zeros(len(dataframe))

    for i in range(len(dataframe)):
        idx_range = consecutive_diff[i - window + 1:i + 1] if i >= window else consecutive_diff[:i + 1]
        avg_consecutive = consecutive_count(idx_range)
        if isinstance(avg_consecutive, np.ndarray):
            avg_consecutive = avg_consecutive.item()
        maj_qual[i] = int(avg_consecutive * (3 * multiplier[i])) if not np.isnan(avg_consecutive) else 0
        min_qual[i] = int(avg_consecutive * (3 * multiplier[i])) if not np.isnan(avg_consecutive) else 0

    return maj_qual, min_qual


def calculate_exhaustion_lengths(dataframe):
    """
    Calculate the average length of peaks and valleys to adjust the exhaustion bands dynamically
    To Do: Apply ML (FreqAI) to make prediction
    """
    high_indices = argrelextrema(dataframe['high'].to_numpy(), np.greater)
    low_indices = argrelextrema(dataframe['low'].to_numpy(), np.less)

    avg_peak_distance = np.mean(np.diff(high_indices))
    std_peak_distance = np.std(np.diff(high_indices))
    avg_valley_distance = np.mean(np.diff(low_indices))
    std_valley_distance = np.std(np.diff(low_indices))

    maj_len = int(avg_peak_distance + std_peak_distance)
    min_len = int(avg_valley_distance + std_valley_distance)

    return maj_len, min_len


if __name__ == "__main__":
    from loaddata import test_data

    df = test_data()
    # df = support_resistance(df)
    df = dynamic_exhaustion_bars(df)
    print(df.tail())