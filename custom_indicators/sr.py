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
    dataframe["leledc_major"] = 1
    dataframe["leledc_minor"] = 1
    for i in range(len(dataframe)):
        close = dataframe['close'][i]

        if i < 1 or i - core_length < 0:
            dataframe.iloc[i]['leledc_major'] = np.nan
            dataframe.iloc[i]['leledc_minor'] = 0
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

        dataframe.loc[i, 'leledc_major'] = trend_maj if update_major else np.nan if trend_maj == 0 else trend_maj

        if bindex_min > min_qual and close < dataframe['open'][i] and dataframe['high'][i] >= dataframe['high'][
                                                                                              i - min_len:i].max():
            bindex_min = 0
            dataframe.loc[i, 'leledc_minor'] = -1
        elif sindex_min > min_qual and close > dataframe['open'][i] and dataframe['low'][i] <= dataframe['low'][
                                                                                               i - min_len:i].min():
            sindex_min = 0
            dataframe.loc[i, 'leledc_minor'] = 1
        else:
            dataframe.loc[i, 'leledc_minor'] = 0

    return dataframe

if __name__ == "__main__":
    from loaddata import test_data

    df = test_data()
    print(df["close"][1])
    # df = support_resistance(df)
    df = exhaustion_bars(df)
    print(df.tail())