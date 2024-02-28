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

from alphas import feature_generation, utils
def create_alphas1(df):
    df['returns'] = utils.returns(df)
    #df['vwap'] = utils.vwap(df)
    df['feature_alpha1'] = feature_generation.alpha1(df)
    df['feature_alpha2'] = feature_generation.alpha2(df)
    df['feature_alpha3'] = feature_generation.alpha3(df)
    df['feature_alpha4'] = feature_generation.alpha4(df)
    df['feature_alpha6'] = feature_generation.alpha6(df)
    df['feature_alpha9'] = feature_generation.alpha9(df)
    df['feature_alpha12'] = feature_generation.alpha12(df)
    df['feature_alpha14'] = feature_generation.alpha14(df)
    df['feature_alpha15'] = feature_generation.alpha15(df)
    df['feature_alpha16'] = feature_generation.alpha16(df)
    df['feature_alpha17'] = feature_generation.alpha17(df)

    df['feature_alpha21'] = feature_generation.alpha21(df)
    df['feature_alpha22'] = feature_generation.alpha22(df)
    df['feature_alpha23'] = feature_generation.alpha23(df)
    df['feature_alpha24'] = feature_generation.alpha24(df)
    df['feature_alpha26'] = feature_generation.alpha26(df)





    return df

def create_alphas(df):
    df['returns'] = utils.returns(df)
    #df['vwap'] = utils.vwap(df)
    df['feature_alpha1'] = feature_generation.alpha1(df)
    df['feature_alpha2'] = feature_generation.alpha2(df)
    df['feature_alpha3'] = feature_generation.alpha3(df)
    df['feature_alpha4'] = feature_generation.alpha4(df)
    df['feature_alpha6'] = feature_generation.alpha6(df)
    df['feature_alpha9'] = feature_generation.alpha9(df)
    df['feature_alpha12'] = feature_generation.alpha12(df)
    df['feature_alpha14'] = feature_generation.alpha14(df)
    df['feature_alpha15'] = feature_generation.alpha15(df)
    df['feature_alpha16'] = feature_generation.alpha16(df)
    df['feature_alpha17'] = feature_generation.alpha17(df)

    df['feature_alpha21'] = feature_generation.alpha21(df)
    df['feature_alpha22'] = feature_generation.alpha22(df)
    df['feature_alpha23'] = feature_generation.alpha23(df)
    df['feature_alpha24'] = feature_generation.alpha24(df)
    df['feature_alpha26'] = feature_generation.alpha26(df)

    df['feature_alpha31'] = feature_generation.alpha31(df)
    df['feature_alpha34'] = feature_generation.alpha34(df)
    df['feature_alpha35'] = feature_generation.alpha35(df)

    df['feature_alpha37'] = feature_generation.alpha37(df)
    df['feature_alpha38'] = feature_generation.alpha38(df)

    df['feature_alpha39'] = feature_generation.alpha39(df)
    df['feature_alpha40'] = feature_generation.alpha40(df)
    df['feature_alpha43'] = feature_generation.alpha43(df)
    # ### df['feature_alpha45'] = feature_generation.alpha45(df)
    df['feature_alpha46'] = feature_generation.alpha46(df)
    df['feature_alpha49'] = feature_generation.alpha49(df)

    # df['feature_alpha52'] = feature_generation.alpha52(df)
    # df['feature_alpha53'] = feature_generation.alpha53(df)
    df['feature_alpha54'] = feature_generation.alpha54(df)
    # df['feature_alpha55'] = feature_generation.alpha55(df)
    # df['feature_alpha60'] = feature_generation.alpha60(df)



    return df

if __name__ == '__main__':
    import pandas as pd

    # df = pd.read_csv(r"D:\rl\alpha-rptr\ohlc\binance_futures\BTCUSDT\['15m']\data.csv", parse_dates=["time"],
    #                  index_col="time")
    df = pd.read_csv(r"D:\rl\alpha-rptr\model\data\BTC_USD-Hourly.csv", parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df = create_alphas(df)
    df.dropna(inplace=True)