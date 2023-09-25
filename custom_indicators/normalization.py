from pandas_ta.statistics import zscore


windows_size = 100

def zscore_ochlv(df):
    df['normal_close'] = zscore(df['close'], length=windows_size )
    df['normal_open'] = zscore(df['open'], length=windows_size )
    df['normal_high'] = zscore(df['high'], length=windows_size )
    df['normal_low'] = zscore(df['low'], length=windows_size )
    df['normal_volume'] = zscore(df['volume'], length=windows_size )
    return df


def highlow_ochlv(df):
    ll = df['low'].rolling(window=windows_size).min()
    hh = df['high'].rolling(window=windows_size).max()
    vll = df['volume'].rolling(window=windows_size).min()
    vhh = df['volume'].rolling(window=windows_size).max()
    scale = hh - ll
    vscale = vhh - vll
    df['normal_open'] = (df['open'] - ll)/scale
    df['normal_low'] = (df['low'] - ll) / scale
    df['normal_high'] = (df['high'] - ll) / scale
    df['normal_close'] = (df['close'] - ll) / scale
    df['normal_volume'] = (df['volume']-vll)/vscale
    return df