import pandas as pd


def test_data():
    df = pd.read_csv(r"D:\rl\alpha-rptr\ohlc\binance_futures\BTCUSDT\['15m']\data.csv", parse_dates=["time"], index_col="time")
    df=df.tz_localize(None)
    df = df.rename(columns={"time": "date"})
    return df