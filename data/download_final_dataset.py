import io
import zipfile

import requests
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/monthly"
FINAL = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XMRUSDT", "DASHUSDT"]
START = pd.Timestamp("2020-02-17 08:00", tz="UTC")
END = pd.Timestamp("2026-06-29 23:00", tz="UTC")
OUT = "data/final_dataset"

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def months(start, end):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m = m + 1 if m < 12 else 1
        y = y if m != 1 else y + 1


def read_zip(url):
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return pd.read_csv(z.open(z.namelist()[0]), header=None)


def to_utc(series):
    s = pd.to_numeric(series)
    unit = "us" if s.max() > 1e14 else "ms"
    return pd.to_datetime(s, unit=unit, utc=True)


def klines(sym):
    frames = []
    for y, m in months(START, END):
        df = read_zip(f"{BASE}/klines/{sym}/1h/{sym}-1h-{y}-{m:02d}.zip")
        if df is None:  # month not archived yet -> fetch its daily files instead
            for d in pd.date_range(f"{y}-{m:02d}-01", periods=31, freq="D"):
                if (d.year, d.month) != (y, m):
                    break
                dd = read_zip(f"{BASE.replace('monthly', 'daily')}/klines/{sym}/1h/{sym}-1h-{d:%Y-%m-%d}.zip")
                if dd is not None:
                    frames.append(dd)
        else:
            frames.append(df)
    df = pd.concat(frames)
    df = df[df.iloc[:, 0] != "open_time"]  # drop header rows present in newer files
    df.columns = KLINE_COLS
    df.index = to_utc(df["open_time"])
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df[~df.index.duplicated()].sort_index()


def funding(sym):
    frames = []
    for y, m in months(START, END):
        df = read_zip(f"{BASE}/fundingRate/{sym}/{sym}-fundingRate-{y}-{m:02d}.zip")
        if df is not None:
            frames.append(df)
    df = pd.concat(frames)
    df = df[df.iloc[:, 0] != "calc_time"]
    idx = to_utc(df.iloc[:, 0]).dt.floor("h")
    rate = pd.Series(pd.to_numeric(df.iloc[:, 2]).values, index=idx)
    return rate.groupby(level=0).last()


for sym in FINAL:
    px = klines(sym).loc[START:END]
    px["funding_rate"] = funding(sym).reindex(px.index).fillna(0.0)
    px.insert(0, "open_time_iso", px.index.strftime("%Y-%m-%dT%H:%M:%S+00:00"))
    px.to_csv(f"{OUT}/{sym}.csv", index=False)
    print(sym, len(px), px["open_time_iso"].iloc[0], "->", px["open_time_iso"].iloc[-1])
