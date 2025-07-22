#!/usr/bin/env python3
import pandas as pd
import numpy as np
from strategies import trend, sideways, downtrend

# ── Paths to your CSVs ───────────────────────────────────────
DATA_1M  = "data/BTCUSDT_1m_5year.csv"
DATA_5M  = "data/BTCUSDT_5m_5year.csv"
DATA_1H  = "data/BTCUSDT_1h_5year.csv"
DATA_15M = "data/BTCUSDT_15m_5year.csv"  

# ── How many days back from the end to include ──────────────
LIMITER_DAYS   = 30  # e.g. last 90 calendar days of data

# ── Regime‐classification constants ─────────────────────────
EMA_SHORT      = 50
ATR_WINDOW     = 14
ADX_WINDOW     = 14
ADX_THRESHOLD  = 20

def load_and_normalize(path):
    df = pd.read_csv(path)
    if 'open_time' in df:
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    elif 'timestamp' in df:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)

def limit_to_last_n_days(df, n):
    days = df['datetime'].dt.normalize().drop_duplicates().sort_values()
    last_n = days.iloc[-n:]
    return df[df['datetime'].dt.normalize().isin(last_n)].reset_index(drop=True)

def split_by_days(df5m):
    days = df5m['datetime'].dt.normalize().drop_duplicates().sort_values()
    cut  = int(round(len(days) * 0.7))
    return set(days.iloc[:cut]), set(days.iloc[cut:])

def classify_regimes(df5m):
    df = df5m.copy()
    df['prev_close'] = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['prev_close']).abs(),
        (df['low']  - df['prev_close']).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_WINDOW).mean()
    m_atr     = df['atr'].mean()
    df['ema50'] = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()

    upm   = df['high'].diff()
    downm = -(df['low'].shift(1).diff())
    plus  = np.where((upm>downm)&(upm>0), upm, 0.0)
    minus = np.where((downm>upm)&(downm>0), downm, 0.0)
    p_ewm  = pd.Series(plus).ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    m_ewm  = pd.Series(minus).ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    df['adx'] = 100 * (p_ewm - m_ewm).abs() / (p_ewm + m_ewm)

    is_trend = (df['close'] > df['ema50']) & (df['adx'] > ADX_THRESHOLD) & (df['atr'] >= m_atr)
    is_down  = (df['close'] < df['ema50']) & (df['adx'] > ADX_THRESHOLD) & (df['atr'] >= m_atr)
    df['regime'] = np.where(is_trend, 'trend',
                     np.where(is_down,  'downtrend', 'sideways'))
    return df[['datetime','regime']]

def detect_segments(df_regime):
    segs = []
    prev = df_regime.iloc[0]
    start, last = prev['datetime'], prev['regime']
    for _, row in df_regime.iloc[1:].iterrows():
        if row['regime'] != last:
            segs.append((start, prev['datetime'], last))
            start, last = row['datetime'], row['regime']
        prev = row
    segs.append((start, prev['datetime'], last))
    return pd.DataFrame(segs, columns=['start','end','regime'])

def main():
    # 1) load all frames
    df1m  = load_and_normalize(DATA_1M)
    df5m  = load_and_normalize(DATA_5M)
    df1h  = load_and_normalize(DATA_1H)
    df15m = load_and_normalize(DATA_15M)

    # 2) limit each to last LIMITER_DAYS
    df1m  = limit_to_last_n_days(df1m, LIMITER_DAYS)
    df5m  = limit_to_last_n_days(df5m, LIMITER_DAYS)
    df1h  = limit_to_last_n_days(df1h, LIMITER_DAYS)
    df15m = limit_to_last_n_days(df15m, LIMITER_DAYS)

    # 3) split train/test by day (still using 5m to define days)
    train_days, test_days = split_by_days(df5m)
    slice_by = lambda df, days: df[df['datetime'].dt.normalize().isin(days)].reset_index(drop=True)

    df5m_tr,  df5m_ts  = slice_by(df5m,  train_days), slice_by(df5m,  test_days)
    df1m_tr,  df1m_ts  = slice_by(df1m,  train_days), slice_by(df1m,  test_days)
    df1h_tr,  df1h_ts  = slice_by(df1h,  train_days), slice_by(df1h,  test_days)
    df15m_tr, df15m_ts = slice_by(df15m, train_days), slice_by(df15m, test_days)

    # 4) regime‐switch detection
    seg_tr = detect_segments(classify_regimes(df5m_tr))
    seg_tr.to_csv('backtest_switching.csv', index=False)
    seg_ts = detect_segments(classify_regimes(df5m_ts))
    seg_ts.to_csv('testrun_switching.csv', index=False)

    # 5) dispatch to Trend
    print("=== RUNNING TREND MODULE ===")
    trend.run_strategy(
        df_5m_train=df5m_tr, df_1h_train=df1h_tr,
         df_5m_test =df5m_ts,  df_1h_test =df1h_ts
    )

    # 6) dispatch to Sideways (now on 15 m + 1 h)
    # 6) dispatch to Sideways (15 m ↔ 1 h micro‐mean‐reversion)
    print("=== RUNNING SIDEWAYS (15m↔1h MICRO‐MR) MODULE ===")
    sideways.run_strategy(
        df15m_tr,   # 15 m train
        df15m_ts,   # 15 m test
        df1h_tr,    # 1 h trainåç
        df1h_ts     # 1 h testå
    )


    # 7) dispatch to Downtrend
    print("=== RUNNING DOWNTREND MODULE ===")
    downtrend.run_strategy(
        df_1m_train=df1m_tr, df_5m_train=df5m_tr, df_1h_train=df1h_tr,
        df_1m_test =df1m_ts,  df_5m_test =df5m_ts,  df_1h_test =df1h_ts
    )

if __name__ == "__main__":
    main()
