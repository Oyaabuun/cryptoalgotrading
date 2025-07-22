#!/usr/bin/env python3
import math
import pandas as pd
import numpy as np
from itertools import product

# ── Static params ─────────────────────────────────────────
ATR_WINDOW    = 14
EMA_KC_15     = 20    # Keltner mid on 15 m
EMA_KC_1H     = 50    # Keltner mid on 1 h
LOOKBACK      = 3     # bars to wait for mid‐line confirm
INITIAL_USDT  = 58.0
LEVERAGE      = 10.0
CONTRACT_SZ   = 0.001   # enforce Binance minQty & stepSize = 0.001
TAKER_FEE     = 0.0005
SLIPPAGE      = 0.0005

# ── Hyper‐grid ────────────────────────────────────────────
param_grid = {
    'KC_ATR_MULT_15': [1.0, 1.5, 2.0],
    'KC_ATR_MULT_1H': [1.0, 1.5],
    'RISK_PCT':       [0.005, 0.01],
}

# ── Resample 5 m → 15 m ───────────────────────────────────
def make_15m(df5):
    df = df5.set_index('datetime')
    df15 = df.resample('15T').agg({
        'open':'first',
        'high':'max',
        'low':'min',
        'close':'last',
        'volume':'sum'
    }).dropna().reset_index()
    return df15

# ── Build 1 h Keltner mid & ATR ───────────────────────────
def prep_1h(df1h):
    df = df1h.sort_values('datetime').reset_index(drop=True)
    prev = df.close.shift(1)
    tr = pd.concat([
        df.high - df.low,
        (df.high - prev).abs(),
        (df.low  - prev).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_WINDOW).mean()
    mid = df.close.ewm(span=EMA_KC_1H, adjust=False).mean()
    df['kc_mid_1h'] = mid
    df['atr_1h']    = atr
    df['close_1h']  = df.close
    return df[['datetime','kc_mid_1h','atr_1h','close_1h']]

# ── Build 15 m Keltner & ATR ──────────────────────────────
def prep_15m(df15, p):
    df = df15.sort_values('datetime').reset_index(drop=True)
    prev = df.close.shift(1)
    tr   = pd.concat([
        df.high - df.low,
        (df.high - prev).abs(),
        (df.low  - prev).abs()
    ], axis=1).max(axis=1)
    atr  = tr.rolling(ATR_WINDOW).mean()
    mid  = df.close.ewm(span=EMA_KC_15, adjust=False).mean()

    df['kc_mid_15']   = mid
    df['kc_upper_15'] = mid + p['KC_ATR_MULT_15'] * atr
    df['kc_lower_15'] = mid - p['KC_ATR_MULT_15'] * atr
    df['atr_15']      = atr
    return df.dropna().reset_index(drop=True)

# ── 15 m mid‐line confirmation ────────────────────────────
def confirm_15(r, df15):
    idx = int(r.name)
    mid = df15.at[idx,'kc_mid_15']
    look = df15.close.iloc[idx+1 : idx+1+LOOKBACK]
    return (look > mid).any() if r.signal=='CALL' else (look < mid).any()

# ── Find entries on 15 m, filtered by 1 h midline ─────────
def find_entries(df15, df1h_f, p):
    df = prep_15m(df15, p)

    # raw fade signals
    df['signal'] = np.where(df.close < df.kc_lower_15, 'CALL',
                     np.where(df.close > df.kc_upper_15, 'PUT', None))
    df = df[df.signal.notnull()]

    # merge in 1 h mid/close
    df = df.merge(df1h_f, on='datetime', how='left').ffill()

    # require 1 h close to have touched its midline already
    df = df[df.apply(lambda r:
        (r.signal=='CALL' and r.close_1h > r.kc_mid_1h) or
        (r.signal=='PUT'  and r.close_1h < r.kc_mid_1h),
        axis=1)]

    # 15 m mid‐line confirmation
    df = df[df.apply(lambda r: confirm_15(r, df), axis=1)]

    df['idx15'] = df.index
    return df.reset_index(drop=True)

# ── Backtest (SL=1×ATR, TP=2×ATR) ─────────────────────────
def backtest(df15, df1h_f, p):
    equity, banked, trades = INITIAL_USDT, 0.0, []
    entries = find_entries(df15, df1h_f, p)

    for _, r in entries.iterrows():
        sig, i15 = r.signal, int(r.idx15)
        epx = df15.at[i15,'open']
        atr = df15.at[i15,'atr_15']
        sl  = epx - atr if sig=='CALL' else epx + atr
        tp  = epx + 2*atr if sig=='CALL' else epx - 2*atr

        # position sizing in 0.001 steps
        risk   = equity * p['RISK_PCT'] * LEVERAGE
        dist   = abs(epx - sl)
        raw    = risk / (dist * CONTRACT_SZ) if dist>0 else 0
        cnt    = math.floor(raw) * CONTRACT_SZ
        if cnt < CONTRACT_SZ:
            continue

        # walk forward
        exit_px = None
        for j in range(i15+1, min(i15+1+LOOKBACK, len(df15))):
            h, l = df15.at[j,'high'], df15.at[j,'low']
            if   sig=='CALL' and l <= sl:
                exit_px, exit_dt = sl, df15.at[j,'datetime']; break
            elif sig=='CALL' and h >= tp:
                exit_px, exit_dt = tp, df15.at[j,'datetime']; break
            elif sig=='PUT'  and h >= sl:
                exit_px, exit_dt = sl, df15.at[j,'datetime']; break
            elif sig=='PUT'  and l <= tp:
                exit_px, exit_dt = tp, df15.at[j,'datetime']; break

        if exit_px is None:
            ei = min(i15+LOOKBACK, len(df15)-1)
            exit_px, exit_dt = df15.at[ei,'close'], df15.at[ei,'datetime']

        ie  = epx*(1+SLIPPAGE) if sig=='CALL' else epx*(1-SLIPPAGE)
        ex  = exit_px*(1-SLIPPAGE) if sig=='CALL' else exit_px*(1+SLIPPAGE)
        pnl = (ex-ie if sig=='CALL' else ie-ex) * cnt
        fee = (ie+ex) * cnt * CONTRACT_SZ * TAKER_FEE
        pnl -= fee

        if pnl > 0:
            banked += pnl * 0.7
            equity += pnl * 0.3
        else:
            equity += pnl

        trades.append({
            'entry_dt':  df15.at[i15,'datetime'],
            'exit_dt':   exit_dt,
            'signal':    sig,
            'contracts': cnt,
            'entry_px':  epx,
            'exit_px':   exit_px,
            'pnl_usd':   pnl,
            'equity':    equity,
            'banked':    banked
        })

    df_tr = pd.DataFrame(trades)
    if df_tr.empty:
        # zeroed‐out metrics
        return df_tr, {k: 0.0 for k in [
            'total_return_pct','annualized_return_pct','sharpe_ratio',
            'max_drawdown_pct','win_rate_pct','num_trades','profit_factor',
            'avg_win_usd','avg_loss_usd','max_win_usd','max_loss_usd',
            'avg_duration_min','final_balance_usdt'
        ]}

    # compute performance metrics
    df_tr['total'] = df_tr.equity + df_tr.banked
    total_ret = df_tr.total.iloc[-1] / INITIAL_USDT - 1
    days      = (df_tr.exit_dt.iloc[-1] - df_tr.entry_dt.iloc[0]).total_seconds()/86400
    cagr      = (1+total_ret)**(365/days)-1 if days>0 else np.nan
    rets      = df_tr.total.pct_change().fillna(0)
    sharpe    = rets.mean()/rets.std() if rets.std() else np.nan
    dd        = (df_tr.total - df_tr.total.cummax()) / df_tr.total.cummax()
    maxdd     = dd.min()
    wins      = df_tr.pnl_usd > 0
    pf        = (df_tr.loc[wins,'pnl_usd'].sum() /
                 -df_tr.loc[~wins,'pnl_usd'].sum()) if (~wins).any() else np.nan
    dur       = (df_tr.exit_dt - df_tr.entry_dt).dt.total_seconds()/60

    metrics = {
      'total_return_pct':      total_ret*100,
      'annualized_return_pct': cagr*100,
      'sharpe_ratio':          sharpe,
      'max_drawdown_pct':      maxdd*100,
      'win_rate_pct':          wins.mean()*100,
      'num_trades':            len(df_tr),
      'profit_factor':         pf,
      'avg_win_usd':           df_tr.loc[wins,'pnl_usd'].mean(),
      'avg_loss_usd':          df_tr.loc[~wins,'pnl_usd'].mean(),
      'max_win_usd':           df_tr.pnl_usd.max(),
      'max_loss_usd':          df_tr.pnl_usd.min(),
      'avg_duration_min':      dur.mean(),
      'final_balance_usdt':    df_tr.total.iloc[-1]
    }
    return df_tr, metrics

# ── Top‐level runner ───────────────────────────────────────
def run_strategy(df5_tr, df5_ts, df1h_tr, df1h_ts):
    df1h_tr_f = prep_1h(df1h_tr)
    df1h_ts_f = prep_1h(df1h_ts)

    df15_tr = make_15m(df5_tr)
    df15_ts = make_15m(df5_ts)

    # 1) grid‐search on train
    grid = []
    for p in [dict(zip(param_grid, v)) for v in product(*param_grid.values())]:
        _, m = backtest(prep_15m(df15_tr, p), df1h_tr_f, p)
        grid.append({**p, **m})
    pd.DataFrame(grid).to_csv('mr15_best_params.csv', index=False)

    # 2) pick best
    best  = pd.DataFrame(grid).loc[pd.DataFrame(grid)['sharpe_ratio'].idxmax()]
    best_p = {k: float(best[k]) for k in param_grid}

    # 3) backtest on train
    bt_tr, bt_m = backtest(prep_15m(df15_tr, best_p), df1h_tr_f, best_p)
    bt_tr.to_csv('mr15_backtest.csv', index=False)
    pd.DataFrame([bt_m]).to_csv('mr15_backtest_metrics.csv', index=False)

    # 4) testrun on test
    ts_tr, ts_m = backtest(prep_15m(df15_ts, best_p), df1h_ts_f, best_p)
    ts_tr.to_csv('mr15_testrun.csv', index=False)
    pd.DataFrame([ts_m]).to_csv('mr15_testrun_metrics.csv', index=False)
