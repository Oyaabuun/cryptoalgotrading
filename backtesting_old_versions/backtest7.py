#!/usr/bin/env python3
"""
Advanced breakout strategy (relaxed VWAP/ATR regime) with:
1) 1m confirmation
2) VWAP band confluence (±0.1%)
3) Volume spike filter
4) Session filter (00:00–00:59 UTC only)
5) Dynamic stops & partial scale-out
6) Hyperparameter grid search (now includes REGIME_MULT)
"""
import math
import pandas as pd
import numpy as np
from itertools import product

# ── Static strategy parameters ─────────────────────────────
LOOKBACK       = 20
ADX_WINDOW     = 14
ADX_THRESHOLD  = 20
INITIAL_USDT   = 58.0
LEVERAGE       = 10.0
CONTRACT_SZ    = 1.0
TAKER_FEE      = 0.0005
SLIPPAGE       = 0.0005

# narrowed VWAP band to ±0.1%
VWAP_BAND_PCT  = 0.001       
# only exclude the midnight hour
EXCLUDE_HOURS  = {0}           

DATA_1M  = "data/BTC_PERPETUAL_1m_1months.csv"
DATA_5M  = "data/BTC_PERPETUAL_5m_1months.csv"
DATA_1H  = "data/BTC_PERPETUAL_60m_1months.csv"

# ── Hyperparameter grid to test ────────────────────────────
param_grid = {
    'ATR_WINDOW':   [14, 20],
    'EMA_LONG':     [50, 100],
    'SL_ATR_MULT':  [1.0, 1.25],
    'TP_ATR_MULT':  [4.0, 5.0],
    'VOL_MULT':     [1.2, 1.5, 2.0],
    'RISK_PCT':     [0.005, 0.008, 0.01, 0.02],
    'REGIME_MULT':  [0.8, 1.0, 1.2],
}

# ── Helpers ─────────────────────────────────────────────────
def prepare_df(path):
    df = pd.read_csv(path)
    if 'open_time' in df:
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    elif 'timestamp' in df:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)

def split_data(df):
    days = df['datetime'].dt.normalize().drop_duplicates().sort_values()
    cut  = int(round(len(days)*0.7))
    back = set(days.iloc[:cut])
    test = set(days.iloc[cut:])
    return (df[df['datetime'].dt.normalize().isin(back)].reset_index(drop=True),
            df[df['datetime'].dt.normalize().isin(test)].reset_index(drop=True))

# ── Preprocessing ───────────────────────────────────────────
def preprocess_1h(df_raw, params):
    df = df_raw.copy()
    df['ema_hf'] = df['close'].ewm(span=int(params['EMA_LONG']), adjust=False).mean()
    return df[['datetime','ema_hf']]

def preprocess_5m(df_raw, params, df_hf):
    df = df_raw.copy()
    atr_win = int(params['ATR_WINDOW'])
    ema_l   = int(params['EMA_LONG'])

    # True Range & ATR
    df['prev_close'] = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['prev_close']).abs(),
        (df['low']  - df['prev_close']).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(atr_win).mean()

    # Slow ATR for regime filtering
    df['atr_slow']  = df['atr'].rolling(atr_win*2).mean()
    df['atr_ratio'] = df['atr'] / df['atr_slow']

    # RSI
    delta = df['close'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi'] = 100 - 100/(1 + up.rolling(atr_win).mean()/down.rolling(atr_win).mean())

    # EMA & breakout levels
    df['ema_long'] = df['close'].ewm(span=ema_l, adjust=False).mean()
    df['highest']  = df['high'].rolling(LOOKBACK).max().shift(1)
    df['lowest']   = df['low'].rolling(LOOKBACK).min().shift(1)

    # ADX
    up_m   = df['high'].diff()
    down_m = -(df['low'].shift(1).diff())
    plus   = np.where((up_m>down_m)&(up_m>0), up_m, 0.0)
    minus  = np.where((down_m>up_m)&(down_m>0), down_m, 0.0)
    sm_tr  = tr.ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    sm_p   = pd.Series(plus).ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    sm_m   = pd.Series(minus).ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    df['adx'] = 100 * (sm_p - sm_m).abs()/(sm_p + sm_m)

    # Volume MA
    df['vol_ma'] = df['volume'].rolling(20).mean()

    # VWAP
    df['date']     = df['datetime'].dt.date
    typ            = (df['high'] + df['low'] + df['close']) / 3
    df['cum_vp']   = typ.mul(df['volume']).groupby(df['date']).cumsum()
    df['cum_vol']  = df['volume'].groupby(df['date']).cumsum()
    df['vwap']     = df['cum_vp'] / df['cum_vol']
    df['vwap_upper'] = df['vwap'] * (1 + VWAP_BAND_PCT)
    df['vwap_lower'] = df['vwap'] * (1 - VWAP_BAND_PCT)

    # Merge 1h trend
    df['hour'] = df['datetime'].dt.floor('h')
    df = df.merge(df_hf.rename(columns={'datetime':'hour'}), on='hour', how='left')
    return df.dropna().reset_index(drop=True)

# ── Optional 1m confirmation ─────────────────────────────────
def confirm_1m(entry, df1m):
    start = entry['datetime']
    end   = start + pd.Timedelta(minutes=5)
    blk   = df1m[(df1m['datetime']>=start)&(df1m['datetime']<end)]
    level = (entry['highest']+0.5*entry['atr']
             if entry['signal']=='CALL'
             else entry['lowest']-0.5*entry['atr'])
    return ((blk['close']>level).any() if entry['signal']=='CALL'
            else (blk['close']<level).any())

# ── Entry detection with regime filter ─────────────────────
def find_entries(df5, df1m, params):
    df = df5.copy()
    total = len(df)
    print(f"\n=== PARAMS {params} ===")
    print(f"Total bars: {total}")

    # Filters:
    ema_f     = df['close'] > df['ema_long']
    th        = 0.5 * df['atr']
    rsi_f     = (df['rsi'] > 50) & (df['close'] > df['highest'] + th)
    af        = df['atr'] >= df['atr'].mean()
    adx_f     = df['adx'] > ADX_THRESHOLD
    trend_f   = df['close'] > df['ema_hf']
    vol_f     = df['volume'] >= params['VOL_MULT'] * df['vol_ma']
    vwap_f    = df['close'] > df['vwap_upper']
    time_f    = ~df['datetime'].dt.hour.isin(EXCLUDE_HOURS)
    reg_f     = df['atr_ratio'] >= params['REGIME_MULT']

    # Diagnostics
    print(f"After EMA:       {ema_f.sum():6d} / {total}")
    print(f"After RSI+BO:    {rsi_f.sum():6d} / {total}")
    print(f"After ADX+ATR:   {(af&adx_f).sum():6d} / {total}")
    print(f"After 1h trend:  {trend_f.sum():6d} / {total}")
    print(f"After vol spike: {vol_f.sum():6d} / {total}")
    print(f"After VWAP band: {vwap_f.sum():6d} / {total}")
    print(f"After session:   {time_f.sum():6d} / {total}")
    print(f"After regime:    {reg_f.sum():6d} / {total}")

    # Build signals
    call_mask = (ema_f & rsi_f & af & adx_f &
                 trend_f & vol_f & vwap_f & time_f & reg_f)

    put_mask  = (
        (df['close'] < df['ema_long']) &
        (df['rsi']   < 50) &
        (df['close'] < df['lowest'] - th) &
        adx_f & af &
        (df['close'] < df['ema_hf']) &
        vol_f &
        (df['close'] < df['vwap_lower']) &
        time_f &
        reg_f
    )

    df['signal'] = np.where(call_mask, 'CALL',
                     np.where(put_mask, 'PUT', None))

    entries = df[df['signal'].notnull()].copy()
    print(f"After combine:   {len(entries):6d} entries\n")
    entries['idx'] = entries.index

    # Uncomment to enforce 1m confirmation:
    # entries = entries[entries.apply(lambda r: confirm_1m(r, df1m), axis=1)]

    return entries.reset_index(drop=True)

# ── Backtest engine with dynamic stops & scale-out ─────────
def run_backtest(df5, df1m, params):
    entries = find_entries(df5, df1m, params)
    equity, banked = INITIAL_USDT, 0.0
    trades = []

    for _, r in entries.iterrows():
        sig, idx = r['signal'], int(r['idx'])
        epx, atr = df5.at[idx,'open'], df5.at[idx,'atr']
        entry_dt = df5.at[idx,'datetime']

        sl = epx - params['SL_ATR_MULT']*atr if sig=='CALL' else epx + params['SL_ATR_MULT']*atr
        tp = epx + params['TP_ATR_MULT']*atr if sig=='CALL' else epx - params['TP_ATR_MULT']*atr

        risk_usd = equity * params['RISK_PCT'] * LEVERAGE
        dist     = abs(epx - sl)
        raw_ct   = risk_usd/(dist*CONTRACT_SZ) if dist>0 else 0
        cnt      = math.floor(raw_ct/0.001)*0.001
        if cnt < 0.001: 
            continue

        rem, peak, moved, scaled = cnt, epx, False, False

        for j in range(idx+1, min(idx+1+LOOKBACK, len(df5))):
            h, l = df5.at[j,'high'], df5.at[j,'low']

            peak = max(peak, h) if sig=='CALL' else min(peak, l)

            # Move stop to breakeven + 0.2 ATR
            if not moved and ((sig=='CALL' and h>=epx+atr) or (sig=='PUT' and l<=epx-atr)):
                sl = epx + (0.2*atr if sig=='CALL' else -0.2*atr)
                moved = True

            # Scale out half at 2*ATR
            if not scaled and ((sig=='CALL' and h>=epx+2*atr) or (sig=='PUT' and l<=epx-2*atr)):
                half = rem/2
                ie   = epx*(1+SLIPPAGE) if sig=='CALL' else epx*(1-SLIPPAGE)
                ex   = (epx+2*atr)*(1-SLIPPAGE) if sig=='CALL' else (epx-2*atr)*(1+SLIPPAGE)
                pnl_pc = (ex-ie) if sig=='CALL' else (ie-ex)
                pnl_usd = pnl_pc * half * CONTRACT_SZ
                fee     = (ie+ex)*half*CONTRACT_SZ*TAKER_FEE
                pnl_usd -= fee

                if pnl_usd > 0:
                    banked += pnl_usd * 0.7
                    equity += pnl_usd * 0.3
                else:
                    equity += pnl_usd

                trades.append({
                    'entry_dt':  entry_dt,
                    'exit_dt':   df5.at[j,'datetime'],
                    'signal':    sig,
                    'contracts': half,
                    'entry_px':  epx,
                    'exit_px':   epx + (2*atr if sig=='CALL' else -2*atr),
                    'pnl_usd':   pnl_usd,
                    'equity':    equity,
                    'banked':    banked
                })
                rem    -= half
                scaled = True

            # Determine current SL (trailing after breakeven)
            cur_sl = sl if not moved else (peak-atr if sig=='CALL' else peak+atr)

            # Exit condition
            exit_now = (
                (sig=='CALL' and (l<=cur_sl or h>=tp)) or
                (sig=='PUT'  and (h>=cur_sl or l<=tp))
            )
            if exit_now:
                exit_px = cur_sl if ((sig=='CALL' and l<=cur_sl) or (sig=='PUT' and h>=cur_sl)) else tp
                ie      = epx*(1+SLIPPAGE) if sig=='CALL' else epx*(1-SLIPPAGE)
                ex      = exit_px*(1-SLIPPAGE) if sig=='CALL' else exit_px*(1+SLIPPAGE)
                pnl_pc  = (ex-ie) if sig=='CALL' else (ie-ex)
                pnl_usd = pnl_pc * rem * CONTRACT_SZ
                fee     = (ie+ex)*rem*CONTRACT_SZ*TAKER_FEE
                pnl_usd -= fee

                if pnl_usd > 0:
                    banked += pnl_usd * 0.7
                    equity += pnl_usd * 0.3
                else:
                    equity += pnl_usd

                trades.append({
                    'entry_dt':  entry_dt,
                    'exit_dt':   df5.at[j,'datetime'],
                    'signal':    sig,
                    'contracts': rem,
                    'entry_px':  epx,
                    'exit_px':   exit_px,
                    'pnl_usd':   pnl_usd,
                    'equity':    equity,
                    'banked':    banked
                })
                break
        else:
            # Time stop at LOOKBACK
            ei = min(idx+LOOKBACK, len(df5)-1)
            xp = df5.at[ei,'close']
            dt2= df5.at[ei,'datetime']
            ie = epx*(1+SLIPPAGE) if sig=='CALL' else epx*(1-SLIPPAGE)
            ex = xp*(1-SLIPPAGE) if sig=='CALL' else xp*(1+SLIPPAGE)
            pnl_pc  = (ex-ie) if sig=='CALL' else (ie-ex)
            pnl_usd = pnl_pc * rem * CONTRACT_SZ
            fee     = (ie+ex)*rem*CONTRACT_SZ*TAKER_FEE
            pnl_usd -= fee

            if pnl_usd > 0:
                banked += pnl_usd * 0.7
                equity += pnl_usd * 0.3
            else:
                equity += pnl_usd

            trades.append({
                'entry_dt':  entry_dt,
                'exit_dt':   dt2,
                'signal':    sig,
                'contracts': rem,
                'entry_px':  epx,
                'exit_px':   xp,
                'pnl_usd':   pnl_usd,
                'equity':    equity,
                'banked':    banked
            })

    # build output
    df_tr = pd.DataFrame(trades)
    if df_tr.empty:
        return df_tr, {
            'total_return_pct':       0.0,
            'annualized_return_pct':  np.nan,
            'sharpe_ratio':           -np.inf,
            'max_drawdown_pct':       np.nan,
            'win_rate_pct':           np.nan,
            'num_trades':             0,
            'profit_factor':          np.nan,
            'avg_win_usd':            np.nan,
            'avg_loss_usd':           np.nan,
            'max_win_usd':            np.nan,
            'max_loss_usd':           np.nan,
            'avg_duration_min':       np.nan,
            'final_balance_usdt':     INITIAL_USDT
        }

    # Compute metrics
    df_tr['total'] = df_tr['equity'] + df_tr['banked']
    tot   = df_tr['total'].iloc[-1]/INITIAL_USDT - 1
    days  = (df_tr['exit_dt'].iloc[-1] - df_tr['entry_dt'].iloc[0]).total_seconds() / 86400
    cagr  = (1+tot)**(365/days)-1 if days>0 else np.nan
    rets  = df_tr['total'].pct_change().fillna(0)
    shar  = rets.mean()/rets.std() if rets.std() else np.nan
    dd    = (df_tr['total'] - df_tr['total'].cummax())/df_tr['total'].cummax()
    maxdd = dd.min()
    wr    = (df_tr['pnl_usd']>0).mean()
    dur   = (df_tr['exit_dt']-df_tr['entry_dt']).dt.total_seconds()/60
    pf    = df_tr.loc[df_tr['pnl_usd']>0,'pnl_usd'].sum() / -df_tr.loc[df_tr['pnl_usd']<0,'pnl_usd'].sum()

    metrics = {
        'total_return_pct':      tot*100,
        'annualized_return_pct': cagr*100,
        'sharpe_ratio':          shar,
        'max_drawdown_pct':      maxdd*100,
        'win_rate_pct':          wr*100,
        'num_trades':            len(df_tr),
        'profit_factor':         pf,
        'avg_win_usd':           df_tr.loc[df_tr['pnl_usd']>0,'pnl_usd'].mean(),
        'avg_loss_usd':          df_tr.loc[df_tr['pnl_usd']<0,'pnl_usd'].mean(),
        'max_win_usd':           df_tr['pnl_usd'].max(),
        'max_loss_usd':          df_tr['pnl_usd'].min(),
        'avg_duration_min':      dur.mean(),
        'final_balance_usdt':    df_tr['total'].iloc[-1]
    }
    return df_tr, metrics

# ── Main ────────────────────────────────────────────────────
def main():
    df1m = prepare_df(DATA_1M)
    df5m = prepare_df(DATA_5M)
    df1h = prepare_df(DATA_1H)

    back5, test5 = split_data(df5m)

    # 1) Grid search on backtest slice
    grid = []
    for combo in [dict(zip(param_grid, v)) for v in product(*param_grid.values())]:
        hf    = preprocess_1h(df1h, combo)
        b5    = preprocess_5m(back5, combo, hf)
        _, m = run_backtest(b5, df1m, combo)
        grid.append({**combo, **m})
    pd.DataFrame(grid).to_csv('best_parameters.csv', index=False)

    # 2) Pick best by Sharpe
    grid_df    = pd.DataFrame(grid)
    best       = grid_df.loc[grid_df['sharpe_ratio'].idxmax()]
    best_p     = {k:int(best[k]) if k in ['ATR_WINDOW','EMA_LONG'] else float(best[k])
                  for k in param_grid}
    print("Best hyperparameters:", best_p)

    # 3) Backtest + Testrun
    hf_b    = preprocess_1h(df1h, best_p)
    hf_t    = hf_b.copy()
    b5_b    = preprocess_5m(back5, best_p, hf_b)
    b5_t    = preprocess_5m(test5,  best_p, hf_t)
    bt_tr, bt_m = run_backtest(b5_b, df1m, best_p)
    ts_tr, ts_m = run_backtest(b5_t, df1m, best_p)

    bt_tr.to_csv('backtest_with_bst_para.csv', index=False)
    pd.DataFrame([bt_m]).to_csv('backtest_results.csv', index=False)

    ts_tr.to_csv('testrun_with_bst_para.csv', index=False)
    pd.DataFrame([ts_m]).to_csv('testrun_results.csv', index=False)

if __name__=="__main__":
    main()
