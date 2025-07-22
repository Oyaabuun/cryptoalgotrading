#!/usr/bin/env python3
"""
Down-trend continuation strategy with micro‐lot sizing:

– Only short (PUT) signals when 1 h close < 1 h EMA
– Breakout to the downside of a 5 m channel
– ATR/RSI/ADX regime filters same as trend model
– Dynamic stops, breakeven + 0.2×ATR, scale‐out half at 2×ATR
– Hyperparameter grid search over ATR_WINDOW, EMA_LONG, SL_ATR_MULT, TP_ATR_MULT, RISK_PCT
"""

import math
import pandas as pd
import numpy as np
from itertools import product

# ── Static params ──────────────────────────────────────────
LOOKBACK      = 20
ADX_WINDOW    = 14
ADX_THRESHOLD = 20
INITIAL_USDT  = 58.0
LEVERAGE      = 10.0
CONTRACT_SZ   = 0.001   # enforce Binance minQty & stepSize = 0.001
TAKER_FEE     = 0.0005
SLIPPAGE      = 0.0005

# ── Hyperparameter grid ────────────────────────────────────
param_grid = {
    'ATR_WINDOW':  [14, 20],
    'EMA_LONG':    [50, 100],
    'SL_ATR_MULT': [1.0, 1.25],
    'TP_ATR_MULT': [4.0, 5.0],
    'RISK_PCT':    [0.005, 0.01],
}

# ── Preprocess 5 m + merge 1 h EMA ─────────────────────────
def preprocess(df5, df1h, params):
    # build 1h EMA
    hf = df1h.sort_values('datetime').copy()
    hf['ema_hf'] = hf['close'].ewm(span=int(params['EMA_LONG']), adjust=False).mean()
    hf = hf[['datetime','ema_hf']]

    df = df5.sort_values('datetime').copy()
    # True range & ATR
    df['prev_close'] = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['prev_close']).abs(),
        (df['low']  - df['prev_close']).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(int(params['ATR_WINDOW'])).mean()

    # Slow indicators
    df['ema_long'] = df['close'].ewm(span=int(params['EMA_LONG']), adjust=False).mean()
    delta = df['close'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi'] = 100 - 100/(1 + up.rolling(int(params['ATR_WINDOW'])).mean()/
                             down.rolling(int(params['ATR_WINDOW'])).mean())

    # Breakout levels
    df['highest'] = df['high'].rolling(LOOKBACK).max().shift(1)
    df['lowest']  = df['low'].rolling(LOOKBACK).min().shift(1)

    # ADX
    up_m   = df['high'].diff()
    down_m = -(df['low'].shift(1).diff())
    plus   = np.where((up_m>down_m)&(up_m>0), up_m, 0.0)
    minus  = np.where((down_m>up_m)&(down_m>0), down_m, 0.0)
    sm_tr  = tr.ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    sm_p   = pd.Series(plus).ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    sm_m   = pd.Series(minus).ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    df['adx'] = 100 * (sm_p - sm_m).abs()/(sm_p + sm_m)

    # Merge 1 h EMA
    df['hour'] = df['datetime'].dt.floor('h')
    df = df.merge(hf.rename(columns={'datetime':'hour'}),
                  on='hour', how='left')
    return df.dropna().reset_index(drop=True)

# ── Find only PUT entries ───────────────────────────────────
def find_entries(df, params):
    m_atr = df['atr'].mean()
    th    = 0.5 * df['atr']

    bear = (
        (df['close'] < df['ema_long']) &
        (df['rsi']   < 50) &
        (df['close'] < df['lowest'] - th) &
        (df['adx'] > ADX_THRESHOLD) &
        (df['atr'] >= m_atr) &
        (df['close'] < df['ema_hf'])
    )

    df['signal'] = np.where(bear, 'PUT', None)
    entries = df[df['signal']=='PUT'].copy()
    entries['idx'] = entries.index
    return entries.reset_index(drop=True)

# ── Backtest engine (shorts only) ──────────────────────────
def backtest(df, params):
    entries = find_entries(df, params)
    equity, banked = INITIAL_USDT, 0.0
    trades = []

    for _, r in entries.iterrows():
        idx = int(r['idx'])
        epx = df.at[idx,'open']
        atr = df.at[idx,'atr']
        entry_dt = df.at[idx,'datetime']

        # SL above entry, TP below
        sl = epx + params['SL_ATR_MULT'] * atr
        tp = epx - params['TP_ATR_MULT'] * atr

        # Position sizing
        risk_usd = equity * params['RISK_PCT'] * LEVERAGE
        dist     = abs(epx - sl)
        raw_ct   = risk_usd / (dist * CONTRACT_SZ) if dist>0 else 0
        cnt      = math.floor(raw_ct / 1.0) * 1.0 * CONTRACT_SZ
        # above, raw_ct is in “number of micro‐contracts” since CONTRACT_SZ=0.001,
        # so math.floor(raw_ct) * CONTRACT_SZ rounds to 0.001 increments
        if cnt < CONTRACT_SZ:
            continue

        rem, peak, moved, scaled = cnt, epx, False, False

        # walk forward
        for j in range(idx+1, min(idx+1+LOOKBACK, len(df))):
            h, l = df.at[j,'high'], df.at[j,'low']
            peak = min(peak, l)

            # breakeven at 1×ATR move
            if not moved and l <= epx - atr:
                sl = epx - 0.2 * atr
                moved = True

            # scale-out half at 2×ATR
            if not scaled and l <= epx - 2*atr:
                half = rem / 2
                ie   = epx*(1-SLIPPAGE)
                ex   = (epx - 2*atr)*(1+SLIPPAGE)
                pnl_pc = ie - ex
                pnl_usd = pnl_pc * half * CONTRACT_SZ
                fee     = (ie+ex) * half * CONTRACT_SZ * TAKER_FEE
                pnl_usd -= fee

                if pnl_usd > 0:
                    banked += pnl_usd * 0.7
                    equity += pnl_usd * 0.3
                else:
                    equity += pnl_usd

                trades.append({
                    'entry_dt': entry_dt,
                    'exit_dt':  df.at[j,'datetime'],
                    'signal':   'PUT',
                    'contracts': half,
                    'entry_px': epx,
                    'exit_px':  epx - 2*atr,
                    'pnl_usd':  pnl_usd,
                    'equity':   equity,
                    'banked':   banked
                })
                rem    -= half
                scaled = True

            # trailing/sl exit
            cur_sl = sl if not moved else (peak + atr)
            if h >= cur_sl or l <= tp:
                exit_px = cur_sl if h >= cur_sl else tp
                ie      = epx*(1-SLIPPAGE)
                ex      = exit_px*(1+SLIPPAGE)
                pnl_pc  = ie - ex
                pnl_usd = pnl_pc * rem * CONTRACT_SZ
                fee     = (ie+ex) * rem * CONTRACT_SZ * TAKER_FEE
                pnl_usd -= fee

                if pnl_usd > 0:
                    banked += pnl_usd * 0.7
                    equity += pnl_usd * 0.3
                else:
                    equity += pnl_usd

                trades.append({
                    'entry_dt': entry_dt,
                    'exit_dt':  df.at[j,'datetime'],
                    'signal':   'PUT',
                    'contracts': rem,
                    'entry_px': epx,
                    'exit_px':  exit_px,
                    'pnl_usd':  pnl_usd,
                    'equity':   equity,
                    'banked':   banked
                })
                break
        else:
            # time‐stop
            ei  = min(idx+LOOKBACK, len(df)-1)
            xp  = df.at[ei,'close']
            dt2 = df.at[ei,'datetime']
            ie  = epx*(1-SLIPPAGE)
            ex  = xp*(1+SLIPPAGE)
            pnl_pc  = ie - ex
            pnl_usd = pnl_pc * rem * CONTRACT_SZ
            fee     = (ie+ex) * rem * CONTRACT_SZ * TAKER_FEE
            pnl_usd -= fee

            if pnl_usd > 0:
                banked += pnl_usd * 0.7
                equity += pnl_usd * 0.3
            else:
                equity += pnl_usd

            trades.append({
                'entry_dt': entry_dt,
                'exit_dt':  dt2,
                'signal':   'PUT',
                'contracts': rem,
                'entry_px': epx,
                'exit_px':  xp,
                'pnl_usd':  pnl_usd,
                'equity':   equity,
                'banked':   banked
            })

    df_tr = pd.DataFrame(trades)
    # …metric calculations unchanged…
    return df_tr, _compute_metrics(df_tr)

def _compute_metrics(df_tr):
    if df_tr.empty:
        return {
            'total_return_pct':      0.0,
            'annualized_return_pct': np.nan,
            'sharpe_ratio':          -np.inf,
            'max_drawdown_pct':      np.nan,
            'win_rate_pct':          np.nan,
            'num_trades':            0,
            'profit_factor':         np.nan,
            'avg_win_usd':           np.nan,
            'avg_loss_usd':          np.nan,
            'max_win_usd':           np.nan,
            'max_loss_usd':          np.nan,
            'avg_duration_min':      np.nan,
            'final_balance_usdt':    INITIAL_USDT
        }
    df_tr['total'] = df_tr['equity'] + df_tr['banked']
    tot   = df_tr['total'].iloc[-1]/INITIAL_USDT - 1
    days  = (df_tr['exit_dt'].iloc[-1] - df_tr['entry_dt'].iloc[0]).total_seconds()/86400
    cagr  = (1+tot)**(365/days)-1 if days>0 else np.nan
    rets  = df_tr['total'].pct_change().fillna(0)
    shar  = rets.mean()/rets.std() if rets.std() else np.nan
    dd    = (df_tr['total'] - df_tr['total'].cummax())/df_tr['total'].cummax()
    maxdd = dd.min()
    wins  = df_tr['pnl_usd']>0
    wr    = wins.mean()
    pf    = df_tr.loc[wins,'pnl_usd'].sum() / -df_tr.loc[~wins,'pnl_usd'].sum()
    dur   = (df_tr['exit_dt'] - df_tr['entry_dt']).dt.total_seconds()/60
    return {
        'total_return_pct':      tot*100,
        'annualized_return_pct': cagr*100,
        'sharpe_ratio':          shar,
        'max_drawdown_pct':      maxdd*100,
        'win_rate_pct':          wr*100,
        'num_trades':            len(df_tr),
        'profit_factor':         pf,
        'avg_win_usd':           df_tr.loc[wins,'pnl_usd'].mean(),
        'avg_loss_usd':          df_tr.loc[~wins,'pnl_usd'].mean(),
        'max_win_usd':           df_tr['pnl_usd'].max(),
        'max_loss_usd':          df_tr['pnl_usd'].min(),
        'avg_duration_min':      dur.mean(),
        'final_balance_usdt':    df_tr['total'].iloc[-1]
    }

# ── Top‐level runner ────────────────────────────────────────
def run_strategy(df_1m_train, df_5m_train, df_1h_train,
                 df_1m_test,  df_5m_test,  df_1h_test):

    # grid‐search on train
    grid = []
    for combo in [dict(zip(param_grid, v)) for v in product(*param_grid.values())]:
        dfb = preprocess(df_5m_train.copy(), df_1h_train, combo)
        _, m = backtest(dfb, combo)
        grid.append({**combo, **m})
    pd.DataFrame(grid).to_csv('downtrend_best_parameters.csv', index=False)

    # select best by Sharpe
    best = pd.DataFrame(grid).loc[pd.DataFrame(grid)['sharpe_ratio'].idxmax()]
    best_p = {
        k: int(best[k]) if k in ['ATR_WINDOW','EMA_LONG'] else float(best[k])
        for k in param_grid
    }

    # backtest on train
    dfb, _   = preprocess(df_5m_train.copy(), df_1h_train, best_p), None
    bt_tr, bt_m = backtest(dfb, best_p)
    bt_tr.to_csv('downtrend_backtest_with_bst_para.csv', index=False)
    pd.DataFrame([bt_m]).to_csv('downtrend_backtest_results.csv', index=False)

    # testrun on test
    dft, _  = preprocess(df_5m_test.copy(), df_1h_test, best_p), None
    ts_tr, ts_m = backtest(dft, best_p)
    ts_tr.to_csv('downtrend_testrun_with_bst_para.csv', index=False)
    pd.DataFrame([ts_m]).to_csv('downtrend_testrun_results.csv', index=False)
