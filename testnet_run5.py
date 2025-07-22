#!/usr/bin/env python3
import os
import math
import json
import logging
import multiprocessing
from collections import deque
import pandas as pd
import numpy as np

# ── bring in your existing helpers ───────────────────────────────
from ml_way4_optuna import (
    preprocess_5m,
    find_entries,
    LOOKBACK,
    INITIAL_USDT,
    LEVERAGE,
    CONTRACT_SZ,
    TAKER_FEE,
    SLIPPAGE
)
# ── end imports ─────────────────────────────────────────────────

# ── Logging & paths ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_5M = r"D:\\algo_crypto\\algo_crypto\\mymodules\\data\\BTCUSDT_5m_1YEAR.csv"
DATA_1H = r"D:\\algo_crypto\\algo_crypto\\mymodules\\data\\BTCUSDT_1h_1YEAR.csv"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Global data holders ─────────────────────────────────────────
df5_vals = {}
df1_vals = {}

# ── Worker initializer: load CSVs once per process ────────────────
def init_worker(csv5m_path, csv1h_path):
    global df5_vals, df1_vals
    df5 = pd.read_csv(csv5m_path)
    df1 = pd.read_csv(csv1h_path)
    # unify datetime
    for df in (df5, df1):
        for col in ['datetime', 'open_time', 'timestamp']:
            if col in df.columns:
                unit = None if col == 'datetime' else 'ms'
                df['datetime'] = pd.to_datetime(df[col], unit=unit)
                break
    # forward-fill
    df5.ffill(inplace=True)
    df1.ffill(inplace=True)
    # prepare arrays
    df1['hour_dt'] = df1['datetime'].values.astype('datetime64[h]')
    df5_vals = {
        'dt': df5['datetime'].values,
        'open': df5['open'].astype(float).to_numpy(),
        'high': df5['high'].astype(float).to_numpy(),
        'low': df5['low'].astype(float).to_numpy(),
        'close': df5['close'].astype(float).to_numpy(),
        'volume': df5['volume'].astype(float).to_numpy()
    }
    df1_vals = {
        'dt': df1['hour_dt'].values,
        'close': df1['close'].astype(float).to_numpy()
    }

# ── Simulation per combo ─────────────────────────────────────────
def sim_livelike(params):
    # build full DataFrames
    df5 = pd.DataFrame({
        'datetime': df5_vals['dt'], 'open': df5_vals['open'],
        'high': df5_vals['high'], 'low': df5_vals['low'],
        'close': df5_vals['close'], 'volume': df5_vals['volume']
    })
    df1 = pd.DataFrame({
        'datetime': df1_vals['dt'], 'close': df1_vals['close']
    }).drop_duplicates('datetime').sort_values('datetime')
    # compute full-hour EMA
    alpha = 2/(params['EMA_LONG']+1)
    df1['ema_hf'] = df1['close'].ewm(alpha=alpha, adjust=False).mean()

    # preprocess and find entries
    df5_ind = preprocess_5m(df5, params, df1[['datetime','ema_hf']])
    # ensure ema present
    df5_ind['hour'] = df5_ind['datetime'].dt.floor('h')
    ema_map = dict(zip(df1['datetime'], df1['ema_hf']))
    df5_ind['ema_hf'] = df5_ind['hour'].map(ema_map)
    entries = find_entries(df5_ind, params)

    # simulate
    trades = []
    equity, banked = INITIAL_USDT, 0.0
    open_pos = None
    entry_dict = {row['datetime']: row for _, row in entries.iterrows()}

    for i, dt in enumerate(df5_vals['dt']):
        if open_pos is None and dt in entry_dict:
            e = entry_dict[dt]
            sig, entry_px, atr = e['signal'], e['open'], e['atr']
            risk = equity * params['RISK_PCT'] * LEVERAGE
            dist = params['SL_ATR_MULT'] * atr
            rem = math.floor(risk/(dist*CONTRACT_SZ))/CONTRACT_SZ*CONTRACT_SZ
            open_pos = (sig, dt, entry_px, atr, rem)
            continue
        if open_pos:
            sig, e_dt, epx, atr, rem = open_pos
            h, l = df5_vals['high'][i], df5_vals['low'][i]
            sl = epx-atr if sig=='CALL' else epx+atr
            tp = epx+params['TP_ATR_MULT']*atr if sig=='CALL' else epx-params['TP_ATR_MULT']*atr
            if (sig=='CALL' and (l<=sl or h>=tp)) or (sig=='PUT' and (h>=sl or l<=tp)):
                exit_px = sl if ((sig=='CALL' and l<=sl) or (sig=='PUT' and h>=sl)) else tp
                ie = epx*(1+SLIPPAGE) if sig=='CALL' else epx*(1-SLIPPAGE)
                ex = exit_px*(1-SLIPPAGE) if sig=='CALL' else exit_px*(1+SLIPPAGE)
                pnl = (ex-ie) if sig=='CALL' else (ie-ex)
                pnl_usd = pnl*rem*CONTRACT_SZ - (ie+ex)*rem*CONTRACT_SZ*TAKER_FEE
                if pnl_usd>0:
                    banked += pnl_usd*0.7; equity+=pnl_usd*0.3
                else:
                    equity+=pnl_usd
                trades.append({'entry_dt':e_dt,'exit_dt':dt,'signal':sig,
                               'entry_px':epx,'exit_px':exit_px,
                               'contracts':rem,'pnl_usd':pnl_usd,
                               'equity':equity,'banked':banked})
                open_pos=None
    return trades

# ── Metrics ─────────────────────────────────────────────────────
def compute_metrics(trades):
    df = pd.DataFrame(trades)
    if df.empty: return {}
    df['total'] = df['equity']+df['banked']
    tot = df['total'].iat[-1]/INITIAL_USDT -1
    days = (df['exit_dt'].iat[-1]-df['entry_dt'].iat[0]).total_seconds()/86400
    cagr = (1+tot)**(365/days)-1 if days>0 else np.nan
    rets = df['total'].pct_change().fillna(0)
    return {'total_return_pct':tot*100,'annualized_return_pct':cagr*100,
            'sharpe_ratio':rets.mean()/rets.std() if rets.std()>0 else np.nan,
            'win_rate_pct':(df['pnl_usd']>0).mean()*100,
            'profit_factor':df.loc[df['pnl_usd']>0,'pnl_usd'].sum()/ -df.loc[df['pnl_usd']<0,'pnl_usd'].sum(),
            'max_drawdown_pct':((df['total']-df['total'].cummax())/df['total'].cummax()).min()*100,
            'final_balance_usdt':df['total'].iat[-1]}

# ── Main ────────────────────────────────────────────────────────
if __name__=='__main__':
    combos = pd.read_csv(r"D:\\algo_crypto\\algo_crypto\\mymodules\\top_optuna_combos_filtered.csv").to_dict('records')
    with multiprocessing.Pool(processes=os.cpu_count(), initializer=init_worker,
                               initargs=(DATA_5M, DATA_1H)) as pool:
        all_trades = pool.map(sim_livelike, combos)
    all_metrics = [compute_metrics(tr) for tr in all_trades]
    with open(os.path.join(OUTPUT_DIR,'metrics_all.json'),'w') as f:
        json.dump(all_metrics,f, indent=2)
    logger.info("Completed all runs.")
