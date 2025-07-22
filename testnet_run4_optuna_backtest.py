#!/usr/bin/env python3
import os
import math
import csv
import logging
from collections import deque
from ml_way4_optuna import find_entries
import pandas as pd
import numpy as np
import multiprocessing

# ── bring in your existing helpers ───────────────────────────────
from ml_way4_optuna import (
    preprocess_1h,
    preprocess_5m,
    LOOKBACK,
    INITIAL_USDT,
    LEVERAGE,
    CONTRACT_SZ,
    TAKER_FEE,
    SLIPPAGE
)
# ── end imports ─────────────────────────────────────────────────

# ── Logging & paths ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_5M    = r"D:\\algo_crypto\\algo_crypto\\mymodules\\data\\BTCUSDT_5m_1YEAR.csv"
DATA_1H    = r"D:\\algo_crypto\\algo_crypto\\mymodules\\data\\BTCUSDT_1h_1YEAR.csv"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── live-style simulation function ───────────────────────────────
def sim_livelike(csv5m_path, csv1h_path, params):
    max_buffer = int(max(params["EMA_LONG"],
                         params["ATR_WINDOW"] * 2,
                         LOOKBACK,
                         20))
    buf5 = deque(maxlen=max_buffer)

    f5 = open(csv5m_path, newline='')
    r5 = csv.DictReader(f5)
    f1 = open(csv1h_path, newline='')
    r1 = csv.DictReader(f1)
    next1 = next(r1, None)
    ema_prev = None
    trades, open_pos = [], None
    equity, banked = INITIAL_USDT, 0.0

    for row5 in r5:
        if 'datetime' in row5:
            dt5 = pd.to_datetime(row5['datetime'])
        elif 'open_time' in row5:
            dt5 = pd.to_datetime(int(row5['open_time']), unit='ms')
        elif 'timestamp' in row5:
            dt5 = pd.to_datetime(int(row5['timestamp']), unit='ms')
        else:
            raise KeyError(f"No recognized time column in 5m CSV: {row5.keys()}")

        bar5 = {k: float(row5.get(k, row5.get(k.capitalize(), 0))) for k in ['open','high','low','close','volume']}
        bar5['datetime'] = dt5

        while next1:
            if 'datetime' in next1:
                dt1 = pd.to_datetime(next1['datetime'])
            elif 'open_time' in next1:
                dt1 = pd.to_datetime(int(next1['open_time']), unit='ms')
            elif 'timestamp' in next1:
                dt1 = pd.to_datetime(int(next1['timestamp']), unit='ms')
            else:
                raise KeyError("No recognized time column in 1h CSV")

            if dt1 <= dt5.floor('h'):
                close1 = float(next1.get('close', next1.get('Close', 0)))
                alpha = 2/(params['EMA_LONG']+1)
                ema_prev = close1 if ema_prev is None else alpha*close1 + (1-alpha)*ema_prev
                next1 = next(r1, None)
            else:
                break

        buf5.append(bar5)
        if len(buf5) < max_buffer:
            continue

        df5_buf = pd.DataFrame(buf5)
        df1h_buf = pd.DataFrame([{'datetime': dt5.floor('h'), 'ema_hf': ema_prev }])
        df5_ind = preprocess_5m(df5_buf.copy(), params, df1h_buf)
        entries = find_entries(df5_ind, params)
        new_row = entries[entries['datetime']==dt5]
        sig = new_row['signal'].iat[0] if not new_row.empty else None

        if open_pos is None and sig in ('CALL','PUT'):
            open_pos = {
                'signal': sig,
                'entry_dt': dt5,
                'entry_px': new_row['open'].iat[0],
                'atr': new_row['atr'].iat[0],
                'rem': None
            }
            risk_usd = equity * params['RISK_PCT'] * LEVERAGE
            dist = params['SL_ATR_MULT'] * open_pos['atr']
            raw_ct = risk_usd/(dist*CONTRACT_SZ) if dist>0 else 0
            cnt = math.floor(raw_ct/CONTRACT_SZ)*CONTRACT_SZ
            open_pos['rem'] = cnt

        if open_pos:
            h,l = bar5['high'], bar5['low']
            epx, atr, sig = open_pos['entry_px'], open_pos['atr'], open_pos['signal']
            cur_sl = epx-atr if sig=='CALL' else epx+atr
            tp = epx+params['TP_ATR_MULT']*atr if sig=='CALL' else epx-params['TP_ATR_MULT']*atr
            exit_now = ((sig=='CALL' and (l<=cur_sl or h>=tp)) or
                        (sig=='PUT' and (h>=cur_sl or l<=tp)))
            if exit_now:
                exit_px = cur_sl if ((sig=='CALL' and l<=cur_sl) or (sig=='PUT' and h>=cur_sl)) else tp
                ie = epx*(1+SLIPPAGE) if sig=='CALL' else epx*(1-SLIPPAGE)
                ex = exit_px*(1-SLIPPAGE) if sig=='CALL' else exit_px*(1+SLIPPAGE)
                pnl_pc = (ex-ie) if sig=='CALL' else (ie-ex)
                pnl_usd = pnl_pc*open_pos['rem']*CONTRACT_SZ
                fee = (ie+ex)*open_pos['rem']*CONTRACT_SZ*TAKER_FEE
                pnl_usd -= fee
                if pnl_usd>0:
                    banked += pnl_usd*0.7; equity += pnl_usd*0.3
                else:
                    equity += pnl_usd
                trades.append({'entry_dt': open_pos['entry_dt'], 'exit_dt': dt5,
                               'signal': sig, 'entry_px': epx, 'exit_px': exit_px,
                               'contracts': open_pos['rem'], 'pnl_usd': pnl_usd,
                               'equity': equity, 'banked': banked})
                open_pos = None
    return trades

# ── metrics calculator ───────────────────────────────────────────
def compute_metrics(trades):
    df = pd.DataFrame(trades)
    if df.empty: return {}
    df['total'] = df['equity']+df['banked']
    tot = df['total'].iat[-1]/INITIAL_USDT - 1
    days = (df['exit_dt'].iat[-1] - df['entry_dt'].iat[0]).total_seconds()/86400
    cagr = (1+tot)**(365/days)-1 if days>0 else np.nan
    rets = df['total'].pct_change().fillna(0)
    sharpe = rets.mean()/rets.std() if rets.std()>0 else np.nan
    win_rate = (df['pnl_usd']>0).mean()
    pf = df.loc[df['pnl_usd']>0,'pnl_usd'].sum()/ -df.loc[df['pnl_usd']<0,'pnl_usd'].sum()
    dd = (df['total']-df['total'].cummax())/df['total'].cummax()
    maxdd = dd.min()
    return {'total_return_pct': tot*100, 'annualized_return_pct': cagr*100,
            'sharpe_ratio': sharpe, 'win_rate_pct': win_rate*100,
            'profit_factor': pf, 'max_drawdown_pct': maxdd*100,
            'final_balance_usdt': df['total'].iat[-1]}

# ── worker & main ───────────────────────────────────────────────
def run_combo(idx, row_dict):
    params = {k: int(row_dict[k]) if k in ('ATR_WINDOW','EMA_LONG') else float(row_dict[k])
              for k in ['ATR_WINDOW','EMA_LONG','SL_ATR_MULT','TP_ATR_MULT','VOL_MULT','RISK_PCT','REGIME_MULT']}
    trades = sim_livelike(DATA_5M, DATA_1H, params)
    df_tr = pd.DataFrame(trades)
    df_tr.to_csv(os.path.join(OUTPUT_DIR, f"trades_{idx+1}.csv"), index=False)
    metrics = compute_metrics(trades)
    pd.DataFrame([metrics]).to_csv(os.path.join(OUTPUT_DIR, f"metrics_{idx+1}.csv"), index=False)
    logger.info(f"Run {idx+1} complete: params={params}")

if __name__ == '__main__':
    combos_file = r"D:\\algo_crypto\\algo_crypto\\mymodules\\top_optuna_combos_filtered.csv"
    df_params = pd.read_csv(combos_file)
    combos = df_params.to_dict('records')
    args = [(i, combos[i]) for i in range(len(combos))]
    cpu = multiprocessing.cpu_count()
    with multiprocessing.Pool(cpu) as pool:
        pool.starmap(run_combo, args)
