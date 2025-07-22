#!/usr/bin/env python3
"""
Filtered Optuna‐driven optimization with “must‐trade” constraint and top‐K proposal.
– Random sample stage to build surrogate data not used here (could be removed)
– Bayesian TPE Optuna with penalties for < N_MIN_ENTRIES on train slice
– After optimization, retest top K Pareto‐best trials on test slice (must trade) and export
"""
import os
import math
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import optuna

# ── Logging Setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────
DATA_5M       = "data/BTCUSDT_5m_5year.csv"
DATA_1H       = "data/BTCUSDT_1h_5year.csv"
LIMITER_DAYS  = 1460
TRAIN_PCT     = 0.7
N_OPT_TRIALS  = 100
TOP_K         = 5
N_MIN_ENTRIES = 10   # must‐trade filter
OUTPUT_DIR    = os.path.abspath('.')
IMAGE_DIR     = os.path.join(OUTPUT_DIR, 'images')

param_bounds = {
    'ATR_WINDOW':  (5, 50),
    'EMA_LONG':    (20, 200),
    'SL_ATR_MULT': (0.5, 2.0),
    'TP_ATR_MULT': (2.0, 8.0),
    'VOL_MULT':    (1.0, 3.0),
    'RISK_PCT':    (0.001, 0.02),
    'REGIME_MULT': (0.5, 2.0),
}

# ── Constants ──────────────────────────────────────────────────────────
LOOKBACK      = 20
ADX_WINDOW    = 14
ADX_THRESHOLD = 20
INITIAL_USDT  = 58.0
LEVERAGE      = 10.0
CONTRACT_SZ   = 0.001
TAKER_FEE     = 0.0005
SLIPPAGE      = 0.0005
VWAP_BAND_PCT = 0.001
EXCLUDE_HOURS = {0}

# ── I/O & slicing helpers ─────────────────────────────────────────────
def load_and_normalize(path):
    df = pd.read_csv(path)
    if 'open_time' in df:
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    elif 'timestamp' in df:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)

def limit_to_first_n_days(df, n):
    days = df['datetime'].dt.normalize().drop_duplicates().sort_values()
    return df[df['datetime'].dt.normalize().isin(days.iloc[:n])].reset_index(drop=True)

def split_by_days(df, pct):
    days = df['datetime'].dt.normalize().drop_duplicates().sort_values()
    cut  = int(len(days) * pct)
    return set(days.iloc[:cut]), set(days.iloc[cut:])

def slice_days(df, days_set):
    return df[df['datetime'].dt.normalize().isin(days_set)].reset_index(drop=True)

# ── Feature preprocessing ───────────────────────────────────────────
def preprocess_1h(df1h, params):
    df = df1h.copy()
    df['ema_hf'] = df['close'].ewm(span=int(params['EMA_LONG']), adjust=False).mean()
    return df[['datetime', 'ema_hf']]

# after defining LOOKBACK, ADX_WINDOW
LOOKBACK      = 20
ADX_WINDOW    = 14
ADX_THRESHOLD = 20
INITIAL_USDT  = 58.0
LEVERAGE      = 10.0
CONTRACT_SZ   = 0.001
TAKER_FEE     = 0.0005
SLIPPAGE      = 0.0005
VWAP_BAND_PCT = 0.001
EXCLUDE_HOURS = {0}

def preprocess_5m(df5m, params, df_hf):
    df = df5m.copy()
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
    # Slow ATR & ratio
    df['atr_slow']  = df['atr'].rolling(atr_win * 2).mean()
    df['atr_ratio'] = df['atr'] / df['atr_slow']
    # RSI
    delta = df['close'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi'] = 100 - 100 / (1 + up.rolling(atr_win).mean() / down.rolling(atr_win).mean())
    # EMA & breakout levels
    df['ema_long'] = df['close'].ewm(span=ema_l, adjust=False).mean()
    df['highest']  = df['high'].rolling(LOOKBACK).max().shift(1)
    df['lowest']   = df['low'].rolling(LOOKBACK).min().shift(1)
    # ADX
    up_m   = df['high'].diff()
    down_m = -(df['low'].shift(1).diff())
    plus   = np.where((up_m > down_m) & (up_m > 0), up_m, 0.0)
    minus  = np.where((down_m > up_m) & (down_m > 0), down_m, 0.0)
    sm_tr  = tr.ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    sm_p   = pd.Series(plus).ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    sm_m   = pd.Series(minus).ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    df['adx'] = 100 * (sm_p - sm_m).abs() / (sm_p + sm_m)
    # Volume MA
    df['vol_ma'] = df['volume'].rolling(20).mean()
    # VWAP & bands
    df['date']     = df['datetime'].dt.date
    typ = (df['high'] + df['low'] + df['close']) / 3
    df['cum_vp']  = typ.mul(df['volume']).groupby(df['date']).cumsum()
    df['cum_vol'] = df['volume'].groupby(df['date']).cumsum()
    df['vwap']    = df['cum_vp'] / df['cum_vol']
    df['vwap_upper'] = df['vwap'] * (1 + VWAP_BAND_PCT)
    df['vwap_lower'] = df['vwap'] * (1 - VWAP_BAND_PCT)
    # Merge 1h EMA
    df['hour'] = df['datetime'].dt.floor('h')
    df = df.merge(df_hf.rename(columns={'datetime': 'hour'}), on='hour', how='left')
    return df.dropna().reset_index(drop=True)

# ── Entry detection & backtest ────────────────────────────────────────
def find_entries(df5, params):
    df = df5.copy()
    total = len(df)
    logger.info(f"=== PARAMS {params} === Total bars: {total}")
    th = 0.5 * df['atr']
    ema_f  = df['close'] > df['ema_long']
    rsi_f  = (df['rsi'] > 50) & (df['close'] > df['highest'] + th)
    atr_f  = df['atr'] >= df['atr'].mean()
    adx_f  = df['adx'] > ADX_THRESHOLD
    trend_f= df['close'] > df['ema_hf']
    vol_f  = df['volume'] >= params['VOL_MULT'] * df['vol_ma']
    vwap_f = df['close'] > df['vwap_upper']
    time_f = ~df['datetime'].dt.hour.isin(EXCLUDE_HOURS)
    reg_f  = df['atr_ratio'] >= params['REGIME_MULT']
    logger.info(f"After EMA:       {ema_f.sum():4d}/{total}")
    logger.info(f"After RSI+BO:    {rsi_f.sum():4d}/{total}")
    logger.info(f"After ADX+ATR:   {(atr_f & adx_f).sum():4d}/{total}")
    logger.info(f"After 1h trend:  {trend_f.sum():4d}/{total}")
    logger.info(f"After vol spike: {vol_f.sum():4d}/{total}")
    logger.info(f"After VWAP band: {vwap_f.sum():4d}/{total}")
    logger.info(f"After session:   {time_f.sum():4d}/{total}")
    logger.info(f"After regime:    {reg_f.sum():4d}/{total}")
    call = ema_f & rsi_f & atr_f & adx_f & trend_f & vol_f & vwap_f & time_f & reg_f
    put  = (
        (df['close'] < df['ema_long']) &
        (df['rsi']   < 50) &
        (df['close'] < df['lowest'] - th) & adx_f & atr_f &
        (df['close'] < df['ema_hf']) &
        vol_f &
        (df['close'] < df['vwap_lower']) &
        time_f &
        reg_f
    )
    df['signal'] = np.where(call, 'CALL', np.where(put, 'PUT', None))
    entries = df[df['signal'].notnull()].copy()
    logger.info(f"After combine:   {len(entries):4d} entries")
    entries['idx'] = entries.index
    return entries.reset_index(drop=True)

def run_backtest(df5, params):
    entries = find_entries(df5, params)
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
        cnt      = math.floor(raw_ct/CONTRACT_SZ)*CONTRACT_SZ
        if cnt < CONTRACT_SZ:
            continue
        rem, peak, moved, scaled = cnt, epx, False, False
        for j in range(idx+1, min(idx+1+LOOKBACK, len(df5))):
            h, l = df5.at[j,'high'], df5.at[j,'low']
            peak = max(peak, h) if sig=='CALL' else min(peak, l)
            if not moved and ((sig=='CALL' and h>=epx+atr) or (sig=='PUT' and l<=epx-atr)):
                sl, moved = (epx+0.2*atr if sig=='CALL' else epx-0.2*atr), True
            if not scaled and ((sig=='CALL' and h>=epx+2*atr) or (sig=='PUT' and l<=epx-2*atr)):
                half = rem/2
                ie   = epx*(1+SLIPPAGE) if sig=='CALL' else epx*(1-SLIPPAGE)
                ex   = ((epx+2*atr)*(1-SLIPPAGE) if sig=='CALL' else (epx-2*atr)*(1+SLIPPAGE))
                pnl_pc  = (ex-ie) if sig=='CALL' else (ie-ex)
                pnl_usd = pnl_pc*half*CONTRACT_SZ
                fee     = (ie+ex)*half*CONTRACT_SZ*TAKER_FEE
                pnl_usd -= fee
                if pnl_usd>0:
                    banked += pnl_usd*0.7; equity += pnl_usd*0.3
                else:
                    equity += pnl_usd
                trades.append({ 'entry_dt':entry_dt, 'exit_dt':df5.at[j,'datetime'], 'signal':sig,
                                'contracts':half, 'entry_px':epx, 'exit_px':ex,'pnl_usd':pnl_usd,
                                'equity':equity, 'banked':banked })
                rem -= half; scaled = True
            cur_sl = sl if not moved else (peak-atr if sig=='CALL' else peak+atr)
            exit_now = (
                (sig=='CALL' and (l<=cur_sl or h>=tp)) or
                (sig=='PUT' and (h>=cur_sl or l<=tp))
            )
            if exit_now:
                exit_px = cur_sl if ((sig=='CALL' and l<=cur_sl) or (sig=='PUT' and h>=cur_sl)) else tp
                ie      = epx*(1+SLIPPAGE) if sig=='CALL' else epx*(1-SLIPPAGE)
                ex      = exit_px*(1-SLIPPAGE) if sig=='CALL' else exit_px*(1+SLIPPAGE)
                pnl_pc  = (ex-ie) if sig=='CALL' else (ie-ex)
                pnl_usd = pnl_pc*rem*CONTRACT_SZ
                fee     = (ie+ex)*rem*CONTRACT_SZ*TAKER_FEE
                pnl_usd -= fee
                if pnl_usd>0:
                    banked += pnl_usd*0.7; equity += pnl_usd*0.3
                else:
                    equity += pnl_usd
                trades.append({ 'entry_dt':entry_dt, 'exit_dt':df5.at[j,'datetime'], 'signal':sig,
                                'contracts':rem, 'entry_px':epx, 'exit_px':exit_px,'pnl_usd':pnl_usd,
                                'equity':equity, 'banked':banked })
                break
        else:
            ei = min(idx+LOOKBACK, len(df5)-1)
            xp, dt2 = df5.at[ei,'close'], df5.at[ei,'datetime']
            ie = epx*(1+SLIPPAGE) if sig=='CALL' else epx*(1-SLIPPAGE)
            ex = xp*(1-SLIPPAGE) if sig=='CALL' else xp*(1+SLIPPAGE)
            pnl_pc  = (ex-ie) if sig=='CALL' else (ie-ex)
            pnl_usd = pnl_pc*rem*CONTRACT_SZ
            fee     = (ie+ex)*rem*CONTRACT_SZ*TAKER_FEE
            pnl_usd -= fee
            if pnl_usd>0:
                banked += pnl_usd*0.7; equity += pnl_usd*0.3
            else:
                equity += pnl_usd
            trades.append({ 'entry_dt':entry_dt, 'exit_dt':dt2, 'signal':sig,
                            'contracts':rem, 'entry_px':epx, 'exit_px':xp,'pnl_usd':pnl_usd,
                            'equity':equity, 'banked':banked })
    df_tr = pd.DataFrame(trades)
    if df_tr.empty:
        return df_tr, {'sortino_ratio': -np.inf}
    # Compute Sortino ratio
    df_tr['total'] = df_tr['equity'] + df_tr['banked']
    rets = df_tr['total'].pct_change().fillna(0)
    downside = rets[rets < 0]
    downside_std = downside.std()
    sortino = rets.mean() / downside_std if downside_std > 0 else np.nan
    return df_tr, {'sortino_ratio': sortino}

# ── Sampling & evaluation ────────────────────────────────────────────
def sample_params(n, bounds):
    rng = np.random.RandomState(42)
    out = []
    for _ in range(n):
        c = {p: int(rng.uniform(lo, hi)) if p in ('ATR_WINDOW','EMA_LONG') else rng.uniform(lo,hi)
             for p,(lo,hi) in bounds.items()}
        out.append(c)
    return out

def evaluate_combo(combo, df5_tr, df1h_tr):
    hf = preprocess_1h(df1h_tr, combo)
    b5 = preprocess_5m(df5_tr, combo, hf)
    entries = find_entries(b5, combo)
    num_entries = len(entries)
    _, mets = run_backtest(b5, combo)
    return {**combo, 'sortino_ratio': mets['sortino_ratio'], 'num_entries': num_entries}

# ── Main pipeline ────────────────────────────────────────────────────
def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    # Load & slice
    df5  = limit_to_first_n_days(load_and_normalize(DATA_5M), LIMITER_DAYS)
    df1h = limit_to_first_n_days(load_and_normalize(DATA_1H), LIMITER_DAYS)
    tr_days, ts_days = split_by_days(df5, TRAIN_PCT)
    df5_tr, df5_ts   = slice_days(df5, tr_days), slice_days(df5, ts_days)
    df1h_tr, df1h_ts = slice_days(df1h, tr_days), slice_days(df1h, ts_days)

    # ── Optuna objective with must‐trade filter on train slice ─────────
    def objective(trial):
        # suggest params
        params = {
            p: trial.suggest_int(p, *param_bounds[p]) if p in ('ATR_WINDOW','EMA_LONG')
               else trial.suggest_float(p, *param_bounds[p])
            for p in param_bounds
        }
        # evaluate on train slice
        df5_proc = preprocess_5m(df5_tr, params, preprocess_1h(df1h_tr, params))
        # run backtest
        trades, mets = run_backtest(df5_proc, params)
        sortino = mets.get('sortino_ratio', -np.inf)
        num_entries = len(trades)
        # penalize no‐trade
        if num_entries < N_MIN_ENTRIES:
            return -1e6
        return sortino

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_OPT_TRIALS)
    logger.info(f"Optuna done. Best train‐slice Sortino: {study.best_value}")

    # ── Retest top‐K trials on test slice ──────────────────────────────
    trials = sorted(study.trials, key=lambda t: t.value or -np.inf, reverse=True)
    top_results = []
    for t in trials:
        if len(top_results) >= TOP_K:
            break
        combo = t.params
        # retest on test slice
        df5_proc = preprocess_5m(df5_ts, combo, preprocess_1h(df1h_ts, combo))
        trades, mets = run_backtest(df5_proc, combo)
        if len(trades) >= N_MIN_ENTRIES:
            rec = {**combo, 'sortino_test': mets['sortino_ratio'], 'num_entries_test': len(trades)}
            top_results.append(rec)

    # Export
    df_top = pd.DataFrame(top_results)
    df_top.to_csv('top_optuna_combos_filtered.csv', index=False)
    logger.info(f"Exported top {len(df_top)} combos to top_optuna_combos_filtered.csv")

if __name__ == "__main__":
    main()