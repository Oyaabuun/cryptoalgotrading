#!/usr/bin/env python3
"""
Multi‐objective Optuna‐driven optimization with:
 - Maximize Sortino, ROI, Win Rate
 - Minimize Max Drawdown
 - Randomized slippage (0.0003–0.0008) based on volume
 - Must–trade & NaN‐safe constraint
"""
import os
import math
import logging
import pandas as pd
import numpy as np
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
N_MIN_ENTRIES = 10
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

# ── Strategy constants ─────────────────────────────────────────────────
LOOKBACK      = 20
ADX_WINDOW    = 14
ADX_THRESHOLD = 20
INITIAL_USDT  = 58.0
LEVERAGE      = 10.0
CONTRACT_SZ   = 0.001
TAKER_FEE     = 0.0005
VWAP_BAND_PCT = 0.001
EXCLUDE_HOURS = {0}

# ── I/O & slicing helpers ─────────────────────────────────────────────
def load_and_normalize(path):
    df = pd.read_csv(path)
    if 'open_time' in df.columns:
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    elif 'timestamp' in df.columns:
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

def preprocess_5m(df5m, params, df_hf):
    df = df5m.copy()
    # True Range & ATR
    atr_win = int(params['ATR_WINDOW'])
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
    ema_l = int(params['EMA_LONG'])
    df['ema_long'] = df['close'].ewm(span=ema_l, adjust=False).mean()
    df['highest']  = df['high'].rolling(LOOKBACK).max().shift(1)
    df['lowest']   = df['low'].rolling(LOOKBACK).min().shift(1)
    # ADX
    up_m   = df['high'].diff()
    down_m = -(df['low'].shift(1).diff())
    plus   = np.where((up_m > down_m) & (up_m > 0), up_m, 0.0)
    minus  = np.where((down_m > up_m) & (down_m > 0), down_m, 0.0)
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
def find_entries(df, params):
    total = len(df)
    th    = 0.5 * df['atr']
    ema_f = df['close'] > df['ema_long']
    rsi_f = (df['rsi'] > 50) & (df['close'] > df['highest'] + th)
    atr_f = df['atr'] >= df['atr'].mean()
    adx_f = df['adx'] > ADX_THRESHOLD
    trend = df['close'] > df['ema_hf']
    vol_f = df['volume'] >= params['VOL_MULT'] * df['vol_ma']
    vwap_f= df['close'] > df['vwap_upper']
    time_f= ~df['datetime'].dt.hour.isin(EXCLUDE_HOURS)
    reg_f = df['atr_ratio'] >= params['REGIME_MULT']

    call = ema_f & rsi_f & atr_f & adx_f & trend & vol_f & vwap_f & time_f & reg_f
    put  = (~ema_f) & (df['rsi']<50) & (df['close']<df['lowest']-th) & adx_f & atr_f \
           & (~trend) & vol_f & (df['close']<df['vwap_lower']) & time_f & reg_f

    df['signal'] = np.where(call, 'CALL', np.where(put, 'PUT', None))
    entries = df[df['signal'].notna()].copy()
    entries['idx'] = entries.index
    return entries

def run_backtest(df, params):
    entries = find_entries(df, params)
    equity, banked = INITIAL_USDT, 0.0
    trades = []

    for _, r in entries.iterrows():
        idx = int(r['idx'])
        sig = r['signal']
        epx = df.at[idx,'open']; atr = df.at[idx,'atr']
        entry_dt = df.at[idx,'datetime']
        sl0 = epx - params['SL_ATR_MULT']*atr if sig=='CALL' else epx + params['SL_ATR_MULT']*atr
        tp  = epx + params['TP_ATR_MULT']*atr if sig=='CALL' else epx - params['TP_ATR_MULT']*atr

        # randomize slippage
        vol, vol_ma = df.at[idx,'volume'], df.at[idx,'vol_ma']
        if vol >= params['VOL_MULT']*vol_ma:
            slippage = np.random.uniform(0.0003, 0.0005)
        else:
            slippage = np.random.uniform(0.0005, 0.0008)

        # position sizing
        risk_usd = equity * params['RISK_PCT'] * LEVERAGE
        dist     = abs(epx - sl0)
        raw_ct   = risk_usd/(dist*CONTRACT_SZ) if dist>0 else 0
        rem      = math.floor(raw_ct/CONTRACT_SZ)*CONTRACT_SZ
        if rem < CONTRACT_SZ:
            continue

        peak, moved, scaled = epx, False, False

        # simulate forward up to LOOKBACK bars
        for j in range(idx+1, min(idx+1+LOOKBACK, len(df))):
            h,l = df.at[j,'high'], df.at[j,'low']
            peak = max(peak,h) if sig=='CALL' else min(peak,l)

            # breakeven scale
            if not moved and ((sig=='CALL' and h>=epx+atr) or (sig=='PUT' and l<=epx-atr)):
                sl0, moved = (epx+0.2*atr if sig=='CALL' else epx-0.2*atr), True

            # half exit at 2×ATR
            if not scaled and ((sig=='CALL' and h>=epx+2*atr) or (sig=='PUT' and l<=epx-2*atr)):
                half = rem/2
                ie   = epx*(1+slippage) if sig=='CALL' else epx*(1-slippage)
                ex   = ((epx+2*atr)*(1-slippage) if sig=='CALL' else (epx-2*atr)*(1+slippage))
                pnl_pc = (ex-ie) if sig=='CALL' else (ie-ex)
                pnl_usd= pnl_pc*half*CONTRACT_SZ - (ie+ex)*half*CONTRACT_SZ*TAKER_FEE
                if pnl_usd>0:
                    banked += pnl_usd*0.7; equity += pnl_usd*0.3
                else:
                    equity += pnl_usd
                trades.append({
                    'entry_dt': entry_dt,
                    'exit_dt':  df.at[j,'datetime'],
                    'signal':   sig,
                    'contracts':half,
                    'entry_px': epx,
                    'exit_px':  ex,
                    'pnl_usd':  pnl_usd,
                    'equity':   equity,
                    'banked':   banked
                })
                rem   -= half
                scaled= True

            # trailing/sl & TP exit
            cur_sl = sl0 if not moved else (peak-atr if sig=='CALL' else peak+atr)
            exit_now = ((sig=='CALL' and (l<=cur_sl or h>=tp)) or
                        (sig=='PUT'  and (h>=cur_sl or l<=tp)))
            if exit_now:
                exit_px = cur_sl if ((sig=='CALL' and l<=cur_sl) or (sig=='PUT' and h>=cur_sl)) else tp
                ie      = epx*(1+slippage) if sig=='CALL' else epx*(1-slippage)
                ex      = exit_px*(1-slippage) if sig=='CALL' else exit_px*(1+slippage)
                pnl_pc  = (ex-ie) if sig=='CALL' else (ie-ex)
                pnl_usd = pnl_pc*rem*CONTRACT_SZ - (ie+ex)*rem*CONTRACT_SZ*TAKER_FEE
                if pnl_usd>0:
                    banked += pnl_usd*0.7; equity += pnl_usd*0.3
                else:
                    equity += pnl_usd
                trades.append({
                    'entry_dt': entry_dt,
                    'exit_dt':  df.at[j,'datetime'],
                    'signal':   sig,
                    'contracts':rem,
                    'entry_px': epx,
                    'exit_px':  exit_px,
                    'pnl_usd':  pnl_usd,
                    'equity':   equity,
                    'banked':   banked
                })
                break
        else:
            # force exit at end of window
            ei = min(idx+LOOKBACK, len(df)-1)
            xp = df.at[ei,'close']; dt2 = df.at[ei,'datetime']
            ie = epx*(1+slippage) if sig=='CALL' else epx*(1-slippage)
            ex = xp*(1-slippage) if sig=='CALL' else xp*(1+slippage)
            pnl_pc = (ex-ie) if sig=='CALL' else (ie-ex)
            pnl_usd= pnl_pc*rem*CONTRACT_SZ - (ie+ex)*rem*CONTRACT_SZ*TAKER_FEE
            if pnl_usd>0:
                banked += pnl_usd*0.7; equity += pnl_usd*0.3
            else:
                equity += pnl_usd
            trades.append({
                'entry_dt': entry_dt,
                'exit_dt':  dt2,
                'signal':   sig,
                'contracts':rem,
                'entry_px': epx,
                'exit_px':  xp,
                'pnl_usd':  pnl_usd,
                'equity':   equity,
                'banked':   banked
            })

    # metrics
    if not trades:
        return {'sortino': -np.inf, 'roi': -np.inf, 'win_rate': 0.0, 'max_dd': np.inf}

    df_tr = pd.DataFrame(trades)
    df_tr['total'] = df_tr['equity'] + df_tr['banked']
    roi       = df_tr['total'].iat[-1]/INITIAL_USDT - 1
    rets      = df_tr['total'].pct_change().fillna(0)
    downside  = rets[rets<0]
    sortino   = rets.mean()/downside.std() if downside.std()>0 else np.nan
    win_rate  = (df_tr['pnl_usd']>0).mean()
    dd        = (df_tr['total'] - df_tr['total'].cummax()) / df_tr['total'].cummax()
    max_dd    = dd.min()

    return {'sortino': sortino, 'roi': roi, 'win_rate': win_rate, 'max_dd': max_dd}

# ── Main pipeline with multi‐objective Optuna ───────────────────────
def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # 1) Load & slice
    df5  = limit_to_first_n_days(load_and_normalize(DATA_5M), LIMITER_DAYS)
    df1h = limit_to_first_n_days(load_and_normalize(DATA_1H), LIMITER_DAYS)
    tr_days, ts_days = split_by_days(df5, TRAIN_PCT)
    df5_tr, df5_ts   = slice_days(df5, tr_days), slice_days(df5, ts_days)
    df1h_tr, df1h_ts = slice_days(df1h, tr_days), slice_days(df1h, ts_days)

    # 2) Multi‐objective study
    def objective(trial):
        params = {
            p: trial.suggest_int(p, *param_bounds[p]) if p in ('ATR_WINDOW','EMA_LONG')
               else trial.suggest_float(p, *param_bounds[p])
            for p in param_bounds
        }
        b5 = preprocess_5m(df5_tr, params, preprocess_1h(df1h_tr, params))
        metrics = run_backtest(b5, params)

        # must-trade & NaN‐safe
        if np.isnan(metrics['sortino']) or metrics['roi']<=-1 or metrics['win_rate']==0:
            return (-np.inf, -np.inf, 0.0, np.inf)

        return (
            metrics['sortino'],
            metrics['roi'],
            metrics['win_rate'],
            metrics['max_dd']
        )

    study = optuna.create_study(
        directions=['maximize','maximize','maximize','minimize'],
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=N_OPT_TRIALS)

    # 3) Retest Pareto‐optimal on test slice and export top K
    pareto = study.best_trials
    results = []
    for t in pareto:
        combo = t.params
        b5_ts = preprocess_5m(df5_ts, combo, preprocess_1h(df1h_ts, combo))
        m = run_backtest(b5_ts, combo)
        if m['win_rate']>0 and not np.isinf(m['max_dd']):
            rec = combo.copy()
            rec.update(m)
            results.append(rec)
    df_out = pd.DataFrame(sorted(results,
                     key=lambda r: (r['sortino'], r['roi'], r['win_rate'], -r['max_dd']),
                     reverse=True)[:TOP_K])
    df_out.to_csv('top_optuna_combos_filtered.csv', index=False)
    logger.info(f"Exported {len(df_out)} combos.")

if __name__ == "__main__":
    main()
