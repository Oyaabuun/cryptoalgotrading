#!/usr/bin/env python3
"""
Filtered ML‐driven hyperparameter optimization for the Trend‐Regime Breakout strategy.
– Samples random combos, backtests them (train slice), records Sharpe & entry‐count
– Filters out “no‐trade” (num_entries < N_MIN_ENTRIES) and infinite‐Sharpe samples
– Trains HistGBM & XGBoost on the remaining data
– Evaluates models → metrics.csv
– Proposes top‐K combos, re‐backtests on test slice, and only keeps those with sufficient entries → top_*.csv


Since we added the “no-trade” filter (requiring ≥ 10 entries) before fitting, the surrogate’s apparent fit drops a bit—but you’re now training only on real, tradable regimes:

Model	R²	MAE	RMSE
HistGBM	0.4563	0.0798	0.1199
XGBoost	0.3207	0.0851	0.1340

R² is down from ~0.60 → ~0.46 (HistGBM) and ~0.52 → ~0.32 (XGBoost) because you’ve removed all the infinite-Sharpe “no-trade” points that made the regression easier.

MAE (≈ 0.08) & RMSE (≈ 0.12) are still in a reasonable range given the tighter domain.

On the test slice (last 180 days), every top-5 combo now also passes the entries ≥ 10 filter—so none of your proposals are “empty” strategies any more. For example:

csv
Copy
Edit
ATR_WINDOW,EMA_LONG,SL_ATR_MULT,TP_ATR_MULT,VOL_MULT,RISK_PCT,REGIME_MULT,sharpe,num_entries
38,188,1.8884,4.7050,1.2265,0.01971,1.7583,1.4549,12
34,199,0.5723,7.8630,1.8138,0.01754,1.6736,1.5534,11
38,170,1.6107,2.8571,2.5069,0.01561,1.4878,1.3786,20
40,183,1.9156,7.7608,2.0429,0.01957,1.6360,1.2926,13
38,199,1.8997,5.8554,1.8425,0.01309,1.6785,1.9716,12
Each of those top-5 HistGBM picks fires at least 11–20 times in 6 months and delivers out-of-sample Sharpe between ~1.29 and ~1.97. That means:

No more “silent” strategies—every top combo actually trades.

You lose a bit of raw fit on the surrogate, but you gain in practical robustness.

HistGBM remains your best regressor (higher R² & lower error than XGBoost), and its top-ranked hyperparameters continue to yield test Sharpe well above 1.

Next steps
Trust the filtered top-5 for live/paper testing—each has ≥ 10 entries and test Sharpe ≥ 1.29.

If you need even tighter robustness, bump N_MIN_ENTRIES higher or add a small classification head on “will trade” vs. “won’t trade.”

You can also re-run a local search around the best cluster (e.g. ATR≈38, EMA≈188–199) with a higher sampling density.
"""

import os, math
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────
DATA_5M          = "data/BTCUSDT_5m_5year.csv"
DATA_1H          = "data/BTCUSDT_1h_5year.csv"
LIMITER_DAYS     = 1460
TRAIN_PCT        = 0.7
N_SAMPLES        = 500
TOP_K            = 5
N_MIN_ENTRIES    = 10   # filter threshold
param_bounds     = {
    'ATR_WINDOW':  (5, 50),
    'EMA_LONG':    (20, 200),
    'SL_ATR_MULT': (0.5, 2.0),
    'TP_ATR_MULT': (2.0, 8.0),
    'VOL_MULT':    (1.0, 3.0),
    'RISK_PCT':    (0.001, 0.02),
    'REGIME_MULT': (0.5, 2.0),
}
LOOKBACK       = 20
ADX_WINDOW     = 14
ADX_THRESHOLD  = 20
INITIAL_USDT   = 58.0
LEVERAGE       = 10.0
CONTRACT_SZ    = 0.001
TAKER_FEE      = 0.0005
SLIPPAGE       = 0.0005
VWAP_BAND_PCT  = 0.001
EXCLUDE_HOURS  = {0}

# ── I/O & date‐slicing helpers ─────────────────────────────────
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
    last = days.iloc[-n:]
    return df[df['datetime'].dt.normalize().isin(last)].reset_index(drop=True)

def split_by_days(df, pct):
    days = df['datetime'].dt.normalize().drop_duplicates().sort_values()
    cut  = int(len(days)*pct)
    return set(days.iloc[:cut]), set(days.iloc[cut:])

# ── Helpers ─────────────────────────────────────────────────
def preprocess_1h(df1h, params):
    df = df1h.copy()
    df['ema_hf'] = df['close'].ewm(span=int(params['EMA_LONG']), adjust=False).mean()
    return df[['datetime','ema_hf']]

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

    # Slow ATR for regime filter
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

    # VWAP & bands
    df['date']     = df['datetime'].dt.date
    typ           = (df['high'] + df['low'] + df['close'])/3
    df['cum_vp']  = typ.mul(df['volume']).groupby(df['date']).cumsum()
    df['cum_vol'] = df['volume'].groupby(df['date']).cumsum()
    df['vwap']    = df['cum_vp'] / df['cum_vol']
    df['vwap_upper'] = df['vwap'] * (1 + VWAP_BAND_PCT)
    df['vwap_lower'] = df['vwap'] * (1 - VWAP_BAND_PCT)

    # Merge 1h EMA
    df['hour'] = df['datetime'].dt.floor('h')
    df = df.merge(df_hf.rename(columns={'datetime':'hour'}), on='hour', how='left')
    return df.dropna().reset_index(drop=True)

def find_entries(df5, params):
    df = df5.copy()
    total = len(df)
    print(f"\n=== PARAMS {params} ===")
    print(f"Total bars: {total}")

    th     = 0.5 * df['atr']
    ema_f  = df['close'] > df['ema_long']
    rsi_f  = (df['rsi'] > 50) & (df['close'] > df['highest'] + th)
    atr_f  = df['atr'] >= df['atr'].mean()
    adx_f  = df['adx'] > ADX_THRESHOLD
    trend_f= df['close'] > df['ema_hf']
    vol_f  = df['volume'] >= params['VOL_MULT'] * df['vol_ma']
    vwap_f = df['close'] > df['vwap_upper']
    time_f = ~df['datetime'].dt.hour.isin(EXCLUDE_HOURS)
    reg_f  = df['atr_ratio'] >= params['REGIME_MULT']

    # Diagnostics
    print(f"After EMA:       {ema_f.sum():4d}/{total}")
    print(f"After RSI+BO:    {rsi_f.sum():4d}/{total}")
    print(f"After ADX+ATR:   {(atr_f&adx_f).sum():4d}/{total}")
    print(f"After 1h trend:  {trend_f.sum():4d}/{total}")
    print(f"After vol spike: {vol_f.sum():4d}/{total}")
    print(f"After VWAP band: {vwap_f.sum():4d}/{total}")
    print(f"After session:   {time_f.sum():4d}/{total}")
    print(f"After regime:    {reg_f.sum():4d}/{total}")

    # Build CALL/PUT masks
    call = ema_f & rsi_f & atr_f & adx_f & trend_f & vol_f & vwap_f & time_f & reg_f
    put  = (
        (df['close'] < df['ema_long']) &
        (df['rsi']   < 50) &
        (df['close'] < df['lowest'] - th) &
        adx_f & atr_f &
        (df['close'] < df['ema_hf']) &
        vol_f &
        (df['close'] < df['vwap_lower']) &
        time_f &
        reg_f
    )

    df['signal'] = np.where(call, 'CALL', np.where(put, 'PUT', None))
    entries = df[df['signal'].notnull()].copy()
    print(f"After combine:   {len(entries):4d} entries\n")
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

        # position sizing
        risk_usd = equity * params['RISK_PCT'] * LEVERAGE
        dist     = abs(epx - sl)
        raw_ct   = risk_usd/(dist*CONTRACT_SZ) if dist>0 else 0
        cnt      = math.floor(raw_ct/0.001)*0.001
        if cnt < 0.001: continue

        rem, peak, moved, scaled = cnt, epx, False, False

        for j in range(idx+1, min(idx+1+LOOKBACK, len(df5))):
            h, l = df5.at[j,'high'], df5.at[j,'low']
            peak = max(peak, h) if sig=='CALL' else min(peak, l)

            # move to breakeven + 0.2 ATR
            if not moved and ((sig=='CALL' and h>=epx+atr) or (sig=='PUT' and l<=epx-atr)):
                sl, moved = (epx+0.2*atr if sig=='CALL' else epx-0.2*atr), True

            # scale out half at 2 ATR
            if not scaled and ((sig=='CALL' and h>=epx+2*atr) or (sig=='PUT' and l<=epx-2*atr)):
                half = rem/2
                ie   = epx*(1+SLIPPAGE) if sig=='CALL' else epx*(1-SLIPPAGE)
                ex   = ((epx+2*atr)*(1-SLIPPAGE) if sig=='CALL'
                        else (epx-2*atr)*(1+SLIPPAGE))
                pnl_pc  = (ex-ie) if sig=='CALL' else (ie-ex)
                pnl_usd = pnl_pc*half*CONTRACT_SZ
                fee     = (ie+ex)*half*CONTRACT_SZ*TAKER_FEE
                pnl_usd-=fee
                if pnl_usd>0:
                    banked+=pnl_usd*0.7; equity+=pnl_usd*0.3
                else:
                    equity+=pnl_usd
                trades.append({
                    'entry_dt':entry_dt, 'exit_dt':df5.at[j,'datetime'],
                    'signal':sig, 'contracts':half,
                    'entry_px':epx, 'exit_px':epx+(2*atr if sig=='CALL' else -2*atr),
                    'pnl_usd':pnl_usd, 'equity':equity, 'banked':banked
                })
                rem-=half; scaled=True

            # trailing stop
            cur_sl = sl if not moved else (peak-atr if sig=='CALL' else peak+atr)
            exit_now = (
                (sig=='CALL' and (l<=cur_sl or h>=tp)) or
                (sig=='PUT'  and (h>=cur_sl or l<=tp))
            )
            if exit_now:
                exit_px = cur_sl if ((sig=='CALL' and l<=cur_sl) or (sig=='PUT' and h>=cur_sl)) else tp
                ie      = epx*(1+SLIPPAGE) if sig=='CALL' else epx*(1-SLIPPAGE)
                ex      = exit_px*(1-SLIPPAGE) if sig=='CALL' else exit_px*(1+SLIPPAGE)
                pnl_pc  = (ex-ie) if sig=='CALL' else (ie-ex)
                pnl_usd = pnl_pc*rem*CONTRACT_SZ
                fee     = (ie+ex)*rem*CONTRACT_SZ*TAKER_FEE
                pnl_usd-=fee
                if pnl_usd>0:
                    banked+=pnl_usd*0.7; equity+=pnl_usd*0.3
                else:
                    equity+=pnl_usd
                trades.append({
                    'entry_dt':entry_dt, 'exit_dt':df5.at[j,'datetime'],
                    'signal':sig, 'contracts':rem,
                    'entry_px':epx, 'exit_px':exit_px,
                    'pnl_usd':pnl_usd, 'equity':equity, 'banked':banked
                })
                break
        else:
            # time stop at LOOKBACK
            ei = min(idx+LOOKBACK, len(df5)-1)
            xp, dt2 = df5.at[ei,'close'], df5.at[ei,'datetime']
            ie = epx*(1+SLIPPAGE) if sig=='CALL' else epx*(1-SLIPPAGE)
            ex = xp*(1-SLIPPAGE)   if sig=='CALL' else xp*(1+SLIPPAGE)
            pnl_pc  = (ex-ie) if sig=='CALL' else (ie-ex)
            pnl_usd = pnl_pc*rem*CONTRACT_SZ
            fee     = (ie+ex)*rem*CONTRACT_SZ*TAKER_FEE
            pnl_usd-=fee
            if pnl_usd>0:
                banked+=pnl_usd*0.7; equity+=pnl_usd*0.3
            else:
                equity+=pnl_usd
            trades.append({
                'entry_dt':entry_dt, 'exit_dt':dt2,
                'signal':sig, 'contracts':rem,
                'entry_px':epx, 'exit_px':xp,
                'pnl_usd':pnl_usd, 'equity':equity, 'banked':banked
            })

    df_tr = pd.DataFrame(trades)
    if df_tr.empty:
        return df_tr, {
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

    # compute metrics
    df_tr['total'] = df_tr['equity'] + df_tr['banked']
    tot   = df_tr['total'].iloc[-1]/INITIAL_USDT - 1
    days  = (df_tr['exit_dt'].iloc[-1] - df_tr['entry_dt'].iloc[0]).total_seconds()/86400
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

# ── ML sampling & filtering ──────────────────────────────────────
def sample_params(n, bounds):
    rng = np.random.RandomState(42)
    out = []
    for _ in range(n):
        c = {}
        for p,(lo,hi) in bounds.items():
            v = rng.uniform(lo, hi)
            if p in ('ATR_WINDOW','EMA_LONG'):
                v = int(v)
            c[p] = v
        out.append(c)
    return out

def evaluate_combo(combo, df5, df1h):
    hf = preprocess_1h(df1h, combo)
    b5 = preprocess_5m(df5.copy(), combo, hf)
    entries = find_entries(b5, combo)
    num_entries = len(entries)
    _, mets   = run_backtest(b5, combo)
    sharpe    = mets['sharpe_ratio']
    return {**combo, 'sharpe': sharpe, 'num_entries': num_entries}

def main():
    # 1) load & slice data
    df5  = limit_to_last_n_days(load_and_normalize(DATA_5M), LIMITER_DAYS)
    df1h = limit_to_last_n_days(load_and_normalize(DATA_1H), LIMITER_DAYS)
    tr_days, ts_days = split_by_days(df5, TRAIN_PCT)
    slice_days = lambda df, ds: df[df['datetime'].dt.normalize().isin(ds)].reset_index(drop=True)
    df5_tr, df5_ts   = slice_days(df5, tr_days), slice_days(df5, ts_days)
    df1h_tr, df1h_ts = slice_days(df1h, tr_days), slice_days(df1h, ts_days)

    # 2) sample & backtest (train slice)
    recs = []
    samples = sample_params(N_SAMPLES, param_bounds)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        futures = {exe.submit(evaluate_combo, c, df5_tr, df1h_tr): c for c in samples}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Sampling"):
            rec = fut.result()
            recs.append(rec)

    df_samp = pd.DataFrame(recs)

    # 3) FILTER: remove infinite/NaN Sharpe and too‐few entries
    df_samp = df_samp.replace([np.inf, -np.inf], np.nan)
    df_samp = df_samp.dropna(subset=['sharpe']).reset_index(drop=True)
    df_samp = df_samp[df_samp['num_entries'] >= N_MIN_ENTRIES].reset_index(drop=True)

    df_samp.to_csv('ml_training_data_filtered.csv', index=False)

    # 4) fit surrogate models
    X = df_samp.drop(columns=['sharpe','num_entries'])
    y = df_samp['sharpe']
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'HistGBM': HistGradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(tree_method='hist', eval_metric='rmse',
                                use_label_encoder=False, random_state=42)
    }
    metrics = []
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        p = mdl.predict(X_val)
        metrics.append({
            'model': name,
            'r2':    r2_score(y_val, p),
            'mae':   mean_absolute_error(y_val, p),
            'rmse':  math.sqrt(mean_squared_error(y_val, p))
        })
    pd.DataFrame(metrics).to_csv('metrics.csv', index=False)

    # 5) propose top‐K & re‐test on test slice
    for name, mdl in models.items():
        pool = sample_params(500, param_bounds)
        dfp  = pd.DataFrame(pool)
        dfp['pred_sharpe'] = mdl.predict(dfp)
        top  = dfp.nlargest(TOP_K*2, 'pred_sharpe').drop(columns=['pred_sharpe'])

        # re-backtest and filter by num_entries
        out = []
        for combo in top.to_dict('records'):
            rec = evaluate_combo(combo, df5_ts, df1h_ts)
            if rec['num_entries'] >= N_MIN_ENTRIES:
                out.append(rec)
            if len(out) >= TOP_K:
                break

        pd.DataFrame(out).to_csv(f'top_{name}_combos_filtered.csv', index=False)

    print("Done. Check:")
    print(" • ml_training_data_filtered.csv")
    print(" • metrics.csv")
    print(" • top_<model>_combos_filtered.csv")

if __name__ == "__main__":
    main()
