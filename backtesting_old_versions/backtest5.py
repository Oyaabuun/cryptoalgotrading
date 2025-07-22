# In this version the data was spilt into train and test first 
# 70 percent of days were backtesting 
# and recent 30 % days were used for testing on historical data 
# and also hyper parameter tuning was introduced 
import math
import pandas as pd
import numpy as np
from itertools import product

# ── Static strategy parameters ─────────────────────────────
LOOKBACK = 20
ADX_WINDOW = 14
ADX_THRESHOLD = 20
RISK_PCT = 0.01
INITIAL_USDT = 58.0
LEVERAGE = 10.0
CONTRACT_SZ = 1.0
TAKER_FEE = 0.0005
SLIPPAGE = 0.0002
DATA_FILE = "data/BTC_PERPETUAL_5m_3months.csv"

# ── Hyperparameter grid to test ────────────────────────────
param_grid = {
    'ATR_WINDOW': [14, 20],
    'EMA_LONG': [50, 100],
    'SL_ATR_MULT': [1.0, 1.25],
    'TP_ATR_MULT': [4.0, 5.0],
}

# ── Raw datetime parsing ────────────────────────────────────
def prepare_df_raw(df):
    if 'open_time' in df:
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    elif 'timestamp' in df:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        df['datetime'] = pd.to_datetime(df['datetime'])
    return df

# ── Preprocessing & Indicators ───────────────────────────────
def preprocess(df, params):
    # Ensure integer window sizes
    atr_win = int(params['ATR_WINDOW'])
    ema_l = int(params['EMA_LONG'])

    # Sort and basic columns
    df = df.sort_values('datetime').reset_index(drop=True)
    df['prev_close'] = df['close'].shift(1)

    # True range and ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['prev_close']).abs(),
        (df['low'] - df['prev_close']).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(atr_win).mean()

    # Long EMA
    df['ema_long'] = df['close'].ewm(span=ema_l, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    df['rsi'] = 100 - 100 / (1 + up.rolling(atr_win).mean() / down.rolling(atr_win).mean())

    # Breakout levels
    df['highest'] = df['high'].rolling(LOOKBACK).max().shift(1)
    df['lowest'] = df['low'].rolling(LOOKBACK).min().shift(1)

    # ADX
    up_m = df['high'].diff()
    down_m = -(df['low'].shift(1).diff())
    plus = np.where((up_m > down_m) & (up_m > 0), up_m, 0.0)
    minus = np.where((down_m > up_m) & (down_m > 0), down_m, 0.0)
    sm_tr = tr.ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    sm_p = pd.Series(plus).ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    sm_m = pd.Series(minus).ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    df['plus_di'] = 100 * sm_p / sm_tr
    df['minus_di'] = 100 * sm_m / sm_tr
    dx = 100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'])
    df['adx'] = dx.ewm(alpha=1/ADX_WINDOW, adjust=False).mean()

    return df.dropna().reset_index(drop=True)

# ── Entry signal detection ───────────────────────────────────
def find_entries(df, params):
    m_atr = df['atr'].mean()
    th = 0.5 * df['atr']
    bull = (
        (df['close'] > df['ema_long']) &
        (df['rsi'] > 50) &
        (df['close'] > df['highest'] + th) &
        (df['adx'] > ADX_THRESHOLD) &
        (df['atr'] >= m_atr)
    )
    bear = (
        (df['close'] < df['ema_long']) &
        (df['rsi'] < 50) &
        (df['close'] < df['lowest'] - th) &
        (df['adx'] > ADX_THRESHOLD) &
        (df['atr'] >= m_atr)
    )
    df['signal'] = np.where(bull, 'CALL', np.where(bear, 'PUT', None))
    entries = df[df['signal'].notnull()].copy()
    entries['idx'] = entries.index
    return entries.reset_index(drop=True)

# ── Backtest engine ─────────────────────────────────────────
def run_backtest(df, params):
    entries = find_entries(df, params)
    equity, banked = INITIAL_USDT, 0.0
    trades = []

    for _, row in entries.iterrows():
        sig = row['signal']
        epx = row['open']
        atr = row['atr']
        idx = int(row['idx'])
        entry_dt = df.at[idx, 'datetime']

        sl_price = epx - params['SL_ATR_MULT'] * atr if sig == 'CALL' else epx + params['SL_ATR_MULT'] * atr
        tp_price = epx + params['TP_ATR_MULT'] * atr if sig == 'CALL' else epx - params['TP_ATR_MULT'] * atr

        risk_usd = equity * RISK_PCT * LEVERAGE
        sl_dist = abs(epx - sl_price)
        raw_ct = risk_usd / (sl_dist * CONTRACT_SZ) if sl_dist > 0 else 0
        contracts = math.floor(raw_ct / 0.001) * 0.001 if raw_ct >= 0.001 else 0
        if contracts == 0:
            continue

        exit_px, exit_idx = None, None
        for j in range(idx + 2, min(idx + 1 + LOOKBACK, len(df))):
            h, l = df.at[j, 'high'], df.at[j, 'low']
            if sig == 'CALL' and l <= sl_price:
                exit_px, exit_idx = sl_price, j
                break
            if sig == 'CALL' and h >= tp_price:
                exit_px, exit_idx = tp_price, j
                break
            if sig == 'PUT' and h >= sl_price:
                exit_px, exit_idx = sl_price, j
                break
            if sig == 'PUT' and l <= tp_price:
                exit_px, exit_idx = tp_price, j
                break
        if exit_px is None:
            exit_idx = min(idx + LOOKBACK, len(df) - 1)
            exit_px = df.at[exit_idx, 'close']
        exit_dt = df.at[exit_idx, 'datetime']

        ie = epx * (1 + SLIPPAGE) if sig == 'CALL' else epx * (1 - SLIPPAGE)
        ex = exit_px * (1 - SLIPPAGE) if sig == 'CALL' else exit_px * (1 + SLIPPAGE)
        pnl_pc = ((ex - ie) if sig == 'CALL' else (ie - ex)) * CONTRACT_SZ
        pnl_usd = pnl_pc * contracts
        fee = (ie + ex) * contracts * CONTRACT_SZ * TAKER_FEE
        pnl_usd -= fee

        if pnl_usd > 0:
            banked += pnl_usd * 0.70
            equity += pnl_usd * 0.30
        else:
            equity += pnl_usd

        trades.append({
            'entry_dt': entry_dt,
            'exit_dt': exit_dt,
            'signal': sig,
            'contracts': contracts,
            'entry_px': epx,
            'exit_px': exit_px,
            'pnl_usd': pnl_usd,
            'equity': equity,
            'banked': banked
        })

    df_trades = pd.DataFrame(trades)
    df_trades['total'] = df_trades['equity'] + df_trades['banked']

    # Compute metrics
    total_return = df_trades['total'].iloc[-1] / INITIAL_USDT - 1
    days = (df_trades['exit_dt'].iloc[-1] - df_trades['entry_dt'].iloc[0]).total_seconds() / 86400
    cagr = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else np.nan
    rets = df_trades['total'].pct_change().fillna(0)
    sharpe = rets.mean() / rets.std() if rets.std() else np.nan
    dd = (df_trades['total'] - df_trades['total'].cummax()) / df_trades['total'].cummax()
    max_dd = dd.min()
    win_rate = (df_trades['pnl_usd'] > 0).mean()
    duration = (df_trades['exit_dt'] - df_trades['entry_dt']).dt.total_seconds() / 60

    metrics = {
        'total_return_pct': total_return * 100,
        'annualized_return_pct': cagr * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd * 100,
        'win_rate_pct': win_rate * 100,
        'num_trades': len(df_trades),
        'profit_factor': (
            df_trades.loc[df_trades['pnl_usd'] > 0, 'pnl_usd'].sum() /
            (-df_trades.loc[df_trades['pnl_usd'] < 0, 'pnl_usd'].sum())
            if (df_trades['pnl_usd'] < 0).any() else np.nan
        ),
        'avg_win_usd': df_trades.loc[df_trades['pnl_usd'] > 0, 'pnl_usd'].mean(),
        'avg_loss_usd': df_trades.loc[df_trades['pnl_usd'] < 0, 'pnl_usd'].mean(),
        'max_win_usd': df_trades['pnl_usd'].max(),
        'max_loss_usd': df_trades['pnl_usd'].min(),
        'avg_duration_min': duration.mean(),
        'final_balance_usdt': df_trades['total'].iloc[-1]
    }
    return df_trades, metrics

# ── Data splitting ───────────────────────────────────────────
def split_data(df):
    days = df['datetime'].dt.normalize().drop_duplicates().sort_values()
    total_days = len(days)
    back_days_count = int(round(total_days * 0.7))
    back_days = set(days.iloc[:back_days_count])
    test_days = set(days.iloc[back_days_count:])

    df_back = df[df['datetime'].dt.normalize().isin(back_days)].reset_index(drop=True)
    df_test = df[df['datetime'].dt.normalize().isin(test_days)].reset_index(drop=True)
    return df_back, df_test

# ── Main execution ───────────────────────────────────────────
def main():
    # Load and split raw data
    df_raw = pd.read_csv(DATA_FILE)
    df_raw = prepare_df_raw(df_raw)
    df_back_raw, df_test_raw = split_data(df_raw)

    # Grid search on backtest slice
    grid_results = []
    combos = [dict(zip(param_grid.keys(), vals)) for vals in product(*param_grid.values())]
    for params in combos:
        df_back = preprocess(df_back_raw.copy(), params)
        _, metrics = run_backtest(df_back, params)
        grid_results.append({**params, **metrics})
    grid_df = pd.DataFrame(grid_results)
    grid_df.to_csv('best_parameters.csv', index=False)

    # Select best by Sharpe
    best_row = grid_df.loc[grid_df['sharpe_ratio'].idxmax()]
    best_params = {
        'ATR_WINDOW': int(best_row['ATR_WINDOW']),
        'EMA_LONG': int(best_row['EMA_LONG']),
        'SL_ATR_MULT': float(best_row['SL_ATR_MULT']),
        'TP_ATR_MULT': float(best_row['TP_ATR_MULT'])
    }
    print("Best hyperparameters:", best_params)

    # Run backtest and testrun with best params
    df_back = preprocess(df_back_raw.copy(), best_params)
    back_trades, back_metrics = run_backtest(df_back, best_params)
    back_trades.to_csv('backtest_with_bst_para.csv', index=False)
    pd.DataFrame([back_metrics]).to_csv('backtest_results.csv', index=False)

    df_test = preprocess(df_test_raw.copy(), best_params)
    test_trades, test_metrics = run_backtest(df_test, best_params)
    test_trades.to_csv('testrun_with_bst_para.csv', index=False)
    pd.DataFrame([test_metrics]).to_csv('testrun_results.csv', index=False)

if __name__ == "__main__":
    main()
