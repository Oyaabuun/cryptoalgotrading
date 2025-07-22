import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

# ── Parameters ────────────────────────────────────────────────
LOOKBACK        = 20
ATR_WINDOW      = 14
EMA_LONG        = 50
RISK_PCT        = 0.01
SL_ATR_MULT     = 1.5
TP_ATR_MULT     = 3.0
MAX_HOLD_BARS   = 12
INITIAL_CAPITAL = 5000.0
USD_INR_RATE    = 82.0

# ── Fees & Slippage ───────────────────────────────────────────
TAKER_FEE      = 0.0005   # 0.05% per side (entry + exit)
SLIPPAGE_PCT   = 0.0002   # 0.02% slippage per side

DATA_FILE       = "data/BTC_PERPETUAL_5m_3months.csv"

# ── Indicator Precompute ──────────────────────────────────────
def preprocess(df):
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_WINDOW).mean()
    df['ema_short'] = close.ewm(span=LOOKBACK, adjust=False).mean()
    df['ema_long']  = close.ewm(span=EMA_LONG, adjust=False).mean()
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up   = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    df['rsi'] = 100 - (100/(1 + roll_up/roll_down))
    df['highest'] = df['high'].rolling(LOOKBACK).max().shift(1)
    df['lowest']  = df['low'].rolling(LOOKBACK).min().shift(1)
    return df.dropna().reset_index(drop=True)

# ── Identify Entry Bars ──────────────────────────────────────
def find_entries(df):
    df = df.copy()
    df['threshold'] = 0.5 * df['atr']
    bull = (
        (df['close'] > df['ema_long']) &
        (df['rsi'] > 50) &
        (df['close'] > df['highest'] + df['threshold'])
    )
    bear = (
        (df['close'] < df['ema_long']) &
        (df['rsi'] < 50) &
        (df['close'] < df['lowest'] - df['threshold'])
    )
    df['signal'] = np.where(bull, 'CALL',
                     np.where(bear, 'PUT', None))
    return df[df['signal'].notnull()].copy()

# ── Simulate One Trade ────────────────────────────────────────
def simulate_trade(args):
    i, row, df = args
    sig      = row['signal']
    entry_px = row['open']
    atr      = row['atr']

    # SL/TP levels
    sl_price = entry_px - SL_ATR_MULT * atr if sig=='CALL' else entry_px + SL_ATR_MULT * atr
    tp_price = entry_px + TP_ATR_MULT * atr if sig=='CALL' else entry_px - TP_ATR_MULT * atr

    # Position sizing (risk 1% of INITIAL_CAPITAL)
    risk_inr = INITIAL_CAPITAL * RISK_PCT
    sl_dist  = abs(entry_px - sl_price)
    qty      = risk_inr / (sl_dist * USD_INR_RATE)

    # Walk-forward to find exit
    exit_px = None
    for j in range(i+1, min(i+1+MAX_HOLD_BARS, len(df))):
        h, l = df.at[j,'high'], df.at[j,'low']
        if sig=='CALL' and l <= sl_price:
            exit_px = sl_price; exit_idx = j; break
        if sig=='CALL' and h >= tp_price:
            exit_px = tp_price; exit_idx = j; break
        if sig=='PUT'  and h >= sl_price:
            exit_px = sl_price; exit_idx = j; break
        if sig=='PUT'  and l <= tp_price:
            exit_px = tp_price; exit_idx = j; break
    if exit_px is None:
        exit_idx = min(i+MAX_HOLD_BARS, len(df)-1)
        exit_px  = df.at[exit_idx,'close']

    # Apply slippage to execution prices
    if sig == 'CALL':
        exec_entry = entry_px * (1 + SLIPPAGE_PCT)
        exec_exit  = exit_px  * (1 - SLIPPAGE_PCT)
    else:  # PUT
        exec_entry = entry_px * (1 - SLIPPAGE_PCT)
        exec_exit  = exit_px  * (1 + SLIPPAGE_PCT)

    # Raw PnL in USD
    pnl_usd = (exec_exit - exec_entry) * qty if sig=='CALL' else (exec_entry - exec_exit) * qty

    # Trading fees (taker on both entry & exit)
    fee_usd = (exec_entry * qty + exec_exit * qty) * TAKER_FEE

    # Net PnL after fees
    pnl_usd_net = pnl_usd - fee_usd
    pnl_inr     = pnl_usd_net * USD_INR_RATE

    return {
        "Trade ID":    i,
        "Signal":      sig,
        "Entry Time":  pd.to_datetime(row['timestamp'], unit='ms'),
        "Entry Price": round(entry_px,2),
        "Exit Time":   pd.to_datetime(df.at[exit_idx,'timestamp'], unit='ms'),
        "Exit Price":  round(exit_px,2),
        "Qty":         round(qty,4),
        "SL":          round(sl_price,2),
        "TP":          round(tp_price,2),
        "ExecEntry":   round(exec_entry,2),
        "ExecExit":    round(exec_exit,2),
        "Fee (USD)":   round(fee_usd,4),
        "PnL (USD)":   round(pnl_usd_net,2),
        "PnL (INR )":  round(pnl_inr,2),
    }

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    df       = pd.read_csv(DATA_FILE)
    df       = preprocess(df)
    entries  = find_entries(df)

    args = [(idx, entries.iloc[k], df)
            for k, idx in enumerate(entries.index)]

    with Pool(cpu_count()) as pool:
        trades = pool.map(simulate_trade, args)

    trades_df = pd.DataFrame(trades).sort_values("Entry Time")
    final_cap = INITIAL_CAPITAL + trades_df["PnL (INR )"].sum()

    # Save and report
    out_csv = "results_backtest_with_fees_slippage.csv"
    trades_df.to_csv(out_csv, index=False)
    print(f"Backtest done: {len(trades_df)} trades")
    print(f"Final capital: ₹{final_cap:.2f}  "
          f"(ROI: {(final_cap/INITIAL_CAPITAL-1)*100:.2f}%)")

