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

DATA_FILE    = "data/BTC_PERPETUAL_5m_1months.csv"
OUTPUT_FILE  = "results_backtest_perp_fixed.csv"

# ── Fees & Slippage ───────────────────────────────────────────
SLIPPAGE_PCT = 0.0002   # 0.02% per side
TAKER_FEE    = 0.0005   # 0.05% per side

def preprocess(df):
    high, low, close = df['high'], df['low'], df['close']
    prev = close.shift(1)
    tr = pd.concat([
        high-low,
        (high-prev).abs(),
        (low-prev).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_WINDOW).mean()
    df['ema_long'] = close.ewm(span=EMA_LONG, adjust=False).mean()
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi'] = 100 - 100/(1 + up.rolling(14).mean()/down.rolling(14).mean())
    df['highest'] = high.rolling(LOOKBACK).max().shift(1)
    df['lowest']  = low.rolling(LOOKBACK).min().shift(1)
    return df.dropna().reset_index(drop=True)

def find_entries(df):
    df = df.copy()
    df['threshold'] = 0.5*df['atr']
    bull = (
      (df['close']>df['ema_long']) &
      (df['rsi']>50) &
      (df['close']>df['highest']+df['threshold'])
    )
    bear = (
      (df['close']<df['ema_long']) &
      (df['rsi']<50) &
      (df['close']<df['lowest']-df['threshold'])
    )
    df['signal'] = np.where(bull,'CALL', np.where(bear,'PUT',None))
    e = df[df['signal'].notnull()].copy()
    e['orig_idx'] = e.index
    return e.reset_index(drop=True)

def simulate_trade(args):
    i, row, df, pot, bank = args
    sig      = row['signal']
    entry_px = row['open']
    atr      = row['atr']
    slp      = SLIPPAGE_PCT

    sl_price = entry_px - SL_ATR_MULT*atr if sig=='CALL' else entry_px + SL_ATR_MULT*atr
    tp_price = entry_px + TP_ATR_MULT*atr if sig=='CALL' else entry_px - TP_ATR_MULT*atr

    # **size on current pot, not initial**
    risk_inr = pot * RISK_PCT
    sl_dist  = abs(entry_px - sl_price)
    qty      = risk_inr / (sl_dist * USD_INR_RATE)
    qty      = max(1e-6, qty)   # avoid zero

    # find exit
    for j in range(i+1, min(i+1+MAX_HOLD_BARS, len(df))):
        h,l = df.at[j,'high'], df.at[j,'low']
        if   sig=='CALL' and l<=sl_price:
            exit_px, ei = sl_price, j; break
        elif sig=='CALL' and h>=tp_price:
            exit_px, ei = tp_price, j; break
        elif sig=='PUT'  and h>=sl_price:
            exit_px, ei = sl_price, j; break
        elif sig=='PUT'  and l<=tp_price:
            exit_px, ei = tp_price, j; break
    else:
        ei      = min(i+MAX_HOLD_BARS, len(df)-1)
        exit_px = df.at[ei,'close']

    # slippage-adjusted execution
    if sig=='CALL':
        exec_in  = entry_px*(1+slp)
        exec_out = exit_px*(1-slp)
        pnl_per = exec_out - exec_in
    else:
        exec_in  = entry_px*(1-slp)
        exec_out = exit_px*(1+slp)
        pnl_per = exec_in - exec_out

    pnl_usd = pnl_per * qty
    notional = (exec_in + exec_out) * qty
    fee_usd  = notional * TAKER_FEE
    pnl_usd -= fee_usd
    pnl_inr  = pnl_usd * USD_INR_RATE

    # **only reinvest 30% of wins, bank 70%**
    if pnl_inr>0:
        reinvest = pnl_inr*0.30
        bank    += pnl_inr*0.70
        pot     += reinvest
    else:
        pot += pnl_inr

    return {
        "orig_idx":   i,
        "Signal":     sig,
        "Entry":      pd.to_datetime(row['timestamp'],unit='ms'),
        "Exit":       pd.to_datetime(df.at[ei,'timestamp'],unit='ms'),
        "Qty":        round(qty,4),
        "PnL_INR":    round(pnl_inr,2),
        "Pot":        round(pot,2),
        "Banked":     round(bank,2)
    }, pot, bank

if __name__=="__main__":
    df      = pd.read_csv(DATA_FILE)
    df      = preprocess(df)
    entries = find_entries(df)

    # we'll fold sequentially so pot/bank carry forward
    pot, bank = INITIAL_CAPITAL, 0.0
    trades = []
    for _, row in entries.iterrows():
        tr, pot, bank = simulate_trade((row.orig_idx, row, df, pot, bank))
        trades.append(tr)

    out = pd.DataFrame(trades).sort_values("Entry")
    out.to_csv(OUTPUT_FILE, index=False)
    total = pot + bank
    print(f"Trades: {len(out)}")
    print(f"Final pot: ₹{pot:.2f}, banked: ₹{bank:.2f}, total: ₹{total:.2f}  (ROI: {(total/INITIAL_CAPITAL-1)*100:.2f}%)")
