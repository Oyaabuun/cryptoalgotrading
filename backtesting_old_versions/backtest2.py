#!/usr/bin/env python3
import pandas as pd
import numpy as np

# ── Strategy parameters ────────────────────────────────────────
LOOKBACK      = 20
ATR_WINDOW    = 14
EMA_LONG      = 50
ADX_WINDOW    = 14
ADX_THRESHOLD = 20
RISK_PCT      = 0.01     # 1% of equity per trade
SL_ATR_MULT   = 1.25
TP_ATR_MULT   = 4.0
MIN_HOLD      = 1
MAX_HOLD      = 20

INITIAL_USDT  = 58.0      # starting capital in USDT
LEVERAGE      = 10.0      # 10× leverage
CONTRACT_SZ   = 1.0       # USDT per contract for BTCUSDT perp
TAKER_FEE     = 0.0005    # 0.05%
SLIPPAGE      = 0.0002    # 0.02%

DATA_FILE     = "data/BTC_PERPETUAL_5m_12months.csv"


def preprocess(df):
    # handle different timestamp columns
    if 'open_time' in df:
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    elif 'timestamp' in df:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    elif 'datetime' in df:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        raise ValueError("DataFrame must contain 'open_time', 'timestamp', or 'datetime' column")
    df = df.sort_values('datetime').reset_index(drop=True)
    df['prev_close'] = df['close'].shift(1)

    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['prev_close']).abs(),
        (df['low']  - df['prev_close']).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_WINDOW).mean()

    # EMA long
    df['ema_long'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi'] = 100 - 100/(1 + up.rolling(14).mean()/down.rolling(14).mean())

    # Donchian channels
    df['highest'] = df['high'].rolling(LOOKBACK).max().shift(1)
    df['lowest']  = df['low'].rolling(LOOKBACK).min().shift(1)

    # ADX
    up_m   = df['high'].diff()
    down_m = df['low'].shift(1).diff() * -1
    plus   = np.where((up_m>down_m)&(up_m>0), up_m, 0.0)
    minus  = np.where((down_m>up_m)&(down_m>0), down_m, 0.0)
    sm_tr  = tr.ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    sm_p   = pd.Series(plus).ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    sm_m   = pd.Series(minus).ewm(alpha=1/ADX_WINDOW, adjust=False).mean()
    df['plus_di']  = 100 * sm_p  / sm_tr
    df['minus_di'] = 100 * sm_m  / sm_tr
    dx            = 100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'])
    df['adx']      = dx.ewm(alpha=1/ADX_WINDOW, adjust=False).mean()

    return df.dropna().reset_index(drop=True)


def find_entries(df):
    m_atr = df['atr'].mean()
    th    = 0.5 * df['atr']
    bull  = (
        (df['close'] > df['ema_long']) &
        (df['rsi']   > 50) &
        (df['close'] > df['highest'] + th) &
        (df['adx']   > ADX_THRESHOLD) &
        (df['atr']   >= m_atr)
    )
    bear  = (
        (df['close'] < df['ema_long']) &
        (df['rsi']   < 50) &
        (df['close'] < df['lowest'] - th) &
        (df['adx']   > ADX_THRESHOLD) &
        (df['atr']   >= m_atr)
    )
    df['signal'] = np.where(bull, 'CALL', np.where(bear, 'PUT', None))
    entries = df[df['signal'].notnull()].copy()
    entries['idx'] = entries.index
    return entries.reset_index(drop=True)


def backtest(compound=True):
    df_raw  = pd.read_csv(DATA_FILE)
    df      = preprocess(df_raw)
    entries = find_entries(df)

    equity   = INITIAL_USDT
    banked   = 0.0
    trades   = []
    results  = []

    for _, row in entries.iterrows():
        sig       = row['signal']
        epx       = row['open']
        atr       = row['atr']
        idx       = int(row['idx'])
        entry_dt  = df.at[idx, 'datetime']

        # SL/TP prices
        sl_price  = epx - SL_ATR_MULT*atr if sig=='CALL' else epx + SL_ATR_MULT*atr
        tp_price  = epx + TP_ATR_MULT*atr if sig=='CALL' else epx - TP_ATR_MULT*atr

        # risk capital with leverage
        base_eq   = equity if compound else INITIAL_USDT
        risk_usd  = base_eq * RISK_PCT * LEVERAGE

        # size contracts
        sl_dist   = abs(epx - sl_price)
        contracts = risk_usd / (sl_dist * CONTRACT_SZ)

        # find exit
        exit_px, exit_idx = None, None
        for j in range(idx+1, min(idx+1+MAX_HOLD, len(df))):
            h,l = df.at[j,'high'], df.at[j,'low']
            if sig=='CALL' and l <= sl_price:
                exit_px, exit_idx = sl_price, j; break
            if sig=='CALL' and h >= tp_price:
                exit_px, exit_idx = tp_price, j; break
            if sig=='PUT' and h >= sl_price:
                exit_px, exit_idx = sl_price, j; break
            if sig=='PUT' and l <= tp_price:
                exit_px, exit_idx = tp_price, j; break
        if exit_px is None:
            exit_idx = min(idx+MAX_HOLD, len(df)-1)
            exit_px  = df.at[exit_idx,'close']
        exit_dt = df.at[exit_idx, 'datetime']

        # slippage
        ie = epx*(1+SLIPPAGE) if sig=='CALL' else epx*(1-SLIPPAGE)
        ex = exit_px*(1-SLIPPAGE) if sig=='CALL' else exit_px*(1+SLIPPAGE)
        pnl_pc = ((ex-ie) if sig=='CALL' else (ie-ex)) * CONTRACT_SZ

        # total PnL and fees
        pnl_usd = pnl_pc * contracts
        fee     = (ie + ex) * contracts * CONTRACT_SZ * TAKER_FEE
        pnl_usd -= fee

        # banking & equity
        if pnl_usd > 0:
            banked_gain = pnl_usd * 0.70
            equity_gain = pnl_usd * 0.30
            banked    += banked_gain
            equity    += equity_gain
        else:
            equity    += pnl_usd

        # record trade
        trades.append({
            'entry_dt':  entry_dt,
            'exit_dt':   exit_dt,
            'signal':    sig,
            'contracts': contracts,
            'entry_px':  epx,
            'exit_px':   exit_px,
            'pnl_usd':   pnl_usd,
            'equity':    equity,
            'banked':    banked
        })

    # save trade log
    df_trades = pd.DataFrame(trades)
    df_trades.to_csv('trade.csv', index=False)

    # performance metrics
    df_trades['total'] = df_trades['equity'] + df_trades['banked']
    returns = df_trades['total'].pct_change().fillna(0)
    sharpe = returns.mean() / returns.std() if returns.std() else np.nan
    cummax = df_trades['total'].cummax()
    drawdowns = (df_trades['total'] - cummax) / cummax
    max_dd = drawdowns.min()
    win_rate = (df_trades['pnl_usd'] > 0).mean()
    total_return = (df_trades['total'].iloc[-1] / INITIAL_USDT - 1)
    duration = (df_trades['exit_dt'] - df_trades['entry_dt']).dt.total_seconds() / 60
    avg_dur = duration.mean()
    profit = df_trades[df_trades['pnl_usd']>0]['pnl_usd'].sum()
    loss   = -df_trades[df_trades['pnl_usd']<0]['pnl_usd'].sum()
    profit_factor = profit / loss if loss else np.nan
    avg_win = df_trades[df_trades['pnl_usd']>0]['pnl_usd'].mean()
    avg_loss= df_trades[df_trades['pnl_usd']<0]['pnl_usd'].mean()
    max_win = df_trades['pnl_usd'].max()
    max_loss= df_trades['pnl_usd'].min()

    metrics = {
        'total_return_pct':      total_return * 100,
        'annualized_return_pct': ((1+total_return)**(365*24*60/5) -1)*100,
        'sharpe_ratio':          sharpe,
        'max_drawdown_pct':      max_dd * 100,
        'win_rate_pct':          win_rate * 100,
        'num_trades':            len(df_trades),
        'profit_factor':         profit_factor,
        'avg_win_usd':           avg_win,
        'avg_loss_usd':          avg_loss,
        'max_win_usd':           max_win,
        'max_loss_usd':          max_loss,
        'avg_duration_min':      avg_dur
    }
    pd.DataFrame([metrics]).to_csv('results.csv', index=False)

    # print metrics
    print(pd.DataFrame([metrics]).to_string(index=False))

if __name__ == '__main__':
    backtest(compound=True)
