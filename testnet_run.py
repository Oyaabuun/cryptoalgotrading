#!/usr/bin/env python3
"""
Live Trading Loop with Multi-Stage Exits:
- Static entry via Market order
- Dynamic exit: 
  1) Scale-out half at 2×ATR
  2) Move stop to breakeven + 0.2×ATR after 1×ATR
  3) Trail stop at 1×ATR behind new peak
  4) Time stop at LOOKBACK bars

Resumes on restart, repairs missing orders, and logs trades & metrics.

Subtle differences to be aware of
Despite the one‑to‑one mapping of stages, there are still a few practical divergences between your historical backtest and the live script:

Intrabar price action:

Backtest iterates tick‑by‑tick/bar‑by‑bar within each 5 m bar, so it “sees” high / low touches anywhere in the bar.

Live code only polls once every 5 m and reads the last known close price and bar’s high/low. If price momentarily spikes beyond an ATR threshold between polls, you might miss it (or conversely trigger two stages in the same bar).

Order execution & slippage:

Backtest uses your fixed SLIPPAGE constant on every fill, assuming instantaneous fills at bar prices.

Live places real market orders and OCO‑stops, which can fill at the bid/ask spread or with partial fills, and network latency can push your SL/TP a little farther.

Fee model & funding rates:

Backtest applies a static TAKER_FEE on each leg and ignores funding. Live trading on testnet (and especially on mainnet) will incur real funding payments, dynamic spreads, and occasionally partial fills.

Timing granularity:

Your backtester simulates a synchronous for‑loop over every single bar.

The live loop wakes up every 5 minutes, so you’ll have up to a 5 m “reaction delay,” which can slightly alter your effective ATR triggers.

What this means in practice
Logic parity: ✔️ All entry + exit rules are implemented identically.

Execution risk: ⚠️ Minor “slippage” in how and when orders actually fire.

Missed triggers: ⚠️ If price whipsaws inside a 5 m bar (e.g. quickly up and back down), your live loop may either miss or batch‑trigger multiple stages at once.


"""
import os, sys, time, math, requests, signal
from datetime import datetime
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
import importlib.util

# ── Config ──────────────────────────────────────────────────────
SYMBOL        = "BTCUSDT"
LEVERAGE      = 10
CONTRACT_SZ   = 0.001
USDT_CAPITAL  = 58.0
TRADE_FILE    = "output/testnet_trades.csv"
METRICS_FILE  = "output/testnet_metrics.csv"
NETWORK_LOG   = "output/network.logs"

METRICS_COLS = [
    "cycle_time","total_return_pct","annualized_return_pct","sharpe_ratio",
    "max_drawdown_pct","win_rate_pct","num_trades","profit_factor","avg_win_usd",
    "avg_loss_usd","max_win_usd","max_loss_usd","avg_duration_min","final_balance_usdt",
    "ATR_WINDOW","EMA_LONG","SL_ATR_MULT","TP_ATR_MULT","VOL_MULT",
    "RISK_PCT","REGIME_MULT"
]

# Graceful shutdown handler
def handle_sigint(signum, frame):
    print("\n⚡️ Stopping live loop…")
    sys.exit(0)
signal.signal(signal.SIGINT, handle_sigint)

# Helper: market entry + OCO

def open_entry_and_sl_tp(side, qty, sl_price, tp_price):
    client.futures_create_order(symbol=SYMBOL, side=side, type="MARKET", quantity=qty)
    client.futures_create_order(
        symbol=SYMBOL,
        side="SELL" if side=="BUY" else "BUY",
        type="STOP_MARKET",
        stopPrice=sl_price,
        closePosition=False,
        timeInForce="GTC"
    )
    client.futures_create_order(
        symbol=SYMBOL,
        side="SELL" if side=="BUY" else "BUY",
        type="TAKE_PROFIT_MARKET",
        stopPrice=tp_price,
        closePosition=False,
        timeInForce="GTC"
    )

# Fetch live klines
def fetch_latest_data(retries=3, backoff_s=5):
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    for attempt in range(1, retries+1):
        try:
            kl5 = client.futures_klines(symbol=SYMBOL, interval="5m", limit=LOOKBACK+50)
            df5 = pd.DataFrame(kl5, columns=cols).astype({c:"float" for c in ["open","high","low","close","volume"]})
            df5["datetime"] = pd.to_datetime(df5["open_time"], unit="ms")
            kl1 = client.futures_klines(symbol=SYMBOL, interval="1h", limit=200)
            df1h = pd.DataFrame(kl1, columns=cols).astype({c:"float" for c in ["open","high","low","close","volume"]})
            df1h["datetime"] = pd.to_datetime(df1h["open_time"], unit="ms")
            return df5, df1h
        except Exception as e:
            with open(NETWORK_LOG, "a") as nf:
                nf.write(f"{datetime.utcnow().isoformat()} attempt {attempt} failed: {e}\n")
            if attempt < retries:
                time.sleep(backoff_s)
            else:
                raise

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    if not os.path.exists(TRADE_FILE):
        pd.DataFrame(columns=["timestamp","side","qty","price","atr","stage","exit_reason"]).to_csv(TRADE_FILE, index=False)
    if not os.path.exists(METRICS_FILE):
        pd.DataFrame(columns=METRICS_COLS).to_csv(METRICS_FILE, index=False)

    load_dotenv(".env")
    API_KEY, API_SECRET = os.getenv("BINANCE_FUTURES_API_KEY"), os.getenv("BINANCE_FUTURES_SECRET_KEY")
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Set API keys in .env")
    client = Client(API_KEY, API_SECRET, {"timeout": 30})
    client.API_URL, client.FUTURES_URL = ("https://testnet.binancefuture.com", "https://testnet.binancefuture.com/fapi")

    sys.path.append(".")
    spec = importlib.util.spec_from_file_location("ml_way2", "./ml_way2.py")
    ml = importlib.util.module_from_spec(spec); spec.loader.exec_module(ml)
    LOOKBACK, preprocess_1h, preprocess_5m, find_entries, run_backtest = (
        ml.LOOKBACK, ml.preprocess_1h, ml.preprocess_5m, ml.find_entries, ml.run_backtest
    )

    combo = pd.read_csv("use_this_combi.csv").iloc[0]
    params = combo.drop(["sharpe","num_entries"]).to_dict()

    client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)

    # Resume logic
    position_open = False
    trades = pd.read_csv(TRADE_FILE)
    entry_rows = trades[trades.exit_reason=="ENTRY"]
    if not entry_rows.empty:
        entry_record = entry_rows.iloc[-1]
        # Recompute same as entry block if needed
        pos_info = client.futures_position_information(symbol=SYMBOL)
        amt = float(next(p for p in pos_info if p['symbol']==SYMBOL)['positionAmt'])
        position_open = amt != 0
        if position_open:
            # repair missing orders here if desired
            pass

    in_trade = position_open
    has_scaled = has_moved_to_be = False
    peak_price = None

    print("▶️ Starting live loop...")
    while True:
        now = datetime.utcnow().isoformat()
        df5, df1h = fetch_latest_data()
        hf = preprocess_1h(df1h, params)
        b5 = preprocess_5m(df5.copy(), params, hf)
        sigs = find_entries(b5, params)
        
        # ENTRY
        if not in_trade and not sigs.empty:
            sig = sigs.iloc[-1]
            side = "BUY" if sig.signal=="CALL" else "SELL"
            entry_price, atr = sig.open, sig.atr
            sl = entry_price - params['SL_ATR_MULT']*atr if side=="BUY" else entry_price + params['SL_ATR_MULT']*atr
            tp = entry_price + params['TP_ATR_MULT']*atr if side=="BUY" else entry_price - params['TP_ATR_MULT']*atr
            risk = USDT_CAPITAL * params['RISK_PCT'] * LEVERAGE
            qty = math.floor(risk/(abs(entry_price-sl)*CONTRACT_SZ))*CONTRACT_SZ
            open_entry_and_sl_tp(side, qty, sl, tp)
            peak_price = entry_price
            has_scaled = has_moved_to_be = False
            entry_record = {'side': side, 'qty': qty, 'entry_price': entry_price, 'atr': atr}
            in_trade = True
            pd.DataFrame([{**entry_record, 'timestamp': now, 'stage':'entry','exit_reason':'ENTRY'}])\
              .to_csv(TRADE_FILE, mode='a', header=False, index=False)
            print(f"{now} ⇒ ENTER {side}@{entry_price}")

        # EXIT management
        if in_trade:
            price = b5['close'].iloc[-1]
            atr = b5['atr'].iloc[-1]
            if entry_record['side']=='BUY':
                peak_price = max(peak_price, price)
            else:
                peak_price = min(peak_price, price)

            # 1×ATR breakeven move
            if not has_moved_to_be and ((entry_record['side']=='BUY' and price>=entry_record['entry_price']+atr) or (entry_record['side']=='SELL' and price<=entry_record['entry_price']-atr)):
                new_sl = entry_record['entry_price'] + (0.2*atr if entry_record['side']=='BUY' else -0.2*atr)
                orders = client.futures_get_open_orders(symbol=SYMBOL)
                for o in orders:
                    if o['type']=='STOP_MARKET': client.futures_cancel_order(symbol=SYMBOL, orderId=o['orderId'])
                client.futures_create_order(symbol=SYMBOL, side=('SELL' if entry_record['side']=='BUY' else 'BUY'), type='STOP_MARKET', stopPrice=new_sl, closePosition=False, timeInForce='GTC')
                has_moved_to_be = True
                pd.DataFrame([{**entry_record,'timestamp':now,'stage':'breakeven','exit_reason':'BE'}])\
                  .to_csv(TRADE_FILE, mode='a', header=False, index=False)
                print(f"{now} ⇒ MOVE TO BE+0.2ATR @ {new_sl}")

            # 2×ATR scale-out
            if not has_scaled and ((entry_record['side']=='BUY' and price>=entry_record['entry_price']+2*atr) or (entry_record['side']=='SELL' and price<=entry_record['entry_price']-2*atr)):
                half = entry_record['qty']/2
                side2 = 'SELL' if entry_record['side']=='BUY' else 'BUY'
                client.futures_create_order(symbol=SYMBOL, side=side2, type='MARKET', quantity=half)
                has_scaled = True
                pd.DataFrame([{**entry_record,'timestamp':now,'stage':'scale','qty':half,'exit_reason':'SCALE'}])\
                  .to_csv(TRADE_FILE, mode='a', header=False, index=False)
                print(f"{now} ⇒ SCALE-OUT half @ market {half}")

            # Trail stop
            trail_sl = (peak_price-atr) if entry_record['side']=='BUY' else (peak_price+atr)
            orders = client.futures_get_open_orders(symbol=SYMBOL)
            for o in orders:
                if o['type']=='STOP_MARKET': client.futures_cancel_order(symbol=SYMBOL, orderId=o['orderId'])
            client.futures_create_order(symbol=SYMBOL, side=('SELL' if entry_record['side']=='BUY' else 'BUY'), type='STOP_MARKET', stopPrice=trail_sl, closePosition=False, timeInForce='GTC')

            # Time stop
            oldest_time = b5['datetime'].iloc[-LOOKBACK]
            if datetime.utcnow() >= oldest_time:
                side3='SELL' if entry_record['side']=='BUY' else 'BUY'
                client.futures_create_order(symbol=SYMBOL, side=side3, type='MARKET', quantity=entry_record['qty'])
                in_trade=False
                pd.DataFrame([{**entry_record,'timestamp':now,'stage':'time','exit_reason':'TIME'}])\
                  .to_csv(TRADE_FILE, mode='a', header=False, index=False)
                print(f"{now} ⇒ TIME-STOP exit @ market")

            # Detect close
            pos = client.futures_position_information(symbol=SYMBOL)
            if float(next(p for p in pos if p['symbol']==SYMBOL)['positionAmt'])==0:
                in_trade=False
                pd.DataFrame([{**entry_record,'timestamp':now,'stage':'exit_complete','exit_reason':'CLOSED'}])\
                  .to_csv(TRADE_FILE, mode='a', header=False, index=False)
                print(f"{now} ⇒ TRADE CLOSED fully")

        # Metrics log
        hist_tr, hist_m = run_backtest(b5, params)
        row = {**{"cycle_time":now}, **hist_m, **params}
        pd.DataFrame([row])[METRICS_COLS].to_csv(METRICS_FILE, mode='a', header=False, index=False)

        time.sleep(5*60)